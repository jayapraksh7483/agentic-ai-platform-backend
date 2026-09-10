"""
Planning logic for the Manager Agent.

IMPORTANT:
    - Capability -> agent mapping is NEVER hard-coded here.
    - Every agent lookup goes through services.discovery_service.
    - The Agent Registry is the source of truth.
    - The Manager selects capabilities from capabilities actually
      registered by ACTIVE agents belonging to the current user.
    - Capability selection is INDEX-BASED: the LLM is shown a numbered
      list of the exact registry strings and returns which indices apply.
    - Agent selection uses a deterministic strategy:
        1. highest current_version
        2. alphabetical agent name
    - New agents are NEVER created silently.
    - If nothing can satisfy the request, the Manager drafts a
      ProposedAgentSpec and waits for explicit user approval.

PHASE 5B USER OWNERSHIP:
    Every registry discovery operation accepts user_id.

    User A:
        Manager -> discovery -> User A agents only

    User B:
        Manager -> discovery -> User B agents only

    A user's Manager must NEVER discover another user's private agents.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from models.agent import Agent, AgentCapability, AgentStatus
from services import discovery_service
from services.llm_service import get_llm_client

from .prompts import (
    AGENT_PROPOSAL_SYSTEM_PROMPT,
    AGENT_PROPOSAL_USER_TEMPLATE,
    CAPABILITY_EXTRACTION_SYSTEM_PROMPT,
    CAPABILITY_EXTRACTION_USER_TEMPLATE,
    REQUEST_CLASSIFICATION_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
)
from .schemas import (
    CapabilityExtraction,
    ExecutionPlan,
    ExecutionStep,
    ProposedAgentSpec,
)

logger = logging.getLogger("manager")

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)```",
    re.DOTALL,
)


# =====================================================================
# JSON HELPERS
# =====================================================================

def _parse_json_object(raw: str) -> Dict[str, Any]:
    """
    Best-effort extraction of a JSON object from an LLM response.

    Handles:
        - normal JSON
        - ```json fenced JSON
        - leading/trailing prose around JSON

    Raises:
        ValueError when no valid JSON object can be extracted.
    """

    if not raw:
        raise ValueError(
            "Empty response from LLM"
        )

    text = raw.strip()

    fence_match = _JSON_FENCE_RE.search(
        text
    )

    if fence_match:
        text = fence_match.group(1).strip()

    if not text.startswith("{"):

        start = text.find("{")
        end = text.rfind("}")

        if (
            start != -1
            and end != -1
            and end > start
        ):
            text = text[
                start:end + 1
            ]

    return json.loads(text)


# =====================================================================
# REGISTERED CAPABILITIES
# =====================================================================

def _get_registered_capabilities(
    db: Session,
    user_id: Optional[int] = None,
) -> List[str]:
    """
    Read currently registered capabilities from the Agent Registry.

    Only ACTIVE agents are considered.

    PHASE 5B:
        When user_id is provided, only capabilities belonging to
        that user's agents are returned.

    There are NO hard-coded capability names.
    """

    query = (
        db.query(
            AgentCapability.capability_name
        )
        .join(
            Agent,
            AgentCapability.agent_id == Agent.id,
        )
        .filter(
            Agent.status == AgentStatus.ACTIVE
        )
    )

    # -------------------------------------------------------------
    # PHASE 5B OWNERSHIP FILTER
    # -------------------------------------------------------------

    if user_id is not None:
        query = query.filter(
            Agent.created_by == user_id
        )

    rows = (
        query
        .distinct()
        .all()
    )

    capabilities = [
        row[0]
        for row in rows
        if row[0]
    ]

    logger.info(
        "manager.registered_capabilities "
        "user_id=%s count=%s",
        user_id,
        len(capabilities),
    )

    return capabilities


# =====================================================================
# REGISTERED AGENT SUMMARY
# =====================================================================

def _get_registered_agents_summary(
    db: Session,
    user_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Return a small summary of ACTIVE agents.

    Used by the agent-proposal prompt so the LLM can avoid proposing
    an unnecessary duplicate.

    PHASE 5B:
        When user_id is provided, only that user's agents are included.
    """

    query = (
        db.query(Agent)
        .filter(
            Agent.status == AgentStatus.ACTIVE
        )
    )

    # -------------------------------------------------------------
    # PHASE 5B OWNERSHIP FILTER
    # -------------------------------------------------------------

    if user_id is not None:
        query = query.filter(
            Agent.created_by == user_id
        )

    agents = query.all()

    summary: List[Dict[str, Any]] = []

    for agent in agents:

        capability_names = [
            c.capability_name
            for c in getattr(
                agent,
                "capabilities",
                [],
            )
            if getattr(
                c,
                "capability_name",
                None,
            )
        ]

        summary.append(
            {
                "name": agent.name,
                "capabilities": capability_names,
            }
        )

    logger.info(
        "manager.registered_agents_summary "
        "user_id=%s count=%s",
        user_id,
        len(summary),
    )

    return summary


# =====================================================================
# CAPABILITY EXTRACTION
# =====================================================================

def extract_capabilities(
    db: Session,
    user_input: str,
    provider: str,
    model: str,
    user_id: Optional[int] = None,
) -> CapabilityExtraction:
    """
    Understand the user's request and select required capabilities
    from capabilities registered for the current user.

    IMPORTANT:

        The LLM never returns capability text.

        It receives a NUMBERED list of exact registry strings and
        returns indices.

        Our code resolves those indices back to the exact registry
        strings.

    PHASE 5B:

        The registry vocabulary itself is user-scoped.

        Therefore User A's LLM cannot select User B's capability
        because User B's capability is never included in the list.

    Example:

        User:
            "Which clothes are suitable for winter?"

        User A registry:

            [0] Suggesting weather-appropriate clothing options
            [1] Write and optimize SQL queries...

        LLM:

            {
                "selected_capabilities": [0]
            }

        Code resolves:

            0 -> "Suggesting weather-appropriate clothing options"
    """

    available_capabilities = (
        _get_registered_capabilities(
            db=db,
            user_id=user_id,
        )
    )

    if not available_capabilities:

        logger.info(
            "manager.no_registered_capabilities "
            "user_id=%s",
            user_id,
        )

        return CapabilityExtraction(
            required_capabilities=[],
            unmatched_description=user_input,
        )

    logger.info(
        "manager.registered_capabilities "
        "user_id=%s count=%s",
        user_id,
        len(available_capabilities),
    )

    # -------------------------------------------------------------
    # Number the exact registry vocabulary.
    # -------------------------------------------------------------

    registry_context = "\n".join(
        f"[{index}] {capability}"
        for index, capability in enumerate(
            available_capabilities
        )
    )

    user_prompt = (
        CAPABILITY_EXTRACTION_USER_TEMPLATE.format(
            user_input=user_input,
            registered_capabilities=registry_context,
        )
    )

    client = get_llm_client(
        provider
    )

    valid_capabilities: List[str] = []

    try:

        raw = client.generate(
            system_prompt=(
                CAPABILITY_EXTRACTION_SYSTEM_PROMPT
            ),
            user_input=user_prompt,
            model_name=model,
        )

        logger.debug(
            "manager.capability_llm_response=%s",
            raw,
        )

        data = _parse_json_object(
            raw
        )

        requested_indices = (
            data.get(
                "selected_capabilities"
            )
            or []
        )

        # ---------------------------------------------------------
        # IMPORTANT SAFETY CHECK
        #
        # The LLM can only refer to indices from the user-scoped
        # registry vocabulary.
        # ---------------------------------------------------------

        rejected_indices = []

        for raw_index in requested_indices:

            try:
                index = int(
                    raw_index
                )

            except (
                TypeError,
                ValueError,
            ):
                rejected_indices.append(
                    raw_index
                )
                continue

            if (
                0
                <= index
                < len(
                    available_capabilities
                )
            ):

                capability = (
                    available_capabilities[
                        index
                    ]
                )

                if (
                    capability
                    not in valid_capabilities
                ):
                    valid_capabilities.append(
                        capability
                    )

            else:
                rejected_indices.append(
                    raw_index
                )

        if rejected_indices:

            logger.warning(
                "manager.rejected_invalid_capability_indices "
                "user_id=%s requested=%s available_count=%s",
                user_id,
                rejected_indices,
                len(
                    available_capabilities
                ),
            )

        unmatched_description = (
            data.get(
                "unmatched_description"
            )
        )

        if valid_capabilities:

            # A capability matched.
            unmatched_description = None

        elif not unmatched_description:

            # Nothing selected and no description.
            unmatched_description = user_input

        extraction = CapabilityExtraction(
            required_capabilities=(
                valid_capabilities
            ),
            execution_mode=data.get(
                "execution_mode",
                "parallel",
            ),
            condition_keyword=data.get(
                "condition_keyword"
            ),
            unmatched_description=(
                unmatched_description
            ),
        )

    except Exception as exc:

        logger.warning(
            "manager.capability_extraction_failed "
            "user_id=%s error=%s",
            user_id,
            exc,
        )

        extraction = CapabilityExtraction(
            required_capabilities=[],
            unmatched_description=user_input,
        )

    logger.info(
        "manager.capabilities_identified "
        "user_id=%s capabilities=%s mode=%s "
        "condition=%s unmatched=%s",
        user_id,
        extraction.required_capabilities,
        extraction.execution_mode,
        extraction.condition_keyword,
        extraction.unmatched_description,
    )

    return extraction


# =====================================================================
# PROPOSE NEW AGENT
# =====================================================================

def propose_new_agent(
    db: Session,
    user_input: str,
    unmatched_description: str,
    provider: str,
    model: str,
    user_id: Optional[int] = None,
) -> ProposedAgentSpec:
    """
    Draft a NEW agent proposal.

    This function NEVER creates an Agent row.

    Creation happens only after explicit user approval through:

        manager.service.approve_pending_agent()

    PHASE 5B:

        Existing-agent context is restricted to the current user.
    """

    existing_agents = (
        _get_registered_agents_summary(
            db=db,
            user_id=user_id,
        )
    )

    existing_agents_context = (
        "\n".join(
            f"- {a['name']}: "
            f"{', '.join(a['capabilities']) or '(no capabilities listed)'}"
            for a in existing_agents
        )
        or "(no agents currently registered)"
    )

    user_prompt = (
        AGENT_PROPOSAL_USER_TEMPLATE.format(
            user_input=user_input,
            unmatched_description=(
                unmatched_description
                or user_input
            ),
            existing_agents=(
                existing_agents_context
            ),
        )
    )

    client = get_llm_client(
        provider
    )

    try:

        raw = client.generate(
            system_prompt=(
                AGENT_PROPOSAL_SYSTEM_PROMPT
            ),
            user_input=user_prompt,
            model_name=model,
        )

        logger.debug(
            "manager.agent_proposal_llm_response=%s",
            raw,
        )

        data = _parse_json_object(
            raw
        )

        proposal = ProposedAgentSpec(
            name=(
                data.get("name")
                or "General Assistant Agent"
            ),
            description=data.get(
                "description"
            ),
            capabilities=(
                data.get(
                    "capabilities"
                )
                or [
                    unmatched_description
                    or user_input
                ]
            ),
            system_prompt=(
                data.get(
                    "system_prompt"
                )
                or (
                    "You are a helpful agent for: "
                    f"{unmatched_description or user_input}"
                )
            ),
            provider=(
                data.get("provider")
                or "gemini"
            ),
            model=data.get(
                "model"
            ),
            input_schema=data.get(
                "input_schema"
            ),
            output_schema=data.get(
                "output_schema"
            ),
            reason=data.get(
                "reason"
            ),
        )

    except Exception as exc:

        logger.warning(
            "manager.agent_proposal_failed "
            "falling_back_to_generic_proposal "
            "user_id=%s error=%s",
            user_id,
            exc,
        )

        # Deterministic fallback.

        proposal = ProposedAgentSpec(
            name="General Assistant Agent",
            description=(
                "Handles requests about: "
                f"{unmatched_description or user_input}"
            ),
            capabilities=[
                unmatched_description
                or user_input
            ],
            system_prompt=(
                "You are a general-purpose "
                "assistant agent. Answer the "
                "user's request directly and "
                "honestly using your own knowledge."
            ),
            provider=(
                provider
                or "gemini"
            ),
            model=model,
            reason=(
                "Generated as a fallback because "
                "agent-proposal generation failed."
            ),
        )

    logger.info(
        "manager.agent_proposed "
        "user_id=%s name=%s capabilities=%s",
        user_id,
        proposal.name,
        proposal.capabilities,
    )

    return proposal


# =====================================================================
# AGENT SELECTION
# =====================================================================

def _select_agent_for_capability(
    db: Session,
    capability: str,
    user_id: Optional[int] = None,
) -> Optional[Agent]:
    """
    Discover an ACTIVE agent advertising the requested capability.

    PHASE 5B:
        Discovery is restricted to the authenticated user's agents.

    No capability -> agent mapping is hard-coded.

    Selection strategy:

        1. Highest current_version
        2. Alphabetical agent name
    """

    candidates = (
        discovery_service.discover_agents(
            db,
            capability=capability,
            status_filter=AgentStatus.ACTIVE,

            # -----------------------------------------------------
            # PHASE 5B OWNERSHIP FILTER
            # -----------------------------------------------------
            owner_id=user_id,
        )
    )

    if not candidates:

        logger.info(
            "manager.no_agent_for_capability "
            "user_id=%s capability=%s",
            user_id,
            capability,
        )

        return None

    candidates.sort(
        key=lambda agent: (
            -agent.current_version,
            agent.name,
        )
    )

    selected_agent = candidates[0]

    logger.info(
        "manager.agent_selected "
        "user_id=%s capability=%s agent=%s version=%s",
        user_id,
        capability,
        selected_agent.name,
        selected_agent.current_version,
    )

    return selected_agent


# =====================================================================
# BUILD EXECUTION PLAN
# =====================================================================

def build_plan(
    db: Session,
    user_input: str,
    extraction: CapabilityExtraction,
    user_id: Optional[int] = None,
) -> ExecutionPlan:
    """
    Discover agents for every required capability and construct the
    execution plan.

    Supports:

        - sequential
        - parallel
        - conditional

    PHASE 5B:

        Only agents belonging to user_id can be placed into the plan.
    """

    plan = ExecutionPlan(
        request=user_input
    )

    resolved_steps: List[
        ExecutionStep
    ] = []

    previous_step_id: Optional[
        str
    ] = None

    for idx, capability in enumerate(
        extraction.required_capabilities,
        start=1,
    ):

        step_id = f"step_{idx}"

        agent = (
            _select_agent_for_capability(
                db=db,
                capability=capability,
                user_id=user_id,
            )
        )

        if agent is None:

            plan.unmatched_capabilities.append(
                capability
            )

            continue

        depends_on: List[str] = []

        condition_on: Optional[
            str
        ] = None

        condition_keyword: Optional[
            str
        ] = None

        # ---------------------------------------------------------
        # SEQUENTIAL
        # ---------------------------------------------------------

        if (
            extraction.execution_mode
            == "sequential"
            and previous_step_id
        ):

            depends_on = [
                previous_step_id
            ]

        # ---------------------------------------------------------
        # CONDITIONAL
        # ---------------------------------------------------------

        elif (
            extraction.execution_mode
            == "conditional"
            and previous_step_id
        ):

            condition_on = (
                previous_step_id
            )

            condition_keyword = (
                extraction.condition_keyword
            )

            depends_on = [
                previous_step_id
            ]

        # ---------------------------------------------------------
        # PARALLEL
        #
        # No dependency between steps.
        # ---------------------------------------------------------

        step = ExecutionStep(
            step_id=step_id,
            capability=capability,
            agent_id=agent.id,
            agent_name=agent.name,
            task=user_input,
            depends_on=depends_on,
            condition_on=condition_on,
            condition_keyword=condition_keyword,
            system_prompt=agent.system_prompt,
            provider=agent.provider,
            model=agent.model,
        )

        resolved_steps.append(
            step
        )

        previous_step_id = step_id

    plan.steps = resolved_steps

    logger.info(
        "manager.plan_created "
        "user_id=%s steps=%s unmatched=%s",
        user_id,
        [
            step.step_id
            for step in plan.steps
        ],
        plan.unmatched_capabilities,
    )

    return plan


# =====================================================================
# SYNTHESIZE FINAL RESPONSE
# =====================================================================

def synthesize_final_response(
    user_input: str,
    step_results: List[
        Dict[str, Any]
    ],
    provider: str,
    model: str,
) -> str:
    """
    Combine real results produced by executed agents into one
    user-facing response.

    The Manager must never fabricate agent results.
    """

    if not step_results:

        return (
            "No agents were executed "
            "for this request."
        )

    client = get_llm_client(
        provider
    )

    try:

        raw = client.generate(
            system_prompt=(
                SYNTHESIS_SYSTEM_PROMPT
            ),
            user_input=(
                SYNTHESIS_USER_TEMPLATE.format(
                    user_input=user_input,
                    step_results_json=json.dumps(
                        step_results,
                        default=str,
                    ),
                )
            ),
            model_name=model,
        )

        if raw and raw.strip():
            return raw.strip()

    except Exception as exc:

        logger.warning(
            "manager.synthesis_failed "
            "falling_back_to_raw_results "
            "error=%s",
            exc,
        )

    # -------------------------------------------------------------
    # Deterministic fallback.
    # -------------------------------------------------------------

    lines: List[str] = []

    for item in step_results:

        capability = (
            item.get("capability")
            or item.get("step_id")
        )

        if (
            item.get("status")
            == "success"
        ):

            lines.append(
                f"- {capability}: "
                f"{item.get('result')}"
            )

        else:

            lines.append(
                f"- {capability}: "
                f"failed ({item.get('error')})"
            )

    return "\n".join(
        lines
    )


# =====================================================================
# REQUEST CLASSIFICATION
# =====================================================================

def classify_request(
    user_input: str,
    provider: str,
    model: str,
) -> str:

    try:

        client = get_llm_client(
            provider
        )

        raw = client.generate(
            system_prompt=(
                REQUEST_CLASSIFICATION_SYSTEM_PROMPT
            ),
            user_input=user_input,
            model_name=model,
        )

        data = _parse_json_object(
            raw
        )

        request_type = data.get(
            "request_type",
            "agent_task",
        )

        if request_type not in {
            "general",
            "agent_task",
        }:
            return "agent_task"

        return request_type

    except Exception as exc:

        logger.warning(
            "Request classification failed: %s",
            exc,
        )

        return "agent_task"