"""
Manager Agent orchestration graph.

Responsibilities:
- Understand the user's request.
- Discover only agents belonging to the authenticated user.
- Build a dynamic execution plan.
- Execute registered agents through the existing Agent Runtime.
- Support parallel/sequential/conditional execution.
- Resolve genuine capability gaps with a temporary worker or a reusable persistent agent.
- Never create an agent because of provider/tool/RAG/planning/timeout failures.
- Preserve authenticated user_id throughout the orchestration path.
- Preserve conversation history throughout chat/orchestration.
"""

import concurrent.futures
import logging
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from core.config import settings
from core.encryption import decrypt_token
from agent_runtime.runtime import agent_runtime
from models.agent import Agent, AgentStatus
from models.knowledge import (
    KnowledgeBase,
    KnowledgeBaseStatus,
    KnowledgeDocument,
    KnowledgeChunk,
    DocumentStatus,
)
from services.llm_service import get_llm_client

from . import planner
from .schemas import (
    ExecutionPlan,
    ExecutionStep,
   
    ManagerFailure,
    ManagerFailureCode,
    ManagerIntent,
    ManagerStage,
)


logger = logging.getLogger("manager")

MANAGER_BUILD = "2026-09-20-power-context-json-fallback-v10"

class ManagerState(TypedDict, total=False):
    deadline: float
    """
    Manager orchestration lifecycle state.

    Phase 1 groups this into Request / Understanding / Planning /
    Lifecycle / Knowledge / Execution / Response, so each later phase
    has an obvious place to attach rather than bolting more loose keys
    onto a flat dict.

    Backwards compatibility: `request_type` ("general" | "agent_task")
    is RETAINED and kept in sync with `intent`, because existing tests
    and the persistence layer still read it. New code should read
    `intent` / `requirements`.
    """

    # ---- Request -------------------------------------------------
    execution_id: str
    user_id: Optional[int]

    user_input: str
    provider: str
    model: str

    conversation_history: List[Dict[str, Any]]
    conversation_id: Optional[str]

    db: Any

    # Bounded, user-scoped, read-only snapshot of the live AgentOS registry.
    # It contains metadata only (never API keys/secrets) and is supplied to
    # request understanding/direct answering so the Manager knows what really
    # exists instead of guessing from model memory.
    platform_context: str

    # ---- Understanding -------------------------------------------
    # Legacy binary view, derived from `intent` (see _sync_legacy_request_type).
    request_type: str

    intent: Optional[str]
    intent_confidence: float
    intent_reason: Optional[str]
    requirements: Dict[str, Any]

    # ---- Planning ------------------------------------------------
    planning_status: str          # "pending" | "ok" | "failed" | "no_capability"
    plan: Optional[Dict[str, Any]]

    iteration: int
    no_agents_found: bool
    unmatched_description: Optional[str]

    # A genuine capability gap: understood + actionable + nothing in
    # the registry satisfies it. Phase 1 only RECORDS this; it never
    # creates anything from it.
    capability_gap: bool
    missing_capabilities: List[str]

    # Capabilities whose providers exist but were all ineligible.
    # Distinct from missing_capabilities: an availability problem,
    # never a capability gap.
    unavailable_capabilities: List[str]

    # ---- Agent lifecycle -----------------------------------------
    proposed_agent: Optional[Dict[str, Any]]
    needs_approval: bool

    # Hybrid capability-gap resolution.
    gap_resolution_strategy: Optional[str]
    auto_created_agent_id: Optional[str]

    # ---- Knowledge -----------------------------------------------
    conversation_knowledge_base_ids: List[str]

    # ---- Execution -----------------------------------------------
    step_results: Dict[str, Dict[str, Any]]
    completed: List[str]
    skipped: List[str]

    # ---- Failure -------------------------------------------------
    # Structured failure detail. `error` (str) is retained for
    # backwards compatibility with the persistence layer and API.
    failure_code: Optional[str]
    failure_stage: Optional[str]
    failure_retryable: bool

    # ---- Response ------------------------------------------------
    final_response: Optional[str]
    status: str
    error: Optional[str]


# ---------------------------------------------------------------------
# Phase 1 state helpers
# ---------------------------------------------------------------------

def _record_failure(
    state: ManagerState,
    failure: ManagerFailure,
) -> ManagerState:
    """
    Record a structured failure without losing the legacy `error`
    string that persistence/API already depend on.

    Only `user_message` is ever surfaced as the user-facing response;
    `message` (which may contain exception text) stays in `error` for
    operators.
    """

    state["failure_code"] = failure.code.value
    state["failure_stage"] = failure.stage.value
    state["failure_retryable"] = failure.retryable
    state["error"] = failure.message or failure.code.value

    logger.info(
        "manager.failure execution_id=%s code=%s stage=%s retryable=%s",
        state.get("execution_id"),
        failure.code.value,
        failure.stage.value,
        failure.retryable,
    )

    return state


def _sync_legacy_request_type(
    state: ManagerState,
) -> None:
    """
    Keep the legacy binary `request_type` consistent with `intent`.

    GENERAL_ANSWER maps to "general"; everything else maps to
    "agent_task". This exists so Phase 1 can introduce the richer
    intent model without breaking existing readers in one step.
    """

    intent = state.get("intent")

    state["request_type"] = (
        "general"
        if intent == ManagerIntent.GENERAL_ANSWER.value
        else "agent_task"
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_list(value: Any) -> List[Any]:
    """
    Convert None/non-list values into a safe list.

    This prevents errors such as:
        "'NoneType' object is not iterable"
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    return [value]


def _safe_string_list(value: Any) -> List[str]:
    """
    Normalize an arbitrary value into a clean list of strings.
    """

    return [
        str(item)
        for item in _safe_list(value)
        if item is not None and str(item).strip()
    ]


def _safe_conversation_history(
    value: Any,
) -> List[Dict[str, Any]]:
    """
    Normalize conversation history into a list of dictionaries.

    Invalid history entries are ignored rather than crashing
    orchestration.
    """

    history = _safe_list(value)

    normalized: List[Dict[str, Any]] = []

    for item in history:
        if isinstance(item, dict):
            normalized.append(item)

    return planner._sanitize_conversation_history(normalized)


def _build_manager_context_pack(
    state: ManagerState,
) -> str:
    """Build a bounded, live, user-scoped AgentOS context snapshot.

    The context intentionally contains metadata only.  No encrypted/raw API
    keys, credentials, prompts, document contents, or other users' resources
    are included.  If one registry is temporarily unavailable, the rest of
    the context still remains usable.
    """

    user_id = state.get("user_id")
    db = state.get("db")

    if user_id is None or db is None:
        return "Authenticated AgentOS registry context is unavailable."

    lines: List[str] = []

    # -------------------------- Agents ---------------------------
    try:
        from services import agent_service

        agents = list(
            agent_service.list_agents(
                db=db,
                owner_id=user_id,
            )
            or []
        )[:25]

        lines.append(f"Agents ({len(agents)} shown):")

        if not agents:
            lines.append("- none")

        for agent in agents:
            status_value = str(
                getattr(getattr(agent, "status", None), "value", getattr(agent, "status", ""))
                or "unknown"
            )
            provider = str(getattr(agent, "provider", "") or "not set")
            model = str(getattr(agent, "model", "") or "not set")
            capabilities = [
                str(getattr(cap, "capability_name", "") or "").strip()
                for cap in (getattr(agent, "capabilities", None) or [])
            ]
            capabilities = [item for item in capabilities if item][:12]

            configured_tools = getattr(agent, "tools", None)
            if configured_tools is None:
                tools_text = "unrestricted/platform-default"
            elif isinstance(configured_tools, (list, tuple, set)):
                tool_names = [
                    str(item).strip()
                    for item in configured_tools
                    if str(item or "").strip()
                ][:12]
                tools_text = ", ".join(tool_names) if tool_names else "none"
            else:
                tools_text = str(configured_tools)

            lines.append(
                "- "
                + str(getattr(agent, "name", "Unnamed Agent"))
                + f" | status={status_value}"
                + f" | provider={provider}"
                + f" | model={model}"
                + f" | rag={bool(getattr(agent, 'is_rag', False))}"
                + f" | capabilities={', '.join(capabilities) if capabilities else 'none'}"
                + f" | tools={tools_text}"
            )
    except Exception as exc:
        logger.warning("manager.context_agents_unavailable error=%s", exc)
        lines.append("Agents: unavailable")

    # ---------------------- Knowledge bases ----------------------
    try:
        knowledge_bases = (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.created_by == user_id)
            .order_by(KnowledgeBase.created_at.desc())
            .limit(25)
            .all()
        )

        lines.append(f"Knowledge bases ({len(knowledge_bases)} shown):")
        if not knowledge_bases:
            lines.append("- none")

        for kb in knowledge_bases:
            status_value = str(
                getattr(getattr(kb, "status", None), "value", getattr(kb, "status", ""))
                or "unknown"
            )
            lines.append(
                f"- {getattr(kb, 'name', 'Unnamed Knowledge Base')}"
                f" | status={status_value}"
            )
    except Exception as exc:
        logger.warning("manager.context_kbs_unavailable error=%s", exc)
        lines.append("Knowledge bases: unavailable")

    # --------------------------- Tools ----------------------------
    try:
        from tools import tool_registry

        tools = list(tool_registry.list_enabled_tools() or [])[:25]
        lines.append(f"Platform tools ({len(tools)} shown):")
        if not tools:
            lines.append("- none")
        for tool in tools:
            name = str(getattr(tool, "name", "") or "Unnamed Tool")
            description = " ".join(
                str(getattr(tool, "description", "") or "").split()
            )[:180]
            lines.append(
                f"- {name}"
                + (f" | {description}" if description else "")
            )
    except Exception as exc:
        logger.warning("manager.context_tools_unavailable error=%s", exc)
        lines.append("Platform tools: unavailable")

    attached_count = len(
        _safe_string_list(
            state.get("conversation_knowledge_base_ids")
        )
    )
    lines.append(f"Conversation attached knowledge bases: {attached_count}")
    lines.append(
        "Pending approval in prior assistant turn: "
        + ("yes" if _previous_assistant_requests_approval(state) else "no")
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Request classification
# ---------------------------------------------------------------------


def _looks_like_simple_calculation(value: str) -> bool:
    text = str(value or "").strip().lower()

    if not text:
        return False

    expression = re.search(
        r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)",
        text,
    )

    if not expression:
        return False

    prefix = text[:expression.start()].strip()
    suffix = text[expression.end():].strip()

    allowed_prefixes = {
        "",
        "what is",
        "calculate",
        "compute",
        "solve",
        "please calculate",
        "please compute",
        "please solve",
    }

    allowed_suffixes = {
        "",
        "?",
        "please",
    }

    return (
        prefix in allowed_prefixes
        and suffix in allowed_suffixes
    )


def _looks_like_current_information(value: str) -> bool:
    """
    Detect requests that should use the registered Web Search agent
    without depending on LLM intent/capability extraction.

    This includes:
      - explicitly current requests: latest/today/recent/live
      - dated external lookups: "2024 top 5 Telugu movies"
      - rankings/lists about changing public data: top movies,
        box-office, releases, sports results, prices, etc.

    General conceptual/educational questions are intentionally not
    matched here and can still be answered directly by Manager.
    """
    raw = str(value or "").strip().casefold()

    if not raw:
        return False

    text = f" {raw} "

    current_markers = (
        " latest ",
        " today ",
        " current ",
        " currently ",
        " recent ",
        " recently ",
        " breaking ",
        " live ",
        " this week ",
        " this month ",
        " this year ",
        " new movies ",
        " new releases ",
        " released movies ",
        " now showing ",
        " box office ",
        " collection ",
        " collections ",
        " opening weekend ",
    )

    if any(
        marker in text
        for marker in current_markers
    ):
        return True

    # A specific year + public/media/result domain normally requires an
    # external lookup rather than model memory.
    has_year = bool(
        re.search(
            r"\b(?:19|20)\d{2}\b",
            raw,
        )
    )

    external_domains = (
        "movie",
        "movies",
        "film",
        "films",
        "tollywood",
        "bollywood",
        "hollywood",
        "song",
        "songs",
        "album",
        "albums",
        "series",
        "webseries",
        "web series",
        "news",
        "election",
        "match",
        "matches",
        "score",
        "scores",
        "tournament",
        "ipl",
        "cricket",
        "football",
        "stock",
        "stocks",
        "share price",
        "market",
        "price",
        "prices",
        "release",
        "releases",
        "award",
        "awards",
        "winner",
        "winners",
        "restaurant",
        "restaurants",
        "hotel",
        "hotels",
        "event",
        "events",
    )

    has_external_domain = any(
        domain in raw
        for domain in external_domains
    )

    ranking_markers = (
        "top ",
        "best ",
        "highest ",
        "most ",
        "popular ",
        "hit ",
        "hits ",
        "blockbuster ",
        "rank ",
        "ranking ",
        "list ",
        "recommend ",
        "recommendation ",
    )

    has_ranking_intent = any(
        marker in text
        for marker in ranking_markers
    )

    # Example:
    #   "2024 top 5 telugu movies in tollywood"
    if (
        has_external_domain
        and (
            has_year
            or has_ranking_intent
        )
    ):
        return True

    # Common "who/what won/released/earned" factual lookups also need
    # external data even when the user did not say "latest".
    lookup_patterns = (
        r"\bwho won\b",
        r"\bwhat won\b",
        r"\bwhich .* won\b",
        r"\bwhen .* released\b",
        r"\bhow much .* earned\b",
        r"\bhow much .* collected\b",
    )

    return any(
        re.search(pattern, raw)
        for pattern in lookup_patterns
    )



_SPECIALIZED_ROUTE_STOPWORDS = {
    # conversational / grammar
    "a", "an", "and", "are", "as", "at", "be", "can", "could",
    "do", "does", "for", "from", "give", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "please", "tell", "that",
    "the", "this", "to", "what", "when", "where", "which", "who",
    "why", "with", "you",

    # generic agent / capability vocabulary
    "agent", "assistant", "general", "help", "information", "inquiry",
    "inquiries", "question", "questions", "support", "guidance",
    "handle", "handles", "handling", "provide", "provides",
    "service", "services", "task", "tasks",
}


def _specialized_route_tokens(
    value: Any,
) -> set[str]:
    """
    Tokenize routing text while removing words that are too generic to
    justify delegating a request to a specialized child agent.
    """
    raw = str(value or "").casefold()

    tokens = {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            raw,
        )
        if (
            len(token) >= 3
            and token not in _SPECIALIZED_ROUTE_STOPWORDS
        )
    }

    return tokens


def _specialized_agent_acronyms(
    agent: Agent,
) -> set[str]:
    """
    Produce useful short forms from the agent name.

    Example:
        Human Resources Assistant -> hr

    Generic suffixes such as "agent" and "assistant" are removed before
    the acronym is built.
    """
    generic_name_words = {
        "agent",
        "assistant",
        "specialist",
        "worker",
        "service",
    }

    words = [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            str(agent.name or "").casefold(),
        )
        if token not in generic_name_words
    ]

    acronyms: set[str] = set()

    if len(words) >= 2:
        acronym = "".join(
            word[0]
            for word in words
            if word
        )

        if 2 <= len(acronym) <= 6:
            acronyms.add(acronym)

    return acronyms


def _find_specialized_registry_agent(
    state: ManagerState,
) -> Optional[tuple[Agent, str]]:
    """
    Find a strong deterministic match between the current user query
    and an ACTIVE user-owned specialized agent.

    Why this exists:
    ----------------
    GENERAL_ANSWER classification used to bypass the registry entirely.
    That meant a request such as:

        "tell me about onboarding formalities"

    was answered directly even when the user already had an active
    "Human Resources Assistant" with capability "onboarding information".

    This matcher runs before generic LLM classification and only routes
    when the registry match is strong. It deliberately ignores platform
    default agents because Calculator/Web Search/Document Reader already
    have their own deterministic paths, and General Assistant should not
    steal normal direct-answer traffic.
    """
    user_id = state.get("user_id")

    if user_id is None:
        return None

    query_text = str(
        state.get("user_input")
        or ""
    ).strip()

    if not query_text:
        return None

    query_tokens = _specialized_route_tokens(
        query_text
    )

    query_raw = " ".join(
        re.findall(
            r"[a-z0-9]+",
            query_text.casefold(),
        )
    )

    query_all_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            query_text.casefold(),
        )
    )

    agents = (
        state["db"]
        .query(Agent)
        .filter(
            Agent.status == AgentStatus.ACTIVE,
            Agent.created_by == user_id,
            Agent.is_default.is_(False),
        )
        .order_by(
            Agent.current_version.desc(),
            Agent.name.asc(),
        )
        .all()
    )

    scored: List[tuple[int, Agent, str]] = []

    for agent in agents:
        # RAG is an execution mode, not a delegation rule. A specialized
        # RAG-enabled agent is still eligible to own a normal domain task;
        # document grounding is decided later from the request itself.
        capability_names = [
            str(
                getattr(
                    capability,
                    "capability_name",
                    "",
                )
                or ""
            ).strip()
            for capability in (
                getattr(
                    agent,
                    "capabilities",
                    [],
                )
                or []
            )
            if str(
                getattr(
                    capability,
                    "capability_name",
                    "",
                )
                or ""
            ).strip()
        ]

        capability_token_map = {
            capability: _specialized_route_tokens(
                capability
            )
            for capability in capability_names
        }

        name_tokens = _specialized_route_tokens(
            agent.name
        )

        description_tokens = (
            _specialized_route_tokens(
                getattr(
                    agent,
                    "description",
                    "",
                )
            )
        )

        score = 0
        best_capability = (
            capability_names[0]
            if capability_names
            else str(
                agent.name
                or "specialized_task"
            )
        )

        best_capability_score = 0

        # Capabilities are the strongest routing signal.
        for capability, cap_tokens in (
            capability_token_map.items()
        ):
            normalized_capability = " ".join(
                re.findall(
                    r"[a-z0-9]+",
                    capability.casefold(),
                )
            )

            capability_score = 0

            if (
                normalized_capability
                and normalized_capability in query_raw
            ):
                capability_score += 12

            overlap = (
                query_tokens
                & cap_tokens
            )

            capability_score += (
                5 * len(overlap)
            )

            if capability_score > best_capability_score:
                best_capability_score = capability_score
                best_capability = capability

        score += best_capability_score

        # Agent/domain name is a strong secondary signal.
        name_overlap = (
            query_tokens
            & name_tokens
        )
        score += 3 * len(name_overlap)

        # Description is useful, but weaker than declared capabilities.
        description_overlap = (
            query_tokens
            & description_tokens
        )
        score += 2 * len(
            description_overlap
        )

        # Support common domain acronyms such as HR.
        for acronym in _specialized_agent_acronyms(
            agent
        ):
            if acronym in query_all_tokens:
                score += 7

        # At least one domain-specific signal is required.
        has_domain_signal = bool(
            best_capability_score
            or name_overlap
            or description_overlap
            or (
                _specialized_agent_acronyms(
                    agent
                )
                & query_all_tokens
            )
        )

        if (
            has_domain_signal
            and score >= 5
        ):
            scored.append(
                (
                    score,
                    agent,
                    best_capability,
                )
            )

    if not scored:
        return None

    scored.sort(
        key=lambda item: (
            -item[0],
            -int(
                getattr(
                    item[1],
                    "current_version",
                    0,
                )
                or 0
            ),
            str(
                item[1].name
                or ""
            ).casefold(),
        )
    )

    top_score, top_agent, top_capability = (
        scored[0]
    )

    # If two different specialized agents are equally strong, do not
    # guess. Let the normal understanding/planning path handle it.
    if (
        len(scored) > 1
        and scored[1][0] == top_score
        and str(scored[1][1].id)
        != str(top_agent.id)
    ):
        logger.info(
            "manager.specialized_registry_match_ambiguous "
            "execution_id=%s top=%s second=%s score=%s",
            state.get("execution_id"),
            top_agent.name,
            scored[1][1].name,
            top_score,
        )
        return None

    logger.info(
        "manager.specialized_registry_match "
        "execution_id=%s agent=%s capability=%s score=%s",
        state.get("execution_id"),
        top_agent.name,
        top_capability,
        top_score,
    )

    return (
        top_agent,
        top_capability,
    )


def _find_semantic_specialized_registry_agent(
    state: ManagerState,
) -> Optional[tuple[Agent, str]]:
    """
    Fallback delegation check for requests that the intent classifier would
    otherwise treat as GENERAL_ANSWER.

    The existing deterministic matcher above is intentionally cheap and
    precise, but token overlap can miss semantically equivalent requests
    (for example, a Kubernetes pod troubleshooting question that never says
    the word "OpenShift").

    Reuse the platform's existing capability extractor as the semantic layer:
    it selects from registered capability names rather than inventing agent
    IDs. We then accept the result only when it resolves to one ACTIVE,
    user-owned, non-default specialized agent. Default agents remain on their
    normal deterministic/direct-response paths.

    Any extraction/provider failure is non-fatal here; the Manager simply
    falls back to the already-classified direct response.
    """
    user_id = state.get("user_id")
    if user_id is None:
        return None

    agents = (
        state["db"]
        .query(Agent)
        .filter(
            Agent.status == AgentStatus.ACTIVE,
            Agent.created_by == user_id,
            Agent.is_default.is_(False),
        )
        .order_by(
            Agent.current_version.desc(),
            Agent.name.asc(),
        )
        .all()
    )

    if not agents:
        return None

    capability_owners: dict[str, list[tuple[Agent, str]]] = {}

    for agent in agents:
        for capability in getattr(agent, "capabilities", []) or []:
            capability_name = str(
                getattr(capability, "capability_name", "") or ""
            ).strip()
            if not capability_name:
                continue

            normalized = " ".join(
                re.findall(
                    r"[a-z0-9]+",
                    capability_name.casefold(),
                )
            )
            if not normalized:
                continue

            capability_owners.setdefault(normalized, []).append(
                (agent, capability_name)
            )

    if not capability_owners:
        return None

    try:
        extraction = planner.extract_capabilities(
            db=state["db"],
            user_input=state["user_input"],
            provider=state["provider"],
            model=state["model"],
            user_id=user_id,
            conversation_document_capability=None,
        )
    except Exception as exc:
        logger.info(
            "manager.semantic_specialized_match_skipped "
            "execution_id=%s reason=%s",
            state.get("execution_id"),
            exc,
        )
        return None

    required_capabilities = _safe_string_list(
        getattr(extraction, "required_capabilities", None)
    )

    resolved: list[tuple[Agent, str]] = []
    for required in required_capabilities:
        normalized = " ".join(
            re.findall(
                r"[a-z0-9]+",
                str(required).casefold(),
            )
        )
        owners = capability_owners.get(normalized, [])

        # A capability shared by several specialized agents is ambiguous;
        # let normal planning handle it instead of guessing.
        if len(owners) == 1:
            resolved.append(owners[0])

    if not resolved:
        return None

    unique_agent_ids = {str(agent.id) for agent, _ in resolved}
    if len(unique_agent_ids) != 1:
        logger.info(
            "manager.semantic_specialized_match_ambiguous "
            "execution_id=%s capabilities=%s",
            state.get("execution_id"),
            required_capabilities,
        )
        return None

    agent, capability = resolved[0]

    logger.info(
        "manager.semantic_specialized_match "
        "execution_id=%s agent=%s capability=%s",
        state.get("execution_id"),
        agent.name,
        capability,
    )

    return agent, capability


def _select_existing_agent(
    state: ManagerState,
    preferred_name: str,
    capability_terms: List[str],
) -> Optional[Agent]:
    query = (
        state["db"]
        .query(Agent)
        .filter(
            Agent.status == AgentStatus.ACTIVE,
        )
    )

    if state.get("user_id") is not None:
        query = query.filter(
            Agent.created_by == state["user_id"]
        )

    agents = query.order_by(
        Agent.is_default.desc(),
        Agent.current_version.desc(),
        Agent.name.asc(),
    ).all()

    preferred = next(
        (
            agent
            for agent in agents
            if str(agent.name or "").strip().casefold()
            == preferred_name.casefold()
        ),
        None,
    )

    if preferred is not None:
        return preferred

    normalized_terms = {
        str(term or "")
        .strip()
        .casefold()
        .replace("_", " ")
        .replace("-", " ")
        for term in capability_terms
        if term
    }

    for agent in agents:
        for capability in getattr(agent, "capabilities", []) or []:
            name = (
                str(
                    getattr(
                        capability,
                        "capability_name",
                        "",
                    )
                    or ""
                )
                .strip()
                .casefold()
                .replace("_", " ")
                .replace("-", " ")
            )

            if (
                name in normalized_terms
                or any(
                    term in name or name in term
                    for term in normalized_terms
                )
            ):
                return agent

    return None


def _single_agent_plan(
    state: ManagerState,
    *,
    agent: Agent,
    step_id: str,
    capability: str,
) -> ManagerState:
    step = ExecutionStep(
        step_id=step_id,
        capability=capability,
        agent_id=str(agent.id),
        agent_name=agent.name,
        agent_version=agent.current_version,
        task=state["user_input"],
        system_prompt=agent.system_prompt,
        provider=agent.provider,
        model=agent.model,
    )

    state["plan"] = ExecutionPlan(
        request=state["user_input"],
        steps=[step],
    ).model_dump()

    state["planning_status"] = "ok"
    state["no_agents_found"] = False
    state["capability_gap"] = False
    state["missing_capabilities"] = []
    state["unavailable_capabilities"] = []
    state["unmatched_description"] = None
    state["step_results"] = {}
    state["completed"] = []
    state["skipped"] = []
    state["iteration"] = 0
    state["error"] = None
    state["failure_code"] = None
    state["failure_stage"] = None
    state["failure_retryable"] = False

    logger.info(
        "manager.fast_path_selected execution_id=%s "
        "agent_id=%s agent_name=%s capability=%s",
        state.get("execution_id"),
        agent.id,
        agent.name,
        capability,
    )

    return state



def _is_routing_introspection(value: str) -> bool:
    """
    Detect questions about which child agent/tool handled a previous
    response. These must be answered from persisted message metadata,
    never from the LLM's guess.
    """
    text = f" {str(value or '').strip().casefold()} "

    if not text.strip():
        return False

    strong_markers = (
        " child agent ",
        " which agent ",
        " what agent ",
        " which child ",
        " web search agent ",
        " websearch agent ",
        " document reader ",
        " calculator agent ",
        " agent use ches",
        " agent use chesa",
        " agent ni use ",
        " agent enduku use ",
        " agent yenduku use ",
        " which tool ",
        " what tool ",
        " tool use ches",
        " did you use ",
        " did u use ",
        " called which agent ",
    )

    if any(marker in text for marker in strong_markers):
        return True

    # Generic "agent" + usage/call/routing wording.
    return (
        " agent " in text
        and any(
            marker in text
            for marker in (
                " use ",
                " used ",
                " call ",
                " called ",
                " route ",
                " routed ",
                " routing ",
                " enduku ",
                " yenduku ",
            )
        )
    )


def _last_assistant_route_metadata(
    state: ManagerState,
) -> Optional[Dict[str, Any]]:
    """
    Read the most recent assistant routing metadata from raw conversation
    history. We intentionally do NOT pass this through the prompt-history
    sanitizer because the sanitizer keeps only role/content.
    """
    history = _safe_list(
        state.get("conversation_history")
    )

    for item in reversed(history):
        if not isinstance(item, dict):
            continue

        role = str(
            item.get("role") or ""
        ).strip().casefold()

        if role != "assistant":
            continue

        metadata = (
            item.get("message_metadata")
            or item.get("metadata")
            or {}
        )

        if isinstance(metadata, dict):
            return metadata

    return None


def _last_assistant_content(
    state: ManagerState,
) -> str:
    """Return the latest assistant text from conversation history."""
    history = _safe_list(
        state.get("conversation_history")
    )

    for item in reversed(history):
        if not isinstance(item, dict):
            continue

        role = str(
            item.get("role") or ""
        ).strip().casefold()

        if role != "assistant":
            continue

        content = item.get("content")
        if isinstance(content, str):
            return content.strip()

    return ""


def _previous_assistant_requests_approval(
    state: ManagerState,
) -> bool:
    """
    Keep short yes/no-style replies available to lifecycle/approval handling
    when the previous assistant message was explicitly asking for approval.
    """
    metadata = _last_assistant_route_metadata(state) or {}
    status = str(
        metadata.get("status") or ""
    ).strip().casefold()

    if status in {
        "pending_agent_approval",
        "pending_approval",
        "approval_required",
    }:
        return True

    previous = f" {_last_assistant_content(state).casefold()} "
    approval_markers = (
        " approval ",
        " approve ",
        " approve it ",
        " should i go ahead and create ",
        " should i create ",
        " go ahead and create ",
        " needs your approval ",
        " explicitly approve ",
    )

    return any(
        marker in previous
        for marker in approval_markers
    )


def _simple_conversation_response(
    state: ManagerState,
) -> Optional[str]:
    """Return a deterministic response for greetings/acks/thanks.

    Tiny social turns should never depend on a JSON classifier.  This also
    keeps provider formatting mistakes such as a non-JSON classifier answer
    from surfacing as "Routing failed" for messages like "ok babu".

    If the previous assistant turn is awaiting approval, do not consume yes/no
    acknowledgements here; the approval/lifecycle path must see them.
    """

    raw = str(state.get("user_input") or "").strip().casefold()
    if not raw:
        return None

    if _previous_assistant_requests_approval(state):
        return None

    normalized = re.sub(
        r"[^\w']+",
        " ",
        raw,
        flags=re.UNICODE,
    )
    normalized = " ".join(normalized.split())
    if not normalized:
        return None

    # Remove a friendly vocative at either end: "ok babu", "bro thanks",
    # "hello anna", etc.  This is intentionally narrow so an actionable
    # command containing one of these words is not swallowed as small talk.
    vocatives = {
        "bro", "brother", "babu", "anna", "mava", "mawa", "boss",
        "dude", "yar", "yaar", "ji", "sir", "madam", "maam", "ma'am",
    }

    tokens = normalized.split()
    while len(tokens) > 1 and tokens[0] in vocatives:
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in vocatives:
        tokens = tokens[:-1]
    core = " ".join(tokens)

    greetings = {
        "hi", "hii", "hiii", "hello", "hey", "heyy", "namaste",
        "namaskaram", "good morning", "good afternoon", "good evening",
    }
    thanks = {
        "thanks", "thank you", "thank u", "tq", "thanks a lot",
        "ధన్యవాదాలు",
    }
    acknowledgements = {
        "ok", "okay", "okk", "fine", "good", "great", "cool", "nice",
        "hmm", "hm", "hmmm", "haha", "lol", "sare", "avunu", "haa",
        "haan", "yes", "yeah", "yep", "సరే", "అవును", "హా", "హ్మ్",
    }
    negatives = {
        "no", "nope", "nah", "nothing", "nothing else", "that's all",
        "thats all", "ledhu", "ledu", "le", "nahi", "nahin", "లేదు",
    }

    if core in greetings:
        return "Hi! How can I help?"
    if core in thanks:
        return "You're welcome."
    if core in acknowledgements:
        return "Okay."
    if core in negatives:
        return "Okay."

    return None


def _looks_like_short_conversational_followup(
    state: ManagerState,
) -> bool:
    """Backward-compatible boolean wrapper around the deterministic reply."""
    return _simple_conversation_response(state) is not None


def _routing_introspection_response(
    state: ManagerState,
) -> str:
    metadata = _last_assistant_route_metadata(
        state
    )

    if not metadata:
        return (
            "I don't have routing metadata for the previous response, "
            "so I can't reliably say which child agent was used."
        )

    orchestrator = metadata.get(
        "orchestrator"
    )

    manager_name = "Manager Agent"

    if isinstance(orchestrator, dict):
        value = orchestrator.get("name")
        if isinstance(value, str) and value.strip():
            manager_name = value.strip()

    child_agents = metadata.get(
        "child_agents"
    )

    status = str(
        metadata.get("status") or ""
    ).strip().casefold()

    if isinstance(child_agents, list) and child_agents:
        route_parts = [manager_name]
        details = []

        for child in child_agents:
            if not isinstance(child, dict):
                continue

            name = str(
                child.get("name")
                or "Child Agent"
            ).strip()

            route_parts.append(name)

            tool_name = str(
                child.get("tool_name")
                or ""
            ).strip()

            provider = str(
                child.get("provider")
                or ""
            ).strip()

            model = str(
                child.get("model")
                or ""
            ).strip()

            if tool_name:
                details.append(
                    f"{name} used tool '{tool_name}'"
                )
            elif provider or model:
                provider_model = " / ".join(
                    item
                    for item in (
                        provider,
                        model,
                    )
                    if item
                )
                details.append(
                    f"{name} ran with {provider_model}"
                )

        route_text = " -> ".join(
            route_parts
        )

        if details:
            return (
                f"The previous response used: {route_text}. "
                + "; ".join(details)
                + "."
            )

        return (
            f"The previous response used: {route_text}."
        )

    if status in {"failed", "error"}:
        return (
            "No child agent completed the previous request. "
            "Routing/planning failed before a child agent could execute."
        )

    return (
        "No child agent was used for the previous response; "
        "the Manager Agent answered it directly."
    )



def _looks_like_knowledge_registry_query(
    state: ManagerState,
) -> bool:
    """Detect read-only questions asking which Knowledge Bases exist."""
    text = " ".join(str(state.get("user_input") or "").strip().casefold().split())
    if not text:
        return False

    if not any(marker in text for marker in (
        "knowledge base", "knowledge bases", "my kb", "my kbs",
        "naa kb", "na kb", "kb list",
    )):
        return False

    if any(marker in text for marker in (
        "create", "delete", "remove", "upload", "attach", "detach",
        "rename", "update", "edit",
    )):
        return False

    return any(marker in text for marker in (
        "what", "which", "list", "show", "available", "do i have",
        "i have", "unnayi", "unnay", "vunnayi", "vunnay", "enti",
    ))


def _knowledge_registry_response(
    state: ManagerState,
) -> str:
    user_id = state.get("user_id")
    if user_id is None:
        return "I need an authenticated user to read the Knowledge Base registry."

    try:
        kbs = (
            state["db"].query(KnowledgeBase)
            .filter(KnowledgeBase.created_by == user_id)
            .order_by(KnowledgeBase.created_at.desc())
            .all()
        )
    except Exception as exc:
        logger.exception("manager.knowledge_registry_query_failed user_id=%s", user_id)
        return f"I couldn't read your Knowledge Base registry right now: {exc}"

    if not kbs:
        return "You do not currently have any Knowledge Bases in AgentOS."

    lines = [
        f"You currently have {len(kbs)} Knowledge Base"
        f"{'s' if len(kbs) != 1 else ''}:"
    ]
    for index, kb in enumerate(kbs, start=1):
        status_value = str(
            getattr(getattr(kb, "status", None), "value", getattr(kb, "status", ""))
            or "unknown"
        )
        lines.append(f"{index}. {getattr(kb, 'name', 'Unnamed Knowledge Base')} — {status_value}")
    return "\n".join(lines)


def _looks_like_tool_registry_query(
    state: ManagerState,
) -> bool:
    """Detect read-only questions asking which platform tools are available."""
    text = " ".join(str(state.get("user_input") or "").strip().casefold().split())
    if not text or "tool" not in text:
        return False

    if any(marker in text for marker in (
        "create tool", "add tool", "delete tool", "remove tool", "connect tool",
        "attach tool", "assign tool", "tool ni add", "tools ni add",
    )):
        return False

    return any(marker in text for marker in (
        "what tools", "which tools", "list tools", "show tools",
        "available tools", "tools available", "tool list", "tools unnayi",
        "tools vunnayi", "tools enti", "my tools",
    ))


def _tool_registry_response(
    state: ManagerState,
) -> str:
    try:
        from tools import tool_registry
        tools = list(tool_registry.list_enabled_tools() or [])
    except Exception as exc:
        logger.exception("manager.tool_registry_query_failed")
        return f"I couldn't read the Tool Registry right now: {exc}"

    if not tools:
        return "There are currently no enabled platform tools registered in AgentOS."

    lines = [
        f"AgentOS currently has {len(tools)} enabled tool"
        f"{'s' if len(tools) != 1 else ''}:"
    ]
    for index, tool in enumerate(tools, start=1):
        name = str(getattr(tool, "name", "") or "Unnamed Tool")
        description = " ".join(
            str(getattr(tool, "description", "") or "").split()
        )
        lines.append(
            f"{index}. {name}"
            + (f" — {description}" if description else "")
        )
    return "\n".join(lines)


def _looks_like_agent_creation_capability_question(
    state: ManagerState,
) -> bool:
    """
    Detect a question about whether AgentOS/Manager is capable of creating
    agents, without treating that question itself as a creation request.

    Examples that are capability questions:
        "Can you create an agent?"
        "Can you create agents?"
        "nuv agent ni create cheyagalava?"

    Requests that include an actual agent specification, such as
    "Can you create an agent for SQL troubleshooting?", intentionally do
    not match these exact/short forms and continue through lifecycle routing.
    """
    raw = str(state.get("user_input") or "").strip().casefold()
    if not raw:
        return False

    normalized = re.sub(r"\s+", " ", raw).strip()

    patterns = (
        r"^(?:can|could)\s+(?:you|u)\s+create\s+(?:an?\s+)?agents?\s*[?.!]*$",
        r"^are\s+you\s+able\s+to\s+create\s+(?:an?\s+)?agents?\s*[?.!]*$",
        r"^do\s+you\s+create\s+agents?\s*[?.!]*$",
        r"^can\s+(?:the\s+)?manager\s+(?:agent\s+)?create\s+agents?\s*[?.!]*$",
        r"^(?:nuv|nuvu|nuvvu|meeru)\s+agents?(?:\s+ni)?\s+create\s+chey{1,2}agalava\s*[?.!]*$",
        r"^agents?(?:\s+ni)?\s+create\s+chey{1,2}agalava\s*[?.!]*$",
        r"^agents?\s+create\s+chey{1,2}acha\s*[?.!]*$",
    )

    return any(re.fullmatch(pattern, normalized) for pattern in patterns)


def _looks_like_agent_registry_query(
    state: ManagerState,
) -> bool:
    """
    Detect read-only questions about the user's registered/child agents.

    This is deliberately narrower than generic "agent" wording so normal
    educational questions such as "what is an AI agent?" stay on the
    GENERAL_ANSWER path. Routing-introspection questions are handled earlier
    by _is_routing_introspection().
    """
    current = str(state.get("user_input") or "").strip().casefold()
    if not current:
        return False

    # Mutating requests must continue through the lifecycle classifier.
    mutation_markers = (
        " create ", " add agent ", " delete ", " remove ", " rename ",
        " update ", " edit ", " modify ", " disable ", " enable ",
        " create chey", " delete chey", " remove chey", " update chey",
    )
    padded = f" {current} "
    if any(marker in padded for marker in mutation_markers):
        return False

    strong_markers = (
        "child agent",
        "child agents",
        "my agents",
        "my agent list",
        "list my agents",
        "list agents",
        "agent list",
        "available agents",
        "registered agents",
        "which agents",
        "what agents",
        "which agent do i have",
        "what agent do i have",
        "agents do i have",
        "agents are there",
        "agents unnayi",
        "agents unnay",
        "agents vunayi",
        "agents vunnayi",
        "agents vunnay",
        "naa agents",
        "na agents",
        "naku agents",
        "nee agents",
        "ni agents",
        "nee child",
        "ni child",
        "child agents unn",
        "child agents vun",
        "dheggara agents",
        "daggara agents",
    )
    if any(marker in current for marker in strong_markers):
        return True

    # Tool/status/model questions explicitly scoped to the user's agents are
    # also registry reads, not generic conceptual questions.
    registry_detail_markers = (
        " tools", "tool ", " status", " active", " inactive",
        " provider", " model", " capability", " capabilities", " rag",
    )
    ownership_markers = (
        "my agent", "my agents", "naa agent", "na agent", "nee agent",
        "ni agent", "child agent", "registered agent",
    )
    return (
        any(marker in current for marker in ownership_markers)
        and any(marker in current for marker in registry_detail_markers)
    )


def _format_agent_registry_response(
    state: ManagerState,
    agents: List[Agent],
) -> str:
    """Build a deterministic, DB-grounded answer for AGENT_QUERY."""
    if not agents:
        return "You do not currently have any registered agents in AgentOS."

    query = str(state.get("user_input") or "").casefold()

    previous_user_text = ""
    for item in reversed(_safe_list(state.get("conversation_history"))):
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").strip().casefold() != "user":
            continue
        previous_user_text = str(item.get("content") or "").casefold()
        break

    # A fragment such as "ni child agents ki" after a prior tools question
    # should remain about tools instead of losing the conversational subject.
    wants_tools = (
        "tool" in query
        or (
            "child agent" in query
            and "tool" in previous_user_text
        )
    )
    wants_details = wants_tools or any(
        token in query
        for token in (
            "status", "active", "inactive", "provider", "model",
            "capability", "rag", "detail",
        )
    )

    lines = [
        f"You currently have {len(agents)} registered agent"
        f"{'s' if len(agents) != 1 else ''} available to the Manager:"
    ]

    for index, agent in enumerate(agents, start=1):
        status_value = getattr(agent.status, "value", agent.status)
        kind = "platform default" if bool(getattr(agent, "is_default", False)) else "user-created"
        provider = str(getattr(agent, "provider", "") or "").strip() or "not set"
        model = str(getattr(agent, "model", "") or "").strip() or "not set"
        lines.append(
            f"{index}. {agent.name} — {status_value} — {kind} — {provider} / {model}"
        )

        capabilities = [
            str(getattr(capability, "capability_name", "") or "").strip()
            for capability in (getattr(agent, "capabilities", None) or [])
        ]
        capabilities = [item for item in capabilities if item]

        if wants_details and capabilities:
            lines.append("   Capabilities: " + ", ".join(capabilities))

        if wants_tools:
            configured_tools = getattr(agent, "tools", None)
            if configured_tools is None:
                lines.append(
                    "   Tools: unrestricted (all currently registered platform tools are available)"
                )
            elif isinstance(configured_tools, (list, tuple, set)):
                tool_names = [
                    str(item).strip()
                    for item in configured_tools
                    if str(item or "").strip()
                ]
                lines.append(
                    "   Tools: " + (", ".join(tool_names) if tool_names else "none")
                )
            else:
                lines.append(f"   Tools: {configured_tools}")

        if wants_details and bool(getattr(agent, "is_rag", False)):
            kb_id = getattr(agent, "knowledge_base_id", None)
            lines.append(
                "   RAG: enabled"
                + (f" (knowledge base {kb_id})" if kb_id else "")
            )

    return "\n".join(lines)


def agent_query(
    state: ManagerState,
) -> ManagerState:
    """
    Answer AGENT_QUERY from the authenticated user's live Agent Registry.

    No LLM is used here, so the Manager cannot invent child agents or claim
    that none exist when the database says otherwise.
    """
    user_id = state.get("user_id")
    if user_id is None:
        _record_failure(
            state,
            ManagerFailure(
                code=ManagerFailureCode.AGENT_NOT_AUTHORIZED,
                stage=ManagerStage.ROUTING,
                message="Agent registry query requires an authenticated user.",
                user_message="I need an authenticated user to read the agent registry.",
                retryable=False,
            ),
        )
        state["status"] = "failed"
        state["final_response"] = (
            "I need an authenticated user to read the agent registry."
        )
        return state

    try:
        # Local import avoids introducing a module-level dependency cycle.
        from services import agent_service

        agents = agent_service.list_agents(
            db=state["db"],
            owner_id=user_id,
        )

        state["final_response"] = _format_agent_registry_response(
            state,
            list(agents or []),
        )
        state["status"] = "success"
        state["error"] = None
        state["failure_code"] = None
        state["failure_stage"] = None
        state["failure_retryable"] = False
        state["needs_approval"] = False
        state["proposed_agent"] = None

        logger.info(
            "manager.agent_query_success execution_id=%s user_id=%s count=%s",
            state.get("execution_id"),
            user_id,
            len(agents or []),
        )
        return state

    except Exception as exc:
        logger.exception(
            "manager.agent_query_failed execution_id=%s user_id=%s",
            state.get("execution_id"),
            user_id,
        )
        _record_failure(
            state,
            ManagerFailure(
                code=ManagerFailureCode.ROUTING_FAILED,
                stage=ManagerStage.ROUTING,
                message=f"Agent registry query failed: {exc}",
                user_message="I couldn't read your agent registry right now. Please try again.",
                retryable=True,
            ),
        )
        state["status"] = "failed"
        state["final_response"] = (
            "I couldn't read your agent registry right now. Please try again."
        )
        return state

def understand_request(
    state: ManagerState,
) -> ManagerState:
    """
    Phase 1 request understanding.

    Produces a validated intent + requirements contract (see
    planner.understand_request) and records it on state. This node
    NEVER decides what to execute -- it only describes the request.
    Routing is a separate, deterministic step.

    Two failure modes are preserved distinctly rather than collapsed:
      CLASSIFICATION_FAILED - no usable answer from the classifier
      UNKNOWN_INTENT        - an answer, but outside the contract

    Neither is converted into GENERAL_ANSWER. Doing so would turn an
    LLM outage into a confident-looking small-talk reply, and would
    erase the distinction the failure model depends on.
    """

    # Routing introspection is answered from persisted execution metadata,
    # never from the LLM.
    if _is_routing_introspection(
        state.get("user_input") or ""
    ):
        state["intent"] = ManagerIntent.GENERAL_ANSWER.value
        state["intent_confidence"] = 1.0
        state["intent_reason"] = "Routing introspection request"
        state["requirements"] = {
            "routing_introspection": True,
            "requires_agent": False,
            "requires_multiple_agents": False,
            "requires_rag": False,
            "requires_document": False,
            "requires_knowledge_base": False,
            "requires_external_tool": False,
            "requires_approval": False,
            "required_capabilities": [],
        }
        _sync_legacy_request_type(state)
        return state

    # Read-only Knowledge Base and Tool Registry questions are answered from
    # live platform state, not model memory.
    if _looks_like_knowledge_registry_query(state):
        state["intent"] = ManagerIntent.GENERAL_ANSWER.value
        state["intent_confidence"] = 1.0
        state["intent_reason"] = "Deterministic Knowledge Base Registry query"
        state["requirements"] = {
            "knowledge_registry_query": True,
            "requires_agent": False,
            "requires_multiple_agents": False,
            "requires_rag": False,
            "requires_document": False,
            "requires_knowledge_base": False,
            "requires_external_tool": False,
            "requires_approval": False,
            "required_capabilities": [],
        }
        _sync_legacy_request_type(state)
        return state

    if _looks_like_tool_registry_query(state):
        state["intent"] = ManagerIntent.GENERAL_ANSWER.value
        state["intent_confidence"] = 1.0
        state["intent_reason"] = "Deterministic Tool Registry query"
        state["requirements"] = {
            "tool_registry_query": True,
            "requires_agent": False,
            "requires_multiple_agents": False,
            "requires_rag": False,
            "requires_document": False,
            "requires_knowledge_base": False,
            "requires_external_tool": False,
            "requires_approval": False,
            "required_capabilities": [],
        }
        _sync_legacy_request_type(state)
        return state

    # A capability question such as "Can you create an agent?" must not
    # trigger the agent-creation lifecycle. It is a read-only question about
    # what the Manager can do. Actual creation requests (for example,
    # "Create a SQL agent" or "Can you create an agent for SQL?") continue
    # through the lifecycle classifier below.
    if _looks_like_agent_creation_capability_question(state):
        state["intent"] = ManagerIntent.GENERAL_ANSWER.value
        state["intent_confidence"] = 1.0
        state["intent_reason"] = "Manager agent-creation capability question"
        state["requirements"] = {
            "manager_creation_capability_question": True,
            "requested_agent": None,
            "requires_agent": False,
            "requires_multiple_agents": False,
            "requires_rag": False,
            "requires_document": False,
            "requires_knowledge_base": False,
            "requires_external_tool": False,
            "requires_approval": False,
            "required_capabilities": [],
        }
        _sync_legacy_request_type(state)
        return state

    # Read-only questions about this user's registered/child agents must be
    # grounded in the live Agent Registry instead of answered from model memory.
    if _looks_like_agent_registry_query(state):
        state["intent"] = ManagerIntent.AGENT_QUERY.value
        state["intent_confidence"] = 1.0
        state["intent_reason"] = "Deterministic Agent Registry query"
        state["requirements"] = {
            "requested_agent": None,
            "requires_agent": False,
            "requires_multiple_agents": False,
            "requires_rag": False,
            "requires_document": False,
            "requires_knowledge_base": False,
            "requires_external_tool": False,
            "requires_approval": False,
            "required_capabilities": [],
        }
        _sync_legacy_request_type(state)
        return state

    # Deterministic platform routes happen before the LLM classifier.
    # Provider issues must not break obvious Calculator, Web Search, or
    # Document Reader requests.
    if _looks_like_simple_calculation(
        state.get("user_input") or ""
    ):
        state["intent"] = ManagerIntent.AGENT_EXECUTION.value
        state["intent_confidence"] = 1.0
        state["intent_reason"] = "Deterministic arithmetic request"
        state["requirements"] = {
            "requested_agent": None,
            "requires_agent": True,
            "requires_multiple_agents": False,
            "requires_rag": False,
            "requires_document": False,
            "requires_knowledge_base": False,
            "requires_external_tool": True,
            "requires_approval": False,
            "required_capabilities": ["calculation"],
        }
        _sync_legacy_request_type(state)
        return state

    if _looks_like_current_information(
        state.get("user_input") or ""
    ):
        state["intent"] = ManagerIntent.AGENT_EXECUTION.value
        state["intent_confidence"] = 1.0
        state["intent_reason"] = "Deterministic current-information request"
        state["requirements"] = {
            "requested_agent": None,
            "requires_agent": True,
            "requires_multiple_agents": False,
            "requires_rag": False,
            "requires_document": False,
            "requires_knowledge_base": False,
            "requires_external_tool": True,
            "requires_approval": False,
            "required_capabilities": ["web_search"],
        }
        _sync_legacy_request_type(state)
        return state

    if _should_route_to_rag(state):
        named_kb_ids = _resolve_named_document_kb_ids(state)

        if named_kb_ids:
            state["conversation_knowledge_base_ids"] = named_kb_ids

        state["intent"] = ManagerIntent.RAG_QUERY.value
        state["intent_confidence"] = 1.0
        state["intent_reason"] = "Deterministic document/RAG request"
        state["requirements"] = {
            "requested_agent": None,
            "requires_agent": True,
            "requires_multiple_agents": False,
            "requires_rag": True,
            "requires_document": True,
            "requires_knowledge_base": True,
            "requires_external_tool": False,
            "requires_approval": False,
            "required_capabilities": ["document_question_answering"],
        }
        _sync_legacy_request_type(state)
        return state

    # --------------------------------------------------------------
    # Existing specialized-agent fast path
    # --------------------------------------------------------------
    # The registry must be considered before treating a request as a
    # generic direct-answer question. This prevents an active specialized
    # agent (for example HR/onboarding) from being bypassed merely because
    # the classifier could answer the question itself.
    specialized_match = (
        _find_specialized_registry_agent(
            state
        )
    )

    if specialized_match is not None:
        specialized_agent, specialized_capability = (
            specialized_match
        )

        state["intent"] = (
            ManagerIntent.AGENT_EXECUTION.value
        )
        state["intent_confidence"] = 1.0
        state["intent_reason"] = (
            "Strong existing specialized-agent registry match"
        )
        state["requirements"] = {
            "requested_agent": specialized_agent.name,
            "requires_agent": True,
            "requires_multiple_agents": False,
            "requires_rag": False,
            "requires_document": False,
            "requires_knowledge_base": False,
            "requires_external_tool": False,
            "requires_approval": False,
            "required_capabilities": [
                specialized_capability
            ],
        }

        _sync_legacy_request_type(
            state
        )

        return state

    # --------------------------------------------------------------
    # Short conversational follow-up fast path
    # --------------------------------------------------------------
    # Run this only after deterministic platform routes and specialized
    # registry matching. That keeps short actionable requests eligible for
    # delegation while preventing replies such as "ledhu", "ok", or
    # "thanks" from becoming routing/planning failures.
    simple_conversation_response = _simple_conversation_response(state)
    if simple_conversation_response is not None:
        state["intent"] = ManagerIntent.GENERAL_ANSWER.value
        state["intent_confidence"] = 1.0
        state["intent_reason"] = "Deterministic conversational turn"
        state["requirements"] = {
            "simple_conversation_response": simple_conversation_response,
            "requested_agent": None,
            "requires_agent": False,
            "requires_multiple_agents": False,
            "requires_rag": False,
            "requires_document": False,
            "requires_knowledge_base": False,
            "requires_external_tool": False,
            "requires_approval": False,
            "required_capabilities": [],
        }
        _sync_legacy_request_type(state)
        return state

    has_conversation_documents = bool(
        state.get("conversation_knowledge_base_ids")
    )

    # Build the live context only when deterministic fast paths did not
    # already resolve the request.  This avoids unnecessary DB work for simple
    # greetings/calculations while grounding ambiguous classification in the
    # user's real AgentOS registry.
    state["platform_context"] = _build_manager_context_pack(state)

    understanding = planner.understand_request(
        user_input=state["user_input"],
        provider=state["provider"],
        model=state["model"],
        has_conversation_documents=has_conversation_documents,
        conversation_history=_safe_conversation_history(
            state.get("conversation_history")
        ),
        platform_context=state.get("platform_context"),
    )

    state["intent_confidence"] = understanding.confidence
    state["intent_reason"] = understanding.reason
    state["requirements"] = understanding.requirements.model_dump()

    if not understanding.ok:

        # Intent stays None on purpose: downstream routing must see
        # "not understood", not a substituted default.
        state["intent"] = None
        _sync_legacy_request_type(state)

        if understanding.failure is not None:
            _record_failure(state, understanding.failure)
            state["final_response"] = (
                understanding.failure.user_message
            )

        return state

    # A GENERAL_ANSWER classification must not bypass an existing
    # specialized agent merely because the user's wording differs from the
    # agent's literal capability text. Run one semantic registry check using
    # the existing capability extractor, then delegate only on an unambiguous
    # specialized-agent match.
    if understanding.intent == ManagerIntent.GENERAL_ANSWER:
        semantic_match = _find_semantic_specialized_registry_agent(state)

        if semantic_match is not None:
            specialized_agent, specialized_capability = semantic_match

            state["intent"] = ManagerIntent.AGENT_EXECUTION.value
            state["intent_confidence"] = max(
                float(understanding.confidence or 0.0),
                0.9,
            )
            state["intent_reason"] = (
                "Semantic existing specialized-agent capability match"
            )
            state["requirements"] = {
                "requested_agent": specialized_agent.name,
                "requires_agent": True,
                "requires_multiple_agents": False,
                "requires_rag": False,
                "requires_document": False,
                "requires_knowledge_base": False,
                "requires_external_tool": False,
                "requires_approval": False,
                "required_capabilities": [
                    specialized_capability
                ],
            }

            _sync_legacy_request_type(state)
            return state

    state["intent"] = understanding.intent.value

    _sync_legacy_request_type(state)

    logger.info(
        "manager.request_understood execution_id=%s intent=%s "
        "confidence=%.2f user_id=%s has_documents=%s",
        state.get("execution_id"),
        state["intent"],
        understanding.confidence,
        state.get("user_id"),
        has_conversation_documents,
    )

    return state


def controlled_failure(
    state: ManagerState,
) -> ManagerState:
    """
    Terminal node for requests that cannot proceed, for a reason that
    has already been recorded on state.

    This is the safety valve that makes the CRITICAL RULE enforceable:
    everything that is "we could not proceed" ends up HERE, never in
    propose_agent. Classification failures, unknown intents, planning
    failures and not-yet-implemented intents all terminate here with
    their reason intact.
    """

    state["status"] = "failed"

    if not state.get("final_response"):
        state["final_response"] = (
            "I wasn't able to process that request."
        )

    logger.info(
        "manager.controlled_failure execution_id=%s code=%s stage=%s",
        state.get("execution_id"),
        state.get("failure_code"),
        state.get("failure_stage"),
    )

    return state


def not_implemented(
    state: ManagerState,
) -> ManagerState:
    """
    Controlled placeholder for intents Phase 1 recognizes but does not
    yet execute (agent lifecycle, knowledge-base management).

    Recognizing-but-deferring is deliberate: the alternative is
    mis-classifying "create an agent that..." as ordinary agent
    execution, which sends it into capability extraction and -- when
    nothing matches -- straight into an agent proposal, i.e. acting on
    a lifecycle request through the wrong path entirely.
    """

    intent = state.get("intent") or "this"

    _record_failure(
        state,
        ManagerFailure(
            code=ManagerFailureCode.NOT_IMPLEMENTED,
            stage=ManagerStage.ROUTING,
            message=f"Intent {intent} is not implemented in Phase 1.",
            user_message=(
                "I understood what you're asking for, but managing "
                "agents and knowledge bases through chat isn't "
                "available yet. You can do this from the agents and "
                "knowledge base screens in the meantime."
            ),
            retryable=False,
        ),
    )

    state["status"] = "failed"
    state["final_response"] = (
        "I understood what you're asking for, but managing agents "
        "and knowledge bases through chat isn't available yet. You "
        "can do this from the agents and knowledge base screens in "
        "the meantime."
    )

    return state


# ---------------------------------------------------------------------
# General conversational response
# ---------------------------------------------------------------------

def direct_response(
    state: ManagerState,
) -> ManagerState:
    """First-class Manager answer path for casual/general/educational queries."""

    requirements = state.get("requirements") or {}

    simple_response = requirements.get("simple_conversation_response")
    if isinstance(simple_response, str) and simple_response.strip():
        state["final_response"] = simple_response.strip()
        state["status"] = "success"
        state["error"] = None
        state["needs_approval"] = False
        state["proposed_agent"] = None
        return state

    if requirements.get("knowledge_registry_query"):
        state["final_response"] = _knowledge_registry_response(state)
        state["status"] = "success"
        state["error"] = None
        state["needs_approval"] = False
        state["proposed_agent"] = None
        return state

    if requirements.get("tool_registry_query"):
        state["final_response"] = _tool_registry_response(state)
        state["status"] = "success"
        state["error"] = None
        state["needs_approval"] = False
        state["proposed_agent"] = None
        return state

    if requirements.get("manager_creation_capability_question"):
        state["final_response"] = (
            "Yes. I can prepare and register a new AgentOS agent, but I do not "
            "create it just because you ask whether I can. Actual creation starts "
            "only when you explicitly ask me to create an agent, and the final "
            "agent is created only after you review and approve the editable form. "
            "For a RAG agent, you can attach an existing ready Knowledge Base or "
            "create a new Knowledge Base, upload documents, and attach it before approval."
        )
        state["status"] = "success"
        state["error"] = None
        state["needs_approval"] = False
        state["proposed_agent"] = None
        return state

    if requirements.get("routing_introspection"):
        state["final_response"] = (
            _routing_introspection_response(
                state
            )
        )
        state["status"] = "success"
        state["error"] = None
        state["needs_approval"] = False
        state["proposed_agent"] = None
        return state

    try:
        history = _safe_conversation_history(state.get("conversation_history"))
        platform_context = state.get("platform_context")
        if not platform_context:
            platform_context = _build_manager_context_pack(state)
            state["platform_context"] = platform_context

        state["final_response"] = planner.generate_direct_response(
            user_input=state["user_input"],
            provider=state["provider"],
            model=state["model"],
            conversation_history=history,
            platform_context=platform_context,
        )
        state["status"] = "success"
        state["error"] = None
        state["needs_approval"] = False
        state["proposed_agent"] = None
    except Exception as exc:
        logger.exception(
            "manager.direct_response_failed execution_id=%s",
            state.get("execution_id"),
        )
        _record_failure(
            state,
            ManagerFailure(
                code=ManagerFailureCode.EXECUTION_FAILED,
                stage=ManagerStage.EXECUTION,
                message=f"Direct response failed: {exc}",
                user_message="I couldn't generate a response right now. Please try again.",
                retryable=True,
            ),
        )
        state["status"] = "failed"
        state["final_response"] = "I couldn't generate a response right now. Please try again."
    return state


# ---------------------------------------------------------------------
# Conversation document routing
# ---------------------------------------------------------------------

def _normalize_document_reference(value: str) -> str:
    """
    Normalize a user phrase or filename for safe document-name matching.

    Examples:
        "Agentic AI Platform — API Ownership & Change Rules.pdf"
        -> "agentic ai platform api ownership change rules"
    """
    cleaned = str(value or "").lower()

    # Remove the common file extension so users can omit/say "pdf".
    cleaned = re.sub(
        r"\.(pdf|docx|doc|txt|md|markdown)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"[^a-z0-9]+",
        " ",
        cleaned,
    )

    return " ".join(cleaned.split())


def _resolve_named_document_kb_ids(
    state: ManagerState,
) -> List[str]:
    """
    If the user explicitly names one of their READY documents, resolve
    that document to its owning ACTIVE Knowledge Base.

    This is intentionally deterministic and ownership-scoped. It avoids
    sending a query about one named PDF to every active KB owned by the
    user, which can make an unrelated failed document block the RAG step.

    Returns [] when no unambiguous named document is found.
    """
    user_id = state.get("user_id")
    user_input = _normalize_document_reference(
        state.get("user_input") or ""
    )

    if user_id is None or not user_input:
        return []

    rows = (
        state["db"]
        .query(
            KnowledgeDocument,
            KnowledgeBase,
        )
        .join(
            KnowledgeBase,
            KnowledgeBase.id == KnowledgeDocument.kb_id,
        )
        .join(
            KnowledgeChunk,
            KnowledgeChunk.doc_id == KnowledgeDocument.id,
        )
        .filter(
            KnowledgeBase.created_by == user_id,
            KnowledgeBase.status == KnowledgeBaseStatus.ACTIVE,
            KnowledgeDocument.status == DocumentStatus.READY,
        )
        .distinct()
        .all()
    )

    matches = []

    for document, kb in rows:
        normalized_name = _normalize_document_reference(
            getattr(document, "filename", "") or ""
        )

        if not normalized_name:
            continue

        # Prefer an explicit filename/title mention.
        if normalized_name in user_input:
            matches.append(
                (
                    len(normalized_name),
                    str(kb.id),
                    str(document.id),
                    str(document.filename),
                )
            )
            continue

        # Tolerant filename/title matching.
        #
        # This must also work for short real-world names such as:
        #   naveen-resume (1).docx -> "tell me about naveen resume"
        #
        # Ignore generic file words and numeric copy suffixes.
        ignored_name_tokens = {
            "pdf",
            "doc",
            "docx",
            "txt",
            "md",
            "markdown",
            "file",
            "document",
            "attachment",
        }

        name_tokens = {
            token
            for token in normalized_name.split()
            if (
                len(token) >= 3
                and not token.isdigit()
                and token not in ignored_name_tokens
            )
        }

        input_tokens = {
            token
            for token in user_input.split()
            if len(token) >= 3
        }

        overlap = name_tokens & input_tokens

        # Two strong filename words are enough for a short filename.
        # For longer names require at least 65% coverage.
        short_name_match = (
            2 <= len(name_tokens) <= 3
            and len(overlap) >= 2
            and len(overlap) / len(name_tokens) >= 0.66
        )

        long_name_match = (
            len(name_tokens) >= 4
            and len(overlap) >= 3
            and len(overlap) / len(name_tokens) >= 0.65
        )

        if short_name_match or long_name_match:
            matches.append(
                (
                    len(overlap) * 100 + len(normalized_name),
                    str(kb.id),
                    str(document.id),
                    str(document.filename),
                )
            )

    if not matches:
        return []

    # Longest/strongest filename match wins.
    matches.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best = matches[0]

    # If the best score is tied across different KBs/documents, leave the
    # scope unchanged rather than guessing.
    tied = [
        item
        for item in matches
        if item[0] == best[0]
    ]

    unique_targets = {
        (item[1], item[2])
        for item in tied
    }

    if len(unique_targets) != 1:
        logger.info(
            "manager.rag_document_scope_ambiguous "
            "execution_id=%s matches=%s",
            state.get("execution_id"),
            [
                {
                    "kb_id": item[1],
                    "document_id": item[2],
                    "filename": item[3],
                }
                for item in tied
            ],
        )
        return []

    logger.info(
        "manager.rag_document_scope_resolved "
        "execution_id=%s kb_id=%s document_id=%s filename=%s",
        state.get("execution_id"),
        best[1],
        best[2],
        best[3],
    )

    return [best[1]]


def _get_preferred_rag_agent(
    state: ManagerState,
) -> tuple[Optional[Agent], Optional[str]]:
    """Select the user's active Document Reader deterministically.

    Preference order:
      1. active user-owned RAG agent named "Document Reader"
      2. active user-owned agent marked is_default + is_rag
      3. any active user-owned RAG agent

    This avoids depending on capability-extraction wording for document
    questions and also works for older rows where is_default was not set.
    """
    # Load active user-owned agents first. Some older provisioned
    # Document Reader rows were stored with is_rag=False even though
    # their name/capabilities clearly identify the platform RAG agent.
    # We still prefer that registered child agent and execution.py
    # safely forces retrieval for document capabilities without
    # mutating the database row.
    query = (
        state["db"].query(Agent)
        .filter(
            Agent.status == AgentStatus.ACTIVE,
        )
    )

    if state.get("user_id") is not None:
        query = query.filter(
            Agent.created_by == state["user_id"]
        )

    agents = query.order_by(
        Agent.is_default.desc(),
        Agent.current_version.desc(),
        Agent.name.asc(),
    ).all()

    if not agents:
        return None, None

    document_reader = next(
        (
            agent
            for agent in agents
            if str(agent.name or "").strip().casefold()
            == "document reader"
        ),
        None,
    )

    rag_agents = [
        agent
        for agent in agents
        if bool(getattr(agent, "is_rag", False))
    ]

    default_rag = next(
        (
            agent
            for agent in rag_agents
            if bool(getattr(agent, "is_default", False))
        ),
        None,
    )

    if document_reader is None and not rag_agents:
        return None, None

    agent = (
        document_reader
        or default_rag
        or rag_agents[0]
    )

    best_name: Optional[str] = None
    best_score = -1

    for capability in getattr(agent, "capabilities", []) or []:
        name = str(
            getattr(capability, "capability_name", "")
            or ""
        ).strip()

        if not name:
            continue

        lowered = name.lower()
        score = (
            (8 if "document_question_answer" in lowered else 0)
            + (7 if "document_search" in lowered else 0)
            + (6 if "document_retriev" in lowered else 0)
            + (5 if lowered == "rag" else 0)
            + (3 if "document" in lowered else 0)
            + (2 if "retriev" in lowered else 0)
            + (
                1
                if any(
                    token in lowered
                    for token in ("question", "answer", "search")
                )
                else 0
            )
        )

        if score > best_score:
            best_score = score
            best_name = name

    if not best_name:
        best_name = "document_question_answering"

    return agent, best_name


def _get_document_capability_for_conversation(
    state: ManagerState,
) -> Optional[str]:
    """
    Backward-compatible capability hint used by generic planning only
    when this conversation actually has document/KB context.
    """
    if not state.get("conversation_knowledge_base_ids"):
        return None

    _, capability = _get_preferred_rag_agent(state)
    return capability


def _ready_rag_scope(
    state: ManagerState,
    candidate_ids: Optional[List[str]] = None,
) -> List[str]:
    """Return only ACTIVE, owned KBs that currently contain indexed data.

    When candidate_ids is supplied (conversation scope), that scope is
    preserved and only unusable KBs are removed. Without candidate_ids,
    all retrieval-ready KBs owned by the authenticated user are returned.
    """
    user_id = state.get("user_id")
    if user_id is None:
        return []

    query = (
        state["db"].query(KnowledgeBase)
        .filter(
            KnowledgeBase.created_by == user_id,
            KnowledgeBase.status == KnowledgeBaseStatus.ACTIVE,
        )
    )

    if candidate_ids:
        wanted = list(
            dict.fromkeys(
                str(kb_id)
                for kb_id in candidate_ids
                if kb_id
            )
        )
        if not wanted:
            return []
        query = query.filter(
            KnowledgeBase.id.in_(wanted)
        )

    knowledge_bases = query.all()

    from services.knowledge_service import knowledge_base_readiness

    ready_ids: List[str] = []
    for knowledge_base in knowledge_bases:
        readiness = knowledge_base_readiness(
            db=state["db"],
            kb_id=str(knowledge_base.id),
            user_id=user_id,
        )
        if readiness.get("ready"):
            ready_ids.append(str(knowledge_base.id))

    return ready_ids


def _should_route_to_rag(
    state: ManagerState,
) -> bool:
    """Deterministic safety net around LLM intent classification.

    A document request must not be answered by the Manager's general
    response path just because the classifier labelled it GENERAL_ANSWER.
    """
    requirements = state.get("requirements") or {}

    # A user does not have to type ".pdf" or "document".
    # If their text clearly names one of their indexed documents,
    # route through Document Reader instead of answering directly.
    if _resolve_named_document_kb_ids(state):
        return True

    if any(
        bool(requirements.get(key))
        for key in (
            "requires_rag",
            "requires_document",
            "requires_knowledge_base",
        )
    ):
        return True

    text = str(state.get("user_input") or "").casefold()
    document_markers = (
        ".pdf",
        ".docx",
        ".doc",
        ".txt",
        ".md",
        " pdf",
        " document",
        "uploaded file",
        "uploaded document",
        "attachment",
        "knowledge base",
        "knowledge-base",
        "according to the file",
        "according to the document",
        "inside this file",
        "inside this pdf",
        "summarize this file",
        "summarize this document",
        " resume",
        " cv ",
        "curriculum vitae",
    )

    if any(marker in text for marker in document_markers):
        return True

    return False


# ---------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------

def plan_request(
    state: ManagerState,
) -> ManagerState:
    """
    Extract required capabilities and dynamically discover the user's
    registered agents.

    user_id is passed to the planner so registry discovery remains
    isolated to the authenticated user.
    """

    logger.info(
        "manager.request_received "
        "execution_id=%s user_id=%s",
        state.get("execution_id"),
        state.get("user_id"),
    )

    requirements = state.get("requirements") or {}

    if _looks_like_simple_calculation(
        state.get("user_input") or ""
    ):
        calculator_agent = _select_existing_agent(
            state,
            preferred_name="Calculator",
            capability_terms=[
                "calculation",
                "mathematics",
                "arithmetic",
            ],
        )

        if calculator_agent is None:
            state["planning_status"] = "failed"
            state["no_agents_found"] = True
            state["capability_gap"] = False
            state["error"] = "The Calculator agent is not available."
            state["failure_code"] = (
                ManagerFailureCode.AGENT_UNAVAILABLE.value
            )
            return state

        return _single_agent_plan(
            state,
            agent=calculator_agent,
            step_id="calculator",
            capability="calculation",
        )

    if _looks_like_current_information(
        state.get("user_input") or ""
    ):
        web_agent = _select_existing_agent(
            state,
            preferred_name="Web Search",
            capability_terms=[
                "web_search",
                "internet_search",
                "current_information",
            ],
        )

        if web_agent is None:
            state["planning_status"] = "failed"
            state["no_agents_found"] = True
            state["capability_gap"] = False
            state["error"] = "The Web Search agent is not available."
            state["failure_code"] = (
                ManagerFailureCode.AGENT_UNAVAILABLE.value
            )
            return state

        return _single_agent_plan(
            state,
            agent=web_agent,
            step_id="web_search",
            capability="web_search",
        )

    requested = requirements.get("requested_agent")
    if requested:
        from .routing import resolve_explicit_agent

        resolution = resolve_explicit_agent(
            state["db"],
            requested,
            state.get("user_id"),
            bool(
                state["requirements"].get(
                    "requires_rag"
                )
            ),
        )

        if not resolution.selected:
            state["planning_status"] = "failed"
            state["capability_gap"] = False
            state["failure_code"] = (
                ManagerFailureCode.AGENT_UNAVAILABLE
                if resolution.candidates
                else ManagerFailureCode.AGENT_NOT_FOUND
            ).value
            state["error"] = (
                "Requested agent is missing or unavailable"
            )
            return state

        selected = resolution.selected

        # Resolve the real Agent row so the standard single-agent plan
        # carries name/prompt/provider/model metadata consistently.
        selected_agent = (
            state["db"]
            .query(Agent)
            .filter(
                Agent.id == selected.agent_id,
                Agent.status == AgentStatus.ACTIVE,
                Agent.created_by == state.get("user_id"),
            )
            .first()
        )

        if selected_agent is None:
            state["planning_status"] = "failed"
            state["capability_gap"] = False
            state["failure_code"] = (
                ManagerFailureCode.AGENT_UNAVAILABLE.value
            )
            state["error"] = (
                "Requested agent is missing or unavailable"
            )
            return state

        required_capabilities = (
            _safe_string_list(
                requirements.get(
                    "required_capabilities"
                )
            )
        )

        capability = (
            required_capabilities[0]
            if required_capabilities
            else str(requested)
        )

        return _single_agent_plan(
            state,
            agent=selected_agent,
            step_id="requested",
            capability=capability,
        )

    intent_name = str(state.get("intent") or "").upper()

    # A RAG/document query should use one RAG agent, not create one
    # execution step per RAG capability. This makes the default
    # Document Reader the deterministic path for KB questions.
    rag_request = (
        intent_name in {"RAG_QUERY", "DOCUMENT_OPERATION"}
        or bool(requirements.get("requires_rag"))
        or bool(requirements.get("requires_document"))
        or bool(requirements.get("requires_knowledge_base"))
        or _should_route_to_rag(state)
    )

    if rag_request:
        # An explicitly named READY document is a stronger scope signal than
        # a broad conversation attachment list, so it narrows retrieval to
        # the KB that owns that document.
        named_kb_ids = _resolve_named_document_kb_ids(state)

        if named_kb_ids:
            state["conversation_knowledge_base_ids"] = named_kb_ids
        else:
            existing_scope = _safe_string_list(
                state.get("conversation_knowledge_base_ids")
            )
            ready_scope = _ready_rag_scope(
                state,
                existing_scope or None,
            )

            # If we found usable KBs, pass only those to retrieval so one
            # empty/failed KB cannot block healthy document sources.
            if ready_scope:
                state["conversation_knowledge_base_ids"] = ready_scope

        rag_agent, rag_capability = _get_preferred_rag_agent(state)

        if rag_agent is None:
            state["planning_status"] = "failed"
            state["capability_gap"] = False
            state["no_agents_found"] = True
            state["missing_capabilities"] = []
            state["unavailable_capabilities"] = ["rag"]
            state["failure_code"] = ManagerFailureCode.AGENT_UNAVAILABLE.value
            state["failure_stage"] = ManagerStage.ROUTING.value
            state["failure_retryable"] = True
            state["error"] = "No active user-owned RAG agent is available"
            state["final_response"] = (
                "Your document/knowledge-base agent is not available right now."
            )
            state["plan"] = ExecutionPlan(
                request=state["user_input"],
                steps=[],
            ).model_dump()
            state["step_results"] = {}
            state["completed"] = []
            state["skipped"] = []
            state["iteration"] = 0
            return state

        rag_step = ExecutionStep(
            step_id="rag_query",
            capability=rag_capability,
            agent_id=str(rag_agent.id),
            agent_name=rag_agent.name,
            agent_version=rag_agent.current_version,
            task=state["user_input"],
            system_prompt=rag_agent.system_prompt,
            provider=rag_agent.provider,
            model=rag_agent.model,
        )

        state["plan"] = ExecutionPlan(
            request=state["user_input"],
            steps=[rag_step],
        ).model_dump()
        state["planning_status"] = "ok"
        state["no_agents_found"] = False
        state["capability_gap"] = False
        state["missing_capabilities"] = []
        state["unavailable_capabilities"] = []
        state["unmatched_description"] = None
        state["step_results"] = {}
        state["completed"] = []
        state["skipped"] = []
        state["iteration"] = 0

        logger.info(
            "manager.rag_agent_selected execution_id=%s agent_id=%s "
            "agent_name=%s capability=%s kb_ids=%s",
            state.get("execution_id"),
            rag_agent.id,
            rag_agent.name,
            rag_capability,
            state.get("conversation_knowledge_base_ids"),
        )

        return state

    conversation_document_capability = _get_document_capability_for_conversation(state)

    try:

        extraction = planner.extract_capabilities(
            db=state["db"],
            user_input=state["user_input"],
            provider=state["provider"],
            model=state["model"],
            user_id=state.get("user_id"),
            conversation_document_capability=conversation_document_capability,
        )

    except Exception as exc:
        logger.exception(
            "manager.capability_extraction_failed "
            "execution_id=%s error=%s",
            state.get("execution_id"),
            exc,
        )

        # Provider/classifier failure must not block an obvious
        # external-information request when a Web Search agent already
        # exists in the user's registry.
        if _looks_like_current_information(
            state.get("user_input") or ""
        ):
            web_agent = _select_existing_agent(
                state,
                preferred_name="Web Search",
                capability_terms=[
                    "web_search",
                    "internet_search",
                    "current_information",
                ],
            )

            if web_agent is not None:
                return _single_agent_plan(
                    state,
                    agent=web_agent,
                    step_id="web_search",
                    capability="web_search",
                )

        state["planning_status"] = "failed"
        state["capability_gap"] = False
        state["failure_code"] = ManagerFailureCode.PLANNING_FAILED.value
        state["error"] = "Capability extraction failed"
        return state

    try:

        required_capabilities = _safe_string_list(
            getattr(
                extraction,
                "required_capabilities",
                None,
            )
        )

        if conversation_document_capability:
            logger.info(
                "manager.conversation_document_capability_available "
                "execution_id=%s capability=%s kb_ids=%s selected=%s",
                state.get("execution_id"),
                conversation_document_capability,
                state.get("conversation_knowledge_base_ids"),
                conversation_document_capability in required_capabilities,
            )

        unmatched_description = getattr(
            extraction,
            "unmatched_description",
            None,
        )

        logger.info(
            "manager.capabilities_extracted "
            "execution_id=%s required=%s",
            state.get("execution_id"),
            required_capabilities,
        )

        if not required_capabilities and not (unmatched_description or "").strip():
            raise ValueError("Empty extraction does not establish a capability gap")

        if not required_capabilities:
            # Route the named unmet requirement against the real registry too.
            # An LLM declining its shortlist is not proof that providers are absent.
            extraction.required_capabilities = [unmatched_description.strip()]
            extraction.unmatched_description = None
            required_capabilities = extraction.required_capabilities

        try:

            plan = planner.build_plan(
                db=state["db"],
                user_input=state["user_input"],
                extraction=extraction,
                user_id=state.get("user_id"),
                requires_rag=bool(
                    (state.get("requirements") or {}).get(
                        "requires_rag"
                    )
                ),
            )

        except Exception:
            raise

        plan_steps = _safe_list(
            getattr(
                plan,
                "steps",
                None,
            )
        )

        unmatched_capabilities = _safe_string_list(
            getattr(
                plan,
                "unmatched_capabilities",
                None,
            )
        )

        state["plan"] = {
            "request": getattr(
                plan,
                "request",
                state["user_input"],
            ),
            "steps": [
                step.model_dump()
                if hasattr(step, "model_dump")
                else step
                for step in plan_steps
            ],
            "unmatched_capabilities":
                unmatched_capabilities,
            "unavailable_capabilities":
                _safe_string_list(
                    getattr(plan, "unavailable_capabilities", None)
                ),
            "execution_mode": (
                getattr(
                    plan,
                    "execution_mode",
                    "parallel",
                )
                or "parallel"
            ),
        }

        state["step_results"] = {}
        state["completed"] = []
        state["skipped"] = []
        state["iteration"] = 0

        # Planning itself SUCCEEDED here -- the planner produced a
        # valid answer. Whether that answer contains usable steps is a
        # separate question, tracked below. Keeping these two apart is
        # the whole point of planning_status vs capability_gap.
        unavailable_capabilities = _safe_string_list(
            getattr(plan, "unavailable_capabilities", None)
        )

        if not plan_steps:

            state["planning_status"] = "no_capability"
            state["no_agents_found"] = True
            state["unavailable_capabilities"] = unavailable_capabilities

            # AGENT_UNAVAILABLE is NOT a capability gap (section 14).
            # If every capability this request needs DOES have a
            # registered provider, and those providers were merely
            # ineligible (disabled, wrong owner, missing RAG support),
            # then the platform already has the capability -- it just
            # cannot run it right now. Proposing a new agent here
            # would ask the user to build a duplicate of an agent they
            # already own.
            if unavailable_capabilities:

                state["capability_gap"] = False
                state["missing_capabilities"] = []

                _record_failure(
                    state,
                    ManagerFailure(
                        code=ManagerFailureCode.AGENT_UNAVAILABLE,
                        stage=ManagerStage.ROUTING,
                        message=(
                            "Capabilities have registered providers, "
                            "but none are currently usable: "
                            + ", ".join(unavailable_capabilities)
                        ),
                        user_message=(
                            "The agent that handles this isn't "
                            "available right now. Please try again "
                            "later."
                        ),
                        retryable=True,
                    ),
                )

                state["final_response"] = (
                    "The agent that handles this isn't available "
                    "right now. Please try again later."
                )

                state["plan"] = ExecutionPlan(
                    request=state["user_input"],
                    steps=[],
                ).model_dump()

                state["step_results"] = {}
                state["completed"] = []
                state["skipped"] = []
                state["iteration"] = 0

                return state

            state["unmatched_description"] = (
                unmatched_description
                or ", ".join(
                    unmatched_capabilities
                )
                or state["user_input"]
            )

            state["missing_capabilities"] = (
                unmatched_capabilities
                or _safe_string_list(required_capabilities)
            )

            # A capability gap requires ALL of: understood request,
            # not answerable directly (we are past understanding, so
            # intent is an execution intent), an actionable
            # requirement, and nothing in the registry that satisfies
            # it. Phase 1 only RECORDS this -- propose_agent still
            # drafts a proposal and still requires explicit approval
            # before anything is created.
            state["capability_gap"] = bool(
                state.get("missing_capabilities")
                or state.get("unmatched_description")
            )

        else:

            # PARTIAL COVERAGE (Phase 2 fix).
            #
            # Previously ANY unmatched capability set no_agents_found
            # and capability_gap, which routed the whole request to
            # propose_agent and THREW AWAY the steps that did resolve.
            # A request needing A+B+C where only C is missing would
            # execute nothing at all and ask to create an agent.
            #
            # Now the resolved steps execute, and the unmatched
            # capabilities are recorded for the Phase 4 lifecycle
            # workflow to consider separately. Routing detects gaps;
            # it does not resolve them.
            state["planning_status"] = "ok"
            state["no_agents_found"] = False
            state["capability_gap"] = False

            state["missing_capabilities"] = unmatched_capabilities
            state["unavailable_capabilities"] = _safe_string_list(
                getattr(plan, "unavailable_capabilities", None)
            )

            state["unmatched_description"] = (
                ", ".join(unmatched_capabilities)
                if unmatched_capabilities
                else None
            )

            if unmatched_capabilities:
                logger.info(
                    "manager.partial_routing execution_id=%s "
                    "matched_steps=%s missing=%s",
                    state.get("execution_id"),
                    len(plan_steps),
                    unmatched_capabilities,
                )

        return state

    except Exception as exc:

        logger.exception(
            "manager.plan_request_failed "
            "execution_id=%s user_id=%s",
            state.get("execution_id"),
            state.get("user_id"),
        )

        # PLANNING_FAILED, explicitly NOT a capability gap.
        #
        # planner.extract_capabilities() re-raises genuine failures
        # (LLM timeout, unparseable output, DB error) precisely so they
        # arrive here and can be told apart from "the registry has
        # nothing matching". capability_gap stays False, so
        # route_after_plan() sends this to controlled_failure and
        # propose_agent is unreachable.
        state["planning_status"] = "failed"
        state["capability_gap"] = False
        state["missing_capabilities"] = []
        state["no_agents_found"] = False

        state["unmatched_description"] = None

        state["plan"] = ExecutionPlan(
            request=state["user_input"],
            steps=[],
        ).model_dump()

        state["step_results"] = {}
        state["completed"] = []
        state["skipped"] = []
        state["iteration"] = 0

        _record_failure(
            state,
            ManagerFailure(
                code=ManagerFailureCode.PLANNING_FAILED,
                stage=ManagerStage.PLANNING,
                message=f"Planning failed: {exc}",
                user_message=(
                    "I ran into a problem while working out how to "
                    "handle that request. Please try again."
                ),
                retryable=True,
            ),
        )

        state["final_response"] = (
            "I ran into a problem while working out how to handle "
            "that request. Please try again."
        )

        return state


# ---------------------------------------------------------------------
# New agent proposal
# ---------------------------------------------------------------------

def _proposal_dict(
    proposal: Any,
) -> Dict[str, Any]:
    if hasattr(proposal, "model_dump"):
        return proposal.model_dump(mode="json")

    if isinstance(proposal, dict):
        return dict(proposal)

    return {}


def _gap_strategy_from_query(
    state: ManagerState,
    proposal_data: Dict[str, Any],
) -> str:
    """
    Decide whether a genuine missing capability should be handled by:

        temporary  -> execute once, do NOT save Agent row
        persistent -> create/register once, then execute/reuse later

    Safety behavior:
    - Strong one-off wording => temporary.
    - Strong recurring/reusable wording => persistent.
    - Otherwise ask the selected LLM for a narrow classification.
    - Any classifier/provider/parsing failure => temporary.

    This function runs ONLY after a real capability gap has already
    been established by planning. It is never used for execution/tool/
    provider/RAG/timeout failures.
    """
    text = f" {str(state.get('user_input') or '').strip().casefold()} "

    persistent_markers = (
        " every time ",
        " whenever ",
        " from now on ",
        " going forward ",
        " for future ",
        " in future ",
        " regularly ",
        " recurring ",
        " repeatedly ",
        " ongoing ",
        " always ",
        " reuse ",
        " reusable ",
        " monitor ",
        " continuously ",
        " each time ",
        " every day ",
        " every week ",
        " every month ",
        " for all ",
    )

    temporary_markers = (
        " just this ",
        " only this ",
        " one time ",
        " one-time ",
        " once ",
        " for now ",
        " this error ",
        " this issue ",
        " this file ",
        " this request ",
        " this message ",
        " this response ",
    )

    if any(marker in text for marker in temporary_markers):
        return "temporary"

    if any(marker in text for marker in persistent_markers):
        return "persistent"

    # If proposal generation itself fell back to a generic proposal,
    # never persist that generic fallback automatically.
    reason = str(
        proposal_data.get("reason")
        or ""
    ).casefold()

    if "fallback" in reason:
        return "temporary"

    capabilities = _safe_string_list(
        proposal_data.get("capabilities")
    )

    classifier_prompt = (
        "Classify how an AgentOS manager should resolve a genuine "
        "missing capability.\n\n"
        "Return ONLY JSON with this shape:\n"
        '{"strategy":"temporary|persistent","confidence":0.0,'
        '"reason":"short reason"}\n\n'
        "Choose persistent ONLY when the missing capability is a broad, "
        "reusable capability that is likely to be useful for future "
        "requests beyond this exact task. Choose temporary for one-off, "
        "narrow, ad-hoc, or query-specific work. Do not choose persistent "
        "just because the task is difficult.\n\n"
        f"User request: {state.get('user_input')}\n"
        f"Missing capabilities: {capabilities}\n"
        f"Candidate worker name: {proposal_data.get('name')}\n"
        f"Candidate description: {proposal_data.get('description')}\n"
    )

    try:
        client = get_llm_client(
            state.get("provider")
            or "gemini"
        )

        raw = client.generate(
            system_prompt=(
                "You are a conservative AgentOS lifecycle classifier. "
                "Prevent agent-registry pollution. Persistent creation "
                "must be reserved for genuinely reusable capabilities."
            ),
            user_input=classifier_prompt,
            model_name=state.get("model"),
        )

        # Reuse the planner's existing robust JSON parser.
        data = planner._parse_json_object(raw)

        strategy = str(
            data.get("strategy")
            or ""
        ).strip().casefold()

        try:
            confidence = float(
                data.get("confidence")
                or 0.0
            )
        except (TypeError, ValueError):
            confidence = 0.0

        # Fail closed: ambiguous decisions stay temporary.
        if (
            strategy == "persistent"
            and confidence >= 0.75
        ):
            return "persistent"

    except Exception as exc:
        logger.warning(
            "manager.gap_strategy_classifier_failed "
            "execution_id=%s error=%s",
            state.get("execution_id"),
            exc,
        )

    return "temporary"


def _append_gap_worker_step(
    state: ManagerState,
    *,
    proposal_data: Dict[str, Any],
    strategy: str,
    agent: Optional[Agent] = None,
) -> ManagerState:
    """
    Append the gap-resolution worker to the existing plan.

    Keeping the previous steps means partial-coverage requests can first
    run the agents that already exist, then run the newly resolved gap.
    """
    raw_plan = dict(
        state.get("plan")
        or {}
    )

    raw_steps = [
        dict(step)
        for step in _safe_list(
            raw_plan.get("steps")
        )
        if isinstance(step, dict)
    ]

    existing_ids = {
        str(
            step.get("step_id")
            or step.get("id")
            or ""
        ).strip()
        for step in raw_steps
    }

    counter = 1

    while True:
        step_id = f"gap_worker_{counter}"
        if step_id not in existing_ids:
            break
        counter += 1

    successful_dependencies = [
        step_id
        for step_id, value in (
            state.get("step_results")
            or {}
        ).items()
        if isinstance(value, dict)
        and value.get("status") == "success"
    ]

    capabilities = (
        _safe_string_list(
            state.get("missing_capabilities")
        )
        or _safe_string_list(
            proposal_data.get("capabilities")
        )
    )

    capability = (
        capabilities[0]
        if capabilities
        else str(
            state.get("unmatched_description")
            or "temporary_task"
        )
    )

    provider = (
        str(state.get("provider") or "").strip()
        or str(proposal_data.get("provider") or "").strip()
        or "gemini"
    )

    model = (
        str(state.get("model") or "").strip()
        or str(proposal_data.get("model") or "").strip()
        or None
    )

    if agent is not None:
        step = ExecutionStep(
            step_id=step_id,
            capability=capability,
            agent_id=str(agent.id),
            agent_name=agent.name,
            agent_version=agent.current_version,
            task=state["user_input"],
            depends_on=successful_dependencies,
            system_prompt=agent.system_prompt,
            provider=agent.provider,
            model=agent.model,
        )
    else:
        worker_name = str(
            proposal_data.get("name")
            or "Specialized Worker"
        ).strip()

        if not worker_name.casefold().startswith("temporary "):
            worker_name = f"Temporary {worker_name}"

        step = ExecutionStep(
            step_id=step_id,
            capability=capability,
            agent_id=None,
            agent_name=worker_name,
            agent_version=None,
            task=state["user_input"],
            depends_on=successful_dependencies,
            system_prompt=(
                str(
                    proposal_data.get("system_prompt")
                    or ""
                ).strip()
                or (
                    "You are a temporary specialized worker. "
                    "Complete the assigned task accurately and return "
                    "only the useful result. Do not claim capabilities "
                    "or tools that were not provided."
                )
            ),
            provider=provider,
            model=model,
        )

    raw_steps.append(
        step.model_dump()
    )

    raw_plan["request"] = (
        raw_plan.get("request")
        or state["user_input"]
    )
    raw_plan["steps"] = raw_steps
    raw_plan["unmatched_capabilities"] = []
    raw_plan["unavailable_capabilities"] = []
    raw_plan["execution_mode"] = (
        "sequential"
        if successful_dependencies
        else "parallel"
    )

    state["plan"] = raw_plan
    state["planning_status"] = "ok"
    state["no_agents_found"] = False
    state["capability_gap"] = False
    state["missing_capabilities"] = []
    state["unavailable_capabilities"] = []
    state["unmatched_description"] = None
    state["needs_approval"] = False
    state["proposed_agent"] = None
    state["gap_resolution_strategy"] = strategy
    state["error"] = None
    state["failure_code"] = None
    state["failure_stage"] = None
    state["failure_retryable"] = False
    state["status"] = "running"

    logger.info(
        "manager.capability_gap_resolved "
        "execution_id=%s strategy=%s worker=%s agent_id=%s",
        state.get("execution_id"),
        strategy,
        step.agent_name,
        step.agent_id,
    )

    return state


def _create_persistent_gap_agent(
    state: ManagerState,
    proposal_data: Dict[str, Any],
) -> Agent:
    """
    Create one private user-owned reusable agent for a validated
    capability gap.

    Auto-created agents start with NO tools by default. Tool access can
    be granted later through the normal Agent configuration UI/API.
    """
    from schemas.agent import (
        AgentCreate,
        ModelConfig,
    )
    from services.agent_service import (
        create_agent,
    )

    user_id = state.get("user_id")

    if user_id is None:
        raise ValueError(
            "Persistent auto-creation requires an authenticated user."
        )

    name = str(
        proposal_data.get("name")
        or "Specialized Agent"
    ).strip()

    # Avoid duplicating an exact-name user agent because of a race or
    # stale registry read.
    existing = (
        state["db"]
        .query(Agent)
        .filter(
            Agent.created_by == user_id,
            Agent.name == name,
        )
        .first()
    )

    if existing is not None:
        if existing.status == AgentStatus.ACTIVE:
            return existing

        raise ValueError(
            f"Agent '{name}' already exists but is not active."
        )

    capabilities = (
        _safe_string_list(
            proposal_data.get("capabilities")
        )
        or _safe_string_list(
            state.get("missing_capabilities")
        )
    )

    if not capabilities:
        raise ValueError(
            "Persistent creation requires at least one capability."
        )

    provider = (
        str(state.get("provider") or "").strip().lower()
        or "gemini"
    )

    model = (
        str(state.get("model") or "").strip()
        or str(proposal_data.get("model") or "").strip()
    )

    if not model:
        raise ValueError(
            "Persistent creation requires a model."
        )

    agent_in = AgentCreate(
        name=name,
        description=proposal_data.get("description"),
        system_prompt=(
            str(
                proposal_data.get("system_prompt")
                or ""
            ).strip()
            or (
                "You are a specialized agent. "
                "Complete requests for your registered capabilities "
                "accurately and concisely."
            )
        ),
        capabilities=capabilities,
        input_schema=proposal_data.get("input_schema"),
        output_schema=proposal_data.get("output_schema"),
        model_cfg=ModelConfig(
            provider=provider,
            model=model,
        ),
        # Least privilege: dynamically created agents do not receive
        # every platform tool automatically.
        tools=[],
        is_rag=False,
        knowledge_base_id=None,
        visibility="private",
        requires_approval=False,
    )

    return create_agent(
        db=state["db"],
        agent_in=agent_in,
        created_by=user_id,
        is_default=False,
    )


def propose_agent(
    state: ManagerState,
) -> ManagerState:
    """
    Resolve a GENUINE capability gap.

    Hybrid behavior:
      - one-off/narrow work -> temporary worker, not persisted
      - broad reusable work -> create/register persistent child agent

    This node is unreachable for planning/provider/tool/RAG/timeout
    failures because route_after_plan() only sends validated
    capability_gap=True here.
    """
    if not state.get("capability_gap"):

        logger.warning(
            "manager.gap_resolution_blocked_no_capability_gap "
            "execution_id=%s planning_status=%s",
            state.get("execution_id"),
            state.get("planning_status"),
        )

        state["no_agents_found"] = True
        state["needs_approval"] = False
        state["proposed_agent"] = None
        state["error"] = (
            "Capability-gap resolution was requested without "
            "a validated capability gap."
        )
        return state

    logger.info(
        "manager.capability_gap_resolution_started "
        "execution_id=%s user_id=%s missing=%s",
        state.get("execution_id"),
        state.get("user_id"),
        state.get("missing_capabilities"),
    )

    try:
        proposal = planner.propose_new_agent(
            db=state["db"],
            user_input=state["user_input"],
            unmatched_description=(
                state.get("unmatched_description")
                or ", ".join(
                    _safe_string_list(
                        state.get("missing_capabilities")
                    )
                )
                or state["user_input"]
            ),
            provider=state["provider"],
            model=state["model"],
            user_id=state.get("user_id"),
        )

        proposal_data = _proposal_dict(
            proposal
        )

        # RAG lifecycle requires a ready/authorized knowledge source.
        # Do not silently create an ungrounded RAG agent or temporary
        # worker in its place.
        if bool(
            (state.get("requirements") or {}).get(
                "requires_rag"
            )
        ):
            state["capability_gap"] = False
            state["no_agents_found"] = True
            state["needs_approval"] = False
            state["proposed_agent"] = None
            state["error"] = (
                "A RAG capability gap needs a ready authorized "
                "Knowledge Base before an agent can be created."
            )
            state["final_response"] = (
                "I found a missing RAG capability, but I won't create "
                "an ungrounded agent. Attach or prepare a ready Knowledge "
                "Base first."
            )
            return state

        strategy = _gap_strategy_from_query(
            state,
            proposal_data,
        )

        if strategy == "persistent":
            try:
                created_agent = (
                    _create_persistent_gap_agent(
                        state,
                        proposal_data,
                    )
                )

                state["auto_created_agent_id"] = str(
                    created_agent.id
                )

                return _append_gap_worker_step(
                    state,
                    proposal_data=proposal_data,
                    strategy="persistent",
                    agent=created_agent,
                )

            except Exception as exc:
                # Creation failure must not cause repeated agent creation.
                # Fall back to an ephemeral worker for this one request.
                logger.warning(
                    "manager.persistent_gap_agent_creation_failed "
                    "execution_id=%s error=%s falling_back=temporary",
                    state.get("execution_id"),
                    exc,
                )

        state["auto_created_agent_id"] = None

        return _append_gap_worker_step(
            state,
            proposal_data=proposal_data,
            strategy="temporary",
            agent=None,
        )

    except Exception as exc:

        logger.exception(
            "manager.capability_gap_resolution_failed "
            "execution_id=%s",
            state.get("execution_id"),
        )

        state["capability_gap"] = False
        state["no_agents_found"] = True
        state["needs_approval"] = False
        state["proposed_agent"] = None
        state["error"] = str(exc)
        state["final_response"] = (
            "I found a capability gap but couldn't safely resolve it."
        )
        return state



def explicit_agent_create(
    state: ManagerState,
) -> ManagerState:
    """
    Handle an explicit user request to create an agent.

    This path is intentionally different from capability-gap proposals:
    an explicit AGENT_CREATE request does not need capability_gap=True,
    but it still NEVER creates the database row directly.  It prepares
    an editable proposal and returns pending_agent_approval so the
    existing Manager service approval endpoint can perform creation.
    """
    logger.info(
        "manager.explicit_agent_create_started execution_id=%s user_id=%s",
        state.get("execution_id"),
        state.get("user_id"),
    )

    try:
        proposal = planner.propose_new_agent(
            db=state["db"],
            user_input=state["user_input"],
            unmatched_description=(
                (state.get("requirements") or {}).get("description")
                or state["user_input"]
            ),
            provider=state["provider"],
            model=state["model"],
            user_id=state.get("user_id"),
        )

        from .schemas import ProposalState

        requirements = state.get("requirements") or {}
        requires_rag = bool(requirements.get("requires_rag"))

        if hasattr(proposal, "is_rag") and requires_rag:
            proposal.is_rag = True

        proposal_data = (
            proposal.model_dump(mode="json")
            if hasattr(proposal, "model_dump")
            else dict(proposal)
            if isinstance(proposal, dict)
            else {}
        )

        if requires_rag or proposal_data.get("is_rag"):
            proposal_data["is_rag"] = True

            if not proposal_data.get("knowledge_base_id"):
                proposal_data["proposal_state"] = "knowledge_source_required"
                state["needs_approval"] = False
                state["status"] = "pending_agent_approval"
                state["final_response"] = (
                    f'I prepared the RAG agent proposal '
                    f'"{proposal_data.get("name", "New Agent")}", but it needs '
                    "a Knowledge Base before approval. Select an existing ready "
                    "Knowledge Base or create one, then attach it to the proposal."
                )
            else:
                proposal_data.setdefault(
                    "proposal_state",
                    ProposalState.READY_FOR_APPROVAL.value,
                )
                state["needs_approval"] = True
                state["status"] = "pending_agent_approval"
                state["final_response"] = (
                    f'I prepared an editable proposal for '
                    f'"{proposal_data.get("name", "New Agent")}". '
                    "Review it and explicitly approve it to create the agent."
                )
        else:
            proposal_data.setdefault(
                "proposal_state",
                ProposalState.READY_FOR_APPROVAL.value,
            )
            state["needs_approval"] = True
            state["status"] = "pending_agent_approval"
            state["final_response"] = (
                f'I prepared an editable proposal for '
                f'"{proposal_data.get("name", "New Agent")}". '
                "Review it and explicitly approve it to create the agent."
            )

        state["proposed_agent"] = proposal_data
        state["error"] = None
        state["failure_code"] = None
        state["failure_stage"] = None
        state["failure_retryable"] = False

        logger.info(
            "manager.explicit_agent_create_proposed execution_id=%s name=%s rag=%s",
            state.get("execution_id"),
            proposal_data.get("name"),
            proposal_data.get("is_rag", False),
        )

        return state

    except Exception as exc:
        logger.exception(
            "manager.explicit_agent_create_failed execution_id=%s",
            state.get("execution_id"),
        )

        _record_failure(
            state,
            ManagerFailure(
                code=ManagerFailureCode.EXECUTION_FAILED,
                stage=ManagerStage.EXECUTION,
                message=f"Agent proposal generation failed: {exc}",
                user_message=(
                    "I couldn't safely build the requested agent proposal. "
                    "Please try again."
                ),
                retryable=True,
            ),
        )

        state["status"] = "failed"
        state["needs_approval"] = False
        state["proposed_agent"] = None
        state["final_response"] = (
            "I couldn't safely build the requested agent proposal. "
            "Please try again."
        )
        return state


# ---------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------

def execute_wave(state: ManagerState) -> ManagerState:
    from .execution import execute_wave as execute_tasks
    return execute_tasks(state)


def check_plan(
    state: ManagerState,
) -> str:

    raw_plan = (
        state.get("plan")
        or {}
    )

    plan = ExecutionPlan(
        **raw_plan
    )

    plan_steps = _safe_list(
        getattr(
            plan,
            "steps",
            None,
        )
    )

    done = (
        set(
            _safe_string_list(
                state.get("completed")
            )
        )
        |
        set(
            _safe_string_list(
                state.get("skipped")
            )
        )
    )

    remaining = [
        step
        for step in plan_steps
        if step.step_id not in done
    ]

    if not remaining:
        if (
            state.get("missing_capabilities")
            and not state.get("unavailable_capabilities")
        ):
            state["capability_gap"] = True
            state["unmatched_description"] = ", ".join(
                _safe_string_list(
                    state.get("missing_capabilities")
                )
            )
            return "resolve_gap"

        return "aggregate"

    if (
        state.get("iteration", 0)
        >= settings.MAX_ORCHESTRATION_STEPS
    ):

        logger.warning(
            "manager.step_limit_reached "
            "execution_id=%s remaining=%s",
            state.get("execution_id"),
            [
                step.step_id
                for step in remaining
            ],
        )

        for step in remaining:

            state["step_results"][
                step.step_id
            ] = {
                "step_id":
                    step.step_id,

                "agent_id":
                    step.agent_id,

                "capability":
                    step.capability,

                "status":
                    "failed",

                "result":
                    None,

                "error":
                    (
                        "Orchestration step limit "
                        f"({settings.MAX_ORCHESTRATION_STEPS}) "
                        "reached before this step could run."
                    ),
            }

            if (
                step.step_id
                not in state["completed"]
            ):

                state["completed"].append(
                    step.step_id
                )

        return "aggregate"

    return "continue"


# ---------------------------------------------------------------------
# Final aggregation
# ---------------------------------------------------------------------

def aggregate(
    state: ManagerState,
) -> ManagerState:

    if state.get("no_agents_found"):

        state["status"] = "failed"

        # Distinguish WHY nothing ran. A planning failure and an empty
        # registry look identical to a user if both say "no agent
        # found", but only one of them is about the registry.
        if state.get("planning_status") == "failed":

            state["final_response"] = (
                "I ran into a problem while working out how to "
                "handle that request. Please try again."
            )

            return state

        if not state.get("failure_code"):
            _record_failure(
                state,
                ManagerFailure(
                    code=ManagerFailureCode.NO_MATCHING_CAPABILITY,
                    stage=ManagerStage.PLANNING,
                    message=(
                        "No registered capability satisfies the "
                        "request."
                    ),
                    user_message=(
                        "I couldn't find a registered agent able to "
                        "handle that request."
                    ),
                    retryable=False,
                ),
            )

        state["error"] = (
            state.get("error")
            or (
                "No suitable registered capability "
                "or agent was found for this request."
            )
        )

        state["final_response"] = (
            "I couldn't find a registered agent "
            "capable of handling this request."
        )

        return state

    results = list(
        (
            state.get(
                "step_results"
            )
            or {}
        ).values()
    )

    non_skipped_results = list(results)
    for missing in (state.get("missing_capabilities") or []) + (state.get("unavailable_capabilities") or []):
        non_skipped_results.append({"step_id": "unresolved", "capability": missing,
                                    "status": "failed", "error": "Required capability was not executed"})

    statuses = [
        result.get("status")
        for result
        in non_skipped_results
    ]

    if (
        statuses
        and all(
            status == "success"
            for status in statuses
        )
    ):

        overall = "success"

    elif any(
        status == "success"
        for status in statuses
    ):

        overall = "partial"

    else:

        overall = "failed"

    state["status"] = overall

    successful_results = [
        item
        for item in non_skipped_results
        if item.get("status") == "success"
    ]

    if (
        overall == "success"
        and len(successful_results) == 1
        and len(non_skipped_results) == 1
    ):
        single_result = (
            successful_results[0].get("result")
            or {}
        )

        if isinstance(single_result, dict):
            single_output = single_result.get("output")
        else:
            single_output = single_result

        if single_output is not None:
            state["final_response"] = str(single_output)
            state["error"] = None
            return state

    try:

        history = _safe_conversation_history(
            state.get("conversation_history")
        )

        state["final_response"] = (
            planner.synthesize_final_response(
                user_input=state["user_input"],
                step_results=non_skipped_results,
                provider=state["provider"],
                model=state["model"],
                conversation_history=history,
            )
        )

    except Exception as exc:

        logger.exception(
            "manager.final_response_synthesis_failed "
            "execution_id=%s",
            state.get("execution_id"),
        )

        state["final_response"] = None

        if not state.get("error"):
            state["error"] = str(exc)

    if overall == "failed":

        failed = [
            result
            for result in non_skipped_results
            if result.get("status")
            == "failed"
        ]

        # For a single document/RAG failure, the retrieval layer already
        # provides a user-safe explanation. Return it directly instead of
        # hiding it behind "Task preparation failed" or an LLM synthesis.
        if len(failed) == 1 and failed[0].get("step_id") == "rag_query":
            rag_error = str(
                failed[0].get("error")
                or "Document retrieval failed."
            )
            state["final_response"] = rag_error

        failed_messages = [
            (
                f"{result.get('step_id', 'unknown')}: "
                f"{result.get('error', 'Execution failed.')}"
            )
            for result in failed
        ]

        state["error"] = (
            "; ".join(failed_messages)
            or state.get("error")
            or "Execution failed."
        )

    else:

        state["error"] = None

    return state


# ---------------------------------------------------------------------
# Graph routing
# ---------------------------------------------------------------------

def route_after_plan(
    state: ManagerState,
) -> str:
    """
    Deterministic post-planning routing.

    CRITICAL RULE enforcement lives here. propose_agent is reachable
    from exactly ONE condition: a genuine, recorded capability gap.

    A planning FAILURE (the planner could not produce a valid plan --
    an LLM error, unparseable output, a DB error) is explicitly NOT a
    capability gap, and routes to controlled_failure instead. Before
    Phase 1 both collapsed into `no_agents_found`, so an LLM timeout
    during capability extraction produced a "shall I create a new
    agent?" prompt for a request that was never even understood.
    """

    if state.get("planning_status") == "failed":
        return "controlled_failure"

    if state.get("capability_gap"):
        return "propose_agent"

    if state.get("no_agents_found"):
        # Understood, actionable, but nothing in the registry matches
        # and it did not qualify as a proposable gap.
        return "controlled_failure"

    return "execute_wave"


def route_after_understanding(
    state: ManagerState,
) -> str:
    """
    Deterministic routing from the understanding contract.

    The LLM chose an intent; this function -- not the LLM -- decides
    which path that intent may take. Unresolved understanding always
    terminates in controlled_failure, never in planning and never in
    agent creation.
    """

    intent_value = state.get("intent")

    if not intent_value:
        # CLASSIFICATION_FAILED / UNKNOWN_INTENT. Already recorded by
        # understand_request().
        return "controlled_failure"

    intent = ManagerIntent.coerce(intent_value)

    if intent is None or intent == ManagerIntent.UNKNOWN:

        _record_failure(
            state,
            ManagerFailure(
                code=ManagerFailureCode.UNKNOWN_INTENT,
                stage=ManagerStage.ROUTING,
                message=f"Unroutable intent {intent_value!r}.",
                user_message=(
                    "I'm not sure what you're asking for. Could you "
                    "rephrase it?"
                ),
                retryable=False,
            ),
        )

        state["final_response"] = (
            "I'm not sure what you're asking for. Could you "
            "rephrase it?"
        )

        return "controlled_failure"

    if intent == ManagerIntent.GENERAL_ANSWER:
        if _should_route_to_rag(state):
            named_kb_ids = _resolve_named_document_kb_ids(state)
            if named_kb_ids:
                state["conversation_knowledge_base_ids"] = named_kb_ids
            return "plan_request"
        return "direct_response"

    # Agent lifecycle must be resolved before generic RAG requirements.
    # For example, "create a RAG agent" has requires_rag=True but is still
    # an AGENT_CREATE lifecycle request, not a document execution request.
    intent_name = str(getattr(intent, "name", "") or "").upper()

    if intent_name in {
        "AGENT_CREATE",
        "CREATE_AGENT",
        "RAG_AGENT_CREATION",
    }:
        return "explicit_agent_create"

    requirements = state.get("requirements") or {}
    if any(
        bool(requirements.get(key))
        for key in (
            "requires_rag",
            "requires_document",
        )
    ):
        return "plan_request"

    # Agent lifecycle and Knowledge Base management are intentionally
    # kept out of a separate manager.lifecycle module. Until those
    # operations are implemented inside the existing Manager/services,
    # route them to the existing controlled not_implemented node.

    if intent_name in {
        "AGENT_QUERY",
        "QUERY_AGENT",
    }:
        return "agent_query"

    if intent_name in {
        "AGENT_UPDATE",
        "UPDATE_AGENT",
        "AGENT_DELETE",
        "DELETE_AGENT",
        "AGENT_LIFECYCLE",
    }:
        return "not_implemented"

    if intent == ManagerIntent.KNOWLEDGE_BASE_OPERATION:
        return "not_implemented"

    # AGENT_EXECUTION, MULTI_AGENT_ORCHESTRATION, RAG_QUERY,
    # DOCUMENT_OPERATION all plan against the registry. RAG/document
    # requests resolve to the user's registered RAG agent through the
    # normal capability path (Phase 6 will formalize this).
    return "plan_request"


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------

def build_manager_graph():
    """
    Phase 1 Manager control plane.

        START -> understand_request -> route_after_understanding
                   |-> direct_response      -> END
                   |-> plan_request         -> route_after_plan
                   |      |-> execute_wave  -> aggregate -> END
                   |      |-> propose_agent -> aggregate -> END
                   |      |-> controlled_failure -> END
                   |-> not_implemented      -> END
                   |-> controlled_failure   -> END

    Note there is no edge from understanding directly to
    propose_agent, and none from controlled_failure to anything except
    END. Agent proposal is reachable only via plan_request AND a
    recorded capability_gap.
    """

    workflow = StateGraph(
        ManagerState
    )

    workflow.add_node(
        "understand_request",
        understand_request,
    )

    workflow.add_node(
        "direct_response",
        direct_response,
    )

    workflow.add_node(
        "agent_query",
        agent_query,
    )

    workflow.add_node(
        "plan_request",
        plan_request,
    )

    workflow.add_node(
        "propose_agent",
        propose_agent,
    )

    workflow.add_node(
        "explicit_agent_create",
        explicit_agent_create,
    )

    workflow.add_node(
        "execute_wave",
        execute_wave,
    )

    workflow.add_node(
        "aggregate",
        aggregate,
    )

    workflow.add_node(
        "controlled_failure",
        controlled_failure,
    )

    workflow.add_node(
        "not_implemented",
        not_implemented,
    )

    workflow.add_edge(
        START,
        "understand_request",
    )

    workflow.add_conditional_edges(
        "understand_request",
        route_after_understanding,
        {
            "direct_response":
                "direct_response",

            "agent_query":
                "agent_query",

            "plan_request":
                "plan_request",

            "explicit_agent_create":
                "explicit_agent_create",

            "not_implemented":
                "not_implemented",

            "controlled_failure":
                "controlled_failure",
        },
    )

    workflow.add_edge(
        "direct_response",
        END,
    )

    workflow.add_edge(
        "agent_query",
        END,
    )

    workflow.add_edge(
        "explicit_agent_create",
        END,
    )

    workflow.add_edge(
        "not_implemented",
        END,
    )

    workflow.add_edge(
        "controlled_failure",
        END,
    )

    workflow.add_conditional_edges(
        "plan_request",
        route_after_plan,
        {
            "propose_agent":
                "propose_agent",

            "execute_wave":
                "execute_wave",

            "controlled_failure":
                "controlled_failure",
        },
    )

    workflow.add_edge(
        "propose_agent",
        "execute_wave",
    )

    workflow.add_conditional_edges(
        "execute_wave",
        check_plan,
        {
            "continue":
                "execute_wave",

            "aggregate":
                "aggregate",

            "resolve_gap":
                "propose_agent",
        },
    )

    workflow.add_edge(
        "aggregate",
        END,
    )

    return workflow.compile()


# ---------------------------------------------------------------------
# Manager Runtime
# ---------------------------------------------------------------------

class ManagerRuntime:
    """
    Runtime wrapper around the Manager LangGraph.

    user_id is propagated for ownership isolation.

    conversation_history is propagated so persistent chat context
    remains available to the Manager and final response synthesis.
    """

    def __init__(self):
        logger.info(
            "manager.build_loaded version=%s",
            MANAGER_BUILD,
        )
        self.graph = build_manager_graph()

    def run(
        self,
        db: Session,
        execution_id: str,
        user_input: str,
        user_id: Optional[int] = None,
        provider: str = "gemini",
        model: Optional[str] = None,
        conversation_history: Optional[
            List[Dict[str, Any]]
        ] = None,
        conversation_id: Optional[str] = None,
        conversation_knowledge_base_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        from core.config import settings as _settings

        history = _safe_conversation_history(
            conversation_history
        )

        initial_state: ManagerState = {
            "deadline": time.monotonic() + settings.MAX_ORCHESTRATION_EXECUTION_TIME,
            "execution_id":
                execution_id,

            "user_id":
                user_id,

            "user_input":
                user_input,

            "provider":
                provider,

            "model":
                (
                    model
                    or {
                        "gemini":
                            _settings.GEMINI_DEFAULT_MODEL,
                        "groq":
                            _settings.GROQ_DEFAULT_MODEL,
                        "anthropic":
                            _settings.ANTHROPIC_DEFAULT_MODEL,
                        "openai":
                            _settings.OPENAI_DEFAULT_MODEL,
                    }.get(
                        str(provider or "").strip().lower(),
                        _settings.GEMINI_DEFAULT_MODEL,
                    )
                ),

            "conversation_history":
                history,

            "conversation_id":
                conversation_id,

            "conversation_knowledge_base_ids":
                _safe_string_list(conversation_knowledge_base_ids),

            "db":
                db,

            "platform_context":
                "",

            "plan":
                None,

            "step_results":
                {},

            "completed":
                [],

            "skipped":
                [],

            "iteration":
                0,

            "no_agents_found":
                False,

            "unmatched_description":
                None,

            # ---- Phase 1 understanding/lifecycle fields ----
            # Every request starts with NO intent and NO capability
            # gap. Both must be positively established by their own
            # stage; neither is ever assumed.
            "intent":
                None,

            "intent_confidence":
                0.0,

            "intent_reason":
                None,

            "requirements":
                {},

            "planning_status":
                "pending",

            "capability_gap":
                False,

            "missing_capabilities":
                [],

            "unavailable_capabilities":
                [],

            "failure_code":
                None,

            "failure_stage":
                None,

            "failure_retryable":
                False,

            "proposed_agent":
                None,

            "needs_approval":
                False,

            "gap_resolution_strategy":
                None,

            "auto_created_agent_id":
                None,

            "final_response":
                None,

            "status":
                "running",

            "error":
                None,
        }

        try:

            result = self.graph.invoke(
                initial_state
            )

            return result

        except Exception as exc:

            logger.exception(
                "manager.orchestration_crashed "
                "execution_id=%s user_id=%s",
                execution_id,
                user_id,
            )

            return {
                **initial_state,
                "failure_code": ManagerFailureCode.EXECUTION_FAILED.value,
                "failure_stage": ManagerStage.SYSTEM.value,

                "status":
                    "failed",

                "error":
                    str(exc),

                "final_response":
                    None,
            }


manager_runtime = ManagerRuntime()
