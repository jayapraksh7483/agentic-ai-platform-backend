"""
Manager Agent orchestration graph.

Responsibilities:
- Understand the user's request.
- Discover only agents belonging to the authenticated user.
- Build a dynamic execution plan.
- Execute registered agents through the existing Agent Runtime.
- Support parallel/sequential/conditional execution.
- Propose a reusable new agent when no suitable agent exists.
- Never create an agent without explicit approval.
- Preserve authenticated user_id throughout the orchestration path.
- Preserve conversation history throughout chat/orchestration.
"""

import concurrent.futures
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from core.config import settings
from agent_runtime.runtime import agent_runtime
from services.llm_service import get_llm_client

from . import planner
from .schemas import ExecutionPlan, ExecutionStep


logger = logging.getLogger("manager")


class ManagerState(TypedDict, total=False):
    execution_id: str
    user_id: Optional[int]

    user_input: str
    provider: str
    model: str
    request_type: str

    # Persistent conversation context.
    conversation_history: List[Dict[str, Any]]

    db: Any

    plan: Optional[Dict[str, Any]]

    step_results: Dict[str, Dict[str, Any]]
    completed: List[str]
    skipped: List[str]

    iteration: int
    no_agents_found: bool

    unmatched_description: Optional[str]

    proposed_agent: Optional[Dict[str, Any]]
    needs_approval: bool

    final_response: Optional[str]
    status: str
    error: Optional[str]


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

    return normalized


# ---------------------------------------------------------------------
# Request classification
# ---------------------------------------------------------------------

def classify_request(
    state: ManagerState,
) -> ManagerState:
    """
    Classify the request as:
        general
    or:
        agent_task
    """

    try:

        request_type = planner.classify_request(
            user_input=state["user_input"],
            provider=state["provider"],
            model=state["model"],
        )

        request_type = (
            str(request_type).strip().lower()
            if request_type
            else "agent_task"
        )

        state["request_type"] = request_type

        logger.info(
            "manager.request_classified "
            "execution_id=%s type=%s user_id=%s",
            state.get("execution_id"),
            request_type,
            state.get("user_id"),
        )

    except Exception as exc:

        logger.warning(
            "manager.request_classification_failed "
            "execution_id=%s error=%s",
            state.get("execution_id"),
            exc,
        )

        state["request_type"] = "agent_task"

    return state


# ---------------------------------------------------------------------
# General conversational response
# ---------------------------------------------------------------------

def direct_response(
    state: ManagerState,
) -> ManagerState:
    """
    Answer simple/general requests directly while preserving
    conversation history.
    """

    try:

        client = get_llm_client(
            state["provider"]
        )

        history = _safe_conversation_history(
            state.get("conversation_history")
        )

        response = client.generate(
            system_prompt="""You are the Manager Agent of an agentic AI platform.

Answer simple conversational requests directly.

Be friendly, concise and natural.

If the user asks what you can do, explain that you can:
- understand requests,
- discover registered specialist agents,
- execute suitable agents,
- coordinate multiple agents,
- and return the final result.

Use the conversation history when it helps maintain context.

Do not claim that an agent was executed when no agent was executed.

Do not invent unavailable capabilities.
""",
            user_input=state["user_input"],
            model_name=state["model"],
            messages=history,
        )

        state["final_response"] = (
            str(response).strip()
            if response is not None
            else ""
        )

        state["status"] = "success"
        state["error"] = None

    except Exception as exc:

        logger.exception(
            "manager.direct_response_failed "
            "execution_id=%s",
            state.get("execution_id"),
        )

        state["status"] = "failed"
        state["error"] = str(exc)
        state["final_response"] = None

    return state


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

    try:

        extraction = planner.extract_capabilities(
            db=state["db"],
            user_input=state["user_input"],
            provider=state["provider"],
            model=state["model"],
            user_id=state.get("user_id"),
        )

    except TypeError:

        extraction = planner.extract_capabilities(
            db=state["db"],
            user_input=state["user_input"],
            provider=state["provider"],
            model=state["model"],
        )

    try:

        required_capabilities = _safe_string_list(
            getattr(
                extraction,
                "required_capabilities",
                None,
            )
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

        if not required_capabilities:

            state["no_agents_found"] = True

            state["unmatched_description"] = (
                unmatched_description
                or state["user_input"]
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

        try:

            plan = planner.build_plan(
                db=state["db"],
                user_input=state["user_input"],
                extraction=extraction,
                user_id=state.get("user_id"),
            )

        except TypeError:

            plan = planner.build_plan(
                state["db"],
                state["user_input"],
                extraction,
            )

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

        if not plan_steps:

            state["no_agents_found"] = True

            state["unmatched_description"] = (
                unmatched_description
                or ", ".join(
                    unmatched_capabilities
                )
                or state["user_input"]
            )

        elif unmatched_capabilities:

            state["no_agents_found"] = True

            state["unmatched_description"] = (
                unmatched_description
                or ", ".join(
                    unmatched_capabilities
                )
                or state["user_input"]
            )

        else:

            state["no_agents_found"] = False
            state["unmatched_description"] = None

        return state

    except Exception as exc:

        logger.exception(
            "manager.plan_request_failed "
            "execution_id=%s user_id=%s",
            state.get("execution_id"),
            state.get("user_id"),
        )

        state["no_agents_found"] = True

        state["unmatched_description"] = (
            state["user_input"]
        )

        state["plan"] = ExecutionPlan(
            request=state["user_input"],
            steps=[],
        ).model_dump()

        state["step_results"] = {}
        state["completed"] = []
        state["skipped"] = []
        state["iteration"] = 0
        state["error"] = str(exc)

        return state


# ---------------------------------------------------------------------
# New agent proposal
# ---------------------------------------------------------------------

def propose_agent(
    state: ManagerState,
) -> ManagerState:
    """
    Draft a reusable new-agent proposal.

    IMPORTANT:
    This function does NOT create an agent.
    """

    logger.info(
        "manager.proposal_started "
        "execution_id=%s user_id=%s",
        state.get("execution_id"),
        state.get("user_id"),
    )

    try:

        try:

            proposal = planner.propose_new_agent(
                db=state["db"],
                user_input=state["user_input"],
                unmatched_description=(
                    state.get(
                        "unmatched_description"
                    )
                    or state["user_input"]
                ),
                provider=state["provider"],
                model=state["model"],
                user_id=state.get("user_id"),
            )

        except TypeError:

            proposal = planner.propose_new_agent(
                db=state["db"],
                user_input=state["user_input"],
                unmatched_description=(
                    state.get(
                        "unmatched_description"
                    )
                    or state["user_input"]
                ),
                provider=state["provider"],
                model=state["model"],
            )

        state["proposed_agent"] = (
            proposal.model_dump()
            if hasattr(
                proposal,
                "model_dump",
            )
            else proposal
        )

        state["needs_approval"] = True
        state["error"] = None

    except Exception as exc:

        logger.exception(
            "manager.propose_agent_failed "
            "execution_id=%s",
            state.get("execution_id"),
        )

        state["needs_approval"] = False
        state["error"] = str(exc)
        state["proposed_agent"] = None

    return state


# ---------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------

def _run_step_with_retries(
    step: ExecutionStep,
    provider: str,
    model: str,
    user_id: Optional[int],
) -> Dict[str, Any]:

    last_error: Optional[str] = None

    max_attempts = (
        1
        + max(
            0,
            settings.MAX_ORCHESTRATION_RETRIES,
        )
    )

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        try:

            result = agent_runtime.execute(
                agent_id=step.agent_id,
                user_input=step.task,
                provider=(
                    step.provider
                    or provider
                ),
                model=(
                    step.model
                    or model
                ),
                system_prompt=step.system_prompt,
                user_id=user_id,
            )

            if (
                result.get("status")
                == "success"
                and not result.get("error")
            ):

                return {
                    "step_id":
                        step.step_id,

                    "agent_id":
                        step.agent_id,

                    "capability":
                        step.capability,

                    "status":
                        "success",

                    "result": {
                        "output":
                            result.get("output")
                    },

                    "error":
                        None,
                }

            last_error = (
                result.get("error")
                or "Agent execution failed"
            )

        except Exception as exc:

            last_error = str(exc)

        logger.info(
            "manager.step_attempt_failed "
            "step_id=%s attempt=%s/%s error=%s",
            step.step_id,
            attempt,
            max_attempts,
            last_error,
        )

    return {
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
                last_error
                or "Agent execution failed"
            ),
    }


# ---------------------------------------------------------------------
# Conditional execution
# ---------------------------------------------------------------------

def _condition_met(
    state: ManagerState,
    step: ExecutionStep,
) -> bool:

    condition_on = getattr(
        step,
        "condition_on",
        None,
    )

    if not condition_on:
        return True

    prior = (
        state.get(
            "step_results",
            {},
        ).get(condition_on)
    )

    if not prior:
        return False

    if prior.get("status") != "success":
        return False

    condition_keyword = getattr(
        step,
        "condition_keyword",
        None,
    )

    if not condition_keyword:
        return True

    haystack = str(
        prior.get("result", "")
    ).lower()

    return (
        str(condition_keyword).lower()
        in haystack
    )


# ---------------------------------------------------------------------
# Execution wave
# ---------------------------------------------------------------------

def execute_wave(
    state: ManagerState,
) -> ManagerState:

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

    completed = set(
        _safe_string_list(
            state.get("completed")
        )
    )

    skipped = set(
        _safe_string_list(
            state.get("skipped")
        )
    )

    state["iteration"] = (
        state.get(
            "iteration",
            0,
        )
        + 1
    )

    ready: List[ExecutionStep] = []

    for step in plan_steps:

        if step.step_id in completed:
            continue

        if step.step_id in skipped:
            continue

        dependencies = _safe_string_list(
            getattr(
                step,
                "depends_on",
                None,
            )
        )

        if not all(
            dep in completed
            or dep in skipped
            for dep in dependencies
        ):
            continue

        ready.append(step)

    if not ready:
        return state

    to_run: List[ExecutionStep] = []

    for step in ready:

        if not _condition_met(
            state,
            step,
        ):

            state["skipped"].append(
                step.step_id
            )

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
                    "skipped",

                "result":
                    None,

                "error":
                    (
                        "Condition not met -- "
                        "step skipped."
                    ),
            }

            continue

        to_run.append(step)

    if not to_run:
        return state

    logger.info(
        "manager.wave_started "
        "execution_id=%s user_id=%s steps=%s",
        state.get("execution_id"),
        state.get("user_id"),
        [
            step.step_id
            for step in to_run
        ],
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(
            1,
            len(to_run),
        )
    ) as pool:

        futures = {
            pool.submit(
                _run_step_with_retries,
                step,
                state["provider"],
                state["model"],
                state.get("user_id"),
            ): step
            for step in to_run
        }

        for future in (
            concurrent.futures.as_completed(
                futures
            )
        ):

            step = futures[future]

            try:

                outcome = future.result()

            except Exception as exc:

                outcome = {
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
                        str(exc),
                }

            state["step_results"][
                step.step_id
            ] = outcome

            logger.info(
                "manager.step_completed "
                "execution_id=%s step_id=%s "
                "status=%s",
                state.get("execution_id"),
                step.step_id,
                outcome.get("status"),
            )

    for step in to_run:

        if (
            step.step_id
            not in state["completed"]
        ):

            state["completed"].append(
                step.step_id
            )

    return state


# ---------------------------------------------------------------------
# Plan routing
# ---------------------------------------------------------------------

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

    if state.get("needs_approval"):

        proposal = (
            state.get(
                "proposed_agent"
            )
            or {}
        )

        capabilities = _safe_string_list(
            proposal.get(
                "capabilities"
            )
        )

        capabilities_text = (
            ", ".join(capabilities)
            or "this request"
        )

        state["status"] = (
            "pending_agent_approval"
        )

        state["error"] = None

        state["final_response"] = (
            "I don't have an agent that can "
            "handle this yet. I'd like to create "
            f"a new agent, "
            f"\"{proposal.get('name', 'New Agent')}\", "
            f"covering: {capabilities_text}. "
            "Should I go ahead and create it?"
        )

        return state

    if state.get("no_agents_found"):

        state["status"] = "failed"

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

    non_skipped_results = [
        result
        for result in results
        if result.get("status")
        != "skipped"
    ]

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

    if state.get("no_agents_found"):
        return "propose_agent"

    return "execute_wave"


def route_after_classification(
    state: ManagerState,
) -> str:

    if (
        state.get("request_type")
        == "general"
    ):
        return "direct_response"

    return "plan_request"


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------

def build_manager_graph():

    workflow = StateGraph(
        ManagerState
    )

    workflow.add_node(
        "classify_request",
        classify_request,
    )

    workflow.add_node(
        "direct_response",
        direct_response,
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
        "execute_wave",
        execute_wave,
    )

    workflow.add_node(
        "aggregate",
        aggregate,
    )

    workflow.add_edge(
        START,
        "classify_request",
    )

    workflow.add_conditional_edges(
        "classify_request",
        route_after_classification,
        {
            "direct_response":
                "direct_response",

            "plan_request":
                "plan_request",
        },
    )

    workflow.add_edge(
        "direct_response",
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
        },
    )

    workflow.add_edge(
        "propose_agent",
        "aggregate",
    )

    workflow.add_conditional_edges(
        "execute_wave",
        check_plan,
        {
            "continue":
                "execute_wave",

            "aggregate":
                "aggregate",
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
    ) -> Dict[str, Any]:

        from core.config import settings as _settings

        history = _safe_conversation_history(
            conversation_history
        )

        initial_state: ManagerState = {
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
                    or _settings.GEMINI_DEFAULT_MODEL
                ),

            "conversation_history":
                history,

            "db":
                db,

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

            "proposed_agent":
                None,

            "needs_approval":
                False,

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

                "status":
                    "failed",

                "error":
                    str(exc),

                "final_response":
                    None,
            }


manager_runtime = ManagerRuntime()