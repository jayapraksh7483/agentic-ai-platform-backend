"""
Manager Agent orchestration graph.

Architecture:

    START
      |
      v
    classify_request
      |
      +--> general --> direct_response --> END
      |
      v
    plan_request
      |
      +--> no matching capability/agent
      |        |
      |        v
      |    propose_agent
      |        |
      |        v
      |     aggregate --> END
      |
      v
    execute_wave
      |
      v
    check_plan
      |
      +--> more work --> execute_wave
      |
      v
    aggregate
      |
      v
     END

Phase 5B ownership rules:

- Every orchestration carries user_id.
- Planner discovery receives user_id and must return only that
  user's agents.
- Agent execution receives user_id.
- The lower-level agent runtime is responsible for enforcing the
  ownership check when user_id is supplied.
- Newly created agents are created by manager.service, not here.
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


# =====================================================================
# MANAGER STATE
# =====================================================================

class ManagerState(TypedDict, total=False):
    execution_id: str

    # Phase 5B:
    # Authenticated platform user who owns this orchestration.
    user_id: Optional[int]

    user_input: str
    provider: str
    model: str
    request_type: str

    # Runtime database session.
    db: Any

    # Execution plan.
    plan: Optional[Dict[str, Any]]

    # Results for individual orchestration steps.
    step_results: Dict[str, Dict[str, Any]]

    # Completed and skipped step IDs.
    completed: List[str]
    skipped: List[str]

    # Safety limit.
    iteration: int

    # Whether no registered agent can satisfy the request.
    no_agents_found: bool

    # Description used when proposing a new agent.
    unmatched_description: Optional[str]

    # Proposed agent waiting for user approval.
    proposed_agent: Optional[Dict[str, Any]]

    # Whether explicit approval is required.
    needs_approval: bool

    final_response: Optional[str]
    status: str
    error: Optional[str]


# =====================================================================
# CLASSIFICATION
# =====================================================================

def classify_request(state: ManagerState) -> ManagerState:
    """
    Decide whether the request is a general conversational request
    or requires a specialist agent.
    """

    try:
        request_type = planner.classify_request(
            user_input=state["user_input"],
            provider=state["provider"],
            model=state["model"],
        )

        state["request_type"] = request_type

        logger.info(
            "Request classified as: %s",
            request_type,
        )

    except Exception as exc:
        logger.warning(
            "Request classification failed: %s",
            exc,
        )

        # Safe fallback.
        state["request_type"] = "agent_task"

    return state


# =====================================================================
# DIRECT RESPONSE
# =====================================================================

def direct_response(state: ManagerState) -> ManagerState:
    """
    Answer simple/general conversational requests directly.

    No specialist agent is executed.
    """

    try:
        client = get_llm_client(
            state["provider"]
        )

        response = client.generate(
            system_prompt="""You are the Manager Agent of an agentic AI platform.

Answer simple conversational requests directly.

Be friendly, concise and natural.

If the user asks what you can do, explain that you can understand
requests, discover registered specialist agents, execute them, and
coordinate their results.

Do not claim that an agent was executed when no agent was executed.
Do not invent unavailable capabilities.
""",
            user_input=state["user_input"],
            model_name=state["model"],
        )

        state["final_response"] = response.strip()
        state["status"] = "success"
        state["error"] = None

    except Exception as exc:
        logger.exception(
            "Direct response failed: %s",
            exc,
        )

        state["status"] = "failed"
        state["error"] = str(exc)
        state["final_response"] = None

    return state


# =====================================================================
# PLAN REQUEST
# =====================================================================

def plan_request(state: ManagerState) -> ManagerState:
    """
    Understand the request, discover the authenticated user's agents,
    and build an execution plan.
    """

    logger.info(
        "manager.request_received execution_id=%s user_id=%s",
        state.get("execution_id"),
        state.get("user_id"),
    )

    # -------------------------------------------------------------
    # Capability extraction
    #
    # Phase 5B:
    # user_id is passed so planner discovery can be user-scoped.
    # -------------------------------------------------------------

    extraction = planner.extract_capabilities(
        db=state["db"],
        user_input=state["user_input"],
        provider=state["provider"],
        model=state["model"],
        user_id=state.get("user_id"),
    )

    if not extraction.required_capabilities:
        state["no_agents_found"] = True

        state["unmatched_description"] = (
            extraction.unmatched_description
            or state["user_input"]
        )

        state["plan"] = (
            ExecutionPlan(
                request=state["user_input"]
            ).model_dump()
        )

        return state

    # -------------------------------------------------------------
    # Build plan
    #
    # Phase 5B:
    # user_id ensures only the authenticated user's agents
    # participate in planning.
    # -------------------------------------------------------------

    plan = planner.build_plan(
        db=state["db"],
        user_input=state["user_input"],
        extraction=extraction,
        user_id=state.get("user_id"),
    )

    state["plan"] = plan.model_dump()

    state["step_results"] = {}
    state["completed"] = []
    state["skipped"] = []
    state["iteration"] = 0

    if not plan.steps:

        # Every identified capability had no matching
        # registered agent for this user.
        state["no_agents_found"] = True

        state["unmatched_description"] = (
            extraction.unmatched_description
            or ", ".join(
                plan.unmatched_capabilities
            )
            or state["user_input"]
        )

    else:
        state["no_agents_found"] = False

    return state


# =====================================================================
# PROPOSE NEW AGENT
# =====================================================================

def propose_agent(state: ManagerState) -> ManagerState:
    """
    Draft a reusable new-agent proposal.

    IMPORTANT:
    This function NEVER creates the agent.

    Creation happens only after explicit user approval through
    manager.service.approve_pending_agent().
    """

    logger.info(
        "manager.no_agent_available_proposing_new_agent "
        "execution_id=%s user_id=%s",
        state.get("execution_id"),
        state.get("user_id"),
    )

    try:
        proposal = planner.propose_new_agent(
            db=state["db"],
            user_input=state["user_input"],
            unmatched_description=(
                state.get("unmatched_description")
                or state["user_input"]
            ),
            provider=state["provider"],
            model=state["model"],
        )

        state["proposed_agent"] = (
            proposal.model_dump()
        )

        state["needs_approval"] = True

    except Exception as exc:
        logger.exception(
            "manager.propose_agent_failed "
            "execution_id=%s",
            state.get("execution_id"),
        )

        state["needs_approval"] = False
        state["error"] = str(exc)

    return state


# =====================================================================
# EXECUTE ONE STEP
# =====================================================================

def _run_step_with_retries(
    step: ExecutionStep,
    provider: str,
    model: str,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute one step through the existing agent runtime.

    Phase 5B:
    user_id is forwarded to agent_runtime.execute() so the runtime
    cannot execute another user's agent.
    """

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

                # -------------------------------------------------
                # Phase 5B ownership enforcement
                # -------------------------------------------------
                user_id=user_id,
            )

            if (
                result.get("status") == "success"
                and not result.get("error")
            ):
                return {
                    "step_id": step.step_id,
                    "agent_id": step.agent_id,
                    "capability": step.capability,
                    "status": "success",
                    "result": {
                        "output": result.get(
                            "output"
                        )
                    },
                    "error": None,
                }

            last_error = (
                result.get("error")
                or "Agent execution failed"
            )

            logger.info(
                "manager.step_attempt_failed "
                "step_id=%s attempt=%s/%s error=%s",
                step.step_id,
                attempt,
                max_attempts,
                last_error,
            )

        except Exception as exc:

            last_error = str(exc)

            logger.info(
                "manager.step_attempt_exception "
                "step_id=%s attempt=%s/%s error=%s",
                step.step_id,
                attempt,
                max_attempts,
                last_error,
            )

    return {
        "step_id": step.step_id,
        "agent_id": step.agent_id,
        "capability": step.capability,
        "status": "failed",
        "result": None,
        "error": last_error,
    }


# =====================================================================
# CONDITION EVALUATION
# =====================================================================

def _condition_met(
    state: ManagerState,
    step: ExecutionStep,
) -> bool:

    if not step.condition_on:
        return True

    prior = state["step_results"].get(
        step.condition_on
    )

    if (
        not prior
        or prior.get("status") != "success"
    ):
        # Fail safe.
        return False

    if not step.condition_keyword:
        return True

    haystack = str(
        prior.get("result", "")
    ).lower()

    return (
        step.condition_keyword.lower()
        in haystack
    )


# =====================================================================
# EXECUTE WAVE
# =====================================================================

def execute_wave(
    state: ManagerState,
) -> ManagerState:

    plan = ExecutionPlan(
        **state["plan"]
    )

    completed = set(
        state["completed"]
    )

    skipped = set(
        state["skipped"]
    )

    state["iteration"] = (
        state.get("iteration", 0)
        + 1
    )

    # -------------------------------------------------------------
    # Find steps whose dependencies are satisfied.
    # -------------------------------------------------------------

    ready: List[ExecutionStep] = []

    for step in plan.steps:

        if (
            step.step_id in completed
            or step.step_id in skipped
        ):
            continue

        if not all(
            dep in completed
            or dep in skipped
            for dep in step.depends_on
        ):
            continue

        ready.append(step)

    if not ready:
        return state

    # -------------------------------------------------------------
    # Evaluate conditions.
    # -------------------------------------------------------------

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
                "step_id": step.step_id,
                "agent_id": step.agent_id,
                "capability": step.capability,
                "status": "skipped",
                "result": None,
                "error": (
                    "Condition not met -- "
                    "step skipped."
                ),
            }

            logger.info(
                "manager.step_skipped "
                "step_id=%s "
                "(condition not met)",
                step.step_id,
            )

            continue

        to_run.append(step)

    # -------------------------------------------------------------
    # Execute independent steps concurrently.
    # -------------------------------------------------------------

    if to_run:

        logger.info(
            "manager.wave_started "
            "execution_id=%s user_id=%s steps=%s",
            state.get("execution_id"),
            state.get("user_id"),
            [
                s.step_id
                for s in to_run
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
                        "step_id": step.step_id,
                        "agent_id": step.agent_id,
                        "capability": step.capability,
                        "status": "failed",
                        "result": None,
                        "error": str(exc),
                    }

                state["step_results"][
                    step.step_id
                ] = outcome

                logger.info(
                    "manager.step_completed "
                    "step_id=%s status=%s",
                    step.step_id,
                    outcome["status"],
                )

    # -------------------------------------------------------------
    # Mark executed steps completed.
    # -------------------------------------------------------------

    for step in to_run:

        state["completed"].append(
            step.step_id
        )

    return state


# =====================================================================
# CHECK PLAN
# =====================================================================

def check_plan(
    state: ManagerState,
) -> str:

    plan = ExecutionPlan(
        **state["plan"]
    )

    done = (
        set(state["completed"])
        | set(state["skipped"])
    )

    remaining = [
        s
        for s in plan.steps
        if s.step_id not in done
    ]

    if not remaining:
        return "aggregate"

    # -------------------------------------------------------------
    # Safety limit
    # -------------------------------------------------------------

    if (
        state.get("iteration", 0)
        >= settings.MAX_ORCHESTRATION_STEPS
    ):

        logger.warning(
            "manager.step_limit_reached "
            "execution_id=%s remaining=%s",
            state.get("execution_id"),
            [
                s.step_id
                for s in remaining
            ],
        )

        for step in remaining:

            state["step_results"][
                step.step_id
            ] = {
                "step_id": step.step_id,
                "agent_id": step.agent_id,
                "capability": step.capability,
                "status": "failed",
                "result": None,
                "error": (
                    f"Orchestration step limit "
                    f"({settings.MAX_ORCHESTRATION_STEPS}) "
                    "reached before this step could run."
                ),
            }

            state["completed"].append(
                step.step_id
            )

        return "aggregate"

    return "continue"


# =====================================================================
# AGGREGATE
# =====================================================================

def aggregate(
    state: ManagerState,
) -> ManagerState:

    # -------------------------------------------------------------
    # Pending new-agent approval
    # -------------------------------------------------------------

    if state.get("needs_approval"):

        proposal = (
            state.get("proposed_agent")
            or {}
        )

        state["status"] = (
            "pending_agent_approval"
        )

        state["error"] = None

        capabilities_text = ", ".join(
            proposal.get(
                "capabilities",
                [],
            )
        ) or "this request"

        state["final_response"] = (
            f"I don't have an agent that can "
            f"handle this yet. I'd like to create "
            f"a new agent, "
            f"\"{proposal.get('name', 'New Agent')}\", "
            f"covering: {capabilities_text}. "
            "Should I go ahead and create it?"
        )

        logger.info(
            "manager.orchestration_completed "
            "execution_id=%s "
            "user_id=%s "
            "status=pending_agent_approval",
            state.get("execution_id"),
            state.get("user_id"),
        )

        return state

    # -------------------------------------------------------------
    # No agent found and proposal failed
    # -------------------------------------------------------------

    if state.get("no_agents_found"):

        state["status"] = "failed"

        state["error"] = (
            state.get("error")
            or (
                "No suitable registered capability "
                "or agent was found for this request, "
                "and drafting a new-agent proposal "
                "failed."
            )
        )

        state["final_response"] = (
            "I couldn't find a registered agent "
            "capable of handling this request."
        )

        logger.info(
            "manager.orchestration_completed "
            "execution_id=%s "
            "status=failed "
            "reason=no_agent",
            state.get("execution_id"),
        )

        return state

    # -------------------------------------------------------------
    # Aggregate actual agent results
    # -------------------------------------------------------------

    results = list(
        state["step_results"].values()
    )

    statuses = [
        r["status"]
        for r in results
    ]

    non_skipped_statuses = [
        s
        for s in statuses
        if s != "skipped"
    ]

    if (
        all(
            s == "success"
            for s in non_skipped_statuses
        )
        and any(
            s == "success"
            for s in statuses
        )
    ):
        overall = "success"

    elif any(
        s == "success"
        for s in statuses
    ):
        overall = "partial"

    else:
        overall = "failed"

    state["status"] = overall

    # -------------------------------------------------------------
    # Synthesize final response.
    # -------------------------------------------------------------

    state["final_response"] = (
        planner.synthesize_final_response(
            user_input=state["user_input"],
            step_results=[
                r
                for r in results
                if r["status"] != "skipped"
            ],
            provider=state["provider"],
            model=state["model"],
        )
    )

    # -------------------------------------------------------------
    # Error aggregation.
    # -------------------------------------------------------------

    if overall == "failed":

        failed = [
            r
            for r in results
            if r["status"] == "failed"
        ]

        state["error"] = (
            "; ".join(
                f"{r['step_id']}: {r['error']}"
                for r in failed
            )
            or "Execution failed."
        )

    else:
        state["error"] = None

    logger.info(
        "manager.orchestration_completed "
        "execution_id=%s "
        "user_id=%s "
        "status=%s",
        state.get("execution_id"),
        state.get("user_id"),
        overall,
    )

    return state


# =====================================================================
# ROUTING
# =====================================================================

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


# =====================================================================
# BUILD LANGGRAPH
# =====================================================================

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

    # START -> classify
    workflow.add_edge(
        START,
        "classify_request",
    )

    # classify -> direct OR planning
    workflow.add_conditional_edges(
        "classify_request",
        route_after_classification,
        {
            "direct_response": "direct_response",
            "plan_request": "plan_request",
        },
    )

    # direct -> END
    workflow.add_edge(
        "direct_response",
        END,
    )

    # planning -> proposal OR execution
    workflow.add_conditional_edges(
        "plan_request",
        route_after_plan,
        {
            "propose_agent": "propose_agent",
            "execute_wave": "execute_wave",
        },
    )

    # proposal -> aggregate
    workflow.add_edge(
        "propose_agent",
        "aggregate",
    )

    # execution -> continue OR aggregate
    workflow.add_conditional_edges(
        "execute_wave",
        check_plan,
        {
            "continue": "execute_wave",
            "aggregate": "aggregate",
        },
    )

    # aggregate -> END
    workflow.add_edge(
        "aggregate",
        END,
    )

    return workflow.compile()


# =====================================================================
# MANAGER RUNTIME
# =====================================================================

class ManagerRuntime:
    """
    Analogous to agent_runtime.AgentRuntime, but orchestrates one
    or more registered agents.

    Phase 5B:
    user_id is propagated through the entire orchestration.
    """

    def __init__(self):
        self.graph = build_manager_graph()

    def run(
        self,
        db: Session,
        execution_id: str,
        user_input: str,
        provider: str = "gemini",
        model: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:

        from core.config import (
            settings as _settings,
        )

        initial_state: ManagerState = {

            # -----------------------------------------------------
            # Execution identity
            # -----------------------------------------------------

            "execution_id": execution_id,

            # -----------------------------------------------------
            # Phase 5B ownership
            # -----------------------------------------------------

            "user_id": user_id,

            # -----------------------------------------------------
            # Request
            # -----------------------------------------------------

            "user_input": user_input,

            "provider": provider,

            "model": (
                model
                or _settings.GEMINI_DEFAULT_MODEL
            ),

            # -----------------------------------------------------
            # Database
            # -----------------------------------------------------

            "db": db,

            # -----------------------------------------------------
            # Planning
            # -----------------------------------------------------

            "plan": None,

            "step_results": {},

            "completed": [],

            "skipped": [],

            "iteration": 0,

            "no_agents_found": False,

            "unmatched_description": None,

            # -----------------------------------------------------
            # New-agent proposal
            # -----------------------------------------------------

            "proposed_agent": None,

            "needs_approval": False,

            # -----------------------------------------------------
            # Final state
            # -----------------------------------------------------

            "final_response": None,

            "status": "running",

            "error": None,
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
                "status": "failed",
                "error": str(exc),
                "final_response": None,
            }


# =====================================================================
# SINGLE MANAGER RUNTIME INSTANCE
# =====================================================================

manager_runtime = ManagerRuntime()