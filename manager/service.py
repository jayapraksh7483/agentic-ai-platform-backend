"""
Orchestration service layer.

Responsibilities:
- Create and persist orchestration executions.
- Pass authenticated user_id into ManagerRuntime.
- Preserve persistent conversation history.
- Validate conversation ownership when conversation_id is supplied.
- Link orchestration executions to persistent conversations.
- Persist orchestration steps.
- Support pending-agent approval.
- Prevent one user from reading or approving another user's execution.
- Ensure newly-created approved agents belong to the authenticated user.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from core.config import settings

from models.conversation import Conversation
from models.orchestration import (
    OrchestrationExecution,
    OrchestrationStatus,
    OrchestrationStep,
    StepStatus,
)

from schemas.orchestration import (
    ExecutionStatusResponse,
    OrchestrateResponse,
)

from .manager_agent import manager_runtime


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _generate_execution_id() -> str:
    return str(uuid.uuid4())


def _safe_list(value: Any) -> List[Any]:
    """
    Convert None/non-list values into a safe list.
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


def _safe_conversation_history(
    value: Any,
) -> List[Dict[str, Any]]:
    """
    Normalize conversation history into a list of dictionaries.
    """

    history = _safe_list(value)

    normalized: List[Dict[str, Any]] = []

    for item in history:
        if isinstance(item, dict):
            normalized.append(item)

    return normalized


_STATUS_MAP = {
    "success":
        OrchestrationStatus.SUCCESS,

    "partial":
        OrchestrationStatus.PARTIAL,

    "failed":
        OrchestrationStatus.FAILED,

    "pending_agent_approval":
        OrchestrationStatus.PENDING_AGENT_APPROVAL,
}


_STEP_STATUS_MAP = {
    "success":
        StepStatus.SUCCESS,

    "failed":
        StepStatus.FAILED,

    "skipped":
        StepStatus.SKIPPED,
}


# ---------------------------------------------------------------------
# Conversation validation
# ---------------------------------------------------------------------

def _validate_conversation_ownership(
    db: Session,
    conversation_id: Optional[str],
    user_id: Optional[int],
) -> Optional[Conversation]:
    """
    Validate that conversation_id belongs to the authenticated user.

    If conversation_id is not supplied, return None.

    If user_id is supplied, a conversation belonging to another user
    is treated as not found. This prevents information disclosure.
    """

    if not conversation_id:
        return None

    query = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id
        )
    )

    if user_id is not None:

        query = query.filter(
            Conversation.user_id == user_id
        )

    conversation = query.first()

    if conversation is None:

        raise ValueError(
            f"Conversation '{conversation_id}' not found"
        )

    return conversation


# ---------------------------------------------------------------------
# Step persistence
# ---------------------------------------------------------------------

def _persist_plan_steps(
    db: Session,
    execution_id: str,
    plan: dict,
    step_results: dict,
) -> None:
    """
    Persist one OrchestrationStep row for every planned step.
    """

    steps = _safe_list(
        (plan or {}).get("steps")
    )

    results = (
        step_results
        if isinstance(
            step_results,
            dict,
        )
        else {}
    )

    for step in steps:

        if not isinstance(
            step,
            dict,
        ):
            continue

        step_id = (
            step.get("step_id")
            or step.get("id")
        )

        if not step_id:
            continue

        outcome = (
            results.get(step_id)
            or {}
        )

        status = _STEP_STATUS_MAP.get(
            outcome.get("status"),
            StepStatus.PENDING,
        )

        row = OrchestrationStep(
            execution_id=
                execution_id,

            step_key=
                step_id,

            agent_id=
                step.get("agent_id"),

            capability=
                step.get("capability"),

            task=
                step.get(
                    "task",
                    "",
                ),

            depends_on=(
                step.get(
                    "depends_on"
                )
                or []
            ),

            status=
                status,

            result=
                outcome.get(
                    "result"
                ),

            error=
                outcome.get(
                    "error"
                ),

            started_at=(
                datetime.now(
                    timezone.utc
                )
                if outcome
                else None
            ),

            ended_at=(
                datetime.now(
                    timezone.utc
                )
                if outcome
                else None
            ),
        )

        db.add(row)


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------

def orchestrate(
    db: Session,
    user_input: str,
    user_id: Optional[int] = None,
    conversation_id: Optional[str] = None,
    conversation_history: Optional[
        List[Dict[str, Any]]
    ] = None,
    provider: str = "gemini",
    model: Optional[str] = None,
) -> OrchestrateResponse:
    """
    Start a Manager orchestration for one authenticated user.

    conversation_id is optional.

    conversation_history is optional.

    When conversation_id is supplied:
        - the conversation must belong to the authenticated user
        - the orchestration execution is linked to that conversation

    When conversation_history is supplied:
        - it is passed to ManagerRuntime
        - it is available for conversational context and final
          response synthesis
    """

    # -------------------------------------------------------------
    # Normalize conversation history
    # -------------------------------------------------------------

    history = _safe_conversation_history(
        conversation_history
    )

    # -------------------------------------------------------------
    # Validate conversation BEFORE creating execution
    # -------------------------------------------------------------

    conversation = (
        _validate_conversation_ownership(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )
    )

    execution_id = (
        _generate_execution_id()
    )

    execution = OrchestrationExecution(
        id=execution_id,

        user_id=user_id,

        user_input=user_input,

        status=(
            OrchestrationStatus.RUNNING
        ),
    )

    # -------------------------------------------------------------
    # Link persistent conversation when supplied.
    # -------------------------------------------------------------

    if conversation is not None:

        execution.conversation_id = (
            conversation.id
        )

    if hasattr(
        execution,
        "provider",
    ):

        execution.provider = provider

    if hasattr(
        execution,
        "model",
    ):

        execution.model = model

    db.add(execution)

    db.commit()

    db.refresh(execution)

    start = time.monotonic()

    try:

        result = manager_runtime.run(
            db=db,

            execution_id=
                execution_id,

            user_input=
                user_input,

            user_id=
                user_id,

            provider=
                provider,

            model=
                model,

            conversation_history=
                history,
        )

    except Exception as exc:

        result = {
            "status":
                "failed",

            "error":
                str(exc),

            "final_response":
                None,

            "plan":
                {},

            "step_results":
                {},
        }

    latency_ms = (
        time.monotonic()
        - start
    ) * 1000

    plan = (
        result.get("plan")
        or {}
    )

    step_results = (
        result.get(
            "step_results"
        )
        or {}
    )

    execution.required_capabilities = [
        step.get("capability")
        for step in _safe_list(
            plan.get("steps")
        )
        if (
            isinstance(
                step,
                dict,
            )
            and step.get(
                "capability"
            )
        )
    ]

    overall_status = (
        result.get("status")
        or "failed"
    )

    # -------------------------------------------------------------
    # Pending new-agent approval
    # -------------------------------------------------------------

    if (
        overall_status
        == "pending_agent_approval"
    ):

        if hasattr(
            execution,
            "proposed_agent",
        ):

            execution.proposed_agent = (
                result.get(
                    "proposed_agent"
                )
            )

        execution.status = (
            OrchestrationStatus
            .PENDING_AGENT_APPROVAL
        )

        execution.final_response = (
            result.get(
                "final_response"
            )
        )

        execution.error_message = None

        execution.end_time = (
            datetime.now(
                timezone.utc
            )
        )

        execution.latency_ms = (
            latency_ms
        )

        db.commit()

        db.refresh(
            execution
        )

        return OrchestrateResponse(
            execution_id=
                execution_id,

            status=
                execution.status.value,

            result=
                execution.final_response,

            error=
                None,
        )

    # -------------------------------------------------------------
    # Normal execution
    # -------------------------------------------------------------

    _persist_plan_steps(
        db=db,

        execution_id=
            execution_id,

        plan=
            plan,

        step_results=
            step_results,
    )

    execution.status = (
        _STATUS_MAP.get(
            overall_status,
            OrchestrationStatus.FAILED,
        )
    )

    execution.final_response = (
        result.get(
            "final_response"
        )
    )

    execution.error_message = (
        result.get(
            "error"
        )
    )

    execution.end_time = (
        datetime.now(
            timezone.utc
        )
    )

    execution.latency_ms = (
        latency_ms
    )

    db.commit()

    db.refresh(
        execution
    )

    return OrchestrateResponse(
        execution_id=
            execution_id,

        status=
            execution.status.value,

        result=
            execution.final_response,

        error=
            execution.error_message,
    )


# ---------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------

def approve_pending_agent(
    db: Session,
    execution_id: str,
    approved: bool,
    user_id: Optional[int] = None,
) -> OrchestrateResponse:
    """
    Resolve an execution waiting for new-agent approval.

    The execution must belong to the authenticated user.
    """

    query = (
        db.query(
            OrchestrationExecution
        )
        .filter(
            OrchestrationExecution.id
            == execution_id
        )
    )

    # -------------------------------------------------------------
    # Ownership isolation
    # -------------------------------------------------------------

    if user_id is not None:

        query = query.filter(
            OrchestrationExecution.user_id
            == user_id
        )

    execution = query.first()

    if not execution:

        raise ValueError(
            f"Execution '{execution_id}' not found"
        )

    if (
        execution.status
        != OrchestrationStatus
        .PENDING_AGENT_APPROVAL
    ):

        raise ValueError(
            f"Execution '{execution_id}' is not "
            "awaiting agent approval"
        )

    # -------------------------------------------------------------
    # User declined
    # -------------------------------------------------------------

    if not approved:

        execution.status = (
            OrchestrationStatus.FAILED
        )

        execution.error_message = (
            "New agent creation was declined "
            "by the user."
        )

        execution.end_time = (
            datetime.now(
                timezone.utc
            )
        )

        if hasattr(
            execution,
            "proposed_agent",
        ):

            execution.proposed_agent = None

        db.commit()

        db.refresh(
            execution
        )

        return OrchestrateResponse(
            execution_id=
                execution_id,

            status=
                execution.status.value,

            result=
                None,

            error=
                execution.error_message,
        )

    # -------------------------------------------------------------
    # User approved
    # -------------------------------------------------------------

    proposal = (
        getattr(
            execution,
            "proposed_agent",
            None,
        )
        or {}
    )

    from services.agent_service import (
        create_agent,
    )

    from schemas.agent import (
        AgentCreate,
    )

    capabilities = (
        proposal.get(
            "capabilities"
        )
        or []
    )

    new_agent = create_agent(
        db,

        AgentCreate(
            name=(
                proposal.get(
                    "name"
                )
                or "New Agent"
            ),

            description=(
                proposal.get(
                    "description"
                )
            ),

            system_prompt=(
                proposal.get(
                    "system_prompt"
                )
                or ""
            ),

            capabilities=
                capabilities,

            input_schema=(
                proposal.get(
                    "input_schema"
                )
            ),

            output_schema=(
                proposal.get(
                    "output_schema"
                )
            ),

            model={
                "provider": (
                    proposal.get(
                        "provider"
                    )
                    or "gemini"
                ),

                "model": (
                    proposal.get(
                        "model"
                    )
                    or settings.GEMINI_DEFAULT_MODEL
                ),
            },
        ),
    )

    # -------------------------------------------------------------
    # Enforce authenticated ownership.
    # -------------------------------------------------------------

    if user_id is not None:

        if hasattr(
            new_agent,
            "created_by",
        ):

            new_agent.created_by = (
                user_id
            )

            db.add(
                new_agent
            )

            db.commit()

            db.refresh(
                new_agent
            )

    start = time.monotonic()

    # -------------------------------------------------------------
    # Preserve conversation history on approval flow.
    #
    # The approval endpoint currently does not receive the history
    # directly, so retrieve it from the linked conversation when
    # available.
    # -------------------------------------------------------------

    conversation_history = []

    conversation_id = getattr(
        execution,
        "conversation_id",
        None,
    )

    if conversation_id:

        conversation = (
            _validate_conversation_ownership(
                db=db,
                conversation_id=
                    conversation_id,
                user_id=(
                    user_id
                    if user_id is not None
                    else execution.user_id
                ),
            )
        )

        if conversation:

            conversation_history = [
                {
                    "role":
                        message.role.value
                        if hasattr(
                            message.role,
                            "value",
                        )
                        else str(
                            message.role
                        ),

                    "content":
                        message.content,
                }
                for message
                in conversation.messages
            ]

    try:

        result = manager_runtime.run(
            db=db,

            execution_id=
                execution_id,

            user_input=
                execution.user_input,

            user_id=(
                user_id
                if user_id is not None
                else execution.user_id
            ),

            provider=(
                getattr(
                    execution,
                    "provider",
                    None,
                )
                or "gemini"
            ),

            model=(
                getattr(
                    execution,
                    "model",
                    None,
                )
            ),

            conversation_history=
                conversation_history,
        )

    except Exception as exc:

        result = {
            "status":
                "failed",

            "error":
                str(exc),

            "final_response":
                None,

            "plan":
                {},

            "step_results":
                {},
        }

    latency_ms = (
        time.monotonic()
        - start
    ) * 1000

    plan = (
        result.get("plan")
        or {}
    )

    step_results = (
        result.get(
            "step_results"
        )
        or {}
    )

    execution.required_capabilities = [
        step.get("capability")
        for step in _safe_list(
            plan.get("steps")
        )
        if (
            isinstance(
                step,
                dict,
            )
            and step.get(
                "capability"
            )
        )
    ]

    _persist_plan_steps(
        db=db,

        execution_id=
            execution_id,

        plan=
            plan,

        step_results=
            step_results,
    )

    overall_status = (
        result.get("status")
        or "failed"
    )

    execution.status = (
        _STATUS_MAP.get(
            overall_status,
            OrchestrationStatus.FAILED,
        )
    )

    execution.final_response = (
        result.get(
            "final_response"
        )
    )

    execution.error_message = (
        result.get(
            "error"
        )
    )

    if hasattr(
        execution,
        "proposed_agent",
    ):

        execution.proposed_agent = None

    execution.end_time = (
        datetime.now(
            timezone.utc
        )
    )

    execution.latency_ms = (
        execution.latency_ms
        or 0
    ) + latency_ms

    db.commit()

    db.refresh(
        execution
    )

    return OrchestrateResponse(
        execution_id=
            execution_id,

        status=
            execution.status.value,

        result=
            execution.final_response,

        error=
            execution.error_message,
    )


# ---------------------------------------------------------------------
# Get execution
# ---------------------------------------------------------------------

def get_execution(
    db: Session,
    execution_id: str,
    user_id: Optional[int] = None,
) -> Optional[
    ExecutionStatusResponse
]:
    """
    Return an orchestration execution.

    If user_id is supplied, only that user's execution can be returned.
    """

    query = (
        db.query(
            OrchestrationExecution
        )
        .filter(
            OrchestrationExecution.id
            == execution_id
        )
    )

    if user_id is not None:

        query = query.filter(
            OrchestrationExecution.user_id
            == user_id
        )

    execution = query.first()

    if not execution:
        return None

    completed_steps = [
        step.step_key
        for step in execution.steps
        if step.status
        == StepStatus.SUCCESS
    ]

    current_step = next(
        (
            step.step_key
            for step in execution.steps
            if step.status
            == StepStatus.PENDING
        ),
        None,
    )

    return ExecutionStatusResponse(
        execution_id=
            execution.id,

        status=
            execution.status,

        user_input=
            execution.user_input,

        required_capabilities=(
            execution.required_capabilities
        ),

        current_step=
            current_step,

        completed_steps=
            completed_steps,

        steps=[
            {
                "step_key":
                    step.step_key,

                "agent_id":
                    step.agent_id,

                "capability":
                    step.capability,

                "task":
                    step.task,

                "depends_on":
                    step.depends_on,

                "status":
                    step.status,

                "result":
                    step.result,

                "error":
                    step.error,
            }
            for step
            in execution.steps
        ],

        final_response=(
            execution.final_response
        ),

        error=(
            execution.error_message
        ),

        start_time=(
            execution.start_time
        ),

        end_time=(
            execution.end_time
        ),

        latency_ms=(
            execution.latency_ms
        ),
    )