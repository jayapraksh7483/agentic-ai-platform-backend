import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from core.config import settings
from models.orchestration import (
    OrchestrationExecution,
    OrchestrationStatus,
    OrchestrationStep,
    StepStatus,
)
from schemas.orchestration import ExecutionStatusResponse, OrchestrateResponse

from .manager_agent import manager_runtime


def _generate_execution_id() -> str:
    return str(uuid.uuid4())


_STATUS_MAP = {
    "success": OrchestrationStatus.SUCCESS,
    "partial": OrchestrationStatus.PARTIAL,
    "failed": OrchestrationStatus.FAILED,
    "pending_agent_approval": getattr(
        OrchestrationStatus,
        "PENDING_AGENT_APPROVAL",
        OrchestrationStatus.FAILED,
    ),
}

_STEP_STATUS_MAP = {
    "success": StepStatus.SUCCESS,
    "failed": StepStatus.FAILED,
    "skipped": StepStatus.SKIPPED,
}


def _persist_plan_steps(
    db: Session,
    execution_id: str,
    plan: dict,
    step_results: dict,
) -> None:
    """Persist one OrchestrationStep row per planned step."""

    for step in plan.get("steps", []):
        outcome = step_results.get(step["step_id"])

        row = OrchestrationStep(
            execution_id=execution_id,
            step_key=step["step_id"],
            agent_id=step.get("agent_id"),
            capability=step.get("capability"),
            task=step.get("task", ""),
            depends_on=step.get("depends_on") or [],
            status=_STEP_STATUS_MAP.get(
                (outcome or {}).get("status"),
                StepStatus.PENDING,
            ),
            result=(outcome or {}).get("result"),
            error=(outcome or {}).get("error"),
            started_at=datetime.now(timezone.utc) if outcome else None,
            ended_at=datetime.now(timezone.utc) if outcome else None,
        )

        db.add(row)


# ============================================================
# ORCHESTRATE
# ============================================================

def orchestrate(
    db: Session,
    user_input: str,
    user_id: Optional[int] = None,
    provider: str = "gemini",
    model: Optional[str] = None,
) -> OrchestrateResponse:

    execution_id = _generate_execution_id()

    execution = OrchestrationExecution(
        id=execution_id,
        user_id=user_id,
        user_input=user_input,
        status=OrchestrationStatus.RUNNING,
    )

    if hasattr(execution, "provider"):
        execution.provider = provider

    if hasattr(execution, "model"):
        execution.model = model

    db.add(execution)
    db.commit()
    db.refresh(execution)

    start = time.monotonic()

    result = manager_runtime.run(
        db=db,
        execution_id=execution_id,
        user_input=user_input,
        provider=provider,
        model=model,
        user_id=user_id,
    )

    latency_ms = (time.monotonic() - start) * 1000

    plan = result.get("plan") or {}
    step_results = result.get("step_results") or {}

    execution.required_capabilities = [
        s.get("capability")
        for s in plan.get("steps", [])
        if s.get("capability")
    ]

    overall_status = result.get("status", "failed")

    # --------------------------------------------------------
    # PENDING AGENT APPROVAL
    # --------------------------------------------------------

    if overall_status == "pending_agent_approval":

        if hasattr(execution, "proposed_agent"):
            execution.proposed_agent = result.get("proposed_agent")

        execution.status = _STATUS_MAP["pending_agent_approval"]
        execution.final_response = result.get("final_response")
        execution.error_message = None
        execution.end_time = datetime.now(timezone.utc)
        execution.latency_ms = latency_ms

        db.commit()
        db.refresh(execution)

        return OrchestrateResponse(
            execution_id=execution_id,
            status=execution.status.value,
            result=execution.final_response,
            error=execution.error_message,
        )

    # --------------------------------------------------------
    # NORMAL EXECUTION
    # --------------------------------------------------------

    _persist_plan_steps(
        db,
        execution_id,
        plan,
        step_results,
    )

    execution.status = _STATUS_MAP.get(
        overall_status,
        OrchestrationStatus.FAILED,
    )

    execution.final_response = result.get("final_response")
    execution.error_message = result.get("error")
    execution.end_time = datetime.now(timezone.utc)
    execution.latency_ms = latency_ms

    db.commit()
    db.refresh(execution)

    return OrchestrateResponse(
        execution_id=execution_id,
        status=execution.status.value,
        result=execution.final_response,
        error=execution.error_message,
    )


# ============================================================
# APPROVE / REJECT PROPOSED AGENT
# ============================================================

def approve_pending_agent(
    db: Session,
    execution_id: str,
    approved: bool,
    user_id: Optional[int] = None,
) -> OrchestrateResponse:
    """
    Resolve an execution waiting for agent approval.

    The execution MUST belong to the authenticated user.
    """

    query = (
        db.query(OrchestrationExecution)
        .filter(
            OrchestrationExecution.id == execution_id,
        )
    )

    # User-scoped access
    if user_id is not None:
        query = query.filter(
            OrchestrationExecution.user_id == user_id
        )

    execution = query.first()

    if not execution:
        raise ValueError(
            f"Execution '{execution_id}' not found"
        )

    pending_status = _STATUS_MAP["pending_agent_approval"]

    if execution.status != pending_status:
        raise ValueError(
            f"Execution '{execution_id}' is not awaiting "
            f"agent approval (status={execution.status})"
        )

    # --------------------------------------------------------
    # USER DECLINED
    # --------------------------------------------------------

    if not approved:

        execution.status = OrchestrationStatus.FAILED
        execution.error_message = (
            "New agent creation was declined by the user."
        )
        execution.end_time = datetime.now(timezone.utc)

        db.commit()
        db.refresh(execution)

        return OrchestrateResponse(
            execution_id=execution_id,
            status=execution.status.value,
            result=None,
            error=execution.error_message,
        )

    # --------------------------------------------------------
    # CREATE PROPOSED AGENT
    # --------------------------------------------------------

    proposal = getattr(
        execution,
        "proposed_agent",
        None,
    ) or {}

    from services.agent_service import create_agent
    from schemas.agent import AgentCreate

    new_agent = create_agent(
        db,
        AgentCreate(
            name=proposal.get(
                "name",
                "New Agent",
            ),
            description=proposal.get(
                "description"
            ),
            system_prompt=proposal.get(
                "system_prompt",
                "",
            ),
            capabilities=proposal.get(
                "capabilities",
                [],
            ),
            input_schema=proposal.get(
                "input_schema"
            ),
            output_schema=proposal.get(
                "output_schema"
            ),
            model={
                "provider": proposal.get(
                    "provider"
                ) or "gemini",
                "model": proposal.get(
                    "model"
                ) or settings.GEMINI_DEFAULT_MODEL,
            },
        ),

        # IMPORTANT:
        # The newly created agent belongs to the
        # authenticated user who approved it.
        created_by=user_id,
    )

    start = time.monotonic()

    # --------------------------------------------------------
    # RE-RUN ORIGINAL REQUEST
    # --------------------------------------------------------

    result = manager_runtime.run(
        db=db,
        execution_id=execution_id,
        user_input=execution.user_input,
        provider=getattr(
            execution,
            "provider",
            None,
        ) or "gemini",
        model=getattr(
            execution,
            "model",
            None,
        ),
        user_id=user_id,
    )

    latency_ms = (
        time.monotonic() - start
    ) * 1000

    plan = result.get("plan") or {}
    step_results = result.get("step_results") or {}

    execution.required_capabilities = [
        s.get("capability")
        for s in plan.get("steps", [])
        if s.get("capability")
    ]

    _persist_plan_steps(
        db,
        execution_id,
        plan,
        step_results,
    )

    overall_status = result.get(
        "status",
        "failed",
    )

    execution.status = _STATUS_MAP.get(
        overall_status,
        OrchestrationStatus.FAILED,
    )

    execution.final_response = result.get(
        "final_response"
    )

    execution.error_message = result.get(
        "error"
    )

    if hasattr(execution, "proposed_agent"):
        execution.proposed_agent = None

    execution.end_time = datetime.now(
        timezone.utc
    )

    execution.latency_ms = (
        execution.latency_ms or 0
    ) + latency_ms

    db.commit()
    db.refresh(execution)

    return OrchestrateResponse(
        execution_id=execution_id,
        status=execution.status.value,
        result=execution.final_response,
        error=execution.error_message,
    )


# ============================================================
# GET EXECUTION
# ============================================================

def get_execution(
    db: Session,
    execution_id: str,
    user_id: Optional[int] = None,
) -> Optional[ExecutionStatusResponse]:

    query = (
        db.query(OrchestrationExecution)
        .filter(
            OrchestrationExecution.id == execution_id
        )
    )

    # IMPORTANT:
    # Prevent User A from reading User B's execution.
    if user_id is not None:
        query = query.filter(
            OrchestrationExecution.user_id == user_id
        )

    execution = query.first()

    if not execution:
        return None

    completed_steps = [
        s.step_key
        for s in execution.steps
        if s.status == StepStatus.SUCCESS
    ]

    current_step = next(
        (
            s.step_key
            for s in execution.steps
            if s.status == StepStatus.PENDING
        ),
        None,
    )

    return ExecutionStatusResponse(
        execution_id=execution.id,
        status=execution.status,
        user_input=execution.user_input,
        required_capabilities=execution.required_capabilities,
        current_step=current_step,
        completed_steps=completed_steps,
        steps=[
            {
                "step_key": s.step_key,
                "agent_id": s.agent_id,
                "capability": s.capability,
                "task": s.task,
                "depends_on": s.depends_on,
                "status": s.status,
                "result": s.result,
                "error": s.error,
            }
            for s in execution.steps
        ],
        final_response=execution.final_response,
        error=execution.error_message,
        start_time=execution.start_time,
        end_time=execution.end_time,
        latency_ms=execution.latency_ms,
    )