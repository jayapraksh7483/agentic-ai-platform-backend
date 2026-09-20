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
- Return proposed_agent details to the frontend.
- Persist approval/rejection reasons.
- Prevent one user from reading or approving another user's execution.
- Ensure newly-created approved agents belong to the authenticated user.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from core.config import settings

from models.conversation import Conversation
from models.attachment import ConversationAttachment
from models.orchestration import (
    OrchestrationExecution,
    OrchestrationStatus,
    OrchestrationStep,
    StepStatus,
)

from schemas.orchestration import (
    ExecutionStatusResponse,
    OrchestrateResponse,
    ProposedAgentOut,
    ProposalEditRequest,
)

from .manager_agent import manager_runtime

logger = logging.getLogger("manager")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _result_sources(results):
    seen, sources = set(), []
    for result in results.values():
        for source in (result.get("result") or {}).get("sources", []):
            key = (source.get("document_id"), source.get("chunk_index"))
            if key not in seen:
                seen.add(key)
                sources.append(source)
    return sources


def _generate_execution_id() -> str:
    return str(uuid.uuid4())


def _get_conversation_knowledge_base_ids(
    db: Session,
    conversation_id: Optional[str],
    user_id: Optional[int],
) -> List[str]:
    """
    Resolve the set of Knowledge Base IDs available to a conversation
    for this user, from every source the conversation can draw on:

        A) Documents uploaded directly to the conversation
           (ConversationAttachment.knowledge_base_id), only once their
           processing has finished (status == "ready").

        B) Existing Knowledge Bases the user explicitly attached to
           the conversation (ConversationKnowledgeBase) -- see
           services/conversation_knowledge_service.py. These are
           references only; nothing is duplicated.

    The Manager does not need to know or care which of the two
    sources a given ID came from -- both are merged into one flat,
    deduplicated list here, which is the ONLY thing manager_agent.py
    ever looks at.

    Always scoped to `user_id` -- a conversation's knowledge sources
    must never include another user's data, even if conversation_id
    were somehow guessed/shared.

    This used to be inlined twice in orchestrate() (redundantly -- the
    first computation was thrown away and immediately recomputed) and
    was MISSING entirely in approve_pending_agent(), which referenced
    the variable without ever defining it -- a NameError on every
    single approve-and-rerun call outside of tests, i.e. Path 1 of the
    "manager agent creation" flow.
    """

    if not conversation_id or user_id is None:
        return []

    uploaded_rows = (
        db.query(ConversationAttachment.knowledge_base_id)
        .filter(
            ConversationAttachment.conversation_id == conversation_id,
            ConversationAttachment.user_id == user_id,
            ConversationAttachment.status == "ready",
            ConversationAttachment.knowledge_base_id.isnot(None),
        )
        .distinct()
        .all()
    )

    from services.conversation_knowledge_service import (
        get_conversation_attached_knowledge_base_ids,
    )

    attached_ids = get_conversation_attached_knowledge_base_ids(
        db=db,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    uploaded_ids = [row[0] for row in uploaded_rows if row[0]]

    # Deduplicate while preserving order (a KB could in principle show
    # up from both sources, e.g. if a user attached the very KB that
    # was created for one of their conversation uploads).
    seen = set()
    merged: List[str] = []

    for kb_id in (*uploaded_ids, *attached_ids):
        if kb_id not in seen:
            seen.add(kb_id)
            merged.append(kb_id)

    return merged


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


def _normalize_proposed_agent(
    value: Any,
) -> Optional[Dict[str, Any]]:
    """Normalize the durable lifecycle proposal for API output."""
    if not isinstance(value, dict):
        return None

    keys = (
        "name", "description", "system_prompt", "capabilities",
        "input_schema", "output_schema", "provider", "model", "tools",
        "is_rag", "knowledge_base_id", "visibility", "timeout_seconds",
        "max_retries", "requires_approval", "reason", "operation",
        "target_agent_id", "proposal_state", "allow_duplicate",
    )
    normalized = {key: value.get(key) for key in keys}
    normalized["capabilities"] = (
        value.get("capabilities") if isinstance(value.get("capabilities"), list) else []
    )
    normalized["tools"] = (
        value.get("tools") if value.get("tools") is None or isinstance(value.get("tools"), list) else []
    )
    normalized["is_rag"] = bool(value.get("is_rag"))
    normalized["visibility"] = value.get("visibility") or "private"
    normalized["timeout_seconds"] = int(value.get("timeout_seconds") or settings.DEFAULT_AGENT_TIMEOUT_SECONDS)
    normalized["max_retries"] = int(
        settings.DEFAULT_AGENT_MAX_RETRIES
        if value.get("max_retries") is None
        else value.get("max_retries")
    )
    normalized["requires_approval"] = bool(value.get("requires_approval", False))
    normalized["operation"] = value.get("operation") or "create"
    normalized["proposal_state"] = value.get("proposal_state") or "proposed"
    normalized["allow_duplicate"] = bool(value.get("allow_duplicate", False))
    normalized["resume_task"] = bool(value.get("resume_task", True))
    return normalized


def _knowledge_options_for_proposal(
    db: Session,
    user_id: Optional[int],
    proposal: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if user_id is None or not proposal or not proposal.get("is_rag"):
        return []
    from services import knowledge_service
    return knowledge_service.list_knowledge_base_options(db, user_id=user_id)


def _proposal_response(
    db: Session,
    execution: OrchestrationExecution,
    *,
    result: Optional[str] = None,
    error: Optional[str] = None,
    created_agent_id: Optional[str] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> OrchestrateResponse:
    proposal = _normalize_proposed_agent(execution.proposed_agent)
    needs_kb = bool(
        proposal
        and proposal.get("proposal_state") == "knowledge_source_required"
    )
    return OrchestrateResponse(
        execution_id=execution.id,
        decision=execution.decision,
        failure_code=execution.failure_code,
        knowledge_source_choices=["attach_existing", "create_new"] if needs_kb else [],
        status=execution.status.value if hasattr(execution.status, "value") else str(execution.status),
        result=result if result is not None else execution.final_response,
        error=error if error is not None else execution.error_message,
        proposed_agent=(proposal if execution.status == OrchestrationStatus.PENDING_AGENT_APPROVAL else None),
        knowledge_source_required=needs_kb,
        available_knowledge_bases=(
            _knowledge_options_for_proposal(db, execution.user_id, proposal)
            if needs_kb
            else []
        ),
        created_agent_id=created_agent_id,
        sources=sources or [],
    )


_STATUS_MAP = {
    "running":
        OrchestrationStatus.RUNNING,

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
            execution_id=execution_id,

            step_key=step_id,
            tool_name=step.get("tool_name"),

            agent_id=step.get("agent_id"),

            capability=step.get("capability"),

            task=step.get(
                "task",
                "",
            ),

            depends_on=(
                step.get(
                    "depends_on"
                )
                or []
            ),

            status=status,

            result=outcome.get(
                "result"
            ),

            error=outcome.get(
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

        existing = db.query(OrchestrationStep).filter(
            OrchestrationStep.execution_id == execution_id,
            OrchestrationStep.step_key == step_id,
        ).first()
        if existing:
            for field in ("agent_id", "tool_name", "capability", "task", "depends_on", "status", "result", "error", "started_at", "ended_at"):
                setattr(existing, field, getattr(row, field))
        else:
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

    # -------------------------------------------------------------
    # Conversation history: single controlled boundary.
    #
    # Two entry points reach this function and they used to behave
    # differently: POST /api/conversations/{id}/chat loads the prior
    # messages and passes them in, while POST /api/orchestrate passed
    # only conversation_id -- so an orchestrate call against an
    # existing conversation silently ran with NO context, and
    # follow-up phrasing ("explain that with an example") could not
    # work there.
    #
    # Loading here (rather than in the API layer) fixes it for every
    # caller and keeps normalization in one place. `is None` is
    # deliberate: a caller that explicitly passes [] is stating "no
    # history", and is respected.
    # -------------------------------------------------------------

    if conversation_history is None and conversation is not None and user_id is not None:

        try:

            from services import conversation_service

            prior_messages = conversation_service.list_messages(
                db=db,
                conversation_id=conversation.id,
                user_id=user_id,
            )

            history = _safe_conversation_history(
                [
                    {
                        "role": (
                            m.role.value
                            if hasattr(m.role, "value")
                            else str(m.role)
                        ),
                        "content": m.content,
                    }
                    for m in prior_messages
                ]
            )

        except Exception as exc:

            # Malformed/unavailable history must never crash
            # understanding -- degrade to no context instead.
            logger.warning(
                "manager.history_load_failed conversation_id=%s error=%s",
                conversation_id,
                exc,
            )

            history = []

    conversation_knowledge_base_ids = (
        _get_conversation_knowledge_base_ids(
            db=db,
            conversation_id=(
                conversation.id if conversation is not None else None
            ),
            user_id=user_id,
        )
    )

    execution_id = _generate_execution_id()

    execution = OrchestrationExecution(
        id=execution_id,

        user_id=user_id,

        user_input=user_input,

        status=OrchestrationStatus.RUNNING,
    )

    # -------------------------------------------------------------
    # Link persistent conversation when supplied.
    # -------------------------------------------------------------

    if conversation is not None:
        execution.conversation_id = conversation.id

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

            execution_id=execution_id,

            user_input=user_input,

            user_id=user_id,

            provider=provider,

            model=model,

            conversation_history=history,
            conversation_id=conversation_id,
            conversation_knowledge_base_ids=conversation_knowledge_base_ids,
        )

    except Exception as exc:

        result = {
            "status": "failed",
            "error": str(exc),
            "final_response": None,
            "plan": {},
            "step_results": {},
            "proposed_agent": None,
        }

    latency_ms = (
        time.monotonic() - start
    ) * 1000
    from .decision import decision_for_result
    execution.decision = decision_for_result(result)
    execution.failure_code = result.get("failure_code")

    plan = (
        result.get("plan")
        or {}
    )

    step_results = (
        result.get("step_results")
        or {}
    )

    proposed_agent = _normalize_proposed_agent(
        result.get("proposed_agent")
    )

    execution.required_capabilities = [
        step.get("capability")
        for step in _safe_list(
            plan.get("steps")
        )
        if (
            isinstance(step, dict)
            and step.get("capability")
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
            execution.proposed_agent = proposed_agent

        execution.status = (
            OrchestrationStatus
            .PENDING_AGENT_APPROVAL
        )

        execution.final_response = (
            result.get("final_response")
        )

        execution.error_message = None

        execution.end_time = (
            datetime.now(
                timezone.utc
            )
        )

        execution.latency_ms = latency_ms

        # Partial-match executions may already have completed useful
        # existing-agent steps before the missing capability proposal.
        # Persist them even while approval is pending so continuation can
        # resume instead of rerunning successful work.
        _persist_plan_steps(
            db=db,
            execution_id=execution_id,
            plan=plan,
            step_results=step_results,
        )

        db.commit()
        db.refresh(execution)

        return _proposal_response(
            db,
            execution,
            result=execution.final_response,
            error=None,
            sources=_result_sources(step_results),
        )

    # -------------------------------------------------------------
    # Normal execution
    # -------------------------------------------------------------

    _persist_plan_steps(
        db=db,

        execution_id=execution_id,

        plan=plan,

        step_results=step_results,
    )

    execution.status = (
        _STATUS_MAP.get(
            overall_status,
            OrchestrationStatus.FAILED,
        )
    )

    execution.final_response = (
        result.get("final_response")
    )

    execution.error_message = (
        result.get("error")
    )

    if hasattr(
        execution,
        "proposed_agent",
    ):
        execution.proposed_agent = proposed_agent

    execution.end_time = (
        datetime.now(
            timezone.utc
        )
    )

    execution.latency_ms = latency_ms

    db.commit()

    db.refresh(execution)

    return OrchestrateResponse(
        execution_id=execution_id,
        decision=execution.decision,
        failure_code=execution.failure_code,

        status=execution.status.value,

        result=execution.final_response,

        error=execution.error_message,

        proposed_agent=proposed_agent,
    )


# ---------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------

def edit_pending_agent_proposal(
    db: Session,
    execution_id: str,
    request: ProposalEditRequest,
    user_id: Optional[int] = None,
) -> OrchestrateResponse:
    """Edit only user-configurable proposal fields before approval."""
    if user_id is None:
        raise ValueError("Authenticated user required")
    query = db.query(OrchestrationExecution).filter(OrchestrationExecution.id == execution_id)
    if user_id is not None:
        query = query.filter(OrchestrationExecution.user_id == user_id)
    execution = query.with_for_update().first()
    if not execution:
        raise ValueError("Execution not found")
    if execution.status != OrchestrationStatus.PENDING_AGENT_APPROVAL:
        raise ValueError("Execution is not awaiting proposal review")

    proposal = dict(execution.proposed_agent or {})
    if not proposal:
        raise ValueError("No active proposal exists")
    if proposal.get("proposal_state") in {"approved", "created", "rejected", "cancelled"}:
        raise ValueError("Proposal can no longer be edited")
    if proposal.get("operation") == "delete":
        raise ValueError("Delete proposals do not have editable agent fields")

    updates = request.model_dump(exclude_unset=True)
    if updates.get("provider") and updates["provider"] != proposal.get("provider") and "model" not in updates:
        from .lifecycle import _default_model
        updates["model"] = _default_model(updates["provider"])
    proposal.update(updates)
    proposal["proposal_state"] = "editing"

    from .schemas import ProposedAgentSpec, ProposalState
    from services import agent_service, knowledge_service

    # A client can edit the form, but cannot use that edit to bypass the
    # lifecycle approval gate. `requires_approval` is the created agent's
    # runtime policy field, not permission to auto-create it.
    try:
        spec = ProposedAgentSpec.model_validate(proposal)
        agent_service.validate_agent_provider(spec.provider)
        agent_service.validate_agent_tools(spec.tools)

        if spec.is_rag:
            if not spec.knowledge_base_id:
                spec.proposal_state = ProposalState.KNOWLEDGE_SOURCE_REQUIRED
            else:
                kb = knowledge_service.get_knowledge_base(
                    db, spec.knowledge_base_id, user_id=execution.user_id
                )
                if not kb:
                    raise ValueError("Knowledge base not found")
                if not knowledge_service.is_knowledge_base_ready(
                    db, kb.id, user_id=execution.user_id
                ):
                    raise ValueError("Selected knowledge base is not ready")
                spec.proposal_state = ProposalState.READY_FOR_APPROVAL
        else:
            spec.knowledge_base_id = None
            spec.proposal_state = ProposalState.READY_FOR_APPROVAL

        execution.proposed_agent = spec.model_dump(mode="json")
        execution.final_response = (
            "Proposal updated. Attach a ready Knowledge Base before approval."
            if spec.proposal_state == ProposalState.KNOWLEDGE_SOURCE_REQUIRED
            else "Proposal updated and ready for explicit approval."
        )
        execution.error_message = None
        db.commit()
        db.refresh(execution)
        return _proposal_response(db, execution)
    except Exception:
        db.rollback()
        raise


def _existing_step_state(execution: OrchestrationExecution):
    """Rehydrate persisted task results for continuation after creation."""
    from .schemas import ExecutionStep
    plan_steps = []
    results: Dict[str, Dict[str, Any]] = {}
    completed: List[str] = []
    for row in execution.steps:
        plan_steps.append(ExecutionStep(
            step_id=row.step_key,
            tool_name=row.tool_name,
            agent_id=row.agent_id,
            capability=row.capability,
            task=row.task,
            depends_on=list(row.depends_on or []),
        ))
        if row.status != StepStatus.PENDING:
            status_value = row.status.value if hasattr(row.status, "value") else str(row.status)
            results[row.step_key] = {
                "step_id": row.step_key,
                "agent_id": row.agent_id,
                "capability": row.capability,
                "status": status_value,
                "result": row.result,
                "error": row.error,
            }
            completed.append(row.step_key)
    return plan_steps, results, completed


def approve_pending_agent(db, execution_id, approved, reason=None, user_id=None):
    """Approve/reject a durable create/update/delete proposal.

    The execution row is locked so double-submits are idempotent and another
    user cannot approve an operation they do not own.
    """
    from services.agent_service import create_agent, update_agent, delete_agent
    from schemas.agent import AgentCreate, AgentUpdate
    from .schemas import (
        ExecutionStep, ExecutionPlan, ProposedAgentSpec,
        LifecycleOperation, ProposalState,
    )
    from .execution import execute_wave
    from .manager_agent import aggregate
    from services import knowledge_service

    if user_id is None:
        raise ValueError("Authenticated user required")
    query = db.query(OrchestrationExecution).filter(OrchestrationExecution.id == execution_id)
    if user_id is not None:
        query = query.filter(OrchestrationExecution.user_id == user_id)
    execution = query.with_for_update().first()
    if execution is None:
        raise ValueError("Execution not found")

    proposal = dict(execution.proposed_agent or {})
    proposal_state = proposal.get("proposal_state")
    if proposal_state in {"approved", "created", "rejected", "cancelled"}:
        return _proposal_response(
            db, execution, created_agent_id=proposal.get("created_agent_id")
        )
    if execution.status != OrchestrationStatus.PENDING_AGENT_APPROVAL:
        raise ValueError("Execution is not awaiting approval")
    if approved and proposal_state == ProposalState.KNOWLEDGE_SOURCE_REQUIRED.value:
        raise ValueError("A ready Knowledge Base must be attached before approval")

    if not approved:
        proposal["proposal_state"] = ProposalState.REJECTED.value
        proposal["rejection_reason"] = reason
        execution.proposed_agent = proposal
        execution.status = OrchestrationStatus.FAILED
        execution.error_message = reason or "Agent proposal declined."
        execution.final_response = "The proposed agent operation was not applied."
        execution.end_time = datetime.now(timezone.utc)
        db.commit()
        db.refresh(execution)
        return _proposal_response(db, execution)

    try:
        spec = ProposedAgentSpec.model_validate(proposal)
        if spec.proposal_state != ProposalState.READY_FOR_APPROVAL:
            raise ValueError("Proposal is not ready for approval")

        if spec.is_rag:
            if not spec.knowledge_base_id or not knowledge_service.is_knowledge_base_ready(
                db, spec.knowledge_base_id, user_id=execution.user_id
            ):
                raise ValueError("Selected knowledge base is not ready")

        operation = spec.operation
        created_agent = None

        if operation == LifecycleOperation.CREATE:
            # Re-check duplicates under the approval lock. A matching agent
            # may have been registered while the user reviewed the form.
            from .routing import route_capabilities
            routes = route_capabilities(
                db, spec.capabilities, user_id=execution.user_id, requires_rag=spec.is_rag
            )
            all_resolved = bool(routes.routings) and all(r.candidates for r in routes.routings)
            selected_ids = {c.agent_id for r in routes.routings for c in r.candidates}
            if all_resolved and selected_ids and not spec.allow_duplicate:
                names = [(r.selected or r.candidates[0]).agent_name for r in routes.routings]
                proposal["proposal_state"] = ProposalState.CANCELLED.value
                proposal["duplicate_resolution"] = list(dict.fromkeys(names))
                execution.proposed_agent = proposal
                execution.status = OrchestrationStatus.SUCCESS
                execution.final_response = (
                    "No duplicate agent was created because existing registered resources now cover the proposal: "
                    + ", ".join(dict.fromkeys(names))
                )
                execution.error_message = None
                execution.end_time = datetime.now(timezone.utc)
                db.commit()
                db.refresh(execution)
                return _proposal_response(db, execution)

            created_agent = create_agent(
                db,
                AgentCreate(
                    name=spec.name,
                    description=spec.description,
                    system_prompt=spec.system_prompt,
                    capabilities=spec.capabilities,
                    input_schema=spec.input_schema,
                    output_schema=spec.output_schema,
                    tools=spec.tools,
                    is_rag=spec.is_rag,
                    knowledge_base_id=spec.knowledge_base_id,
                    visibility=spec.visibility,
                    timeout_seconds=spec.timeout_seconds,
                    max_retries=spec.max_retries,
                    requires_approval=spec.requires_approval,
                    model={
                        "provider": spec.provider,
                        "model": spec.model or settings.GEMINI_DEFAULT_MODEL,
                    },
                ),
                created_by=execution.user_id,
                commit=False,
            )
            proposal.update(
                proposal_state=ProposalState.APPROVED.value,
                created_agent_id=created_agent.id,
                created_agent_version=created_agent.current_version,
            )
            execution.proposed_agent = proposal
            execution.status = OrchestrationStatus.RUNNING
            if not spec.resume_task:
                proposal["proposal_state"] = ProposalState.CREATED.value
                execution.proposed_agent = dict(proposal)
                execution.status = OrchestrationStatus.SUCCESS
                execution.final_response = f'Agent "{created_agent.name}" was created successfully.'
                execution.end_time = datetime.now(timezone.utc)
                db.commit()
                db.refresh(execution)
                return _proposal_response(db, execution, created_agent_id=created_agent.id)
            db.commit()

        elif operation == LifecycleOperation.UPDATE:
            if not spec.target_agent_id:
                raise ValueError("Update proposal is missing its target agent")
            updated = update_agent(
                db,
                spec.target_agent_id,
                AgentUpdate(
                    name=spec.name,
                    description=spec.description,
                    system_prompt=spec.system_prompt,
                    capabilities=spec.capabilities,
                    input_schema=spec.input_schema,
                    output_schema=spec.output_schema,
                    tools=spec.tools,
                    is_rag=spec.is_rag,
                    knowledge_base_id=(spec.knowledge_base_id if spec.is_rag else ""),
                    visibility=spec.visibility,
                    timeout_seconds=spec.timeout_seconds,
                    max_retries=spec.max_retries,
                    requires_approval=spec.requires_approval,
                    model={"provider": spec.provider, "model": spec.model or settings.GEMINI_DEFAULT_MODEL},
                ),
                owner_id=execution.user_id,
            )
            if not updated:
                raise ValueError("Target agent not found")
            proposal["proposal_state"] = ProposalState.CREATED.value
            execution.proposed_agent = proposal
            execution.status = OrchestrationStatus.SUCCESS
            execution.final_response = f'Agent "{updated.name}" was updated successfully.'
            execution.error_message = None
            execution.end_time = datetime.now(timezone.utc)
            db.commit()
            db.refresh(execution)
            return _proposal_response(db, execution)

        elif operation == LifecycleOperation.DELETE:
            if not spec.target_agent_id:
                raise ValueError("Delete proposal is missing its target agent")
            deleted = delete_agent(db, spec.target_agent_id, owner_id=execution.user_id)
            if not deleted:
                raise ValueError("Target agent not found")
            proposal["proposal_state"] = ProposalState.CREATED.value
            execution.proposed_agent = proposal
            execution.status = OrchestrationStatus.SUCCESS
            execution.final_response = f'Agent "{spec.name}" was deleted successfully.'
            execution.error_message = None
            execution.end_time = datetime.now(timezone.utc)
            db.commit()
            db.refresh(execution)
            return _proposal_response(db, execution)
        else:
            raise ValueError("Unsupported approval operation")

    except Exception:
        db.rollback()
        raise

    # CREATE continuation: preserve already-completed steps when the proposal
    # came from a partial capability gap, and run only the new unresolved task.
    execution = (
        db.query(OrchestrationExecution)
        .filter(OrchestrationExecution.id == execution_id)
        .one()
    )
    old_steps, previous_results, completed = _existing_step_state(execution)
    dependency_ids = [
        key for key, result in previous_results.items() if result.get("status") == "success"
    ]
    new_step_id = "approved_task"
    existing_ids = {step.step_id for step in old_steps}
    suffix = 1
    while new_step_id in existing_ids:
        suffix += 1
        new_step_id = f"approved_task_{suffix}"

    old_steps.append(ExecutionStep(
        step_id=new_step_id,
        agent_id=created_agent.id,
        agent_version=created_agent.current_version,
        capability=spec.capabilities[0],
        task=(
            "Complete the unresolved capability for the original request: "
            + ", ".join(spec.capabilities)
        ),
        depends_on=dependency_ids,
    ))
    plan = ExecutionPlan(request=execution.user_input, steps=old_steps)

    state = dict(
        db=db, execution_id=execution.id, user_id=execution.user_id,
        user_input=execution.user_input, provider=execution.provider or "gemini",
        model=execution.model or settings.GEMINI_DEFAULT_MODEL,
        plan=plan.model_dump(), step_results=previous_results,
        completed=completed, skipped=[], iteration=0,
        deadline=time.monotonic() + settings.MAX_ORCHESTRATION_EXECUTION_TIME,
        conversation_knowledge_base_ids=_get_conversation_knowledge_base_ids(
            db, execution.conversation_id, execution.user_id
        ),
        missing_capabilities=[], unavailable_capabilities=[],
        proposed_agent=None, needs_approval=False, capability_gap=False,
    )
    try:
        execute_wave(state)
        aggregate(state)
        _persist_plan_steps(db, execution.id, state["plan"], state["step_results"])
        execution.status = _STATUS_MAP.get(state["status"], OrchestrationStatus.FAILED)
        execution.final_response = state.get("final_response")
        execution.error_message = state.get("error")
        proposal["proposal_state"] = ProposalState.CREATED.value
        execution.proposed_agent = proposal
    except Exception:
        logger.exception("Approved agent continuation failed execution_id=%s", execution_id)
        db.rollback()
        execution = db.query(OrchestrationExecution).filter(OrchestrationExecution.id == execution_id).one()
        execution.status = OrchestrationStatus.FAILED
        execution.error_message = "Agent was registered, but continuation failed."
        execution.final_response = "The agent was created, but the original task could not be continued."
    execution.end_time = datetime.now(timezone.utc)
    db.commit()
    db.refresh(execution)
    return _proposal_response(
        db, execution, created_agent_id=created_agent.id,
        sources=_result_sources(state.get("step_results", {})),
    )


def maybe_handle_natural_language_approval(
    db: Session,
    conversation_id: str,
    user_id: int,
    text: str,
) -> Optional[OrchestrateResponse]:
    """Resolve exact approve/reject replies only against one pending proposal.

    A bare "yes" with no pending proposal returns None and is handled as an
    ordinary new chat message. Multiple pending proposals are intentionally not
    guessed; the user must approve by execution ID/form.
    """
    normalized = " ".join((text or "").strip().casefold().split())
    approve_words = {"yes", "approve", "approved", "go ahead", "create it", "do it"}
    reject_words = {"no", "reject", "decline", "cancel", "don't create it", "do not create it"}
    if normalized not in approve_words | reject_words:
        return None

    rows = (
        db.query(OrchestrationExecution)
        .filter(
            OrchestrationExecution.user_id == user_id,
            OrchestrationExecution.conversation_id == conversation_id,
            OrchestrationExecution.status == OrchestrationStatus.PENDING_AGENT_APPROVAL,
        )
        .order_by(OrchestrationExecution.start_time.desc())
        .all()
    )
    if not rows:
        return None
    if len(rows) != 1:
        latest = rows[0]
        response = _proposal_response(
            db, latest,
            result="There is more than one pending agent proposal in this conversation. Please approve the specific proposal from its form/execution ID.",
            error=None,
        )
        response.decision = "CLARIFICATION_REQUIRED"
        response.failure_code = "AMBIGUOUS_APPROVAL"
        return response

    execution = rows[0]
    if normalized in reject_words:
        return approve_pending_agent(
            db, execution.id, approved=False, reason="Rejected in conversation", user_id=user_id
        )

    proposal = execution.proposed_agent or {}
    if proposal.get("proposal_state") != "ready_for_approval":
        return _proposal_response(
            db, execution,
            result=(
                "This proposal is not ready for approval yet. Attach a ready Knowledge Base first."
                if proposal.get("proposal_state") == "knowledge_source_required"
                else "This proposal is not currently ready for approval."
            ),
            error=None,
        )
    return approve_pending_agent(
        db, execution.id, approved=True, reason="Approved in conversation", user_id=user_id
    )


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

    proposed_agent = _normalize_proposed_agent(
        getattr(
            execution,
            "proposed_agent",
            None,
        )
    )

    if execution.status != OrchestrationStatus.PENDING_AGENT_APPROVAL:
        proposed_agent = None
    needs_kb = bool(proposed_agent and proposed_agent.get("proposal_state") == "knowledge_source_required")
    return ExecutionStatusResponse(
        execution_id=execution.id,
        decision=execution.decision,
        failure_code=execution.failure_code,
        knowledge_source_required=needs_kb,
        knowledge_source_choices=["attach_existing", "create_new"] if needs_kb else [],
        available_knowledge_bases=_knowledge_options_for_proposal(db, execution.user_id, proposed_agent) if needs_kb else [],

        status=execution.status,

        user_input=execution.user_input,

        required_capabilities=(
            execution.required_capabilities
        ),

        current_step=current_step,

        completed_steps=completed_steps,

        steps=[
            {
                "step_key": step.step_key,
                "tool_name": step.tool_name,

                "agent_id": step.agent_id,

                "capability": step.capability,

                "task": step.task,

                "depends_on": step.depends_on,

                "status": step.status,

                "result": step.result,

                "error": step.error,
            }
            for step in execution.steps
        ],

        final_response=(
            execution.final_response
        ),

        error=(
            execution.error_message
        ),

        proposed_agent=proposed_agent,

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
