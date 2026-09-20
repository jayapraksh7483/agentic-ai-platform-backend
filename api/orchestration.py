import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user

from schemas.orchestration import (
    ApprovalRequest,
    ExecutionStatusResponse,
    OrchestrateRequest,
    OrchestrateResponse,
    ProposalEditRequest,
)

from manager import service as manager_service


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/orchestrate",
    tags=["Orchestration"],
)


# ============================================================
# MANAGER AGENT / ORCHESTRATION
# ============================================================

@router.post(
    "",
    response_model=OrchestrateResponse,
)
def orchestrate(
    request: OrchestrateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Manager Agent entry point.

    Understands the request, dynamically discovers registered
    agents belonging to the authenticated user, executes them,
    and returns the final response.

    If conversation_id is provided, the orchestration execution
    is linked to that persistent conversation.
    """

    try:
        return manager_service.orchestrate(
            db=db,
            user_input=request.user_input,
            user_id=current_user.id,
            conversation_id=getattr(
                request,
                "conversation_id",
                None,
            ),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception:
        logger.exception("Orchestration failed")

        raise HTTPException(
            status_code=500,
            detail="Orchestration failed unexpectedly.",
        )


# ============================================================
# GET ORCHESTRATION STATUS
# ============================================================

@router.get(
    "/{execution_id}",
    response_model=ExecutionStatusResponse,
)
def get_orchestration_status(
    execution_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Return orchestration status only for the authenticated user's
    orchestration execution.

    When the execution is pending agent approval, proposed_agent
    is included in the response.
    """

    try:
        result = manager_service.get_execution(
            db=db,
            execution_id=execution_id,
            user_id=current_user.id,
        )

        if not result:
            raise HTTPException(
                status_code=404,
                detail="Execution not found",
            )

        return result

    except HTTPException:
        raise

    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Execution not found",
        )

    except Exception as exc:
        logger.exception(
            "Failed to retrieve orchestration execution"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# EDIT PENDING AGENT PROPOSAL / ATTACH KNOWLEDGE BASE
# ============================================================

@router.patch(
    "/{execution_id}/proposal",
    response_model=OrchestrateResponse,
)
def edit_orchestration_proposal(
    execution_id: str,
    request: ProposalEditRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Edit the Manager-generated form before explicit approval.

    For RAG agents, setting knowledge_base_id here validates ownership and
    READY indexing state. The client can create/upload a new KB using the
    existing /api/knowledge-bases endpoints, then attach its ID here.
    """
    try:
        return manager_service.edit_pending_agent_proposal(
            db=db,
            execution_id=execution_id,
            request=request,
            user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Proposal edit failed")
        raise HTTPException(status_code=500, detail="Proposal edit failed unexpectedly.")


# ============================================================
# APPROVE / REJECT PROPOSED AGENT
# ============================================================

@router.post(
    "/{execution_id}/approve",
    response_model=OrchestrateResponse,
)
def approve_orchestration(
    execution_id: str,
    request: ApprovalRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Approve or reject a proposed agent.

    Only the user who owns the orchestration execution can
    approve or reject it.

    If approved, the manager re-runs the original request using
    the conversation associated with the execution.

    If rejected, the optional reason is persisted with the
    orchestration execution.
    """

    try:
        # Save the user's edited form values immediately before approval.
        # This keeps "edit + create" atomic from the Chat UI.
        if (
            request.approved
            and request.proposed_agent is not None
        ):
            manager_service.edit_pending_agent_proposal(
                db=db,
                execution_id=execution_id,
                request=request.proposed_agent,
                user_id=current_user.id,
            )

        return manager_service.approve_pending_agent(
            db=db,
            execution_id=execution_id,
            approved=request.approved,
            reason=request.reason,
            user_id=current_user.id,
        )

    except ValueError as exc:
        # Previously every ValueError became 404, which hid the real
        # creation/validation error behind "Request failed with 404".
        detail = str(exc)
        lowered = detail.casefold()

        if "not found" in lowered:
            status_code = 404
        elif (
            "not awaiting" in lowered
            or "already" in lowered
            or "pending" in lowered
        ):
            status_code = 409
        else:
            status_code = 422

        raise HTTPException(
            status_code=status_code,
            detail=detail,
        )

    except Exception as exc:
        logger.exception("Approval failed")

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )