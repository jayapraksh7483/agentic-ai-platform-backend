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
    """

    try:
        return manager_service.orchestrate(
            db=db,
            user_input=request.user_input,
            user_id=current_user.id,
        )

    except Exception as exc:
        logger.exception("Orchestration failed")

        raise HTTPException(
            status_code=500,
            detail=str(exc),
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
    """

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
    """

    try:
        return manager_service.approve_pending_agent(
            db=db,
            execution_id=execution_id,
            approved=request.approved,
            user_id=current_user.id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:
        logger.exception("Approval failed")

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )