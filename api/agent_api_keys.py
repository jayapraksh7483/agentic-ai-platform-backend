from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user

from schemas.agent_api_key import (
    AgentAPIKeyCreateRequest,
    AgentAPIKeyCreatedResponse,
    AgentAPIKeyOut,
)

from services import agent_api_key_service


router = APIRouter(
    prefix="/api/agents",
    tags=["Agent API Access"],
)


@router.post(
    "/{agent_id}/api-keys",
    response_model=AgentAPIKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_agent_api_key(
    agent_id: str,
    request: AgentAPIKeyCreateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        record, raw_key = (
            agent_api_key_service.create_api_key(
                db=db,
                agent_id=agent_id,
                user_id=current_user.id,
                expires_at=request.expires_at,
            )
        )

    except agent_api_key_service.AgentAPIKeyNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    return AgentAPIKeyCreatedResponse(
        id=record.id,
        agent_id=record.agent_id,
        key_prefix=record.key_prefix,
        is_active=record.is_active,
        created_at=record.created_at,
        expires_at=record.expires_at,
        last_used_at=record.last_used_at,
        revoked_at=record.revoked_at,
        api_key=raw_key,
        invoke_path=(
            f"/api/v1/agents/{record.agent_id}/invoke"
        ),
    )


@router.get(
    "/{agent_id}/api-keys",
    response_model=List[AgentAPIKeyOut],
)
def list_agent_api_keys(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        return agent_api_key_service.list_api_keys(
            db=db,
            agent_id=agent_id,
            user_id=current_user.id,
        )

    except agent_api_key_service.AgentAPIKeyNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


@router.delete(
    "/{agent_id}/api-keys/{key_id}",
    response_model=AgentAPIKeyOut,
)
def revoke_agent_api_key(
    agent_id: str,
    key_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        return agent_api_key_service.revoke_api_key(
            db=db,
            agent_id=agent_id,
            key_id=key_id,
            user_id=current_user.id,
        )

    except agent_api_key_service.AgentAPIKeyNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
