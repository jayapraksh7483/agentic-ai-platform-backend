from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db

from schemas.agent import (
    AgentExecuteRequest,
    AgentExecuteResponse,
)

from services import (
    agent_api_key_service,
    executor_service,
)


router = APIRouter(
    prefix="/api/v1/agents",
    tags=["Public Agent API"],
)


def _extract_bearer_token(
    authorization: Optional[str],
) -> str:
    value = str(authorization or "").strip()

    if not value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = value.split(None, 1)

    if (
        len(parts) != 2
        or parts[0].lower() != "bearer"
        or not parts[1].strip()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization must use Bearer authentication",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return parts[1].strip()


@router.post(
    "/{agent_id}/invoke",
    response_model=AgentExecuteResponse,
)
def invoke_published_agent(
    agent_id: str,
    request: AgentExecuteRequest,
    authorization: Optional[str] = Header(
        default=None,
        alias="Authorization",
    ),
    db: Session = Depends(get_db),
):
    """
    External application endpoint.

    Authentication:
        Authorization: Bearer ag_live_...

    The API key is scoped to one Agent. On successful authentication
    we reuse the existing executor_service and pass the owning
    user_id so existing Agent ownership and RAG/KB isolation rules
    remain intact.
    """

    raw_key = _extract_bearer_token(
        authorization
    )

    try:
        api_key = (
            agent_api_key_service
            .authenticate_external_api_key(
                db=db,
                agent_id=agent_id,
                raw_key=raw_key,
            )
        )

    except (
        agent_api_key_service
        .AgentAPIKeyUnauthorizedError
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return executor_service.execute_agent(
            db=db,
            agent_id=agent_id,
            request=request,
            user_id=api_key.user_id,
        )

    except executor_service.AgentNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except executor_service.AgentInactiveError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )
