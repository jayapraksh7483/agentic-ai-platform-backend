from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user

from models.agent import AgentStatus

from schemas.agent import (
    AgentCreate,
    AgentUpdate,
    AgentOut,
    AgentStatusUpdate,
    ModelConfigOut,
    AgentExecuteRequest,
    AgentExecuteResponse,
)

from services import (
    agent_service,
    executor_service,
    builder_service,
    validator_service,
)
from services.agent_service import _mask_api_key


router = APIRouter(
    prefix="/api/agents",
    tags=["Agents"],
)


class AgentBuildRequest(BaseModel):
    prompt: str
    provider: str = "gemini"
    model: str = "gemini-3.5-flash-lite"


class AgentValidateRequest(BaseModel):
    spec: dict


def _to_out(agent) -> AgentOut:
    return AgentOut(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        status=agent.status,
        current_version=agent.current_version,
        system_prompt=agent.system_prompt,

        model_cfg=ModelConfigOut(
            provider=agent.provider,
            model=agent.model,
            temperature=agent.temperature,
            api_key_configured=bool(agent.api_key_encrypted),
            api_key_preview=_mask_api_key(agent.api_key_last4),
        ),

        input_schema=agent.input_schema,
        output_schema=agent.output_schema,

        tools=agent.tools,

        is_rag=agent.is_rag,
        knowledge_base_id=agent.knowledge_base_id,

        visibility=getattr(agent, "visibility", "private"),
        timeout_seconds=getattr(agent, "timeout_seconds", 30),
        max_retries=getattr(agent, "max_retries", 2),
        requires_approval=getattr(agent, "requires_approval", False),

        # Important for frontend grouping
        is_default=agent.is_default,

        capabilities=[
            c.capability_name
            for c in agent.capabilities
        ],
    )


# ============================================================
# AI AGENT BUILDER
# ============================================================

@router.post("/build")
def build_agent(
    request: AgentBuildRequest,
    current_user=Depends(get_current_user),
):
    try:

        service = builder_service.BuilderService()

        specification = service.build_agent(
            user_prompt=request.prompt,
            provider=request.provider,
            model=request.model,
        )

        return {
            "data": specification,
            "message": (
                "Agent specification generated successfully"
            ),
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# AI AGENT VALIDATOR
# ============================================================

@router.post("/validate")
def validate_agent(
    request: AgentValidateRequest,
    current_user=Depends(get_current_user),
):

    try:

        result = validator_service.validator_service.validate(
            request.spec
        )

        return {
            "data": result,
            "message": (
                "Agent specification validation completed"
            ),
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# AGENT REGISTRY
# ============================================================

@router.post(
    "",
    response_model=AgentOut,
    status_code=status.HTTP_201_CREATED,
)
def register_agent(
    agent_in: AgentCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):

    try:

        agent = agent_service.create_agent(
            db,
            agent_in,
            created_by=current_user.id,
            is_default=False,
        )

    except ValueError as e:

        raise HTTPException(
            status_code=422,
            detail=str(e),
        )

    return _to_out(agent)


@router.get(
    "",
    response_model=List[AgentOut],
)
def list_agents(
    status_filter: Optional[AgentStatus] = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Return all agents belonging to the authenticated user.

    This includes:
        - platform default agents
        - user-created agents

    Frontend separates them using is_default.
    """

    agents = agent_service.list_agents(
        db,
        status_filter=status_filter,
        owner_id=current_user.id,
    )

    return [
        _to_out(agent)
        for agent in agents
    ]


@router.get(
    "/{agent_id}",
    response_model=AgentOut,
)
def get_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):

    agent = agent_service.get_agent(
        db,
        agent_id,
        owner_id=current_user.id,
    )

    if not agent:

        raise HTTPException(
            status_code=404,
            detail="Agent not found",
        )

    return _to_out(agent)


@router.put(
    "/{agent_id}",
    response_model=AgentOut,
)
def update_agent(
    agent_id: str,
    agent_in: AgentUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):

    try:

        agent = agent_service.update_agent(
            db,
            agent_id,
            agent_in,
            owner_id=current_user.id,
        )

    except ValueError as e:

        raise HTTPException(
            status_code=422,
            detail=str(e),
        )

    if not agent:

        raise HTTPException(
            status_code=404,
            detail="Agent not found",
        )

    return _to_out(agent)


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):

    try:
        deleted = agent_service.delete_agent(
            db,
            agent_id,
            owner_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    if not deleted:

        raise HTTPException(
            status_code=404,
            detail="Agent not found",
        )

    return None


@router.patch(
    "/{agent_id}/status",
    response_model=AgentOut,
)
def set_agent_status(
    agent_id: str,
    status_in: AgentStatusUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):

    agent = agent_service.set_agent_status(
        db,
        agent_id,
        status_in.status,
        owner_id=current_user.id,
    )

    if not agent:

        raise HTTPException(
            status_code=404,
            detail="Agent not found",
        )

    return _to_out(agent)


# ============================================================
# AGENT EXECUTION
# ============================================================

@router.post(
    "/{agent_id}/execute",
    response_model=AgentExecuteResponse,
)
def execute_agent(
    agent_id: str,
    request: AgentExecuteRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):

    try:

        result = executor_service.execute_agent(
            db,
            agent_id,
            request,
            user_id=current_user.id,
        )

        return result

    except executor_service.AgentNotFoundError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    except executor_service.AgentInactiveError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except ValueError as e:

        raise HTTPException(
            status_code=422,
            detail=str(e),
        )
