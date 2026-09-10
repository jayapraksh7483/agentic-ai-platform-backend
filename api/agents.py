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
    ModelConfig,
    AgentExecuteRequest,
    AgentExecuteResponse,
)

from services import (
    agent_service,
    executor_service,
    builder_service,
    validator_service,
)


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
        model_cfg=ModelConfig(
            provider=agent.provider,
            model=agent.model,
        ),
        input_schema=agent.input_schema,
        output_schema=agent.output_schema,
        is_rag=agent.is_rag,
        knowledge_base_id=agent.knowledge_base_id,
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
            "message": "Agent specification generated successfully",
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
):
    try:
        result = validator_service.validator_service.validate(
            request.spec
        )

        return {
            "data": result,
            "message": "Agent specification validation completed",
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
    """
    Register a new agent for the authenticated user.
    """

    try:
        agent = agent_service.create_agent(
            db,
            agent_in,
            created_by=current_user.id,
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
    Return only agents owned by the authenticated user.
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
    """
    Return an agent only if it belongs to the authenticated user.
    """

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
    """
    Update an agent only if it belongs to the authenticated user.
    """

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
    """
    Delete an agent only if it belongs to the authenticated user.
    """

    deleted = agent_service.delete_agent(
        db,
        agent_id,
        owner_id=current_user.id,
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
    """
    Change agent status only if it belongs to the authenticated user.
    """

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
    """
    Execute an agent.

    Authentication is required here, but ownership enforcement
    is handled in the executor layer in the next Phase 5B step.
    """

    try:
        result = executor_service.execute_agent(
            db,
            agent_id,
            request,
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