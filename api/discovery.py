from typing import List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.database import get_db
 
from models.agent import AgentStatus
 
from schemas.agent import AgentOut, ModelConfig
from services import discovery_service


router = APIRouter(prefix="/api/agents", tags=["Discovery"])


@router.get("/discover", response_model=List[AgentOut])
def discover_agents(
    capability: Optional[str] = None,
    status_filter: Optional[AgentStatus] = AgentStatus.ACTIVE,
    db: Session = Depends(get_db),
 
):
    """
    Example: GET /api/agents/discover?capability=sql_query
    Returns agents matching the requested capability, filtered by status.

    Consumed by the AI orchestration layer to find the right agent for a task.
    """

    agents = discovery_service.discover_agents(
        db,
        capability=capability,
        status_filter=status_filter,
    )

    return [
        AgentOut(
            id=a.id,
            name=a.name,
            description=a.description,
            status=a.status,
            current_version=a.current_version,
            system_prompt=a.system_prompt,
            model_cfg=ModelConfig(
                provider=a.provider,
                model=a.model,
            ),
            input_schema=a.input_schema,
            output_schema=a.output_schema,
            capabilities=[c.capability_name for c in a.capabilities],
        )
        for a in agents
    ]