from typing import List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from api.agents import _to_out
 
from models.agent import AgentStatus
 
from schemas.agent import AgentOut, ModelConfig
from services import discovery_service


router = APIRouter(prefix="/api/agents", tags=["Discovery"])


@router.get("/discover", response_model=List[AgentOut])
def discover_agents(
    capability: Optional[str] = None,
    status_filter: Optional[AgentStatus] = AgentStatus.ACTIVE,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
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
        owner_id=current_user.id,
    )

    return [_to_out(agent) for agent in agents]
