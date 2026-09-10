from typing import List, Optional

from sqlalchemy.orm import Session, joinedload

from models.agent import Agent, AgentCapability, AgentStatus


def discover_agents(
    db: Session,
    capability: Optional[str] = None,
    status_filter: Optional[AgentStatus] = AgentStatus.ACTIVE,
    owner_id: Optional[int] = None,
) -> List[Agent]:
    """
    Capability-based agent discovery.

    Phase 5B - User Ownership & Isolation
    --------------------------------------

    When owner_id is provided:
        Only agents belonging to that user are returned.

    When owner_id is None:
        Discovery remains unscoped for internal/system use.

    Examples:

        discover_agents(
            db,
            capability="weather",
            owner_id=10,
        )

        -> Returns only ACTIVE weather agents owned by user 10.

    No capability -> returns all agents matching the status filter
    and ownership filter.
    """

    query = (
        db.query(Agent)
        .options(
            joinedload(Agent.capabilities)
        )
    )

    # -------------------------------------------------------------
    # STATUS FILTER
    # -------------------------------------------------------------

    if status_filter:
        query = query.filter(
            Agent.status == status_filter
        )

    # -------------------------------------------------------------
    # PHASE 5B - OWNER FILTER
    # -------------------------------------------------------------
    #
    # This is the important isolation rule:
    #
    # User A -> only User A's agents
    # User B -> only User B's agents
    #
    # The Manager passes current_user.id as owner_id.
    # -------------------------------------------------------------

    if owner_id is not None:
        query = query.filter(
            Agent.created_by == owner_id
        )

    # -------------------------------------------------------------
    # CAPABILITY FILTER
    # -------------------------------------------------------------

    if capability:
        query = (
            query
            .join(AgentCapability)
            .filter(
                AgentCapability.capability_name
                == capability
            )
        )

    return query.all()