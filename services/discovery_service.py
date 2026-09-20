from typing import List, Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session, joinedload

from models.agent import Agent, AgentCapability, AgentStatus


def _normalize_capability(value: str) -> str:
    """
    Canonical capability form: lowercase, underscores/hyphens to
    spaces, collapsed whitespace. Shared by the SQL filter above and
    the routing layer so both agree on what "the same capability"
    means.
    """

    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())

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

# =====================================================================
# PHASE 2 -- BATCHED / STRUCTURED DISCOVERY
# =====================================================================

def discover_agents_for_capabilities(
    db: Session,
    capabilities: List[str],
    owner_id: Optional[int] = None,
    include_inactive: bool = False,
) -> List[Agent]:
    """
    Discover, in ONE query, every agent that could serve ANY of the
    supplied capabilities.

    Why this exists: routing previously called discover_agents() once
    per required capability, so a 3-capability request issued 3
    separate registry queries (N+1). At 1,000+ agents and multi-
    capability requests that is the dominant cost of routing. One
    IN-clause over an indexed column replaces it, and the caller
    groups the result in memory.

    `include_inactive` is deliberate: the eligibility layer needs to
    SEE inactive/unusable agents in order to distinguish
    "this capability has no provider at all" (a capability gap) from
    "every provider is currently disabled" (an availability problem).
    Filtering them out in SQL would destroy that distinction -- which
    is exactly how a temporarily disabled Calculator could end up
    looking like a missing calculation capability.

    Matching here is deliberately BROAD (normalized equality on the
    capability name). Ranking and final selection are the routing
    layer's job, not the database's.
    """

    if not capabilities:
        return []

    # Normalize for a case-insensitive IN-clause. The normalized form
    # is matched again (more strictly) in the routing layer.
    wanted = {
        _normalize_capability(c)
        for c in capabilities
        if c and str(c).strip()
    }

    if not wanted:
        return []

    # Build the Agent query WITHOUT joining AgentCapability directly.
    #
    # PostgreSQL JSON columns do not have a default equality operator.
    # The previous implementation joined AgentCapability and then used
    # query.distinct(). Because SQLAlchemy selected the entire Agent row,
    # PostgreSQL tried to apply DISTINCT to JSON columns such as
    # agents.tools/input_schema/output_schema and failed with:
    #
    #   could not identify an equality operator for type json
    #
    # Instead, find distinct matching agent IDs in a small subquery and
    # filter the Agent query by those IDs. DISTINCT is therefore applied
    # only to AgentCapability.agent_id, never to JSON columns.
    query = (
        db.query(Agent)
        .options(joinedload(Agent.capabilities))
    )

    if owner_id is not None:
        query = query.filter(
            Agent.created_by == owner_id
        )

    if not include_inactive:
        query = query.filter(
            Agent.status == AgentStatus.ACTIVE
        )

    # DB-side narrowing: lower()/replace() on the capability column so
    # the registry is filtered in SQL rather than in Python. This is
    # what keeps routing from loading every agent the user owns.
    normalized_column = func.replace(
        func.replace(
            func.lower(AgentCapability.capability_name),
            "_",
            " ",
        ),
        "-",
        " ",
    )

    # Narrowing must be BROADER than exact-normalized equality,
    # because the routing layer also accepts token-overlap matches
    # ("calculation" should find an agent registered as "arithmetic
    # calculation"). An IN-clause alone would filter those out in SQL
    # before routing ever saw them, silently capping match quality at
    # NORMALIZED and making token matching dead code.
    #
    # So: exact-normalized IN (fast, indexed) OR a LIKE per significant
    # token. Still DB-side, so the full registry is never loaded into
    # memory; routing then does the precise ranking on a small set.
    conditions = [normalized_column.in_(wanted)]

    significant_tokens = {
        token
        for capability in wanted
        for token in capability.split()
        if len(token) > 2
    }

    for token in significant_tokens:
        conditions.append(
            normalized_column.like(f"%{token}%")
        )

    matching_agent_ids = (
        db.query(AgentCapability.agent_id)
        .filter(or_(*conditions))
        .distinct()
    )

    query = query.filter(
        Agent.id.in_(matching_agent_ids)
    )

    return query.all()


def discover_agent_by_name(
    db: Session,
    name: str,
    owner_id: Optional[int] = None,
) -> Optional[Agent]:
    """
    Resolve an agent the user named explicitly ("run the calculator
    agent").

    Returns the agent regardless of status -- the caller must still
    run it through eligibility. Returning None here means "no such
    agent in YOUR scope", which the caller reports as AGENT_NOT_FOUND;
    it must never be treated as a reason to create one.

    Ownership scoping is applied in SQL, so another user's private
    agent is indistinguishable from a nonexistent one.
    """

    if not name or not str(name).strip():
        return None

    cleaned = str(name).strip().lower()

    query = (
        db.query(Agent)
        .options(joinedload(Agent.capabilities))
        .filter(func.lower(Agent.name) == cleaned)
    )

    if owner_id is not None:
        query = query.filter(
            Agent.created_by == owner_id
        )

    return query.first()