"""
Phase 2 -- Agent routing.

    required capabilities (from Phase 1 understanding)
        -> discovery      : which registered agents COULD serve this?
        -> eligibility    : which of those MAY actually be used?
        -> selection      : which eligible candidate SHOULD be used?
        -> RoutingResult

Three properties this module exists to guarantee:

1. The LLM never selects an agent. It names capabilities; every
   AgentCandidate here is built from a real registry row fetched by
   application code under the caller's ownership scope. An agent_id
   that the registry did not return cannot be constructed here, so it
   cannot reach execution.

2. "No provider exists" and "providers exist but none are usable" stay
   distinct. Only the former is a capability gap. Conflating them is
   how a temporarily disabled agent becomes a proposal to build a
   duplicate of it.

3. Partial matches survive. A request needing A, B, C where only C is
   missing still routes A and B. Discarding the whole plan because one
   capability is unmatched is a correctness bug, not a safety measure.
"""

import logging
from typing import Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from models.agent import Agent, AgentStatus
from services import discovery_service
from services.discovery_service import _normalize_capability

from .schemas import (
    AgentCandidate,
    CapabilityRouting,
    MatchQuality,
    RejectionReason,
    RoutingResult,
    RoutingStatus,
)

logger = logging.getLogger("manager")


# Words that carry no discriminating power when comparing capability
# phrases, so they must not create spurious TOKEN_OVERLAP matches
# ("document processing" vs "payment processing" share only "processing").
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "the", "of", "for", "to", "with", "on", "in",
        "or", "by", "using", "via", "that", "this", "it", "handling",
        "general", "generic", "task", "tasks", "request", "requests",
        "operation", "operations", "support", "agent",
    }
)


def _tokens(value: str) -> frozenset:
    return frozenset(
        token
        for token in _normalize_capability(value).split()
        if token and token not in _STOPWORDS and len(token) > 2
    )


def _match_quality(
    requirement: str,
    capability: str,
) -> Optional[MatchQuality]:
    """
    Layered capability matching, best tier first.

    Deliberately NOT semantic search: the tiers below are cheap,
    deterministic, and explainable, which matters more here than
    recall. A semantic tier can be added later as a final fallback
    without disturbing these, because match quality is an explicit
    ranked enum rather than an opaque score.
    """

    if requirement == capability:
        return MatchQuality.EXACT

    normalized_requirement = _normalize_capability(requirement)
    normalized_capability = _normalize_capability(capability)

    if normalized_requirement == normalized_capability:
        return MatchQuality.NORMALIZED

    requirement_tokens = _tokens(requirement)
    capability_tokens = _tokens(capability)

    if not requirement_tokens or not capability_tokens:
        return None

    # One side fully contained in the other ("calculation" vs
    # "arithmetic calculation"). Bare intersection is too loose and
    # produces the false positives the stopword list above guards
    # against.
    if (
        requirement_tokens <= capability_tokens
        or capability_tokens <= requirement_tokens
    ):
        return MatchQuality.TOKEN_OVERLAP

    return None


def _evaluate_eligibility(
    agent: Agent,
    user_id: Optional[int],
    requires_rag: bool,
) -> Optional[RejectionReason]:
    """
    Policy gate. Returns the reason this agent may NOT be used, or None
    if it is eligible.

    Ownership is re-checked here even though discovery already scoped
    the query, because defence in depth at the point of use is cheap
    and this is the boundary that protects one user's agents from
    another's.

    KNOWN GAP (documented rather than invented): the current Agent
    model has no tenant column, no per-agent permission/ACL table, and
    no operational health/heartbeat field. Tenant isolation is
    therefore equivalent to owner isolation today, and "availability"
    is equivalent to status ACTIVE. The hooks are here so Phase 7 can
    add real availability and multi-tenant scoping without reshaping
    callers -- no authorization semantics have been fabricated.
    """

    if user_id is not None and agent.created_by is not None:
        if agent.created_by != user_id:
            return RejectionReason.NOT_OWNED

    if agent.status != AgentStatus.ACTIVE:
        return RejectionReason.INACTIVE

    # Versioning: the existing model tracks current_version on the
    # agent and keeps AgentVersion history rows. There is no
    # per-version enabled/disabled flag, so "executable version" today
    # means "the agent has a sane current_version". This is the clean
    # interface for a future per-version gate.
    if not agent.current_version or agent.current_version < 1:
        return RejectionReason.NO_EXECUTABLE_VERSION

    if requires_rag and not getattr(agent, "is_rag", False):
        return RejectionReason.MISSING_RAG_SUPPORT

    return None


def _to_candidate(
    agent: Agent,
    capability: str,
    match_quality: MatchQuality,
    rejection: Optional[RejectionReason],
) -> AgentCandidate:
    return AgentCandidate(
        agent_id=str(agent.id),
        agent_name=agent.name,
        agent_version=agent.current_version or 1,
        capability=capability,
        match_quality=match_quality,
        is_default=bool(getattr(agent, "is_default", False)),
        is_rag=bool(getattr(agent, "is_rag", False)),
        knowledge_base_id=getattr(agent, "knowledge_base_id", None),
        provider=agent.provider,
        model=agent.model,
        system_prompt=agent.system_prompt,
        eligible=rejection is None,
        rejection_reason=rejection,
    )


def route_capabilities(
    db: Session,
    capabilities: Sequence[str],
    user_id: Optional[int] = None,
    requires_rag: bool = False,
) -> RoutingResult:
    """
    Route every required capability in one pass.

    Discovery is a SINGLE batched query for all capabilities (see
    discovery_service.discover_agents_for_capabilities), then grouping,
    eligibility and selection happen in memory. Routing therefore costs
    one query regardless of how many capabilities a request needs.
    """

    requirements = [
        str(c).strip()
        for c in (capabilities or [])
        if c and str(c).strip()
    ]

    if not requirements:
        return RoutingResult(
            status=RoutingStatus.NO_REQUIREMENT,
            reason="No capabilities were required.",
        )

    # include_inactive=True on purpose: the eligibility layer needs to
    # see unusable providers to tell a capability gap apart from an
    # availability problem.
    agents = discovery_service.discover_agents_for_capabilities(
        db=db,
        capabilities=requirements,
        owner_id=user_id,
        include_inactive=True,
    )

    logger.info(
        "manager.routing_discovered user_id=%s requirements=%s agents=%s",
        user_id,
        len(requirements),
        len(agents),
    )

    routings: List[CapabilityRouting] = []

    for requirement in requirements:

        routing = CapabilityRouting(capability=requirement)

        for agent in agents:

            best: Optional[MatchQuality] = None

            for capability_row in (agent.capabilities or []):

                name = getattr(
                    capability_row,
                    "capability_name",
                    None,
                )

                if not name:
                    continue

                quality = _match_quality(requirement, name)

                if quality is None:
                    continue

                if best is None or quality.rank < best.rank:
                    best = quality

            if best is None:
                continue

            rejection = _evaluate_eligibility(
                agent=agent,
                user_id=user_id,
                requires_rag=requires_rag,
            )

            routing.candidates.append(
                _to_candidate(
                    agent=agent,
                    capability=requirement,
                    match_quality=best,
                    rejection=rejection,
                )
            )

        eligible = [c for c in routing.candidates if c.eligible]

        if eligible:
            # Deterministic: see AgentCandidate.sort_key().
            eligible.sort(key=lambda c: c.sort_key())
            routing.selected = eligible[0]

        elif routing.candidates:
            # Providers exist but none are usable. Record WHY, so this
            # surfaces as AGENT_UNAVAILABLE rather than a capability gap.
            routing.blocked_reason = (
                routing.candidates[0].rejection_reason
            )

        routings.append(routing)

        logger.info(
            "manager.routing_capability user_id=%s capability=%r "
            "candidates=%s eligible=%s selected=%s blocked=%s",
            user_id,
            requirement,
            len(routing.candidates),
            len(eligible),
            routing.selected.agent_name if routing.selected else None,
            routing.blocked_reason.value if routing.blocked_reason else None,
        )

    matched = sum(1 for r in routings if r.matched)

    if matched == len(routings):
        status = RoutingStatus.RESOLVED
    elif matched > 0:
        # PARTIAL is a first-class outcome, not a failure: matched
        # capabilities stay executable and the unmatched ones are
        # reported separately.
        status = RoutingStatus.PARTIAL
    else:
        status = RoutingStatus.UNRESOLVED

    result = RoutingResult(
        status=status,
        requirements=requirements,
        routings=routings,
        reason=(
            f"{matched}/{len(routings)} capabilities routed."
        ),
    )

    logger.info(
        "manager.routing_result user_id=%s status=%s matched=%s "
        "missing=%s unavailable=%s",
        user_id,
        status.value,
        result.matched_capabilities,
        result.missing_capabilities,
        result.unavailable_capabilities,
    )

    return result


def resolve_explicit_agent(
    db: Session,
    agent_name: str,
    user_id: Optional[int] = None,
    requires_rag: bool = False,
) -> CapabilityRouting:
    """
    Resolve an agent the user named directly ("run the calculator
    agent").

    The name still goes through the registry and through the same
    eligibility gate as any other candidate -- naming an agent is a
    request, not an authorization. A name that resolves to nothing in
    the caller's scope yields no candidate, which the caller reports
    as AGENT_NOT_FOUND and never as a reason to create one.
    """

    routing = CapabilityRouting(capability=agent_name)

    agent = discovery_service.discover_agent_by_name(
        db=db,
        name=agent_name,
        owner_id=user_id,
    )

    if agent is None:
        logger.info(
            "manager.explicit_agent_not_found user_id=%s name=%r",
            user_id,
            agent_name,
        )
        return routing

    rejection = _evaluate_eligibility(
        agent=agent,
        user_id=user_id,
        requires_rag=requires_rag,
    )

    candidate = _to_candidate(
        agent=agent,
        capability=agent_name,
        match_quality=MatchQuality.EXPLICIT,
        rejection=rejection,
    )

    routing.candidates.append(candidate)

    if candidate.eligible:
        routing.selected = candidate
    else:
        routing.blocked_reason = rejection

    logger.info(
        "manager.explicit_agent_resolved user_id=%s name=%r eligible=%s "
        "rejection=%s",
        user_id,
        agent_name,
        candidate.eligible,
        rejection.value if rejection else None,
    )

    return routing