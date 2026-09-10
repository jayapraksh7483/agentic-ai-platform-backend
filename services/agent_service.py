from typing import List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from models.agent import Agent, AgentVersion, AgentCapability, AgentStatus
from models.orchestration import OrchestrationStep
from models.execution import AgentExecution

from schemas.agent import AgentCreate, AgentUpdate


def _agent_to_dict_extras(agent: Agent) -> dict:
    """Helper to expose capabilities as a plain list for API responses."""
    return {
        "capabilities": [c.capability_name for c in agent.capabilities],
    }


def create_agent(
    db: Session,
    agent_in: AgentCreate,
    created_by: Optional[int] = None,
) -> Agent:

    if agent_in.knowledge_base_id is not None:
        from services import knowledge_service

        kb = knowledge_service.get_knowledge_base(
            db,
            agent_in.knowledge_base_id,
        )

        if not kb:
            raise ValueError(
                f"Knowledge base '{agent_in.knowledge_base_id}' not found"
            )

    agent = Agent(
        name=agent_in.name,
        description=agent_in.description,
        system_prompt=agent_in.system_prompt,
        provider=agent_in.model_cfg.provider,
        model=agent_in.model_cfg.model,
        input_schema=agent_in.input_schema,
        output_schema=agent_in.output_schema,
        knowledge_base_id=agent_in.knowledge_base_id,
        status=AgentStatus.ACTIVE,
        current_version=1,
        created_by=created_by,
    )

    db.add(agent)

    try:
        # Get agent.id before adding child records.
        db.flush()

    except IntegrityError:
        # Most likely cause: agents.name has a unique constraint.
        db.rollback()

        raise ValueError(
            f"An agent named '{agent_in.name}' already exists. "
            "Please rename it (e.g. add a version number or edit the "
            "generated name) before registering."
        )

    for cap in agent_in.capabilities:
        db.add(
            AgentCapability(
                agent_id=agent.id,
                capability_name=cap,
            )
        )

    # Initial version snapshot
    db.add(
        AgentVersion(
            agent_id=agent.id,
            version_number=1,
            system_prompt=agent.system_prompt,
            model_config_snapshot={
                "provider": agent.provider,
                "model": agent.model,
            },
        )
    )

    db.commit()
    db.refresh(agent)

    return agent


def list_agents(
    db: Session,
    status_filter: Optional[AgentStatus] = None,
    owner_id: Optional[int] = None,
) -> List[Agent]:
    """
    List agents.

    When owner_id is supplied, only agents belonging to that
    user are returned.

    owner_id=None is intentionally supported for internal platform
    operations such as registry/manager discovery. User-facing API
    endpoints must always pass the authenticated user's ID.
    """

    query = db.query(Agent).options(
        joinedload(Agent.capabilities)
    )

    if owner_id is not None:
        query = query.filter(
            Agent.created_by == owner_id
        )

    if status_filter:
        query = query.filter(
            Agent.status == status_filter
        )

    return query.order_by(
        Agent.created_at.desc()
    ).all()


def get_agent(
    db: Session,
    agent_id: str,
    owner_id: Optional[int] = None,
) -> Optional[Agent]:
    """
    Get an agent by ID.

    When owner_id is supplied, the agent must belong to that user.
    """

    query = db.query(Agent).options(
        joinedload(Agent.capabilities)
    ).filter(
        Agent.id == agent_id
    )

    if owner_id is not None:
        query = query.filter(
            Agent.created_by == owner_id
        )

    return query.first()


def update_agent(
    db: Session,
    agent_id: str,
    agent_in: AgentUpdate,
    owner_id: Optional[int] = None,
) -> Optional[Agent]:
    """
    Update an agent.

    When owner_id is supplied, only the owner's agent can be updated.
    """

    agent = get_agent(
        db,
        agent_id,
        owner_id=owner_id,
    )

    if not agent:
        return None

    bumped_version = False

    if agent_in.description is not None:
        agent.description = agent_in.description

    if (
        agent_in.system_prompt is not None
        and agent_in.system_prompt != agent.system_prompt
    ):
        agent.system_prompt = agent_in.system_prompt
        bumped_version = True

    if agent_in.model_cfg is not None:
        agent.provider = agent_in.model_cfg.provider
        agent.model = agent_in.model_cfg.model
        bumped_version = True

    if agent_in.input_schema is not None:
        agent.input_schema = agent_in.input_schema

    if agent_in.output_schema is not None:
        agent.output_schema = agent_in.output_schema

    if agent_in.knowledge_base_id is not None:
        from services import knowledge_service

        kb = knowledge_service.get_knowledge_base(
            db,
            agent_in.knowledge_base_id,
        )

        if not kb:
            raise ValueError(
                f"Knowledge base '{agent_in.knowledge_base_id}' not found"
            )

        agent.knowledge_base_id = agent_in.knowledge_base_id

    if agent_in.capabilities is not None:
        db.query(AgentCapability).filter(
            AgentCapability.agent_id == agent.id
        ).delete(
            synchronize_session=False
        )

        for cap in agent_in.capabilities:
            db.add(
                AgentCapability(
                    agent_id=agent.id,
                    capability_name=cap,
                )
            )

    if agent_in.status is not None:
        # Allows enabling/disabling through the general edit form
        # as well as the dedicated status endpoint.
        agent.status = agent_in.status

    if bumped_version:
        agent.current_version += 1

        db.add(
            AgentVersion(
                agent_id=agent.id,
                version_number=agent.current_version,
                system_prompt=agent.system_prompt,
                model_config_snapshot={
                    "provider": agent.provider,
                    "model": agent.model,
                },
            )
        )

    try:
        db.commit()

    except IntegrityError:
        db.rollback()

        raise ValueError(
            "Could not save changes -- check for a duplicate value "
            "(e.g. name)."
        )

    db.refresh(agent)

    return agent


def delete_agent(
    db: Session,
    agent_id: str,
    owner_id: Optional[int] = None,
) -> bool:
    """
    Hard-delete an agent.

    When owner_id is supplied, only the owner's agent can be deleted.

    Removes dependent execution/orchestration records first because
    they reference agents through foreign keys.
    """

    agent = get_agent(
        db,
        agent_id,
        owner_id=owner_id,
    )

    if not agent:
        return False

    # Remove agent execution history
    db.query(AgentExecution).filter(
        AgentExecution.agent_id == agent_id
    ).delete(
        synchronize_session=False
    )

    # Remove orchestration steps referencing this agent
    db.query(OrchestrationStep).filter(
        OrchestrationStep.agent_id == agent_id
    ).delete(
        synchronize_session=False
    )

    # Now delete the agent
    db.delete(agent)

    try:
        db.commit()

    except IntegrityError:
        db.rollback()

        raise ValueError(
            "Could not delete agent because another record still "
            "references it."
        )

    return True


def set_agent_status(
    db: Session,
    agent_id: str,
    status: AgentStatus,
    owner_id: Optional[int] = None,
) -> Optional[Agent]:
    """
    Change agent status.

    When owner_id is supplied, only the owner's agent can be changed.
    """

    agent = get_agent(
        db,
        agent_id,
        owner_id=owner_id,
    )

    if not agent:
        return None

    agent.status = status

    db.commit()
    db.refresh(agent)

    return agent