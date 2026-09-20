from typing import List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from models.agent import (
    Agent,
    AgentVersion,
    AgentCapability,
    AgentStatus,
)

from models.orchestration import OrchestrationStep
from models.execution import AgentExecution

from schemas.agent import AgentCreate, AgentUpdate

from core.encryption import encrypt_token, decrypt_token

from services.llm_service import PROVIDER_REGISTRY


# A caller sends this exact sentinel as model.api_key on an UPDATE to
# explicitly remove a previously-configured key. Distinguishing "not
# provided" (None -- leave the existing key alone) from "clear it" is
# not possible with a single Optional[str] field, so an empty string
# is reserved for "clear it". A real API key is never an empty string.
CLEAR_API_KEY_SENTINEL = ""


def _agent_to_dict_extras(agent: Agent) -> dict:
    """
    Helper to expose capabilities as a plain list.
    """

    return {
        "capabilities": [
            c.capability_name
            for c in agent.capabilities
        ],
    }


def _mask_api_key(last4: Optional[str]) -> Optional[str]:
    """
    Build a display-only masked preview, e.g. "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022abcd".

    Never touches the encrypted/raw key -- built purely from the
    last4 characters stored alongside it at write time.
    """

    if not last4:
        return None

    return "\u2022" * 8 + last4


def get_decrypted_api_key(agent: Agent) -> Optional[str]:
    """
    Decrypt and return this agent's own stored API key, or None if it
    has no agent-specific key configured.

    Called only at execution time (services/executor_service.py,
    manager/planner.py). The decrypted value must never be logged,
    persisted anywhere else, or included in any API response.
    """

    if not agent.api_key_encrypted:
        return None

    return decrypt_token(agent.api_key_encrypted)


def _validate_provider(provider: str) -> None:
    if (provider or "").lower() not in PROVIDER_REGISTRY:
        supported = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            f"Supported providers: {supported}"
        )


def _validate_tools(tools: Optional[List[str]]) -> None:
    if tools is None:
        return

    from tools import tool_registry

    unknown = [
        name
        for name in tools
        if not tool_registry.exists(name)
    ]

    if unknown:
        raise ValueError(
            f"Unknown tool(s): {', '.join(unknown)}"
        )


def validate_agent_provider(provider: str) -> None:
    """Public lifecycle-validation wrapper."""
    _validate_provider(provider)


def validate_agent_tools(tools: Optional[List[str]]) -> None:
    """Public lifecycle-validation wrapper."""
    _validate_tools(tools)


def create_agent(
    db: Session,
    agent_in: AgentCreate,
    created_by: Optional[int] = None,
    is_default: bool = False,
    commit: bool = True,
) -> Agent:
    """
    Create an agent.

    created_by:
        Owner of the agent.

    is_default:
        Internal flag used by platform provisioning.

        False -> user-created agent
        True  -> platform default agent
    """

    _validate_provider(agent_in.model_cfg.provider)
    _validate_tools(agent_in.tools)

    # Agent names are unique within the caller's registry scope even though
    # the legacy database column itself is not globally unique. Enforce the
    # product rule explicitly instead of relying on an IntegrityError that
    # PostgreSQL cannot raise for a non-unique column.
    duplicate = (
        db.query(Agent)
        .filter(
            Agent.created_by == created_by,
            Agent.name == agent_in.name.strip(),
        )
        .first()
    )
    if duplicate:
        raise ValueError(
            f"An agent named '{agent_in.name}' already exists for this user."
        )

    if agent_in.is_rag:
        from services import knowledge_service

        if not agent_in.knowledge_base_id:
            raise ValueError(
                "A user-created RAG agent requires a knowledge base."
            )

        kb = knowledge_service.get_knowledge_base(
            db,
            agent_in.knowledge_base_id,
            user_id=created_by,
        )

        if not kb:
            raise ValueError(
                f"Knowledge base '{agent_in.knowledge_base_id}' not found"
            )

        if not knowledge_service.is_knowledge_base_ready(
            db, kb.id, user_id=created_by
        ):
            raise ValueError(
                "The selected knowledge base is not ready for RAG execution."
            )
    elif agent_in.knowledge_base_id:
        raise ValueError(
            "knowledge_base_id can only be set when is_rag=true."
        )

    raw_api_key = agent_in.model_cfg.api_key
    api_key_encrypted = None
    api_key_last4 = None

    if raw_api_key:
        api_key_encrypted = encrypt_token(raw_api_key)
        api_key_last4 = raw_api_key[-4:]

    agent = Agent(
        name=agent_in.name,
        description=agent_in.description,
        system_prompt=agent_in.system_prompt,
        provider=agent_in.model_cfg.provider,
        model=agent_in.model_cfg.model,
        temperature=agent_in.model_cfg.temperature,
        api_key_encrypted=api_key_encrypted,
        api_key_last4=api_key_last4,
        tools=agent_in.tools,
        input_schema=agent_in.input_schema,
        output_schema=agent_in.output_schema,
        knowledge_base_id=agent_in.knowledge_base_id,
        is_rag=agent_in.is_rag,
        visibility=agent_in.visibility,
        timeout_seconds=agent_in.timeout_seconds,
        max_retries=agent_in.max_retries,
        requires_approval=agent_in.requires_approval,
        status=AgentStatus.ACTIVE,
        current_version=1,
        created_by=created_by,
        is_default=is_default,
    )

    db.add(agent)

    try:
        db.flush()

    except IntegrityError:
        db.rollback()

        raise ValueError(
            f"An agent named '{agent_in.name}' "
            f"already exists for this user."
        )

    for cap in agent_in.capabilities:
        db.add(
            AgentCapability(
                agent_id=agent.id,
                capability_name=cap,
            )
        )

    db.add(
        AgentVersion(
            agent_id=agent.id,
            version_number=1,
            system_prompt=agent.system_prompt,
            model_config_snapshot={
                "provider": agent.provider,
                "model": agent.model,
                "temperature": agent.temperature,
                # Never snapshot the raw or encrypted key itself --
                # only whether one was configured at this version.
                "api_key_configured": bool(
                    agent.api_key_encrypted
                ),
                "tools": agent.tools,
                "visibility": agent.visibility,
                "timeout_seconds": agent.timeout_seconds,
                "max_retries": agent.max_retries,
                "requires_approval": agent.requires_approval,
            },
        )
    )

    if commit:
        db.commit()
    else:
        db.flush()
    db.refresh(agent)

    return agent


def list_agents(
    db: Session,
    status_filter: Optional[AgentStatus] = None,
    owner_id: Optional[int] = None,
) -> List[Agent]:
    """
    List agents.

    User-facing calls pass owner_id.

    Therefore users see only their own agents.

    Default agents are provisioned for each user and therefore
    are also included naturally.
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

    When owner_id is supplied, the agent must belong
    to that user.
    """

    query = (
        db.query(Agent)
        .options(joinedload(Agent.capabilities))
        .filter(Agent.id == agent_id)
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

    agent = get_agent(
        db,
        agent_id,
        owner_id=owner_id,
    )

    if not agent:
        return None

    if agent.is_default:
        raise ValueError("Platform default agents cannot be modified.")

    bumped_version = False

    if agent_in.name is not None:
        new_name = agent_in.name.strip()
        if not new_name:
            raise ValueError("Agent name cannot be blank.")
        if new_name != agent.name:
            duplicate = (
                db.query(Agent)
                .filter(
                    Agent.created_by == owner_id,
                    Agent.name == new_name,
                    Agent.id != agent.id,
                )
                .first()
            )
            if duplicate:
                raise ValueError(f"An agent named '{new_name}' already exists for this user.")
            agent.name = new_name

    if agent_in.description is not None:
        agent.description = agent_in.description

    if (
        agent_in.system_prompt is not None
        and agent_in.system_prompt != agent.system_prompt
    ):
        agent.system_prompt = agent_in.system_prompt
        bumped_version = True

    if agent_in.model_cfg is not None:

        _validate_provider(agent_in.model_cfg.provider)

        if (
            agent.provider
            != agent_in.model_cfg.provider
            or agent.model
            != agent_in.model_cfg.model
            or agent.temperature
            != agent_in.model_cfg.temperature
        ):
            bumped_version = True

        agent.provider = agent_in.model_cfg.provider
        agent.model = agent_in.model_cfg.model
        agent.temperature = agent_in.model_cfg.temperature

        raw_api_key = agent_in.model_cfg.api_key

        if raw_api_key == CLEAR_API_KEY_SENTINEL:
            # Explicit clear -- see CLEAR_API_KEY_SENTINEL docstring.
            if agent.api_key_encrypted is not None:
                bumped_version = True

            agent.api_key_encrypted = None
            agent.api_key_last4 = None

        elif raw_api_key:
            # A new/changed key was supplied. We can't compare it to
            # the existing one without decrypting on every update just
            # to diff strings, so any supplied non-empty key is
            # treated as a change and always bumps the version --
            # matching the spec's explicit requirement that changing
            # the API key bumps the configuration/version.
            agent.api_key_encrypted = encrypt_token(raw_api_key)
            agent.api_key_last4 = raw_api_key[-4:]
            bumped_version = True

        # raw_api_key is None (field omitted entirely): leave the
        # agent's existing stored key untouched.

    if agent_in.tools is not None:

        _validate_tools(agent_in.tools)

        if agent.tools != agent_in.tools:
            bumped_version = True

        agent.tools = agent_in.tools

    if agent_in.input_schema is not None:
        agent.input_schema = agent_in.input_schema

    if agent_in.output_schema is not None:
        agent.output_schema = agent_in.output_schema

    if agent_in.is_rag is not None:
        if agent.is_rag != agent_in.is_rag:
            bumped_version = True
        agent.is_rag = agent_in.is_rag

    if agent_in.knowledge_base_id is not None:

        from services import knowledge_service

        if agent_in.knowledge_base_id == "":
            if agent.knowledge_base_id is not None:
                bumped_version = True
            agent.knowledge_base_id = None
        else:
            kb = knowledge_service.get_knowledge_base(
                db,
                agent_in.knowledge_base_id,
                user_id=owner_id,
            )

            if not kb:
                raise ValueError(
                    f"Knowledge base '{agent_in.knowledge_base_id}' not found"
                )

            if not knowledge_service.is_knowledge_base_ready(
                db, kb.id, user_id=owner_id
            ):
                raise ValueError(
                    "The selected knowledge base is not ready for RAG execution."
                )

            if agent.knowledge_base_id != agent_in.knowledge_base_id:
                bumped_version = True
            agent.knowledge_base_id = agent_in.knowledge_base_id

    if agent.is_rag and not agent.knowledge_base_id:
        raise ValueError("A RAG agent requires a ready knowledge base.")

    if not agent.is_rag and agent.knowledge_base_id is not None:
        agent.knowledge_base_id = None
        bumped_version = True

    for attr in ("visibility", "timeout_seconds", "max_retries", "requires_approval"):
        value = getattr(agent_in, attr)
        if value is not None and getattr(agent, attr) != value:
            setattr(agent, attr, value)
            bumped_version = True

    if agent_in.capabilities is not None:

        db.query(
            AgentCapability
        ).filter(
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
                    "temperature": agent.temperature,
                    "api_key_configured": bool(
                        agent.api_key_encrypted
                    ),
                    "tools": agent.tools,
                    "visibility": agent.visibility,
                    "timeout_seconds": agent.timeout_seconds,
                    "max_retries": agent.max_retries,
                    "requires_approval": agent.requires_approval,
                },
            )
        )

    try:
        db.commit()

    except IntegrityError:
        db.rollback()

        raise ValueError(
            "Could not save changes -- check for "
            "a duplicate value."
        )

    db.refresh(agent)

    return agent


def delete_agent(
    db: Session,
    agent_id: str,
    owner_id: Optional[int] = None,
) -> bool:

    agent = get_agent(
        db,
        agent_id,
        owner_id=owner_id,
    )

    if not agent:
        return False

    if agent.is_default:
        raise ValueError("Platform default agents cannot be deleted.")

    db.query(
        AgentExecution
    ).filter(
        AgentExecution.agent_id == agent_id
    ).delete(
        synchronize_session=False
    )

    db.query(
        OrchestrationStep
    ).filter(
        OrchestrationStep.agent_id == agent_id
    ).delete(
        synchronize_session=False
    )

    db.delete(agent)

    try:
        db.commit()

    except IntegrityError:
        db.rollback()

        raise ValueError(
            "Could not delete agent because another "
            "record still references it."
        )

    return True


def set_agent_status(
    db: Session,
    agent_id: str,
    status: AgentStatus,
    owner_id: Optional[int] = None,
) -> Optional[Agent]:

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
