import enum
import uuid

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Enum,
    Float,
    Text,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database import Base


class AgentStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Agent(Base):
    __tablename__ = "agents"

    id = Column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    name = Column(
        String(150),
        index=True,
        nullable=False,
    )

    description = Column(
        Text,
        nullable=True,
    )

    status = Column(
        Enum(
            AgentStatus,
            values_callable=lambda x: [e.value for e in x],
        ),
        default=AgentStatus.ACTIVE,
        nullable=False,
    )

    current_version = Column(
        Integer,
        default=1,
        nullable=False,
    )

    system_prompt = Column(
        Text,
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------

    provider = Column(
        String(50),
        default="gemini",
        nullable=False,
    )

    model = Column(
        String(100),
        nullable=False,
    )

    # Per-agent LLM sampling temperature. NULL means "use the provider
    # client's own default" -- existing agents (created before this
    # field existed) keep behaving exactly as before.
    temperature = Column(
        Float,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Custom agent credentials
    #
    # NULL (both columns) means "no agent-specific key configured" --
    # execution falls back to the platform-wide key for `provider`
    # (core/config.py / .env), exactly like every agent before this
    # feature existed. This is what keeps the Manager Agent and the
    # four platform default agents completely unaffected: they are
    # never given an api_key, so they always take this fallback path.
    #
    # The raw key is NEVER stored. api_key_encrypted holds only a
    # Fernet-encrypted ciphertext (core/encryption.py, reusing the
    # same TOKEN_ENCRYPTION_KEY already used for Google OAuth tokens).
    # api_key_last4 is stored purely so the API can show a masked
    # "••••••••abcd" preview without ever decrypting the key again
    # outside of actual execution.
    # ------------------------------------------------------------------

    api_key_encrypted = Column(
        Text,
        nullable=True,
    )

    api_key_last4 = Column(
        String(4),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Tool access
    #
    # NULL means "unrestricted" -- every registered platform tool is
    # made available to this agent, which is the existing behavior for
    # every agent created before this field existed (see
    # agent_runtime/nodes.py::_tool_specs). A non-null list restricts
    # the agent to exactly those registered tool names (validated at
    # write time in services/agent_service.py against tools.tool_registry).
    # An explicit empty list ([]) means "no tools" -- deliberately
    # distinct from NULL.
    # ------------------------------------------------------------------

    tools = Column(
        JSON,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Agent I/O contract
    # ------------------------------------------------------------------

    input_schema = Column(
        JSON,
        nullable=True,
    )

    output_schema = Column(
        JSON,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # RAG / Knowledge Base
    # ------------------------------------------------------------------

    is_rag = Column(
        Boolean,
        default=False,
        nullable=False,
    )

    knowledge_base_id = Column(
        String(36),
        ForeignKey("knowledge_bases.id"),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Runtime / visibility policy for user-created agents
    # ------------------------------------------------------------------

    visibility = Column(
        String(20),
        default="private",
        nullable=False,
    )

    timeout_seconds = Column(
        Integer,
        default=30,
        nullable=False,
    )

    max_retries = Column(
        Integer,
        default=2,
        nullable=False,
    )

    # Reserved for workflows that require an execution-time approval
    # gate. Agent creation itself is always explicitly approved by the
    # Manager lifecycle workflow.
    requires_approval = Column(
        Boolean,
        default=False,
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Ownership / Default Agent
    # ------------------------------------------------------------------

    # True  -> Platform-provided default agent
    # False -> User-created agent
    is_default = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    created_by = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    owner = relationship(
        "User",
        back_populates="agents",
    )

    # ------------------------------------------------------------------
    # Agent registry relationships
    # ------------------------------------------------------------------

    versions = relationship(
        "AgentVersion",
        back_populates="agent",
        cascade="all, delete-orphan",
    )

    capabilities = relationship(
        "AgentCapability",
        back_populates="agent",
        cascade="all, delete-orphan",
    )

    executions = relationship(
        "AgentExecution",
        back_populates="agent",
    )


class AgentVersion(Base):
    """
    Snapshot of an agent's configuration at a point in time.
    """

    __tablename__ = "agent_versions"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    agent_id = Column(
        String(36),
        ForeignKey("agents.id"),
        nullable=False,
    )

    version_number = Column(
        Integer,
        nullable=False,
    )

    system_prompt = Column(
        Text,
        nullable=False,
    )

    model_config_snapshot = Column(
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    agent = relationship(
        "Agent",
        back_populates="versions",
    )


class AgentCapability(Base):
    """
    Capability records used for dynamic agent discovery.
    """

    __tablename__ = "agent_capabilities"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    agent_id = Column(
        String(36),
        ForeignKey("agents.id"),
        nullable=False,
    )

    capability_name = Column(
        String(150),
        index=True,
        nullable=False,
    )

    agent = relationship(
        "Agent",
        back_populates="capabilities",
    )