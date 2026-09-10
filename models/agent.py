import enum
import uuid

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Enum, Text, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database import Base


class AgentStatus(str, enum.Enum):
    # Lowercase values so the API returns "active"/"inactive" to match
    # the AI layer / frontend contract.
    ACTIVE = "active"
    INACTIVE = "inactive"


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Agent(Base):
    __tablename__ = "agents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(150), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    status = Column(
        Enum(AgentStatus, values_callable=lambda x: [e.value for e in x]),
        default=AgentStatus.ACTIVE,
        nullable=False,
    )
    current_version = Column(Integer, default=1, nullable=False)

    system_prompt = Column(Text, nullable=False)

    # --- Model configuration (Phase 7 - Gemini only for V1) ---
    provider = Column(String(50), default="gemini", nullable=False)
    model = Column(String(100), nullable=False)

    # --- Agent I/O contract (used by the AI layer / Agent Builder) ---
    input_schema = Column(JSON, nullable=True)
    output_schema = Column(JSON, nullable=True)

    # --- RAG / Knowledge Base (Feature 3) ---
    # is_rag: set at creation time when the user chooses "RAG Agent".
    # knowledge_base_id: starts NULL for a RAG agent -- the frontend shows
    # an upload prompt on first Execute (not at creation time) to attach
    # documents, which sets this via PUT /api/agents/{id}.
    # An agent with is_rag=True and knowledge_base_id=None is a "RAG agent
    # waiting for its first document upload".
    is_rag = Column(Boolean, default=False, nullable=False)
    knowledge_base_id = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=True)

    created_by = Column(
    Integer,
    ForeignKey("users.id"),
    nullable=True,
    index=True,
)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    owner = relationship("User", back_populates="agents")

    versions = relationship("AgentVersion", back_populates="agent", cascade="all, delete-orphan")
    capabilities = relationship("AgentCapability", back_populates="agent", cascade="all, delete-orphan")
     
    executions = relationship("AgentExecution", back_populates="agent")


class AgentVersion(Base):
    """Snapshot of an agent's config at a point in time (Phase 2 - version management)."""
    __tablename__ = "agent_versions"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    system_prompt = Column(Text, nullable=False)
    model_config_snapshot = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    agent = relationship("Agent", back_populates="versions")


class AgentCapability(Base):
    """Used for capability-based discovery (Phase 5)."""
    __tablename__ = "agent_capabilities"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False)
    capability_name = Column(String(150), index=True, nullable=False)

    agent = relationship("Agent", back_populates="capabilities")