"""
Manager Agent / orchestration persistence models (Phase 4).

These are intentionally separate from the Agent Registry tables
(models/agent.py) -- an orchestration execution coordinates *calls to*
one or more registered agents, it does not own or duplicate the agent
definitions themselves.
"""
import enum
import uuid

from sqlalchemy import (
    Column, Integer, String, DateTime, Enum, Text, ForeignKey, JSON, Float
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class OrchestrationStatus(str, enum.Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # some steps succeeded, some failed
    PENDING_AGENT_APPROVAL = "pending_agent_approval"


class StepStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  # e.g. a conditional step whose condition wasn't met


class OrchestrationExecution(Base):
    """One row per POST /api/orchestrate request."""

    __tablename__ = "orchestration_executions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(Integer, nullable=True)

    user_input = Column(Text, nullable=False)

    # Capabilities the Manager identified for this request (informational /
    # observability -- Phase 8 can use this to analyze capability demand).
    required_capabilities = Column(JSON, nullable=True)

    # The provider/model the ORIGINAL request used. Persisted so that if
    # this execution ends up PENDING_AGENT_APPROVAL, a later call to
    # approve_pending_agent() can re-run the same user_input with the same
    # provider/model instead of silently falling back to a default.
    provider = Column(String(50), nullable=True)
    model = Column(String(100), nullable=True)

    # Set only while status == PENDING_AGENT_APPROVAL. Holds the drafted
    # ProposedAgentSpec (see manager/schemas.py) so approve_pending_agent()
    # can register the *actual* proposed agent instead of a blank one.
    # Cleared back to null once the proposal is approved/declined.
    proposed_agent = Column(JSON, nullable=True)

    status = Column(
        Enum(OrchestrationStatus, values_callable=lambda x: [e.value for e in x]),
        default=OrchestrationStatus.RUNNING,
        nullable=False,
    )

    final_response = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)
    latency_ms = Column(Float, nullable=True)

    steps = relationship(
        "OrchestrationStep",
        back_populates="execution",
        cascade="all, delete-orphan",
        order_by="OrchestrationStep.id",
    )


class OrchestrationStep(Base):
    """One row per planned step within an orchestration execution."""

    __tablename__ = "orchestration_steps"

    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(
        String(36), ForeignKey("orchestration_executions.id"), nullable=False, index=True
    )

    # Logical id within the plan, e.g. "step_1" -- referenced by depends_on.
    step_key = Column(String(50), nullable=False)

    # Nullable: a step can exist in a *proposed* plan for a capability that
    # had no matching registered agent (surfaced as a failed/skipped step
    # rather than silently dropped).
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=True)
    capability = Column(String(150), nullable=True)

    task = Column(Text, nullable=False)
    depends_on = Column(JSON, nullable=True)  # list[str] of step_keys

    status = Column(
        Enum(StepStatus, values_callable=lambda x: [e.value for e in x]),
        default=StepStatus.PENDING,
        nullable=False,
    )

    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)

    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    execution = relationship("OrchestrationExecution", back_populates="steps")