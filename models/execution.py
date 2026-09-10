import enum
import uuid

from sqlalchemy import Column, String, DateTime, Enum, Text, ForeignKey, Integer, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database import Base


class ExecutionStatus(str, enum.Enum):
    # Lowercase values so the API returns "success"/"failed" etc.
    # to match the AI layer / frontend contract.
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


def generate_task_id() -> str:
    return str(uuid.uuid4())


class AgentExecution(Base):
    __tablename__ = "agent_executions"

    id = Column(String(36), primary_key=True, default=generate_task_id)
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False)
    user_id = Column(Integer, nullable=True)

    input_payload = Column(Text, nullable=False)
    output_payload = Column(Text, nullable=True)

    status = Column(
        Enum(ExecutionStatus, values_callable=lambda x: [e.value for e in x]),
        default=ExecutionStatus.PENDING,
        nullable=False,
    )

    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)
    latency_ms = Column(Float, nullable=True)

    error_message = Column(Text, nullable=True)

    agent = relationship("Agent", back_populates="executions")