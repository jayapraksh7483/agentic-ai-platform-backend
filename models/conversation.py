import uuid

from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database import Base


def generate_conversation_uuid() -> str:
    return str(uuid.uuid4())


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(
        String(36),
        primary_key=True,
        default=generate_conversation_uuid,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    title = Column(
        String(255),
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ---------------------------------------------------------
    # User relationship
    # ---------------------------------------------------------
    user = relationship(
        "User",
        back_populates="conversations",
    )

    # ---------------------------------------------------------
    # Messages belonging to this conversation
    # ---------------------------------------------------------
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    # ---------------------------------------------------------
    # Orchestration executions belonging to this conversation
    # ---------------------------------------------------------
    orchestration_executions = relationship(
        "OrchestrationExecution",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )