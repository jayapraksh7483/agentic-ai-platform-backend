"""
Phase 5C -- Persistent Messages.

Messages belong to a conversation.

Supported roles:
    user
    assistant
    system
    tool
"""

import enum
import uuid

from sqlalchemy import (
    Column,
    String,
    DateTime,
    ForeignKey,
    Text,
    Enum,
    JSON,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database import Base


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


def generate_message_uuid() -> str:
    return str(uuid.uuid4())


class Message(Base):
    __tablename__ = "messages"

    id = Column(
        String(36),
        primary_key=True,
        default=generate_message_uuid,
    )

    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    role = Column(
        Enum(
            MessageRole,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )

    content = Column(
        Text,
        nullable=False,
    )

    message_metadata = Column(
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    conversation = relationship(
        "Conversation",
        back_populates="messages",
    )