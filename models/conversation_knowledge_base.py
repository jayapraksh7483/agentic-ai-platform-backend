"""
Conversation ↔ Knowledge Base association model.

Allows users to attach existing knowledge bases to conversations
without copying or modifying the underlying documents/chunks.
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from core.database import Base


class ConversationKnowledgeBase(Base):
    __tablename__ = "conversation_knowledge_bases"

    id = Column(String, primary_key=True)

    conversation_id = Column(
        String,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )

    knowledge_base_id = Column(
        String,
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "knowledge_base_id",
            name="uq_conversation_knowledge_base",
        ),
    )

    conversation = relationship(
        "Conversation",
        back_populates="knowledge_base_links",
    )

    knowledge_base = relationship(
        "KnowledgeBase",
        back_populates="conversation_links",
    )

    user = relationship("User")