from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import relationship

from core.database import Base


class ConversationAttachment(Base):
    __tablename__ = "conversation_attachments"

    id = Column(Integer, primary_key=True, index=True)

    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Original uploaded filename
    filename = Column(
        String(255),
        nullable=False,
    )

    # MIME type received from frontend
    content_type = Column(
        String(150),
        nullable=True,
    )

    # File size in bytes
    file_size = Column(
        Integer,
        nullable=True,
    )

    # Existing KnowledgeDocument created by knowledge_service.process_upload()
    knowledge_document_id = Column(
        String(36),
        ForeignKey("knowledge_documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Existing Knowledge Base used for this attachment
    knowledge_base_id = Column(
        String(36),
        ForeignKey("knowledge_bases.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    status = Column(
        String(30),
        nullable=False,
        default="processing",
    )

    error_message = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    conversation = relationship(
        "Conversation",
        backref="attachments",
    )

    user = relationship(
        "User",
        backref="conversation_attachments",
    )

    knowledge_document = relationship(
        "KnowledgeDocument",
        foreign_keys=[knowledge_document_id],
    )

    knowledge_base = relationship(
        "KnowledgeBase",
        foreign_keys=[knowledge_base_id],
    )