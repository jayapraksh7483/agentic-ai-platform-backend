
import enum
import uuid

from sqlalchemy import Column, String, Text, Integer, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from core.database import Base


# Must match the output dimension of the embedding model used in
# services/embedding_service.py.
#
# Current embedding model:
#   gemini-embedding-001
#
# Current configured output dimension:
#   768
#
# If the embedding model or dimension changes, existing embeddings
# must be regenerated. Vector dimensions cannot be mixed in the
# same database column.
EMBEDDING_DIM = 768


def generate_uuid() -> str:
    return str(uuid.uuid4())


class KnowledgeBaseStatus(str, enum.Enum):
    ACTIVE = "active"
    PROCESSING = "processing"


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String(36), primary_key=True, default=generate_uuid)

    name = Column(
        String(150),
        unique=True,
        index=True,
        nullable=False,
    )

    description = Column(Text, nullable=True)

    document_count = Column(
        Integer,
        default=0,
        nullable=False,
    )

    status = Column(
        Enum(
            KnowledgeBaseStatus,
            values_callable=lambda x: [e.value for e in x],
        ),
        default=KnowledgeBaseStatus.ACTIVE,
        nullable=False,
    )

    # User ownership.
    #
    # Keep nullable=True temporarily because existing knowledge bases
    # may contain NULL values. We will migrate existing records and
    # enforce NOT NULL at the database level after the data migration.
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

    # Relationships
    owner = relationship(
        "User",
        back_populates="knowledge_bases",
    )

    documents = relationship(
        "KnowledgeDocument",
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
    )

    # Requirement 2 -- conversations that have explicitly attached this
    # Knowledge Base (Option B). This is a reference-only join table
    # (models/conversation_knowledge_base.py); it does not cascade
    # deletes onto documents/chunks and a KB can be attached to many
    # conversations at once.
    conversation_links = relationship(
        "ConversationKnowledgeBase",
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
    )


class DocumentStatus(str, enum.Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id = Column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    kb_id = Column(
        String(36),
        ForeignKey("knowledge_bases.id"),
        nullable=False,
        index=True,
    )

    filename = Column(
        String(300),
        nullable=False,
    )

    status = Column(
        Enum(
            DocumentStatus,
            values_callable=lambda x: [e.value for e in x],
        ),
        default=DocumentStatus.PROCESSING,
        nullable=False,
    )

    chunk_count = Column(
        Integer,
        default=0,
        nullable=False,
    )

    error_message = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    knowledge_base = relationship(
        "KnowledgeBase",
        back_populates="documents",
    )

    chunks = relationship(
        "KnowledgeChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    doc_id = Column(
        String(36),
        ForeignKey("knowledge_documents.id"),
        nullable=False,
        index=True,
    )

    # Denormalized KB ID for efficient vector filtering.
    kb_id = Column(
        String(36),
        ForeignKey("knowledge_bases.id"),
        nullable=False,
        index=True,
    )

    text = Column(
        Text,
        nullable=False,
    )

    chunk_index = Column(
        Integer,
        nullable=False,
    )

    # pgvector embedding.
    #
    # Current dimension = 768.
    embedding = Column(
        Vector(EMBEDDING_DIM),
        nullable=True,
    )

    document = relationship(
        "KnowledgeDocument",
        back_populates="chunks",
    )

