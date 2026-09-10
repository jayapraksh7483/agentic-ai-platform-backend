import enum
import uuid

from sqlalchemy import Column, String, Text, Integer, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from core.database import Base

# Must match the output dimension of the embedding model used in
# services/embedding_service.py (text-embedding-004 default = 768).
# If you change embedding models to one with a different dimension,
# this constant AND every existing row's embedding must be regenerated --
# you cannot mix vector dimensions in one column.
EMBEDDING_DIM = 768


def generate_uuid() -> str:
    return str(uuid.uuid4())


class KnowledgeBaseStatus(str, enum.Enum):
    ACTIVE = "active"
    PROCESSING = "processing"


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(150), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    document_count = Column(Integer, default=0, nullable=False)
    status = Column(
        Enum(KnowledgeBaseStatus, values_callable=lambda x: [e.value for e in x]),
        default=KnowledgeBaseStatus.ACTIVE,
        nullable=False,
    )
    created_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    documents = relationship("KnowledgeDocument", back_populates="knowledge_base", cascade="all, delete-orphan")


class DocumentStatus(str, enum.Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    kb_id = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    filename = Column(String(300), nullable=False)
    status = Column(
        Enum(DocumentStatus, values_callable=lambda x: [e.value for e in x]),
        default=DocumentStatus.PROCESSING,
        nullable=False,
    )
    chunk_count = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship("KnowledgeChunk", back_populates="document", cascade="all, delete-orphan")


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    doc_id = Column(String(36), ForeignKey("knowledge_documents.id"), nullable=False, index=True)
    # Denormalized for fast filtering during search (avoid a join on every query)
    kb_id = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)

    document = relationship("KnowledgeDocument", back_populates="chunks")