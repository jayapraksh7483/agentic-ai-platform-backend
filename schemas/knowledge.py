
from typing import Optional, List

from pydantic import BaseModel, ConfigDict

from models.knowledge import (
    KnowledgeBaseStatus,
    DocumentStatus,
)


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: Optional[str] = None


class KnowledgeBaseOut(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    document_count: int
    status: KnowledgeBaseStatus

    model_config = ConfigDict(
        from_attributes=True,
    )


class KnowledgeDocumentOut(BaseModel):
    id: str
    kb_id: str
    filename: str
    status: DocumentStatus
    chunk_count: int
    error_message: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True,
    )


class KnowledgeSearchResultItem(BaseModel):
    """
    Search result returned by the RAG retrieval layer.

    text:
        Retrieved chunk content.

    score:
        Similarity score. Higher means more relevant.

    document_id:
        ID of the actual uploaded document.

    filename:
        Original uploaded filename. Used for source/citation display.

    chunk_index:
        Position of the retrieved chunk inside the document.
    """

    text: str
    score: float

    document_id: Optional[str] = None
    filename: Optional[str] = None
    chunk_index: Optional[int] = None



class KnowledgeBaseReadinessOut(BaseModel):
    exists: bool
    ready: bool
    state: str
    ready_documents: int = 0
    processing_documents: int = 0
    failed_documents: int = 0
