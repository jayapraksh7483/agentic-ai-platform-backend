from typing import Optional, List

from pydantic import BaseModel, ConfigDict

from models.knowledge import KnowledgeBaseStatus, DocumentStatus


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: Optional[str] = None


class KnowledgeBaseOut(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    document_count: int
    status: KnowledgeBaseStatus

    model_config = ConfigDict(from_attributes=True)


class KnowledgeDocumentOut(BaseModel):
    id: str
    kb_id: str
    filename: str
    status: DocumentStatus
    chunk_count: int
    error_message: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class KnowledgeSearchResultItem(BaseModel):
    """Matches Sumith's exact spec: [{ "text": "chunk", "score": 0.9 }]"""
    text: str
    score: float