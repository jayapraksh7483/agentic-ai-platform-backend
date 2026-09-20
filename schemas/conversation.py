"""
Phase 5C / 5D -- Conversation and Message API schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# Conversations
# ============================================================

class ConversationCreate(BaseModel):
    title: Optional[str] = Field(
        default=None,
        max_length=255,
    )


class ConversationResponse(BaseModel):
    id: str
    user_id: int
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse]


# ============================================================
# Messages
# ============================================================

class MessageCreate(BaseModel):
    content: str = Field(
        min_length=1
    )


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    message_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )


class MessageListResponse(BaseModel):
    messages: List[MessageResponse]


# ============================================================
# Phase 5D -- Chat
# ============================================================

class ChatRequest(BaseModel):
    content: str = Field(
        min_length=1
    )

    # Optional per-message override of which LLM the Manager itself
    # reasons with (classification, capability extraction, direct
    # response, synthesis). None/omitted -> orchestrate()'s existing
    # default ("gemini", settings default model) is unchanged.
    provider: Optional[str] = None
    model: Optional[str] = None


class ChatResponse(BaseModel):
    conversation_id: str

    user_message: MessageResponse

    assistant_message: Optional[MessageResponse] = None

    execution_id: Optional[str] = None

    status: str

    error: Optional[str] = None

class AttachmentResponse(BaseModel):
    id: int
    conversation_id: str
    user_id: int
    filename: str
    content_type: Optional[str] = None
    file_size: Optional[int] = None
    knowledge_document_id: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


class AttachmentListResponse(BaseModel):
    attachments: List[AttachmentResponse]


# ============================================================
# Requirement 2 -- Attaching an EXISTING Knowledge Base
# ============================================================

class AttachKnowledgeBaseRequest(BaseModel):
    knowledge_base_id: str


class AttachedKnowledgeBaseResponse(BaseModel):
    """
    A Knowledge Base attached to a conversation (Option B). Deliberately
    the same shape a frontend would want for the "Attach Knowledge"
    checklist -- id/name/description -- so the same response can
    populate both "already attached" and "available to attach" lists.
    """

    id: str
    name: str
    description: Optional[str] = None
    document_count: int

    model_config = ConfigDict(
        from_attributes=True,
    )


class AttachedKnowledgeBaseListResponse(BaseModel):
    knowledge_bases: List[AttachedKnowledgeBaseResponse]