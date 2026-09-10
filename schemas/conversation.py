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


class ChatResponse(BaseModel):
    conversation_id: str

    user_message: MessageResponse

    assistant_message: Optional[MessageResponse] = None

    execution_id: Optional[str] = None

    status: str

    error: Optional[str] = None