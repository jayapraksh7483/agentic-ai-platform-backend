"""
Phase 5C -- Conversation and Message API schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# Conversation schemas
# ============================================================

class ConversationCreate(BaseModel):
    """
    Request body for creating a conversation.

    Title is optional because the frontend may initially create
    an untitled conversation and assign a title later.
    """

    title: Optional[str] = Field(
        default=None,
        max_length=255,
    )


class ConversationResponse(BaseModel):
    """
    Conversation returned by the API.
    """

    id: str
    user_id: int
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationListResponse(BaseModel):
    """
    Response wrapper for conversation listing.
    """

    conversations: List[ConversationResponse]


# ============================================================
# Message schemas
# ============================================================

class MessageCreate(BaseModel):
    """
    Request body for adding a message to a conversation.

    The client can only create normal user messages through
    this endpoint. Assistant/system/tool messages are created
    internally by the backend when orchestration is connected.
    """

    content: str = Field(
        min_length=1,
    )


class MessageResponse(BaseModel):
    """
    Message returned by the API.
    """

    id: str
    conversation_id: str
    role: str
    content: str
    message_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageListResponse(BaseModel):
    """
    Response wrapper for conversation messages.
    """

    messages: List[MessageResponse]