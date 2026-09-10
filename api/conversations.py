"""
Phase 5C -- Persistent Conversation APIs.

All endpoints require JWT authentication.

Ownership rule:
    A user can only access their own conversations and messages.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user

from models.user import User
from schemas.conversation import (
    ConversationCreate,
    ConversationResponse,
    ConversationListResponse,
    MessageCreate,
    MessageResponse,
    MessageListResponse,
)

from services import conversation_service
from services.conversation_service import ConversationNotFoundError


router = APIRouter(
    prefix="/api/conversations",
    tags=["Conversations"],
)


# ============================================================
# Conversations
# ============================================================

@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new conversation for the authenticated user.
    """

    return conversation_service.create_conversation(
        db=db,
        user_id=current_user.id,
        title=request.title,
    )


@router.get(
    "",
    response_model=ConversationListResponse,
)
def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return only conversations owned by the authenticated user.
    """

    conversations = conversation_service.list_conversations(
        db=db,
        user_id=current_user.id,
    )

    return {
        "conversations": conversations,
    }


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get one conversation.

    Ownership is enforced by the service layer.
    """

    try:
        return conversation_service.get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )

    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a conversation owned by the authenticated user.

    Messages are automatically deleted because the
    messages.conversation_id foreign key uses ON DELETE CASCADE.
    """

    try:
        conversation_service.delete_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )

    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return None


# ============================================================
# Messages
# ============================================================

@router.get(
    "/{conversation_id}/messages",
    response_model=MessageListResponse,
)
def list_messages(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return all messages belonging to the user's conversation.
    """

    try:
        messages = conversation_service.list_messages(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )

    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return {
        "messages": messages,
    }


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_message(
    conversation_id: str,
    request: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Add a user message to a conversation.

    Assistant/system/tool messages will later be created
    internally by the orchestration/runtime layer.
    """

    try:
        return conversation_service.add_message(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
            content=request.content,
        )

    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )