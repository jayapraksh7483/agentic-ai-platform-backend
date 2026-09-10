
"""
Phase 5C / 5D -- Persistent Conversation APIs.

All endpoints require JWT authentication.

Ownership rule:
    A user can only access their own conversations and messages.

Phase 5D:
    Conversation chat requests are connected to the existing
    Manager / Orchestration layer.

Conversation flow:

    JWT
      ↓
    Conversation
      ↓
    Persistent message history
      ↓
    Manager / Orchestration
      ↓
    Agent Runtime / LangGraph
      ↓
    Assistant response
      ↓
    Persistent assistant message
"""

from typing import Any, Dict, List

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
    ChatRequest,
    ChatResponse,
)

from services import conversation_service
from services.conversation_service import (
    ConversationNotFoundError,
)

from manager import service as manager_service


router = APIRouter(
    prefix="/api/conversations",
    tags=["Conversations"],
)


# ------------------------------------------------------------------
# Create conversation
# ------------------------------------------------------------------

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
    return conversation_service.create_conversation(
        db=db,
        user_id=current_user.id,
        title=request.title,
    )


# ------------------------------------------------------------------
# List user's conversations
# ------------------------------------------------------------------

@router.get(
    "",
    response_model=ConversationListResponse,
)
def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conversations = conversation_service.list_conversations(
        db=db,
        user_id=current_user.id,
    )

    return {
        "conversations": conversations,
    }


# ------------------------------------------------------------------
# Get one conversation
# ------------------------------------------------------------------

@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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


# ------------------------------------------------------------------
# Delete conversation
# ------------------------------------------------------------------

@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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


# ------------------------------------------------------------------
# List messages
# ------------------------------------------------------------------

@router.get(
    "/{conversation_id}/messages",
    response_model=MessageListResponse,
)
def list_messages(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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


# ------------------------------------------------------------------
# Create standalone message
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Chat
# ------------------------------------------------------------------

@router.post(
    "/{conversation_id}/chat",
    response_model=ChatResponse,
)
def chat(
    conversation_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Send a message into a persistent conversation.

    Flow:

        1. Verify conversation ownership.
        2. Load previous messages.
        3. Persist current user message.
        4. Send previous history + current input to Manager.
        5. Link orchestration execution to this conversation.
        6. Persist generated assistant response.
        7. Return orchestration execution information.

    The current user message is NOT included in conversation_history
    because Manager receives it separately as user_input.
    """

    # --------------------------------------------------------------
    # Verify conversation ownership.
    # --------------------------------------------------------------

    try:
        conversation_service.get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )

    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # --------------------------------------------------------------
    # Load previous messages.
    # --------------------------------------------------------------

    previous_messages = conversation_service.list_messages(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )

    conversation_history: List[Dict[str, Any]] = []

    for message in previous_messages:
        conversation_history.append(
            {
                "role": message.role.value,
                "content": message.content,
            }
        )

    # --------------------------------------------------------------
    # Persist current user message.
    # --------------------------------------------------------------

    try:
        user_message = conversation_service.add_message(
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

    # --------------------------------------------------------------
    # Execute Manager.
    #
    # IMPORTANT:
    # conversation_id is passed so the orchestration execution
    # can be permanently linked to this conversation.
    # --------------------------------------------------------------

    try:
        execution = manager_service.orchestrate(
            db=db,
            user_input=request.content,
            user_id=current_user.id,
            conversation_history=conversation_history,
            conversation_id=conversation_id,
        )

    except Exception as exc:
        try:
            conversation_service.touch_conversation(
                db=db,
                conversation_id=conversation_id,
                user_id=current_user.id,
            )
        except Exception:
            pass

        return ChatResponse(
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=None,
            execution_id=None,
            status="failed",
            error=str(exc),
        )

    execution_id = execution.execution_id
    manager_status = execution.status
    result = execution.result
    error = execution.error

    # --------------------------------------------------------------
    # Agent approval required.
    # --------------------------------------------------------------

    if manager_status == "pending_agent_approval":

        assistant_message = None

        if result:
            assistant_message = conversation_service.add_message(
                db=db,
                conversation_id=conversation_id,
                user_id=current_user.id,
                content=str(result),
                message_metadata={
                    "execution_id": execution_id,
                    "status": manager_status,
                    "type": "agent_approval_required",
                },
            )

        try:
            conversation_service.touch_conversation(
                db=db,
                conversation_id=conversation_id,
                user_id=current_user.id,
            )
        except Exception:
            pass

        return ChatResponse(
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=assistant_message,
            execution_id=execution_id,
            status=manager_status,
            error=error,
        )

    # --------------------------------------------------------------
    # Manager execution failed.
    # --------------------------------------------------------------

    if manager_status in (
        "failed",
        "error",
    ):
        try:
            conversation_service.touch_conversation(
                db=db,
                conversation_id=conversation_id,
                user_id=current_user.id,
            )
        except Exception:
            pass

        return ChatResponse(
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=None,
            execution_id=execution_id,
            status=manager_status,
            error=error or "Manager execution failed",
        )

    # --------------------------------------------------------------
    # No generated response.
    # --------------------------------------------------------------

    if result is None:
        result = (
            "The request completed but "
            "no response was generated."
        )

    # --------------------------------------------------------------
    # Persist assistant response.
    # --------------------------------------------------------------

    assistant_message = conversation_service.add_message(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
        content=str(result),
        message_metadata={
            "execution_id": execution_id,
            "status": manager_status,
            "type": "assistant_response",
        },
    )

    # --------------------------------------------------------------
    # Update conversation timestamp.
    # --------------------------------------------------------------

    try:
        conversation_service.touch_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )
    except Exception:
        pass

    # --------------------------------------------------------------
    # Return chat response.
    # --------------------------------------------------------------

    return ChatResponse(
        conversation_id=conversation_id,
        user_message=user_message,
        assistant_message=assistant_message,
        execution_id=execution_id,
        status=manager_status,
        error=error,
    )
 