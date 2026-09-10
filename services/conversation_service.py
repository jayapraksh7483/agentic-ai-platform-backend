"""
Phase 5C -- Conversation service.

All conversation operations are ownership-scoped.

A user can only access conversations where:
    conversations.user_id == current_user.id
"""

from typing import List, Optional

from sqlalchemy.orm import Session
from datetime import datetime, timezone

from models.conversation import Conversation
from models.message import Message, MessageRole


class ConversationNotFoundError(Exception):
    """Raised when a conversation does not exist or is not owned by the user."""


class MessageNotFoundError(Exception):
    """Raised when a message does not exist."""


def create_conversation(
    db: Session,
    user_id: int,
    title: Optional[str] = None,
) -> Conversation:
    conversation = Conversation(
        user_id=user_id,
        title=title,
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation


def list_conversations(
    db: Session,
    user_id: int,
) -> List[Conversation]:
    return (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )


def get_conversation(
    db: Session,
    conversation_id: str,
    user_id: int,
) -> Conversation:
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        .first()
    )

    if conversation is None:
        raise ConversationNotFoundError(
            "Conversation not found"
        )

    return conversation


def delete_conversation(
    db: Session,
    conversation_id: str,
    user_id: int,
) -> None:
    conversation = get_conversation(
        db,
        conversation_id,
        user_id,
    )

    db.delete(conversation)
    db.commit()


def add_message(
    db: Session,
    conversation_id: str,
    user_id: int,
    content: str,
    role: MessageRole = MessageRole.USER,
    message_metadata: Optional[dict] = None,
) -> Message:

    # First verify ownership.
    get_conversation(
        db,
        conversation_id,
        user_id,
    )

    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        message_metadata=message_metadata,
    )

    db.add(message)

    # Touch conversation updated_at so the conversation moves
    # to the top of the user's conversation list.
    conversation = get_conversation(
        db,
        conversation_id,
        user_id,
    )

    db.add(conversation)

    db.commit()
    db.refresh(message)

    return message


def list_messages(
    db: Session,
    conversation_id: str,
    user_id: int,
) -> List[Message]:

    # Verify ownership before returning any messages.
    get_conversation(
        db,
        conversation_id,
        user_id,
    )

    return (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id
        )
        .order_by(Message.created_at.asc())
        .all()
    )
def touch_conversation(
    db: Session,
    conversation_id: str,
    user_id: int,
) -> Conversation:
    conversation = get_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    conversation.updated_at = datetime.now(timezone.utc)

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation