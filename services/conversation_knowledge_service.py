"""
Requirement 2 -- Attach an EXISTING Knowledge Base to a conversation.

This is Option B ("Attach Knowledge") alongside Option A ("Upload
Documents", handled by services/attachment_service.py). Attaching an
existing Knowledge Base creates ONLY a reference row
(ConversationKnowledgeBase) -- it never copies, re-embeds, or
otherwise touches the underlying KnowledgeDocument / KnowledgeChunk
rows or their pgvector embeddings.

Every function here re-verifies ownership of BOTH sides before doing
anything:

    - the conversation must belong to the calling user
      (services/conversation_service.get_conversation)

    - the Knowledge Base must belong to the calling user
      (services/knowledge_service.get_knowledge_base, which already
      returns None for another user's KB -- that is what lets us
      return the exact same 404 for "not found" and "not yours")
"""

import uuid
from typing import List, Optional

from sqlalchemy.orm import Session, joinedload

from models.conversation_knowledge_base import ConversationKnowledgeBase
from models.knowledge import KnowledgeBase

from services import knowledge_service
from services.conversation_service import (
    get_conversation,
    ConversationNotFoundError,
)


class KnowledgeBaseNotFoundError(Exception):
    """Raised when a Knowledge Base does not exist or is not owned by the user."""


# Re-exported so callers only need to import this module, e.g.
#   except conversation_knowledge_service.ConversationNotFoundError:
# rather than also importing conversation_service.
__all__ = [
    "ConversationNotFoundError",
    "KnowledgeBaseNotFoundError",
    "attach_knowledge_base",
    "detach_knowledge_base",
    "list_attached_knowledge_bases",
    "get_conversation_attached_knowledge_base_ids",
]


def _generate_id() -> str:
    return str(uuid.uuid4())


def attach_knowledge_base(
    db: Session,
    conversation_id: str,
    knowledge_base_id: str,
    user_id: int,
) -> ConversationKnowledgeBase:
    """
    Attach an existing Knowledge Base to a conversation.

    Idempotent: attaching the same (conversation, knowledge_base)
    pair twice returns the existing link rather than raising an
    IntegrityError on the unique constraint.
    """

    # Ownership check #1: the conversation must belong to this user.
    get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)

    # Ownership check #2: the Knowledge Base must belong to this user.
    # get_knowledge_base() already scopes by user_id and returns None
    # for another user's KB -- we deliberately don't distinguish that
    # from "doesn't exist" (see KnowledgeBaseNotFoundError usage in
    # api/conversations.py).
    kb = knowledge_service.get_knowledge_base(
        db,
        knowledge_base_id,
        user_id=user_id,
    )

    if not kb:
        raise KnowledgeBaseNotFoundError(
            f"Knowledge base '{knowledge_base_id}' not found"
        )

    existing = (
        db.query(ConversationKnowledgeBase)
        .filter(
            ConversationKnowledgeBase.conversation_id == conversation_id,
            ConversationKnowledgeBase.knowledge_base_id == knowledge_base_id,
            ConversationKnowledgeBase.user_id == user_id,
        )
        .first()
    )

    if existing:
        return existing

    link = ConversationKnowledgeBase(
        id=_generate_id(),
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        user_id=user_id,
    )

    db.add(link)
    db.commit()
    db.refresh(link)

    return link


def detach_knowledge_base(
    db: Session,
    conversation_id: str,
    knowledge_base_id: str,
    user_id: int,
) -> None:
    """
    Detach a Knowledge Base from a conversation.

    Deletes ONLY the join row. The Knowledge Base itself, its
    documents, and its embeddings are left completely untouched and
    remain available to the user everywhere else.

    Detaching a link that doesn't exist (or was already detached) is
    a no-op, not an error -- DELETE is naturally idempotent here.
    """

    # Ownership check: the conversation must belong to this user.
    get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)

    (
        db.query(ConversationKnowledgeBase)
        .filter(
            ConversationKnowledgeBase.conversation_id == conversation_id,
            ConversationKnowledgeBase.knowledge_base_id == knowledge_base_id,
            ConversationKnowledgeBase.user_id == user_id,
        )
        .delete()
    )

    db.commit()


def list_attached_knowledge_bases(
    db: Session,
    conversation_id: str,
    user_id: int,
) -> List[KnowledgeBase]:
    """
    List the Knowledge Bases explicitly attached to a conversation
    (Option B only -- this does NOT include Knowledge Bases created
    implicitly from uploaded documents, Option A).
    """

    # Ownership check: the conversation must belong to this user.
    get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)

    links = (
        db.query(ConversationKnowledgeBase)
        .options(joinedload(ConversationKnowledgeBase.knowledge_base))
        .filter(
            ConversationKnowledgeBase.conversation_id == conversation_id,
            ConversationKnowledgeBase.user_id == user_id,
        )
        .order_by(ConversationKnowledgeBase.created_at.asc())
        .all()
    )

    return [link.knowledge_base for link in links]


def get_conversation_attached_knowledge_base_ids(
    db: Session,
    conversation_id: Optional[str],
    user_id: Optional[int],
) -> List[str]:
    """
    Return the list of Knowledge Base IDs explicitly attached to a
    conversation for this user.

    Used by manager/service.py::_get_conversation_knowledge_base_ids()
    to merge Option B (attached KBs) with Option A (uploaded
    attachments) into the single flat list the Manager consumes.

    Deliberately does NOT raise ConversationNotFoundError -- this is
    called from deep inside the orchestration path (including
    approve_pending_agent(), which has no HTTP request/response cycle
    to turn a 404 into), so an unknown/foreign conversation_id simply
    resolves to "no attached knowledge", matching how the uploaded-
    attachment side of the same helper already behaves.
    """

    if not conversation_id or user_id is None:
        return []

    rows = (
        db.query(ConversationKnowledgeBase.knowledge_base_id)
        .filter(
            ConversationKnowledgeBase.conversation_id == conversation_id,
            ConversationKnowledgeBase.user_id == user_id,
        )
        .distinct()
        .all()
    )

    return [row[0] for row in rows if row[0]]
