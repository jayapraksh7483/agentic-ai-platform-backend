
from typing import List, Optional

from sqlalchemy.orm import Session

from models.attachment import ConversationAttachment
from models.conversation import Conversation
from models.knowledge import KnowledgeBase
from schemas.knowledge import KnowledgeBaseCreate
from services import knowledge_service


def create_attachment(
    db: Session,
    conversation_id: str,
    user_id: int,
    filename: str,
    content_type: Optional[str],
    file_bytes: bytes,
) -> ConversationAttachment:
    """
    Create a conversation attachment and process it through
    the existing Knowledge/RAG pipeline.

    One Knowledge Base is reused for all attachments belonging
    to the same conversation.
    """

    # --------------------------------------------------------------
    # Verify conversation ownership.
    # --------------------------------------------------------------

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        .first()
    )

    if not conversation:
        raise ValueError("Conversation not found")

    # --------------------------------------------------------------
    # Create attachment record.
    # --------------------------------------------------------------

    attachment = ConversationAttachment(
        conversation_id=conversation_id,
        user_id=user_id,
        filename=filename,
        content_type=content_type,
        file_size=len(file_bytes),
        status="processing",
    )

    db.add(attachment)
    db.commit()
    db.refresh(attachment)

    try:
        # ----------------------------------------------------------
        # Find the existing Knowledge Base for this conversation.
        #
        # KnowledgeBase names are globally unique in the current
        # database schema, so the conversation ID makes the name
        # unique.
        # ----------------------------------------------------------

        kb_name = f"Conversation {conversation_id}"

        kb = (
            db.query(KnowledgeBase)
            .filter(
                KnowledgeBase.name == kb_name,
                KnowledgeBase.created_by == user_id,
            )
            .first()
        )

        # ----------------------------------------------------------
        # Create the conversation Knowledge Base if it does not
        # already exist.
        # ----------------------------------------------------------

        if not kb:
            kb = knowledge_service.create_knowledge_base(
                db=db,
                kb_in=KnowledgeBaseCreate(
                    name=kb_name,
                    description="Conversation-scoped uploaded documents",
                ),
                created_by=user_id,
            )

        # ----------------------------------------------------------
        # Reuse the existing document-processing pipeline.
        #
        # Upload
        #   ↓
        # Parser
        #   ↓
        # Chunking
        #   ↓
        # Gemini Embeddings
        #   ↓
        # PostgreSQL + pgvector
        # ----------------------------------------------------------

        document = knowledge_service.process_upload(
            db=db,
            kb_id=kb.id,
            filename=filename,
            file_bytes=file_bytes,
            user_id=user_id,
        )

        # ----------------------------------------------------------
        # Link attachment to the Knowledge Document and Knowledge
        # Base.
        # ----------------------------------------------------------

        attachment.knowledge_document_id = document.id
        attachment.knowledge_base_id = kb.id
        attachment.status = "ready"
        attachment.error_message = None

        db.commit()
        db.refresh(attachment)

        return attachment

    except Exception as exc:
        attachment.status = "failed"
        attachment.error_message = str(exc)

        db.commit()
        db.refresh(attachment)

        raise


def get_attachment(
    db: Session,
    attachment_id: int,
    user_id: int,
) -> Optional[ConversationAttachment]:
    """
    Get one attachment belonging to the current user.
    """

    return (
        db.query(ConversationAttachment)
        .filter(
            ConversationAttachment.id == attachment_id,
            ConversationAttachment.user_id == user_id,
        )
        .first()
    )


def list_conversation_attachments(
    db: Session,
    conversation_id: str,
    user_id: int,
) -> List[ConversationAttachment]:
    """
    List attachments for a conversation owned by the current user.
    """

    return (
        db.query(ConversationAttachment)
        .filter(
            ConversationAttachment.conversation_id == conversation_id,
            ConversationAttachment.user_id == user_id,
        )
        .order_by(ConversationAttachment.created_at.asc())
        .all()
    )
 