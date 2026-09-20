
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

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user

from models.user import User
from models.message import MessageRole
from models.agent import Agent
from models.orchestration import (
    OrchestrationExecution,
    OrchestrationStep,
)

from schemas.conversation import (
    ConversationCreate,
    ConversationResponse,
    ConversationListResponse,
    MessageCreate,
    MessageResponse,
    MessageListResponse,
    ChatRequest,
    ChatResponse,
    AttachmentResponse,
    AttachmentListResponse,
)

from services import conversation_service, attachment_service
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


MAX_CONVERSATION_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024


# ------------------------------------------------------------------
# Upload conversation attachment
# ------------------------------------------------------------------

@router.post(
    "/{conversation_id}/attachments",
    response_model=AttachmentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_conversation_attachment(
    conversation_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Uploaded file is empty")
    if len(file_bytes) > MAX_CONVERSATION_UPLOAD_SIZE_BYTES:
        raise HTTPException(413, "Uploaded file exceeds the 20 MB limit")
    try:
        return attachment_service.create_attachment(
            db=db, conversation_id=conversation_id, user_id=current_user.id,
            filename=file.filename or "uploaded_file",
            content_type=file.content_type, file_bytes=file_bytes,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(422, str(exc))


@router.get(
    "/{conversation_id}/attachments",
    response_model=AttachmentListResponse,
)
def list_conversation_attachments(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        conversation_service.get_conversation(
            db=db, conversation_id=conversation_id, user_id=current_user.id
        )
    except ConversationNotFoundError:
        raise HTTPException(404, "Conversation not found")
    attachments = attachment_service.list_conversation_attachments(
        db=db, conversation_id=conversation_id, user_id=current_user.id
    )
    return {"attachments": attachments}


# ------------------------------------------------------------------
# Child-agent metadata for chat UI
# ------------------------------------------------------------------

def _execution_child_agents(
    db: Session,
    execution_id: str,
    user_id: int,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return real persisted orchestration children.

    Supports both:
      - registered Agent rows, which report their own stored provider/model
      - temporary hybrid workers, which report the Manager/chat execution
        provider/model because they do not have a persisted Agent row
    """
    if not execution_id:
        return []

    execution = (
        db.query(OrchestrationExecution)
        .filter(
            OrchestrationExecution.id == execution_id,
            OrchestrationExecution.user_id == user_id,
        )
        .first()
    )

    if execution is None:
        return []

    steps = (
        db.query(OrchestrationStep)
        .filter(
            OrchestrationStep.execution_id == execution_id,
        )
        .order_by(
            OrchestrationStep.id.asc(),
        )
        .all()
    )

    agent_ids = list(
        dict.fromkeys(
            str(step.agent_id)
            for step in steps
            if step.agent_id
        )
    )

    agents = []

    if agent_ids:
        agents = (
            db.query(Agent)
            .filter(
                Agent.id.in_(agent_ids),
                Agent.created_by == user_id,
            )
            .all()
        )

    agents_by_id = {
        str(agent.id): agent
        for agent in agents
    }

    normalized_provider = (
        str(provider_override or "")
        .strip()
        .lower()
        or None
    )

    normalized_model = (
        str(model_override or "").strip()
        or None
    )

    child_agents: List[Dict[str, Any]] = []
    seen = set()

    for step in steps:
        agent_id = str(
            step.agent_id or ""
        ).strip()

        result_payload = getattr(
            step,
            "result",
            None,
        )

        worker_type = None
        worker_name = None
        tool_name = None

        if isinstance(result_payload, dict):
            worker_type = (
                str(
                    result_payload.get(
                        "worker_type"
                    )
                    or ""
                ).strip()
                or None
            )
            worker_name = (
                str(
                    result_payload.get(
                        "worker_name"
                    )
                    or ""
                ).strip()
                or None
            )
            tool_name = (
                str(
                    result_payload.get(
                        "tool_name"
                    )
                    or ""
                ).strip()
                or None
            )

        # Ignore rows that are neither a registered child nor a known
        # temporary/persistent hybrid worker.
        if not agent_id and not worker_type:
            continue

        identity = (
            agent_id
            if agent_id
            else f"temporary-step:{step.id}"
        )

        if identity in seen:
            continue

        seen.add(identity)

        agent = (
            agents_by_id.get(agent_id)
            if agent_id
            else None
        )

        raw_status = getattr(
            step,
            "status",
            None,
        )

        if hasattr(raw_status, "value"):
            step_status = raw_status.value
        elif raw_status is None:
            step_status = None
        else:
            step_status = str(raw_status)

        stored_provider = (
            str(
                getattr(agent, "provider", "")
                or ""
            ).strip()
            if agent is not None
            else ""
        )

        stored_model = (
            str(
                getattr(agent, "model", "")
                or ""
            ).strip()
            if agent is not None
            else ""
        )

        child_agents.append(
            {
                "agent_id": (
                    agent_id
                    or None
                ),
                "name": (
                    worker_name
                    or getattr(
                        agent,
                        "name",
                        None,
                    )
                    or "Child Agent"
                ),
                "capability": getattr(
                    step,
                    "capability",
                    None,
                ),
                "status": step_status,
                "provider": (
                    (stored_provider or None)
                    if agent is not None
                    else normalized_provider
                ),
                "model": (
                    (stored_model or None)
                    if agent is not None
                    else normalized_model
                ),
                "tool_name": tool_name,
                "worker_type": worker_type,
            }
        )

    return child_agents


def _assistant_message_metadata(
    db: Session,
    execution_id: str,
    user_id: int,
    manager_status: str,
    message_type: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Metadata consumed by Chat.tsx.

    The route is derived from real persisted orchestration steps.
    The orchestrator reports the chat-selected Manager provider/model,
    while registered child agents report their own stored provider/model.
    """
    normalized_provider = (
        str(provider or "").strip().lower()
        or None
    )
    normalized_model = (
        str(model or "").strip()
        or None
    )

    return {
        "execution_id": execution_id,
        "status": manager_status,
        "type": message_type,
        "manager_build": "2026-09-19-hybrid-agent-v6",
        "orchestrator": {
            "name": "Manager Agent",
            "provider": normalized_provider,
            "model": normalized_model,
        },
        "child_agents": _execution_child_agents(
            db=db,
            execution_id=execution_id,
            user_id=user_id,
            provider_override=normalized_provider,
            model_override=normalized_model,
        ),
    }


def _compact_chat_error(
    error: Optional[str],
    provider: Optional[str] = None,
) -> str:
    """
    Keep provider failures useful without dumping a full SDK payload into
    the chat UI.
    """
    raw = str(error or "").strip()

    if not raw:
        return "The request could not be completed."

    lowered = raw.lower()
    provider_name = (
        str(provider or "").strip().title()
        or "LLM"
    )

    if (
        "resource_exhausted" in lowered
        or "quota" in lowered
        or "rate limit" in lowered
        or "429" in lowered
    ):
        return (
            f"{provider_name} request failed because the selected "
            "provider/model reached a quota or rate limit."
        )

    if (
        "unauthorized" in lowered
        or "authentication" in lowered
        or "invalid api key" in lowered
        or "401" in lowered
    ):
        return (
            f"{provider_name} authentication failed. Check the API key "
            "configured for that provider."
        )

    if len(raw) > 400:
        return raw[:397] + "..."

    return raw


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
                "message_metadata": (
                    message.message_metadata
                    if isinstance(
                        message.message_metadata,
                        dict,
                    )
                    else {}
                ),
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

    selected_provider = (
        str(request.provider or "gemini")
        .strip()
        .lower()
    )
    selected_model = (
        str(request.model or "").strip()
        or None
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
            provider=selected_provider,
            model=selected_model,
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
                db=db, conversation_id=conversation_id, user_id=current_user.id,
                content=str(result), role=MessageRole.ASSISTANT,
                message_metadata=_assistant_message_metadata(
                    db=db,
                    execution_id=execution_id,
                    user_id=current_user.id,
                    manager_status=manager_status,
                    message_type="agent_approval_required",
                    provider=selected_provider,
                    model=selected_model,
                ),
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
        compact_error = _compact_chat_error(
            error or "Manager execution failed",
            provider=selected_provider,
        )

        assistant_message = conversation_service.add_message(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
            content=compact_error,
            role=MessageRole.ASSISTANT,
            message_metadata=_assistant_message_metadata(
                db=db,
                execution_id=execution_id,
                user_id=current_user.id,
                manager_status=manager_status,
                message_type="assistant_error",
                provider=selected_provider,
                model=selected_model,
            ),
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
            error=compact_error,
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
        db=db, conversation_id=conversation_id, user_id=current_user.id,
        content=str(result), role=MessageRole.ASSISTANT,
        message_metadata=_assistant_message_metadata(
            db=db,
            execution_id=execution_id,
            user_id=current_user.id,
            manager_status=manager_status,
            message_type="assistant_response",
            provider=selected_provider,
            model=selected_model,
        ),
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
 