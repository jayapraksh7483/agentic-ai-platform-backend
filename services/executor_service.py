
"""
Agent Execution Engine.

RAG behavior:

1. Agent is RAG-enabled.
2. Authenticated user_id is required.
3. If the agent has an explicit knowledge_base_id:
       - verify the KB belongs to the authenticated user
       - search only that KB
4. If knowledge_base_id is NULL:
       - find all active KBs owned by the authenticated user
       - search across those KBs
       - merge and rank the results
       - use the best matching chunks
5. Build strict document-grounded prompt.
6. Execute through the LangGraph runtime.
7. Return the answer together with actual retrieved sources.

This allows:

    General Document Reader
        -> knowledge_base_id = NULL
        -> searches the current user's KBs

    Specialized Agent
        -> knowledge_base_id = specific KB
        -> searches only that authorized KB
"""

import concurrent.futures
import json
import time

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from core.config import settings
from core.execution import run_with_timeout

from models.agent import Agent, AgentStatus
from models.execution import AgentExecution, ExecutionStatus
from models.knowledge import KnowledgeBase, KnowledgeBaseStatus

from schemas.agent import (
    AgentExecuteRequest,
    AgentExecuteResponse,
)

from agent_runtime.runtime import agent_runtime

from services.agent_service import get_decrypted_api_key
from services.knowledge_service import (
    retrieve_documents,
    build_strict_rag_prompt,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAG_TOP_K = 5


# ---------------------------------------------------------------------------
# Input Helpers
# ---------------------------------------------------------------------------

def _extract_text(user_input) -> str:
    """
    Convert supported execution input formats into plain text.
    """

    if isinstance(user_input, str):
        return user_input

    if isinstance(user_input, dict):

        if "query" in user_input:
            return str(user_input["query"])

        if "text" in user_input:
            return str(user_input["text"])

        return json.dumps(user_input)

    return str(user_input)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AgentNotFoundError(Exception):
    pass


class AgentInactiveError(Exception):
    pass


class RAGConfigurationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Knowledge Base Resolution
# ---------------------------------------------------------------------------

def _get_user_knowledge_bases(
    db: Session,
    user_id: int,
) -> List[KnowledgeBase]:
    """
    Get all active knowledge bases belonging to the authenticated user.

    User ownership is enforced here so another user's KB can never be
    selected for retrieval.
    """

    return (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.created_by == user_id,
            KnowledgeBase.status == KnowledgeBaseStatus.ACTIVE,
        )
        .order_by(
            KnowledgeBase.created_at.desc()
        )
        .all()
    )


def _resolve_rag_knowledge_base_ids(
    db: Session,
    agent: Agent,
    user_id: int,
) -> List[str]:
    """
    Resolve which knowledge bases the RAG agent is allowed to search.

    Case 1:
        Agent has an explicit knowledge_base_id.

        Only that KB is searched after ownership validation.

    Case 2:
        Agent has knowledge_base_id = NULL.

        All active KBs owned by the authenticated user are searched.

    Returns:
        Authorized knowledge-base IDs.
    """

    # ============================================================
    # Option 1: Explicitly bound knowledge base
    # ============================================================

    if agent.knowledge_base_id:

        knowledge_base = (
            db.query(KnowledgeBase)
            .filter(
                KnowledgeBase.id == agent.knowledge_base_id,
                KnowledgeBase.created_by == user_id,
                KnowledgeBase.status == KnowledgeBaseStatus.ACTIVE,
            )
            .first()
        )

        if not knowledge_base:
            raise RAGConfigurationError(
                f"Knowledge base '{agent.knowledge_base_id}' "
                "was not found or does not belong to the authenticated user"
            )

        return [knowledge_base.id]

    # ============================================================
    # Option 2: Dynamic user-scoped knowledge bases
    # ============================================================

    knowledge_bases = _get_user_knowledge_bases(
        db=db,
        user_id=user_id,
    )

    if not knowledge_bases:
        raise RAGConfigurationError(
            "No active knowledge bases are available for this user"
        )

    return [
        knowledge_base.id
        for knowledge_base in knowledge_bases
    ]


# ---------------------------------------------------------------------------
# RAG Retrieval
# ---------------------------------------------------------------------------

def _retrieve_rag_context(
    db: Session,
    agent: Agent,
    user_input: str,
    user_id: Optional[int],
    knowledge_base_ids: Optional[List[str]] = None,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve user-owned documents for a RAG-enabled agent.

    Dynamic Document Reader:
        Searches across all active KBs owned by the current user.

    Specialized RAG agent:
        Searches only its explicitly bound KB after ownership validation.

    Returns:
        (
            strict_rag_system_prompt,
            source_metadata
        )
    """

    # ============================================================
    # Non-RAG agent
    # ============================================================

    if not agent.is_rag:
        return agent.system_prompt or "", []

    # ============================================================
    # RAG requires authentication
    # ============================================================

    if user_id is None:
        raise RAGConfigurationError(
            "RAG execution requires an authenticated user"
        )

    # ============================================================
    # Resolve authorized KBs
    # ============================================================
    #
    # When conversation KB IDs are supplied, they are the explicit
    # document scope for this chat. Every supplied KB is still checked
    # against the authenticated user and ACTIVE status.
    #
    # Without an explicit conversation scope, preserve the existing
    # behavior: an explicitly bound agent KB is used, otherwise the
    # authenticated user's active KBs are searched.
    # ============================================================

    if knowledge_base_ids and agent.knowledge_base_id:
        if agent.knowledge_base_id not in knowledge_base_ids:
            raise RAGConfigurationError("Conversation scope excludes the agent's configured knowledge base")
        knowledge_base_ids = [agent.knowledge_base_id]
    if knowledge_base_ids:
        requested_ids = list(dict.fromkeys(
            str(kb_id) for kb_id in knowledge_base_ids if kb_id
        ))

        authorized_rows = (
            db.query(KnowledgeBase)
            .filter(
                KnowledgeBase.id.in_(requested_ids),
                KnowledgeBase.created_by == user_id,
                KnowledgeBase.status == KnowledgeBaseStatus.ACTIVE,
            )
            .all()
        )

        authorized_by_id = {row.id: row for row in authorized_rows}

        missing_ids = [
            kb_id
            for kb_id in requested_ids
            if kb_id not in authorized_by_id
        ]

        if missing_ids:
            raise RAGConfigurationError(
                "Conversation document scope contains a knowledge base "
                "that was not found, is inactive, or does not belong "
                "to the authenticated user"
            )

        resolved_knowledge_base_ids = requested_ids
    else:
        resolved_knowledge_base_ids = _resolve_rag_knowledge_base_ids(
            db=db,
            agent=agent,
            user_id=user_id,
        )

    # ============================================================
    # Retrieve from every authorized KB
    # ============================================================

    all_documents = []

    for kb_id in resolved_knowledge_base_ids:
        from services.knowledge_service import is_knowledge_base_ready
        if not is_knowledge_base_ready(db, kb_id, user_id):
            raise RAGConfigurationError("Knowledge base indexing is incomplete or failed")

        documents = retrieve_documents(
            db=db,
            kb_id=kb_id,
            query=user_input,
            top_k=RAG_TOP_K,
            user_id=user_id,
        )

        all_documents.extend(documents)

    # ============================================================
    # No documents found
    # ============================================================

    if not all_documents:

        retrieved_chunks: List[Dict[str, Any]] = []

        rag_system_prompt = build_strict_rag_prompt(
            base_system_prompt=agent.system_prompt or "",
            retrieved_chunks=retrieved_chunks,
        )

        return rag_system_prompt, retrieved_chunks

    # ============================================================
    # Rank all retrieved documents
    # ============================================================

    def _document_score(document) -> float:
        """
        Safely extract retrieval score from LangChain metadata.
        """

        metadata = document.metadata or {}

        try:
            return float(
                metadata.get(
                    "score",
                    0.0,
                )
            )
        except (TypeError, ValueError):
            return 0.0

    all_documents.sort(
        key=_document_score,
        reverse=True,
    )

    # ============================================================
    # Keep strongest matches
    # ============================================================

    selected_documents = all_documents[:RAG_TOP_K]

    # ============================================================
    # Convert LangChain Documents to API source format
    # ============================================================

    retrieved_chunks: List[Dict[str, Any]] = []

    for document in selected_documents:

        metadata = document.metadata or {}

        retrieved_chunks.append(
            {
                "text": document.page_content,
                "score": _document_score(document),
                "document_id": metadata.get(
                    "document_id"
                ),
                "filename": metadata.get(
                    "filename"
                ),
                "chunk_index": metadata.get(
                    "chunk_index"
                ),
                "kb_id": metadata.get(
                    "kb_id"
                ),
            }
        )

    # ============================================================
    # Build strict document-only prompt
    # ============================================================

    rag_system_prompt = build_strict_rag_prompt(
        base_system_prompt=agent.system_prompt or "",
        retrieved_chunks=retrieved_chunks,
    )

    return rag_system_prompt, retrieved_chunks


# ---------------------------------------------------------------------------
# LLM / Runtime Execution
# ---------------------------------------------------------------------------

def _call_llm(
    db: Session,
    agent: Agent,
    system_prompt: str,
    user_input: str,
    user_id: Optional[int] = None,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Execute the agent through the LangGraph runtime.

    RAG:
        LangChain Retriever -> pgvector -> strict RAG prompt -> runtime.

    Normal:
        Direct runtime execution.
    """

    effective_system_prompt = system_prompt
    sources: List[Dict[str, Any]] = []

    # ============================================================
    # RAG path
    # ============================================================

    if agent.is_rag:

        effective_system_prompt, sources = _retrieve_rag_context(
            db=db,
            agent=agent,
            user_input=user_input,
            user_id=user_id,
        )

    # ============================================================
    # LangGraph runtime
    # ============================================================

    runtime_input = dict(
        agent_id=str(agent.id), user_input=user_input, provider=agent.provider,
        model=agent.model, system_prompt=effective_system_prompt, user_id=user_id,
        api_key=get_decrypted_api_key(agent), temperature=agent.temperature,
        allowed_tools=agent.tools,
    )
    result = run_with_timeout(lambda: agent_runtime.execute(**runtime_input), settings.EXECUTION_TIMEOUT_SECONDS)

    if result.get("error"):
        raise RuntimeError(
            result["error"]
        )

    output_text = str(
        result.get(
            "output",
            "",
        )
    )

    return output_text, sources


# ---------------------------------------------------------------------------
# Agent Execution
# ---------------------------------------------------------------------------

def execute_agent(
    db: Session,
    agent_id: str,
    request: AgentExecuteRequest,
    user_id: Optional[int] = None,
) -> AgentExecuteResponse:
    """
    Execute an agent.

    If user_id is supplied, the agent must belong to that user.

    User-facing endpoints must always provide the authenticated user's ID.
    """

    # ============================================================
    # 1. Find agent
    # ============================================================

    query = db.query(Agent).filter(
        Agent.id == agent_id
    )

    # ============================================================
    # 2. Enforce agent ownership
    # ============================================================

    if user_id is not None:
        query = query.filter(
            Agent.created_by == user_id
        )

    agent: Optional[Agent] = query.first()

    if not agent:
        raise AgentNotFoundError(
            f"Agent '{agent_id}' not found"
        )

    # ============================================================
    # 3. Validate agent status
    # ============================================================

    if agent.status != AgentStatus.ACTIVE:
        raise AgentInactiveError(
            f"Agent '{agent_id}' is not active"
        )

    # ============================================================
    # 4. Validate request input
    # ============================================================

    input_text = _extract_text(
        request.input
    )

    if not input_text or not input_text.strip():
        raise ValueError(
            "Execution input cannot be empty"
        )

    # ============================================================
    # 5. Create execution record
    # ============================================================

    execution = AgentExecution(
        agent_id=agent.id,
        user_id=user_id,
        input_payload=(
            json.dumps(request.input)
            if isinstance(request.input, dict)
            else request.input
        ),
        status=ExecutionStatus.PENDING,
    )

    db.add(execution)
    db.commit()
    db.refresh(execution)

    start = time.monotonic()

    try:

        # ========================================================
        # 6. Execute through runtime
        # ========================================================

        output_text, sources = _call_llm(
            db=db, agent=agent, system_prompt=agent.system_prompt or "",
            user_input=input_text, user_id=user_id,
        )

        # ========================================================
        # 7. Successful execution
        # ========================================================

        latency_ms = (
            time.monotonic() - start
        ) * 1000

        output_obj: Dict[str, Any] = {
            "result": output_text,
        }

        # --------------------------------------------------------
        # Return actual retrieved sources for RAG.
        # --------------------------------------------------------

        if agent.is_rag:
            output_obj["sources"] = sources

        execution.output_payload = json.dumps(
            output_obj
        )

        execution.status = ExecutionStatus.SUCCESS
        execution.end_time = datetime.now(
            timezone.utc
        )
        execution.latency_ms = latency_ms

        db.commit()

        return AgentExecuteResponse(
            execution_id=execution.id,
            agent_id=agent.id,
            status=execution.status.value,
            output=output_obj,
            latency_ms=latency_ms,
        )

    # ============================================================
    # 8. Timeout
    # ============================================================

    except TimeoutError as e:

        latency_ms = (
            time.monotonic() - start
        ) * 1000

        execution.status = ExecutionStatus.TIMEOUT
        execution.error_message = str(e)
        execution.end_time = datetime.now(
            timezone.utc
        )
        execution.latency_ms = latency_ms

        db.commit()

        return AgentExecuteResponse(
            execution_id=execution.id,
            agent_id=agent.id,
            status=execution.status.value,
            error=str(e),
            latency_ms=latency_ms,
        )

    # ============================================================
    # 9. General failure
    # ============================================================

    except Exception as e:

        latency_ms = (
            time.monotonic() - start
        ) * 1000

        execution.status = ExecutionStatus.FAILED
        execution.error_message = str(e)
        execution.end_time = datetime.now(
            timezone.utc
        )
        execution.latency_ms = latency_ms

        db.commit()

        return AgentExecuteResponse(
            execution_id=execution.id,
            agent_id=agent.id,
            status=execution.status.value,
            error=str(e),
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Timeout Helper
# ---------------------------------------------------------------------------

def _run_with_timeout(fn, timeout_seconds):
    return run_with_timeout(fn, timeout_seconds)
