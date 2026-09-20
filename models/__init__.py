"""
Database model registration.

Import all SQLAlchemy models here so that every mapped class is registered
before SQLAlchemy configures relationships between models.
"""

from .agent import (
    Agent,
    AgentStatus,
    AgentVersion,
    AgentCapability,
)

from .execution import (
    AgentExecution,
    ExecutionStatus,
)

from .knowledge import (
    KnowledgeBase,
    KnowledgeBaseStatus,
    KnowledgeDocument,
    DocumentStatus,
    KnowledgeChunk,
)

from .user import (
    User,
    RefreshToken,
)

from .conversation import (
    Conversation,
)

from .message import (
    Message,
    MessageRole,
)

from .orchestration import (
    OrchestrationExecution,
    OrchestrationStatus,
)

from .attachment import (
    ConversationAttachment,
)

from .conversation_knowledge_base import (
    ConversationKnowledgeBase,
)

from . import google_connection


__all__ = [
    # Agent
    "Agent",
    "AgentStatus",
    "AgentVersion",
    "AgentCapability",

    # Execution
    "AgentExecution",
    "ExecutionStatus",

    # Knowledge / RAG
    "KnowledgeBase",
    "KnowledgeBaseStatus",
    "KnowledgeDocument",
    "DocumentStatus",
    "KnowledgeChunk",

    # Authentication
    "User",
    "RefreshToken",

    # Conversations
    "Conversation",
    "Message",
    "MessageRole",

    # Orchestration
    "OrchestrationExecution",
    "OrchestrationStatus",

    # Conversation attachments (Requirement 1 -- uploaded documents)
    "ConversationAttachment",

    # Conversation <-> Knowledge Base links (Requirement 2 -- attach
    # an existing Knowledge Base)
    "ConversationKnowledgeBase",

    # Google OAuth
    "google_connection",
]