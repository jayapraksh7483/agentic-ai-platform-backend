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

__all__ = [
    "Agent",
    "AgentStatus",
    "AgentVersion",
    "AgentCapability",
    "AgentExecution",
    "ExecutionStatus",
    "KnowledgeBase",
    "KnowledgeBaseStatus",
    "KnowledgeDocument",
    "DocumentStatus",
    "KnowledgeChunk",
    "User",
    "RefreshToken",
]