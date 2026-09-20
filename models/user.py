"""
Phase 5A -- Authentication foundation.

User owns platform resources such as agents, conversations,
knowledge bases, and Google Workspace connections.

Refresh tokens are database-backed sessions. Only the SHA-256 hash
of each refresh token is stored in the database.
"""

import enum

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    email = Column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    password_hash = Column(
        String(255),
        nullable=False,
    )

    name = Column(
        String(150),
        nullable=True,
    )

    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # ---------------------------------------------------------
    # Authentication / sessions
    # ---------------------------------------------------------

    refresh_tokens = relationship(
        "RefreshToken",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    # ---------------------------------------------------------
    # Agent ownership
    # ---------------------------------------------------------

    agents = relationship(
        "Agent",
        back_populates="owner",
    )

    # ---------------------------------------------------------
    # Conversation ownership
    # ---------------------------------------------------------

    conversations = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    # ---------------------------------------------------------
    # Knowledge-base ownership
    # ---------------------------------------------------------

    knowledge_bases = relationship(
        "KnowledgeBase",
        back_populates="owner",
        cascade="all, delete-orphan",
    )

    # ---------------------------------------------------------
    # Google Workspace OAuth
    # ---------------------------------------------------------

    # google_connections = relationship(
    #     "GoogleConnection",
    #     back_populates="user",
    #     cascade="all, delete-orphan",
    # )


class RefreshToken(Base):
    """
    One row per issued refresh token / login session.

    token_hash:
        SHA-256 hash of the raw refresh token.

    revoked_at:
        Non-null means this session has been logged out/revoked.

    Keeping revoked rows provides an audit trail of past sessions.
    """

    __tablename__ = "refresh_tokens"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    token_hash = Column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    revoked_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_used_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    user = relationship(
        "User",
        back_populates="refresh_tokens",
    )