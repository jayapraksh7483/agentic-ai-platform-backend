"""
Phase 5A -- Authentication foundation.

Two tables:
  - users: one row per registered account.
  - refresh_tokens: a database-backed session table -- NOT a second JWT.
    A refresh token issued to a client is a random opaque string; only
    its SHA-256 hash is stored here, so a stolen database dump can't be
    used to log in as anyone. This is what makes logout/revocation
    actually possible, which a stateless JWT refresh token cannot do on
    its own (see core/security.py for the hashing).

Deliberately NOT touched in this phase (per Phase 5A scope): no
ForeignKey from here to Agent.created_by or AgentExecution.user_id --
those stay exactly as they are (plain nullable Integer, unlinked) until
a later "agent ownership" phase wires them up on purpose.
"""
import enum

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(150), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    refresh_tokens = relationship(
        "RefreshToken", back_populates="user", cascade="all, delete-orphan"
    )

    agents = relationship(
        "Agent",
        back_populates="owner",
    )

class RefreshToken(Base):
    """
    One row per issued refresh token (i.e. one row per login "session").

    token_hash is unique+indexed so /api/auth/refresh can look a
    presented token up in O(1) without ever storing or logging the raw
    value. revoked_at being non-null means this specific session has
    been logged out; it is never deleted so there's an audit trail of
    past sessions.
    """
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    token_hash = Column(String(255), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="refresh_tokens")