import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.sql import func

from core.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class AgentAPIKey(Base):
    """
    External application credential for one user-owned Agent.

    The raw ag_live_* token is never stored. Only its SHA-256 hash
    and a short display prefix are persisted.
    """

    __tablename__ = "agent_api_keys"

    id = Column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    agent_id = Column(
        String(36),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    key_hash = Column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
    )

    # Safe display value only, for example:
    # ag_live_ab12cd34
    key_prefix = Column(
        String(32),
        nullable=False,
    )

    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_used_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    revoked_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )
