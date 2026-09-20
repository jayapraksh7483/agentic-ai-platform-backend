import hashlib
import secrets
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from models.agent import Agent, AgentStatus
from models.agent_api_key import AgentAPIKey


AGENT_API_KEY_PREFIX = "ag_live_"


class AgentAPIKeyNotFoundError(Exception):
    pass


class AgentAPIKeyUnauthorizedError(Exception):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()


def _generate_raw_api_key() -> str:
    # 32 random bytes -> high-entropy opaque application credential.
    return AGENT_API_KEY_PREFIX + secrets.token_urlsafe(32)


def _display_prefix(raw_key: str) -> str:
    # Enough to identify the key in the UI without exposing it.
    return raw_key[:16]


def _get_owned_agent(
    db: Session,
    agent_id: str,
    user_id: int,
) -> Optional[Agent]:
    return (
        db.query(Agent)
        .filter(
            Agent.id == agent_id,
            Agent.created_by == user_id,
        )
        .first()
    )


def create_api_key(
    db: Session,
    agent_id: str,
    user_id: int,
    expires_at: Optional[datetime] = None,
) -> Tuple[AgentAPIKey, str]:
    """
    Create one external API key for a user-created Agent.

    The raw token is returned exactly once. Only its SHA-256 hash is
    persisted.
    """

    agent = _get_owned_agent(
        db=db,
        agent_id=agent_id,
        user_id=user_id,
    )

    if agent is None:
        raise AgentAPIKeyNotFoundError(
            "Agent not found"
        )

    if agent.is_default:
        raise ValueError(
            "Platform default agents cannot be published for "
            "external API access."
        )

    if agent.status != AgentStatus.ACTIVE:
        raise ValueError(
            "Only active agents can receive external API keys."
        )

    if expires_at is not None:
        now = _utcnow()

        # Normalize naive timestamps to UTC for predictable behavior.
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(
                tzinfo=timezone.utc
            )

        if expires_at <= now:
            raise ValueError(
                "expires_at must be in the future."
            )

    raw_key = _generate_raw_api_key()

    record = AgentAPIKey(
        user_id=user_id,
        agent_id=agent_id,
        key_hash=_hash_api_key(raw_key),
        key_prefix=_display_prefix(raw_key),
        is_active=True,
        expires_at=expires_at,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return record, raw_key


def list_api_keys(
    db: Session,
    agent_id: str,
    user_id: int,
) -> List[AgentAPIKey]:
    agent = _get_owned_agent(
        db=db,
        agent_id=agent_id,
        user_id=user_id,
    )

    if agent is None:
        raise AgentAPIKeyNotFoundError(
            "Agent not found"
        )

    return (
        db.query(AgentAPIKey)
        .filter(
            AgentAPIKey.agent_id == agent_id,
            AgentAPIKey.user_id == user_id,
        )
        .order_by(
            AgentAPIKey.created_at.desc()
        )
        .all()
    )


def revoke_api_key(
    db: Session,
    agent_id: str,
    key_id: str,
    user_id: int,
) -> AgentAPIKey:
    agent = _get_owned_agent(
        db=db,
        agent_id=agent_id,
        user_id=user_id,
    )

    if agent is None:
        raise AgentAPIKeyNotFoundError(
            "Agent not found"
        )

    record = (
        db.query(AgentAPIKey)
        .filter(
            AgentAPIKey.id == key_id,
            AgentAPIKey.agent_id == agent_id,
            AgentAPIKey.user_id == user_id,
        )
        .first()
    )

    if record is None:
        raise AgentAPIKeyNotFoundError(
            "API key not found"
        )

    if record.is_active:
        record.is_active = False
        record.revoked_at = _utcnow()
        db.commit()
        db.refresh(record)

    return record


def authenticate_external_api_key(
    db: Session,
    agent_id: str,
    raw_key: str,
) -> AgentAPIKey:
    """
    Authenticate an external application key for exactly one Agent.

    Returns the key record on success. The caller should execute the
    Agent with user_id=record.user_id so the existing ownership and
    RAG isolation rules remain enforced end-to-end.
    """

    raw_key = str(raw_key or "").strip()

    if not raw_key.startswith(AGENT_API_KEY_PREFIX):
        raise AgentAPIKeyUnauthorizedError(
            "Invalid agent API key"
        )

    key_hash = _hash_api_key(raw_key)

    record = (
        db.query(AgentAPIKey)
        .filter(
            AgentAPIKey.agent_id == agent_id,
            AgentAPIKey.key_hash == key_hash,
            AgentAPIKey.is_active.is_(True),
        )
        .first()
    )

    if record is None:
        raise AgentAPIKeyUnauthorizedError(
            "Invalid or revoked agent API key"
        )

    now = _utcnow()

    if (
        record.expires_at is not None
        and record.expires_at <= now
    ):
        record.is_active = False
        record.revoked_at = now
        db.commit()

        raise AgentAPIKeyUnauthorizedError(
            "Agent API key has expired"
        )

    # Confirm the Agent still exists, is owned by the same user and
    # is active. executor_service performs the same ownership check
    # again at execution time (defense in depth).
    agent = (
        db.query(Agent)
        .filter(
            Agent.id == agent_id,
            Agent.created_by == record.user_id,
            Agent.status == AgentStatus.ACTIVE,
        )
        .first()
    )

    if agent is None:
        raise AgentAPIKeyUnauthorizedError(
            "Agent is unavailable"
        )

    record.last_used_at = now
    db.commit()
    db.refresh(record)

    return record
