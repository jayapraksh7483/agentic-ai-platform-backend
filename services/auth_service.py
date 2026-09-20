
"""
Phase 5A -- Authentication business logic.

This service handles:

    - User registration
    - User authentication
    - Access-token creation
    - Refresh-token rotation
    - Refresh-token revocation
    - Default-agent provisioning for newly registered users

Default agents are provisioned after a user is successfully created.
Each default agent is owned by that user through Agent.created_by.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from core.config import settings
from core.security import (
    create_access_token,
    generate_refresh_token_value,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from models.user import RefreshToken, User
from schemas.auth import UserRegister
from services.default_agent_service import provision_default_agents


logger = logging.getLogger(__name__)


class DuplicateEmailError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


class InvalidRefreshTokenError(Exception):
    pass


def register_user(
    db: Session,
    data: UserRegister,
) -> User:
    """
    Register a new user and provision the platform's default agents.

    Every newly registered user receives:

        - Document Reader
        - Web Search
        - Calculator
        - General Assistant

    The default agents are user-owned through Agent.created_by.
    """

    normalized_email = str(
        data.email
    ).strip().lower()

    existing = (
        db.query(User)
        .filter(User.email == normalized_email)
        .first()
    )

    if existing:
        raise DuplicateEmailError(
            f"An account with email '{normalized_email}' already exists"
        )

    user = User(
        email=normalized_email,
        password_hash=hash_password(data.password),
        name=data.name,
        is_active=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    # ---------------------------------------------------------
    # Provision the four default agents for this user.
    #
    # This happens after the user has been committed so that
    # user.id is guaranteed to exist.
    # ---------------------------------------------------------
    try:
        provision_default_agents(
            db=db,
            user_id=user.id,
        )

    except Exception:
        # If default-agent provisioning fails, do not leave a
        # partially-created account behind.
        db.rollback()

        # The user was already committed before provisioning.
        # Delete the user explicitly so registration remains
        # atomic from the application's perspective.
        persisted_user = (
            db.query(User)
            .filter(User.id == user.id)
            .first()
        )

        if persisted_user is not None:
            db.delete(persisted_user)
            db.commit()

        raise

    db.refresh(user)

    return user


def authenticate_user(
    db: Session,
    email: str,
    password: str,
) -> User:
    """
    Authenticate a user using email and password.
    """

    normalized_email = str(
        email
    ).strip().lower()

    user = (
        db.query(User)
        .filter(User.email == normalized_email)
        .first()
    )

    if user is None:
        raise InvalidCredentialsError(
            "Incorrect email or password"
        )

    if not verify_password(
        password,
        user.password_hash,
    ):
        raise InvalidCredentialsError(
            "Incorrect email or password"
        )

    if not user.is_active:
        raise InvalidCredentialsError(
            "Incorrect email or password"
        )

    # Bug #1 fix: provision_default_agents() previously only ran in
    # register_user(). Any account created before that code existed
    # (or that otherwise lost its default agents) had zero default
    # agents forever, since login never re-ran it. Also run it here,
    # idempotently, on every successful login.
    #
    # provision_default_agents() must already be safe to call
    # repeatedly for the same user (skip/upsert agents that already
    # exist) -- it is not re-implemented here. Failures are swallowed
    # so a provisioning problem never blocks login itself.
    try:
        provision_default_agents(
            db=db,
            user_id=user.id,
        )
    except Exception:
        # Keep login valid even if default-agent sync fails.
        # Roll back because a failed PostgreSQL statement leaves
        # the SQLAlchemy transaction unusable until rollback.
        db.rollback()
        logger.exception(
            "Default agent provisioning failed during login for user_id=%s",
            user.id,
        )

    return user


def _issue_refresh_token(
    db: Session,
    user_id: int,
) -> str:
    """
    Create and persist a new opaque refresh token.

    Only the SHA-256 hash is stored in the database.
    """

    raw_token = generate_refresh_token_value()

    expires_at = (
        datetime.now(timezone.utc)
        + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
    )

    db.add(
        RefreshToken(
            user_id=user_id,
            token_hash=hash_refresh_token(raw_token),
            expires_at=expires_at,
        )
    )

    db.commit()

    return raw_token


def create_tokens_for_user(
    db: Session,
    user: User,
) -> Tuple[str, str, int]:
    """
    Returns:

        access_token
        refresh_token_raw
        expires_in_seconds
    """

    access_token = create_access_token(user.id)

    refresh_token_raw = _issue_refresh_token(
        db,
        user.id,
    )

    expires_in = (
        settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

    return (
        access_token,
        refresh_token_raw,
        expires_in,
    )


def _get_valid_refresh_token_row(
    db: Session,
    raw_token: str,
) -> RefreshToken:
    """
    Validate a refresh token and return its database row.
    """

    token_hash = hash_refresh_token(raw_token)

    row = (
        db.query(RefreshToken)
        .filter(
            RefreshToken.token_hash == token_hash
        )
        .first()
    )

    if row is None:
        raise InvalidRefreshTokenError(
            "Refresh token not recognized"
        )

    if row.revoked_at is not None:
        raise InvalidRefreshTokenError(
            "Refresh token has been revoked"
        )

    if (
        row.expires_at.replace(
            tzinfo=timezone.utc
        )
        < datetime.now(timezone.utc)
    ):
        raise InvalidRefreshTokenError(
            "Refresh token has expired"
        )

    return row


def refresh_access_token(
    db: Session,
    raw_token: str,
) -> Tuple[str, str, int]:
    """
    Rotate a refresh token.

    The old refresh token is revoked and a new refresh token
    is issued.
    """

    row = _get_valid_refresh_token_row(
        db,
        raw_token,
    )

    user = (
        db.query(User)
        .filter(User.id == row.user_id)
        .first()
    )

    if user is None or not user.is_active:
        raise InvalidRefreshTokenError(
            "Refresh token not recognized"
        )

    now = datetime.now(timezone.utc)

    row.revoked_at = now
    row.last_used_at = now

    db.add(row)

    access_token = create_access_token(
        user.id
    )

    new_refresh_token_raw = _issue_refresh_token(
        db,
        user.id,
    )

    expires_in = (
        settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

    db.commit()

    return (
        access_token,
        new_refresh_token_raw,
        expires_in,
    )


def revoke_refresh_token(
    db: Session,
    raw_token: str,
) -> None:
    """
    Revoke a refresh token if it exists and has not already
    been revoked.
    """

    token_hash = hash_refresh_token(raw_token)

    row = (
        db.query(RefreshToken)
        .filter(
            RefreshToken.token_hash == token_hash
        )
        .first()
    )

    if row is not None and row.revoked_at is None:
        row.revoked_at = datetime.now(
            timezone.utc
        )

        db.add(row)
        db.commit()

