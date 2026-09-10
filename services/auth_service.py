"""
Phase 5A business logic. Kept separate from api/auth.py the same way
every other feature in this project separates services/ from api/
(e.g. agent_service.py vs api/agents.py).
"""
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


class DuplicateEmailError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


class InvalidRefreshTokenError(Exception):
    pass


def register_user(db: Session, data: UserRegister) -> User:
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        # Generic on purpose -- doesn't say "email already registered"
        # vs "invalid email", just that this address can't be used.
        raise DuplicateEmailError(f"An account with email '{data.email}' already exists")

    user = User(
        email=data.email,
        password_hash=hash_password(data.password),
        name=data.name,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User:
    """Raises InvalidCredentialsError for EITHER a wrong email or a
    wrong password -- deliberately the same error/message for both, so
    a login attempt can't be used to enumerate which emails are
    registered."""
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        raise InvalidCredentialsError("Incorrect email or password")

    if not user.is_active:
        raise InvalidCredentialsError("Incorrect email or password")

    return user


def _issue_refresh_token(db: Session, user_id: int) -> str:
    raw_token = generate_refresh_token_value()
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    db.add(RefreshToken(
        user_id=user_id,
        token_hash=hash_refresh_token(raw_token),
        expires_at=expires_at,
    ))
    db.commit()
    return raw_token


def create_tokens_for_user(db: Session, user: User) -> Tuple[str, str, int]:
    """Returns (access_token, refresh_token_raw, expires_in_seconds)."""
    access_token = create_access_token(user.id)
    refresh_token_raw = _issue_refresh_token(db, user.id)
    expires_in = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    return access_token, refresh_token_raw, expires_in


def _get_valid_refresh_token_row(db: Session, raw_token: str) -> RefreshToken:
    token_hash = hash_refresh_token(raw_token)
    row = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()

    if row is None:
        raise InvalidRefreshTokenError("Refresh token not recognized")
    if row.revoked_at is not None:
        raise InvalidRefreshTokenError("Refresh token has been revoked")
    if row.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise InvalidRefreshTokenError("Refresh token has expired")

    return row


def refresh_access_token(db: Session, raw_token: str) -> Tuple[str, str, int]:
    """
    Validates the presented refresh token, then ROTATES it: the old
    token row is revoked and a brand-new refresh token is issued along
    with the new access token. The client must start using the new
    refresh token -- the old one is now dead, even though it hadn't
    expired yet.

    Rotation means a leaked-and-reused-later refresh token becomes
    immediately detectable/dead after its first legitimate use, rather
    than staying valid for its full multi-day lifetime.
    """
    row = _get_valid_refresh_token_row(db, raw_token)

    user = db.query(User).filter(User.id == row.user_id).first()
    if user is None or not user.is_active:
        raise InvalidRefreshTokenError("Refresh token not recognized")

    now = datetime.now(timezone.utc)
    row.revoked_at = now
    row.last_used_at = now
    db.add(row)

    access_token = create_access_token(user.id)
    new_refresh_token_raw = _issue_refresh_token(db, user.id)
    expires_in = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

    db.commit()
    return access_token, new_refresh_token_raw, expires_in


def revoke_refresh_token(db: Session, raw_token: str) -> None:
    """
    Logout. Deliberately idempotent/quiet: whether the token was valid,
    already revoked, expired, or simply never existed, this always
    succeeds from the caller's point of view -- a logout endpoint that
    can fail is more confusing than useful, and there's no information
    worth protecting by distinguishing these cases here (unlike login).
    """
    token_hash = hash_refresh_token(raw_token)
    row = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()

    if row is not None and row.revoked_at is None:
        row.revoked_at = datetime.now(timezone.utc)
        db.add(row)
        db.commit()