
"""
Phase 5A -- centralized auth utilities.

Everything token/password-related lives here so there is exactly one
place that knows how a password is hashed or a JWT is signed.

Access tokens are JWTs.

Refresh tokens are opaque, high-entropy values. Only their SHA-256
hash should be persisted by the authentication service.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from core.config import settings
from core.database import get_db
from models.user import User


# ---------------------------------------------------------------------------
# OAuth2 / Swagger
# ---------------------------------------------------------------------------
#
# Swagger's Authorize button sends username/password to this endpoint.
# The normal application login endpoint remains /api/auth/login and
# continues accepting the JSON LoginRequest body.
#
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login/oauth2",
    auto_error=False,
)


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


# ---------------------------------------------------------------------------
# Token constants
# ---------------------------------------------------------------------------

ACCESS_TOKEN_TYPE = "access"


# ---------------------------------------------------------------------------
# Password hashing utilities
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt.
    """
    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    password_hash: str,
) -> bool:
    """
    Verify a plaintext password against its bcrypt hash.
    """
    return pwd_context.verify(
        plain_password,
        password_hash,
    )


# ---------------------------------------------------------------------------
# JWT access tokens
# ---------------------------------------------------------------------------

def create_access_token(user_id: int) -> str:
    """
    Create a signed JWT access token for a platform user.
    """

    now = datetime.now(timezone.utc)

    expire = now + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": expire,
        "type": ACCESS_TOKEN_TYPE,
    }

    return jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


class InvalidTokenError(Exception):
    """
    Raised when an access token is invalid, expired, malformed,
    or is not an access token.
    """

    pass


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT access token.

    Raises:
        InvalidTokenError: when the token is invalid, expired,
        malformed, or has the wrong token type.
    """

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )

    except JWTError:
        raise InvalidTokenError(
            "Invalid or expired token"
        )

    if payload.get("type") != ACCESS_TOKEN_TYPE:
        raise InvalidTokenError(
            "Wrong token type"
        )

    return payload


# ---------------------------------------------------------------------------
# Opaque refresh tokens
# ---------------------------------------------------------------------------

def generate_refresh_token_value() -> str:
    """
    Generate a high-entropy refresh token.

    The raw token is returned to the client but should never be
    persisted in the database or written to logs.
    """

    return secrets.token_urlsafe(48)


def hash_refresh_token(raw_token: str) -> str:
    """
    Hash a refresh token using SHA-256.

    Refresh tokens are already cryptographically random, so a fast
    deterministic hash is appropriate for database lookup.
    """

    return hashlib.sha256(
        raw_token.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Authentication errors
# ---------------------------------------------------------------------------

def _unauthorized(detail: str) -> HTTPException:
    """
    Build a standard HTTP 401 response.
    """

    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={
            "WWW-Authenticate": "Bearer",
        },
    )


# ---------------------------------------------------------------------------
# Current-user dependency
# ---------------------------------------------------------------------------

def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Resolve the currently authenticated platform user.

    Validation order:

    1. Authorization header / bearer token exists.
    2. JWT is valid.
    3. JWT is not expired.
    4. JWT contains a valid user ID.
    5. User exists in the database.
    6. User is active.
    """

    if token is None:
        raise _unauthorized(
            "Not authenticated"
        )

    try:
        payload = decode_access_token(
            token
        )

    except InvalidTokenError:
        raise _unauthorized(
            "Invalid or expired token"
        )

    user_id_raw = payload.get("sub")

    if user_id_raw is None:
        raise _unauthorized(
            "Invalid token payload"
        )

    try:
        user_id = int(user_id_raw)

    except (TypeError, ValueError):
        raise _unauthorized(
            "Invalid token payload"
        )

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if user is None:
        raise _unauthorized(
            "User not found"
        )

    if not user.is_active:
        raise _unauthorized(
            "Inactive user"
        )

    return user
