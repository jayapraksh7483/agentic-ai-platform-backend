"""
Authentication and security utilities.

Phase 5A:
    - Password hashing
    - JWT access-token creation
    - JWT access-token validation
    - Bearer-token authentication for protected APIs
    - Opaque refresh-token generation and hashing
    - FastAPI dependency for the authenticated user

Swagger authentication:
    Login is handled by POST /api/auth/login using JSON.
    Protected APIs use:
        Authorization: Bearer <access_token>

    We intentionally use HTTPBearer instead of OAuth2PasswordBearer
    because /api/auth/login is a JSON endpoint, not an OAuth2
    password-form endpoint.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from core.config import settings
from core.database import get_db
from models.user import User


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


def hash_password(password: str) -> str:
    """Hash a user's password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """Verify a plaintext password against its bcrypt hash.

    Invalid or malformed stored hashes are treated as an ordinary
    authentication failure instead of crashing the login endpoint.
    """
    if not plain_password or not hashed_password:
        return False

    try:
        return pwd_context.verify(
            plain_password,
            hashed_password,
        )
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Bearer authentication
# ---------------------------------------------------------------------------

# HTTPBearer makes Swagger show a simple Bearer-token authorization
# instead of the OAuth2 username/password popup.
#
# IMPORTANT:
# Stock HTTPBearer(auto_error=True) raises 403 Forbidden when the
# Authorization header is missing/malformed entirely, but our own
# code below raises 401 Unauthorized for an invalid/expired token.
# That inconsistency breaks frontends whose axios/fetch interceptors
# only listen for 401 to trigger a token refresh or redirect-to-login
# -- requests made before the token is attached come back as 403,
# slip past that logic, and just fail silently.
#
# Bearer401 normalizes both cases to 401 so "not authenticated" always
# means the same status code, regardless of whether the header was
# missing or the token was bad.
class Bearer401(HTTPBearer):
    async def __call__(
        self,
        request: Request,
    ) -> HTTPAuthorizationCredentials:
        try:
            return await super().__call__(request)
        except HTTPException:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )


bearer_scheme = Bearer401(
    auto_error=True,
)


# ---------------------------------------------------------------------------
# JWT access tokens
# ---------------------------------------------------------------------------

def create_access_token(
    user_id: int,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT access token.

    The user's database ID is stored in the JWT `sub` claim.
    """

    if expires_delta is None:
        expires_delta = timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    expire = datetime.now(timezone.utc) + expires_delta

    payload = {
        "sub": str(user_id),
        "exp": expire,
    }

    if len(settings.SECRET_KEY) < 32 or settings.SECRET_KEY == "CHANGE_ME_TO_A_RANDOM_SECRET":
        raise RuntimeError("Configure a random SECRET_KEY of at least 32 characters.")
    return jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def decode_access_token(token: str) -> Optional[int]:
    """
    Decode and validate an access token.

    Returns:
        User ID when valid.
        None when invalid or expired.
    """

    if len(settings.SECRET_KEY) < 32 or settings.SECRET_KEY == "CHANGE_ME_TO_A_RANDOM_SECRET":
        return None
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )

        subject = payload.get("sub")

        if subject is None:
            return None

        return int(subject)

    except (
        JWTError,
        ValueError,
        TypeError,
    ):
        return None


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme
    ),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency that returns the authenticated user.

    Every protected endpoint should use:

        current_user: User = Depends(get_current_user)

    Swagger will send:

        Authorization: Bearer <access_token>
    """

    # HTTPBearer has already parsed:
    #
    # Authorization: Bearer <token>
    #
    # We only need the actual token value.
    token = credentials.credentials

    user_id = decode_access_token(token)

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# ---------------------------------------------------------------------------
# Refresh tokens
# ---------------------------------------------------------------------------

def generate_refresh_token_value() -> str:
    """
    Generate a cryptographically secure opaque refresh token.

    The raw value is returned to the client.
    Only its hash is persisted.
    """

    return secrets.token_urlsafe(64)


def hash_refresh_token(raw_token: str) -> str:
    """
    SHA-256 hash of an opaque refresh token.

    Raw refresh tokens are never stored in the database.
    """

    return hashlib.sha256(
        raw_token.encode("utf-8")
    ).hexdigest()