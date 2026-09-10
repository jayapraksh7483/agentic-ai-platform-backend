"""
Phase 5A -- centralized auth utilities.

Everything token/password-related lives here so there is exactly one
place that knows how a password is hashed or a JWT is signed. Reuses
the project's existing settings.SECRET_KEY / settings.ALGORITHM /
settings.ACCESS_TOKEN_EXPIRE_MINUTES (already present in core/config.py
from an earlier, since-removed auth attempt) rather than introducing a
second, differently-named set of JWT settings.
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

# tokenUrl is only used by Swagger's "Authorize" button to know which
# endpoint issues tokens -- it does not change how this dependency
# validates a token that's already been supplied.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ACCESS_TOKEN_TYPE = "access"


# --- Password hashing --------------------------------------------------

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


# --- JWT access tokens ---------------------------------------------------

def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": expire,
        "type": ACCESS_TOKEN_TYPE,
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


class InvalidTokenError(Exception):
    pass


def decode_access_token(token: str) -> dict:
    """Raises InvalidTokenError for anything wrong with the token
    (bad signature, expired, wrong type, malformed) -- callers don't
    need to distinguish why, they all map to the same 401."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        raise InvalidTokenError("Invalid or expired token")

    if payload.get("type") != ACCESS_TOKEN_TYPE:
        raise InvalidTokenError("Wrong token type")

    return payload


# --- Refresh tokens (opaque, DB-backed -- see models/user.py) ----------

def generate_refresh_token_value() -> str:
    """High-entropy random string handed to the client. Only its hash
    is ever stored (see hash_refresh_token) -- this raw value exists
    only in the response body and the client's storage, never in the DB
    or in logs."""
    return secrets.token_urlsafe(48)


def hash_refresh_token(raw_token: str) -> str:
    """SHA-256 is appropriate here (unlike for passwords): this input
    is already a 48-byte random value, not a human-guessable password,
    so a fast hash is fine and lets /refresh look it up by exact match
    instead of checking it against every stored hash."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


# --- get_current_user dependency ----------------------------------------

def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Reusable dependency for any protected endpoint:

        @router.get("/something")
        def something(current_user: User = Depends(get_current_user)):
            ...

    Order of checks, matching the Phase 5A spec exactly:
    1. Authorization header present
    2. Bearer token well-formed / JWT valid
    3. Not expired (decode_access_token raises on this)
    4. User referenced by `sub` still exists
    5. User is still active
    """
    if token is None:
        raise _unauthorized("Not authenticated")

    try:
        payload = decode_access_token(token)
    except InvalidTokenError:
        raise _unauthorized("Invalid or expired token")

    user_id_raw = payload.get("sub")
    if user_id_raw is None:
        raise _unauthorized("Invalid token payload")

    try:
        user_id = int(user_id_raw)
    except (TypeError, ValueError):
        raise _unauthorized("Invalid token payload")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise _unauthorized("User not found")

    if not user.is_active:
        raise _unauthorized("Inactive user")

    return user