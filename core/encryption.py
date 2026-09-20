"""
Encryption helpers for sensitive secrets stored at rest.

Originally written for Google OAuth access/refresh tokens. Reused as-is
(same Fernet key, same functions) for encrypting user-supplied custom
agent LLM API keys (models/agent.py: Agent.api_key_encrypted), so the
platform has exactly one place that does symmetric encryption/decryption
of secrets rather than two near-duplicate implementations.
"""

from cryptography.fernet import Fernet, InvalidToken

from core.config import settings


def _get_fernet() -> Fernet:
    """
    Create a Fernet cipher using the application encryption key.
    """
    key = settings.TOKEN_ENCRYPTION_KEY

    if not key:
        raise RuntimeError(
            "TOKEN_ENCRYPTION_KEY is not configured. "
            "Set it in the backend .env file."
        )

    try:
        return Fernet(key.encode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "TOKEN_ENCRYPTION_KEY is invalid. "
            "Generate a valid Fernet key."
        ) from exc


def encrypt_token(token: str | None) -> str | None:
    """
    Encrypt a token/secret for database storage.
    """
    if token is None:
        return None

    if not token:
        return token

    fernet = _get_fernet()
    return fernet.encrypt(token.encode("utf-8")).decode("utf-8")


def decrypt_token(encrypted_token: str | None) -> str | None:
    """
    Decrypt a token/secret retrieved from database storage.
    """
    if encrypted_token is None:
        return None

    if not encrypted_token:
        return encrypted_token

    fernet = _get_fernet()

    try:
        return fernet.decrypt(
            encrypted_token.encode("utf-8")
        ).decode("utf-8")
    except InvalidToken as exc:
        raise RuntimeError(
            "Unable to decrypt stored secret. "
            "Check that TOKEN_ENCRYPTION_KEY has not changed."
        ) from exc
