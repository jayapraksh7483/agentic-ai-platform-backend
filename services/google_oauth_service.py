# """
# Google Workspace OAuth. Handles everything up to "we have a valid,
# usable Credentials object for this platform user" -- Gmail/Drive/
# Calendar/Sheets/Meet-specific logic lives in
# services/google_workspace_service.py and tools/*.py (Part 3), not here.

# CSRF-safe `state` design
# -------------------------
# The OAuth callback (GET /api/oauth/google/callback) is hit by the
# user's browser as a plain redirect from Google -- it does NOT carry
# our platform's Authorization: Bearer header, so we can't use
# get_current_user() there. We also must NOT trust a user_id the browser
# might pass us directly (a classic OAuth account-mixup vulnerability).

# Instead, `state` is a short-lived JWT we sign ourselves (reusing the
# exact same jose + settings.SECRET_KEY already used for platform access
# tokens in core/security.py -- no new signing mechanism, no new table
# just to hold temporary state) at the moment the user clicks "Connect
# Google Workspace" *while authenticated*. It encodes which user started
# the flow. On callback, we verify the signature and expiry ourselves --
# we are reading back a token we minted, not trusting anything the
# browser or Google added -- which is exactly what prevents both CSRF
# and user-mixup.
# """
# import secrets
# from datetime import datetime, timedelta, timezone
# from typing import List, Optional

# import requests as http_requests
# from google.auth.transport.requests import Request as GoogleAuthRequest
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import Flow
# from jose import JWTError, jwt
# from sqlalchemy.orm import Session

# from core.config import settings
# from core.encryption import decrypt_token, encrypt_token
# from models.google_connection import GoogleConnection

# OAUTH_STATE_PURPOSE = "google_oauth_state"
# OAUTH_STATE_TTL_MINUTES = 10

# # Which granted scope unlocks which user-facing "service" name -- used
# # only to build the `services: [...]` list in the status response, and
# # nowhere else (tools check the actual connection, not this list).
# _SCOPE_SERVICE_MAP = {
#     "https://www.googleapis.com/auth/gmail.readonly": "gmail",
#     "https://www.googleapis.com/auth/gmail.send": "gmail",
#     "https://www.googleapis.com/auth/gmail.compose": "gmail",
#     "https://www.googleapis.com/auth/drive.readonly": "drive",
#     "https://www.googleapis.com/auth/drive.file": "drive",
#     "https://www.googleapis.com/auth/calendar": "calendar",
#     "https://www.googleapis.com/auth/calendar.events": "calendar",
#     "https://www.googleapis.com/auth/spreadsheets": "sheets",
# }
# # Meet has no dedicated OAuth scope of its own in this phase -- Meet
# # links are created as a side effect of a Calendar event
# # (conferenceData), so "meet" is available whenever "calendar" is.


# class GoogleOAuthError(Exception):
#     pass


# class InvalidOAuthStateError(GoogleOAuthError):
#     pass


# class GoogleAuthorizationRequiredError(Exception):
#     """Raised by get_valid_credentials() -- callers (tools, Part 3)
#     catch this specifically to build the `authorization_required`
#     response shape rather than letting it surface as a generic 500."""
#     def __init__(self, message: str, authorization_url: Optional[str] = None):
#         super().__init__(message)
#         self.message = message
#         self.authorization_url = authorization_url


# class GoogleNotConnectedError(GoogleAuthorizationRequiredError):
#     pass


# # --- datetime helpers -----------------------------------------------
# # google-auth's Credentials.expiry is, by that library's own internal
# # convention, a NAIVE UTC datetime -- comparing it against an aware one
# # (e.g. straight from our DateTime(timezone=True) column) raises
# # TypeError. We store token_expiry as tz-aware UTC in Postgres (for
# # clarity/consistency with every other timestamp column in this
# # project) and convert at the two boundaries where it crosses into/out
# # of a google-auth Credentials object.

# def _to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
#     if dt is None:
#         return None
#     if dt.tzinfo is not None:
#         return dt.astimezone(timezone.utc).replace(tzinfo=None)
#     return dt


# def _to_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
#     if dt is None:
#         return None
#     if dt.tzinfo is None:
#         return dt.replace(tzinfo=timezone.utc)
#     return dt


# # --- CSRF-safe state ------------------------------------------------

# def create_oauth_state(user_id: int) -> str:
#     now = datetime.now(timezone.utc)
#     payload = {
#         "sub": str(user_id),
#         "purpose": OAUTH_STATE_PURPOSE,
#         "nonce": secrets.token_urlsafe(16),
#         "iat": now,
#         "exp": now + timedelta(minutes=OAUTH_STATE_TTL_MINUTES),
#     }
#     return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


# def verify_oauth_state(state: str) -> int:
#     """Returns the user_id that initiated this flow. Raises
#     InvalidOAuthStateError for anything wrong (expired, tampered,
#     wrong purpose, malformed) -- deliberately one error type, since the
#     caller (the callback endpoint) treats all of these identically."""
#     if not state:
#         raise InvalidOAuthStateError("Missing OAuth state")
#     try:
#         payload = jwt.decode(state, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
#     except JWTError:
#         raise InvalidOAuthStateError("Invalid or expired OAuth state")

#     if payload.get("purpose") != OAUTH_STATE_PURPOSE:
#         raise InvalidOAuthStateError("Invalid OAuth state purpose")

#     try:
#         return int(payload["sub"])
#     except (KeyError, TypeError, ValueError):
#         raise InvalidOAuthStateError("Invalid OAuth state payload")


# # --- Flow construction -----------------------------------------------

# def _client_config() -> dict:
#     return {
#         "web": {
#             "client_id": settings.GOOGLE_CLIENT_ID,
#             "client_secret": settings.GOOGLE_CLIENT_SECRET,
#             "auth_uri": "https://accounts.google.com/o/oauth2/auth",
#             "token_uri": "https://oauth2.googleapis.com/token",
#             "redirect_uris": [settings.GOOGLE_REDIRECT_URI],
#         }
#     }


# def _build_flow() -> Flow:
#     if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
#         raise GoogleOAuthError(
#             "Google OAuth is not configured on this server "
#             "(GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET missing)."
#         )
#     flow = Flow.from_client_config(
#         _client_config(),
#         scopes=settings.GOOGLE_OAUTH_SCOPES.split(),
#     )
#     flow.redirect_uri = settings.GOOGLE_REDIRECT_URI
#     return flow


# # --- Authorization URL -------------------------------------------------

# def build_authorization_url(user_id: int) -> str:
#     state = create_oauth_state(user_id)
#     flow = _build_flow()
#     # access_type=offline -> we get a refresh_token back.
#     # prompt=consent -> forces Google to re-issue a refresh_token even
#     # if this user connected before (otherwise a returning user often
#     # gets NO refresh_token on the second consent, which would silently
#     # break re-connection after a disconnect).
#     authorization_url, _ = flow.authorization_url(
#         access_type="offline",
#         include_granted_scopes="true",
#         prompt="consent",
#         state=state,
#     )
#     return authorization_url


# # --- Callback / token exchange -----------------------------------------

# def _fetch_google_email(credentials: Credentials) -> Optional[str]:
#     """Best-effort -- if this fails for any reason, we still complete
#     the connection without an email on file rather than failing the
#     whole OAuth flow over a non-essential detail."""
#     try:
#         resp = http_requests.get(
#             "https://www.googleapis.com/oauth2/v2/userinfo",
#             headers={"Authorization": f"Bearer {credentials.token}"},
#             timeout=10,
#         )
#         if resp.status_code == 200:
#             return resp.json().get("email")
#     except http_requests.RequestException:
#         pass
#     return None


# def handle_oauth_callback(
#     db: Session,
#     code: Optional[str],
#     state: Optional[str],
#     error: Optional[str] = None,
# ) -> GoogleConnection:
#     if error:
#         raise GoogleOAuthError(f"Google denied authorization: {error}")
#     if not code:
#         raise GoogleOAuthError("Missing authorization code")

#     user_id = verify_oauth_state(state)

#     flow = _build_flow()
#     flow.fetch_token(code=code)
#     credentials = flow.credentials

#     email = _fetch_google_email(credentials)
#     granted_scopes = " ".join(credentials.scopes or [])

#     connection = (
#         db.query(GoogleConnection)
#         .filter(GoogleConnection.user_id == user_id, GoogleConnection.provider == "google")
#         .first()
#     )
#     if connection is None:
#         connection = GoogleConnection(user_id=user_id, provider="google")
#         db.add(connection)

#     connection.email = email
#     connection.access_token = encrypt_token(credentials.token)
#     # Google sometimes omits refresh_token on a re-consent even with
#     # prompt=consent in edge cases (observed, not just theoretical) --
#     # never overwrite a good stored refresh_token with nothing.
#     if credentials.refresh_token:
#         connection.refresh_token = encrypt_token(credentials.refresh_token)
#     connection.token_expiry = _to_aware_utc(credentials.expiry)
#     connection.scopes = granted_scopes
#     connection.revoked_at = None

#     db.commit()
#     db.refresh(connection)
#     return connection


# # --- Reading / using the connection -------------------------------------

# def get_connection(db: Session, user_id: int) -> Optional[GoogleConnection]:
#     return (
#         db.query(GoogleConnection)
#         .filter(
#             GoogleConnection.user_id == user_id,
#             GoogleConnection.provider == "google",
#             GoogleConnection.revoked_at.is_(None),
#         )
#         .first()
#     )


# def get_status(db: Session, user_id: int) -> dict:
#     connection = get_connection(db, user_id)
#     if connection is None:
#         return {"connected": False, "provider": "google", "email": None, "services": []}

#     granted = set((connection.scopes or "").split())
#     services = sorted({
#         service_name
#         for scope, service_name in _SCOPE_SERVICE_MAP.items()
#         if scope in granted
#     })
#     if "calendar" in services:
#         services.append("meet")

#     return {
#         "connected": True,
#         "provider": "google",
#         "email": connection.email,
#         "services": services,
#     }


# def get_valid_credentials(db: Session, user_id: int) -> Credentials:
#     """
#     Used by every Google tool (Part 3) before doing anything with
#     Gmail/Drive/Calendar/Sheets. Returns a live, guaranteed-non-expired
#     Credentials object, refreshing and persisting a new access token
#     first if needed.

#     Raises GoogleNotConnectedError / GoogleAuthorizationRequiredError
#     for every failure mode (no connection, revoked, refresh itself
#     fails) -- callers build the `authorization_required` response from
#     that, never a raw 500.
#     """
#     connection = get_connection(db, user_id)
#     if connection is None:
#         raise GoogleNotConnectedError(
#             "Google Workspace is not connected for this account.",
#             authorization_url=build_authorization_url(user_id),
#         )

#     credentials = Credentials(
#         token=decrypt_token(connection.access_token) if connection.access_token else None,
#         refresh_token=decrypt_token(connection.refresh_token) if connection.refresh_token else None,
#         token_uri="https://oauth2.googleapis.com/token",
#         client_id=settings.GOOGLE_CLIENT_ID,
#         client_secret=settings.GOOGLE_CLIENT_SECRET,
#         scopes=(connection.scopes or "").split(),
#     )
#     credentials.expiry = _to_naive_utc(connection.token_expiry)

#     if not credentials.valid:
#         if not credentials.refresh_token:
#             connection.revoked_at = datetime.now(timezone.utc)
#             db.commit()
#             raise GoogleAuthorizationRequiredError(
#                 "Google Workspace authorization has expired and cannot be "
#                 "refreshed automatically. Please reconnect.",
#                 authorization_url=build_authorization_url(user_id),
#             )
#         try:
#             credentials.refresh(GoogleAuthRequest())
#         except Exception:
#             # Most commonly: the refresh token itself was revoked
#             # (user removed access in their Google Account settings).
#             connection.revoked_at = datetime.now(timezone.utc)
#             db.commit()
#             raise GoogleAuthorizationRequiredError(
#                 "Google Workspace authorization has expired or been "
#                 "revoked. Please reconnect.",
#                 authorization_url=build_authorization_url(user_id),
#             )

#         connection.access_token = encrypt_token(credentials.token)
#         connection.token_expiry = _to_aware_utc(credentials.expiry)
#         db.commit()

#     return credentials


# def disconnect(db: Session, user_id: int) -> bool:
#     """Idempotent, like logout in auth_service.py -- returns True if
#     an active connection was actually revoked, False if there was
#     nothing to do. Best-effort revokes the token with Google itself so
#     it stops showing up in the user's Google Account permissions page;
#     a failure there does not block disconnecting on our side."""
#     connection = get_connection(db, user_id)
#     if connection is None:
#         return False

#     if connection.refresh_token:
#         try:
#             http_requests.post(
#                 "https://oauth2.googleapis.com/revoke",
#                 params={"token": decrypt_token(connection.refresh_token)},
#                 headers={"content-type": "application/x-www-form-urlencoded"},
#                 timeout=10,
#             )
#         except http_requests.RequestException:
#             pass

#     connection.revoked_at = datetime.now(timezone.utc)
#     db.commit()
#     return True