# from typing import List, Optional

# from pydantic import BaseModel


# class AuthorizationUrlResponse(BaseModel):
#     authorization_url: str


# class GoogleStatusResponse(BaseModel):
#     """Exactly what the frontend is allowed to know -- never tokens,
#     never scopes verbatim (that's an implementation detail), just
#     connection state and which services are usable."""
#     connected: bool
#     provider: str = "google"
#     email: Optional[str] = None
#     services: List[str] = []


# class AuthorizationRequiredResponse(BaseModel):
#     """Returned by the Manager/tool layer (not this router directly)
#     when a Google-dependent request can't proceed. Matches the shape
#     specified for this phase so the frontend can branch on `status`
#     and, when present, immediately redirect the user to
#     authorization_url."""
#     status: str = "authorization_required"
#     provider: str = "google"
#     connection: str = "google_workspace"
#     message: str
#     authorization_url: Optional[str] = None