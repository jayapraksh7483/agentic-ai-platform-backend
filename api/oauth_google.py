# from fastapi import APIRouter, Depends, HTTPException, Query, status
# from sqlalchemy.orm import Session
# from typing import Optional

# from core.database import get_db
# from core.security import get_current_user
# from models.user import User
# from schemas.oauth import AuthorizationUrlResponse, GoogleStatusResponse
# from services import google_oauth_service

# router = APIRouter(prefix="/api/oauth/google", tags=["Google OAuth"])


# @router.get("/connect", response_model=AuthorizationUrlResponse)
# def connect(current_user: User = Depends(get_current_user)):
#     """
#     Authenticated endpoint -- the ONLY entry point that starts a Google
#     OAuth flow. Never called automatically at register/login (see
#     module docstring in services/google_oauth_service.py for why the
#     `state` this issues is safe to trust on the way back).
#     """
#     try:
#         url = google_oauth_service.build_authorization_url(current_user.id)
#     except google_oauth_service.GoogleOAuthError as e:
#         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
#     return AuthorizationUrlResponse(authorization_url=url)


# @router.get("/callback")
# def callback(
#     code: Optional[str] = Query(None),
#     state: Optional[str] = Query(None),
#     error: Optional[str] = Query(None),
#     db: Session = Depends(get_db),
# ):
#     """
#     Hit directly by the browser as a redirect FROM Google -- there is
#     no Authorization header here, which is exactly why `state` (not a
#     JWT bearer token) is what identifies the user (see
#     services/google_oauth_service.py). Do not add
#     Depends(get_current_user) to this endpoint; it will never have a
#     token to check.
#     """
#     try:
#         connection = google_oauth_service.handle_oauth_callback(db, code=code, state=state, error=error)
#     except google_oauth_service.InvalidOAuthStateError:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired OAuth state")
#     except google_oauth_service.GoogleOAuthError as e:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

#     # A plain confirmation body is enough for Part 2 -- if/when the
#     # frontend wants this callback to land on a specific app URL
#     # (e.g. to close a popup window), swap this for a
#     # RedirectResponse(url=f"{FRONTEND_URL}/settings?google=connected")
#     # once that URL is decided; nothing else in this file would change.
#     return {"connected": True, "email": connection.email}


# @router.get("/status", response_model=GoogleStatusResponse)
# def status_endpoint(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db),
# ):
#     return google_oauth_service.get_status(db, current_user.id)


# @router.delete("/disconnect", status_code=status.HTTP_204_NO_CONTENT)
# def disconnect(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db),
# ):
#     google_oauth_service.disconnect(db, current_user.id)
#     return None