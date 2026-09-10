
"""
Phase 5A -- Authentication API.

Responsibilities:
- Register users
- Login using JSON credentials
- Login using OAuth2 form credentials for Swagger
- Refresh and rotate opaque refresh tokens
- Logout / revoke refresh tokens

Token creation and persistence are delegated to auth_service.py.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from core.database import get_db
from schemas.auth import (
    LoginRequest,
    RefreshRequest,
    TokenResponse,
    UserRegister,
    AccessTokenResponse,
    LogoutRequest,
)
from services.auth_service import (
    DuplicateEmailError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
    authenticate_user,
    create_tokens_for_user,
    refresh_access_token,
    register_user,
    revoke_refresh_token,
)


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: UserRegister,
    db: Session = Depends(get_db),
):
    """
    Register a new platform user.
    """

    try:
        user = register_user(
            db=db,
            data=request,
        )

    except DuplicateEmailError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Unable to create account with the supplied information",
        )

    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "is_active": user.is_active,
        "created_at": user.created_at,
    }


# ---------------------------------------------------------------------------
# Internal token-response helper
# ---------------------------------------------------------------------------

def _build_token_response(
    db: Session,
    user,
) -> TokenResponse:
    """
    Issue both access and refresh tokens using the Phase 5A service.

    create_tokens_for_user() is the single source of truth for token
    creation and refresh-token persistence.
    """

    (
        access_token,
        refresh_token,
        expires_in,
    ) = create_tokens_for_user(
        db=db,
        user=user,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


# ---------------------------------------------------------------------------
# Normal application login
# ---------------------------------------------------------------------------

@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    """
    Normal application login.

    JSON body:

    {
        "email": "user@example.com",
        "password": "password"
    }
    """

    try:
        user = authenticate_user(
            db=db,
            email=request.email,
            password=request.password,
        )

    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    return _build_token_response(
        db=db,
        user=user,
    )


# ---------------------------------------------------------------------------
# Swagger OAuth2 login
# ---------------------------------------------------------------------------

@router.post(
    "/login/oauth2",
    response_model=TokenResponse,
)
def login_oauth2(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    OAuth2 Password Flow endpoint used by Swagger UI.

    Swagger sends:

        username=<email>
        password=<password>

    as application/x-www-form-urlencoded.
    """

    try:
        user = authenticate_user(
            db=db,
            email=form_data.username,
            password=form_data.password,
        )

    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    return _build_token_response(
        db=db,
        user=user,
    )


# ---------------------------------------------------------------------------
# Refresh token rotation
# ---------------------------------------------------------------------------

@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
)
def refresh_token(
    request: RefreshRequest,
    db: Session = Depends(get_db),
):
    """
    Validate and rotate an opaque refresh token.

    The submitted refresh token is revoked and a new refresh token
    is returned.
    """

    try:
        (
            access_token,
            new_refresh_token,
            expires_in,
        ) = refresh_access_token(
            db=db,
            raw_token=request.refresh_token,
        )

    except InvalidRefreshTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    return AccessTokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------

@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
def logout(
    request: LogoutRequest,
    db: Session = Depends(get_db),
):
    """
    Revoke the supplied refresh token.

    Logout is deliberately idempotent. The endpoint succeeds even if
    the token was already revoked or does not exist.
    """

    revoke_refresh_token(
        db=db,
        raw_token=request.refresh_token,
    )

    return None

