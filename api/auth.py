from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from models.user import User
from schemas.auth import (
    UserRegister,
    UserResponse,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    AccessTokenResponse,
    LogoutRequest,
)
from services import auth_service
from services.auth_service import InvalidCredentialsError


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: UserRegister,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.register_user expects:
        # register_user(db, data: UserRegister)
        user = auth_service.register_user(
            db=db,
            data=request,
        )

    except auth_service.DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.authenticate_user expects:
        # authenticate_user(db, email, password)
        user = auth_service.authenticate_user(
            db=db,
            email=str(request.email).strip().lower(),
            password=request.password,
        )

    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
)
def refresh(
    request: RefreshRequest,
    db: Session = Depends(get_db),
):
    try:
        result = auth_service.refresh_access_token(
            db=db,
            raw_token=request.refresh_token,
        )

    except auth_service.InvalidRefreshTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    access_token, new_refresh_token, expires_in = result

    return AccessTokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
def logout(
    request: LogoutRequest,
    db: Session = Depends(get_db),
):
    auth_service.revoke_refresh_token(
        db=db,
        raw_token=request.refresh_token,
    )

    return None


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return current_user
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from models.user import User
from schemas.auth import (
    UserRegister,
    UserResponse,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    AccessTokenResponse,
    LogoutRequest,
)
from services import auth_service
from services.auth_service import InvalidCredentialsError


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: UserRegister,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.register_user expects:
        # register_user(db, data: UserRegister)
        user = auth_service.register_user(
            db=db,
            data=request,
        )

    except auth_service.DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.authenticate_user expects:
        # authenticate_user(db, email, password)
        user = auth_service.authenticate_user(
            db=db,
            email=str(request.email).strip().lower(),
            password=request.password,
        )

    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
)
def refresh(
    request: RefreshRequest,
    db: Session = Depends(get_db),
):
    try:
        result = auth_service.refresh_access_token(
            db=db,
            raw_token=request.refresh_token,
        )

    except auth_service.InvalidRefreshTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    access_token, new_refresh_token, expires_in = result

    return AccessTokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
def logout(
    request: LogoutRequest,
    db: Session = Depends(get_db),
):
    auth_service.revoke_refresh_token(
        db=db,
        raw_token=request.refresh_token,
    )

    return None


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return current_user
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from models.user import User
from schemas.auth import (
    UserRegister,
    UserResponse,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    AccessTokenResponse,
    LogoutRequest,
)
from services import auth_service
from services.auth_service import InvalidCredentialsError


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: UserRegister,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.register_user expects:
        # register_user(db, data: UserRegister)
        user = auth_service.register_user(
            db=db,
            data=request,
        )

    except auth_service.DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.authenticate_user expects:
        # authenticate_user(db, email, password)
        user = auth_service.authenticate_user(
            db=db,
            email=str(request.email).strip().lower(),
            password=request.password,
        )

    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
)
def refresh(
    request: RefreshRequest,
    db: Session = Depends(get_db),
):
    try:
        result = auth_service.refresh_access_token(
            db=db,
            raw_token=request.refresh_token,
        )

    except auth_service.InvalidRefreshTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    access_token, new_refresh_token, expires_in = result

    return AccessTokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
def logout(
    request: LogoutRequest,
    db: Session = Depends(get_db),
):
    auth_service.revoke_refresh_token(
        db=db,
        raw_token=request.refresh_token,
    )

    return None


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return current_user
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from models.user import User
from schemas.auth import (
    UserRegister,
    UserResponse,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    AccessTokenResponse,
    LogoutRequest,
)
from services import auth_service
from services.auth_service import InvalidCredentialsError


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: UserRegister,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.register_user expects:
        # register_user(db, data: UserRegister)
        user = auth_service.register_user(
            db=db,
            data=request,
        )

    except auth_service.DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.authenticate_user expects:
        # authenticate_user(db, email, password)
        user = auth_service.authenticate_user(
            db=db,
            email=str(request.email).strip().lower(),
            password=request.password,
        )

    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
)
def refresh(
    request: RefreshRequest,
    db: Session = Depends(get_db),
):
    try:
        result = auth_service.refresh_access_token(
            db=db,
            raw_token=request.refresh_token,
        )

    except auth_service.InvalidRefreshTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    access_token, new_refresh_token, expires_in = result

    return AccessTokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
def logout(
    request: LogoutRequest,
    db: Session = Depends(get_db),
):
    auth_service.revoke_refresh_token(
        db=db,
        raw_token=request.refresh_token,
    )

    return None


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return current_user
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from models.user import User
from schemas.auth import (
    UserRegister,
    UserResponse,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    AccessTokenResponse,
    LogoutRequest,
)
from services import auth_service
from services.auth_service import InvalidCredentialsError


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: UserRegister,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.register_user expects:
        # register_user(db, data: UserRegister)
        user = auth_service.register_user(
            db=db,
            data=request,
        )

    except auth_service.DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.authenticate_user expects:
        # authenticate_user(db, email, password)
        user = auth_service.authenticate_user(
            db=db,
            email=str(request.email).strip().lower(),
            password=request.password,
        )

    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
)
def refresh(
    request: RefreshRequest,
    db: Session = Depends(get_db),
):
    try:
        result = auth_service.refresh_access_token(
            db=db,
            raw_token=request.refresh_token,
        )

    except auth_service.InvalidRefreshTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    access_token, new_refresh_token, expires_in = result

    return AccessTokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
def logout(
    request: LogoutRequest,
    db: Session = Depends(get_db),
):
    auth_service.revoke_refresh_token(
        db=db,
        raw_token=request.refresh_token,
    )

    return None


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return current_user
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from models.user import User
from schemas.auth import (
    UserRegister,
    UserResponse,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    AccessTokenResponse,
    LogoutRequest,
)
from services import auth_service
from services.auth_service import InvalidCredentialsError


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: UserRegister,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.register_user expects:
        # register_user(db, data: UserRegister)
        user = auth_service.register_user(
            db=db,
            data=request,
        )

    except auth_service.DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.authenticate_user expects:
        # authenticate_user(db, email, password)
        user = auth_service.authenticate_user(
            db=db,
            email=str(request.email).strip().lower(),
            password=request.password,
        )

    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
)
def refresh(
    request: RefreshRequest,
    db: Session = Depends(get_db),
):
    try:
        result = auth_service.refresh_access_token(
            db=db,
            raw_token=request.refresh_token,
        )

    except auth_service.InvalidRefreshTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    access_token, new_refresh_token, expires_in = result

    return AccessTokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
def logout(
    request: LogoutRequest,
    db: Session = Depends(get_db),
):
    auth_service.revoke_refresh_token(
        db=db,
        raw_token=request.refresh_token,
    )

    return None


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return current_user
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from models.user import User
from schemas.auth import (
    UserRegister,
    UserResponse,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    AccessTokenResponse,
    LogoutRequest,
)
from services import auth_service
from services.auth_service import InvalidCredentialsError


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: UserRegister,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.register_user expects:
        # register_user(db, data: UserRegister)
        user = auth_service.register_user(
            db=db,
            data=request,
        )

    except auth_service.DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.authenticate_user expects:
        # authenticate_user(db, email, password)
        user = auth_service.authenticate_user(
            db=db,
            email=str(request.email).strip().lower(),
            password=request.password,
        )

    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
)
def refresh(
    request: RefreshRequest,
    db: Session = Depends(get_db),
):
    try:
        result = auth_service.refresh_access_token(
            db=db,
            raw_token=request.refresh_token,
        )

    except auth_service.InvalidRefreshTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    access_token, new_refresh_token, expires_in = result

    return AccessTokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
def logout(
    request: LogoutRequest,
    db: Session = Depends(get_db),
):
    auth_service.revoke_refresh_token(
        db=db,
        raw_token=request.refresh_token,
    )

    return None


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return current_user
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from models.user import User
from schemas.auth import (
    UserRegister,
    UserResponse,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    AccessTokenResponse,
    LogoutRequest,
)
from services import auth_service
from services.auth_service import InvalidCredentialsError


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: UserRegister,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.register_user expects:
        # register_user(db, data: UserRegister)
        user = auth_service.register_user(
            db=db,
            data=request,
        )

    except auth_service.DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    try:
        # auth_service.authenticate_user expects:
        # authenticate_user(db, email, password)
        user = auth_service.authenticate_user(
            db=db,
            email=str(request.email).strip().lower(),
            password=request.password,
        )

    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token, refresh_token, expires_in = (
        auth_service.create_tokens_for_user(
            db=db,
            user=user,
        )
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
)
def refresh(
    request: RefreshRequest,
    db: Session = Depends(get_db),
):
    try:
        result = auth_service.refresh_access_token(
            db=db,
            raw_token=request.refresh_token,
        )

    except auth_service.InvalidRefreshTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    access_token, new_refresh_token, expires_in = result

    return AccessTokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
def logout(
    request: LogoutRequest,
    db: Session = Depends(get_db),
):
    auth_service.revoke_refresh_token(
        db=db,
        raw_token=request.refresh_token,
    )

    return None


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return current_user
