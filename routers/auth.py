import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

from core.auth import create_access_token, hash_password, verify_password
from core.storage.users import (
    create_refresh_token,
    create_user,
    get_refresh_token,
    get_user_by_email,
    get_user_by_id,
    revoke_refresh_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])

_REFRESH_TOKEN_EXPIRE_DAYS = 30


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest) -> TokenResponse:
    if len(request.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")

    existing = await get_user_by_email(str(request.email))
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = await create_user(str(request.email), hash_password(request.password))

    access_token = create_access_token(user["id"], user["email"])
    refresh_token = secrets.token_hex(32)  # 64 hex chars
    expires_at = (
        datetime.now(timezone.utc) + timedelta(days=_REFRESH_TOKEN_EXPIRE_DAYS)
    ).isoformat()
    await create_refresh_token(refresh_token, user["id"], expires_at)

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    user = await get_user_by_email(str(request.email))
    if not user or not verify_password(request.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(user["id"], user["email"])
    refresh_token = secrets.token_hex(32)
    expires_at = (
        datetime.now(timezone.utc) + timedelta(days=_REFRESH_TOKEN_EXPIRE_DAYS)
    ).isoformat()
    await create_refresh_token(refresh_token, user["id"], expires_at)

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=RefreshResponse)
async def refresh(request: RefreshRequest) -> RefreshResponse:
    record = await get_refresh_token(request.refresh_token)
    if not record or record["revoked"]:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    expires_at = datetime.fromisoformat(record["expires_at"])
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = await get_user_by_id(record["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    return RefreshResponse(access_token=create_access_token(user["id"], user["email"]))


@router.post("/logout")
async def logout(request: LogoutRequest) -> dict:
    await revoke_refresh_token(request.refresh_token)
    return {"ok": True}
