from fastapi import APIRouter, Depends
from pydantic import BaseModel

from core.auth import get_current_user
from core.storage.users import upsert_push_token

router = APIRouter(prefix="/users", tags=["users"])


class PushTokenRequest(BaseModel):
    token: str


@router.post("/push-token")
async def register_push_token(
    request: PushTokenRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    await upsert_push_token(current_user["id"], request.token)
    return {"ok": True}
