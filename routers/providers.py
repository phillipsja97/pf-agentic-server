from fastapi import APIRouter, Depends

from config import settings
from core.auth import get_current_user

router = APIRouter(tags=["providers"])


@router.get("/providers")
async def get_providers(current_user: dict = Depends(get_current_user)) -> list[dict]:
    providers = []
    if settings.anthropic_api_key:
        providers.append({
            "name": "anthropic",
            "label": "Anthropic",
            "models": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        })
    if settings.openai_api_key:
        providers.append({
            "name": "openai",
            "label": "OpenAI",
            "models": ["gpt-4o", "gpt-4o-mini"],
        })
    return providers
