"""Agent route module placeholder."""

from fastapi import APIRouter

router = APIRouter(prefix="/v1/agent", tags=["agent"])


@router.get("/ping")
async def ping_agent() -> dict[str, str]:
    return {"status": "ok"}
