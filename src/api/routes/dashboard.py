"""Dashboard route module placeholder."""

from fastapi import APIRouter

router = APIRouter(prefix="/v1/dashboard", tags=["dashboard"])


@router.get("/ping")
async def ping_dashboard() -> dict[str, str]:
    return {"status": "ok"}
