"""Transaction route module placeholder."""

from fastapi import APIRouter

router = APIRouter(prefix="/v1/transactions", tags=["transactions"])


@router.get("/ping")
async def ping_transactions() -> dict[str, str]:
    return {"status": "ok"}
