"""Authentication middleware placeholder."""

from __future__ import annotations

from fastapi import HTTPException, Request


async def require_authentication(request: Request) -> None:
    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
