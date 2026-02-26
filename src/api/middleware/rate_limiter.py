"""Rate limiting middleware placeholder."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class InMemoryRateLimiter:
    def __init__(self, max_requests: int = 120, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def check(self, request: Request) -> None:
        key = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = self._hits[key]
        while bucket and now - bucket[0] > self.window_seconds:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        bucket.append(now)
