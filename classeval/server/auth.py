import hmac
import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from classeval.config import Settings

api_key_header = APIKeyHeader(name="X-Access-Code", auto_error=False)


class RateLimiter:
    """Sliding-window rate limiter per client IP (60 requests per 60 seconds)."""

    def __init__(self, max_requests: int = 60, window_s: float = 60.0):
        self.max_requests = max_requests
        self.window_s = window_s
        self.requests: dict[str, list[float]] = defaultdict(list)

    def check(self, client_ip: str) -> None:
        now = time.monotonic()
        timestamps = self.requests[client_ip]
        # Drop timestamps older than the window
        cutoff = now - self.window_s
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)

        if len(timestamps) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        timestamps.append(now)


rate_limiter = RateLimiter()


def client_ip(request: Request) -> str:
    """Peer address; behind a local reverse proxy (Tailscale Funnel) use the LAST X-Forwarded-For entry,
    which is the one the proxy itself appended (earlier entries can be forged by the caller)."""
    host = request.client.host if request.client else "127.0.0.1"
    if host in ("127.0.0.1", "::1"):
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[-1].strip() or host
    return host


def get_auth_dependency(settings: Settings, limiter: RateLimiter | None = None) -> Callable:
    expected_code = settings.access_code.get_secret_value()
    active_limiter = limiter if limiter is not None else rate_limiter

    async def verify_auth(
        request: Request,
        header_code: str | None = Security(api_key_header),
    ) -> bool:
        # Check rate limit first
        active_limiter.check(client_ip(request))

        # Extract code from header or cookie
        provided_code = header_code or request.cookies.get("access")
        if not provided_code:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            )

        if not hmac.compare_digest(provided_code.encode("utf-8"), expected_code.encode("utf-8")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            )
        return True

    return verify_auth
