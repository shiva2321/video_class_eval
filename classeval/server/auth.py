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


class FailureTracker:
    """Per-client lockout after repeated wrong or missing access codes (brute-force guard).

    A locked client is refused even with the correct code, so the lockout cannot be used as a guessing oracle.
    """

    def __init__(
        self,
        max_failures: int = 10,
        window_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ):
        self.max_failures = max_failures
        self.window_s = window_s
        self.max_keys = max_keys
        self._clock = clock
        self._failures: dict[str, list[float]] = defaultdict(list)

    def _gc(self) -> None:
        """Bound memory without wiping active lockouts: drop expired clients, then evict the client with the
        oldest last failure, preferring clients that are not locked out (locked ones are evicted last)."""
        cutoff = self._clock() - self.window_s
        self._failures = defaultdict(list, {k: v for k, v in self._failures.items() if v and v[-1] >= cutoff})
        while len(self._failures) >= self.max_keys:
            unlocked = [k for k, v in self._failures.items() if len(v) < self.max_failures]
            victim = min(unlocked or list(self._failures), key=lambda k: self._failures[k][-1])
            del self._failures[victim]

    def _recent(self, key: str) -> list[float]:
        cutoff = self._clock() - self.window_s
        recent = [t for t in self._failures[key] if t >= cutoff]
        self._failures[key] = recent
        return recent

    def is_locked(self, key: str) -> bool:
        return len(self._recent(key)) >= self.max_failures

    def record(self, key: str) -> None:
        if len(self._failures) >= self.max_keys:
            self._gc()
        self._recent(key).append(self._clock())


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


def is_shared_client(request: Request) -> bool:
    """True when visitors cannot be told apart: the peer is loopback and no proxy-added address is present.
    Locking such a key out would lock out everybody, so the lockout does not apply to it. Accepted trade-off:
    guessing is then bounded only by the request rate limit and the code's length (random, at least ten
    characters), which is far safer than letting one visitor lock out every evaluator. When the tunnel forwards
    the client address (checked via /healthz) remote visitors never use this key."""
    host = request.client.host if request.client else "127.0.0.1"
    return host in ("127.0.0.1", "::1") and not request.headers.get("x-forwarded-for")


def get_auth_dependency(
    settings: Settings,
    limiter: RateLimiter | None = None,
    failures: FailureTracker | None = None,
) -> Callable:
    expected_code = settings.access_code.get_secret_value()
    active_limiter = limiter if limiter is not None else rate_limiter
    active_failures = failures if failures is not None else FailureTracker()

    async def verify_auth(
        request: Request,
        header_code: str | None = Security(api_key_header),
    ) -> bool:
        ip = client_ip(request)
        active_limiter.check(ip)

        lockout_applies = not is_shared_client(request)
        if lockout_applies and active_failures.is_locked(ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed attempts. Try again in a few minutes.",
                headers={"Retry-After": str(int(active_failures.window_s))},
            )

        provided_code = header_code or request.cookies.get("access")
        if not provided_code:  # not a guess: never counts towards a lockout
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        if not hmac.compare_digest(provided_code.encode("utf-8"), expected_code.encode("utf-8")):
            if lockout_applies:
                active_failures.record(ip)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            )
        return True

    return verify_auth
