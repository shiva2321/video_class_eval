import pytest
from httpx import ASGITransport, AsyncClient

from classeval.config import load_settings
from classeval.server.auth import RateLimiter
from classeval.server.main import create_app


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "correct_token_123")
    settings = load_settings("config.yaml", env_file=None)
    return create_app(settings)


@pytest.mark.asyncio
async def test_healthz_open(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_missing_or_wrong_code_401_on_all_api_routes(app):
    transport = ASGITransport(app=app)
    routes = [
        ("POST", "/api/uploads", {"size": 100, "filename": "test.mp4"}),
        ("PUT", "/api/uploads/test_id/chunks/0", b"data"),
        ("POST", "/api/uploads/test_id/complete", {}),
        ("POST", "/api/jobs", {"type": "upload", "upload_id": "test_id"}),
        ("GET", "/api/jobs/test_id", None),
        ("GET", "/api/jobs/test_id/events", None),
        ("GET", "/api/jobs/test_id/report.json", None),
        ("GET", "/api/jobs/test_id/report.html", None),
        ("DELETE", "/api/jobs/test_id", None),
    ]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for method, path, body in routes:
            # 1. No auth header or cookie
            kw = {}
            if isinstance(body, bytes):
                kw["content"] = body
            elif isinstance(body, dict):
                kw["json"] = body
            resp = await client.request(method, path, **kw)
            assert resp.status_code == 401, f"{method} {path} without auth expected 401, got {resp.status_code}"

            # 2. Wrong access code
            resp = await client.request(method, path, headers={"X-Access-Code": "wrong_code"}, **kw)
            assert resp.status_code == 401, f"{method} {path} with wrong auth expected 401, got {resp.status_code}"

            # 3. Valid access code in cookie
            resp = await client.request(method, path, headers={"Cookie": "access=correct_token_123"}, **kw)
            assert resp.status_code != 401, f"{method} {path} with cookie auth gave 401"


@pytest.mark.asyncio
async def test_rate_limit_429(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "correct_token_123")
    settings = load_settings("config.yaml", env_file=None)
    app = create_app(settings, limiter=RateLimiter(max_requests=60, window_s=60.0))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"X-Access-Code": "correct_token_123"}
        for _ in range(60):
            resp = await client.get("/api/jobs/nonexistent_id", headers=headers)
            assert resp.status_code != 429
        # 61st request in a minute from the same client returns 429
        resp = await client.get("/api/jobs/nonexistent_id", headers=headers)
        assert resp.status_code == 429
