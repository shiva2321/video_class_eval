from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from classeval.config import load_settings
from classeval.server.auth import FailureTracker
from classeval.server.main import create_app

CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
GOOD = {"X-Access-Code": "correct_token_123"}


def _client(monkeypatch, failures=None):
    monkeypatch.setenv("ACCESS_CODE", "correct_token_123")
    app = create_app(load_settings(CONFIG, env_file=None), failures=failures)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _from(ip: str, code: str = "correct_token_123") -> dict[str, str]:
    return {"X-Access-Code": code, "X-Forwarded-For": f"203.0.113.9, {ip}"}


@pytest.mark.asyncio
async def test_lockout_after_ten_wrong_codes_even_for_the_right_code(monkeypatch):
    async with _client(monkeypatch) as client:
        for _ in range(10):
            resp = await client.get("/api/jobs/nonexistent_id_000000", headers=_from("198.51.100.7", "wrong-code-xyz"))
            assert resp.status_code == 401
        locked = await client.get("/api/jobs/nonexistent_id_000000", headers=_from("198.51.100.7"))
        assert locked.status_code == 429
        assert "Retry-After" in locked.headers


@pytest.mark.asyncio
async def test_lockout_is_per_client(monkeypatch):
    async with _client(monkeypatch) as client:
        for _ in range(10):
            await client.get("/api/jobs/nonexistent_id_000000", headers=_from("198.51.100.7", "wrong-code-xyz"))
        other = await client.get("/api/jobs/nonexistent_id_000000", headers=_from("198.51.100.8"))
        assert other.status_code == 404  # authenticated, job simply does not exist


@pytest.mark.asyncio
async def test_missing_code_counts_as_a_failure(monkeypatch):
    async with _client(monkeypatch) as client:
        for _ in range(10):
            resp = await client.get("/api/jobs/nonexistent_id_000000", headers={"X-Forwarded-For": "198.51.100.9"})
            assert resp.status_code == 401
        locked = await client.get("/api/jobs/nonexistent_id_000000", headers={**GOOD, "X-Forwarded-For": "198.51.100.9"})
        assert locked.status_code == 429


def test_lockout_expires_after_the_window():
    now = [0.0]
    tracker = FailureTracker(max_failures=3, window_s=300.0, clock=lambda: now[0])
    for _ in range(3):
        tracker.record("ip")
    assert tracker.is_locked("ip")
    now[0] = 301.0
    assert not tracker.is_locked("ip")
