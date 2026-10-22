from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from classeval.config import load_settings
from classeval.server.auth import RateLimiter
from classeval.server.main import create_app

HEADERS = {"X-Access-Code": "correct_token_123"}
CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


def _client(monkeypatch, tmp_path, limiter=None):
    monkeypatch.setenv("ACCESS_CODE", "correct_token_123")
    monkeypatch.chdir(tmp_path)  # jobs/ and uploads/ are relative to the working directory
    settings = load_settings(CONFIG, env_file=None)
    app = create_app(settings, limiter=limiter)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), settings


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["%2e%2e", "..%5C..", "..%2F..", "a", "x" * 70])
async def test_delete_never_leaves_the_jobs_dir(monkeypatch, tmp_path, bad_id):
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("keep me")
    (tmp_path.parent / "sentinel_parent.txt").write_text("keep me too")
    client, _ = _client(monkeypatch, tmp_path)
    async with client:
        resp = await client.delete(f"/api/jobs/{bad_id}", headers=HEADERS)
    assert resp.status_code == 404
    assert sentinel.exists()
    assert (tmp_path.parent / "sentinel_parent.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["%2e%2e", "..%5C..", "short"])
async def test_upload_routes_reject_malformed_ids(monkeypatch, tmp_path, bad_id):
    client, _ = _client(monkeypatch, tmp_path)
    async with client:
        put = await client.put(f"/api/uploads/{bad_id}/chunks/0", content=b"x", headers=HEADERS)
        done = await client.post(f"/api/uploads/{bad_id}/complete", headers=HEADERS)
    assert put.status_code == 404
    assert done.status_code == 404
    assert not list(tmp_path.rglob("chunk_*"))


@pytest.mark.asyncio
async def test_chunk_index_and_size_are_bounded(monkeypatch, tmp_path):
    client, settings = _client(monkeypatch, tmp_path)
    async with client:
        init = await client.post("/api/uploads", json={"size": 100, "filename": "a.mp4"}, headers=HEADERS)
        upload_id = init.json()["upload_id"]
        out_of_range = await client.put(f"/api/uploads/{upload_id}/chunks/5", content=b"x", headers=HEADERS)
        too_big = await client.put(
            f"/api/uploads/{upload_id}/chunks/0",
            content=b"x" * (settings.limits.chunk_bytes + 1),
            headers=HEADERS,
        )
    assert out_of_range.status_code == 400
    assert too_big.status_code == 413


@pytest.mark.asyncio
async def test_complete_does_not_leak_server_paths(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)
    async with client:
        init = await client.post("/api/uploads", json={"size": 3, "filename": "a.mp4"}, headers=HEADERS)
        upload_id = init.json()["upload_id"]
        await client.put(f"/api/uploads/{upload_id}/chunks/0", content=b"abc", headers=HEADERS)
        done = await client.post(f"/api/uploads/{upload_id}/complete", headers=HEADERS)
    assert done.status_code == 200
    assert str(tmp_path) not in done.text and "path" not in done.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("hostile_name", ["meta.json", "chunk_0.part", "upload.tmp", "..", "..\\..\\x.mp4", "a.mp4:stream"])
async def test_client_filename_is_never_used_on_disk(monkeypatch, tmp_path, hostile_name):
    client, _ = _client(monkeypatch, tmp_path)
    async with client:
        init = await client.post("/api/uploads", json={"size": 3, "filename": hostile_name}, headers=HEADERS)
        upload_id = init.json()["upload_id"]
        await client.put(f"/api/uploads/{upload_id}/chunks/0", content=b"abc", headers=HEADERS)
        done = await client.post(f"/api/uploads/{upload_id}/complete", headers=HEADERS)
    assert done.status_code == 200
    up_dir = tmp_path / "uploads" / upload_id
    assert (up_dir / "upload.bin").read_bytes() == b"abc"
    assert "total_chunks" in (up_dir / "meta.json").read_text(encoding="utf-8")  # metadata not overwritten
    assert sorted(p.name for p in up_dir.iterdir()) == ["meta.json", "upload.bin"]


@pytest.mark.asyncio
async def test_completed_upload_is_closed_and_complete_is_idempotent(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)
    async with client:
        init = await client.post("/api/uploads", json={"size": 3, "filename": "a.mp4"}, headers=HEADERS)
        upload_id = init.json()["upload_id"]
        await client.put(f"/api/uploads/{upload_id}/chunks/0", content=b"abc", headers=HEADERS)
        first = await client.post(f"/api/uploads/{upload_id}/complete", headers=HEADERS)
        again = await client.post(f"/api/uploads/{upload_id}/complete", headers=HEADERS)
        late = await client.put(f"/api/uploads/{upload_id}/chunks/0", content=b"zzz", headers=HEADERS)
    assert first.status_code == 200 and again.status_code == 200
    assert late.status_code == 409
    assert (tmp_path / "uploads" / upload_id / "upload.bin").read_bytes() == b"abc"


@pytest.mark.asyncio
async def test_rate_limit_uses_last_forwarded_entry_behind_local_proxy(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path, limiter=RateLimiter(max_requests=2, window_s=60.0))
    async with client:
        # Same real client (last entry); the forged first entry must not buy extra requests.
        for forged in ("1.1.1.1", "2.2.2.2"):
            r = await client.get("/api/jobs/nonexistent_id_000000", headers={**HEADERS, "X-Forwarded-For": f"{forged}, 9.9.9.9"})
            assert r.status_code != 429
        blocked = await client.get("/api/jobs/nonexistent_id_000000", headers={**HEADERS, "X-Forwarded-For": "3.3.3.3, 9.9.9.9"})
        other = await client.get("/api/jobs/nonexistent_id_000000", headers={**HEADERS, "X-Forwarded-For": "3.3.3.3, 8.8.8.8"})
    assert blocked.status_code == 429
    assert other.status_code != 429
