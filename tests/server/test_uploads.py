import hashlib
import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from classeval.config import load_settings
from classeval.server.main import create_app
from classeval.server.uploads import UploadStore


def _store(settings) -> UploadStore:
    return UploadStore(settings.paths.uploads_dir, settings.limits.max_upload_bytes, settings.limits.chunk_bytes)


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ACCESS_CODE", "correct_token_123")
    s = load_settings("config.yaml", env_file=None)
    s.paths.uploads_dir = tmp_path / "uploads"
    s.paths.jobs_dir = tmp_path / "jobs"
    s.paths.uploads_dir.mkdir(parents=True, exist_ok=True)
    s.paths.jobs_dir.mkdir(parents=True, exist_ok=True)
    return s


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.mark.asyncio
async def test_upload_roundtrip_out_of_order_and_duplicate_chunks(app, settings):
    transport = ASGITransport(app=app)
    headers = {"X-Access-Code": "correct_token_123"}

    # 20 MB random bytes in 8 MiB chunks -> 3 chunks: 8 MB, 8 MB, 4 MB
    total_size = 20 * 1024 * 1024
    chunk_size = 8 * 1024 * 1024
    data = os.urandom(total_size)
    source_sha256 = hashlib.sha256(data).hexdigest()

    chunks = [
        data[0:chunk_size],
        data[chunk_size : 2 * chunk_size],
        data[2 * chunk_size : total_size],
    ]

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Initialize upload
        resp = await client.post(
            "/api/uploads",
            json={"size": total_size, "filename": "sample.mp4"},
            headers=headers,
        )
        assert resp.status_code == 200
        upload_id = resp.json()["upload_id"]

        # 2. Put chunks in order 2, 0, 1, 1 (duplicate chunk 1)
        for idx in [2, 0, 1, 1]:
            resp = await client.put(
                f"/api/uploads/{upload_id}/chunks/{idx}",
                content=chunks[idx],
                headers={"Content-Type": "application/octet-stream", **headers},
            )
            assert resp.status_code == 200, f"Put chunk {idx} failed: {resp.text}"

        # 3. Complete upload
        resp = await client.post(f"/api/uploads/{upload_id}/complete", headers=headers)
        assert resp.status_code == 200
        completed_path = _store(settings).final_path(upload_id)

        # Verify completed file hash equals source
        completed_bytes = Path(completed_path).read_bytes()
        assert hashlib.sha256(completed_bytes).hexdigest() == source_sha256


@pytest.mark.asyncio
async def test_complete_with_missing_chunk_409_lists_indices(app, settings):
    transport = ASGITransport(app=app)
    headers = {"X-Access-Code": "correct_token_123"}
    total_size = 20 * 1024 * 1024
    chunk_size = 8 * 1024 * 1024
    data = os.urandom(total_size)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/uploads",
            json={"size": total_size, "filename": "sample.mp4"},
            headers=headers,
        )
        upload_id = resp.json()["upload_id"]

        # Send chunk 0 and 2, missing chunk 1
        await client.put(f"/api/uploads/{upload_id}/chunks/0", content=data[:chunk_size], headers=headers)
        await client.put(f"/api/uploads/{upload_id}/chunks/2", content=data[2 * chunk_size :], headers=headers)

        resp = await client.post(f"/api/uploads/{upload_id}/complete", headers=headers)
        assert resp.status_code == 409
        body = resp.json()
        assert body["missing"] == [1]


@pytest.mark.asyncio
async def test_init_over_limit_413(app, settings):
    transport = ASGITransport(app=app)
    headers = {"X-Access-Code": "correct_token_123"}
    huge_size = settings.limits.max_upload_bytes + 1

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/uploads",
            json={"size": huge_size, "filename": "sample.mp4"},
            headers=headers,
        )
        assert resp.status_code == 413


@pytest.mark.asyncio
async def test_filename_path_traversal_ignored(app, settings):
    transport = ASGITransport(app=app)
    headers = {"X-Access-Code": "correct_token_123"}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/uploads",
            json={"size": 100, "filename": "..\\..\\evil.mp4"},
            headers=headers,
        )
        assert resp.status_code == 200
        upload_id = resp.json()["upload_id"]

        await client.put(f"/api/uploads/{upload_id}/chunks/0", content=b"x" * 100, headers=headers)
        resp = await client.post(f"/api/uploads/{upload_id}/complete", headers=headers)
        assert resp.status_code == 200
        completed_path = _store(settings).final_path(upload_id)

        # Must reside strictly within uploads_dir
        assert str(settings.paths.uploads_dir.resolve()) in str(completed_path)
        assert "evil.mp4" not in os.path.dirname(completed_path)
