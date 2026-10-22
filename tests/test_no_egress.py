import socket
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from classeval.config import load_settings
from classeval.pipeline import orchestrator
from classeval.schema import MediaInfo, Report
from classeval.server.jobs import JobStore
from classeval.server.main import create_app
from classeval.worker import run_worker


@pytest.fixture(autouse=True)
def guard_network_egress(monkeypatch):
    orig_connect = socket.socket.connect

    def patched_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"Outbound network egress attempted to: {address}")
        return orig_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", patched_connect)


def test_socket_patch_rejects_external_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(AssertionError, match="Outbound network egress attempted to"):
        s.connect(("8.8.8.8", 53))
    s.close()


@pytest.mark.asyncio
async def test_no_egress_end_to_end(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "egress_test_token_123")
    settings = load_settings("config.yaml", env_file=None)
    settings.paths.uploads_dir = tmp_path / "uploads"
    settings.paths.jobs_dir = tmp_path / "jobs"
    settings.paths.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.jobs_dir.mkdir(parents=True, exist_ok=True)

    app = create_app(settings)
    transport = ASGITransport(app=app)
    headers = {"X-Access-Code": "egress_test_token_123"}
    store = JobStore(settings.paths.jobs_dir)

    def fake_execute(input_path, job_dir, settings, stages=None, job_id="test", store=None):
        media = MediaInfo(duration_s=30.0, has_audio=True, has_video=True, container="mp4", size_bytes=3)
        rep = Report(
            job_id=job_id,
            created_at="2026-10-22T00:00:00+00:00",
            media=media,
            language="en",
            language_confidence=0.9,
            grade="Good",
            confidence=0.9,
            coverage=1.0,
            summary="Fake canned report",
            degraded=False,
        )
        (job_dir / "report.json").write_text(rep.model_dump_json(), encoding="utf-8")
        (job_dir / "report.html").write_text("<!doctype html><html><body>Report</body></html>", encoding="utf-8")
        return rep

    monkeypatch.setattr(orchestrator, "execute_pipeline", fake_execute)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Init upload
        init_resp = await client.post(
            "/api/uploads",
            json={"size": 4, "filename": "lesson.mp4"},
            headers=headers,
        )
        assert init_resp.status_code == 200
        upload_id = init_resp.json()["upload_id"]

        # 2. Put chunk
        chunk_resp = await client.put(
            f"/api/uploads/{upload_id}/chunks/0",
            content=b"test",
            headers=headers,
        )
        assert chunk_resp.status_code == 200

        # 3. Complete upload
        comp_resp = await client.post(
            f"/api/uploads/{upload_id}/complete",
            headers=headers,
        )
        assert comp_resp.status_code == 200

        # 4. Create job
        job_resp = await client.post(
            "/api/jobs",
            json={"type": "upload", "upload_id": upload_id},
            headers=headers,
        )
        assert job_resp.status_code == 200
        job_id = job_resp.json()["job_id"]

        # Run worker loop in-process
        run_worker(settings, store, once=True)

        # Check job completion
        final_job_resp = await client.get(f"/api/jobs/{job_id}", headers=headers)
        assert final_job_resp.status_code == 200
        assert final_job_resp.json()["status"] == "done"
