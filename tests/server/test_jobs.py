
import pytest
from httpx import ASGITransport, AsyncClient

from classeval.config import load_settings
from classeval.pipeline import orchestrator
from classeval.schema import Report
from classeval.server.jobs import JobStore
from classeval.server.main import create_app


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
async def test_job_stub_flow_stages_in_order(app, settings, monkeypatch):
    """A job walks through the stages and ends with a valid report (pipeline faked, upload real)."""
    from classeval.schema import MediaInfo

    transport = ASGITransport(app=app)
    headers = {"X-Access-Code": "correct_token_123"}
    store = JobStore(settings.paths.jobs_dir)
    stage_names = ["ingest", "asr", "diarize", "audio", "video", "understand", "score", "report"]
    seen: list[str] = []

    def fake_execute(input_path, job_dir, settings, stages=None, job_id="x", store=None):
        for idx, stage in enumerate(stage_names):
            store.set_stage(job_id, stage, (idx + 1) / len(stage_names))
            seen.append(stage)
        media = MediaInfo(duration_s=30.0, has_audio=True, has_video=True, container="mp4", size_bytes=3)
        rep = Report(
            job_id=job_id,
            created_at="2026-10-22T00:00:00+00:00",
            media=media,
            language="en",
            language_confidence=0.9,
            grade="Insufficient evidence",
            confidence=0.5,
            coverage=0.0,
            summary="fake pipeline",
            degraded=False,
        )
        (job_dir / "report.json").write_text(rep.model_dump_json(), encoding="utf-8")
        (job_dir / "report.html").write_text("<!doctype html><html><body>Report</body></html>", encoding="utf-8")
        return rep

    monkeypatch.setattr(orchestrator, "execute_pipeline", fake_execute)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        init = await client.post("/api/uploads", json={"size": 3, "filename": "a.mp4"}, headers=headers)
        upload_id = init.json()["upload_id"]
        await client.put(f"/api/uploads/{upload_id}/chunks/0", content=b"abc", headers=headers)
        await client.post(f"/api/uploads/{upload_id}/complete", headers=headers)

        resp = await client.post("/api/jobs", json={"type": "upload", "upload_id": upload_id}, headers=headers)
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        assert len(job_id) >= 22

        resp = await client.get(f"/api/jobs/{job_id}", headers=headers)
        assert resp.json()["status"] == "queued"

        orchestrator.run(job_id, store, settings)
        assert seen == stage_names

        resp = await client.get(f"/api/jobs/{job_id}", headers=headers)
        job_data = resp.json()
        assert job_data["status"] == "done"
        assert job_data["stage"] == "done"
        assert job_data["progress"] == 1.0

        resp = await client.get(f"/api/jobs/{job_id}/report.json", headers=headers)
        assert resp.status_code == 200
        report = Report.model_validate(resp.json())
        assert report.job_id == job_id
        assert report.grade == "Insufficient evidence"

        resp = await client.get(f"/api/jobs/{job_id}/report.html", headers=headers)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "<html" in resp.text.lower()

    assert not (settings.paths.uploads_dir / upload_id).exists()  # the recording is deleted when the job ends


@pytest.mark.asyncio
async def test_job_ids_unguessable(app):
    transport = ASGITransport(app=app)
    headers = {"X-Access-Code": "correct_token_123"}
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # No job listing route exists
        resp = await client.get("/api/jobs", headers=headers)
        assert resp.status_code in (404, 405)


@pytest.mark.asyncio
async def test_delete_removes_job_dir(app, settings):
    transport = ASGITransport(app=app)
    headers = {"X-Access-Code": "correct_token_123"}
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/jobs",
            json={"type": "upload", "upload_id": "to_delete_1234567890123"},
            headers=headers,
        )
        job_id = resp.json()["job_id"]
        job_dir = settings.paths.jobs_dir / job_id
        assert job_dir.exists()

        resp = await client.delete(f"/api/jobs/{job_id}", headers=headers)
        assert resp.status_code == 200
        assert not job_dir.exists()

        resp = await client.get(f"/api/jobs/{job_id}", headers=headers)
        assert resp.status_code == 404
