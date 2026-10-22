
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
async def test_job_stub_flow_stages_in_order(app, settings):
    transport = ASGITransport(app=app)
    headers = {"X-Access-Code": "correct_token_123"}
    store = JobStore(settings.paths.jobs_dir)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create job
        resp = await client.post(
            "/api/jobs",
            json={"type": "upload", "upload_id": "test_up_1234567890123"},
            headers=headers,
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        assert len(job_id) >= 22

        # Check initial state
        resp = await client.get(f"/api/jobs/{job_id}", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

        # Run orchestrator stub
        orchestrator.run(job_id, store, settings)

        # Job is now done
        resp = await client.get(f"/api/jobs/{job_id}", headers=headers)
        assert resp.status_code == 200
        job_data = resp.json()
        assert job_data["status"] == "done"
        assert job_data["stage"] == "done"
        assert job_data["progress"] == 1.0

        # Validate report.json against Report pydantic model
        resp = await client.get(f"/api/jobs/{job_id}/report.json", headers=headers)
        assert resp.status_code == 200
        report = Report.model_validate(resp.json())
        assert report.job_id == job_id
        assert report.grade == "Insufficient evidence"

        # Validate report.html returns HTML content
        resp = await client.get(f"/api/jobs/{job_id}/report.html", headers=headers)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "<html" in resp.text.lower()


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
