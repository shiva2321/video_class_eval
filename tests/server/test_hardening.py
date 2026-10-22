import shutil
from collections import namedtuple
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from classeval.config import Settings, load_settings
from classeval.server.jobs import JobStore
from classeval.server.main import create_app
from classeval.worker import run_worker

Usage = namedtuple("Usage", ["total", "used", "free"])


@pytest.fixture
def settings(tmp_path: Path, monkeypatch) -> Settings:
    monkeypatch.setenv("ACCESS_CODE", "test_hardening_token_123")
    s = load_settings("config.yaml", env_file=None)
    s.paths.jobs_dir = tmp_path / "jobs"
    s.paths.uploads_dir = tmp_path / "uploads"
    s.paths.jobs_dir.mkdir(parents=True, exist_ok=True)
    s.paths.uploads_dir.mkdir(parents=True, exist_ok=True)
    return s


def test_security_headers_present_on_every_response(settings: Settings):
    app = create_app(settings)
    client = TestClient(app)

    # 1. 200 OK response
    r_ok = client.get("/healthz")
    assert r_ok.status_code == 200
    for resp in [r_ok]:
        assert resp.headers["Content-Security-Policy"] == (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'self'"
        )
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["Referrer-Policy"] == "no-referrer"
        assert resp.headers["X-Frame-Options"] == "SAMEORIGIN"

    # 2. 401 Unauthorized response (missing/wrong token)
    r_unauth = client.post("/api/uploads", json={"size": 100, "filename": "test.mp4"})
    assert r_unauth.status_code == 401
    assert r_unauth.headers["Content-Security-Policy"] == (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'self'"
    )
    assert r_unauth.headers["X-Content-Type-Options"] == "nosniff"
    assert r_unauth.headers["Referrer-Policy"] == "no-referrer"
    assert r_unauth.headers["X-Frame-Options"] == "SAMEORIGIN"

    # 3. 404 Not Found response
    r_404 = client.get("/non_existent_path_404")
    assert r_404.status_code == 404
    assert r_404.headers["Content-Security-Policy"] == (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'self'"
    )
    assert r_404.headers["X-Content-Type-Options"] == "nosniff"
    assert r_404.headers["Referrer-Policy"] == "no-referrer"
    assert r_404.headers["X-Frame-Options"] == "SAMEORIGIN"


def test_job_responses_not_cacheable(settings: Settings):
    app = create_app(settings)
    client = TestClient(app)
    headers = {"X-Access-Code": "test_hardening_token_123"}

    store = JobStore(settings.paths.jobs_dir)
    job = store.create(source={"type": "upload", "upload_id": "test_upload_12345"})

    # 1. Job status response
    r_job = client.get(f"/api/jobs/{job.id}", headers=headers)
    assert r_job.status_code == 200
    assert r_job.headers["Cache-Control"] == "no-store"

    # 2. Mock report files
    j_dir = store.job_dir(job.id)
    (j_dir / "report.json").write_text('{"status": "ok"}', encoding="utf-8")
    (j_dir / "report.html").write_text("<html>Report</html>", encoding="utf-8")

    # 3. report.json
    r_rep_json = client.get(f"/api/jobs/{job.id}/report.json", headers=headers)
    assert r_rep_json.status_code == 200
    assert r_rep_json.headers["Cache-Control"] == "no-store"

    # 4. report.html
    r_rep_html = client.get(f"/api/jobs/{job.id}/report.html", headers=headers)
    assert r_rep_html.status_code == 200
    assert r_rep_html.headers["Cache-Control"] == "no-store"


def test_disk_guard_rejects_when_low_space(settings: Settings, monkeypatch):
    app = create_app(settings)
    client = TestClient(app)
    headers = {"X-Access-Code": "test_hardening_token_123"}

    # Monkeypatch shutil.disk_usage so free space is very low (e.g. 500 bytes)
    # Requested size = 1000 bytes; required = 3 * 1000 = 3000 bytes
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: Usage(total=10_000, used=9_500, free=500))

    resp = client.post(
        "/api/uploads",
        json={"size": 1000, "filename": "video.mp4"},
        headers=headers,
    )
    assert resp.status_code == 507
    assert resp.json()["detail"] == "Not enough free disk space on the server."


def test_purge_task_survives_exception(settings: Settings, monkeypatch):
    # Monkeypatch purge_expired to raise an exception
    def broken_purge(*args, **kwargs):
        raise RuntimeError("Simulated purge disk error")

    monkeypatch.setattr("classeval.server.main.purge_expired", broken_purge)

    app = create_app(settings)
    with TestClient(app) as client:
        # App should still boot up and serve requests without crashing
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_worker_restart_requeues_orphan(settings: Settings, monkeypatch):
    store = JobStore(settings.paths.jobs_dir)
    job = store.create(source={"type": "upload", "upload_id": "test_up_12345678"})

    # Make the job "running" but with updated_at from 3600 seconds ago (> lease_s = 1800)
    old_time = (datetime.now(UTC) - timedelta(seconds=3600)).isoformat()
    raw_orphan = job.model_copy(
        update={
            "status": "running",
            "stage": "asr",
            "progress": 0.3,
            "updated_at": old_time,
        }
    )
    (store.job_dir(job.id) / "job.json").write_text(raw_orphan.model_dump_json(indent=2), encoding="utf-8")

    handled_jobs: list[str] = []

    def fake_orchestrator_run(job_id, s_store, s_settings, stages=None):
        handled_jobs.append(job_id)
        s_store.update(job_id, status="done", stage="done", progress=1.0)

    monkeypatch.setattr("classeval.pipeline.orchestrator.run", fake_orchestrator_run)

    # Worker starts with once=True
    run_worker(settings, store, once=True)

    # Orphaned job must have been requeued and claimed/run to completion
    assert handled_jobs == [job.id]
    final_job = store.get(job.id)
    assert final_job.status == "done"
