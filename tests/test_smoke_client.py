import threading
import time

import pytest
from httpx import ASGITransport

from classeval.config import load_settings
from classeval.server.jobs import JobStore
from classeval.server.main import create_app
from classeval.worker import run_worker
from scripts.smoke_public import run_smoke


@pytest.fixture
def server_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ACCESS_CODE", "test_secret_smoke_123")
    s = load_settings("config.yaml", env_file=None)
    s.paths.uploads_dir = tmp_path / "uploads"
    s.paths.jobs_dir = tmp_path / "jobs"
    s.paths.uploads_dir.mkdir(parents=True, exist_ok=True)
    s.paths.jobs_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(s)
    return s, app


@pytest.mark.gpu  # runs the real pipeline, which loads speech models on the GPU
def test_smoke_client_in_process(server_env):
    settings, app = server_env
    transport = ASGITransport(app=app)
    store = JobStore(settings.paths.jobs_dir)

    # Run worker in background thread
    stop_event = threading.Event()

    def worker_loop():
        while not stop_event.is_set():
            run_worker(settings, store, poll_interval=0.1, once=True)
            time.sleep(0.1)

    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

    try:
        ok, msg = run_smoke("http://testserver", "test_secret_smoke_123", transport=transport)
        assert ok, f"Smoke run failed: {msg}"
        assert msg.startswith("OK")
    finally:
        stop_event.set()
        t.join(timeout=2.0)
