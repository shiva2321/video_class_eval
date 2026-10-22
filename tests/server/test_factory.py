from fastapi.testclient import TestClient

from classeval.server.main import create_app


def test_the_factory_runs_without_arguments_like_uvicorn_calls_it(monkeypatch):
    """scripts/run.py starts the server with `uvicorn classeval.server.main:create_app --factory`."""
    monkeypatch.setenv("ACCESS_CODE", "test_factory_token_123")
    client = TestClient(create_app())
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/api/jobs/does_not_exist_0123456789", headers={"X-Access-Code": "wrong"}).status_code == 401
