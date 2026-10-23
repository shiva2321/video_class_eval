import re
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from classeval.config import load_settings
from classeval.server.main import create_app


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "correct_token_123")
    settings = load_settings("config.yaml", env_file=None)
    return create_app(settings)


def test_static_has_no_external_urls():
    static_dir = Path("classeval/server/static")
    for fname in ["index.html", "app.js", "style.css"]:
        fpath = static_dir / fname
        assert fpath.exists(), f"Static file {fname} must exist"
        content = fpath.read_text(encoding="utf-8")
        # Check no external http:// or https:// URLs exist
        matches = re.findall(r"https?://[^\s\"'>]+", content)
        assert not matches, f"External URLs found in {fname}: {matches}"


@pytest.mark.asyncio
async def test_root_serves_index_html(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "<!doctype html>" in resp.text.lower() or "<html" in resp.text.lower()


def test_a_code_in_the_link_fragment_is_taken_over_and_removed_from_the_address_bar():
    js = Path("classeval/server/static/app.js").read_text(encoding="utf-8")
    assert 'new URLSearchParams(location.hash.slice(1)).get("code")' in js
    assert "history.replaceState" in js and 'localStorage.setItem("classeval_access_code"' in js


def test_the_cookie_value_is_encoded_and_a_fragment_code_is_validated():
    js = Path("classeval/server/static/app.js").read_text(encoding="utf-8")
    assert "encodeURIComponent(value" in js and "decodeURIComponent(raw)" in js
    assert r"/^[!-~]{1,128}$/.test(linkCode)" in js


@pytest.mark.asyncio
async def test_a_percent_encoded_cookie_with_special_characters_is_accepted(monkeypatch):
    from urllib.parse import quote

    code = "odd;code=with space%and!chars_123"
    monkeypatch.setenv("ACCESS_CODE", code)
    app = create_app(load_settings("config.yaml", env_file=None))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        ok = await client.get("/api/jobs/abcdefghijklmnop0123", cookies={"access": quote(code, safe="")})
        wrong = await client.get("/api/jobs/abcdefghijklmnop0123", cookies={"access": "nope"})
    assert ok.status_code == 404 and wrong.status_code == 401  # authorised (the job does not exist) versus refused
