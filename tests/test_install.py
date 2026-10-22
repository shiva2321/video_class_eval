import pytest

from scripts import install


def test_a_new_env_file_gets_a_random_access_code(tmp_path):
    (tmp_path / ".env.example").write_text("ACCESS_CODE=change_this_to_a_secure_token\nHF_TOKEN=\n", encoding="utf-8")
    message = install.ensure_env_file(tmp_path)
    lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    code = next(line.split("=", 1)[1] for line in lines if line.startswith("ACCESS_CODE="))
    assert "created .env" in message and "generated" in message
    assert len(code) >= 10 and code != install.PLACEHOLDER
    assert "HF_TOKEN=" in lines  # the other keys of the example are kept
    assert code not in message  # the code is never printed


def test_a_real_access_code_is_never_overwritten(tmp_path):
    (tmp_path / ".env").write_text("ACCESS_CODE=my_own_long_secret_code\nHF_TOKEN=hf_abcdefghijklmnop\n", encoding="utf-8")
    message = install.ensure_env_file(tmp_path)
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "ACCESS_CODE=my_own_long_secret_code" in text and "HF_TOKEN=hf_abcdefghijklmnop" in text
    assert "kept" in message


@pytest.mark.parametrize("existing", ["ACCESS_CODE=\n", "ACCESS_CODE=short\n", f"ACCESS_CODE={install.PLACEHOLDER}\n", "OTHER=1\n"])
def test_a_missing_placeholder_or_too_short_code_is_replaced(tmp_path, existing):
    (tmp_path / ".env").write_text(existing, encoding="utf-8")
    install.ensure_env_file(tmp_path)
    lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    codes = [line.split("=", 1)[1] for line in lines if line.startswith("ACCESS_CODE=")]
    assert len(codes) == 1 and len(codes[0]) >= 10 and codes[0] != install.PLACEHOLDER


def test_running_the_setup_twice_keeps_the_same_code(tmp_path):
    (tmp_path / ".env.example").write_text("ACCESS_CODE=\n", encoding="utf-8")
    install.ensure_env_file(tmp_path)
    first = (tmp_path / ".env").read_text(encoding="utf-8")
    install.ensure_env_file(tmp_path)
    assert (tmp_path / ".env").read_text(encoding="utf-8") == first


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_every_missing_tool_has_an_install_hint_on_every_platform(monkeypatch, platform):
    monkeypatch.setattr(install.sys, "platform", platform)
    for tool in ("uv", "ffmpeg", "ollama"):
        assert len(install.hint(tool)) > 10


def test_an_unreachable_ollama_is_reported_as_none_not_an_error(monkeypatch):
    monkeypatch.setattr(install, "OLLAMA_URL", "http://127.0.0.1:9")
    assert install.ollama_models() is None


def test_the_setup_stops_with_a_clear_code_when_a_required_tool_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    assert install.main(["--skip-selftest"]) == 2
    out = capsys.readouterr().out
    assert "MISSING" in out and "uv" in out and "ffmpeg" in out and "STOP" in out


def test_the_model_tag_comes_from_the_config_with_the_local_override_winning(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text('models:\n  llm: "model-a:1b"\n', encoding="utf-8")
    monkeypatch.setattr(install, "ROOT", tmp_path)
    assert install.llm_name() == "model-a:1b"
    (tmp_path / "config.local.yaml").write_text("models:\n  llm: 'model-b:2b'\n", encoding="utf-8")
    assert install.llm_name() == "model-b:2b"
