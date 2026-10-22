import shutil
from pathlib import Path

import pytest

from classeval.config import load_settings


def test_weights_sum_100_and_ids_r1_r12(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "testcode123")
    settings = load_settings("config.yaml", env_file=None)
    expected_ids = {f"R{i}" for i in range(1, 13)}
    assert set(settings.weights.keys()) == expected_ids
    assert sum(settings.weights.values()) == 100


def test_groups_split_75_25(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "testcode123")
    settings = load_settings("config.yaml", env_file=None)
    pres_weight = sum(settings.weights[rid] for rid in settings.groups.presentation)
    teach_weight = sum(settings.weights[rid] for rid in settings.groups.teaching)
    assert pres_weight == 25
    assert teach_weight == 75


def test_access_code_required(tmp_path, monkeypatch):
    # Pass even when a real .env containing ACCESS_CODE exists in the current directory
    fake_env = tmp_path / ".env"
    fake_env.write_text("ACCESS_CODE=real_code_in_dir\n", encoding="utf-8")
    cfg_copy = tmp_path / "config.yaml"
    shutil.copy("config.yaml", cfg_copy)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ACCESS_CODE", raising=False)
    with pytest.raises(ValueError, match="ACCESS_CODE"):
        load_settings(cfg_copy, env_file=None)


def test_env_file_is_read_when_requested(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ACCESS_CODE=custom_file_code\n", encoding="utf-8")
    monkeypatch.delenv("ACCESS_CODE", raising=False)
    settings = load_settings("config.yaml", env_file=env_file)
    assert settings.access_code.get_secret_value() == "custom_file_code"


def test_real_environment_beats_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ACCESS_CODE=from_file\n", encoding="utf-8")
    monkeypatch.setenv("ACCESS_CODE", "from_environment")
    settings = load_settings("config.yaml", env_file=env_file)
    assert settings.access_code.get_secret_value() == "from_environment"


def test_short_access_code_rejected(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "short9chr")
    with pytest.raises(ValueError, match="at least 10"):
        load_settings("config.yaml", env_file=None)


def test_placeholder_access_code_rejected(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "change_this_to_a_secure_token")
    with pytest.raises(ValueError, match="ACCESS_CODE"):
        load_settings("config.yaml", env_file=None)


def test_scoring_keys_and_ranges(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "testcode123")
    settings = load_settings("config.yaml", env_file=None)
    expected_keys = {
        "branded_ratio_tutor_speaking",
        "student_words_per_min",
        "dead_air_per_min",
        "distinct_responders",
        "question_reply_rate",
        "pitch_std_semitones",
        "energy_cv",
        "board_share_ratio",
    }
    assert set(settings.scoring.keys()) == expected_keys
    for k, v in settings.scoring.items():
        assert len(v) == 2, f"Key {k} does not have exactly 2 numbers"
        assert all(isinstance(x, (int, float)) for x in v), f"Key {k} contains non-numeric values"


def test_stage_sections_load_from_yaml(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "testcode123")
    settings = load_settings("config.yaml", env_file=None)
    assert settings.asr.compute_type == "int8_float16"
    assert settings.asr.language_windows == 5
    assert settings.diarization.prefer == "pyannote"
    assert settings.screen_state.flat_card_luma_range == (20.0, 40.0)
    assert settings.screen_state.logo_scales[0] == 0.35 and settings.screen_state.logo_scales[-1] == 1.6
    assert set(settings.screen_state.clip_prompts) == {
        "BOARD_SHARE",
        "LESSON_VIDEO_SHARE",
        "OTHER_TALKING_HEAD",
        "OTHER_SHARE",
    }


def test_stage_sections_have_defaults_when_missing_from_yaml(tmp_path, monkeypatch):
    import yaml

    monkeypatch.setenv("ACCESS_CODE", "testcode123")
    data = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    for key in ("asr", "diarization", "screen_state"):
        data.pop(key)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    settings = load_settings(cfg, env_file=None)
    assert settings.asr.batch_size == 8
    assert settings.diarization.ecapa_hop_s == 0.75
    assert settings.screen_state.clip_prompts == {}


def test_env_override(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "secret_token_val")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_123")
    monkeypatch.setenv("RECORDINGS_DIR", "F:\\test_recordings")
    settings = load_settings("config.yaml", env_file=None)
    assert settings.access_code.get_secret_value() == "secret_token_val"
    assert settings.hf_token is not None
    assert settings.hf_token.get_secret_value() == "hf_secret_123"
    assert settings.recordings_dir == Path("F:\\test_recordings")

    rep = repr(settings)
    assert "secret_token_val" not in rep
    assert "hf_secret_123" not in rep
