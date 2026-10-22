import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_python_312():
    assert sys.version_info[:2] == (3, 12)


def test_ffmpeg_and_ffprobe_on_path():
    res1 = subprocess.run(["ffmpeg", "-version"], check=False, capture_output=True, text=True, timeout=10)
    assert res1.returncode == 0
    res2 = subprocess.run(["ffprobe", "-version"], check=False, capture_output=True, text=True, timeout=10)
    assert res2.returncode == 0


def test_imports():
    import cv2
    import fastapi
    import faster_whisper
    import httpx
    import librosa
    import parselmouth
    import pyannote.audio
    import pydantic
    import pyloudnorm
    import rapidfuzz
    import speechbrain
    import torch
    import transformers
    import yaml as _yaml

    assert torch is not None
    assert faster_whisper is not None
    assert cv2 is not None
    assert librosa is not None
    assert pyloudnorm is not None
    assert parselmouth is not None
    assert pyannote.audio is not None
    assert transformers is not None
    assert fastapi is not None
    assert httpx is not None
    assert pydantic is not None
    assert _yaml is not None
    assert rapidfuzz is not None
    assert speechbrain is not None


def test_config_has_model_names():
    cfg_path = Path("config.yaml")
    assert cfg_path.exists(), "config.yaml must exist"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), "config.yaml must be a dictionary"
    models = cfg.get("models", {})
    keys = ["whisper", "whisper_dir", "clip", "ecapa", "pyannote", "llm"]
    for k in keys:
        val = models.get(k)
        assert isinstance(val, str) and len(val.strip()) > 0, f"models.{k} missing or empty"
    ollama = cfg.get("ollama", {})
    assert isinstance(ollama.get("base_url"), str) and len(ollama["base_url"].strip()) > 0, "ollama.base_url missing"


def test_markers_registered():
    res = subprocess.run([sys.executable, "-m", "pytest", "--markers"], check=False, capture_output=True, text=True, timeout=15)
    assert res.returncode == 0
    for marker in ["gpu", "ollama", "e2e"]:
        assert marker in res.stdout, f"pytest marker '{marker}' not registered"


def test_make_clip_is_probeable(tmp_path):
    from tests.fixtures.clips import make_clip
    clip_path = tmp_path / "test_clip.mp4"
    make_clip(clip_path, seconds=2.0)
    assert clip_path.exists()
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(clip_path),
    ]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    dur = float(data["format"]["duration"])
    assert 1.8 <= dur <= 2.2
