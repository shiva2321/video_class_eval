import subprocess

import numpy as np
import pytest
import soundfile as sf
import torch

from classeval.config import load_settings
from classeval.pipeline import asr
from tests.fixtures.speech import make_speech_wav


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_asr_token_123")
    return load_settings("config.yaml", env_file=None)


def _get_gpu_mem_used_mb() -> int:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        return int(res.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return int(torch.cuda.memory_allocated() / (1024 * 1024))


@pytest.mark.gpu
def test_transcribes_synthetic_speech(tmp_path, settings):
    wav = tmp_path / "speech.wav"
    try:
        make_speech_wav("Hello everyone, welcome to today's chess class", wav)
    except RuntimeError as exc:
        pytest.skip(f"SAPI voice unavailable: {exc}")

    tr = asr.transcribe(wav, settings)
    assert len(tr.segments) > 0
    full_text = " ".join(s.text.lower() for s in tr.segments)
    assert "welcome" in full_text
    assert "chess" in full_text

    for seg in tr.segments:
        for w in seg.words:
            assert w.start < w.end


@pytest.mark.gpu
def test_silence_gives_no_segments(tmp_path, settings):
    wav = tmp_path / "silence.wav"
    silence = np.zeros(16000 * 3, dtype=np.float32)
    sf.write(str(wav), silence, 16000)

    tr = asr.transcribe(wav, settings)
    assert len(tr.segments) == 0


@pytest.mark.gpu
def test_tone_gives_at_most_3_words(tmp_path, settings):
    wav = tmp_path / "tone.wav"
    t = np.linspace(0, 3, 16000 * 3, endpoint=False, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(str(wav), tone, 16000)

    tr = asr.transcribe(wav, settings)
    total_words = sum(len(s.words) for s in tr.segments)
    assert total_words <= 3


@pytest.mark.gpu
def test_language_detect_english(tmp_path, settings):
    wav = tmp_path / "detect_en.wav"
    try:
        make_speech_wav(
            "Hello everyone, welcome to today's class. We will analyze the Italian game.", wav
        )
    except RuntimeError as exc:
        pytest.skip(f"SAPI voice unavailable: {exc}")

    lang, prob = asr.detect_language(wav, settings)
    assert lang == "en"
    assert prob > 0.6


@pytest.mark.gpu
def test_vram_released(tmp_path, settings):
    wav = tmp_path / "vram_test.wav"
    t = np.linspace(0, 2, 16000 * 2, endpoint=False, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(str(wav), tone, 16000)

    mem_before = _get_gpu_mem_used_mb()
    _ = asr.transcribe(wav, settings)
    mem_after = _get_gpu_mem_used_mb()

    # VRAM allocated should return to near baseline (<= before + 400 MB)
    assert mem_after <= mem_before + 400
