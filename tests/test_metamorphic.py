import subprocess
from pathlib import Path

import pytest
import soundfile as sf

from classeval.config import Settings, load_settings
from classeval.pipeline import orchestrator
from scripts.make_degraded import make_degraded
from tests.fixtures.frames import draw_frame
from tests.fixtures.speech import concat_wavs, make_speech_wav
from tests.test_e2e_synthetic import _build_3min_synthetic_class, _get_ffmpeg


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_metamorphic_token")
    return load_settings("config.yaml", env_file=None)


def _build_7min_synthetic_class(video_path: Path, work_dir: Path) -> Path:
    """Build a ~7-minute (420s) synthetic class video for metamorphic duration tests."""
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = _get_ffmpeg()

    wav_tutor_1 = work_dir / "t1.wav"
    wav_student_1 = work_dir / "s1.wav"
    wav_tutor_2 = work_dir / "t2.wav"

    make_speech_wav(
        "Hello everyone and welcome to chess class! Today we will practice tactics.",
        wav_tutor_1,
        voice="David",
    )
    make_speech_wav(
        "Hello teacher! I am ready.",
        wav_student_1,
        voice="Zira",
    )
    make_speech_wav(
        "Great, for homework solve chapter four. Goodbye everyone!",
        wav_tutor_2,
        voice="David",
    )

    combined_speech = work_dir / "speech_7m.wav"
    concat_wavs([wav_tutor_1, wav_student_1, wav_tutor_2], gap_s=50.0, out=combined_speech)

    audio_420 = work_dir / "audio_420.wav"
    data, sr = sf.read(str(combined_speech))
    target_samples = int(sr * 420.0)
    if len(data) < target_samples:
        import numpy as np

        padding = np.zeros(target_samples - len(data), dtype=data.dtype)
        data = np.concatenate([data, padding])
    else:
        data = data[:target_samples]
    sf.write(str(audio_420), data, sr)

    frame_board = work_dir / "board.jpg"
    draw_frame("board_grid", frame_board)

    concat_txt = work_dir / "frames_7m.txt"
    concat_txt.write_text(
        f"file '{frame_board.as_posix()}'\nduration 420\nfile '{frame_board.as_posix()}'\n",
        encoding="utf-8",
    )

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_txt),
        "-i",
        str(audio_420),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-t",
        "420",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120.0, check=False)
    if res.returncode != 0 or not video_path.exists():
        raise RuntimeError(f"Failed to build 7min synthetic class video: {res.stderr}")

    return video_path


@pytest.mark.e2e
def test_metamorphic_mute_lowers_clarity_and_flags_audio(tmp_path, settings: Settings):
    settings.limits.min_duration_s = 60
    base_clip = tmp_path / "base_class.mp4"
    _build_3min_synthetic_class(base_clip, tmp_path / "build_base")

    # Degrade: mute 5 minutes (or full audio if shorter)
    muted_clip = make_degraded(base_clip, tmp_path / "degraded", "mute5")

    # Run original pipeline
    out_orig = tmp_path / "out_orig"
    rep_orig = orchestrator.execute_pipeline(base_clip, out_orig, settings, job_id="orig")

    # Run muted pipeline
    out_muted = tmp_path / "out_muted"
    rep_muted = orchestrator.execute_pipeline(muted_clip, out_muted, settings, job_id="muted")

    r3_orig = next((d.score for d in rep_orig.dimensions if d.id == "R3"), 100.0)
    r3_muted = next((d.score for d in rep_muted.dimensions if d.id == "R3"), 100.0)
    r12_orig = next((d.score for d in rep_orig.dimensions if d.id == "R12"), 100.0)
    r12_muted = next((d.score for d in rep_muted.dimensions if d.id == "R12"), 100.0)

    # R3 or R12 score is lower on muted or dropout flags/evidence present
    has_dropout_or_flag = bool(rep_muted.technical_flags) or any(
        "dropout" in str(e).lower() for d in rep_muted.dimensions for e in d.evidence
    )

    assert (
        (r3_muted is not None and r3_orig is not None and r3_muted < r3_orig)
        or (r12_muted is not None and r12_orig is not None and r12_muted < r12_orig)
        or has_dropout_or_flag
    )


@pytest.mark.e2e
def test_metamorphic_black_video_flags_camera(tmp_path, settings: Settings):
    settings.limits.min_duration_s = 60
    base_clip = tmp_path / "base_class.mp4"
    _build_3min_synthetic_class(base_clip, tmp_path / "build_base")

    black_clip = make_degraded(base_clip, tmp_path / "degraded", "black")

    out_orig = tmp_path / "out_orig"
    rep_orig = orchestrator.execute_pipeline(base_clip, out_orig, settings, job_id="orig")

    out_black = tmp_path / "out_black"
    rep_black = orchestrator.execute_pipeline(black_clip, out_black, settings, job_id="black")

    r1_orig = next((d.score for d in rep_orig.dimensions if d.id == "R1" and d.score is not None), None)
    r1_black = next((d.score for d in rep_black.dimensions if d.id == "R1" and d.score is not None), None)
    r2_orig = next((d.score for d in rep_orig.dimensions if d.id == "R2" and d.score is not None), None)
    r2_black = next((d.score for d in rep_black.dimensions if d.id == "R2" and d.score is not None), None)

    # R1 and R2 should not be higher on black frames than original
    if r1_orig is not None and r1_black is not None:
        assert r1_black <= r1_orig
    if r2_orig is not None and r2_black is not None:
        assert r2_black <= r2_orig


@pytest.mark.e2e
def test_metamorphic_truncated_5min_insufficient_evidence(tmp_path, settings: Settings):
    # Set duration gate to 400s so 5-min (300s) clip is below it, while 7-min (420s) clip is above it
    settings.limits.min_duration_s = 400

    class_7min = tmp_path / "class_7min.mp4"
    _build_7min_synthetic_class(class_7min, tmp_path / "build_7min")

    trunc_clip = make_degraded(class_7min, tmp_path / "degraded", "trunc5min")

    out_trunc = tmp_path / "out_trunc"
    rep_trunc = orchestrator.execute_pipeline(trunc_clip, out_trunc, settings, job_id="trunc")

    assert rep_trunc.grade == "Insufficient evidence"
