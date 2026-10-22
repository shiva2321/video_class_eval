import os
import subprocess
import time
from pathlib import Path

import pytest
import soundfile as sf
import torch

from classeval.config import Settings, load_settings
from classeval.pipeline import orchestrator
from classeval.schema import Report
from classeval.server.jobs import JobStore
from classeval.server.uploads import UploadStore
from tests.fixtures.frames import draw_frame
from tests.fixtures.speech import concat_wavs, make_speech_wav


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_e2e_token")
    return load_settings("config.yaml", env_file=None)


def _get_ffmpeg() -> str:
    return "ffmpeg"


def _build_3min_synthetic_class(video_path: Path, work_dir: Path) -> Path:
    """Build a 3-minute synthetic class video with two alternating voices and cycling frames."""
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = _get_ffmpeg()

    # 1. Synthesize alternating dialogue clips
    wav_tutor_1 = work_dir / "t1.wav"
    wav_student_1 = work_dir / "s1.wav"
    wav_tutor_2 = work_dir / "t2.wav"
    wav_student_2 = work_dir / "s2.wav"
    wav_tutor_3 = work_dir / "t3.wav"

    make_speech_wav(
        "Hello everyone and welcome to our chess class! Today we will practice pawn endgames and checkmating patterns.",
        wav_tutor_1,
        voice="David",
    )
    make_speech_wav(
        "Hello teacher! I am ready to practice.",
        wav_student_1,
        voice="Zira",
    )
    make_speech_wav(
        "Great! Vihaan, what is the best move for White in this position? Look at the queen on e4.",
        wav_tutor_2,
        voice="David",
    )
    make_speech_wav(
        "I think queen to e7 is checkmate!",
        wav_student_2,
        voice="Zira",
    )
    make_speech_wav(
        "Excellent move! That is correct. For homework, please solve the chapter four tactics. Good job today, goodbye everyone!",
        wav_tutor_3,
        voice="David",
    )

    combined_speech = work_dir / "speech_combined.wav"
    concat_wavs(
        [wav_tutor_1, wav_student_1, wav_tutor_2, wav_student_2, wav_tutor_3],
        gap_s=25.0,  # distribute turns across the 3 minutes
        out=combined_speech,
    )

    # Pad or trim combined speech to exactly 180 seconds (3 minutes)
    audio_180 = work_dir / "audio_180.wav"
    data, sr = sf.read(str(combined_speech))
    target_samples = int(sr * 180.0)
    if len(data) < target_samples:
        import numpy as np
        padding = np.zeros(target_samples - len(data), dtype=data.dtype)
        data = np.concatenate([data, padding])
    else:
        data = data[:target_samples]
    sf.write(str(audio_180), data, sr)

    # 2. Draw 3 distinct frame kinds
    frame_branded = work_dir / "branded.jpg"
    frame_board = work_dir / "board.jpg"
    frame_card = work_dir / "card.jpg"
    draw_frame("branded_head", frame_branded)
    draw_frame("board_grid", frame_board)
    draw_frame("camera_off_card", frame_card)

    # Create concat script for 3x60s frame segments = 180s
    concat_txt = work_dir / "frames.txt"
    concat_txt.write_text(
        f"file '{frame_branded.as_posix()}'\nduration 60\n"
        f"file '{frame_board.as_posix()}'\nduration 60\n"
        f"file '{frame_card.as_posix()}'\nduration 60\n"
        f"file '{frame_card.as_posix()}'\n",
        encoding="utf-8",
    )

    # 3. Mux into 180s MP4 with ffmpeg
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
        str(audio_180),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-t",
        "180",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120.0, check=False)
    if res.returncode != 0 or not video_path.exists():
        raise RuntimeError(f"FFmpeg muxing failed: {res.stderr}")

    return video_path


@pytest.mark.e2e
def test_synthetic_class_end_to_end(tmp_path, settings: Settings):
    """End-to-end pipeline run on a 3-minute synthetic class video using real models."""
    # Ensure offline environment
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Lower min_duration_s for this test
    settings.limits.min_duration_s = 60
    settings.paths.uploads_dir = tmp_path / "uploads"
    settings.paths.jobs_dir = tmp_path / "jobs"
    settings.paths.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.jobs_dir.mkdir(parents=True, exist_ok=True)

    # Baseline VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        baseline_vram_bytes = torch.cuda.memory_allocated()
    else:
        baseline_vram_bytes = 0

    # Build synthetic class media
    video_path = tmp_path / "class_3min.mp4"
    _build_3min_synthetic_class(video_path, tmp_path / "build")

    # Create upload and job
    up_store = UploadStore(settings.paths.uploads_dir, settings.limits.max_upload_bytes, settings.limits.chunk_bytes)
    up_id = up_store.init(size=video_path.stat().st_size, filename="class_3min.mp4")
    video_bytes = video_path.read_bytes()
    up_store.put_chunk(up_id, 0, video_bytes)
    up_store.complete(up_id)

    j_store = JobStore(settings.paths.jobs_dir)
    job = j_store.create({"type": "upload", "upload_id": up_id})

    # Execute real pipeline end-to-end
    t_start = time.monotonic()
    orchestrator.run(job.id, j_store, settings)
    elapsed_s = time.monotonic() - t_start

    # Check job status
    job_after = j_store.get(job.id)
    assert job_after is not None
    assert job_after.status == "done", f"Job execution failed with error: {job_after.error}"

    # Check report
    job_dir = j_store.job_dir(job.id)
    report_file = job_dir / "report.json"
    assert report_file.exists()

    report = Report.model_validate_json(report_file.read_text(encoding="utf-8"))
    assert report.job_id == job.id
    assert report.grade in ["Poor", "Average", "Good", "Excellent", "Insufficient evidence"]
    assert report.media.duration_s == 180.0 or abs(report.media.duration_s - 180.0) <= 2.0

    # Wall time assertion: under 8 minutes (480s)
    assert elapsed_s < 480.0, f"Pipeline took {elapsed_s}s, exceeding 480s limit"

    # GPU memory back near baseline afterwards (<= baseline + 400 MB)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_after_bytes = torch.cuda.memory_allocated()
        delta_mb = (vram_after_bytes - baseline_vram_bytes) / (1024 * 1024)
        assert delta_mb <= 400.0, f"VRAM leaked {delta_mb} MB above baseline"
