import subprocess
import wave
from pathlib import Path

import pytest

from classeval.config import load_settings
from classeval.pipeline import ingest
from classeval.schema import MediaInfo
from tests.fixtures.clips import make_clip


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ACCESS_CODE", "test_token_ingest")
    s = load_settings("config.yaml", env_file=None)
    s.paths.jobs_dir = tmp_path / "jobs"
    return s


def test_no_shell_true():
    src = Path("classeval/pipeline/ingest.py").read_text(encoding="utf-8")
    assert "shell=True" not in src


def test_probe_duration_within_0_1s(tmp_path):
    clip = tmp_path / "test_10s.mp4"
    make_clip(clip, seconds=10.0)
    info = ingest.probe(clip)
    assert abs(info.duration_s - 10.0) < 0.1
    assert info.has_audio is True
    assert info.has_video is True
    assert info.width == 1280
    assert info.height == 720
    assert info.size_bytes > 0


def test_audio_only_has_no_frames(tmp_path, settings):
    clip = tmp_path / "audio_only.wav"
    make_clip(clip, seconds=5.0, with_video=False, with_audio=True)
    info = ingest.probe(clip)
    assert info.has_audio is True
    assert info.has_video is False
    warnings = ingest.validate(info, settings)
    assert "no_video" in warnings

    job_dir = tmp_path / "job_audio"
    prepared = ingest.prepare(clip, job_dir, settings)
    assert prepared.frames == [] and prepared.tiles == []
    assert prepared.wav is not None
    assert prepared.wav.exists()


def test_video_only_has_no_wav(tmp_path, settings):
    clip = tmp_path / "video_only.mp4"
    make_clip(clip, seconds=5.0, with_video=True, with_audio=False)
    info = ingest.probe(clip)
    assert info.has_video is True
    assert info.has_audio is False
    warnings = ingest.validate(info, settings)
    assert "no_audio" in warnings

    job_dir = tmp_path / "job_video"
    prepared = ingest.prepare(clip, job_dir, settings)
    assert prepared.wav is None
    assert len(prepared.frames) > 0


def test_frame_count_matches_fps(tmp_path, settings):
    # 60s at 0.2 fps = 12 frames (+- 1)
    clip = tmp_path / "test_60s.mp4"
    make_clip(clip, seconds=60.0)
    job_dir = tmp_path / "job_60s"
    prepared = ingest.prepare(clip, job_dir, settings)
    assert abs(len(prepared.frames) - 12) <= 1
    # Check FrameRef timestamps
    for i, fr in enumerate(prepared.frames):
        assert fr.path.exists()
        expected_t = i / settings.sampling.frame_fps
        assert abs(fr.t - expected_t) < 0.01


def test_wav_is_16k_mono_s16(tmp_path, settings):
    clip = tmp_path / "test_wav.mp4"
    make_clip(clip, seconds=5.0)
    job_dir = tmp_path / "job_wav"
    prepared = ingest.prepare(clip, job_dir, settings)
    assert prepared.wav is not None
    assert prepared.wav.exists()

    with wave.open(str(prepared.wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getframerate() == 16000


def test_corrupt_file_raises_unreadable(tmp_path):
    bad_file = tmp_path / "corrupt.mp4"
    bad_file.write_bytes(b"not a valid video format at all 123456789")
    with pytest.raises(ingest.IngestError) as exc_info:
        ingest.probe(bad_file)
    assert exc_info.value.code in ("unreadable", "no_streams")


def test_too_long_by_mocked_probe(tmp_path, settings):
    info = MediaInfo(
        duration_s=14400.0,
        has_audio=True,
        has_video=True,
        width=1280,
        height=720,
        fps=25.0,
        video_kbps=1000.0,
        container="mp4",
        size_bytes=100000,
    )
    with pytest.raises(ingest.IngestError) as exc_info:
        ingest.validate(info, settings)
    assert exc_info.value.code == "too_long"


def test_ffmpeg_timeout_maps_to_timeout_error(tmp_path, monkeypatch, settings):
    clip = tmp_path / "test_timeout.mp4"
    make_clip(clip, seconds=2.0)
    job_dir = tmp_path / "job_timeout"

    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1.0)

    monkeypatch.setattr(subprocess, "run", mock_run)
    with pytest.raises(ingest.IngestError) as exc_info:
        ingest.prepare(clip, job_dir, settings)
    assert exc_info.value.code == "timeout"


def test_variable_frame_rate_clip_ok(tmp_path, settings):
    clip = tmp_path / "test_vfr.mp4"
    make_clip(clip, seconds=5.0, extra_args=["-fps_mode", "vfr"])
    info = ingest.probe(clip)
    assert info.duration_s > 0
    job_dir = tmp_path / "job_vfr"
    prepared = ingest.prepare(clip, job_dir, settings)
    assert prepared.info.duration_s > 0


def test_the_tutors_tile_is_cut_out_in_the_same_pass_at_native_resolution(tmp_path, settings):
    import cv2

    clip = tmp_path / "test_tiles.mp4"
    make_clip(clip, seconds=60.0)
    prepared = ingest.prepare(clip, tmp_path / "job_tiles", settings)
    cfg = settings.screen_state
    assert abs(len(prepared.tiles) - 60.0 / cfg.tile_every_s) <= 1
    for i, tile in enumerate(prepared.tiles):
        assert tile.path.exists() and abs(tile.t - i * cfg.tile_every_s) < 0.01
    full = prepared.info.width, prepared.info.height
    h, w = cv2.imread(str(prepared.tiles[0].path)).shape[:2]
    _, _, tw, th = cfg.thumb_region
    assert abs(w - full[0] * tw) <= 2 and abs(h - full[1] * th) <= 2  # a crop of the original, not of the scaled frame
