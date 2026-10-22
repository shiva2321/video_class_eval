"""Extracted frames show the children and the extracted audio carries their voices: neither may outlive a run."""

from pathlib import Path

import pytest

from classeval.config import load_settings
from classeval.pipeline.ingest import Prepared
from classeval.pipeline.orchestrator import execute_pipeline
from classeval.schema import FrameRef
from tests.pipeline.test_orchestrator import _build_fake_stages, _make_dummy_media

MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".wav", ".mp4", ".npz", ".npy"}


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_media_purge_token")
    return load_settings("config.yaml", env_file=None)


def _media_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES]


def _stages_that_write_real_files(tmp_path, calls):
    """Fake stages whose ingest writes frames and audio into the job's work folder, like the real one."""
    stages = _build_fake_stages(tmp_path)

    def ingest(src, ingest_dir, settings):
        calls.append(1)
        ingest_dir.mkdir(parents=True, exist_ok=True)
        wav = ingest_dir / "audio_16k.wav"
        wav.write_bytes(b"RIFFdummywav")
        frames = []
        for i in range(12):
            path = ingest_dir / f"frame_{i:06d}.jpg"
            path.write_bytes(b"fakejpg")
            frames.append(FrameRef(t=5.0 * i, path=path))
        return Prepared(info=_make_dummy_media(60.0), wav=wav, frames=frames)

    stages.ingest = ingest
    return stages


def test_frames_and_audio_are_gone_after_a_successful_run(tmp_path, settings):
    job_dir, calls = tmp_path / "job", []
    stages = _stages_that_write_real_files(tmp_path, calls)
    execute_pipeline(tmp_path / "in.mp4", job_dir, settings, stages=stages)
    assert calls and _media_files(job_dir) == []
    assert not (job_dir / "work" / "ingest").exists()
    assert (job_dir / "work" / "stages" / "asr.done.json").exists()  # numbers and transcripts stay for a re-run


def test_frames_and_audio_are_gone_after_a_failed_run(tmp_path, settings):
    job_dir, calls = tmp_path / "job", []
    stages = _stages_that_write_real_files(tmp_path, calls)

    def explode(*args, **kwargs):
        raise RuntimeError("scripted failure")

    stages.audio = explode
    with pytest.raises(RuntimeError):
        execute_pipeline(tmp_path / "in.mp4", job_dir, settings, stages=stages)
    assert calls and _media_files(job_dir) == []


def test_scratch_crops_of_the_tutor_tile_are_removed_too(tmp_path, settings):
    job_dir = tmp_path / "job"
    for name in ("thumbs", "visual"):
        (job_dir / "work" / name).mkdir(parents=True)
        (job_dir / "work" / name / "c0.jpg").write_bytes(b"jpg")
    execute_pipeline(tmp_path / "in.mp4", job_dir, settings, stages=_stages_that_write_real_files(tmp_path, []))
    assert _media_files(job_dir) == []


def test_a_rerun_extracts_the_media_again_because_the_cached_ingest_points_at_deleted_files(tmp_path, settings):
    job_dir, calls = tmp_path / "job", []
    stages = _stages_that_write_real_files(tmp_path, calls)
    execute_pipeline(tmp_path / "in.mp4", job_dir, settings, stages=stages)
    execute_pipeline(tmp_path / "in.mp4", job_dir, settings, stages=stages)
    assert len(calls) == 2 and _media_files(job_dir) == []
