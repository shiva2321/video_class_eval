import logging
from pathlib import Path

import pytest

from classeval.config import Settings, load_settings
from classeval.pipeline.checklist import ChecklistItem, ChecklistResult
from classeval.pipeline.ingest import Prepared
from classeval.pipeline.orchestrator import StageFns
from classeval.pipeline.scoring import ScoreResult
from classeval.schema import (
    AudioFacts,
    Detection,
    Diarization,
    DimensionScore,
    FrameRef,
    FrameState,
    MediaInfo,
    Segment,
    SpeakerTurn,
    Transcript,
    TranscriptFacts,
    VideoFacts,
)
from classeval.server.jobs import JobStore
from classeval.server.uploads import UploadStore
from classeval.worker import run_worker

SENTINEL = "zebra-quartz-17"


def _make_dummy_media(duration_s: float = 60.0) -> MediaInfo:
    return MediaInfo(
        duration_s=duration_s,
        has_audio=True,
        has_video=True,
        width=1280,
        height=720,
        fps=25.0,
        video_kbps=1000.0,
        container="mp4",
        size_bytes=10000,
    )


def _make_dummy_transcript() -> Transcript:
    return Transcript(
        language="en",
        language_confidence=0.95,
        asr_model="large-v3-turbo",
        segments=[
            Segment(start=0.0, end=10.0, text="Hello and welcome to chess class.", speaker="TUTOR"),
            Segment(start=10.0, end=20.0, text=f"My child secret is {SENTINEL}", speaker="CHILD_1"),
        ],
    )


def _make_dummy_diarization() -> Diarization:
    return Diarization(
        turns=[
            SpeakerTurn(start=0.0, end=10.0, speaker="TUTOR"),
            SpeakerTurn(start=10.0, end=20.0, speaker="CHILD_1"),
        ],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=2,
    )


def _make_dummy_audio_facts() -> AudioFacts:
    return AudioFacts(
        duration_s=60.0,
        speech_ratio=0.5,
        tutor_talk_share=0.8,
        student_words_per_min=20.0,
        dead_air_per_min=0.2,
        median_response_latency_s=2.5,
        tutor_speech_rate_wpm=120.0,
        loudness_lufs=-20.0,
        clipping_ratio=0.0,
        snr_db_est=25.0,
        pitch_std_semitones_tutor=2.5,
        energy_cv_tutor=0.6,
        dropouts=[],
    )


def _make_dummy_video_facts(n_frames: int = 12) -> VideoFacts:
    return VideoFacts(
        states=[FrameState(t=float(i * 5), state="BRANDED_TALKING_HEAD", confidence=0.9) for i in range(n_frames)],
        tutor_name=None,
        branded_ratio_tutor_speaking=0.8,
        board_share_ratio=0.3,
        lesson_video_ratio=0.0,
        camera_off_total_ratio=0.0,
        camera_off_while_tutor_speaks_s=0.0,
        first5_branded=True,
        last5_branded=True,
        talking_head_luma_mean=120.0,
        n_frames=n_frames,
    )


def _make_dummy_tfacts() -> TranscriptFacts:
    return TranscriptFacts(
        detections={
            "greeting_opening": Detection(assessed=True, score=1.0, hits=[]),
            "rapport": Detection(assessed=True, score=0.0, hits=[]),
            "self_intro_new_joiners": Detection(assessed=True, score=0.0, hits=[]),
            "individual_interaction": Detection(assessed=True, score=1.0, hits=[]),
            "homework": Detection(assessed=True, score=0.0, hits=[]),
            "closing": Detection(assessed=True, score=1.0, hits=[]),
            "chess_engagement": Detection(assessed=True, score=0.5, hits=[]),
            "praise_density": Detection(assessed=True, score=0.0, hits=[]),
        },
        distinct_responders=1,
        question_reply_rate=0.8,
        question_count=2,
    )


def _make_dummy_checklist() -> ChecklistResult:
    return ChecklistResult(
        items={
            "R9": ChecklistItem(dimension="R9", verdict="met", evidence=[]),
            "R10": ChecklistItem(dimension="R10", verdict="not_observed", evidence=[]),
            "R11": ChecklistItem(dimension="R11", verdict="met", evidence=[]),
        },
        strengths=[],
        improvements=[],
        degraded=False,
    )


def _make_dummy_scoring() -> ScoreResult:
    dims = [
        DimensionScore(
            id="R1",
            name="Professional backdrop & attire",
            group="presentation",
            score=85.0,
            weight=8,
            confidence=0.9,
            assessed=True,
            evidence=[],
        ),
        DimensionScore(
            id="R3",
            name="Audio clarity",
            group="presentation",
            score=90.0,
            weight=4,
            confidence=0.9,
            assessed=True,
            evidence=[],
        ),
        DimensionScore(
            id="R4",
            name="Welcoming & warm-up",
            group="teaching",
            score=100.0,
            weight=10,
            confidence=0.9,
            assessed=True,
            evidence=[],
        ),
    ]
    return ScoreResult(
        dimensions=dims,
        overall_score=88.0,
        coverage=1.0,
        grade="Good",
        confidence=0.85,
        flags=[],
    )


def test_log_scrub_child_sentinel_never_logged_or_persisted(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ACCESS_CODE", "scrub_test_token_123")
    monkeypatch.setattr("classeval.pipeline.orchestrator.OllamaClient.unload", lambda self: None)
    caplog.set_level(logging.DEBUG)

    settings: Settings = load_settings("config.yaml", env_file=None)
    settings.paths.uploads_dir = tmp_path / "uploads"
    settings.paths.jobs_dir = tmp_path / "jobs"
    settings.paths.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.jobs_dir.mkdir(parents=True, exist_ok=True)

    up_store = UploadStore(settings.paths.uploads_dir, settings.limits.max_upload_bytes, settings.limits.chunk_bytes)
    upload_id = up_store.init(size=10, filename="test.mp4")
    up_store.put_chunk(upload_id, 0, b"0123456789")
    up_store.complete(upload_id)

    job_store = JobStore(settings.paths.jobs_dir)
    job = job_store.create(source={"type": "upload", "upload_id": upload_id})

    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_bytes(b"RIFFdummyWAVEfmt ")
    frames = [FrameRef(t=float(i * 5), path=tmp_path / f"frame_{i}.jpg") for i in range(12)]
    for f in frames:
        f.path.write_bytes(b"fakejpg")

    stages = StageFns(
        ingest=lambda *a, **k: Prepared(wav=dummy_wav, frames=frames, info=_make_dummy_media()),
        asr=lambda *a, **k: _make_dummy_transcript(),
        classify_frames=lambda frs, s: [FrameState(t=f.t, state="BRANDED_TALKING_HEAD", confidence=0.9) for f in frs],
        diarize=lambda *a, **k: _make_dummy_diarization(),
        assign_speakers=lambda tr, d: tr,
        audio=lambda *a, **k: _make_dummy_audio_facts(),
        video=lambda states, d, tr, s, frames=None: _make_dummy_video_facts(n_frames=len(states)),
        thumbnail=lambda *a, **k: (0, None),
        visual=lambda *a, **k: {},
        narrative=lambda *a, **k: None,
        tfacts=lambda *a, **k: _make_dummy_tfacts(),
        understand=lambda *a, **k: _make_dummy_checklist(),
        score=lambda *a, **k: _make_dummy_scoring(),
        cleanup_gpu=lambda: None,
    )

    run_worker(settings, job_store, once=True, stages=stages)

    final_job = job_store.get(job.id)
    assert final_job is not None
    assert final_job.status == "done"

    # 1. Assert sentinel appears in no log record
    for record in caplog.records:
        assert SENTINEL not in record.getMessage(), f"Sentinel found in log record: {record.getMessage()}"

    # 2. Assert sentinel appears in no stdout / stderr
    captured = capsys.readouterr()
    assert SENTINEL not in captured.out, "Sentinel found in stdout"
    assert SENTINEL not in captured.err, "Sentinel found in stderr"

    # 3. Assert sentinel appears in no file under jobs/ or uploads/ dirs after completion
    all_files = list(settings.paths.jobs_dir.rglob("*")) + list(settings.paths.uploads_dir.rglob("*"))
    for file_path in all_files:
        if file_path.is_file():
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            assert SENTINEL not in text, f"Sentinel found persisted in {file_path}"
