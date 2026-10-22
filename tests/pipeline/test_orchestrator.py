from pathlib import Path
from unittest.mock import MagicMock

import pytest

from classeval.config import Settings, load_settings
from classeval.pipeline.checklist import ChecklistItem, ChecklistResult
from classeval.pipeline.ingest import Prepared
from classeval.pipeline.orchestrator import StageFns, run
from classeval.pipeline.scoring import ScoreResult
from classeval.schema import (
    AudioFacts,
    Detection,
    Diarization,
    DimensionScore,
    FrameRef,
    FrameState,
    MediaInfo,
    Report,
    Segment,
    SpeakerTurn,
    Transcript,
    TranscriptFacts,
    VideoFacts,
)
from classeval.server.jobs import JobStore
from classeval.server.uploads import UploadStore


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_orchestrator_token")
    return load_settings("config.yaml", env_file=None)


def _make_dummy_media(duration_s: float = 60.0, has_video: bool = True, has_audio: bool = True) -> MediaInfo:
    return MediaInfo(
        duration_s=duration_s,
        has_audio=has_audio,
        has_video=has_video,
        width=1280 if has_video else None,
        height=720 if has_video else None,
        fps=25.0 if has_video else None,
        video_kbps=1000.0 if has_video else None,
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
            Segment(start=10.0, end=20.0, text="Today we practice pawns.", speaker="TUTOR"),
        ],
    )


def _make_dummy_diarization() -> Diarization:
    return Diarization(
        turns=[SpeakerTurn(start=0.0, end=20.0, speaker="TUTOR")],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=1,
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
    if n_frames == 0:
        return VideoFacts(
            states=[],
            tutor_name=None,
            branded_ratio_tutor_speaking=None,
            board_share_ratio=0.0,
            lesson_video_ratio=0.0,
            camera_off_total_ratio=0.0,
            camera_off_while_tutor_speaks_s=0.0,
            first5_branded=None,
            last5_branded=None,
            talking_head_luma_mean=None,
            n_frames=0,
        )
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
            "R4": ChecklistItem(dimension="R4", verdict="met", evidence=[]),
            "R5": ChecklistItem(dimension="R5", verdict="not_observed", evidence=[]),
            "R6": ChecklistItem(dimension="R6", verdict="met", evidence=[]),
            "R7": ChecklistItem(dimension="R7", verdict="met", evidence=[]),
            "R9": ChecklistItem(dimension="R9", verdict="met", evidence=[]),
            "R10": ChecklistItem(dimension="R10", verdict="not_observed", evidence=[]),
            "R11": ChecklistItem(dimension="R11", verdict="met", evidence=[]),
        },
        strengths=[],
        improvements=[],
        degraded=False,
    )


def _make_dummy_score_result(video_assessed: bool = True) -> ScoreResult:
    dims = [
        DimensionScore(
            id="R1",
            name="Professional backdrop & attire",
            group="presentation",
            score=85.0 if video_assessed else None,
            weight=8,
            confidence=0.9,
            assessed=video_assessed,
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
    flags = [] if video_assessed else ["no_video"]
    return ScoreResult(
        dimensions=dims,
        overall_score=88.0,
        coverage=1.0 if video_assessed else 0.8,
        grade="Good",
        confidence=0.85,
        flags=flags,
    )


def _setup_job_and_upload(tmp_path: Path, settings: Settings, filename: str = "test.mp4") -> tuple[str, JobStore, UploadStore, Path]:
    settings.paths.uploads_dir = tmp_path / "uploads"
    settings.paths.jobs_dir = tmp_path / "jobs"
    settings.paths.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.jobs_dir.mkdir(parents=True, exist_ok=True)

    up_store = UploadStore(settings.paths.uploads_dir, settings.limits.max_upload_bytes, settings.limits.chunk_bytes)
    up_id = up_store.init(size=1024, filename=filename)
    up_store.put_chunk(up_id, 0, b"fake_media_bytes" * 64)
    final_p = up_store.complete(up_id)

    j_store = JobStore(settings.paths.jobs_dir)
    job = j_store.create({"type": "upload", "upload_id": up_id})
    return job.id, j_store, up_store, final_p


def _build_fake_stages(
    tmp_path: Path,
    has_video: bool = True,
    has_audio: bool = True,
    cleanup_cb: MagicMock | None = None,
) -> StageFns:
    media = _make_dummy_media(duration_s=60.0, has_video=has_video, has_audio=has_audio)
    wav_path = (tmp_path / "dummy.wav") if has_audio else None
    if wav_path:
        wav_path.write_bytes(b"RIFFdummywav")

    frames = [FrameRef(t=float(i * 5), path=tmp_path / f"frame_{i}.jpg") for i in range(12)] if has_video else []
    for f in frames:
        f.path.write_bytes(b"fakejpg")

    def fake_ingest(src, job_dir, s):
        return Prepared(info=media, wav=wav_path, frames=frames)

    def fake_asr(wav, s, progress=None):
        return _make_dummy_transcript()

    def fake_classify(frs, s):
        return [FrameState(t=f.t, state="BRANDED_TALKING_HEAD", confidence=0.9) for f in frs]

    def fake_diarize(wav, s):
        return _make_dummy_diarization()

    def fake_assign(tr, d):
        return tr

    def fake_audio(wav, tr, d):
        return _make_dummy_audio_facts()

    def fake_video(states, d, tr, s, frames=None):
        return _make_dummy_video_facts(n_frames=len(states))

    def fake_tfacts(tr, d, s):
        return _make_dummy_tfacts()

    def fake_understand(tr, d, tfacts, llm, s):
        return _make_dummy_checklist()

    def fake_score(m, tr, d, audio, video, tfacts, chk, s):
        return _make_dummy_score_result(video_assessed=m.has_video)

    return StageFns(
        ingest=fake_ingest,
        asr=fake_asr,
        classify_frames=fake_classify,
        diarize=fake_diarize,
        assign_speakers=fake_assign,
        audio=fake_audio,
        video=fake_video,
        tfacts=fake_tfacts,
        understand=fake_understand,
        score=fake_score,
        cleanup_gpu=cleanup_cb,
    )


def test_stage_order_and_progress_monotonic(tmp_path, settings):
    job_id, j_store, _up_store, _ = _setup_job_and_upload(tmp_path, settings)

    observed_stages: list[tuple[str, float]] = []
    orig_set_stage = j_store.set_stage

    def tracked_set_stage(jid: str, stage: str, prog: float):
        observed_stages.append((stage, prog))
        return orig_set_stage(jid, stage, prog)

    j_store.set_stage = tracked_set_stage

    stages = _build_fake_stages(tmp_path)
    run(job_id, j_store, settings, stages=stages)

    job = j_store.get(job_id)
    assert job is not None
    assert job.status == "done"

    stage_names = [st[0] for st in observed_stages]
    # Check that expected stages occurred in order
    expected = ["ingest", "asr", "diarize", "audio", "video", "understand", "score", "report"]
    assert stage_names == expected

    # Check progress values are strictly monotonic
    progresses = [st[1] for st in observed_stages]
    for i in range(len(progresses) - 1):
        assert progresses[i] < progresses[i + 1], f"Progress not monotonic at step {i}: {progresses}"
    assert progresses[0] > 0.0
    assert progresses[-1] <= 1.0


def test_failure_in_diarize_marks_job_failed_with_code_no_traceback(tmp_path, settings):
    job_id, j_store, _up_store, _ = _setup_job_and_upload(tmp_path, settings)

    stages = _build_fake_stages(tmp_path)

    def failing_diarize(wav, s):
        raise RuntimeError("ECAPA clustering failed due to internal math domain error")

    stages.diarize = failing_diarize

    run(job_id, j_store, settings, stages=stages)

    job = j_store.get(job_id)
    assert job is not None
    assert job.status == "failed"
    assert job.error is not None
    assert "code" in job.error
    assert "message" in job.error
    assert len(job.error["code"]) > 0
    # No traceback elements in the error message
    msg = job.error["message"]
    assert "Traceback" not in msg
    assert "File " not in msg
    assert ".py" not in msg


def test_resume_skips_completed_stages(tmp_path, settings):
    job_id, j_store, _up_store, _ = _setup_job_and_upload(tmp_path, settings)
    job_dir = j_store.job_dir(job_id)
    stages_dir = job_dir / "work" / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    # Pre-write done markers for ingest and asr
    media = _make_dummy_media()
    dummy_wav = tmp_path / "resume_audio.wav"
    dummy_wav.write_bytes(b"RIFFwav")
    prep = Prepared(info=media, wav=dummy_wav, frames=[])
    (stages_dir / "ingest.done.json").write_text(prep.model_dump_json(), encoding="utf-8")

    tr = _make_dummy_transcript()
    (stages_dir / "asr.done.json").write_text(tr.model_dump_json(), encoding="utf-8")

    stages = _build_fake_stages(tmp_path)
    ingest_mock = MagicMock(side_effect=stages.ingest)
    asr_mock = MagicMock(side_effect=stages.asr)
    diarize_mock = MagicMock(side_effect=stages.diarize)
    stages.ingest = ingest_mock
    stages.asr = asr_mock
    stages.diarize = diarize_mock

    run(job_id, j_store, settings, stages=stages)

    job = j_store.get(job_id)
    assert job is not None
    assert job.status == "done"

    # Ingest and asr should have been skipped
    assert ingest_mock.call_count == 0
    assert asr_mock.call_count == 0
    # Diarize should have run
    assert diarize_mock.call_count == 1


def test_gpu_cleanup_called_between_gpu_stages(tmp_path, settings):
    job_id, j_store, _up_store, _ = _setup_job_and_upload(tmp_path, settings)

    cleanup_mock = MagicMock()
    stages = _build_fake_stages(tmp_path, cleanup_cb=cleanup_mock)

    run(job_id, j_store, settings, stages=stages)

    job = j_store.get(job_id)
    assert job is not None
    assert job.status == "done"

    # Cleanup must be called at least 3 times (after ASR/CLIP, after diarize, after understand)
    assert cleanup_mock.call_count >= 3


def test_audio_only_flow_continues_with_video_dimensions_not_assessed(tmp_path, settings):
    job_id, j_store, _up_store, _ = _setup_job_and_upload(tmp_path, settings)

    stages = _build_fake_stages(tmp_path, has_video=False, has_audio=True)
    run(job_id, j_store, settings, stages=stages)

    job = j_store.get(job_id)
    assert job is not None
    assert job.status == "done"

    job_dir = j_store.job_dir(job_id)
    report_path = job_dir / "report.json"
    assert report_path.exists()
    report = Report.model_validate_json(report_path.read_text(encoding="utf-8"))

    # R1 should be not assessed
    r1 = next((d for d in report.dimensions if d.id == "R1"), None)
    assert r1 is not None
    assert r1.score is None
    assert "R1" in report.not_assessed


def test_video_only_fails_with_no_audio(tmp_path, settings):
    job_id, j_store, _up_store, _ = _setup_job_and_upload(tmp_path, settings)

    stages = _build_fake_stages(tmp_path, has_video=True, has_audio=False)
    run(job_id, j_store, settings, stages=stages)

    job = j_store.get(job_id)
    assert job is not None
    assert job.status == "failed"
    assert job.error is not None
    assert job.error["code"] == "no_audio"
    assert job.error["message"] == "This recording has no audio track."


def test_input_deleted_after_success(tmp_path, settings):
    job_id, j_store, up_store, _ = _setup_job_and_upload(tmp_path, settings)
    job = j_store.get(job_id)
    up_id = job.source["upload_id"]
    up_dir = up_store.root / up_id
    assert up_dir.exists()

    stages = _build_fake_stages(tmp_path)
    run(job_id, j_store, settings, stages=stages)

    job_after = j_store.get(job_id)
    assert job_after is not None
    assert job_after.status == "done"

    # Upload directory must be deleted after report is written
    assert not up_dir.exists()

    # Job directory and reports must still exist
    job_dir = j_store.job_dir(job_id)
    assert (job_dir / "report.json").exists()
    assert (job_dir / "report.html").exists()


def test_missing_upload_fails_with_upload_missing(tmp_path, settings):
    settings.paths.uploads_dir = tmp_path / "uploads"
    settings.paths.jobs_dir = tmp_path / "jobs"
    settings.paths.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.jobs_dir.mkdir(parents=True, exist_ok=True)

    j_store = JobStore(settings.paths.jobs_dir)
    job = j_store.create({"type": "upload", "upload_id": "nonexistent_upload_1234567890"})

    stages = _build_fake_stages(tmp_path)
    run(job.id, j_store, settings, stages=stages)

    job_after = j_store.get(job.id)
    assert job_after is not None
    assert job_after.status == "failed"
    assert job_after.error is not None
    assert job_after.error["code"] == "upload_missing"


def test_no_upload_id_is_special_cased_and_failed_jobs_leave_no_recording(tmp_path, settings, monkeypatch):
    """A job never skips the pipeline because of its id, and failures also remove the recording and work files."""
    from classeval.pipeline import orchestrator
    from classeval.server.jobs import JobStore
    from classeval.server.uploads import UploadStore

    settings.paths.uploads_dir = tmp_path / "uploads"
    settings.paths.jobs_dir = tmp_path / "jobs"
    uploads = UploadStore(settings.paths.uploads_dir, settings.limits.max_upload_bytes, settings.limits.chunk_bytes)
    upload_id = uploads.init(3, "a.mp4")
    uploads.put_chunk(upload_id, 0, b"abc")
    uploads.complete(upload_id)

    store = JobStore(settings.paths.jobs_dir)
    job = store.create({"type": "upload", "upload_id": upload_id})
    work = store.job_dir(job.id) / "work" / "stages"
    work.mkdir(parents=True)
    (work / "asr.done.json").write_text("{}", encoding="utf-8")

    def boom(*args, **kwargs):
        raise ValueError("input_value='secret transcript words'")

    monkeypatch.setattr(orchestrator, "execute_pipeline", boom)
    orchestrator.run(job.id, store, settings)

    after = store.get(job.id)
    assert after.status == "failed"
    assert "secret transcript" not in str(after.error)
    assert after.error["code"] == "valueerror"
    assert not (settings.paths.uploads_dir / upload_id).exists()
    assert not (store.job_dir(job.id) / "work").exists()

    ghost = store.create({"type": "upload", "upload_id": "test_up_1234567890123"})
    orchestrator.run(ghost.id, store, settings)
    assert store.get(ghost.id).status == "failed"
    assert store.get(ghost.id).error["code"] == "upload_missing"


def test_ingest_errors_reach_the_client_only_as_fixed_messages(tmp_path, settings, monkeypatch):
    from classeval.pipeline import orchestrator
    from classeval.pipeline.ingest import IngestError
    from classeval.server.jobs import JobStore
    from classeval.server.uploads import UploadStore

    settings.paths.uploads_dir = tmp_path / "uploads"
    settings.paths.jobs_dir = tmp_path / "jobs"
    uploads = UploadStore(settings.paths.uploads_dir, settings.limits.max_upload_bytes, settings.limits.chunk_bytes)
    upload_id = uploads.init(3, "a.mp4")
    uploads.put_chunk(upload_id, 0, b"abc")
    uploads.complete(upload_id)
    store = JobStore(settings.paths.jobs_dir)
    job = store.create({"type": "upload", "upload_id": upload_id})

    def bad_media(*args, **kwargs):
        raise IngestError(code="unreadable", message="ffprobe failed: F:/secret/path/upload.bin: Invalid data")

    monkeypatch.setattr(orchestrator, "execute_pipeline", bad_media)
    orchestrator.run(job.id, store, settings)
    err = store.get(job.id).error
    assert err["code"] == "unreadable"
    assert "secret" not in err["message"] and "upload.bin" not in err["message"]
