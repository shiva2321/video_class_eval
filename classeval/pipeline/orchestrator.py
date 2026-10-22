import concurrent.futures
import gc
import logging
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel

from classeval.config import Settings
from classeval.pipeline import (
    asr as asr_mod,
)
from classeval.pipeline import (
    audio_features as audio_features_mod,
)
from classeval.pipeline import (
    checklist as checklist_mod,
)
from classeval.pipeline import (
    class_stats as class_stats_mod,
)
from classeval.pipeline import (
    diarize as diarize_mod,
)
from classeval.pipeline import (
    ingest as ingest_mod,
)
from classeval.pipeline import (
    report as report_mod,
)
from classeval.pipeline import (
    scoring as scoring_mod,
)
from classeval.pipeline import (
    screen_state as screen_state_mod,
)
from classeval.pipeline import (
    transcript_features as transcript_features_mod,
)
from classeval.pipeline import (
    video_features as video_features_mod,
)
from classeval.pipeline.checklist import ChecklistResult
from classeval.pipeline.ingest import IngestError, Prepared
from classeval.pipeline.llm import OllamaClient
from classeval.pipeline.scoring import ScoreResult
from classeval.schema import (
    AudioFacts,
    Diarization,
    FrameState,
    Report,
    Transcript,
    TranscriptFacts,
    VideoFacts,
)
from classeval.server.jobs import JobStore
from classeval.server.uploads import UploadStore

logger = logging.getLogger("classeval.orchestrator")


@dataclass
class StageFns:
    ingest: Callable[..., Any] = ingest_mod.prepare
    asr: Callable[..., Any] = asr_mod.transcribe
    classify_frames: Callable[..., Any] = screen_state_mod.classify_frames
    diarize: Callable[..., Any] = diarize_mod.diarize
    assign_speakers: Callable[..., Any] = diarize_mod.assign_speakers
    audio: Callable[..., Any] = audio_features_mod.compute
    video: Callable[..., Any] = video_features_mod.compute
    tfacts: Callable[..., Any] = transcript_features_mod.compute
    understand: Callable[..., Any] = checklist_mod.extract
    score: Callable[..., Any] = scoring_mod.score
    report_summary: Callable[..., Any] = report_mod.summarize
    report_build: Callable[..., Any] = report_mod.build_report
    report_render: Callable[..., Any] = report_mod.render_html
    cleanup_gpu: Callable[[], None] | None = None


class _FrameStatesCache(BaseModel):
    states: list[FrameState]


class _DiarizeCache(BaseModel):
    diarization: Diarization
    transcript: Transcript


class _UnderstandCache(BaseModel):
    tfacts: TranscriptFacts
    checklist: ChecklistResult


def _clean_gpu(custom_cleanup: Callable[[], None] | None = None) -> None:
    if custom_cleanup is not None:
        custom_cleanup()
    else:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def execute_pipeline(
    input_path: Path,
    job_dir: Path,
    settings: Settings,
    stages: StageFns | None = None,
    job_id: str = "pipeline_run",
    store: JobStore | None = None,
) -> Report:
    """Run all pipeline stages with caching, timeouts, concurrency, and monotonic progress."""
    fns = stages if stages is not None else StageFns()
    stages_dir = job_dir / "work" / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    timings_s: dict[str, float] = {}

    def set_stage(st: str, prog: float) -> None:
        if store is not None:
            store.set_stage(job_id, st, prog)

    # 1. Ingest stage
    set_stage("ingest", 0.10)
    ingest_cache_path = stages_dir / "ingest.done.json"
    if ingest_cache_path.exists():
        prepared = Prepared.model_validate_json(ingest_cache_path.read_text(encoding="utf-8"))
    else:
        t0 = time.monotonic()
        prepared = fns.ingest(input_path, job_dir / "work" / "ingest", settings)
        timings_s["ingest"] = round(time.monotonic() - t0, 2)
        ingest_cache_path.write_text(prepared.model_dump_json(indent=2), encoding="utf-8")

    # Audio track check: video-only recordings fail cleanly
    if not prepared.info.has_audio or prepared.wav is None:
        raise IngestError(code="no_audio", message="This recording has no audio track.")

    # 2. ASR & Frame Classification concurrently
    set_stage("asr", 0.30)
    asr_cache_path = stages_dir / "asr.done.json"
    frames_cache_path = stages_dir / "classify_frames.done.json"

    tr: Transcript | None = None
    frame_states: list[FrameState] | None = None

    if asr_cache_path.exists():
        tr = Transcript.model_validate_json(asr_cache_path.read_text(encoding="utf-8"))
    if frames_cache_path.exists():
        frame_states = _FrameStatesCache.model_validate_json(frames_cache_path.read_text(encoding="utf-8")).states

    if tr is None or (prepared.info.has_video and prepared.frames and frame_states is None):
        t0 = time.monotonic()
        if prepared.info.has_video and prepared.frames:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                fut_asr = executor.submit(fns.asr, prepared.wav, settings) if tr is None else None
                fut_clip = (
                    executor.submit(fns.classify_frames, prepared.frames, settings)
                    if frame_states is None
                    else None
                )
                if fut_asr:
                    tr = fut_asr.result()
                if fut_clip:
                    frame_states = fut_clip.result()
        else:
            if tr is None:
                tr = fns.asr(prepared.wav, settings)
            frame_states = []

        timings_s["asr"] = round(time.monotonic() - t0, 2)
        if tr is not None:
            asr_cache_path.write_text(tr.model_dump_json(indent=2), encoding="utf-8")
        if frame_states is not None:
            frames_cache_path.write_text(
                _FrameStatesCache(states=frame_states).model_dump_json(indent=2),
                encoding="utf-8",
            )
        _clean_gpu(fns.cleanup_gpu)
    elif frame_states is None:
        frame_states = []

    # 3. Diarization stage
    set_stage("diarize", 0.50)
    diarize_cache_path = stages_dir / "diarize.done.json"
    if diarize_cache_path.exists():
        cached_diar = _DiarizeCache.model_validate_json(diarize_cache_path.read_text(encoding="utf-8"))
        d = cached_diar.diarization
        tr = cached_diar.transcript
    else:
        t0 = time.monotonic()
        d = fns.diarize(prepared.wav, settings)
        tr = fns.assign_speakers(tr, d)
        timings_s["diarize"] = round(time.monotonic() - t0, 2)
        diarize_cache_path.write_text(
            _DiarizeCache(diarization=d, transcript=tr).model_dump_json(indent=2),
            encoding="utf-8",
        )
        _clean_gpu(fns.cleanup_gpu)

    # 4. Audio features stage
    set_stage("audio", 0.60)
    audio_cache_path = stages_dir / "audio.done.json"
    if audio_cache_path.exists():
        audio_facts = AudioFacts.model_validate_json(audio_cache_path.read_text(encoding="utf-8"))
    else:
        t0 = time.monotonic()
        audio_facts = fns.audio(prepared.wav, tr, d)
        timings_s["audio"] = round(time.monotonic() - t0, 2)
        audio_cache_path.write_text(audio_facts.model_dump_json(indent=2), encoding="utf-8")

    # 5. Video features stage
    set_stage("video", 0.70)
    video_cache_path = stages_dir / "video.done.json"
    if video_cache_path.exists():
        video_facts = VideoFacts.model_validate_json(video_cache_path.read_text(encoding="utf-8"))
    else:
        t0 = time.monotonic()
        if prepared.info.has_video and frame_states:
            video_facts = fns.video(frame_states, d, tr, settings, frames=prepared.frames)
        else:
            video_facts = VideoFacts(
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
        timings_s["video"] = round(time.monotonic() - t0, 2)
        video_cache_path.write_text(video_facts.model_dump_json(indent=2), encoding="utf-8")

    # 6. Understand (Transcript facts + LLM checklist) stage
    set_stage("understand", 0.85)
    understand_cache_path = stages_dir / "understand.done.json"
    if understand_cache_path.exists():
        cached_und = _UnderstandCache.model_validate_json(understand_cache_path.read_text(encoding="utf-8"))
        tfacts = cached_und.tfacts
        checklist_res = cached_und.checklist
    else:
        t0 = time.monotonic()
        tfacts = fns.tfacts(tr, d, settings)
        client = OllamaClient(base_url=settings.ollama.base_url, model=settings.models.llm)
        try:
            checklist_res = fns.understand(tr, d, tfacts, client, settings)
        finally:
            try:
                client.unload()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Ollama unload error: %s", exc)
            _clean_gpu(fns.cleanup_gpu)
        timings_s["understand"] = round(time.monotonic() - t0, 2)
        understand_cache_path.write_text(
            _UnderstandCache(tfacts=tfacts, checklist=checklist_res).model_dump_json(indent=2),
            encoding="utf-8",
        )

    # 7. Scoring stage
    set_stage("score", 0.92)
    score_cache_path = stages_dir / "score.done.json"
    if score_cache_path.exists():
        score_res = ScoreResult.model_validate_json(score_cache_path.read_text(encoding="utf-8"))
    else:
        t0 = time.monotonic()
        score_res = fns.score(prepared.info, tr, d, audio_facts, video_facts, tfacts, checklist_res, settings)
        timings_s["score"] = round(time.monotonic() - t0, 2)
        score_cache_path.write_text(score_res.model_dump_json(indent=2), encoding="utf-8")

    # 8. Report stage
    set_stage("report", 0.98)
    t0 = time.monotonic()
    summary = fns.report_summary(score_res, checklist_res)

    versions = {
        "classeval": "0.1.0",
        "whisper": settings.models.whisper,
        "clip": settings.models.clip,
        "ecapa": settings.models.ecapa,
        "llm": settings.models.llm,
    }
    if torch is not None:
        versions["torch"] = getattr(torch, "__version__", "unknown")
    try:
        import faster_whisper

        versions["faster_whisper"] = getattr(faster_whisper, "__version__", "unknown")
    except (ImportError, AttributeError):
        pass

    rep = fns.report_build(
        job_id,
        prepared.info,
        tr,
        d,
        audio_facts,
        video_facts,
        tfacts,
        checklist_res,
        score_res,
        timings_s,
        versions,
        summary,
    )
    rep.stats = class_stats_mod.compute(tr, d, audio_facts, video_facts, tfacts, prepared.info)
    html_content = fns.report_render(rep)
    timings_s["report"] = round(time.monotonic() - t0, 2)

    # Write report.json and report.html under job_dir
    (job_dir / "report.json").write_text(rep.model_dump_json(indent=2), encoding="utf-8")
    (job_dir / "report.html").write_text(html_content, encoding="utf-8")

    return rep



_UPLOAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
FRIENDLY_FAILURE = "The assessment could not be completed. Please try another recording or try again later."
# Fixed, client-safe texts: raw error messages can contain server paths or transcript text.
_INGEST_MESSAGES = {
    "unreadable": "This file could not be read as audio or video. Please upload a normal MP4, MOV, MKV or WEBM recording.",
    "no_streams": "This file contains no audio or video.",
    "too_long": "This recording is longer than the supported maximum.",
    "too_short_zero": "This recording has no playable duration.",
    "timeout": "Reading this file took too long. Please try a different export of the recording.",
    "no_audio": "This recording has no audio track.",
}


def _purge_inputs(settings: Settings, upload_id: str | None, job_dir: Path) -> None:
    """When a job ends, the recording, the extracted audio/frames and the transcripts are not kept (only the report)."""
    if upload_id and _UPLOAD_ID_RE.fullmatch(upload_id):
        shutil.rmtree(settings.paths.uploads_dir / upload_id, ignore_errors=True)
    shutil.rmtree(job_dir / "work", ignore_errors=True)


def run(
    job_id: str,
    store: JobStore,
    settings: Settings,
    stages: StageFns | None = None,
) -> None:
    """Execute the full assessment pipeline for a claimed job."""
    job = store.get(job_id)
    if not job:
        logger.error("Job not found in store")
        return

    job_dir = store.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    upload_id = job.source.get("upload_id")

    up_store = UploadStore(settings.paths.uploads_dir, settings.limits.max_upload_bytes, settings.limits.chunk_bytes)
    input_path = up_store.final_path(upload_id) if upload_id else None
    if input_path is None:
        store.update(job_id, status="failed", error={"code": "upload_missing", "message": "Upload file not found"})
        return

    try:
        execute_pipeline(
            input_path=input_path,
            job_dir=job_dir,
            settings=settings,
            stages=stages,
            job_id=job_id,
            store=store,
        )
    except IngestError as exc:
        logger.warning("Ingest error: code=%s", exc.code)
        store.update(
            job_id,
            status="failed",
            error={"code": exc.code, "message": _INGEST_MESSAGES.get(exc.code, FRIENDLY_FAILURE)},
        )
    except Exception as exc:  # noqa: BLE001
        code = type(exc).__name__.lower()
        # Class name only: exception messages (e.g. validation errors) can contain transcript text.
        logger.error("Pipeline failure: code=%s", code)
        store.update(job_id, status="failed", error={"code": code, "message": FRIENDLY_FAILURE})
    else:
        store.update(job_id, status="done", stage="done", progress=1.0)
    finally:
        _purge_inputs(settings, upload_id, job_dir)
