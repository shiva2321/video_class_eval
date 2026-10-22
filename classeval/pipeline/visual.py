"""Looks at the tutor: is the tutor on camera while speaking, is the face clearly visible, is the light good?

A local vision model answers three closed questions about the tutor's video at moments when the diarization says the
tutor is speaking. While a board or video is shared the tutor is a small top-right tile, which is cropped at native
resolution; while the tutor is full-screen the whole frame is used. Only categories leave this module, never what the
pictures show. Everything is best effort: when the model is unavailable the dimension falls back to the older measure.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from classeval.config import Settings
from classeval.pipeline.ingest import _ensure_ffmpeg_path
from classeval.pipeline.llm import LLMBadOutput, LLMUnavailable, OllamaClient
from classeval.schema import Diarization, FrameRef, FrameState

logger = logging.getLogger("classeval.visual")

MIN_TURN_S = 3.0  # only turns long enough that the tile surely shows the speaker
FFMPEG_TIMEOUT_S = 120.0
_FULL_FRAME_STATES = {"BRANDED_TALKING_HEAD", "OTHER_TALKING_HEAD"}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "shows": {"type": "string", "enum": ["person_on_camera", "name_card_or_blank", "shared_content", "other"]},
        "face_clearly_visible": {"type": "string", "enum": ["yes", "no"]},
        "lighting": {"type": "string", "enum": ["good", "dim", "very_dark", "overexposed", "not_applicable"]},
    },
    "required": ["shows", "face_clearly_visible", "lighting"],
}

_QUESTIONS = (
    "shows: person_on_camera if a live camera view of a person is visible; name_card_or_blank if it is a name card or a "
    "grey or black placeholder; shared_content if it is part of a shared screen (board, slides, video); other otherwise. "
    "face_clearly_visible: yes only if one person's face is unobstructed and large enough to read their expression. "
    "lighting of that person: good, dim, very_dark, overexposed, or not_applicable."
)
PROMPT_TILE = (
    "This is a crop of the top-right corner of an online class recording. Describe ONLY what the small video tile shows. "
    + _QUESTIONS
)
PROMPT_FULL = "This is one frame of an online class recording in which one camera fills the screen. " + _QUESTIONS
SYSTEM = "You answer closed questions about one picture. Output only JSON."


def pick_moments(d: Diarization, n: int) -> list[float]:
    """Midpoints of up to n tutor turns, evenly spread over the class."""
    turns = [t for t in d.turns if d.tutor is not None and t.speaker == d.tutor and t.end - t.start >= MIN_TURN_S]
    if not turns:
        return []
    step = max(1, len(turns) // n)
    return [(t.start + t.end) / 2.0 for t in turns[::step][:n]]


def _nearest(items: list, t: float):
    return min(items, key=lambda x: abs(x.t - t)) if items else None


def _tile_crop(video: Path, t: float, out: Path, settings: Settings) -> bytes | None:
    _ensure_ffmpeg_path()
    x, y, w, h = settings.screen_state.thumb_region
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-ss", f"{t:.1f}", "-i", str(video), "-frames:v", "1",
        "-vf", f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}", "-q:v", "3", str(out),
    ]  # fmt: skip
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S)
        return out.read_bytes()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("tile crop failed: %s", type(exc).__name__)
        return None


def _ask(client: OllamaClient, image: bytes, prompt: str) -> dict[str, str] | None:
    try:
        answer = client.chat_json(SYSTEM, prompt, SCHEMA, images=[image])
    except LLMBadOutput:
        return None
    return answer if isinstance(answer, dict) and all(k in answer for k in SCHEMA["required"]) else None


def summarize(answers: list[dict[str, str]], camera_off: int, min_samples: int) -> dict[str, Any]:
    """Ratios over the sampled tutor-speaking moments; empty of ratios when there were too few to say anything."""
    n = len(answers) + camera_off
    if n < min_samples:
        return {"visual_samples": n}
    on_camera = [a for a in answers if a["shows"] == "person_on_camera"]
    visible = sum(1 for a in on_camera if a["face_clearly_visible"] == "yes")
    good_light = sum(1 for a in on_camera if a["lighting"] == "good")
    return {
        "visual_samples": n,
        "tutor_visible_ratio": visible / n,
        "tutor_lighting_good_ratio": good_light / len(on_camera) if on_camera else None,
    }


def analyze(
    video: Path,
    work_dir: Path,
    states: list[FrameState],
    d: Diarization,
    frames: list[FrameRef],
    settings: Settings,
    client: OllamaClient,
) -> dict[str, Any]:
    cfg = settings.screen_state
    crops_dir = work_dir / "visual"
    shutil.rmtree(crops_dir, ignore_errors=True)
    crops_dir.mkdir(parents=True)
    answers: list[dict[str, str]] = []
    camera_off = 0
    try:
        for i, t in enumerate(pick_moments(d, cfg.visual_samples)):
            state = _nearest(states, t)
            if state is not None and state.state == "CAMERA_OFF_CARD":
                camera_off += 1
                continue
            frame = _nearest(frames, t) if state is not None and state.state in _FULL_FRAME_STATES else None
            if frame is not None:
                image, prompt = Path(frame.path).read_bytes(), PROMPT_FULL
            else:
                image, prompt = _tile_crop(video, t, crops_dir / f"c{i}.jpg", settings), PROMPT_TILE
            if image is None:
                continue
            try:
                answer = _ask(client, image, prompt)
            except LLMUnavailable:
                logger.warning("the vision model is unavailable; skipping the tutor visibility check")
                return {}
            if answer is not None:
                answers.append(answer)
    except OSError as exc:
        logger.warning("tutor visibility check failed: %s", type(exc).__name__)
        return {}
    finally:
        shutil.rmtree(crops_dir, ignore_errors=True)
    return summarize(answers, camera_off, cfg.visual_min_samples)
