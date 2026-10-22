"""Looks at the tutor: is the tutor on camera while speaking, is the face clearly visible, is the light good?

A local vision model answers three closed questions about the tutor's video at moments when the diarization says the
tutor is speaking. While a board or video is shared the tutor is a small top-right tile, which is cropped at native
resolution; while the tutor is full-screen the whole frame is used. Only categories leave this module, never what the
pictures show. Everything is best effort: when the model is unavailable the dimension falls back to the older measure.
"""

import logging
from pathlib import Path
from typing import Any

from classeval.config import Settings
from classeval.pipeline.llm import LLMBadOutput, LLMUnavailable, OllamaClient
from classeval.schema import Diarization, FrameRef, FrameState

logger = logging.getLogger("classeval.visual")

MIN_TURN_S = 3.0  # only turns long enough that the tile surely shows the speaker
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
    states: list[FrameState],
    d: Diarization,
    frames: list[FrameRef],
    tiles: list[FrameRef],
    settings: Settings,
    client: OllamaClient,
) -> dict[str, Any]:
    cfg = settings.screen_state
    answers: list[dict[str, str]] = []
    camera_off = 0
    for t in pick_moments(d, cfg.visual_samples):
        state = _nearest(states, t)
        if state is not None and state.state == "CAMERA_OFF_CARD":
            camera_off += 1
            continue
        frame = _nearest(frames, t) if state is not None and state.state in _FULL_FRAME_STATES else None
        tile = _nearest(tiles, t)
        if frame is not None:
            path, prompt = frame.path, PROMPT_FULL
        elif tile is not None and abs(tile.t - t) <= cfg.tile_every_s:
            path, prompt = tile.path, PROMPT_TILE
        else:
            continue
        try:
            image = Path(path).read_bytes()
            answer = _ask(client, image, prompt)
        except OSError as exc:
            logger.warning("tutor visibility check could not read a picture: %s", type(exc).__name__)
            continue
        except LLMUnavailable:
            logger.warning("the vision model is unavailable; skipping the tutor visibility check")
            return {}
        if answer is not None:
            answers.append(answer)
    return summarize(answers, camera_off, cfg.visual_min_samples)
