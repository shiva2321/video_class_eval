
import cv2
import numpy as np
import pytest

from classeval.config import load_settings
from classeval.pipeline import video_features
from classeval.schema import Diarization, FrameRef, FrameState, SpeakerTurn, Transcript


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_video_feat_token")
    return load_settings("config.yaml", env_file=None)


def test_compute_ratios_exact(settings):
    # 20 states: 10 BOARD_SHARE, 4 LESSON_VIDEO_SHARE, 6 CAMERA_OFF_CARD
    states = []
    for i in range(10):
        states.append(FrameState(t=float(i * 5), state="BOARD_SHARE", confidence=0.9))
    for i in range(4):
        states.append(FrameState(t=float(50 + i * 5), state="LESSON_VIDEO_SHARE", confidence=0.9))
    for i in range(6):
        states.append(FrameState(t=float(70 + i * 5), state="CAMERA_OFF_CARD", confidence=0.9))

    d = Diarization(turns=[], tutor=None, tutor_confidence=0.0, method="ecapa", n_speakers=0)
    tr = Transcript(language="en", language_confidence=1.0, asr_model="w", segments=[])

    facts = video_features.compute(states, d, tr, settings)

    assert facts.n_frames == 20
    assert facts.board_share_ratio == 0.5  # 10 / 20
    assert facts.lesson_video_ratio == 0.2  # 4 / 20
    assert facts.camera_off_total_ratio == 0.3  # 6 / 20


def test_camera_off_seconds_only_when_tutor_speaks(settings):
    # sampling.frame_fps = 0.2 -> frame_period = 5.0 s
    # 4 CAMERA_OFF frames at t=0, 5, 10, 15
    # Tutor speaks only from t=0 to t=10 (covering t=0, 5, 10 -> 3 frames)
    states = [
        FrameState(t=0.0, state="CAMERA_OFF_CARD", confidence=0.95),
        FrameState(t=5.0, state="CAMERA_OFF_CARD", confidence=0.95),
        FrameState(t=10.0, state="CAMERA_OFF_CARD", confidence=0.95),
        FrameState(t=15.0, state="CAMERA_OFF_CARD", confidence=0.95),
    ]
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker="TUTOR"),
        SpeakerTurn(start=12.0, end=20.0, speaker="STUDENT"),
    ]
    d = Diarization(turns=turns, tutor="TUTOR", tutor_confidence=0.9, method="ecapa", n_speakers=2)
    tr = Transcript(language="en", language_confidence=1.0, asr_model="w", segments=[])

    facts = video_features.compute(states, d, tr, settings)
    # 3 frames * 5.0s = 15.0s
    assert facts.camera_off_while_tutor_speaks_s == 15.0


def test_branded_ratio_none_when_few_frames(settings):
    # Only 2 talking-head frames during tutor speech (< 3 threshold)
    states = [
        FrameState(t=0.0, state="BRANDED_TALKING_HEAD", confidence=0.9),
        FrameState(t=5.0, state="OTHER_TALKING_HEAD", confidence=0.8),
    ]
    turns = [SpeakerTurn(start=0.0, end=10.0, speaker="TUTOR")]
    d = Diarization(turns=turns, tutor="TUTOR", tutor_confidence=0.9, method="ecapa", n_speakers=1)
    tr = Transcript(language="en", language_confidence=1.0, asr_model="w", segments=[])

    facts = video_features.compute(states, d, tr, settings)
    assert facts.branded_ratio_tutor_speaking is None


def test_luma_only_from_talking_head(tmp_path, settings):
    # One talking-head frame (mean luma 100), one board-share frame (mean luma 200)
    img_th = tmp_path / "th.jpg"
    img_board = tmp_path / "board.jpg"

    cv2.imwrite(str(img_th), np.full((100, 100, 3), 100, dtype=np.uint8))
    cv2.imwrite(str(img_board), np.full((100, 100, 3), 200, dtype=np.uint8))

    states = [
        FrameState(t=0.0, state="BRANDED_TALKING_HEAD", confidence=0.9),
        FrameState(t=5.0, state="BOARD_SHARE", confidence=0.9),
    ]
    frames = [
        FrameRef(t=0.0, path=img_th),
        FrameRef(t=5.0, path=img_board),
    ]
    d = Diarization(turns=[], tutor=None, tutor_confidence=0.0, method="ecapa", n_speakers=0)
    tr = Transcript(language="en", language_confidence=1.0, asr_model="w", segments=[])

    facts = video_features.compute(states, d, tr, settings, frames=frames)
    assert facts.talking_head_luma_mean == pytest.approx(100.0, 1.0)
