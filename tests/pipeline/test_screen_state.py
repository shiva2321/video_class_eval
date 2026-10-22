
import pytest

from classeval.config import load_settings
from classeval.pipeline import screen_state
from classeval.schema import FrameRef
from tests.fixtures.frames import draw_frame


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_screen_state_token")
    return load_settings("config.yaml", env_file=None)


def test_flat_grey_card_is_camera_off(tmp_path, settings):
    img_path = tmp_path / "camera_off.jpg"
    draw_frame("camera_off_card", img_path)

    frames = [FrameRef(t=0.0, path=img_path)]
    states = screen_state.classify_frames(frames, settings)

    assert len(states) == 1
    assert states[0].state == "CAMERA_OFF_CARD"
    assert states[0].confidence > 0.8


def test_logo_pasted_on_navy_is_branded(tmp_path, settings):
    img_path1 = tmp_path / "branded_05.jpg"
    img_path2 = tmp_path / "branded_07.jpg"
    draw_frame("branded_head", img_path1, scale=0.5)
    draw_frame("branded_head", img_path2, scale=0.7)

    frames = [
        FrameRef(t=0.0, path=img_path1),
        FrameRef(t=5.0, path=img_path2),
    ]
    states = screen_state.classify_frames(frames, settings)

    assert len(states) == 2
    assert states[0].state == "BRANDED_TALKING_HEAD"
    assert states[1].state == "BRANDED_TALKING_HEAD"
    assert states[0].confidence >= 0.6
    assert states[1].confidence >= 0.6


@pytest.mark.gpu
def test_checkerboard_is_board_share(tmp_path, settings):
    img_path = tmp_path / "board.jpg"
    draw_frame("board_grid", img_path)

    frames = [FrameRef(t=0.0, path=img_path)]
    states = screen_state.classify_frames(frames, settings)

    assert len(states) == 1
    assert states[0].state == "BOARD_SHARE"
    assert states[0].confidence > 0.35


def test_unknown_for_noise(tmp_path, settings):
    img_path = tmp_path / "noise.jpg"
    draw_frame("noise", img_path)

    frames = [FrameRef(t=0.0, path=img_path)]
    states = screen_state.classify_frames(frames, settings)

    assert len(states) == 1
    # Noise should not match flat grey or logo
    assert states[0].state in ("UNKNOWN", "OTHER_SHARE")
