import cv2
import numpy as np
import pytest

from classeval.config import load_settings
from classeval.pipeline import scoring, thumbnail
from classeval.schema import FrameRef
from tests.pipeline.test_scoring import _make_dummy_inputs


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_thumbnail_token")
    return load_settings("config.yaml", env_file=None)


def _template() -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(thumbnail.LOGO_PATH)), cv2.COLOR_BGR2GRAY)


def _noise(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (300, 400), dtype=np.uint8)


def _with_logo(seed: int, scale: float = 0.28) -> np.ndarray:
    image = _noise(seed)
    tpl = _template()
    small = cv2.resize(tpl, (int(tpl.shape[1] * scale), int(tpl.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    image[20 : 20 + small.shape[0], 30 : 30 + small.shape[1]] = small
    return image


def test_the_logo_is_found_at_a_small_scale_and_absent_from_noise(settings):
    scales = settings.screen_state.thumb_scales
    assert thumbnail.best_logo_score(_with_logo(1), _template(), scales) > 0.9
    assert thumbnail.best_logo_score(_noise(2), _template(), scales) < settings.screen_state.thumb_logo_threshold


def _tiles(tmp_path, images, every_s=10.0):
    refs = []
    for i, image in enumerate(images):
        path = tmp_path / f"tile_{i:05d}.jpg"
        cv2.imwrite(str(path), image)
        refs.append(FrameRef(t=i * every_s, path=path))
    return refs


def test_share_counts_the_moments_with_the_backdrop(tmp_path, settings):
    settings.screen_state.tile_every_s = settings.screen_state.thumb_every_s = 10.0  # every tile is looked at
    tiles = _tiles(tmp_path, [_with_logo(1), _noise(2), _with_logo(3), _noise(4)])
    assert thumbnail.branded_share(tiles, settings) == (4, 0.5)


def test_only_every_nth_tile_is_looked_at_when_tiles_are_cut_more_often_than_they_are_judged(tmp_path, settings):
    settings.screen_state.tile_every_s, settings.screen_state.thumb_every_s = 10.0, 20.0
    tiles = _tiles(tmp_path, [_with_logo(1), _noise(2), _with_logo(3), _noise(4)])
    assert thumbnail.branded_share(tiles, settings) == (2, 1.0)  # tiles 0 and 2


def test_no_tiles_and_unreadable_tiles_give_no_measurement(tmp_path, settings):
    assert thumbnail.branded_share([], settings) == (0, None)
    broken = tmp_path / "tile_00000.jpg"
    broken.write_bytes(b"not an image")
    assert thumbnail.branded_share([FrameRef(t=0.0, path=broken)], settings) == (0, None)


def _r1(settings, **video_fields):
    media, tr, d, audio, video, tfacts, checklist = _make_dummy_inputs(branded_ratio=None)
    video = video.model_copy(update=video_fields)
    res = scoring.score(media, tr, d, audio, video, tfacts, checklist, settings)
    return next(x for x in res.dimensions if x.id == "R1")


def test_r1_uses_the_tutor_tile_when_the_full_frame_view_is_missing(settings):
    assert _r1(settings, thumb_crops=40, thumb_branded_share=0.30).score == 100.0
    assert _r1(settings, thumb_crops=40, thumb_branded_share=0.075).score == pytest.approx(50.0)
    assert _r1(settings, thumb_crops=40, thumb_branded_share=0.0).score == 0.0


def test_r1_is_not_assessed_with_too_few_tile_samples(settings):
    dim = _r1(settings, thumb_crops=2, thumb_branded_share=1.0)
    assert dim.score is None


def test_r1_takes_the_better_of_the_two_views(settings):
    dim = _r1(settings, branded_ratio_tutor_speaking=0.7, thumb_crops=40, thumb_branded_share=0.0)
    assert dim.score == 100.0
