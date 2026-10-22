from pathlib import Path

import pytest

from classeval.pipeline import ingest
from scripts.make_degraded import make_degraded, make_long
from tests.fixtures.clips import make_clip


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("variants") / "base.mp4"
    make_clip(path, seconds=12.0)
    return path


@pytest.mark.parametrize("kind,height", [("scale720", 720), ("scale480", 480)])
def test_scaled_copies_have_the_requested_height_and_keep_the_length(clip, tmp_path, kind, height):
    original = ingest.probe(clip)
    out = ingest.probe(make_degraded(clip, tmp_path, kind))
    assert out.height == height and out.has_audio and abs(out.duration_s - original.duration_s) < 1.0


def test_first20_keeps_a_short_recording_whole_and_cuts_a_long_one(clip, tmp_path):
    assert abs(ingest.probe(make_degraded(clip, tmp_path, "first20")).duration_s - ingest.probe(clip).duration_s) < 1.0


def test_concatenation_adds_the_lengths_up(clip, tmp_path):
    joined = ingest.probe(make_long([clip, clip, clip], tmp_path))
    assert abs(joined.duration_s - 3 * ingest.probe(clip).duration_s) < 2.0
