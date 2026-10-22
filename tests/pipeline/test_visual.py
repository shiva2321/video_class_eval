import pytest

from classeval.config import load_settings
from classeval.pipeline import scoring, visual
from classeval.pipeline.llm import LLMBadOutput, LLMUnavailable
from classeval.schema import Diarization, FrameRef, FrameState, SpeakerTurn
from tests.pipeline.test_scoring import _make_dummy_inputs


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_visual_token")
    return load_settings("config.yaml", env_file=None)


def _diar(n_turns=20, tutor="T"):
    turns = []
    for i in range(n_turns):
        turns.append(SpeakerTurn(start=i * 20.0, end=i * 20.0 + 8.0, speaker=tutor))
        turns.append(SpeakerTurn(start=i * 20.0 + 9, end=i * 20.0 + 12, speaker="S1"))
    return Diarization(turns=turns, tutor=tutor, tutor_confidence=1.0, method="ecapa", n_speakers=2)


ON = {"shows": "person_on_camera", "face_clearly_visible": "yes", "lighting": "good"}
OFF = {"shows": "name_card_or_blank", "face_clearly_visible": "no", "lighting": "not_applicable"}
DARK = {"shows": "person_on_camera", "face_clearly_visible": "yes", "lighting": "dim"}


def test_moments_are_tutor_turns_spread_over_the_class():
    moments = visual.pick_moments(_diar(40), 16)
    assert len(moments) == 16
    assert all(m % 20 == pytest.approx(4.0) for m in moments)  # midpoints of the tutor's 8 s turns, never the students'
    assert moments == sorted(moments) and moments[-1] > 400


def test_short_turns_and_a_missing_tutor_give_no_moments():
    d = Diarization(turns=[SpeakerTurn(start=0, end=1.5, speaker="T")], tutor="T", tutor_confidence=1, method="ecapa", n_speakers=1)
    assert visual.pick_moments(d, 16) == []
    assert visual.pick_moments(_diar().model_copy(update={"tutor": None}), 16) == []


def test_summary_counts_visible_faces_over_all_sampled_moments():
    out = visual.summarize([ON, ON, ON, OFF, DARK], camera_off=3, min_samples=4)
    assert out["visual_samples"] == 8
    assert out["tutor_visible_ratio"] == pytest.approx(4 / 8)  # three good + one dim, the name card and 3 camera-off cards are not
    assert out["tutor_lighting_good_ratio"] == pytest.approx(3 / 4)


def test_summary_with_too_few_samples_makes_no_claim():
    assert visual.summarize([ON, ON], camera_off=1, min_samples=4) == {"visual_samples": 3}


def test_summary_when_the_tutor_was_never_on_camera():
    out = visual.summarize([OFF, OFF, OFF, OFF], camera_off=0, min_samples=4)
    assert out["tutor_visible_ratio"] == 0.0 and out["tutor_lighting_good_ratio"] is None


class FakeClient:
    def __init__(self, answers):
        self.answers, self.calls, self.images = list(answers), 0, []

    def chat_json(self, system, user, schema, images=None):
        self.images.append(images)
        step = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        if isinstance(step, type) and issubclass(step, Exception):
            raise step("scripted")
        return step


def _inputs(tmp_path, state="BOARD_SHARE"):
    frames, states = [], []
    for i in range(0, 420, 5):
        path = tmp_path / f"f{i}.jpg"
        path.write_bytes(b"jpg")
        frames.append(FrameRef(t=float(i), path=path))
        states.append(FrameState(t=float(i), state=state, confidence=0.9))
    return states, frames


def test_camera_off_cards_count_without_asking_the_model(monkeypatch, tmp_path, settings):
    states, frames = _inputs(tmp_path, state="CAMERA_OFF_CARD")
    client = FakeClient([ON])
    out = visual.analyze(tmp_path / "v.mp4", tmp_path, states, _diar(), frames, settings, client)
    assert client.calls == 0
    assert out["tutor_visible_ratio"] == 0.0 and out["visual_samples"] == 16


def test_full_screen_tutor_moments_use_the_whole_frame_and_shared_screens_use_the_tile(monkeypatch, tmp_path, settings):
    tiles = []
    monkeypatch.setattr(visual, "_tile_crop", lambda video, t, out, s: tiles.append(t) or b"tile")
    states, frames = _inputs(tmp_path, state="OTHER_TALKING_HEAD")
    client = FakeClient([ON])
    visual.analyze(tmp_path / "v.mp4", tmp_path, states, _diar(), frames, settings, client)
    assert tiles == [] and all(img == [b"jpg"] for img in client.images)

    states, frames = _inputs(tmp_path, state="BOARD_SHARE")
    client = FakeClient([ON])
    visual.analyze(tmp_path / "v.mp4", tmp_path, states, _diar(), frames, settings, client)
    assert len(tiles) == 16 and all(img == [b"tile"] for img in client.images)


def test_an_unusable_answer_skips_that_moment_and_an_unavailable_model_gives_nothing(monkeypatch, tmp_path, settings):
    monkeypatch.setattr(visual, "_tile_crop", lambda *a: b"tile")
    states, frames = _inputs(tmp_path)
    out = visual.analyze(tmp_path / "v.mp4", tmp_path, states, _diar(), frames, settings, FakeClient([LLMBadOutput, ON]))
    assert out["visual_samples"] == 15  # the first moment is skipped, the rest are counted
    assert visual.analyze(tmp_path / "v.mp4", tmp_path, states, _diar(), frames, settings, FakeClient([LLMUnavailable])) == {}


def test_incomplete_answers_are_ignored(monkeypatch, tmp_path, settings):
    monkeypatch.setattr(visual, "_tile_crop", lambda *a: b"tile")
    states, frames = _inputs(tmp_path)
    out = visual.analyze(tmp_path / "v.mp4", tmp_path, states, _diar(), frames, settings, FakeClient([{"shows": "person_on_camera"}]))
    assert out == {"visual_samples": 0}


def test_no_crops_are_left_behind(monkeypatch, tmp_path, settings):
    monkeypatch.setattr(visual, "_tile_crop", lambda *a: b"tile")
    states, frames = _inputs(tmp_path)
    visual.analyze(tmp_path / "v.mp4", tmp_path, states, _diar(), frames, settings, FakeClient([ON]))
    assert not (tmp_path / "visual").exists()


def _r2(settings, **fields):
    media, tr, d, audio, video, tfacts, checklist = _make_dummy_inputs()
    video = video.model_copy(update=fields)
    res = scoring.score(media, tr, d, audio, video, tfacts, checklist, settings)
    return next(x for x in res.dimensions if x.id == "R2")


def test_r2_follows_how_often_the_tutor_face_was_clearly_visible(settings):
    assert _r2(settings, visual_samples=16, tutor_visible_ratio=0.8, tutor_lighting_good_ratio=1.0).score == 100.0
    assert _r2(settings, visual_samples=16, tutor_visible_ratio=0.35, tutor_lighting_good_ratio=1.0).score == pytest.approx(50.0)
    assert _r2(settings, visual_samples=16, tutor_visible_ratio=0.05, tutor_lighting_good_ratio=None).score == 0.0


def test_r2_is_reduced_when_the_light_was_mostly_poor(settings):
    dim = _r2(settings, visual_samples=16, tutor_visible_ratio=0.8, tutor_lighting_good_ratio=0.25)
    assert dim.score == pytest.approx(80.0)
    assert any("light was good in 25%" in e.text for e in dim.evidence)


def test_r2_falls_back_to_the_camera_off_measure_without_the_vision_check(settings):
    dim = _r2(settings, visual_samples=0, tutor_visible_ratio=None)
    assert dim.score == 100.0 and "camera was off" in dim.evidence[0].text
