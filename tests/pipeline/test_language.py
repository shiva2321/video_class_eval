from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from classeval.config import load_settings
from classeval.pipeline import asr, class_stats, report, scoring, transcript_features
from classeval.pipeline.checklist import extract
from classeval.pipeline.language import text_checks_supported
from classeval.schema import Diarization, Segment, Transcript, TranscriptFacts
from tests.pipeline import test_class_stats, test_report, test_scoring


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_language_token")
    return load_settings("config.yaml", env_file=None)


def _tr(language: str, conf: float = 0.95) -> Transcript:
    seg = Segment(start=1, end=3, text="Hello everyone welcome to class", speaker="S0")
    return Transcript(language=language, language_confidence=conf, asr_model="t", segments=[seg])


@pytest.mark.parametrize(
    "language,conf,expected",
    [("en", 0.95, True), ("en", 0.4, False), ("ml", 0.95, False), ("hi", 0.99, False)],
)
def test_text_checks_only_for_confident_english(settings, language, conf, expected):
    assert text_checks_supported(_tr(language, conf), settings) is expected


def test_other_languages_mark_text_dimensions_not_assessed_but_still_score_the_rest(settings):
    media, tr, d, audio, video, tfacts, checklist = test_scoring._make_dummy_inputs()
    tr = tr.model_copy(update={"language": "ml", "language_confidence": 0.97})
    res = scoring.score(media, tr, d, audio, video, tfacts, checklist, settings)
    by_id = {x.id: x for x in res.dimensions}
    assert "language_unsupported" in res.flags
    for dim in ("R5", "R10", "R11"):
        assert by_id[dim].score is None
    assert by_id["R4"].score is not None  # judged from the start time only
    for dim in ("R2", "R3", "R8", "R12"):  # language-independent measurements stay
        assert by_id[dim].score is not None
    assert res.coverage < 1.0


def test_the_checklist_does_not_call_the_model_for_an_unsupported_language(settings):
    class Boom:
        def chat_json(self, *args, **kwargs):
            raise AssertionError("the model must not judge a transcript it cannot be trusted on")

    d = Diarization(turns=[], tutor="S0", tutor_confidence=1.0, method="ecapa", n_speakers=2)
    facts = TranscriptFacts(detections={}, distinct_responders=0, question_reply_rate=None, question_count=0)
    res = extract(_tr("ml"), d, facts, Boom(), settings)
    assert res.degraded
    assert all(item.verdict == "not_observed" and not item.evidence for item in res.items.values())


def test_word_list_detectors_are_not_assessed_for_an_unsupported_language(settings):
    d = Diarization(turns=[], tutor="S0", tutor_confidence=1.0, method="ecapa", n_speakers=2)
    facts = transcript_features.compute(_tr("ml"), d, settings)
    assert all(not det.assessed and not det.hits for det in facts.detections.values())


def test_lexicon_based_stats_are_n_a_for_an_unsupported_language():
    stats = class_stats.compute(*test_class_stats._inputs(), text_supported=False)
    assert stats.open_questions is None and stats.praise_count is None
    assert stats.students_addressed_by_name is None and stats.name_address_count is None
    assert stats.tutor_questions == 2  # a question mark is language independent


def test_the_report_explains_an_unsupported_language(settings):
    media, tr, d, audio, video, tfacts, checklist, score_res = test_report._make_report_inputs(settings)
    score_res.flags.append("language_unsupported")
    tr = tr.model_copy(update={"language": "ml"})
    rep = report.build_report(
        job_id="test_job_12345", media=media, tr=tr, d=d, audio=audio, video=video, tfacts=tfacts,
        checklist=checklist, result=score_res, timings_s={}, versions={}, summary="s",
    )
    page = report.render_html(rep)
    assert "Language notice" in page and "Malayalam" in page


@pytest.mark.parametrize(
    "detected,model_attr,expected_options",
    [
        (("en", 0.99), "whisper", {}),
        (("ml", 0.95), "whisper_multilingual", {"language": "ml"}),
        (("hi", 0.6), "whisper_multilingual", {"multilingual": True}),
    ],
)
def test_speech_model_follows_the_detected_language(monkeypatch, tmp_path, settings, detected, model_attr, expected_options):
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
    seen: dict = {}

    class FakeBatched:
        def transcribe(self, audio, **options):
            seen["options"] = options
            return iter([]), SimpleNamespace(language=detected[0], language_probability=detected[1], duration=1.0)

    def fake_load(_settings, model_name=None):
        seen["model"] = model_name
        return object(), FakeBatched()

    monkeypatch.setattr(asr, "_load_model", fake_load)
    monkeypatch.setattr(asr, "detect_language", lambda wav_path, s: detected)
    tr = asr.transcribe(wav, settings)

    expected_model = getattr(settings.models, model_attr)
    assert seen["model"] == expected_model and tr.asr_model == expected_model
    assert {k: v for k, v in seen["options"].items() if k in ("language", "multilingual")} == expected_options
    assert tr.language == detected[0]
