import pytest

from classeval.config import load_settings
from classeval.pipeline import transcript_features
from classeval.schema import Diarization, Segment, SpeakerTurn, Transcript


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_tfacts_token")
    return load_settings("config.yaml", env_file=None)


def test_hi_does_not_match_inside_other_words(settings):
    # Words like "this", "which", "think" contain "hi", but must not match "hi"
    seg = Segment(start=10.0, end=15.0, text="This which think is good.", speaker="TUTOR")
    tr = Transcript(language="en", language_confidence=0.99, asr_model="w", segments=[seg])
    d = Diarization(
        turns=[SpeakerTurn(start=10.0, end=15.0, speaker="TUTOR")],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=1,
    )

    facts = transcript_features.compute(tr, d, settings)
    assert facts.detections["greeting_opening"].assessed is True
    # Should not have matched greeting
    assert len(facts.detections["greeting_opening"].hits) == 0


def test_phrases_match_case_insensitively(settings):
    seg = Segment(start=10.0, end=15.0, text="GOOD MORNING EVERYONE", speaker="TUTOR")
    tr = Transcript(language="en", language_confidence=0.99, asr_model="w", segments=[seg])
    d = Diarization(
        turns=[SpeakerTurn(start=10.0, end=15.0, speaker="TUTOR")],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=1,
    )

    facts = transcript_features.compute(tr, d, settings)
    assert len(facts.detections["greeting_opening"].hits) > 0


def test_puzzle_alone_is_not_homework(settings):
    # Homework lexicon was narrowed: "puzzle" or "practice" alone is not homework
    seg = Segment(start=100.0, end=110.0, text="Let's solve this puzzle and practice.", speaker="TUTOR")
    tr = Transcript(language="en", language_confidence=0.99, asr_model="w", segments=[seg])
    d = Diarization(
        turns=[SpeakerTurn(start=100.0, end=110.0, speaker="TUTOR")],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=1,
    )

    facts = transcript_features.compute(tr, d, settings)
    assert len(facts.detections["homework"].hits) == 0


def test_not_assessed_when_language_uncertain(settings):
    seg = Segment(start=10.0, end=15.0, text="Hello class, how are you?", speaker="TUTOR")
    tr = Transcript(language="en", language_confidence=0.3, asr_model="w", segments=[seg])
    d = Diarization(
        turns=[SpeakerTurn(start=10.0, end=15.0, speaker="TUTOR")],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=1,
    )

    facts = transcript_features.compute(tr, d, settings)
    for det_name, det in facts.detections.items():
        assert det.assessed is False, f"Detector {det_name} must not be assessed on uncertain language"
        assert det.score is None


def test_homework_in_last_10_min_counts_double(settings):
    # Class duration 30 min (1800 s). Last 10 min starts at t=1200 s.
    # Early homework at t=300 (score 1), late homework at t=1500 (score 2)
    s1 = Segment(start=300.0, end=305.0, text="Please do the worksheet at home.", speaker="TUTOR")
    s2 = Segment(start=1500.0, end=1505.0, text="Remember your homework assignment.", speaker="TUTOR")
    tr = Transcript(language="en", language_confidence=0.99, asr_model="w", segments=[s1, s2])
    d = Diarization(
        turns=[SpeakerTurn(start=0.0, end=1800.0, speaker="TUTOR")],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=1,
    )

    facts = transcript_features.compute(tr, d, settings)
    hw_det = facts.detections["homework"]
    assert hw_det.assessed is True
    # 1 + 2 = 3 points
    assert hw_det.score == pytest.approx(3.0)


def test_distinct_responders_counts_speakers_not_utterances(settings):
    # Tutor asks questions, Student1 answers twice, Student2 answers once -> distinct_responders = 2
    segments = [
        Segment(start=10.0, end=12.0, text="What is this move?", speaker="TUTOR"),
        Segment(start=14.0, end=15.0, text="Knight to f3", speaker="STUDENT_1"),
        Segment(start=20.0, end=22.0, text="Can you capture?", speaker="TUTOR"),
        Segment(start=23.0, end=24.0, text="Yes", speaker="STUDENT_1"),
        Segment(start=30.0, end=32.0, text="Who can defend?", speaker="TUTOR"),
        Segment(start=33.0, end=34.0, text="The bishop", speaker="STUDENT_2"),
    ]
    tr = Transcript(language="en", language_confidence=0.99, asr_model="w", segments=segments)
    d = Diarization(turns=[], tutor="TUTOR", tutor_confidence=0.9, method="ecapa", n_speakers=3)

    facts = transcript_features.compute(tr, d, settings)
    assert facts.distinct_responders == 2
    assert facts.question_count == 3
    assert facts.question_reply_rate == pytest.approx(1.0)


def test_quotes_capped_15_words_and_names_masked(settings):
    # Long quote > 15 words with a name
    long_text = "Hello Aarav welcome to our class today we will play chess and analyze every opening move very carefully"
    seg = Segment(start=10.0, end=20.0, text=long_text, speaker="TUTOR")
    tr = Transcript(language="en", language_confidence=0.99, asr_model="w", segments=[seg])
    d = Diarization(
        turns=[SpeakerTurn(start=10.0, end=20.0, speaker="TUTOR")],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=1,
    )

    facts = transcript_features.compute(tr, d, settings)
    hits = facts.detections["greeting_opening"].hits
    assert len(hits) > 0
    quote = hits[0].text
    words = quote.split()
    assert len(words) <= 15
    assert "Aarav" not in quote


def test_praise_density_present_but_weight_zero_flag(settings):
    seg = Segment(start=10.0, end=15.0, text="Great job! Very good move! Excellent!", speaker="TUTOR")
    tr = Transcript(language="en", language_confidence=0.99, asr_model="w", segments=[seg])
    d = Diarization(
        turns=[SpeakerTurn(start=10.0, end=15.0, speaker="TUTOR")],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=1,
    )

    facts = transcript_features.compute(tr, d, settings)
    assert "praise_density" in facts.detections
    praise_det = facts.detections["praise_density"]
    assert praise_det.assessed is True
    assert praise_det.score is not None
    assert len(praise_det.hits) > 0


def test_empty_transcript_no_crash(settings):
    tr = Transcript(language="en", language_confidence=0.99, asr_model="w", segments=[])
    d = Diarization(turns=[], tutor="S0", tutor_confidence=0.5, method="ecapa", n_speakers=1)

    facts = transcript_features.compute(tr, d, settings)
    assert facts.distinct_responders == 0
    assert facts.question_count == 0
    assert facts.question_reply_rate is None
    for det in facts.detections.values():
        assert det.assessed is True
        assert det.score in (0.0, None)


def test_no_identified_tutor_means_nothing_is_assessed_or_quoted():
    import os

    from classeval.config import load_settings
    from classeval.pipeline.transcript_features import compute
    from classeval.schema import Diarization, Segment, Transcript

    os.environ.setdefault("ACCESS_CODE", "test_features_token")
    settings = load_settings("config.yaml", env_file=None)
    tr = Transcript(
        language="en",
        language_confidence=0.95,
        asr_model="t",
        segments=[
            Segment(start=5, end=8, text="Hello everyone welcome to class", speaker="S1"),
            Segment(start=40, end=44, text="I did my homework yesterday", speaker="S2"),
        ],
    )
    d = Diarization(turns=[], tutor=None, tutor_confidence=0.0, method="ecapa", n_speakers=2)
    facts = compute(tr, d, settings)
    assert all(not det.assessed and not det.hits for det in facts.detections.values())
