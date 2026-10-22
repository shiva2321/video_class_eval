import hashlib

import pytest

from classeval.config import load_settings
from classeval.pipeline import checklist
from classeval.pipeline.checklist import (
    ChecklistItem,
    chunk_transcript,
    extract,
    verify_evidence,
)
from classeval.pipeline.llm import OllamaClient
from classeval.schema import (
    Detection,
    Diarization,
    Evidence,
    Segment,
    SpeakerTurn,
    Transcript,
    TranscriptFacts,
)


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_checklist_token")
    return load_settings("config.yaml", env_file=None)


def test_chunking_60min_gives_7_to_8_chunks_each_under_12k_tokens_estimate():
    # 60 min = 3600 s. Segments every 5 seconds.
    segments = []
    for i in range(720):
        t0 = i * 5.0
        t1 = t0 + 4.5
        speaker = "TUTOR" if i % 2 == 0 else "STUDENT_1"
        segments.append(
            Segment(
                start=t0,
                end=t1,
                text=f"Utterance number {i} talking about knight to e4 and pawn moves.",
                speaker=speaker,
            )
        )
    tr = Transcript(language="en", language_confidence=0.98, asr_model="w", segments=segments)
    d = Diarization(
        turns=[SpeakerTurn(start=s.start, end=s.end, speaker=s.speaker or "SPEAKER_00") for s in segments],
        tutor="TUTOR",
        tutor_confidence=0.95,
        method="ecapa",
        n_speakers=2,
    )

    chunks = chunk_transcript(tr, d, minutes=8.0)
    assert 7 <= len(chunks) <= 8

    for t_start, t_end, text in chunks:
        assert t_end > t_start
        # Token estimate: word count * 1.3 must be well under 12k tokens
        word_count = len(text.split())
        token_est = word_count * 1.3
        assert token_est < 12000
        assert "TUTOR:" in text
        assert "STUDENT" in text


def test_prompt_is_deterministic():
    chunk_text = "TUTOR: Hello everyone, welcome to chess class.\nSTUDENT: Hello coach!"
    prompt1 = checklist.build_chunk_prompt(chunk_text, chunk_index=0, total_chunks=8)
    prompt2 = checklist.build_chunk_prompt(chunk_text, chunk_index=0, total_chunks=8)

    hash1 = hashlib.sha256(prompt1.encode("utf-8")).hexdigest()
    hash2 = hashlib.sha256(prompt2.encode("utf-8")).hexdigest()
    assert hash1 == hash2


def test_hallucinated_quote_dropped():
    # Transcript only talks about rook moves
    seg = Segment(start=100.0, end=105.0, text="Move your rook to d1 right now.", speaker="TUTOR")
    tr = Transcript(language="en", language_confidence=0.95, asr_model="w", segments=[seg])

    item = ChecklistItem(
        dimension="R6",
        verdict="met",
        evidence=[
            Evidence(
                t_start=100.0,
                t_end=105.0,
                kind="quote",
                text="Completely fabricated quotation never uttered by tutor",
            )
        ],
    )

    verified = verify_evidence(item, tr)
    # The hallucinated quote must be dropped
    assert len(verified.evidence) == 0
    # If all evidence dropped, verdict falls back to not_observed
    assert verified.verdict == "not_observed"


def test_names_masked_and_quote_truncated_to_15_words():
    # Tutor says a sentence with a child name
    seg = Segment(
        start=30.0,
        end=45.0,
        text="Hello Vihaan let us move the queen to f7 and see if black defends",
        speaker="TUTOR",
    )
    tr = Transcript(language="en", language_confidence=0.95, asr_model="w", segments=[seg])

    # Model returned a quote containing child name "Vihaan"
    item = ChecklistItem(
        dimension="R4",
        verdict="met",
        evidence=[
            Evidence(
                t_start=30.0,
                t_end=45.0,
                kind="quote",
                text="Hello Vihaan let us move the queen to f7 and see if black defends",
            )
        ],
    )

    verified = verify_evidence(item, tr)
    assert len(verified.evidence) == 1
    ev = verified.evidence[0]
    assert "Vihaan" not in ev.text
    assert "[Name]" in ev.text or "[student]" in ev.text or "[tutor]" in ev.text
    words = ev.text.split()
    assert len(words) <= 15

    # Also test capping helper for long raw quote from LLM
    raw_long_quote = "Hello Vihaan let us move the queen to f7 and see if black can defend against the checkmate threat here"
    capped = checklist._mask_and_cap_quote(raw_long_quote, max_words=15)
    assert len(capped.split()) <= 15
    assert "Vihaan" not in capped


def test_reduce_rule_homework_ignores_early_chunks(settings):
    # R10 (homework) reduce rule: only last 2 chunks are considered.
    # If chunk 0 (early) claimed met, but chunks 6 and 7 (last 2) did not observe homework,
    # R10 must NOT be met.
    c_early = {
        "items": [
            {
                "dimension": "R10",
                "verdict": "met",
                "evidence": [{"t": 60.0, "quote": "homework assignment here"}],
            }
        ],
        "strengths": [],
        "improvements": [],
    }
    c_late = {
        "items": [
            {
                "dimension": "R10",
                "verdict": "not_observed",
                "evidence": [],
            }
        ],
        "strengths": [],
        "improvements": [],
    }

    # Simulate 8 chunks where only chunk 0 had R10 met
    chunk_evals = [c_early] + [c_late] * 7
    chunk_times = [(i * 480.0, (i + 1) * 480.0) for i in range(8)]

    # Tutor segment to verify the quote
    seg = Segment(start=60.0, end=65.0, text="homework assignment here", speaker="TUTOR")
    tr = Transcript(language="en", language_confidence=0.95, asr_model="w", segments=[seg])

    res = checklist.reduce_chunks(chunk_evals, chunk_times, tr)
    assert res.items["R10"].verdict != "met"


def test_fallback_to_rules_sets_degraded(settings):
    seg = Segment(start=10.0, end=15.0, text="Good morning everyone, welcome to class.", speaker="TUTOR")
    tr = Transcript(language="en", language_confidence=0.95, asr_model="w", segments=[seg])
    d = Diarization(
        turns=[SpeakerTurn(start=10.0, end=15.0, speaker="TUTOR")],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=1,
    )
    tfacts = TranscriptFacts(
        detections={
            "greeting_opening": Detection(
                assessed=True,
                score=1.0,
                hits=[Evidence(t_start=10.0, t_end=15.0, kind="quote", text="Good morning everyone welcome to class.")],
            ),
            "rapport": Detection(assessed=True, score=0.0, hits=[]),
            "self_intro_new_joiners": Detection(assessed=True, score=0.0, hits=[]),
            "individual_interaction": Detection(assessed=True, score=0.0, hits=[]),
            "homework": Detection(assessed=True, score=0.0, hits=[]),
            "closing": Detection(assessed=True, score=0.0, hits=[]),
            "chess_engagement": Detection(assessed=True, score=0.0, hits=[]),
            "praise_density": Detection(assessed=True, score=0.0, hits=[]),
        },
        distinct_responders=0,
        question_reply_rate=None,
        question_count=0,
    )

    # When llm is None -> fallback to rules
    res = extract(tr, d, tfacts, llm=None, settings=settings)
    assert res.degraded is True
    assert "R4" in res.items
    assert res.items["R4"].verdict == "met"
    assert len(res.items["R4"].evidence) > 0


def test_not_observed_when_transcript_language_uncertain(settings):
    seg = Segment(start=10.0, end=15.0, text="Uncertain language segment", speaker="TUTOR")
    tr = Transcript(language="en", language_confidence=0.3, asr_model="w", segments=[seg])
    d = Diarization(
        turns=[SpeakerTurn(start=10.0, end=15.0, speaker="TUTOR")],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=1,
    )
    tfacts = TranscriptFacts(
        detections={
            k: Detection(assessed=False, score=None, hits=[])
            for k in [
                "greeting_opening",
                "rapport",
                "self_intro_new_joiners",
                "individual_interaction",
                "homework",
                "closing",
                "chess_engagement",
                "praise_density",
            ]
        },
        distinct_responders=0,
        question_reply_rate=None,
        question_count=0,
    )

    res = extract(tr, d, tfacts, llm=None, settings=settings)
    assert res.degraded is True
    for item in res.items.values():
        assert item.verdict == "not_observed"


@pytest.mark.ollama
def test_live_roundtrip_schema_valid(settings):
    seg1 = Segment(start=10.0, end=20.0, text="Hello everyone and welcome to our chess class!", speaker="TUTOR")
    seg2 = Segment(start=25.0, end=35.0, text="Today we will practice pawn endgames and checkmating patterns.", speaker="TUTOR")
    tr = Transcript(language="en", language_confidence=0.95, asr_model="w", segments=[seg1, seg2])
    d = Diarization(
        turns=[
            SpeakerTurn(start=10.0, end=20.0, speaker="TUTOR"),
            SpeakerTurn(start=25.0, end=35.0, speaker="TUTOR"),
        ],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=1,
    )
    tfacts = TranscriptFacts(
        detections={
            "greeting_opening": Detection(assessed=True, score=1.0, hits=[]),
            "rapport": Detection(assessed=True, score=0.0, hits=[]),
            "self_intro_new_joiners": Detection(assessed=True, score=0.0, hits=[]),
            "individual_interaction": Detection(assessed=True, score=0.0, hits=[]),
            "homework": Detection(assessed=True, score=0.0, hits=[]),
            "closing": Detection(assessed=True, score=0.0, hits=[]),
            "chess_engagement": Detection(assessed=True, score=0.0, hits=[]),
            "praise_density": Detection(assessed=True, score=0.0, hits=[]),
        },
        distinct_responders=0,
        question_reply_rate=None,
        question_count=0,
    )

    client = OllamaClient(
        base_url=settings.ollama.base_url,
        model=settings.models.llm,
        timeout_s=180,
    )
    res = extract(tr, d, tfacts, llm=client, settings=settings)
    assert res.degraded is False
    assert len(res.items) > 0


def test_evidence_quote_is_dropped_when_only_a_student_said_it():
    from classeval.pipeline.checklist import ChecklistItem, verify_evidence
    from classeval.schema import Evidence, Segment, Transcript

    tr = Transcript(
        language="en",
        language_confidence=0.9,
        asr_model="t",
        segments=[
            Segment(start=10, end=14, text="I always do my homework after dinner", speaker="S1"),
            Segment(start=20, end=24, text="Please finish the puzzles for next class", speaker="S0"),
        ],
    )
    item = ChecklistItem(
        dimension="R10",
        verdict="met",
        evidence=[Evidence(t_start=12, kind="quote", text="I always do my homework after dinner")],
    )
    assert verify_evidence(item, tr, tutor="S0").verdict == "not_observed"
    assert verify_evidence(item, tr).verdict == "met"  # without the tutor filter the child's words would match


def test_chunk_lines_carry_seconds_and_numbered_students():
    from classeval.pipeline.checklist import chunk_transcript
    from classeval.schema import Diarization, Segment, Transcript

    tr = Transcript(
        language="en",
        language_confidence=0.9,
        asr_model="t",
        segments=[
            Segment(start=125.4, end=128, text="Hello everyone", speaker="S0"),
            Segment(start=130, end=132, text="Hi teacher", speaker="S1"),
            Segment(start=135, end=137, text="Hello", speaker="S2"),
        ],
    )
    d = Diarization(turns=[], tutor="S0", tutor_confidence=1.0, method="ecapa", n_speakers=3)
    lines = chunk_transcript(tr, d)[0][2].splitlines()
    assert lines[0] == "[125s] TUTOR: Hello everyone"
    assert lines[1].startswith("[130s] STUDENT 1:")
    assert lines[2].startswith("[135s] STUDENT 2:")
