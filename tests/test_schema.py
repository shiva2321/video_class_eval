import pytest
from pydantic import ValidationError

from classeval.schema import (
    DimensionScore,
    Evidence,
    MediaInfo,
    Observation,
    Report,
)


def test_quote_over_15_words_rejected():
    words = "word " * 16
    with pytest.raises(ValidationError):
        Evidence(t_start=1.0, kind="quote", text=words.strip())


def test_measure_may_exceed_15_words():
    words = "word " * 20
    ev = Evidence(t_start=1.0, kind="measure", text=words.strip())
    assert ev.kind == "measure"


def test_grade_literals_reject_unknown():
    media = MediaInfo(
        duration_s=100.0,
        has_audio=True,
        has_video=True,
        width=1280,
        height=720,
        fps=25.0,
        video_kbps=1000.0,
        container="mp4",
        size_bytes=1000000,
    )
    with pytest.raises(ValidationError):
        Report(
            job_id="test1",
            created_at="2026-10-22T00:00:00Z",
            media=media,
            language="en",
            language_confidence=0.95,
            grade="Great",  # type: ignore[arg-type]
            overall_score=80.0,
            confidence=0.9,
            coverage=1.0,
            summary="summary",
            dimensions=[],
            strengths=[],
            improvements=[],
            technical_flags=[],
            not_assessed=[],
            downweighted_signals=[],
            degraded=False,
            timings_s={},
            versions={},
        )


def test_dimension_score_none_means_not_assessed():
    ds = DimensionScore(
        id="R1",
        name="Audio clarity",
        group="presentation",
        score=None,
        weight=8.0,
        confidence=0.0,
        evidence=[],
        sources=[],
    )
    assert ds.score is None


def test_report_roundtrip():
    media = MediaInfo(
        duration_s=600.0,
        has_audio=True,
        has_video=True,
        width=1280,
        height=720,
        fps=25.0,
        video_kbps=1500.0,
        container="mp4",
        size_bytes=10485760,
    )
    rep = Report(
        job_id="job_xyz",
        created_at="2026-10-22T10:00:00Z",
        media=media,
        language="en",
        language_confidence=0.98,
        grade="Good",
        overall_score=72.5,
        confidence=0.88,
        coverage=0.95,
        summary="Good session overall.",
        dimensions=[
            DimensionScore(
                id="R1",
                name="Audio clarity",
                group="presentation",
                score=80.0,
                weight=8.0,
                confidence=0.9,
                evidence=[Evidence(t_start=10.0, t_end=15.0, kind="measure", text="snr 24db")],
                sources=["audio"],
                note=None,
            )
        ],
        strengths=[
            Observation(
                text="Clear voice",
                dimension="R1",
                evidence=[Evidence(t_start=10.0, kind="quote", text="Welcome everyone")],
            )
        ],
        improvements=[],
        technical_flags=[],
        not_assessed=[],
        downweighted_signals=[],
        degraded=False,
        timings_s={"asr": 5.2},
        versions={"torch": "2.6.0"},
    )
    dumped = rep.model_dump_json()
    validated = Report.model_validate_json(dumped)
    assert rep == validated
