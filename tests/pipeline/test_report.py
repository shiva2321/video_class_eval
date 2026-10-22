import re

import pytest

from classeval.config import load_settings
from classeval.pipeline import report, scoring
from classeval.pipeline.checklist import ChecklistItem, ChecklistResult
from classeval.schema import (
    AudioFacts,
    Detection,
    Diarization,
    Evidence,
    MediaInfo,
    Observation,
    Report,
    SpeakerTurn,
    Transcript,
    TranscriptFacts,
    VideoFacts,
)


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_report_token")
    return load_settings("config.yaml", env_file=None)


def _make_report_inputs(settings):
    media = MediaInfo(
        duration_s=1800.0,
        has_audio=True,
        has_video=True,
        container="mp4",
        size_bytes=100_000_000,
    )
    tr = Transcript(
        language="en",
        language_confidence=0.95,
        asr_model="large-v3-turbo",
        segments=[],
    )
    d = Diarization(
        turns=[SpeakerTurn(start=0.0, end=1800.0, speaker="TUTOR")],
        tutor="TUTOR",
        tutor_confidence=0.9,
        method="ecapa",
        n_speakers=2,
    )
    audio = AudioFacts(
        duration_s=1800.0,
        speech_ratio=0.75,
        tutor_talk_share=0.6,
        student_words_per_min=25.0,
        dead_air_per_min=0.1,
        median_response_latency_s=2.0,
        tutor_speech_rate_wpm=125.0,
        loudness_lufs=-20.0,
        clipping_ratio=0.001,
        snr_db_est=25.0,
        pitch_std_semitones_tutor=2.5,
        energy_cv_tutor=0.65,
        dropouts=[],
    )
    video = VideoFacts(
        states=[],
        tutor_name="Coach",
        branded_ratio_tutor_speaking=0.8,
        board_share_ratio=0.45,
        lesson_video_ratio=0.05,
        camera_off_total_ratio=0.05,
        camera_off_while_tutor_speaks_s=10.0,
        first5_branded=True,
        last5_branded=True,
        talking_head_luma_mean=110.0,
        n_frames=360,
    )
    tfacts = TranscriptFacts(
        detections={
            "greeting_opening": Detection(assessed=True, score=1.0, hits=[]),
        },
        distinct_responders=3,
        question_reply_rate=0.7,
        question_count=8,
    )
    checklist = ChecklistResult(
        items={
            "R4": ChecklistItem(
                dimension="R4",
                verdict="met",
                evidence=[Evidence(t_start=15.0, kind="quote", text="Welcome everyone to chess!")],
            ),
            "R5": ChecklistItem(
                dimension="R5",
                verdict="met",
                evidence=[Evidence(t_start=45.0, kind="quote", text="Today we study the knight fork.")],
            ),
            "R6": ChecklistItem(
                dimension="R6",
                verdict="met",
                evidence=[Evidence(t_start=120.0, kind="quote", text="Who can find the best move here?")],
            ),
            "R7": ChecklistItem(
                dimension="R7",
                verdict="met",
                evidence=[Evidence(t_start=200.0, kind="quote", text="Brilliant idea, well played!")],
            ),
            "R9": ChecklistItem(
                dimension="R9",
                verdict="met",
                evidence=[Evidence(t_start=300.0, kind="visual", text="Board demonstration of endgame.")],
            ),
            "R10": ChecklistItem(
                dimension="R10",
                verdict="partial",
                evidence=[Evidence(t_start=1700.0, kind="quote", text="Practice your puzzles for next time.")],
            ),
            "R11": ChecklistItem(
                dimension="R11",
                verdict="met",
                evidence=[Evidence(t_start=1780.0, kind="quote", text="Good bye everyone, see you next week!")],
            ),
        },
        strengths=[
            Observation(
                text="Engaging opening with warm personal welcome",
                dimension="R4",
                evidence=[Evidence(t_start=15.0, kind="quote", text="Welcome everyone to chess!")],
            )
        ],
        improvements=[
            Observation(
                text="Assign explicit homework worksheets at closing",
                dimension="R10",
                evidence=[Evidence(t_start=1700.0, kind="quote", text="Practice your puzzles for next time.")],
            )
        ],
        degraded=False,
    )

    res = scoring.score(media, tr, d, audio, video, tfacts, checklist, settings=settings)
    return media, tr, d, audio, video, tfacts, checklist, res


def test_report_validates_and_html_has_grade_and_all_dimension_ids(settings):
    media, tr, d, audio, video, tfacts, checklist, score_res = _make_report_inputs(settings)
    summary_text = report.summarize(score_res, checklist)

    rep = report.build_report(
        job_id="test_job_12345",
        media=media,
        tr=tr,
        d=d,
        audio=audio,
        video=video,
        tfacts=tfacts,
        checklist=checklist,
        result=score_res,
        timings_s={"asr": 5.0, "total": 12.0},
        versions={"whisper": "turbo", "model": "qwen3"},
        summary=summary_text,
    )

    # Validates as schema.Report
    assert isinstance(rep, Report)
    dumped = rep.model_dump_json()
    assert Report.model_validate_json(dumped) == rep

    html = report.render_html(rep)
    assert score_res.grade in html
    # Check all R1 to R12 appear in the HTML
    for i in range(1, 13):
        assert f"R{i}" in html


def test_html_has_no_external_urls(settings):
    media, tr, d, audio, video, tfacts, checklist, score_res = _make_report_inputs(settings)
    summary_text = report.summarize(score_res, checklist)
    rep = report.build_report(
        job_id="test_job_12345",
        media=media,
        tr=tr,
        d=d,
        audio=audio,
        video=video,
        tfacts=tfacts,
        checklist=checklist,
        result=score_res,
        timings_s={},
        versions={},
        summary=summary_text,
    )
    html = report.render_html(rep)

    # Find any http:// or https://
    urls = re.findall(r"https?://", html, flags=re.IGNORECASE)
    assert len(urls) == 0, f"Found external URLs in HTML: {urls}"


def test_summary_names_grade_best_and_weakest_dimension(settings):
    *_, checklist, score_res = _make_report_inputs(settings)
    summary = report.summarize(score_res, checklist)

    assert score_res.grade in summary
    # Best and weakest dimensions should be mentioned
    assessed = [d for d in score_res.dimensions if d.score is not None]
    best_dim = max(assessed, key=lambda x: x.score)
    weakest_dim = min(assessed, key=lambda x: x.score)

    assert best_dim.id in summary or best_dim.name in summary
    assert weakest_dim.id in summary or weakest_dim.name in summary


def test_report_strengths_improvements_each_have_evidence(settings):
    media, tr, d, audio, video, tfacts, checklist, score_res = _make_report_inputs(settings)
    summary_text = report.summarize(score_res, checklist)
    rep = report.build_report(
        job_id="test_job_12345",
        media=media,
        tr=tr,
        d=d,
        audio=audio,
        video=video,
        tfacts=tfacts,
        checklist=checklist,
        result=score_res,
        timings_s={},
        versions={},
        summary=summary_text,
    )

    assert len(rep.strengths) > 0
    for s in rep.strengths:
        assert len(s.evidence) > 0

    assert len(rep.improvements) > 0
    for imp in rep.improvements:
        assert len(imp.evidence) > 0
