import pytest

from classeval.config import load_settings
from classeval.pipeline import scoring
from classeval.pipeline.checklist import ChecklistItem, ChecklistResult
from classeval.schema import (
    AudioFacts,
    Detection,
    Diarization,
    Evidence,
    MediaInfo,
    SpeakerTurn,
    Transcript,
    TranscriptFacts,
    VideoFacts,
)


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_scoring_token")
    return load_settings("config.yaml", env_file=None)


def _make_dummy_inputs(
    duration_s=1800.0,
    speech_ratio=0.75,
    language_conf=0.95,
    tutor_conf=0.95,
    branded_ratio=0.8,
    cam_off_tutor_s=0.0,
    talking_luma=120.0,
    lufs=-20.0,
    clipping=0.001,
    snr=25.0,
    dropouts=None,
    qrr=0.75,
    distinct_resp=4,
    pitch_std=2.8,
    energy_cv=0.7,
    board_share=0.5,
    lesson_video=0.05,
    cam_off_total=0.05,
    wpm=30.0,
    dead_air=0.1,
    last5_branded=True,
    all_verdicts="met",
    degraded=False,
):
    if dropouts is None:
        dropouts = []

    media = MediaInfo(
        duration_s=duration_s,
        has_audio=True,
        has_video=True,
        container="mp4",
        size_bytes=100_000_000,
    )
    tr = Transcript(
        language="en",
        language_confidence=language_conf,
        asr_model="large-v3-turbo",
        segments=[],
    )
    d = Diarization(
        turns=[SpeakerTurn(start=0.0, end=duration_s, speaker="TUTOR")],
        tutor="TUTOR",
        tutor_confidence=tutor_conf,
        method="ecapa",
        n_speakers=3,
    )
    audio = AudioFacts(
        duration_s=duration_s,
        speech_ratio=speech_ratio,
        tutor_talk_share=0.6,
        student_words_per_min=wpm,
        dead_air_per_min=dead_air,
        median_response_latency_s=2.5,
        tutor_speech_rate_wpm=130.0,
        loudness_lufs=lufs,
        clipping_ratio=clipping,
        snr_db_est=snr,
        pitch_std_semitones_tutor=pitch_std,
        energy_cv_tutor=energy_cv,
        dropouts=dropouts,
    )
    video = VideoFacts(
        states=[],
        tutor_name="Coach",
        branded_ratio_tutor_speaking=branded_ratio,
        board_share_ratio=board_share,
        lesson_video_ratio=lesson_video,
        camera_off_total_ratio=cam_off_total,
        camera_off_while_tutor_speaks_s=cam_off_tutor_s,
        first5_branded=True,
        last5_branded=last5_branded,
        talking_head_luma_mean=talking_luma,
        n_frames=360,
    )
    tfacts = TranscriptFacts(
        detections={
            "greeting_opening": Detection(assessed=True, score=1.0, hits=[]),
            "praise_density": Detection(assessed=True, score=0.8, hits=[]),
        },
        distinct_responders=distinct_resp,
        question_reply_rate=qrr,
        question_count=10,
    )
    checklist = ChecklistResult(
        items={
            dim: ChecklistItem(
                dimension=dim,
                verdict=all_verdicts,
                evidence=[Evidence(t_start=10.0, kind="quote", text="Sample quote")] if all_verdicts != "not_observed" else [],
            )
            for dim in ["R4", "R5", "R6", "R7", "R9", "R10", "R11"]
        },
        strengths=[],
        improvements=[],
        degraded=degraded,
    )
    return media, tr, d, audio, video, tfacts, checklist


def test_all_good_facts_grade_excellent_over_85(settings):
    args = _make_dummy_inputs(all_verdicts="met")
    res = scoring.score(*args, settings=settings)
    assert res.overall_score is not None
    assert res.overall_score > 85.0
    assert res.grade == "Excellent"


def test_all_bad_facts_grade_poor(settings):
    args = _make_dummy_inputs(
        branded_ratio=0.0,
        cam_off_tutor_s=400.0,
        talking_luma=30.0,
        lufs=-45.0,
        clipping=0.05,
        snr=8.0,
        dropouts=[(10.0, 15.0), (40.0, 45.0), (100.0, 105.0), (200.0, 205.0), (300.0, 305.0)],
        qrr=0.1,
        distinct_resp=0,
        pitch_std=1.0,
        energy_cv=0.2,
        board_share=0.05,
        lesson_video=0.0,
        cam_off_total=0.7,
        wpm=2.0,
        dead_air=1.2,
        last5_branded=False,
        all_verdicts="not_met",
    )
    res = scoring.score(*args, settings=settings)
    assert res.overall_score is not None
    assert res.overall_score < 45.0
    assert res.grade == "Poor"


def test_camera_off_60pct_caps_at_average(settings):
    # tutor speech = 0.6 * 1800 = 1080s. 60% camera off = 648s
    args = _make_dummy_inputs(
        cam_off_tutor_s=648.0,
        all_verdicts="met",
    )
    res = scoring.score(*args, settings=settings)
    assert "camera_off_high" in res.flags
    assert res.grade == "Average"


def test_not_assessed_excluded_and_weights_renormalised(settings):
    # If video.branded_ratio_tutor_speaking is None -> R1 not assessed (weight 8).
    # Coverage should be exactly 92 / 100 = 0.92
    args = list(_make_dummy_inputs(branded_ratio=None, all_verdicts="met"))
    res = scoring.score(*args, settings=settings)
    assert res.coverage == pytest.approx(0.92)
    # R1 score must be None
    r1_dim = next(d for d in res.dimensions if d.id == "R1")
    assert r1_dim.score is None


def test_silent_recording_insufficient_evidence(settings):
    args = _make_dummy_inputs(speech_ratio=0.01)
    res = scoring.score(*args, settings=settings)
    assert res.grade == "Insufficient evidence"


def test_short_recording_insufficient_evidence(settings):
    args = _make_dummy_inputs(duration_s=300.0)
    res = scoring.score(*args, settings=settings)
    assert res.grade == "Insufficient evidence"


def test_non_english_marks_text_dimensions_not_assessed_not_zero(settings):
    args = _make_dummy_inputs(language_conf=0.3)
    res = scoring.score(*args, settings=settings)
    assert "language_uncertain" in res.flags
    for dim_id in ["R4", "R5", "R6", "R10", "R11"]:
        d_score = next(d for d in res.dimensions if d.id == dim_id)
        assert d_score.score is None


def test_praise_density_has_zero_weight(settings):
    # Praise density is reported in transcript facts, but has no weight in scoring
    args = _make_dummy_inputs(all_verdicts="met")
    res = scoring.score(*args, settings=settings)
    dim_ids = [d.id for d in res.dimensions]
    assert "praise_density" not in dim_ids


def test_confidence_penalised_when_degraded(settings):
    args_normal = _make_dummy_inputs(degraded=False)
    res_normal = scoring.score(*args_normal, settings=settings)

    args_degraded = _make_dummy_inputs(degraded=True)
    res_degraded = scoring.score(*args_degraded, settings=settings)

    # 0.2 * (1.0 - 0.4) = 0.12 difference
    assert res_normal.confidence - res_degraded.confidence == pytest.approx(0.12)


def test_low_tutor_confidence_adds_flag_and_lowers_confidence(settings):
    args_high = _make_dummy_inputs(tutor_conf=1.0)
    res_high = scoring.score(*args_high, settings=settings)

    args_low = _make_dummy_inputs(tutor_conf=0.2)
    res_low = scoring.score(*args_low, settings=settings)

    assert "tutor_unclear" in res_low.flags
    # Difference should be exactly 0.2 * (1.0 - 0.2) = 0.16
    assert res_high.confidence - res_low.confidence == pytest.approx(0.16)


def test_audio_only_recording_does_not_score_video_dimensions(settings):
    media, tr, d, audio, video, tfacts, checklist = _make_dummy_inputs()
    media = media.model_copy(update={"has_video": False})
    video = video.model_copy(update={"n_frames": 0})
    res = scoring.score(media, tr, d, audio, video, tfacts, checklist, settings)
    by_id = {x.id: x for x in res.dimensions}
    assert by_id["R1"].score is None and by_id["R2"].score is None
    assert res.coverage < 1.0


def test_camera_off_gate_uses_tutor_speech_seconds_not_whole_duration(settings):
    # 1800 s x 0.75 speech x 0.6 tutor share = 810 s of tutor speech; 405 s camera-off is exactly half of it.
    res = scoring.score(*_make_dummy_inputs(cam_off_tutor_s=405.0), settings)
    assert "camera_off_high" in res.flags
    assert res.grade in ("Average", "Poor")


def test_measured_dimensions_carry_measure_evidence_and_rubric_names(settings):
    res = scoring.score(*_make_dummy_inputs(), settings)
    by_id = {x.id: x for x in res.dimensions}
    for dim in ("R1", "R2", "R3", "R8", "R12"):
        assert by_id[dim].evidence and by_id[dim].evidence[0].kind == "measure"
    assert by_id["R5"].name == "New-joiner introductions"
    assert by_id["R6"].name == "Individual interaction"


def test_checklist_reason_becomes_the_dimension_note(settings):
    media, tr, d, audio, video, tfacts, checklist = _make_dummy_inputs()
    checklist.items["R4"] = ChecklistItem(
        dimension="R4",
        verdict="met",
        evidence=[Evidence(t_start=10.0, kind="quote", text="Hello everyone welcome to class")],
        reason="Greeted the children warmly before starting.",
    )
    res = scoring.score(media, tr, d, audio, video, tfacts, checklist, settings)
    by_id = {x.id: x for x in res.dimensions}
    assert by_id["R4"].note == "Greeted the children warmly before starting."
