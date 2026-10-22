from typing import Any

from pydantic import BaseModel, Field

from classeval.config import Settings
from classeval.pipeline.checklist import ChecklistResult
from classeval.pipeline.language import text_checks_supported
from classeval.schema import (
    AudioFacts,
    Diarization,
    DimensionScore,
    Evidence,
    FinalGrade,
    MediaInfo,
    Transcript,
    TranscriptFacts,
    VideoFacts,
)

_SHARED_STATES = {"BOARD_SHARE", "LESSON_VIDEO_SHARE", "OTHER_SHARE"}

DIMENSION_META: dict[str, dict[str, Any]] = {
    "R1": {"name": "8x8 background", "group": "presentation"},
    "R2": {"name": "Tutor camera and lighting", "group": "presentation"},
    "R3": {"name": "Audio clarity", "group": "presentation"},
    "R4": {"name": "Opening and rapport", "group": "teaching"},
    "R5": {"name": "New-joiner introductions", "group": "teaching"},
    "R6": {"name": "Individual interaction", "group": "teaching"},
    "R7": {"name": "Warmth and energy", "group": "teaching"},
    "R8": {"name": "Student participation", "group": "teaching"},
    "R9": {"name": "Content moves (revision, puzzles, videos)", "group": "teaching"},
    "R10": {"name": "Homework", "group": "teaching"},
    "R11": {"name": "Conclusion", "group": "teaching"},
    "R12": {"name": "Technical stability", "group": "presentation"},
}

VERDICT_MAP: dict[str, float | None] = {
    "met": 100.0,
    "partial": 60.0,
    "not_met": 10.0,
    "not_observed": None,
}


class ScoreResult(BaseModel):
    dimensions: list[DimensionScore]
    overall_score: float | None
    coverage: float
    grade: FinalGrade
    confidence: float
    flags: list[str] = Field(default_factory=list)


# Opening: full marks when the tutor speaks within 30 s of the recording start, none from 3 minutes on. Chosen after
# seeing the ten labelled classes (first-speech time ranked them almost perfectly): treat it as exploratory.
START_FULL_S = 30.0
START_ZERO_S = 180.0
# Camera: the tutor's face counts as shown when clearly visible in at least 60 % of the sampled moments they spoke,
# and not at all below 10 %.
VISIBLE_NONE = 0.1
VISIBLE_FULL = 0.6
PROVISIONAL_COVERAGE = 0.8  # below this share of the rubric assessed, the top grade is not awarded
START_WEIGHT = 0.4  # share of R4 that comes from the start time when the opening was also judged from the words


def _value(item) -> float | None:
    """The 0-100 value of a checklist item: its own mean score when it has one, else the fixed value of its verdict."""
    if item is None:
        return None
    return item.score if item.score is not None else VERDICT_MAP.get(item.verdict)


def _norm(x: float | None, lo: float, hi: float) -> float:
    """Normalize x to [0, 1] clamping to bounds. Inverts scale if lo > hi."""
    if x is None:
        return 0.0
    if lo < hi:
        val = (x - lo) / (hi - lo)
    else:
        val = (lo - x) / (lo - hi)
    return max(0.0, min(1.0, float(val)))


def _measure(text: str) -> list[Evidence]:
    """Evidence for a dimension that is computed from measurements rather than quotes."""
    return [Evidence(t_start=0.0, kind="measure", text=text)]


def score(
    media: MediaInfo,
    tr: Transcript,
    d: Diarization,
    audio: AudioFacts,
    video: VideoFacts,
    tfacts: TranscriptFacts,
    checklist: ChecklistResult,
    settings: Settings,
) -> ScoreResult:
    """Evaluate all R1-R12 dimensions with weights, gates, cutoffs and confidence."""
    weights = settings.weights
    flags: list[str] = []

    # Flag checks
    if tr.language_confidence < 0.5:
        flags.append("language_uncertain")
    elif not text_checks_supported(tr, settings):
        flags.append("language_unsupported")
    if d.tutor_confidence < 0.5:
        flags.append("tutor_unclear")

    lang_uncertain = bool({"language_uncertain", "language_unsupported"} & set(flags))
    # tutor_talk_share is a share of speech time, so tutor seconds = share x speech seconds
    tutor_speech_s = audio.tutor_talk_share * audio.speech_ratio * audio.duration_s
    has_video = media.has_video and video.n_frames > 0

    dim_scores: dict[str, float | None] = {}
    dim_evidence: dict[str, list[Evidence]] = {}
    dim_sources: dict[str, list[str]] = {}
    dim_notes: dict[str, str | None] = {}

    # R1: Branded Backdrop
    dim_sources["R1"] = ["video:branded_ratio_tutor_speaking", "video:thumb_branded_share"]
    if not has_video:
        dim_scores["R1"] = None
        dim_notes["R1"] = "No video in this recording"
    else:
        # Two views of the same thing: the full-frame tutor video, and the small tutor tile shown during screen shares.
        full = None if video.branded_ratio_tutor_speaking is None else _norm(video.branded_ratio_tutor_speaking, 0.0, 0.7)
        tile = None
        if video.thumb_branded_share is not None and video.thumb_crops >= settings.screen_state.thumb_min_crops:
            tile = min(1.0, video.thumb_branded_share / settings.screen_state.thumb_present_share)
        seen = [v for v in (full, tile) if v is not None]
        if not seen:
            dim_scores["R1"] = None
            dim_notes["R1"] = "The tutor's video was not visible often enough to judge the background"
        else:
            dim_scores["R1"] = 100.0 * max(seen)
            parts = []
            if video.branded_ratio_tutor_speaking is not None:
                parts.append(f"{video.branded_ratio_tutor_speaking:.0%} of the on-camera time the tutor spoke")
            if tile is not None and video.thumb_branded_share is not None:
                parts.append(f"{video.thumb_branded_share:.0%} of sampled moments in the tutor's small video tile")
            dim_evidence["R1"] = _measure("8x8 backdrop visible on " + " and ".join(parts))

    # R2: Camera Presence & Lighting
    dim_sources["R2"] = ["video:tutor_visible_ratio", "video:camera_off_while_tutor_speaks_s", "video:talking_head_luma_mean"]
    looked = video.visual_samples >= settings.screen_state.visual_min_samples and video.tutor_visible_ratio is not None
    if not has_video:
        dim_scores["R2"] = None
        dim_notes["R2"] = "No video in this recording"
    elif looked and video.tutor_visible_ratio is not None:
        # A vision model looked at the tutor's video at moments when the tutor was speaking.
        visible = video.tutor_visible_ratio
        r2_val = 100.0 * _norm(visible, VISIBLE_NONE, VISIBLE_FULL)
        light = video.tutor_lighting_good_ratio
        if light is not None and light < 0.5:
            r2_val *= 0.8
        dim_scores["R2"] = r2_val
        detail = f"The tutor's face was clearly visible in {visible:.0%} of {video.visual_samples} sampled moments when they spoke"
        if light is not None:
            detail += f"; the light was good in {light:.0%} of the moments they were on camera"
        dim_evidence["R2"] = _measure(detail)
    else:
        denom = max(1.0, 0.25 * tutor_speech_s)
        cam_ratio = min(1.0, video.camera_off_while_tutor_speaks_s / denom)
        r2_val = 100.0 * (1.0 - cam_ratio)
        if video.talking_head_luma_mean is not None and video.talking_head_luma_mean < 60.0:
            r2_val *= 0.8
        dim_scores["R2"] = max(0.0, min(100.0, r2_val))
        dim_evidence["R2"] = _measure(
            f"Tutor camera was off for {video.camera_off_while_tutor_speaks_s:.0f} s while the tutor was speaking"
        )

    # R3: Audio Quality
    dim_sources["R3"] = ["audio:loudness_lufs", "audio:clipping_ratio", "audio:snr_db_est", "audio:dropouts"]
    r3_val = 100.0
    if audio.loudness_lufs < -30.0 or audio.loudness_lufs > -14.0:
        r3_val -= 20.0
    if audio.clipping_ratio > 0.01:
        r3_val -= 20.0
    if audio.snr_db_est < 15.0:
        r3_val -= 20.0
    # Digital silences ("dropouts") are NOT penalised: real meeting recordings contain many during normal pauses.
    dim_scores["R3"] = max(0.0, min(100.0, r3_val))
    dim_evidence["R3"] = _measure(
        f"Loudness {audio.loudness_lufs:.0f} LUFS, clipping {audio.clipping_ratio:.1%}, "
        f"signal-to-noise about {audio.snr_db_est:.0f} dB"
    )

    # Pedagogical items from checklist
    r4_item = checklist.items.get("R4")
    r5_item = checklist.items.get("R5")
    r6_item = checklist.items.get("R6")
    r7_item = checklist.items.get("R7")
    r9_item = checklist.items.get("R9")
    r10_item = checklist.items.get("R10")
    r11_item = checklist.items.get("R11")

    # R4: Welcoming & Warmup = what the tutor said at the start (words) + how soon the tutor started (language free)
    dim_sources["R4"] = ["checklist:R4", "diarization:first_tutor_speech_s"]
    tutor_starts = [t.start for t in d.turns if d.tutor is not None and t.speaker == d.tutor]
    start_s = min(tutor_starts) if tutor_starts else None
    start_score = None if start_s is None else 100.0 * _norm(start_s, START_ZERO_S, START_FULL_S)
    words_score = None if lang_uncertain else _value(r4_item)
    if words_score is not None and start_score is not None:
        dim_scores["R4"] = (1.0 - START_WEIGHT) * words_score + START_WEIGHT * start_score
    else:
        dim_scores["R4"] = words_score if words_score is not None else start_score
    if lang_uncertain:
        dim_notes["R4"] = "Only the start time could be measured; the words of the opening are not assessed for this language"
    r4_evidence = list(r4_item.evidence) if r4_item and not lang_uncertain else []
    if start_s is not None:
        r4_evidence += _measure(f"The tutor first spoke {start_s:.0f} s into the recording")
    dim_evidence["R4"] = r4_evidence

    # R5: Objective Setting
    dim_sources["R5"] = ["checklist:R5"]
    if lang_uncertain:
        dim_scores["R5"] = None
        dim_notes["R5"] = "Text-derived dimension not assessed due to uncertain language"
    else:
        dim_scores["R5"] = _value(r5_item)
        dim_evidence["R5"] = r5_item.evidence if r5_item else []

    # R6: Interactive Questioning
    dim_sources["R6"] = ["checklist:R6", "tfacts:question_reply_rate"]
    if lang_uncertain:
        dim_scores["R6"] = None
        dim_notes["R6"] = "Text-derived dimension not assessed due to uncertain language"
    else:
        v_r6 = _value(r6_item)
        qrr_norm = _norm(tfacts.question_reply_rate, 0.2, 0.8)
        if v_r6 is not None:
            dim_scores["R6"] = 0.6 * v_r6 + 0.4 * 100.0 * qrr_norm
        else:
            dim_scores["R6"] = 100.0 * qrr_norm if tfacts.question_reply_rate is not None else None
        dim_evidence["R6"] = r6_item.evidence if r6_item else []

    # R7: Voice & Delivery
    dim_sources["R7"] = ["checklist:R7", "audio:pitch_std_semitones_tutor", "audio:energy_cv_tutor"]
    v_r7 = _value(r7_item)
    pitch_norm = _norm(audio.pitch_std_semitones_tutor, 1.5, 4.0)
    energy_norm = _norm(audio.energy_cv_tutor, 0.4, 1.0)
    if v_r7 is not None:
        dim_scores["R7"] = 0.5 * v_r7 + 0.25 * 100.0 * pitch_norm + 0.25 * 100.0 * energy_norm
    else:
        if audio.pitch_std_semitones_tutor is not None and audio.energy_cv_tutor is not None:
            dim_scores["R7"] = 50.0 * pitch_norm + 50.0 * energy_norm
        else:
            dim_scores["R7"] = None
    dim_evidence["R7"] = r7_item.evidence if r7_item else []

    # R8: Student Engagement & Pacing
    dim_sources["R8"] = ["audio:student_words_per_min", "audio:dead_air_per_min", "tfacts:distinct_responders"]
    wpm_norm = _norm(audio.student_words_per_min, 5.0, 45.0)
    air_norm = _norm(audio.dead_air_per_min, 0.8, 0.05)
    resp_norm = _norm(float(tfacts.distinct_responders), 0.0, 5.0)
    dim_scores["R8"] = 100.0 * (0.5 * wpm_norm + 0.25 * air_norm + 0.25 * resp_norm)
    dim_evidence["R8"] = _measure(
        f"Students spoke about {audio.student_words_per_min:.0f} words per minute; "
        f"{audio.dead_air_per_min:.2f} long silences per minute; "
        f"{tfacts.distinct_responders} different students answered the tutor"
    )

    # R9: Screen & Visual Material
    dim_sources["R9"] = ["video:shared_material_ratio", "video:lesson_video_ratio", "checklist:R9"]
    # Any shared lesson material counts (a chess board, slides, a document, a video): the subject can be anything.
    shared_ratio = (
        sum(1 for s in video.states if s.state in _SHARED_STATES) / len(video.states)
        if video.states
        else video.board_share_ratio
    )
    board_norm = _norm(shared_ratio, 0.15, 0.6)
    video_term = 100.0 if video.lesson_video_ratio > 0.02 else 0.0
    v_r9 = _value(r9_item)
    if not has_video:
        dim_scores["R9"] = v_r9  # no screen evidence without video: judge from the transcript alone
    elif v_r9 is not None:
        dim_scores["R9"] = 0.4 * 100.0 * board_norm + 0.2 * video_term + 0.4 * v_r9
    else:
        dim_scores["R9"] = 100.0 * (0.67 * board_norm + 0.33 * (1.0 if video.lesson_video_ratio > 0.02 else 0.0))
    dim_evidence["R9"] = list(r9_item.evidence if r9_item else [])
    if has_video:
        dim_evidence["R9"] += _measure(
            f"Lesson material (board, puzzle, slides or video) on screen in {shared_ratio:.0%} of sampled frames; "
            f"lesson video {'shown' if video.lesson_video_ratio > 0.02 else 'not shown'}"
        )

    # R10: Homework & Practice
    dim_sources["R10"] = ["checklist:R10"]
    if lang_uncertain:
        dim_scores["R10"] = None
        dim_notes["R10"] = "Text-derived dimension not assessed due to uncertain language"
    else:
        dim_scores["R10"] = _value(r10_item)
        dim_evidence["R10"] = r10_item.evidence if r10_item else []

    # R11: Class Wrap-up & Closing
    dim_sources["R11"] = ["checklist:R11", "video:last5_branded"]
    if lang_uncertain:
        dim_scores["R11"] = None
        dim_notes["R11"] = "Text-derived dimension not assessed due to uncertain language"
    else:
        v_r11 = _value(r11_item)
        if v_r11 is not None:
            dim_scores["R11"] = min(100.0, v_r11 + (10.0 if video.last5_branded else 0.0))
        else:
            dim_scores["R11"] = None
        dim_evidence["R11"] = r11_item.evidence if r11_item else []

    # R12: Technical Stability
    dim_sources["R12"] = ["audio:dropouts", "video:camera_off_total_ratio"]
    cam_penalty = 30.0 if has_video and video.camera_off_total_ratio > 0.5 else 0.0
    r12_val = 100.0 - cam_penalty
    dim_scores["R12"] = max(0.0, min(100.0, r12_val))
    dim_evidence["R12"] = _measure(
        f"Camera off in {video.camera_off_total_ratio:.0%} of sampled frames"
        if has_video
        else "No video track to check"
    )

    reasons = {k: v.reason for k, v in checklist.items.items() if v.reason}

    # Build DimensionScore list
    dimensions: list[DimensionScore] = []
    for d_id, meta in DIMENSION_META.items():
        score_val = dim_scores.get(d_id)
        if score_val is not None:
            score_val = round(score_val, 1)
        dimensions.append(
            DimensionScore(
                id=d_id,
                name=meta["name"],
                group=meta["group"],
                score=score_val,
                weight=weights.get(d_id, 0.0),
                confidence=1.0 if score_val is not None else 0.0,
                evidence=dim_evidence.get(d_id, []),
                sources=dim_sources.get(d_id, []),
                note=dim_notes.get(d_id) or reasons.get(d_id),
            )
        )

    # Calculate coverage and overall score
    assessed_dims = [d for d in dimensions if d.score is not None]
    assessed_weight = sum(weights.get(d.id, 0.0) for d in assessed_dims)
    coverage = round(assessed_weight / 100.0, 4)

    if assessed_weight > 0:
        overall_score = sum(weights.get(d.id, 0.0) * d.score for d in assessed_dims) / assessed_weight
        overall_score = round(overall_score, 1)
    else:
        overall_score = None

    # Grade determination
    cutoffs = settings.cutoffs
    grade: FinalGrade
    if overall_score is None:
        grade = "Insufficient evidence"
    elif overall_score < cutoffs.poor_lt:
        grade = "Poor"
    elif overall_score < cutoffs.average_lt:
        grade = "Average"
    elif overall_score < cutoffs.good_lt:
        grade = "Good"
    else:
        grade = "Excellent"

    # Gates
    # Gate 1: Insufficient evidence conditions
    if (
        media.duration_s < settings.limits.min_duration_s
        or audio.speech_ratio < 0.05
        or coverage < 0.5
    ):
        grade = "Insufficient evidence"

    # Gate 3: the language model analysed none of the class, so the word lists alone must not give a top grade
    if checklist.degraded and "language_unsupported" not in flags and "language_uncertain" not in flags:
        flags.append("llm_degraded")
        if grade == "Excellent":
            grade = "Good"
    if checklist.chunks_failed:
        flags.append("partial_analysis")
    if coverage < PROVISIONAL_COVERAGE and grade == "Excellent":
        flags.append("grade_provisional")
        grade = "Good"

    # Gate 2: Camera-off while tutor speaks >= 50%
    if tutor_speech_s > 0 and (video.camera_off_while_tutor_speaks_s / tutor_speech_s) >= 0.5:
        flags.append("camera_off_high")
        if grade in ("Good", "Excellent"):
            grade = "Average"

    # Confidence calculation:
    # 0.4*coverage + 0.2*asr_conf + 0.2*tutor_conf + 0.2*(1.0 if not degraded else 0.4)
    failed_share = checklist.chunks_failed / checklist.chunks_total if checklist.chunks_total else 0.0
    deg_factor = 0.4 if checklist.degraded else 1.0 - 0.6 * failed_share
    conf = (
        0.4 * coverage
        + 0.2 * tr.language_confidence
        + 0.2 * d.tutor_confidence
        + 0.2 * deg_factor
    )
    confidence = round(max(0.0, min(1.0, conf)), 4)

    return ScoreResult(
        dimensions=dimensions,
        overall_score=overall_score,
        coverage=coverage,
        grade=grade,
        confidence=confidence,
        flags=flags,
    )
