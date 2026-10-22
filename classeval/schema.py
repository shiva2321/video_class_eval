from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Grade = Literal["Poor", "Average", "Good", "Excellent"]
FinalGrade = Literal["Poor", "Average", "Good", "Excellent", "Insufficient evidence"]
DimId = Literal["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12"]
ScreenState = Literal[
    "BRANDED_TALKING_HEAD",
    "OTHER_TALKING_HEAD",
    "CAMERA_OFF_CARD",
    "BOARD_SHARE",
    "LESSON_VIDEO_SHARE",
    "OTHER_SHARE",
    "UNKNOWN",
]
STAGES: tuple[str, ...] = (
    "queued",
    "ingest",
    "asr",
    "diarize",
    "audio",
    "video",
    "understand",
    "score",
    "report",
    "done",
)


class MediaInfo(BaseModel):
    duration_s: float
    has_audio: bool
    has_video: bool
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_kbps: float | None = None
    container: str
    size_bytes: int


class FrameRef(BaseModel):
    t: float
    path: Path


class Word(BaseModel):
    start: float
    end: float
    text: str
    prob: float


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None
    avg_logprob: float | None = None
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    language: str
    language_confidence: float
    asr_model: str
    segments: list[Segment]


class SpeakerTurn(BaseModel):
    start: float
    end: float
    speaker: str


class Diarization(BaseModel):
    turns: list[SpeakerTurn]
    tutor: str | None
    tutor_confidence: float
    method: Literal["pyannote", "ecapa", "none"]
    n_speakers: int


class FrameState(BaseModel):
    t: float
    state: ScreenState
    confidence: float


class Evidence(BaseModel):
    t_start: float
    t_end: float | None = None
    kind: Literal["quote", "measure", "visual"]
    text: str

    @model_validator(mode="after")
    def validate_quote_length(self) -> "Evidence":
        if self.kind == "quote":
            words = self.text.strip().split()
            if len(words) > 15:
                raise ValueError(f"Evidence quote must be at most 15 words, got {len(words)}")
        return self


class Observation(BaseModel):
    text: str
    dimension: DimId
    evidence: list[Evidence] = Field(default_factory=list)


class Detection(BaseModel):
    assessed: bool
    score: float | None = None
    hits: list[Evidence] = Field(default_factory=list)


class AudioFacts(BaseModel):
    duration_s: float
    speech_ratio: float
    tutor_talk_share: float
    student_words_per_min: float
    dead_air_per_min: float
    median_response_latency_s: float | None
    tutor_speech_rate_wpm: float
    loudness_lufs: float
    clipping_ratio: float
    snr_db_est: float
    pitch_std_semitones_tutor: float | None
    energy_cv_tutor: float | None
    dropouts: list[tuple[float, float]] = Field(default_factory=list)


class VideoFacts(BaseModel):
    states: list[FrameState]
    tutor_name: str | None
    branded_ratio_tutor_speaking: float | None
    board_share_ratio: float
    lesson_video_ratio: float
    camera_off_total_ratio: float
    camera_off_while_tutor_speaks_s: float
    first5_branded: bool | None
    last5_branded: bool | None
    talking_head_luma_mean: float | None
    n_frames: int
    thumb_branded_share: float | None = None  # share of sampled moments with the backdrop in the tutor's top-right tile
    thumb_crops: int = 0
    tutor_visible_ratio: float | None = None  # share of sampled tutor-speaking moments with the face clearly visible
    tutor_lighting_good_ratio: float | None = None  # of the moments the tutor was on camera, share with good light
    visual_samples: int = 0


class TranscriptFacts(BaseModel):
    detections: dict[str, Detection]
    distinct_responders: int
    question_reply_rate: float | None
    question_count: int


class DimensionScore(BaseModel):
    id: DimId
    name: str
    group: Literal["teaching", "presentation"]
    score: float | None = None
    weight: float
    confidence: float
    evidence: list[Evidence] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    note: str | None = None


class ExchangeExample(BaseModel):
    """One tutor question with how it went. `question` is the tutor's own words (masked, short); the child's
    answer is never quoted, only measured."""

    t_s: float
    question: str
    kind: str  # open | closed
    answered: bool
    wait_s: float | None = None
    answer_words: int | None = None
    feedback: str = "none"  # praise | correction | explanation | follow_up_question | acknowledgement | moved_on | none


class InteractionStats(BaseModel):
    """Question - answer - feedback facts. Type counts are None when word lists do not apply (other languages)."""

    questions_total: int
    content_questions: int  # open + closed (attention and checking questions excluded)
    open_questions: int | None = None
    closed_questions: int | None = None
    checking_questions: int | None = None  # "do you understand?"
    attention_questions: int | None = None  # "can you hear me?"
    content_answered_pct: float | None = None
    median_wait_s: float | None = None
    median_answer_words: float | None = None
    feedback_pct: float | None = None  # of answered content questions, share where the tutor reacted explicitly
    feedback_mix: dict[str, int] = Field(default_factory=dict)
    student_initiated: int = 0
    longest_exchange_chain: int = 0
    examples: list[ExchangeExample] = Field(default_factory=list)


class TimelineBin(BaseModel):
    """One minute of the class: seconds the tutor and the students spoke, and the dominant screen state."""

    tutor_s: float
    student_s: float
    screen: str  # dominant ScreenState in that minute, or "NONE" when no frame was sampled


class ClassStats(BaseModel):
    """Objective, countable facts about what happened in the class (the report's "at a glance" section)."""

    duration_min: float
    tutor_talk_pct: float
    student_talk_pct: float
    silence_pct: float
    tutor_words: int
    student_words: int
    tutor_wpm: float
    student_words_per_min: float
    student_voices: int
    tutor_questions: int
    open_questions: int | None = None  # None = not assessed (word lists need a supported language)
    question_reply_pct: float | None = None
    median_wait_s: float | None = None
    student_questions: int
    student_utterances: int
    students_addressed_by_name: int | None = None
    name_address_count: int | None = None
    praise_count: int | None = None
    longest_tutor_monologue_s: float
    first_tutor_speech_s: float | None = None
    last_tutor_speech_s: float | None = None
    screen_pct: dict[str, float] = Field(default_factory=dict)
    timeline: list[TimelineBin] = Field(default_factory=list)
    interaction: InteractionStats | None = None


class Narrative(BaseModel):
    """Short written explanation of the grade, produced by the local model from the report's own facts."""

    headline: str
    why: str
    improve: str


class Report(BaseModel):
    job_id: str
    created_at: str
    media: MediaInfo
    language: str
    language_confidence: float
    grade: FinalGrade
    overall_score: float | None = None
    confidence: float
    coverage: float
    summary: str
    dimensions: list[DimensionScore] = Field(default_factory=list)
    strengths: list[Observation] = Field(default_factory=list)
    improvements: list[Observation] = Field(default_factory=list)
    technical_flags: list[str] = Field(default_factory=list)
    not_assessed: list[str] = Field(default_factory=list)
    downweighted_signals: list[str] = Field(default_factory=list)
    degraded: bool
    timings_s: dict[str, float] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)
    stats: ClassStats | None = None
    topics: list[str] = Field(default_factory=list)
    narrative: Narrative | None = None
