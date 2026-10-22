"""Objective, countable facts about what happened in the class (the report's "at a glance" section).

Everything here is a count, a duration or a share computed from the transcript, the speaker turns and the
sampled frames. No text is stored: names are only counted, never kept.
"""

import math
import re
from collections import Counter

from classeval.pipeline.checklist import find_names
from classeval.schema import (
    AudioFacts,
    ClassStats,
    Diarization,
    MediaInfo,
    TimelineBin,
    Transcript,
    TranscriptFacts,
    VideoFacts,
)

MONOLOGUE_GAP_S = 3.0  # tutor utterances closer than this belong to one uninterrupted stretch
MIN_BREAKER_WORDS = 2  # a student utterance of at least this many words interrupts a tutor stretch
MIN_STUDENT_VOICE_WORDS = 25  # a voice counts as a distinct student once it has said this many words
MAX_TIMELINE_MINUTES = 360

_PRAISE = re.compile(
    r"(?<!\w)(great|excellent|well done|very good|very nice|good job|nice job|perfect|correct|wonderful|"
    r"awesome|brilliant|super|exactly|that's right|that is right)(?!\w)",
    re.IGNORECASE,
)
_OPEN_QUESTION = re.compile(
    r"^\W*(?:(?:okay|ok|so|now|and|well|alright)\b[\s,]*)*"
    r"(why|how|what do you think|what would|what if|can you explain|explain|tell me why|what makes)",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_SCREEN_GROUPS = {
    "BOARD_SHARE": "board",
    "LESSON_VIDEO_SHARE": "lesson_video",
    "BRANDED_TALKING_HEAD": "face_on_screen",
    "OTHER_TALKING_HEAD": "face_on_screen",
    "CAMERA_OFF_CARD": "camera_off",
    "OTHER_SHARE": "other",
    "UNKNOWN": "other",
}


def screen_group(state: str) -> str:
    """Reader-friendly group of a screen state ("NONE" when no frame was sampled)."""
    return _SCREEN_GROUPS.get(state, "NONE")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def _count_questions(texts: list[str]) -> tuple[int, int]:
    """(questions, open questions) among the sentences of the given utterances."""
    questions = open_questions = 0
    for text in texts:
        for sentence in _sentences(text):
            if sentence.endswith("?"):
                questions += 1
                if _OPEN_QUESTION.match(sentence):
                    open_questions += 1
    return questions, open_questions


def _longest_tutor_stretch(segments: list, tutor: str | None) -> float:
    """Longest run of tutor speech that no student interrupts (gaps up to MONOLOGUE_GAP_S are bridged)."""
    best = 0.0
    run_start: float | None = None
    run_end = 0.0
    for seg in segments:
        if tutor is not None and seg.speaker == tutor:
            if run_start is not None and seg.start - run_end <= MONOLOGUE_GAP_S:
                run_end = max(run_end, seg.end)
            else:
                run_start, run_end = seg.start, seg.end
            best = max(best, run_end - run_start)
        elif len(seg.text.split()) >= MIN_BREAKER_WORDS:
            run_start = None
    return best


def _timeline(
    d: Diarization, video: VideoFacts, duration_s: float
) -> list[TimelineBin]:
    n = min(MAX_TIMELINE_MINUTES, max(1, math.ceil(duration_s / 60.0)))
    tutor_s = [0.0] * n
    student_s = [0.0] * n
    for turn in d.turns:
        target = tutor_s if turn.speaker == d.tutor else student_s
        a, b = turn.start, min(turn.end, n * 60.0)
        while a < b:
            k = int(a // 60)
            edge = (k + 1) * 60.0
            target[k] += min(b, edge) - a
            a = min(b, edge)
    screens: list[Counter[str]] = [Counter() for _ in range(n)]
    for fs in video.states:
        k = int(fs.t // 60)
        if 0 <= k < n:
            screens[k][screen_group(fs.state)] += 1
    return [
        TimelineBin(
            tutor_s=round(tutor_s[k], 1),
            student_s=round(student_s[k], 1),
            screen=screens[k].most_common(1)[0][0] if screens[k] else "NONE",
        )
        for k in range(n)
    ]


def compute(
    tr: Transcript,
    d: Diarization,
    audio: AudioFacts,
    video: VideoFacts,
    tfacts: TranscriptFacts,
    media: MediaInfo,
    text_supported: bool = True,
) -> ClassStats:
    duration_s = max(media.duration_s, 1e-6)
    tutor = d.tutor
    tutor_turn_s = sum(t.end - t.start for t in d.turns if t.speaker == tutor) if tutor else 0.0
    student_turn_s = sum(t.end - t.start for t in d.turns if t.speaker != tutor)
    tutor_pct = round(100.0 * tutor_turn_s / duration_s, 1)
    student_pct = round(100.0 * student_turn_s / duration_s, 1)

    segments = sorted(tr.segments, key=lambda s: s.start)
    tutor_segs = [s for s in segments if tutor is not None and s.speaker == tutor]
    student_segs = [s for s in segments if s.speaker is not None and s.speaker != tutor]

    tutor_questions, open_questions = _count_questions([s.text for s in tutor_segs])
    student_questions, _ = _count_questions([s.text for s in student_segs])

    words_by_voice: Counter[str] = Counter()
    for s in student_segs:
        words_by_voice[s.speaker or ""] += len(s.text.split())

    address_utterances = 0
    names: set[str] = set()
    for s in tutor_segs:
        found = find_names(s.text)
        if found:
            address_utterances += 1
            names.update(n.lower() for n in found)

    frames = Counter(screen_group(fs.state) for fs in video.states)
    total_frames = sum(frames.values())
    screen_pct = {k: round(100.0 * v / total_frames, 1) for k, v in frames.items()} if total_frames else {}

    tutor_turns = [t for t in d.turns if t.speaker == tutor]
    return ClassStats(
        duration_min=round(duration_s / 60.0, 1),
        tutor_talk_pct=tutor_pct,
        student_talk_pct=student_pct,
        silence_pct=round(max(0.0, 100.0 * (1.0 - (tutor_turn_s + student_turn_s) / duration_s)), 1),
        tutor_words=sum(len(s.text.split()) for s in tutor_segs),
        student_words=sum(len(s.text.split()) for s in student_segs),
        tutor_wpm=round(audio.tutor_speech_rate_wpm, 1),
        student_words_per_min=round(audio.student_words_per_min, 1),
        student_voices=sum(1 for w in words_by_voice.values() if w >= MIN_STUDENT_VOICE_WORDS),
        tutor_questions=tutor_questions,
        open_questions=open_questions if text_supported else None,
        question_reply_pct=(
            round(100.0 * tfacts.question_reply_rate, 1) if tfacts.question_reply_rate is not None else None
        ),
        median_wait_s=(
            round(audio.median_response_latency_s, 1) if audio.median_response_latency_s is not None else None
        ),
        student_questions=student_questions,
        student_utterances=len(student_segs),
        students_addressed_by_name=len(names) if text_supported else None,
        name_address_count=address_utterances if text_supported else None,
        praise_count=sum(len(_PRAISE.findall(s.text)) for s in tutor_segs) if text_supported else None,
        longest_tutor_monologue_s=round(_longest_tutor_stretch(segments, tutor), 1),
        first_tutor_speech_s=round(min(t.start for t in tutor_turns), 1) if tutor_turns else None,
        last_tutor_speech_s=round(max(t.end for t in tutor_turns), 1) if tutor_turns else None,
        screen_pct=screen_pct,
        timeline=_timeline(d, video, duration_s),
    )
