import re

import numpy as np

from classeval.config import Settings
from classeval.schema import (
    Detection,
    Diarization,
    Evidence,
    Segment,
    Transcript,
    TranscriptFacts,
)

# Default praise lexicon if not in config
DEFAULT_PRAISE = [
    "good job",
    "very good",
    "great job",
    "excellent",
    "well done",
    "awesome",
    "nice move",
    "brilliant",
    "super",
    "fantastic",
    "wonderful",
]

CHESS_TERMS = [
    "king",
    "queen",
    "rook",
    "bishop",
    "knight",
    "pawn",
    "check",
    "checkmate",
    "mate",
    "move",
    "capture",
    "castle",
    "castling",
]


def _matches_phrase(text: str, phrase: str) -> bool:
    """Case-insensitive whole-word or whole-phrase boundary match."""
    pattern = r"\b" + re.escape(phrase.strip().lower()) + r"\b"
    return bool(re.search(pattern, text.lower()))


def _mask_and_cap_quote(text: str, max_words: int = 15) -> str:
    """Mask student/tutor names and cap quote length to at most 15 words."""
    # Mask greetings followed by name
    masked = re.sub(
        r"\b(hello|hi|hey|welcome)\s+([A-Z][a-z]+)\b",
        r"\1 [Name]",
        text,
        flags=re.IGNORECASE,
    )
    # Mask common Indian/Western child names
    masked = re.sub(
        r"\b(Aarav|Vivaan|Aditya|Vihaan|Arjun|Reyansh|Muhammad|Sai|Ayaan|Krishna|Ishaan|Shaurya|Dhruv|Atharv|Kabir|Ria|Ananya|Diya|Aadhya|Myra|Pari|Isha|Anvi|Saanvi|Fatima|Zoya|Sara|John|David|Zira)\b",
        "[Name]",
        masked,
        flags=re.IGNORECASE,
    )
    words = masked.split()
    if len(words) > max_words:
        masked = " ".join(words[:max_words])
    return masked


def _make_evidence(seg: Segment) -> Evidence:
    text = _mask_and_cap_quote(seg.text)
    return Evidence(t_start=seg.start, t_end=seg.end, kind="quote", text=text)


def compute(tr: Transcript, d: Diarization, settings: Settings) -> TranscriptFacts:
    """Extract rule-based detectors and interaction facts from transcript."""
    lex = settings.lexicon
    segments = tr.segments
    tutor_speaker = d.tutor

    max_t = max((s.end for s in segments), default=0.0)
    duration_min = max(max_t / 60.0, 1e-6)

    # Gate check: language confidence and logprob quality
    logprobs = [s.avg_logprob for s in segments if s.avg_logprob is not None]
    median_lp = float(np.median(logprobs)) if logprobs else 0.0

    # Without an identified tutor nothing can be attributed to the tutor, and children's speech must never be
    # quoted as tutor speech, so every detector is "not assessed" (fail closed).
    unreliable = tr.language_confidence < 0.5 or (logprobs and median_lp < -1.0) or tutor_speaker is None
    if unreliable or not segments:
        default_dets = {
            "greeting_opening": Detection(assessed=not unreliable, score=0.0 if not unreliable else None, hits=[]),
            "rapport": Detection(assessed=not unreliable, score=0.0 if not unreliable else None, hits=[]),
            "self_intro_new_joiners": Detection(assessed=not unreliable, score=0.0 if not unreliable else None, hits=[]),
            "individual_interaction": Detection(assessed=not unreliable, score=0.0 if not unreliable else None, hits=[]),
            "homework": Detection(assessed=not unreliable, score=0.0 if not unreliable else None, hits=[]),
            "closing": Detection(assessed=not unreliable, score=0.0 if not unreliable else None, hits=[]),
            "chess_engagement": Detection(assessed=not unreliable, score=0.0 if not unreliable else None, hits=[]),
            "praise_density": Detection(assessed=not unreliable, score=0.0 if not unreliable else None, hits=[]),
        }
        return TranscriptFacts(
            detections=default_dets,
            distinct_responders=0,
            question_reply_rate=None,
            question_count=0,
        )

    # 1. Greeting opening (first 4 min: t <= 240s)
    greeting_hits: list[Evidence] = []
    for s in segments:
        if (
            s.start <= 240.0
            and (tutor_speaker is None or s.speaker == tutor_speaker)
            and any(_matches_phrase(s.text, phrase) for phrase in lex.greeting)
        ):
            greeting_hits.append(_make_evidence(s))

    # 2. Rapport (first 5 min: t <= 300s)
    rapport_hits: list[Evidence] = []
    for s in segments:
        if (
            s.start <= 300.0
            and (tutor_speaker is None or s.speaker == tutor_speaker)
            and any(_matches_phrase(s.text, phrase) for phrase in lex.rapport)
        ):
            rapport_hits.append(_make_evidence(s))

    # 3. Self intro
    intro_hits: list[Evidence] = []
    for s in segments:
        if (tutor_speaker is None or s.speaker == tutor_speaker) and any(
            _matches_phrase(s.text, phrase) for phrase in lex.self_intro
        ):
            intro_hits.append(_make_evidence(s))

    # 4. Homework (matches in last 10 min count double)
    hw_hits: list[Evidence] = []
    hw_score = 0.0
    last_10_min_start = max(0.0, max_t - 600.0)
    for s in segments:
        if (tutor_speaker is None or s.speaker == tutor_speaker) and any(
            _matches_phrase(s.text, phrase) for phrase in lex.homework
        ):
            hw_hits.append(_make_evidence(s))
            if s.start >= last_10_min_start:
                hw_score += 2.0
            else:
                hw_score += 1.0

    # 5. Closing (last 5 min: t >= max_t - 300s)
    closing_hits: list[Evidence] = []
    last_5_min_start = max(0.0, max_t - 300.0)
    for s in segments:
        if (
            s.start >= last_5_min_start
            and (tutor_speaker is None or s.speaker == tutor_speaker)
            and any(_matches_phrase(s.text, phrase) for phrase in lex.closing)
        ):
            closing_hits.append(_make_evidence(s))

    # 6. Praise density (reported, weight 0)
    praise_lexicon = getattr(lex, "praise", DEFAULT_PRAISE)
    praise_hits: list[Evidence] = []
    for s in segments:
        if (tutor_speaker is None or s.speaker == tutor_speaker) and any(
            _matches_phrase(s.text, phrase) for phrase in praise_lexicon
        ):
            praise_hits.append(_make_evidence(s))
    praise_rate = round(float(len(praise_hits)) / duration_min, 2)

    # 7. Chess engagement (student utterances with squares or chess terms per min)
    chess_hits: list[Evidence] = []
    square_pattern = re.compile(r"\b[a-h][1-8]\b", re.IGNORECASE)
    for s in segments:
        if tutor_speaker is not None and s.speaker != tutor_speaker:
            has_square = bool(square_pattern.search(s.text))
            has_term = any(_matches_phrase(s.text, term) for term in CHESS_TERMS)
            if has_square or has_term:
                chess_hits.append(_make_evidence(s))
    chess_engagement_rate = round(float(len(chess_hits)) / duration_min, 2)

    # 8. Individual interaction & question-reply rate
    question_count = 0
    answered_questions = 0
    responders: set[str] = set()

    for idx, s in enumerate(segments):
        if (tutor_speaker is None or s.speaker == tutor_speaker) and s.text.strip().endswith("?"):
            question_count += 1
            # Look for student answer within 6 seconds
            for next_s in segments[idx + 1 :]:
                gap = next_s.start - s.end
                if 0.0 <= gap <= 6.0:
                    if tutor_speaker is None or next_s.speaker != tutor_speaker:
                        answered_questions += 1
                        if next_s.speaker:
                            responders.add(next_s.speaker)
                        break
                elif gap > 6.0:
                    break

    reply_rate = round(float(answered_questions) / float(question_count), 2) if question_count > 0 else None
    distinct_responders = len(responders)

    detections = {
        "greeting_opening": Detection(
            assessed=True,
            score=float(len(greeting_hits)),
            hits=greeting_hits,
        ),
        "rapport": Detection(
            assessed=True,
            score=float(len(rapport_hits)),
            hits=rapport_hits,
        ),
        "self_intro_new_joiners": Detection(
            assessed=True,
            score=float(len(intro_hits)),
            hits=intro_hits,
        ),
        "individual_interaction": Detection(
            assessed=True,
            score=float(distinct_responders),
            hits=[],
        ),
        "homework": Detection(
            assessed=True,
            score=hw_score,
            hits=hw_hits,
        ),
        "closing": Detection(
            assessed=True,
            score=float(len(closing_hits)),
            hits=closing_hits,
        ),
        "chess_engagement": Detection(
            assessed=True,
            score=chess_engagement_rate,
            hits=chess_hits,
        ),
        "praise_density": Detection(
            assessed=True,
            score=praise_rate,
            hits=praise_hits,
        ),
    }

    return TranscriptFacts(
        detections=detections,
        distinct_responders=distinct_responders,
        question_reply_rate=reply_rate,
        question_count=question_count,
    )
