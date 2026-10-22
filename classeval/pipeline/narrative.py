"""The written explanation of the grade ("why this grade").

The local model is given only the finished report's facts (scores, weights, reasons, measurements, verified tutor quotes,
class statistics), never the transcript, and writes a short explanation. The result is sanitised and every number in it
must be one of the facts, otherwise the explanation is dropped and the template summary stays. `points_lost` is the
deterministic companion: how many points of the 100 each dimension cost.
"""

import logging
import re
from typing import Any

from classeval.pipeline.checklist import find_names
from classeval.pipeline.llm import LLMBadOutput, LLMUnavailable, OllamaClient
from classeval.schema import DimensionScore, Narrative, Report

logger = logging.getLogger("classeval.narrative")

MAX_QUOTES = 5
FIELD_LIMITS = {"headline": 180, "why": 800, "improve": 500}
_DIM_ID = re.compile(r"\bR\d{1,2}\b")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_QUOTED = re.compile(r'["“”][^"“”]*["“”]')
_UNSAFE = re.compile(r"https?:|www\.|[<>{}\[\]`]|\\", re.IGNORECASE)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "maxLength": FIELD_LIMITS["headline"]},
        "why": {"type": "string", "maxLength": FIELD_LIMITS["why"]},
        "improve": {"type": "string", "maxLength": FIELD_LIMITS["improve"]},
    },
    "required": ["headline", "why", "improve"],
}

SYSTEM_PROMPT = """You explain a class-quality grade to the tutor's quality-review team. The facts below were measured from one recording
of an online class for children. Use ONLY these facts: never add anything that is not in them, never guess, and never write a
number that is not in the facts. Do not quote anyone, do not name any person, and do not mention children's words.
Write plain, specific, kind sentences. Give a reason for a score only if the facts state it. If a dimension has no reason or
measurement, just give its score and weight and say nothing about the cause (never invent causes such as tone, gestures or
clarity of explanation).
headline: one sentence stating the grade and the two things that mattered most.
why: three to five sentences on what drove the grade, naming the dimensions (by name, not code) and their measurements or reasons.
improve: two or three concrete suggestions for the tutor, each tied to a dimension that lost points and to a measured fact.
If a dimension was not assessed, say so rather than judging it."""


def points_lost(dimensions: list[DimensionScore]) -> list[tuple[DimensionScore, float]]:
    """Points of the 100-point score each assessed dimension cost, largest first (they sum to 100 minus the score)."""
    assessed = [d for d in dimensions if d.score is not None]
    total = sum(d.weight for d in assessed)
    if total <= 0:
        return []
    lost = [(d, d.weight / total * (100.0 - float(d.score or 0.0))) for d in assessed]
    return sorted(lost, key=lambda x: -x[1])


def fact_sheet(rep: Report) -> str:
    lines = [
        (
            f"GRADE: {rep.grade}. Score {rep.overall_score} of 100. Confidence {rep.confidence:.2f}. "
            f"Share of the rubric assessed: {round(rep.coverage * 100)}%."
        ),
        f"FLAGS: {', '.join(rep.technical_flags) if rep.technical_flags else 'none'}.",
        "DIMENSIONS (name | weight | score | points lost | reason or measurement):",
    ]
    lost = {d.id: points for d, points in points_lost(rep.dimensions)}
    for d in rep.dimensions:
        if d.score is None:
            lines.append(f"- {d.name} | weight {d.weight:g} | not assessed | {d.note or 'could not be measured'}")
            continue
        detail = " ".join(e.text for e in d.evidence if e.kind == "measure")[:220]
        reason = d.note or ""
        lines.append(
            f"- {d.name} | weight {d.weight:g} | score {d.score:g} | points lost {lost.get(d.id, 0.0):.1f} | "
            f"{(reason + ' ' + detail).strip()}"
        )
    quotes = [(d.name, e) for d in rep.dimensions for e in d.evidence if e.kind == "quote"][:MAX_QUOTES]
    if quotes:
        lines.append("VERIFIED TUTOR QUOTES (for context only, do not repeat them):")
        lines += [f"- {name} at {e.t_start:.0f} s" for name, e in quotes]
    s = rep.stats
    if s is not None:
        lines.append(
            f"CLASS: {s.duration_min:g} minutes. Tutor spoke {s.tutor_talk_pct:g}% of the time, students {s.student_talk_pct:g}%, "
            f"silence {s.silence_pct:g}%. The tutor first spoke at {s.first_tutor_speech_s} s."
        )
        it = s.interaction
        if it is not None and it.content_questions:
            lines.append(
                f"QUESTIONS: {it.content_questions} content questions, {it.content_answered_pct}% answered by a student, "
                f"typical wait {it.median_wait_s} s, tutor reacted to {it.feedback_pct}% of answers."
            )
    if rep.topics:
        lines.append("TOPICS COVERED: " + "; ".join(rep.topics[:6]))
    return "\n".join(lines)


def _numbers(text: str) -> list[float]:
    return [float(n) for n in _NUMBER.findall(_DIM_ID.sub("", text))]


def _clean(text: str, limit: int) -> str:
    text = " ".join(_QUOTED.sub("", str(text)).replace('"', "").split())
    if not text or _UNSAFE.search(text):
        return ""
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return text


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _sentence_is_grounded(sentence: str, words: set[str], known: list[float]) -> bool:
    """Numbers must be facts (to rounding for numbers of ten and over, exactly below) and a capitalised name-like word
    must be a whole word of the facts (dimension names are; a person's name is not, even when it happens to be part of
    a longer fact word)."""
    if any(name.lower() not in words for name in find_names(sentence)):
        return False
    return all(any(abs(n - k) <= (0.51 if n >= 10 else 0.051) for k in known) for n in _numbers(sentence))


def grounded(text: str, facts: str) -> str:
    """Only the sentences of `text` whose numbers and names are facts."""
    known = _numbers(facts)
    words = {w.lower() for w in _WORD.findall(facts)}
    return " ".join(x for x in _SENTENCE.split(text) if x and _sentence_is_grounded(x, words, known))


def _default_headline(rep: Report) -> str:
    worst = [d.name for d, lost in points_lost(rep.dimensions)[:2] if lost >= 1.0]
    held = f" {' and '.join(worst)} held the score back." if worst else ""
    return f"{rep.grade}, score {rep.overall_score} of 100.{held}"


def _default_improve(rep: Report) -> str:
    worst = [d.name for d, lost in points_lost(rep.dimensions)[:2] if lost >= 1.0]
    return f"The most points to gain are in {' and '.join(worst)}." if worst else "No dimension lost a notable number of points."


def generate(rep: Report, client: OllamaClient) -> Narrative | None:
    """The model's explanation keeping only grounded sentences. If the headline or the improvement advice has nothing
    grounded left, a plain sentence built from the facts replaces it; without a grounded explanation there is none."""
    if rep.overall_score is None:
        return None
    facts = fact_sheet(rep)
    try:
        answer = client.chat_json(SYSTEM_PROMPT, f"<facts>\n{facts}\n</facts>", SCHEMA)
    except (LLMBadOutput, LLMUnavailable) as exc:
        logger.warning("no explanation written: %s", type(exc).__name__)
        return None
    if not isinstance(answer, dict):
        return None
    parts = {k: grounded(_clean(answer.get(k, ""), FIELD_LIMITS[k]), facts) for k in FIELD_LIMITS}
    if not parts["why"]:
        logger.warning("no grounded sentence was left in the written explanation; it was dropped")
        return None
    return Narrative(
        headline=parts["headline"] or _default_headline(rep),
        why=parts["why"],
        improve=parts["improve"] or _default_improve(rep),
    )
