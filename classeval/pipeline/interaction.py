"""Question - answer - feedback analysis: what the tutor asked, how it was answered and what the tutor did next.

A tutor question is a tutor utterance whose last sentence ends with a question mark. Its answer is the student
utterance that follows within ANSWER_WINDOW_S; the feedback is the tutor utterance that follows the answer within
FEEDBACK_WINDOW_S. The tutor's own question may be quoted (short, names masked); what a child answered is never
quoted, only measured (timing and length).
"""

import re
from statistics import median

from classeval.pipeline.checklist import _mask_and_cap_quote
from classeval.schema import ExchangeExample, InteractionStats, Segment

ANSWER_WINDOW_S = 10.0  # a student reply this soon after a question is its answer
FEEDBACK_WINDOW_S = 8.0  # a tutor reply this soon after the answer is the feedback
CHAIN_GAP_S = 12.0  # answered exchanges closer than this form one back-and-forth stretch
MAX_EXAMPLES = 4
MIN_INITIATIVE_WORDS = 3  # a student remark of at least this many words, not prompted by a question

PRAISE = re.compile(
    r"(?<!\w)(great|excellent|well done|very good|very nice|good job|nice job|perfect|correct|wonderful|"
    r"awesome|brilliant|super|exactly|that's right|that is right)(?!\w)",
    re.IGNORECASE,
)
OPEN_QUESTION = re.compile(
    r"^\W*(?:(?:okay|ok|so|now|and|well|alright)\b[\s,]*)*"
    r"(why|how|what do you think|what would|what if|can you explain|explain|tell me why|what makes)",
    re.IGNORECASE,
)
_ATTENTION = re.compile(
    r"(can you (hear|see)|are you (there|all there|listening)|turn on|unmute|\bmute\b|camera|audible|"
    r"is everyone|everyone ready|\bhello\?)",
    re.IGNORECASE,
)
_CHECKING = re.compile(
    r"(do you (understand|follow|get it)|understood|any (doubt|question)s?|is it clear|got it\?|make(s)? sense|clear\?)",
    re.IGNORECASE,
)
_CORRECTION = re.compile(
    r"(\bno\b|not (quite|exactly|really)|try again|wrong|careful|almost|think again|look again|not correct|incorrect)",
    re.IGNORECASE,
)
_EXPLANATION = re.compile(
    r"(because|that's why|that is why|the reason|so that|for example|remember|notice|look at)", re.IGNORECASE
)
_ACKNOWLEDGE = re.compile(r"^\W*(okay|ok|yes|yeah|right|alright|hmm|uh-huh|mm|i see)\W*$", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

FEEDBACK_KINDS = ("praise", "correction", "explanation", "follow_up_question", "acknowledgement")


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def _question_kind(question: str) -> str:
    if _ATTENTION.search(question):
        return "attention"
    if _CHECKING.search(question):
        return "checking"
    if OPEN_QUESTION.match(question):
        return "open"
    return "closed"


def _feedback_kind(text: str) -> str:
    if PRAISE.search(text):
        return "praise"
    if _CORRECTION.search(text):
        return "correction"
    if _EXPLANATION.search(text):
        return "explanation"
    last = sentences(text)
    if last and last[-1].endswith("?"):
        return "follow_up_question"
    if _ACKNOWLEDGE.match(text) or len(text.split()) <= 4:
        return "acknowledgement"
    return "moved_on"  # the tutor carried on teaching without reacting to the answer


class _Exchange:
    def __init__(self, question_seg: Segment, question: str, kind: str):
        self.question_seg = question_seg
        self.question = question
        self.kind = kind
        self.answer: Segment | None = None
        self.feedback: str = "none"
        self.feedback_seg: Segment | None = None

    @property
    def wait_s(self) -> float | None:
        return None if self.answer is None else max(0.0, self.answer.start - self.question_seg.end)

    @property
    def answer_words(self) -> int | None:
        return None if self.answer is None else len(self.answer.text.split())

    @property
    def end(self) -> float:
        for seg in (self.feedback_seg, self.answer, self.question_seg):
            if seg is not None:
                return seg.end
        return self.question_seg.end


def _exchanges(segs: list[Segment], tutor: str | None, text_supported: bool) -> list[_Exchange]:
    found: list[_Exchange] = []
    for i, seg in enumerate(segs):
        if seg.speaker != tutor:
            continue
        parts = sentences(seg.text)
        if not parts or not parts[-1].endswith("?"):
            continue
        ex = _Exchange(seg, parts[-1], _question_kind(parts[-1]) if text_supported else "unknown")
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if nxt is not None and nxt.speaker != tutor and nxt.start - seg.end <= ANSWER_WINDOW_S:
            ex.answer = nxt
            after = segs[i + 2] if i + 2 < len(segs) else None
            if after is not None and after.speaker == tutor and after.start - nxt.end <= FEEDBACK_WINDOW_S:
                ex.feedback_seg = after
                ex.feedback = _feedback_kind(after.text) if text_supported else "unknown"
        found.append(ex)
    return found


def _longest_chain(exchanges: list[_Exchange]) -> int:
    best = run = 0
    prev_end: float | None = None
    for ex in exchanges:
        if ex.answer is None:
            run, prev_end = 0, None
            continue
        run = run + 1 if prev_end is not None and ex.question_seg.start - prev_end <= CHAIN_GAP_S else 1
        prev_end = ex.end
        best = max(best, run)
    return best


def _pick_examples(content: list[_Exchange]) -> list[ExchangeExample]:
    """Up to MAX_EXAMPLES content questions, open ones first, spread over the class."""
    chosen: list[_Exchange] = []
    for wanted in ("open", "closed"):
        pool = [e for e in content if e.kind == wanted]
        if not pool:
            continue
        room = MAX_EXAMPLES - len(chosen)
        step = max(1, len(pool) // max(1, min(room, 2)))
        chosen.extend(pool[::step][: min(room, 2)])
    chosen.sort(key=lambda e: e.question_seg.start)
    return [
        ExchangeExample(
            t_s=round(e.question_seg.start, 1),
            question=_mask_and_cap_quote(e.question),
            kind=e.kind,
            answered=e.answer is not None,
            wait_s=None if e.wait_s is None else round(e.wait_s, 1),
            answer_words=e.answer_words,
            feedback=e.feedback,
        )
        for e in chosen
    ]


def analyze(segments: list[Segment], tutor: str | None, text_supported: bool = True) -> InteractionStats:
    segs = sorted((s for s in segments if s.speaker is not None and s.text.strip()), key=lambda s: s.start)
    exchanges = _exchanges(segs, tutor, text_supported)

    # student remarks that were not an answer to a tutor question
    answers = {id(ex.answer) for ex in exchanges if ex.answer is not None}
    initiated = sum(
        1
        for s in segs
        if s.speaker != tutor and id(s) not in answers and len(s.text.split()) >= MIN_INITIATIVE_WORDS
    )

    if text_supported:
        content = [e for e in exchanges if e.kind in ("open", "closed")]
        counts = {k: sum(1 for e in exchanges if e.kind == k) for k in ("open", "closed", "checking", "attention")}
    else:
        content = exchanges  # without word lists every question is simply a question
        counts = {}

    answered = [e for e in content if e.answer is not None]
    waits = [e.wait_s for e in answered if e.wait_s is not None]
    lengths = [e.answer_words for e in answered if e.answer_words is not None]
    with_feedback = [e for e in answered if e.feedback in FEEDBACK_KINDS]
    mix = {k: sum(1 for e in answered if e.feedback == k) for k in FEEDBACK_KINDS if any(e.feedback == k for e in answered)}

    return InteractionStats(
        questions_total=len(exchanges),
        content_questions=len(content),
        open_questions=counts.get("open"),
        closed_questions=counts.get("closed"),
        checking_questions=counts.get("checking"),
        attention_questions=counts.get("attention"),
        content_answered_pct=round(100.0 * len(answered) / len(content), 1) if content else None,
        median_wait_s=round(median(waits), 1) if waits else None,
        median_answer_words=round(float(median(lengths)), 1) if lengths else None,
        feedback_pct=(round(100.0 * len(with_feedback) / len(answered), 1) if answered and text_supported else None),
        feedback_mix=mix if text_supported else {},
        student_initiated=initiated,
        longest_exchange_chain=_longest_chain(exchanges),
        examples=_pick_examples(content) if text_supported else [],
    )
