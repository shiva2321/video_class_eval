import pytest

from classeval.config import load_settings
from classeval.pipeline import class_stats, interaction, report
from classeval.schema import ExchangeExample, InteractionStats, Segment
from tests.pipeline import test_class_stats, test_report


def _seg(start: float, end: float, text: str, who: str) -> Segment:
    return Segment(start=start, end=end, text=text, speaker=who)


def test_an_answered_question_is_measured_from_the_end_of_the_question():
    segs = [
        _seg(0, 3, "Why does the knight move in an L shape?", "T"),
        _seg(5, 9, "Because it jumps over the other pieces", "S1"),
        _seg(10, 12, "Great answer, that is correct.", "T"),
    ]
    it = interaction.analyze(segs, "T")
    assert (it.questions_total, it.content_questions, it.open_questions, it.closed_questions) == (1, 1, 1, 0)
    assert it.content_answered_pct == 100.0
    assert it.median_wait_s == 2.0 and it.median_answer_words == 7.0
    assert it.feedback_pct == 100.0 and it.feedback_mix == {"praise": 1}


def test_a_question_nobody_answers_is_counted_as_unanswered():
    segs = [
        _seg(0, 3, "What is the next move?", "T"),
        _seg(20, 25, "Let us look at the board again.", "T"),
    ]
    it = interaction.analyze(segs, "T")
    assert it.content_questions == 1 and it.content_answered_pct == 0.0
    assert it.median_wait_s is None and it.feedback_pct is None
    assert it.examples[0].answered is False


def test_a_reply_after_the_window_is_not_an_answer():
    segs = [_seg(0, 3, "Which piece moves next?", "T"), _seg(3 + interaction.ANSWER_WINDOW_S + 1, 20, "The rook", "S1")]
    assert interaction.analyze(segs, "T").content_answered_pct == 0.0


@pytest.mark.parametrize(
    "question,kind",
    [
        ("Can you hear me?", "attention"),
        ("Any doubts so far?", "checking"),
        ("Do you understand this move?", "checking"),
        ("Why is the king in danger?", "open"),
        ("So how would you defend it?", "open"),
        ("Is it a checkmate?", "closed"),
    ],
)
def test_question_types(question, kind):
    assert interaction._question_kind(question) == kind


def test_attention_and_checking_questions_are_not_content_questions():
    segs = [
        _seg(0, 2, "Can you hear me?", "T"),
        _seg(3, 4, "Yes", "S1"),
        _seg(10, 12, "Any doubts?", "T"),
        _seg(13, 14, "No", "S1"),
        _seg(20, 23, "Which square does the bishop attack?", "T"),
    ]
    it = interaction.analyze(segs, "T")
    assert (it.questions_total, it.content_questions) == (3, 1)
    assert (it.attention_questions, it.checking_questions, it.closed_questions) == (1, 1, 1)


@pytest.mark.parametrize(
    "reply,kind",
    [
        ("Well done, that is right.", "praise"),
        ("Not quite, try again.", "correction"),
        ("It moves there because the square is protected.", "explanation"),
        ("And what happens after that?", "follow_up_question"),
        ("Okay.", "acknowledgement"),
        ("Now let us open the next position and set up the pieces on the board.", "moved_on"),
    ],
)
def test_tutor_feedback_types(reply, kind):
    assert interaction._feedback_kind(reply) == kind


def test_feedback_only_counts_when_the_tutor_replies_soon_after_the_answer():
    segs = [
        _seg(0, 2, "What is the best move here?", "T"),
        _seg(3, 4, "The knight", "S1"),
        _seg(4 + interaction.FEEDBACK_WINDOW_S + 5, 40, "Excellent work everyone.", "T"),
    ]
    it = interaction.analyze(segs, "T")
    assert it.content_answered_pct == 100.0 and it.feedback_pct == 0.0 and it.feedback_mix == {}


def test_student_remarks_nobody_asked_for_are_counted_separately_from_answers():
    segs = [
        _seg(0, 3, "Which piece is that?", "T"),
        _seg(4, 5, "The bishop", "S1"),
        _seg(30, 33, "I have a question about castling", "S2"),
        _seg(40, 41, "Okay", "S2"),
    ]
    assert interaction.analyze(segs, "T").student_initiated == 1  # the one-word reply is too short to count


def test_the_longest_run_of_connected_exchanges():
    segs = []
    t = 0.0
    for _ in range(3):
        segs += [_seg(t, t + 2, "What is the next move?", "T"), _seg(t + 3, t + 4, "Knight takes", "S1"), _seg(t + 5, t + 6, "Good.", "T")]
        t += 8
    segs += [_seg(200, 202, "Which piece is attacked?", "T"), _seg(203, 204, "The pawn", "S1")]
    assert interaction.analyze(segs, "T").longest_exchange_chain == 3


def test_examples_quote_the_tutor_only_and_never_the_childs_answer():
    segs = [
        _seg(0, 3, "Why does the bishop need an open diagonal?", "T"),
        _seg(4, 8, "Pineapple banana secret answer words", "S1"),
        _seg(9, 10, "Good.", "T"),
    ]
    it = interaction.analyze(segs, "T")
    assert it.examples and it.examples[0].question.startswith("Why does the bishop")
    dumped = it.model_dump_json().lower()
    assert "pineapple" not in dumped and "banana" not in dumped
    assert it.examples[0].answer_words == 5


def test_examples_are_capped_and_questions_are_short():
    segs = []
    for i in range(12):
        long_q = "Why " + " ".join(["word"] * 30) + "?"
        segs += [_seg(i * 20, i * 20 + 2, long_q, "T"), _seg(i * 20 + 3, i * 20 + 4, "Yes it is", "S1")]
    it = interaction.analyze(segs, "T")
    assert len(it.examples) <= interaction.MAX_EXAMPLES
    assert all(len(ex.question.split()) <= 15 for ex in it.examples)


def test_unsupported_language_counts_questions_but_does_not_type_or_quote_them():
    segs = [
        _seg(0, 3, "Which piece is that?", "T"),
        _seg(4, 6, "The bishop", "S1"),
        _seg(7, 9, "Great, correct.", "T"),
    ]
    it = interaction.analyze(segs, "T", text_supported=False)
    assert it.questions_total == 1 and it.content_questions == 1 and it.content_answered_pct == 100.0
    assert it.open_questions is None and it.closed_questions is None
    assert it.feedback_pct is None and it.feedback_mix == {} and it.examples == []
    assert it.median_wait_s == 1.0  # timing is language independent


def test_no_tutor_means_no_questions():
    segs = [_seg(0, 3, "Which piece is that?", "S1")]
    assert interaction.analyze(segs, None).questions_total == 0


def test_class_stats_carries_the_interaction_block():
    stats = class_stats.compute(*test_class_stats._inputs())
    assert stats.interaction is not None and stats.interaction.questions_total == 2
    assert stats.interaction.content_answered_pct == 100.0


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_interaction_token")
    return load_settings("config.yaml", env_file=None)


def _page_with_interaction(settings, interaction_stats):
    media, tr, d, audio, video, tfacts, checklist, score_res = test_report._make_report_inputs(settings)
    rep = report.build_report(
        job_id="test_job_12345", media=media, tr=tr, d=d, audio=audio, video=video, tfacts=tfacts,
        checklist=checklist, result=score_res, timings_s={}, versions={}, summary="s",
    )
    stats = class_stats.compute(tr, d, audio, video, tfacts, media)
    rep.stats = stats.model_copy(update={"interaction": interaction_stats})
    return report.render_html(rep)


def test_the_report_shows_the_questions_and_answers_panel(settings):
    computed = class_stats.compute(*test_class_stats._inputs()).interaction
    assert computed is not None and computed.examples
    page = _page_with_interaction(settings, computed)
    assert "Questions and answers" in page and "never quoted" in page
    assert "http://" not in page and "https://" not in page


def test_the_report_escapes_a_question_taken_from_the_transcript(settings):
    nasty = InteractionStats(
        questions_total=1,
        content_questions=1,
        open_questions=1,
        closed_questions=0,
        content_answered_pct=100.0,
        examples=[ExchangeExample(t_s=3.0, question="<script>alert(1)</script> why", kind="open", answered=True, wait_s=1.0, answer_words=3, feedback="praise")],
    )
    page = _page_with_interaction(settings, nasty)
    assert "<script>alert" not in page and "&lt;script&gt;" in page


def test_the_panel_is_left_out_when_there_were_no_questions(settings):
    page = _page_with_interaction(settings, interaction.analyze([], "T"))
    assert "Questions and answers" not in page
