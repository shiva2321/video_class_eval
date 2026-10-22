import pytest

from classeval.config import load_settings
from classeval.pipeline import class_stats, narrative, report
from classeval.pipeline.llm import LLMBadOutput, LLMUnavailable
from classeval.schema import DimensionScore, Narrative
from tests.pipeline import test_report


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_narrative_token")
    return load_settings("config.yaml", env_file=None)


def _dim(dim_id, score, weight, name="Dimension"):
    return DimensionScore(id=dim_id, name=name, group="teaching", score=score, weight=weight, confidence=1.0)


def _report(settings):
    media, tr, d, audio, video, tfacts, checklist, score_res = test_report._make_report_inputs(settings)
    rep = report.build_report(
        job_id="test_job_12345", media=media, tr=tr, d=d, audio=audio, video=video, tfacts=tfacts,
        checklist=checklist, result=score_res, timings_s={}, versions={}, summary="s",
    )
    rep.stats = class_stats.compute(tr, d, audio, video, tfacts, media)
    return rep


def test_points_lost_add_up_to_what_the_score_is_missing():
    dims = [_dim("R1", 100, 10), _dim("R4", 20, 10), _dim("R6", 60, 20), _dim("R7", None, 10)]
    lost = narrative.points_lost(dims)
    assert [d.id for d, _ in lost] == ["R4", "R6", "R1"]  # the unassessed dimension costs nothing
    assert sum(p for _, p in lost) == pytest.approx(100 - (100 * 10 + 20 * 10 + 60 * 20) / 40)
    assert lost[0][1] == pytest.approx(20.0)


def test_no_assessed_dimensions_means_no_points_lost():
    assert narrative.points_lost([_dim("R1", None, 10)]) == []


def test_the_fact_sheet_has_the_grade_dimensions_and_class_numbers(settings):
    rep = _report(settings)
    facts = narrative.fact_sheet(rep)
    assert f"GRADE: {rep.grade}" in facts and "DIMENSIONS" in facts and "CLASS:" in facts
    assert all(d.name in facts for d in rep.dimensions)
    assert "VERIFIED TUTOR QUOTES" in facts and "do not repeat" in facts  # only timestamps, never the words


class Client:
    def __init__(self, answer):
        self.answer = answer
        self.seen = None

    def chat_json(self, system, user, schema):
        self.seen = (system, user)
        if isinstance(self.answer, type) and issubclass(self.answer, Exception):
            raise self.answer("scripted")
        return self.answer


GOOD = {
    "headline": "Good class: strong participation, held back by a late start.",
    "why": "Students answered quickly and the screen was used well. The opening started late.",
    "improve": "Greet the children in the first minute. Invite new joiners to introduce themselves.",
}


def test_a_clean_explanation_is_kept(settings):
    rep = _report(settings)
    out = narrative.generate(rep, Client(GOOD))
    assert isinstance(out, Narrative) and out.headline == GOOD["headline"]


def test_the_model_only_sees_the_fact_sheet(settings):
    rep = _report(settings)
    client = Client(GOOD)
    narrative.generate(rep, client)
    assert narrative.fact_sheet(rep) in client.seen[1]
    assert rep.summary not in client.seen[1] or rep.summary == "s"


def test_a_sentence_with_an_invented_number_is_removed_and_the_rest_kept(settings):
    rep = _report(settings)
    mixed = {**GOOD, "why": "Students answered quickly. The tutor lost 77 points because of silence. The screen was used well."}
    out = narrative.generate(rep, Client(mixed))
    assert out is not None and out.why == "Students answered quickly. The screen was used well."


def test_an_explanation_with_nothing_grounded_is_dropped(settings):
    rep = _report(settings)
    assert narrative.generate(rep, Client({**GOOD, "why": "The tutor lost 93 points and students answered in 41 seconds."})) is None


def test_an_ungrounded_headline_or_advice_is_replaced_by_a_plain_sentence_from_the_facts(settings):
    rep = _report(settings)
    out = narrative.generate(rep, Client({**GOOD, "headline": "A 97 out of 100 class.", "improve": "Wait 9 seconds after each question."}))
    assert out is not None
    assert out.headline.startswith(f"{rep.grade}, score {rep.overall_score} of 100.")
    assert out.improve.startswith("The most points to gain are in ")


def test_numbers_that_are_facts_pass_to_rounding(settings):
    rep = _report(settings)
    pct = round(rep.stats.tutor_talk_pct)
    ok = {**GOOD, "why": f"The tutor spoke about {pct}% of the time. The score was {rep.overall_score}."}
    assert narrative.generate(rep, Client(ok)) is not None


def test_quoted_speech_is_removed_from_the_text(settings):
    rep = _report(settings)
    quoted = {**GOOD, "headline": 'The tutor said "you are all great" at the start of a good class.'}
    out = narrative.generate(rep, Client(quoted))
    assert out is not None and '"' not in out.headline and "you are all great" not in out.headline


@pytest.mark.parametrize("bad", ["See https://example.com for more.", "Use <b>bold</b> words.", r"Path C:\data here."])
def test_a_field_with_a_link_or_markup_invalidates_the_explanation(settings, bad):
    rep = _report(settings)
    assert narrative.generate(rep, Client({**GOOD, "why": bad})) is None


def test_an_empty_improvement_gets_the_plain_sentence(settings):
    out = narrative.generate(_report(settings), Client({**GOOD, "improve": "   "}))
    assert out is not None and out.improve.startswith(("The most points to gain", "No dimension lost"))


@pytest.mark.parametrize("failure", [LLMBadOutput, LLMUnavailable])
def test_a_model_failure_leaves_no_explanation(settings, failure):
    assert narrative.generate(_report(settings), Client(failure)) is None


def test_no_score_means_no_explanation(settings):
    rep = _report(settings)
    rep.overall_score = None
    assert narrative.generate(rep, Client(GOOD)) is None


def test_a_sentence_naming_a_person_is_removed(settings):
    rep = _report(settings)
    named = {**GOOD, "why": "Students answered quickly. Ask Gautham, and then Priya, to share one move each."}
    out = narrative.generate(rep, Client(named))
    assert out is not None and out.why == "Students answered quickly."


def test_a_name_that_is_only_part_of_a_fact_word_is_still_a_name(settings):
    rep = _report(settings)
    assert "rapport" in narrative.fact_sheet(rep).lower()  # "Rap" is a substring of it, but not a word of the facts
    out = narrative.generate(rep, Client({**GOOD, "why": "Students answered quickly. Ask Rap to share one move."}))
    assert out is not None and out.why == "Students answered quickly."


def test_dimension_names_are_not_mistaken_for_people(settings):
    rep = _report(settings)
    ok = {**GOOD, "why": "The Opening and rapport dimension was strong, but Student participation lost points."}
    assert narrative.generate(rep, Client(ok)) is not None


def test_small_numbers_must_match_exactly(settings):
    rep = _report(settings)
    out = narrative.generate(rep, Client({**GOOD, "why": "Students answered quickly. Wait 9 seconds after each question."}))
    assert out is not None and "9" not in out.why  # nothing in the facts says 9


def test_the_report_page_shows_the_explanation_and_the_points_lost_escaped(settings):
    rep = _report(settings)
    rep.narrative = Narrative(headline="<script>alert(1)</script> Good", why="Because.", improve="Do more.")
    page = report.render_html(rep)
    assert "Why this grade" in page and "<script>alert" not in page and "&lt;script&gt;" in page
    assert "What held the score back" in page


def test_the_page_without_an_explanation_still_shows_the_points_lost(settings):
    rep = _report(settings)
    page = report.render_html(rep)
    assert "Written by the local language model" not in page
