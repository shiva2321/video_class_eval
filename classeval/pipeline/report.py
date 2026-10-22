import html
from datetime import UTC, datetime

from classeval.pipeline.checklist import ChecklistResult
from classeval.pipeline.class_stats import screen_group
from classeval.pipeline.scoring import ScoreResult
from classeval.schema import (
    AudioFacts,
    ClassStats,
    Diarization,
    InteractionStats,
    MediaInfo,
    Report,
    Transcript,
    TranscriptFacts,
    VideoFacts,
)


def summarize(result: ScoreResult, checklist: ChecklistResult) -> str:
    """Generate deterministic executive summary from scores and verified observations without LLM."""
    grade = result.grade
    overall = result.overall_score
    conf_pct = round(result.confidence * 100)

    if grade == "Insufficient evidence":
        reasons = []
        if result.coverage < 0.5:
            reasons.append(f"low rubric coverage ({round(result.coverage * 100)}%)")
        if "language_uncertain" in result.flags:
            reasons.append("uncertain spoken language confidence")
        if "language_unsupported" in result.flags:
            reasons.append("a spoken language the transcript checks do not support yet")
        reason_str = ", ".join(reasons) if reasons else "recording duration or speech activity below evaluation threshold"
        return (
            f"Overall Assessment: Insufficient evidence. "
            f"The recording could not be reliably evaluated due to {reason_str}. "
            f"Confidence: {conf_pct}%."
        )

    score_str = f"{overall:.1f}/100" if overall is not None else "N/A"
    summary_parts = [
        f"Overall Assessment: {grade} (Score: {score_str}, Confidence: {conf_pct}%)."
    ]

    assessed = [d for d in result.dimensions if d.score is not None]
    if assessed:
        best_dim = max(assessed, key=lambda x: x.score if x.score is not None else -1)
        weakest_dim = min(assessed, key=lambda x: x.score if x.score is not None else 999)

        summary_parts.append(
            f"Highest performing area is {best_dim.id} ({best_dim.name}) at {best_dim.score:.1f}/100."
        )
        if weakest_dim.id != best_dim.id:
            summary_parts.append(
                f"Primary area for growth is {weakest_dim.id} ({weakest_dim.name}) at {weakest_dim.score:.1f}/100."
            )

    if "language_unsupported" in result.flags:
        summary_parts.append(
            "Most of this class was not in English, so transcript-based checks are marked not assessed; "
            "measurements that do not depend on language are still reported."
        )

    if result.flags:
        flags_str = ", ".join(result.flags)
        summary_parts.append(f"Technical flags noted: {flags_str}.")

    if checklist.degraded and "llm_degraded" in result.flags:
        summary_parts.append("Note: the language model could not analyse this class, so the grade rests on word-list signals only and is capped at Good.")
    elif checklist.chunks_failed:
        summary_parts.append(
            f"Note: {checklist.chunks_failed} of {checklist.chunks_total} sections of the class could not be analysed in depth; confidence is lowered."
        )

    return " ".join(summary_parts)


def build_report(
    job_id: str,
    media: MediaInfo,
    tr: Transcript,
    d: Diarization,
    audio: AudioFacts,
    video: VideoFacts,
    tfacts: TranscriptFacts,
    checklist: ChecklistResult,
    result: ScoreResult,
    timings_s: dict[str, float],
    versions: dict[str, str],
    summary: str,
) -> Report:
    """Assemble final validated Report object."""
    not_assessed = [d_score.id for d_score in result.dimensions if d_score.score is None]
    downweighted_signals = [
        "praise-word count (weight 0: weak classes use praise words as much as strong ones)",
        "audio dropouts (weight 0: recorded meetings contain many digital silences during normal pauses)",
        "recording length and bitrate (not related to teaching quality)",
    ]

    return Report(
        job_id=job_id,
        created_at=datetime.now(UTC).isoformat(),
        media=media,
        language=tr.language,
        language_confidence=tr.language_confidence,
        grade=result.grade,
        overall_score=result.overall_score,
        confidence=result.confidence,
        coverage=result.coverage,
        summary=summary,
        dimensions=result.dimensions,
        strengths=checklist.strengths,
        improvements=checklist.improvements,
        topics=list(checklist.topics),
        technical_flags=result.flags,
        not_assessed=not_assessed,
        downweighted_signals=downweighted_signals,
        degraded=checklist.degraded,
        timings_s=timings_s,
        versions=versions,
    )


def _fmt_ts(sec: float | None) -> str:
    if sec is None or sec < 0:
        return "[--:--]"
    m = int(sec // 60)
    s = int(sec % 60)
    return f"[{m:02d}:{s:02d}]"


_SCREEN_COLORS = {
    "board": "#f59e0b",
    "lesson_video": "#8b5cf6",
    "face_on_screen": "#3b82f6",
    "camera_off": "#ef4444",
    "other": "#94a3b8",
    "NONE": "#e2e8f0",
}
_SCREEN_LABELS = {
    "board": "board / puzzle / game",
    "lesson_video": "lesson video",
    "face_on_screen": "a face on screen",
    "camera_off": "camera off",
    "other": "slides or other shared screen",
    "NONE": "no frame",
}


_LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "ml": "Malayalam", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu",
}


def _language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code.upper())


def _stat_card(value: str, label: str) -> str:
    return (
        f'<div class="stat-card"><div class="stat-value">{html.escape(value)}</div>'
        f'<div class="stat-label">{html.escape(label)}</div></div>'
    )


_FEEDBACK_LABELS = {
    "praise": "praise",
    "correction": "correction",
    "explanation": "explanation",
    "follow_up_question": "follow-up question",
    "acknowledgement": "acknowledgement",
    "moved_on": "carried on without reacting",
    "none": "no reaction",
    "unknown": "n/a",
}


def _render_interaction(it: InteractionStats | None) -> str:
    """Questions and answers panel: what was asked, how it was answered, what the tutor did next."""
    if it is None or it.questions_total == 0:
        return ""

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.0f}%"

    typed = it.open_questions is not None
    cards = [
        _stat_card(str(it.content_questions), "content questions"),
        _stat_card(pct(it.content_answered_pct), "answered by a student"),
        _stat_card("n/a" if it.median_wait_s is None else f"{it.median_wait_s:.1f} s", "typical wait for the answer"),
        _stat_card("n/a" if it.median_answer_words is None else f"{it.median_answer_words:.0f} words", "typical answer length"),
        _stat_card(pct(it.feedback_pct), "answers the tutor reacted to"),
        _stat_card(str(it.student_initiated), "student remarks nobody asked for"),
        _stat_card(str(it.longest_exchange_chain), "longest run of question-answer rounds"),
    ]
    types_html = ""
    if typed:
        parts = [
            (it.open_questions, "open (why / how)"),
            (it.closed_questions, "closed"),
            (it.checking_questions, 'checking understanding ("any doubts?")'),
            (it.attention_questions, 'attention or technical ("can you hear me?")'),
        ]
        types_html = (
            '<div class="legend">'
            + "".join(f"<span class=\"legend-item\"><strong>{n}</strong>&nbsp;{html.escape(label)}</span>" for n, label in parts)
            + "</div>"
        )
    mix_html = ""
    if it.feedback_mix:
        mix_html = (
            '<div class="tl-label">What the tutor did after an answer</div><div class="legend">'
            + "".join(
                f"<span class=\"legend-item\"><strong>{n}</strong>&nbsp;{html.escape(_FEEDBACK_LABELS.get(k, k))}</span>"
                for k, n in it.feedback_mix.items()
            )
            + "</div>"
        )
    examples_html = ""
    if it.examples:
        rows = []
        for ex in it.examples:
            outcome = (
                f"answered after {ex.wait_s:.1f} s with {ex.answer_words} words; tutor reaction: "
                f"{html.escape(_FEEDBACK_LABELS.get(ex.feedback, ex.feedback))}"
                if ex.answered and ex.wait_s is not None
                else "not answered by a student"
            )
            rows.append(
                f"<li><span class=\"chip\">{_fmt_ts(ex.t_s)}</span> &ldquo;{html.escape(ex.question)}&rdquo; "
                f"&mdash; {html.escape(ex.kind)} question; {outcome}</li>"
            )
        examples_html = '<div class="tl-label">Examples (tutor\'s own words; children\'s answers are never quoted)</div><ul class="topic-list">' + "".join(rows) + "</ul>"
    return (
        '<div class="tl-label">Questions and answers</div>'
        f'<div class="stat-grid">{"".join(cards)}</div>{types_html}{mix_html}{examples_html}'
    )


def _render_glance(stats: ClassStats | None, topics: list[str]) -> str:
    """The 'Class at a glance' section: countable facts, two minute-by-minute strips and the topics covered."""
    if stats is None:
        return ""

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.0f}%"

    cards = [
        _stat_card(f"{stats.duration_min:.0f} min", "class length"),
        _stat_card(pct(stats.tutor_talk_pct), "tutor speaking"),
        _stat_card(pct(stats.student_talk_pct), "students speaking"),
        _stat_card(pct(stats.silence_pct), "silence"),
        _stat_card(str(stats.student_voices), "student voices heard"),
        _stat_card(
            str(stats.tutor_questions) if stats.open_questions is None else f"{stats.tutor_questions} ({stats.open_questions} open)",
            "tutor questions",
        ),
        _stat_card(pct(stats.question_reply_pct), "questions answered"),
        _stat_card("n/a" if stats.median_wait_s is None else f"{stats.median_wait_s:.1f} s", "typical wait for an answer"),
        _stat_card(str(stats.student_questions), "questions from students"),
        _stat_card("n/a" if stats.students_addressed_by_name is None else str(stats.students_addressed_by_name), "children called by name"),
        _stat_card("n/a" if stats.praise_count is None else str(stats.praise_count), "praise moments"),
        _stat_card(f"{stats.longest_tutor_monologue_s:.0f} s", "longest tutor stretch without a student"),
        _stat_card(f"{stats.tutor_wpm:.0f}", "tutor words per minute"),
        _stat_card(f"{stats.student_words_per_min:.0f}", "student words per minute"),
    ]

    speech_cols = []
    for k, b in enumerate(stats.timeline):
        tutor_h = min(100.0, 100.0 * b.tutor_s / 60.0)
        student_h = min(100.0 - tutor_h, 100.0 * b.student_s / 60.0)
        speech_cols.append(
            f'<div class="tl-col" title="minute {k + 1}: tutor {b.tutor_s:.0f} s, students {b.student_s:.0f} s">'
            f'<div class="tl-student" style="height:{student_h:.1f}%"></div>'
            f'<div class="tl-tutor" style="height:{tutor_h:.1f}%"></div></div>'
        )
    screen_cells = [
        f'<div class="tl-cell" style="background:{_SCREEN_COLORS.get(screen_group(b.screen) if b.screen != "NONE" else "NONE", "#e2e8f0")}" '
        f'title="minute {k + 1}: {html.escape(_SCREEN_LABELS.get(b.screen, b.screen))}"></div>'
        for k, b in enumerate(stats.timeline)
    ]
    legend = "".join(
        f'<span class="legend-item"><span class="legend-dot" style="background:{_SCREEN_COLORS[g]}"></span>'
        f"{html.escape(_SCREEN_LABELS[g])} {stats.screen_pct.get(g, 0):.0f}%</span>"
        for g in ("board", "lesson_video", "face_on_screen", "camera_off", "other")
        if stats.screen_pct.get(g)
    )
    topics_html = (
        '<div class="tl-label">What the class covered</div><ul class="topic-list">'
        + "".join(f"<li>{html.escape(t)}</li>" for t in topics)
        + "</ul>"
        if topics
        else ""
    )
    return f"""
    <div class="section">
      <div class="section-title">Class at a glance</div>
      <div class="stat-grid">{"".join(cards)}</div>
      {_render_interaction(stats.interaction)}
      <div class="tl-label">Who spoke, minute by minute</div>
      <div class="tl-speech">{"".join(speech_cols)}</div>
      <div class="legend"><span class="legend-item"><span class="legend-dot" style="background:#3b82f6"></span>tutor</span>
        <span class="legend-item"><span class="legend-dot" style="background:#10b981"></span>students</span>
        <span class="legend-item"><span class="legend-dot" style="background:#e2e8f0"></span>silence</span></div>
      <div class="tl-label">What was on screen, minute by minute</div>
      <div class="tl-screen">{"".join(screen_cells)}</div>
      <div class="legend">{legend}</div>
      {topics_html}
    </div>
"""


def render_html(report: Report) -> str:
    """Render self-contained single-page HTML report without external asset URLs."""
    grade_colors = {
        "Excellent": ("#065f46", "#d1fae5", "#10b981"),
        "Good": ("#1e40af", "#dbeafe", "#3b82f6"),
        "Average": ("#92400e", "#fef3c7", "#f59e0b"),
        "Poor": ("#991b1b", "#fee2e2", "#ef4444"),
        "Insufficient evidence": ("#374151", "#f3f4f6", "#6b7280"),
    }
    text_color, bg_color, bar_color = grade_colors.get(
        report.grade, ("#374151", "#f3f4f6", "#6b7280")
    )

    score_display = f"{report.overall_score:.1f} / 100" if report.overall_score is not None else "N/A"
    conf_pct = round(report.confidence * 100)
    cov_pct = round(report.coverage * 100)

    # Dimension rows
    dim_rows = []
    for d in report.dimensions:
        score_text = f"{d.score:.1f}" if d.score is not None else "Not Assessed"
        fill_pct = max(0.0, min(100.0, d.score)) if d.score is not None else 0.0
        bar_html = f'<div class="bar-track"><div class="bar-fill" style="width: {fill_pct:.1f}%;"></div></div>' if d.score is not None else '<div class="na-badge">Not Assessed</div>'
        dim_rows.append(
            f"""<tr>
              <td class="dim-id">{html.escape(d.id)}</td>
              <td class="dim-name">
                <strong>{html.escape(d.name)}</strong>
                <span class="dim-group">({html.escape(d.group)})</span>
                {f'<div class="dim-note">{html.escape(d.note)}</div>' if d.note else ''}
              </td>
              <td class="dim-weight">{d.weight:.0f}%</td>
              <td class="dim-bar-col">{bar_html}</td>
              <td class="dim-score">{score_text}</td>
            </tr>"""
        )

    # Strengths cards
    strength_cards = []
    for s in report.strengths:
        chips = " ".join(
            f'<span class="chip">{_fmt_ts(ev.t_start)} {html.escape(ev.text)}</span>'
            for ev in s.evidence
        )
        strength_cards.append(
            f"""<div class="card strength-card">
              <div class="card-title">[{html.escape(s.dimension)}] {html.escape(s.text)}</div>
              <div class="card-evidence">{chips}</div>
            </div>"""
        )
    if not strength_cards:
        strength_cards.append('<div class="empty-state">No specific strengths observed.</div>')

    # Improvement cards
    improvement_cards = []
    for imp in report.improvements:
        chips = " ".join(
            f'<span class="chip">{_fmt_ts(ev.t_start)} {html.escape(ev.text)}</span>'
            for ev in imp.evidence
        )
        improvement_cards.append(
            f"""<div class="card improvement-card">
              <div class="card-title">[{html.escape(imp.dimension)}] {html.escape(imp.text)}</div>
              <div class="card-evidence">{chips}</div>
            </div>"""
        )
    if not improvement_cards:
        improvement_cards.append('<div class="empty-state">No specific improvement areas flagged.</div>')

    # Metadata & Flags
    flags_html = (
        "".join(f'<span class="badge flag-badge">{html.escape(f)}</span>' for f in report.technical_flags)
        if report.technical_flags
        else '<span class="badge ok-badge">None</span>'
    )
    not_assessed_html = (
        ", ".join(html.escape(d) for d in report.not_assessed)
        if report.not_assessed
        else "None (all assessed)"
    )

    lang_notice = ""
    if "language_unsupported" in report.technical_flags:
        lang_notice = (
            f'<div class="section notice">Language notice: this class was mostly in {html.escape(_language_name(report.language))}. '
            "Transcript-based checks (opening, introductions, homework, conclusion and similar) are validated for English "
            "only, so they are marked not assessed instead of guessed. Everything that does not depend on language "
            "(talk time, participation, pace, screen use, audio) is still measured and shown below.</div>"
        )
    elif "language_uncertain" in report.technical_flags:
        lang_notice = (
            '<div class="section notice">Language notice: the spoken language could not be identified reliably, so '
            "transcript-based checks are marked not assessed. Language-independent measurements are still shown.</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Class Quality Assessment Report - {html.escape(report.job_id)}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; padding: 24px; line-height: 1.5; }}
    .container {{ max-width: 960px; margin: 0 auto; background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; overflow: hidden; }}
    .banner {{ background: {bg_color}; color: {text_color}; padding: 28px 32px; border-bottom: 2px solid {bar_color}; }}
    .banner-title {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; opacity: 0.8; margin-bottom: 8px; }}
    .grade-display {{ font-size: 38px; font-weight: 800; line-height: 1.1; margin-bottom: 12px; }}
    .meta-row {{ display: flex; gap: 24px; font-size: 14px; font-weight: 500; margin-top: 8px; }}
    .section {{ padding: 24px 32px; border-bottom: 1px solid #f1f5f9; }}
    .section-title {{ font-size: 18px; font-weight: 700; color: #0f172a; margin-bottom: 16px; }}
    .summary-text {{ font-size: 15px; color: #334155; line-height: 1.6; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th {{ text-align: left; padding: 10px 12px; font-weight: 600; color: #64748b; border-bottom: 1px solid #e2e8f0; }}
    td {{ padding: 12px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }}
    .dim-id {{ font-weight: 700; color: #2563eb; width: 48px; }}
    .dim-group {{ font-size: 12px; color: #94a3b8; font-weight: 400; }}
    .dim-note {{ font-size: 12px; color: #475569; margin-top: 4px; }}
    .dim-weight {{ color: #64748b; width: 50px; text-align: center; }}
    .dim-score {{ font-weight: 700; width: 100px; text-align: right; }}
    .bar-track {{ height: 10px; background: #e2e8f0; border-radius: 5px; overflow: hidden; width: 100%; }}
    .bar-fill {{ height: 100%; background: #3b82f6; border-radius: 5px; }}
    .na-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; background: #e2e8f0; font-size: 12px; color: #64748b; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }}
    .card {{ padding: 16px; border-radius: 6px; border: 1px solid #e2e8f0; }}
    .strength-card {{ background: #f0fdf4; border-color: #bbf7d0; }}
    .improvement-card {{ background: #fffbeb; border-color: #fef3c7; }}
    .card-title {{ font-weight: 700; font-size: 14px; margin-bottom: 8px; }}
    .card-evidence {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
    .chip {{ display: inline-block; padding: 3px 8px; background: #ffffff; border-radius: 4px; font-size: 12px; color: #334155; border: 1px solid #cbd5e1; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
    .flag-badge {{ background: #fee2e2; color: #991b1b; margin-right: 6px; }}
    .ok-badge {{ background: #f1f5f9; color: #475569; }}
    .empty-state {{ color: #94a3b8; font-style: italic; font-size: 14px; }}
    .notice {{ background: #fffbeb; color: #92400e; font-size: 14px; border-left: 4px solid #f59e0b; }}
    .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 16px; }}
    .stat-card {{ padding: 10px 12px; border: 1px solid #e2e8f0; border-radius: 6px; background: #f8fafc; }}
    .stat-value {{ font-size: 20px; font-weight: 800; color: #0f172a; }}
    .stat-label {{ font-size: 12px; color: #64748b; }}
    .tl-label {{ font-size: 13px; font-weight: 600; color: #475569; margin: 14px 0 6px; }}
    .tl-speech {{ display: flex; align-items: flex-end; gap: 1px; height: 56px; }}
    .tl-col {{ flex: 1; height: 100%; display: flex; flex-direction: column-reverse; background: #f1f5f9; min-width: 2px; }}
    .tl-tutor {{ background: #3b82f6; }}
    .tl-student {{ background: #10b981; }}
    .tl-screen {{ display: flex; gap: 1px; height: 16px; }}
    .tl-cell {{ flex: 1; min-width: 2px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 6px; font-size: 12px; color: #475569; }}
    .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }}
    .topic-list {{ padding-left: 20px; font-size: 14px; color: #334155; }}
    @media (max-width: 700px) {{
      body {{ padding: 8px; }}
      .grid-2 {{ grid-template-columns: 1fr; }}
      .section {{ padding: 16px; }}
      .meta-row {{ flex-wrap: wrap; gap: 8px 16px; }}
      .grade-display {{ font-size: 30px; }}
    }}
    .footer {{ padding: 16px 32px; background: #f8fafc; font-size: 12px; color: #64748b; display: flex; justify-content: space-between; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="banner">
      <div class="banner-title">Class Quality Assessment</div>
      <div class="grade-display">{html.escape(report.grade)}</div>
      <div class="meta-row">
        <span>Score: {score_display}</span>
        <span>Rubric Coverage: {cov_pct}%</span>
        <span>Confidence: {conf_pct}%</span>
        <span>Language: {html.escape(report.language)} ({int(report.language_confidence * 100)}%)</span>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Executive Summary</div>
      <div class="summary-text">{html.escape(report.summary)}</div>
    </div>
    {lang_notice}
    {_render_glance(report.stats, report.topics)}

    <div class="section">
      <div class="section-title">Rubric Dimension Scores</div>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Dimension</th>
            <th>Weight</th>
            <th>Performance</th>
            <th style="text-align: right;">Score</th>
          </tr>
        </thead>
        <tbody>
          {"".join(dim_rows)}
        </tbody>
      </table>
    </div>

    <div class="section">
      <div class="grid-2">
        <div>
          <div class="section-title">Key Strengths</div>
          {"".join(strength_cards)}
        </div>
        <div>
          <div class="section-title">Areas for Improvement</div>
          {"".join(improvement_cards)}
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Evaluation Details</div>
      <p style="font-size: 13px; color: #475569; margin-bottom: 8px;"><strong>Technical Flags:</strong> {flags_html}</p>
      <p style="font-size: 13px; color: #475569; margin-bottom: 8px;"><strong>Not Assessed Dimensions:</strong> {not_assessed_html}</p>
      <p style="font-size: 13px; color: #475569; margin-bottom: 8px;"><strong>Down-weighted Signals:</strong> {", ".join(html.escape(s) for s in report.downweighted_signals)}</p>
      <p style="font-size: 13px; color: #475569;"><strong>Mode:</strong> {"Degraded Rule Fallback" if report.degraded else "Standard Full Pipeline"}</p>
    </div>

    <div class="footer">
      <span>Job ID: {html.escape(report.job_id)}</span>
      <span>Evaluated: {html.escape(report.created_at)}</span>
    </div>
  </div>
</body>
</html>"""
