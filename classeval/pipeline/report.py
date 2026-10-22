import html
from datetime import UTC, datetime

from classeval.pipeline.checklist import ChecklistResult
from classeval.pipeline.scoring import ScoreResult
from classeval.schema import (
    AudioFacts,
    Diarization,
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

    if result.flags:
        flags_str = ", ".join(result.flags)
        summary_parts.append(f"Technical flags noted: {flags_str}.")

    if checklist.degraded:
        summary_parts.append("Note: Evaluated using rule-based fallback signals (LLM degraded mode).")

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
    downweighted_signals = ["praise_density (weight 0)"]

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
