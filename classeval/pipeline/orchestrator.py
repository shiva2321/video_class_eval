from collections.abc import Sequence
from datetime import UTC, datetime

from classeval.config import Settings
from classeval.schema import STAGES, MediaInfo, Report
from classeval.server.jobs import JobStore


def run(
    job_id: str,
    store: JobStore,
    settings: Settings,
    stages: Sequence[str] | None = None,
) -> None:
    """Stub orchestrator: walks STAGES, writes a schema-valid Report and HTML."""
    target_stages = stages if stages is not None else STAGES

    # Walk through stages
    for idx, stage in enumerate(target_stages):
        prog = float(idx + 1) / float(len(target_stages))
        store.set_stage(job_id, stage, prog)

    job_dir = store.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Build schema-valid stub report
    media_info = MediaInfo(
        duration_s=30.0,
        has_audio=True,
        has_video=True,
        width=1280,
        height=720,
        fps=25.0,
        video_kbps=1000.0,
        container="mp4",
        size_bytes=1024,
    )

    report = Report(
        job_id=job_id,
        created_at=datetime.now(UTC).isoformat(),
        media=media_info,
        language="en",
        language_confidence=0.99,
        grade="Insufficient evidence",
        overall_score=None,
        confidence=0.5,
        coverage=1.0,
        summary="Stub assessment completed with insufficient evidence placeholder.",
        dimensions=[],
        strengths=[],
        improvements=[],
        technical_flags=[],
        not_assessed=[],
        downweighted_signals=[],
        degraded=False,
        timings_s={},
        versions={"classeval": "0.1.0"},
    )

    # Write report.json
    report_json_path = job_dir / "report.json"
    report_json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    # Write report.html
    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Assessment Report - {job_id}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; padding: 2rem; background: #0f172a; color: #f8fafc; }}
    .card {{ background: #1e293b; border-radius: 8px; padding: 1.5rem; max-width: 600px; margin: 0 auto; }}
    .badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; background: #f59e0b; color: #000; font-weight: bold; }}
  </style>
</head>
<body>
  <div class="card">
    <h2>Class Quality Assessment</h2>
    <p>Job ID: <code>{job_id}</code></p>
    <p>Grade: <span class="badge">{report.grade}</span></p>
    <p>{report.summary}</p>
  </div>
</body>
</html>
"""
    report_html_path = job_dir / "report.html"
    report_html_path.write_text(html_content, encoding="utf-8")

    # Mark job as done
    store.update(job_id, status="done", stage="done", progress=1.0)
