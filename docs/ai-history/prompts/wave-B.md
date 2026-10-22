# Prompt: Wave B (Task 2 + Task 3: server, job store, UI shell, ops scripts) — to Gemini via Antigravity
Issued by: Claude (planner). Date: 2026-10-22. Runs in parallel with waves C and D. Contains no recording content.

```
Workspace is F:\Eval_class\video_class_eval only. Read GEMINI.md first, including section 2b on efficiency (its rules apply: you never stage, commit or tag, and never touch git hooks or the Truthgate kit), then docs/superpowers/plans/2026-10-22-class-quality-assessor.md: "Global Constraints", "Review Focus", "Task 2" and "Task 3". Execute Task 2 then Task 3 in order, step by step (write tests -> run and see them fail -> implement -> run and see them pass), skipping every "Commit" step.

PARALLEL WORK: two other agents are editing other files in this same workspace right now. You may only create or edit the files under OWNS. Never edit config.yaml, classeval/config.py, classeval/schema.py, pyproject.toml, uv.lock, tests/conftest.py or tests/fixtures/*. If you need a new setting, dependency or schema change, STOP and say so in the REPORT. Do not run `uv sync`, `uv add` or `uv run`; use .venv\Scripts\python.exe -m pytest and -m ruff, and run only your own test files. Git: read-only commands only (status, diff, log). Create an empty __init__.py in a package directory you own if it is missing. If the Truthgate stop-gate reports problems in files you do not own, do not touch them; disclose them in the REPORT.

OWNS: classeval/server/** (including static/), classeval/worker.py, classeval/pipeline/orchestrator.py (the stub only; create classeval/pipeline/__init__.py empty if missing), scripts/run.ps1, scripts/funnel.ps1, scripts/smoke_public.py, tests/server/** (with __init__.py), tests/test_smoke_client.py.

EXTRA REQUIREMENTS (beyond the plan text):
 1. Compare the access code with hmac.compare_digest, never ==.
 2. load_settings(...) is the only place settings are read; create_app(settings) receives them. For tests build Settings via load_settings("config.yaml", env_file=None) with ACCESS_CODE set through monkeypatch.
 3. Do NOT run scripts/funnel.ps1 (any option) or powercfg: public exposure and power settings are the human's call. You may run scripts/run.ps1 locally on 127.0.0.1 only. For a local end-to-end check set a throwaway ACCESS_CODE in the process environment (do not read or print .env; real environment variables now beat the .env file).
 4. The stub orchestrator.run must walk every stage in schema.STAGES with progress values and write a schema-valid Report with grade "Insufficient evidence"; a later task replaces its internals, so keep the signature exactly as in the plan.
 5. The UI must not request anything external (no CDN, fonts or analytics); the test in the plan enforces it. Keep the UI clean and readable on a phone: single column, large tap targets, visible stage list with the current stage highlighted, human-friendly error text for every API error code.
 6. Never log upload file names, job contents or the access code.
 7. Upload only: no link/URL input, no fetch.py, no url job source (cut for time and to avoid a server-side request risk).

Finish with the REPORT block from GEMINI.md; "files changed" lists only your files; "numbers" are measured values only (test counts), never estimates.
```
