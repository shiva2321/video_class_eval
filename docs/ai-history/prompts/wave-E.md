# Prompt: Wave E (Task 10 + Task 11: LLM checklist extraction, scoring, report) — to Gemini via Antigravity
Issued by: Claude (planner). Date: 2026-10-22. Contains no recording content.

```
Workspace is F:\Eval_class\video_class_eval only. Read GEMINI.md first, including section 2b on efficiency (its rules apply: you never stage, commit or tag, and never touch git hooks or the Truthgate kit), then docs/superpowers/plans/2026-10-22-class-quality-assessor.md: "Global Constraints", "Review Focus", "Task 10" and "Task 11". Execute Task 10 then Task 11 in order, step by step (write tests -> run and see them fail -> implement -> run and see them pass), skipping every "Commit" step. Tasks 1-9 are done and committed; import their real types from classeval.schema and the real functions from classeval.pipeline.*.

OWNS (only these): classeval/pipeline/llm.py, classeval/pipeline/checklist.py, classeval/pipeline/scoring.py, classeval/pipeline/report.py, tests/pipeline/test_llm.py, test_checklist.py, test_scoring.py, test_report.py. Never edit config.yaml, classeval/config.py, classeval/schema.py, pyproject.toml, uv.lock or any other existing file; if you need a new setting, dependency or schema change, STOP and say so in the REPORT. Do not run `uv sync`, `uv add` or `uv run`; use .venv\Scripts\python.exe -m pytest / -m ruff and run only your own test files while iterating. Git: read-only commands only.

EXTRA REQUIREMENTS (beyond the plan text):
 1. summarize() has NO llm parameter and makes NO model call: it is a deterministic template (grade, strongest and weakest dimension, one sentence on confidence) — see the edited Task 11 interface.
 2. Ollama chat requests must send "think": false so Qwen3-VL returns the JSON directly; if a response still arrives with empty content, treat it as LLMBadOutput. Model name comes from settings.models.llm, URL from settings.ollama.base_url, timeout default 180 s, temperature 0 and seed 7, num_ctx 16384, keep_alive set so the model stays loaded between chunks of one job and unload() sends keep_alive 0.
 3. Every unit test uses httpx.MockTransport (no network, no GPU). Write exactly ONE live test, marked @ollama, with a tiny synthetic transcript, and run it once at the end.
 4. The LLM sees transcript text only through the chunk prompt. Evidence quotes returned by the model are accepted only after verify_evidence (fuzzy match against the real transcript, tutor speech only, at most 15 words, person names masked as [student]/[tutor]); anything unverifiable is dropped and, if a dimension loses all its evidence, its verdict becomes not_observed.
 5. Scoring: put every threshold, weight, range and cut-off behind settings (settings.weights, settings.scoring, settings.cutoffs); no magic numbers in scoring.py except the verdict map (met 100, partial 60, not_met 10) and the pinned gate constants from the plan, defined once as module constants.
 6. render_html: one self-contained page, no external URLs, readable on a phone, grade banner with colour AND text (not colour alone), dimension table with bars, strengths and improvements with [mm:ss] timestamp chips, flags, "not assessed", "signals we down-weighted". Escape every string (html.escape) — report text includes model output.
 7. Never log transcript text or quotes.

Finish with the REPORT block from GEMINI.md; "files changed" lists only your files; "numbers" are measured values only (test counts), never estimates.
```
