# Prompt: Wave D (Task 7 + Task 8 + Task 9: audio features, screen-state timeline, rule detectors) — to Gemini via Antigravity
Issued by: Claude (planner). Date: 2026-10-22. Runs in parallel with waves B and C. Contains no recording content.

```
Workspace is F:\Eval_class\video_class_eval only. Read GEMINI.md first, including section 2b on efficiency (its rules apply: you never stage, commit or tag, and never touch git hooks or the Truthgate kit), then docs/superpowers/plans/2026-10-22-class-quality-assessor.md: "Global Constraints", "Review Focus", "Task 7", "Task 8" and "Task 9". Execute Tasks 7, 8, 9 in order, step by step (write tests -> run and see them fail -> implement -> run and see them pass), skipping every "Commit" step.

PARALLEL WORK: two other agents are editing other files in this same workspace right now. You may only create or edit the files under OWNS. Never edit config.yaml, classeval/config.py, classeval/schema.py, pyproject.toml, uv.lock, tests/conftest.py or tests/fixtures/clips.py. Settings you need already exist: settings.screen_state (thresholds, logo scales, CLIP prompts), settings.lexicon, settings.scoring. If you need a new setting, dependency or schema change, STOP and say so in the REPORT. Do not run `uv sync`, `uv add` or `uv run`; use .venv\Scripts\python.exe -m pytest and -m ruff, and run only your own test files. Git: read-only commands only (status, diff, log). Create an empty __init__.py in a package directory you own if it is missing (another agent creating the same empty file is fine). If the Truthgate stop-gate reports problems in files you do not own, do not touch them; disclose them in the REPORT.

OWNS: classeval/pipeline/audio_features.py, screen_state.py, video_features.py, transcript_features.py, classeval/pipeline/__init__.py (create if missing), tests/fixtures/frames.py, tests/pipeline/__init__.py (create if missing), tests/pipeline/test_audio_features.py, test_screen_state.py, test_video_features.py, test_transcript_features.py.

EXTRA REQUIREMENTS (beyond the plan text):
 1. Task 9 matching must be on word boundaries: add test_hi_does_not_match_inside_other_words (the lexicon entry "hi" must not fire on "this", "which" or "think") and test_phrases_match_case_insensitively. The homework lexicon was deliberately narrowed (no "puzzle" or "practice"): add test_puzzle_alone_is_not_homework.
 2. Task 9 detectors read their word lists only from settings.lexicon; no word lists hard-coded in the module.
 3. Task 8 logo template is assets/8x8_logo_left8.png (already in the repo). The real recordings may show the logo at other scales: use settings.screen_state.logo_scales. Camera-off inside a small thumbnail is out of scope; say so in the module docstring.
 4. Task 8 CLIP model loads from the Hugging Face cache by settings.models.clip with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1. Another agent may use the GPU at the same time: batch size is settings.screen_state.clip_batch_size; free GPU memory afterwards.
 5. Task 8: do NOT implement tutor-name reading, VisionLLM or read_name_card (cut for time). VideoFacts.tutor_name stays None and compute() has no name_reader parameter. Do not import anything from classeval/pipeline/llm.py.
 6. Task 7 must be pure computation over arrays and the schema objects: no model loading. All thresholds that the plan pins (8 s dead air, 2 s dropouts, -70 dBFS, 50 ms frames, 0.999 clipping) are module constants with those exact values and a one-line comment pointing to the plan.
 7. Never log transcript text; log counts only.

Finish with the REPORT block from GEMINI.md; "files changed" lists only your files; "numbers" are measured values only (test counts), never estimates.
```
