# Prompt: Wave C (Task 4 + Task 5 + Task 6: ingest, ASR, diarization) — to Gemini via Antigravity
Issued by: Claude (planner). Date: 2026-10-22. Runs in parallel with waves B and D. Contains no recording content.

```
Workspace is F:\Eval_class\video_class_eval only. Read GEMINI.md first, including section 2b on efficiency (its rules apply: you never stage, commit or tag, and never touch git hooks or the Truthgate kit), then docs/superpowers/plans/2026-10-22-class-quality-assessor.md: "Global Constraints", "Review Focus", "Task 4", "Task 5" and "Task 6". Execute Tasks 4, 5, 6 in order, step by step (write tests -> run and see them fail -> implement -> run and see them pass), skipping every "Commit" step.

PARALLEL WORK: two other agents are editing other files in this same workspace right now. You may only create or edit the files under OWNS. Never edit config.yaml, classeval/config.py, classeval/schema.py, pyproject.toml, uv.lock, tests/conftest.py or tests/fixtures/clips.py. Settings you need already exist: settings.asr, settings.diarization, settings.models. If you need a new setting, dependency or schema change, STOP and say so in the REPORT. Do not run `uv sync`, `uv add` or `uv run`; use .venv\Scripts\python.exe -m pytest and -m ruff, and run only your own test files. Git: read-only commands only (status, diff, log). Create an empty __init__.py in a package directory you own if it is missing (another agent creating the same empty file is fine). If the Truthgate stop-gate reports problems in files you do not own, do not touch them; disclose them in the REPORT.

OWNS: classeval/pipeline/ingest.py, classeval/pipeline/asr.py, classeval/pipeline/diarize.py, classeval/pipeline/__init__.py, classeval/eval/__init__.py, classeval/eval/vtt.py, scripts/asr_bench.py, scripts/diar_check.py, tests/fixtures/speech.py, tests/pipeline/__init__.py, tests/pipeline/test_ingest.py, test_asr.py, test_diarize.py, tests/eval/__init__.py, tests/eval/test_vtt.py.

EXTRA REQUIREMENTS (beyond the plan text):
 1. The installed ffmpeg is version 9: for the variable-frame-rate test use `-fps_mode vfr` (the old `-vsync` flag is gone). tests/fixtures/clips.py already accepts extra_args.
 2. The SAPI speech fixture needs Windows PowerShell 5.1 (`powershell.exe`, System.Speech), not pwsh. Skip (not fail) the speech-based tests with a clear reason if no voice is installed.
 3. Run all GPU tests with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1; models are already cached (Whisper turbo is at models.whisper_dir; ECAPA and the pyannote repos are in the default Hugging Face cache). Another agent may use the GPU at the same time: keep batch sizes as configured and free GPU memory after every stage.
 4. DIARIZATION IS ECAPA ONLY (decision made for time): do not load, test or work around pyannote in any way. Diarization.method is always "ecapa".
 5. scripts/asr_bench.py and scripts/diar_check.py must be implemented and unit-tested on synthetic input only. Do NOT run them against any real recording and do not look for recordings: the human runs them. They must print numbers only, never text or names.
 6. Never log transcript text; log stage names, durations and counts only.

Finish with the REPORT block from GEMINI.md; "files changed" lists only your files; "numbers" are measured values only (test counts, wall seconds of your own synthetic runs), never estimates.```
