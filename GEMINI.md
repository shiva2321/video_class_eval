# GEMINI.md — rules for the implementing agent (Antigravity / Gemini)

## Project
Class-quality assessor for 8x8 (online chess classes for children). Input: one class recording (upload or link). Output: grade **Poor / Average / Good / Excellent** (or "Insufficient evidence") with specific, evidence-linked observations. Runs **100 % locally** on this Windows 11 PC (RTX 3060 12 GB, 32 GB RAM).

- Design: `docs/planning/00-decision-brief.md`
- Plan: `docs/superpowers/plans/2026-10-22-class-quality-assessor.md`
- Planner/reviewer: Claude. You implement. Work on **one task (or the wave you are told)** at a time.

## 1. DATA FIREWALL — non-negotiable
1. The recordings are confidential and children are identifiable. They live **outside this workspace** (`F:\Eval_class\class recordings`). **Never** open, read, copy, move, upload, print or describe any file there, nor any transcript, frame, audio clip or per-recording output. Never widen the workspace.
2. Code that needs real recordings takes the directory from a CLI argument or `RECORDINGS_DIR` and prints **only aggregate numbers** (counts, durations, rates, scores). Never print transcript text, speaker names, file names, or image data. Write outputs only under the git-ignored dirs `data/`, `jobs/`, `uploads/` and `results/` (numbers only). Per-recording outputs of the evaluation harness (transcripts, caches) live OUTSIDE this workspace, in `F:\Eval_class\derived_data\` (pass it as `--out` / `--runs`), because frames and audio show and carry the children and an IDE agent may index everything inside the workspace. The pipeline deletes extracted frames, audio and scratch crops at the end of every run, successful or not; never write a frame, crop or audio clip anywhere else, and never open one.
3. Tests use **synthetic fixtures generated at test time** (`tests/fixtures.py`: ffmpeg `lavfi`, Windows SAPI speech via PowerShell, PIL/OpenCV drawn frames). Never commit media.
4. Runtime network: **only `127.0.0.1`** (Ollama at `http://127.0.0.1:11434`). Model downloads happen only in `scripts/install.py` and `scripts/download_models.py`. No CDN assets, analytics or telemetry. Set `HF_HUB_OFFLINE=1` at runtime.
5. Never read, print, echo or commit `.env` or tokens. Document variables in `.env.example` only.
6. **Do not stage, commit, tag or push.** Leave your changes in the working tree; the planner (Claude) and the human review and commit them together. Never use `git add -f` or `--no-verify`, and never edit git hooks, `core.hooksPath` or the Truthgate kit (they are the human's own guardrails). If any hook or gate blocks something, stop and report the finding ids.

## 2. Working agreement
- Follow the plan literally: write tests → run (see FAIL) → implement → run (PASS) → report. Wherever the plan says "Commit" or "tag", skip it and list the changed files in your REPORT.
- Touch only files named in your task. If another task's interface must change, **stop and report**; do not improvise.
- Add no dependency that the plan does not list without saying why in the report.
- Windows + PowerShell. Python 3.12 via `uv` (`uv sync`, `uv run ...`). Use `pathlib`. Type hints everywhere; pydantic v2 for contracts; `ruff check .` clean; no module-level mutable state; functions ≲ 60 lines.
- `ffmpeg`/`ffprobe` only through `subprocess.run([...], timeout=...)` with an argument list — **never `shell=True`**.
- Config in `config.yaml`; secrets only from the environment. Logs may contain stage names, durations and counts — never transcript text, names or upload file names.
- GPU stages run sequentially. After each GPU stage: `del model; gc.collect(); torch.cuda.empty_cache()`.
- **Scale rule (a real class is 1 to 3 hours, not 3 minutes):** any code that touches whole-recording data (audio, frames, embeddings, transcripts) must work in bounded chunks and must never put an unbounded number of items through a model in one call. Every such stage needs a test with at least one hour of synthetic input (fake the model if needed) that asserts the batch size stays bounded or peak memory stays small. Memory runaway freezes the user's PC; that is never acceptable.
- Measured numbers (accuracy, speed, counts) may appear in docs or comments only if a script wrote them to a result file under `results/`; otherwise describe the target in words and leave the number to the config or the tests.

## 2b. Efficiency — hard rule (this is a deadline build, not research)
- Build only what the task lists. No extra features, helper scripts, abstractions, refactors, docs or long comments.
- Do not explore: read only the files your task names or owns; no repo tours, no searches outside them.
- Long commands (downloads, model loads, GPU/e2e tests): start once, wait with one long timeout until they finish. Never poll in a loop and never narrate progress.
- While iterating run only the test file you are working on. Run the whole non-GPU suite once at the end; run each GPU/e2e test once at the end.
- Do not re-verify work that already passed. If an approach fails once, report instead of trying alternatives.
- Keep the final answer to the REPORT block plus at most a few lines.

## 3. Task is done when
All task tests pass; `uv run ruff check .` is clean; `uv run pytest -q -m "not gpu and not ollama and not e2e"` passes; GPU/Ollama tests pass when the task says so; `git status` shows only the files the task names (no stray files). Nothing is staged or committed by you.

## 4. Reply format (the human pastes it back to the planner)
End your final message with exactly:

```
REPORT
task: T0N
files changed: <paths>
tests: <command> -> <passed/failed/skipped counts>
deviations: <none | list>
numbers: <only the aggregates the task asks for>
blockers/questions: <none | list>
```
