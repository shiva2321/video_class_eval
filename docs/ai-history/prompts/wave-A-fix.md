# Prompt: Wave A fixes (review of Task 0 + Task 1) — to Gemini via Antigravity
Issued by: Claude (planner/reviewer). Date: 2026-10-22. Contains no recording content.
Review verdict: Task 1 (schema, config, tests) is faithful to the plan; Task 0 works on the GPU stack but has the defects below.

```
Workspace is F:\Eval_class\video_class_eval only. Re-read GEMINI.md (commit rules changed: you never stage, commit or tag) and the "Global Constraints" and "Task 0" sections of docs/superpowers/plans/2026-10-22-class-quality-assessor.md (they were edited). Do NOT touch git hooks or the Truthgate kit. Fix exactly the items below, nothing else, then run the checks and report.

FIX 1 (scripts/bootstrap.ps1): delete the whole branch that tests/copies from F:\IELTS_GURU. That path is another project outside this workspace; it must not appear anywhere in this repo. ffmpeg logic becomes: use ffmpeg/ffprobe from PATH if they run; else use F:\tools\ffmpeg\bin if present (session PATH only, do not persist PATH at User scope); else `winget install --id Gyan.FFmpeg -e`; else print a clear instruction. Also stop hard-coding F:\models paths inside the embedded Python: read them from config.yaml.

FIX 2 (scripts/bootstrap.ps1, ACCESS_CODE): generate it with System.Security.Cryptography.RandomNumberGenerator (not System.Random), 12 characters from letters and digits, never echo it. Keep "append missing keys only, never overwrite existing values".

FIX 3 (bootstrap model downloads): make bootstrap and runtime use the same mechanism. Download CLIP, ECAPA and, when HF_TOKEN works, the pyannote repos (pyannote/speaker-diarization-3.1, pyannote/segmentation-3.0, pyannote/wespeaker-voxceleb-resnet34-LM) into the DEFAULT Hugging Face cache (snapshot_download without local_dir). Whisper stays under models.whisper_dir via faster-whisper's download_root; the Systran/faster-whisper-large-v3 fallback is also downloaded there. Do not create F:\models\hf or F:\models\speechbrain. Remove the "local_dir" downloads.

FIX 4 (classeval/config.py + tests/test_config.py): load_settings must not read the repo's real .env implicitly. New signature: load_settings(path="config.yaml", env_file=".env"). tests pass env_file=None. Reject ACCESS_CODE equal to the placeholder "change_this_to_a_secure_token" with a ValueError that names ACCESS_CODE. Add tests: (a) test_access_code_required must pass even when a real .env containing ACCESS_CODE exists in the current directory (use tmp_path + monkeypatch.chdir to prove it), (b) test_env_file_is_read_when_requested (tmp_path/.env with ACCESS_CODE), (c) test_placeholder_access_code_rejected.

FIX 5 (config.yaml lexicon.homework): remove "puzzle" and "practice" (they occur in nearly every chess lesson). Use exactly: "homework", "home work", "assignment", "worksheet", "practice sheet", "practice at home", "do at home". Leave the other lexicon lists unchanged.

FIX 6 (config.yaml scoring): replace the whole `scoring:` block with the plan's Task 11 priors, one key per quantity, as [lo, hi] (inverse when lo > hi):
  branded_ratio_tutor_speaking: [0.0, 0.7]
  student_words_per_min: [5.0, 45.0]
  dead_air_per_min: [0.8, 0.05]
  distinct_responders: [0.0, 5.0]
  question_reply_rate: [0.2, 0.8]
  pitch_std_semitones: [1.5, 4.0]
  energy_cv: [0.4, 1.0]
  board_share_ratio: [0.15, 0.6]
Add a test that these eight keys exist and each has two numbers. Do not invent other keys.

FIX 7 (scripts/gpu_smoke.py): (a) the ECAPA fallback savedir "tmp_ecapa" must not be created in the repo root: use ".cache/ecapa" (git-ignored) or the default cache; delete the existing tmp_ecapa folder. (b) pyannote: "skipped: ..." only when HF_TOKEN is unset; if a token is set and loading fails, report "error: <exception class and first line>". (c) read model names/paths from config.yaml, no F:\models literals except the config default. (d) the script must pass with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 set.

FIX 8 (tests/test_env.py): add `import pyannote.audio` to test_imports.

CHECKS TO RUN AND REPORT:
 1. uv run ruff check .   2. uv run pytest -q -m "not gpu and not ollama and not e2e"
 3. with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1: uv run python scripts/gpu_smoke.py (paste the JSON)
 4. run scripts/bootstrap.ps1 end to end once (idempotent) and report ONLY: which keys exist in .env as "KEY: set/empty" (never values), and the disk size of each downloaded model.
 5. `git status --short` (must show no tmp_ecapa, nothing staged).
Do not read or print .env values. Finish with the REPORT block from GEMINI.md.
```
