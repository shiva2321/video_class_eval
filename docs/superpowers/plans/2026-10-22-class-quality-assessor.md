# Class Quality Assessor Implementation Plan

> **For the implementing agent (Antigravity / Gemini):** follow `GEMINI.md`. Execute **one wave at a time**, tasks in order, steps literally. Steps use checkbox syntax. The human pastes your `REPORT` block back to the planner (Claude), who reviews the diff and runs the tests independently before the next wave.

**Goal:** a locally-run web app that takes a chess-class recording and returns Poor / Average / Good / Excellent with evidence-linked, specific observations.

**Architecture:** "sandwich": deterministic perception (ffmpeg, VAD+ASR, diarization, audio DSP, screen-state timeline) → local LLM/VLM reads transcript + measured facts and returns schema-constrained checklist JSON with quotes/timestamps → deterministic weighted-rubric scoring with gates, calibrated thresholds, confidence and an evidence-linked report. FastAPI + vanilla-JS UI, chunked uploads, one GPU worker, exposed through Tailscale Funnel.

**Tech Stack:** Python 3.12 (uv), PyTorch 2.6 cu124, faster-whisper, pyannote.audio 3.3.2 (SpeechBrain ECAPA fallback), OpenCV, CLIP ViT-B/32 (zero-shot screen states), librosa/pyloudnorm/parselmouth, Ollama (Qwen3-VL-8B, fallback Qwen2.5-VL-7B), FastAPI/uvicorn/SSE, pytest, ruff.

**Spec:** `docs/planning/00-decision-brief.md` (v0.2). Read §3.4–3.5 (rubric, screen layout) and §4.5 (scoring) before Waves D–F.

## Global Constraints
- Windows 11, PowerShell, Python **3.12** via `uv`; venv at `.venv`; RTX 3060 **12 GB** — GPU stages strictly sequential; after a job GPU memory used must be ≤ baseline + 400 MB.
- Whisper under `F:\models\whisper`; every other Hugging Face model lives in the **default Hugging Face cache** (reviewed: loads with `HF_HUB_OFFLINE=1`); Ollama keeps its default model dir. Names live in `config.yaml`, never hard-coded. Runtime always sets `HF_HUB_OFFLINE=1`.
- Runtime network: **127.0.0.1 only** (Ollama). No CDN/fonts/analytics. `HF_HUB_OFFLINE=1` at runtime.
- Grade strings exactly: `"Poor"`, `"Average"`, `"Good"`, `"Excellent"`, and `"Insufficient evidence"`.
- Dimension ids `R1`–`R12`; weights (sum 100): R1 8, R2 10, R3 4, R4 10, R5 5, R6 15, R7 12, R8 15, R9 8, R10 6, R11 4, R12 3. Teaching = R4–R11 (75), Presentation = R1,R2,R3,R12 (25).
- Provisional grade cut-offs on the 0–100 overall score: Poor < 45; Average 45 ≤ s < 62; Good 62 ≤ s < 78; Excellent ≥ 78 (replaced by calibrated values in Task 13).
- Limits: upload ≤ **2.5 GB**; duration ≤ **10 800 s**; analysable minimum **600 s** (shorter → "Insufficient evidence"); chunk ≤ **8 MiB**; job TTL **24 h**; one GPU worker.
- Sampling: audio 16 kHz mono PCM s16 WAV; frames 0.2 fps, 640 px wide JPEG; 24 VLM keyframes.
- Evidence quotes: tutor speech only, ≤ **15 words**, person names masked as `[student]` / `[tutor]`.
- Auth: header `X-Access-Code` (or cookie `access`) equal to env `ACCESS_CODE`; job/upload ids = `secrets.token_urlsafe(16)`.
- Logging: stage names, durations, counts only. Never transcript text, names, upload file names.
- Tests: synthetic fixtures only. Scripts touching real recordings print numbers only and write under ignored dirs.
- **Commit policy:** agents never stage, commit, tag or push. Wherever a task says "Commit" or "tag", skip it and list the changed files in the REPORT; the planner and the human commit after review. Measured numbers in docs come only from result files under `results/` that scripts wrote.

## Review Focus
Inputs the spec implies but no happy-path test exercises, most likely first (each is pinned by a test in the owning task):
1. **Silent / music-only / no-speech recording** → "Insufficient evidence", no crash (Task 11).
2. **Non-English-dominant speech** (Malayalam/Hindi) → language banner, text-derived dimensions "not assessed" instead of scored 0 (Tasks 9, 11).
3. **Odd containers:** audio-only, video-only, corrupt/truncated, > 3 h, variable frame rate → friendly error or flagged partial (Tasks 2, 4).
4. **Tutor not identifiable** (one speaker; two speakers within 15 % talk time; 1-on-1 demo) → low `tutor_confidence`, no crash, R6/R8 flagged (Task 6).
5. **LLM misbehaves:** invalid JSON, hallucinated quote, timeout, Ollama down → retry once, drop unverifiable evidence, fall back to rule detectors, `degraded=true` (Task 10).
Also pinned: resumed/duplicate/out-of-order upload chunks, path-traversal filenames, brute-force of the access code (Tasks 2, 14).

## File Structure
```
GEMINI.md  config.yaml  .env.example  pyproject.toml
assets/8x8_logo_left8.png                    # logo template (already in repo)
classeval/
  schema.py config.py maintenance.py worker.py
  server/   main.py auth.py uploads.py jobs.py static/{index.html,app.js,style.css}
  pipeline/ ingest.py asr.py diarize.py audio_features.py screen_state.py video_features.py
            transcript_features.py llm.py checklist.py scoring.py report.py orchestrator.py
  eval/     vtt.py metrics.py thresholds.py
scripts/    bootstrap.ps1 gpu_smoke.py run.ps1 funnel.ps1 smoke_public.py run_pipeline.py
            asr_bench.py diar_check.py eval_loo.py make_degraded.py release_scan.py
tests/      conftest.py  fixtures/{clips,speech,frames}.py  + test_*.py (server/, pipeline/, eval/)   # one fixture module per owner task → parallel waves never collide
docs/       planning/ superpowers/ ai-history/ explanation-300w.md
```

## Waves (what the human pastes, and what may run in parallel)
| Wave | Tasks | Parallel-safe with |
|---|---|---|
| A | T00 env + models, T01 contracts/config | — (first) |
| B | T02 server + job store + stub worker, T03 ops scripts + **public walking skeleton** | C, D |
| C | T04 ingest, T05 ASR, T06 diarization | B, D |
| D | T07 audio features, T08 screen state + video features, T09 rule detectors | B, C |
| E | T10 LLM + checklist, T11 scoring + report | needs A–D |
| F | T12 orchestrator + real pipeline in worker (then **batch-run the ten labelled recordings in the background**) | — |
| G | T13 eval harness + calibration + metamorphic tests | after F |
| H | T14 security/ops hardening, T15 README + release | after F |

Parallel agents must edit disjoint files.

**Prompt template** (replace `{TASKS}`):
> Workspace is `F:\Eval_class\video_class_eval` only. Read `GEMINI.md`, then `docs/superpowers/plans/2026-10-22-class-quality-assessor.md`: "Global Constraints", "Review Focus" and {TASKS}. Execute {TASKS} in order, following each step literally (test → see it fail → implement → pass → report). Do not stage, commit or tag anything; skip every "Commit" step and list the changed files in the REPORT. Obey the data firewall. If an interface owned by another task must change, stop and report. Finish with the REPORT block from `GEMINI.md`.

---

### Task 0: Environment and model bootstrap

**Files:** Create `pyproject.toml`, `config.yaml` (model names/paths only for now), `.env.example`, `scripts/bootstrap.ps1`, `scripts/gpu_smoke.py`, `tests/test_env.py`, `tests/conftest.py`, `tests/fixtures/__init__.py`, `tests/fixtures/clips.py`. Modify none.

**Interfaces:** Produces: `tests.fixtures.clips.make_clip(path: Path, seconds: float, with_audio: bool = True, with_video: bool = True, size: tuple[int, int] = (1280, 720)) -> Path` (ffmpeg `testsrc2` + `sine`; argument-list subprocess); `config.yaml` keys `models.whisper` (`"large-v3-turbo"`), `models.whisper_dir` (`F:\models\whisper`), `models.clip` (`"openai/clip-vit-base-patch32"`), `models.ecapa` (`"speechbrain/spkrec-ecapa-voxceleb"`), `models.pyannote` (`"pyannote/speaker-diarization-3.1"`), `models.llm` (exact Ollama tag chosen here), `ollama.base_url` (`http://127.0.0.1:11434`); env vars `ACCESS_CODE`, `HF_TOKEN`, `RECORDINGS_DIR`; pytest markers `gpu`, `ollama`, `e2e`.

- [ ] **Step 1: Baseline (made by the planner and the human — skip).** Only confirm that `git status` shows nothing under `data/`, then continue with Step 2.
- [ ] **Step 2: Write `tests/test_env.py`** (CPU-only): `test_python_312` (`sys.version_info[:2] == (3, 12)`); `test_ffmpeg_and_ffprobe_on_path` (both run `-version`, exit 0); `test_imports` (`torch, faster_whisper, cv2, librosa, pyloudnorm, parselmouth, transformers, fastapi, httpx, pydantic, yaml, rapidfuzz, speechbrain` import without error); `test_config_has_model_names` (yaml loads; the 6 keys above exist and are non-empty strings); `test_markers_registered` (`pytest --markers` output contains `gpu`, `ollama`, `e2e`); `test_make_clip_is_probeable` (a 2 s clip exists and `ffprobe` reports duration 2.0 ± 0.2 s). Run → FAIL.
- [ ] **Step 3: Write `pyproject.toml`**: `requires-python = ">=3.12,<3.13"`; `package = false`; torch/torchaudio **2.6.0** from the explicit uv index `https://download.pytorch.org/whl/cu124`; deps: faster-whisper>=1.1, pyannote.audio==3.3.2, speechbrain>=1.0, transformers>=4.46, opencv-python-headless>=4.10, numpy, scipy, scikit-learn, librosa, soundfile, pyloudnorm, praat-parselmouth, rapidfuzz, pillow, fastapi>=0.115, uvicorn[standard], python-multipart, sse-starlette, httpx, pydantic>=2.8, pyyaml, python-dotenv, huggingface_hub, nvidia-ml-py; dev: pytest, pytest-asyncio, ruff. `[tool.pytest.ini_options]`: `pythonpath=["."]`, markers `gpu`, `ollama`, `e2e`. Resolve version conflicts yourself; list final pinned versions in the REPORT.
- [ ] **Step 4: Write `scripts/bootstrap.ps1`** (idempotent, prints progress, never prints secrets): (a) ensure `ffmpeg`+`ffprobe` (try `winget install --id Gyan.FFmpeg -e`; fallback unzip a release build into `F:\tools\ffmpeg\bin` and prepend to the user PATH for the session); (b) `uv sync`; (c) if `.env` is missing create it from `.env.example`; if it exists (the human may have created it with `HF_TOKEN=...`) **append only the missing keys and never touch existing values**; a missing `ACCESS_CODE` gets a random 10-char value; never echo any value; (d) download `Systran/faster-whisper-large-v3` **and** the turbo model (`large-v3-turbo`; the HF id now redirects to `dropbox-dash/faster-whisper-large-v3-turbo`, MIT, CTranslate2 binaries — log the resolved repo id and total size) to `F:\models\whisper`; CLIP, ECAPA and (when accessible) the pyannote repos into the **default Hugging Face cache** (no `local_dir`, so they also load with `HF_HUB_OFFLINE=1`); (e) if `HF_TOKEN` is set, try `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` (gated: on 401/403 print "accept terms at huggingface.co/<repo>" and continue); (f) `ollama pull qwen3-vl:8b`; if the tag does not exist, `ollama pull qwen2.5vl:7b`; write the tag that worked into `config.yaml` `models.llm`.
- [ ] **Step 5: Write `scripts/gpu_smoke.py`**: prints one JSON object: `torch`, `cuda` (bool), `vram_free_mb`, `ffmpeg`, `whisper_cuda_load` (`"ok"`/error; load turbo with `device="cuda", compute_type="int8_float16"`, transcribe 3 s of a generated tone, free it), `clip` (`"ok"`: zero-shot a generated checkerboard image on cuda), `ollama` (`{"model":..., "json_roundtrip":"ok"}`: `/api/chat` with `format` schema and one generated 256×256 PNG; then `keep_alive=0`), `pyannote` (`"ok"` or `"skipped: <reason>"`), `ecapa` (`"ok"`). Known Windows gotcha: CTranslate2 needs cuDNN 9/cuBLAS DLLs — add `torch\lib` with `os.add_dll_directory`. No recordings involved.
- [ ] **Step 6: Run** `uv run pytest -q tests/test_env.py` → all PASS; `uv run python scripts/gpu_smoke.py` → `cuda:true`, whisper/clip/ollama/ecapa `ok`; pyannote `ok`, or `skipped: <reason>` only when `HF_TOKEN` is unset — a set token that fails must report `error: <reason>`; the same run must also pass with `HF_HUB_OFFLINE=1` set.
- [ ] **Step 7: Commit** (`feat: environment, model bootstrap and gpu smoke test`), tag `task-T0`. REPORT includes the smoke JSON (it has no private data) and installed versions.

**Human prerequisites (before pasting Wave A):** accept terms at huggingface.co/pyannote/speaker-diarization-3.1 and /segmentation-3.0; create a **read-only** token and put it in `.env` as `HF_TOKEN=` yourself (never in chat); Ollama app running; Antigravity workspace = the repo folder only.

---

### Task 1: Contracts and configuration

**Files:** Create `classeval/__init__.py`, `classeval/schema.py`, `classeval/config.py`, extend `config.yaml`, Test `tests/test_schema.py`, `tests/test_config.py`.

**Interfaces — Produces** (every later task imports these exact names):
```python
# classeval/schema.py  (pydantic v2)
Grade = Literal["Poor","Average","Good","Excellent"]
FinalGrade = Literal["Poor","Average","Good","Excellent","Insufficient evidence"]
DimId = Literal["R1","R2","R3","R4","R5","R6","R7","R8","R9","R10","R11","R12"]
ScreenState = Literal["BRANDED_TALKING_HEAD","OTHER_TALKING_HEAD","CAMERA_OFF_CARD",
                      "BOARD_SHARE","LESSON_VIDEO_SHARE","OTHER_SHARE","UNKNOWN"]
STAGES = ("queued","ingest","asr","diarize","audio","video","understand","score","report","done")

class MediaInfo: duration_s: float; has_audio: bool; has_video: bool; width: int|None; height: int|None
                 fps: float|None; video_kbps: float|None; container: str; size_bytes: int
class FrameRef: t: float; path: Path
class Word: start: float; end: float; text: str; prob: float
class Segment: start: float; end: float; text: str; speaker: str|None=None; avg_logprob: float|None=None; words: list[Word]=[]
class Transcript: language: str; language_confidence: float; asr_model: str; segments: list[Segment]
class SpeakerTurn: start: float; end: float; speaker: str
class Diarization: turns: list[SpeakerTurn]; tutor: str|None; tutor_confidence: float
                   method: Literal["pyannote","ecapa","none"]; n_speakers: int
class FrameState: t: float; state: ScreenState; confidence: float
class Evidence: t_start: float; t_end: float|None=None; kind: Literal["quote","measure","visual"]; text: str
                # validator: kind=="quote" -> at most 15 words
class Observation: text: str; dimension: DimId; evidence: list[Evidence]
class Detection: assessed: bool; score: float|None; hits: list[Evidence]
class AudioFacts: duration_s; speech_ratio; tutor_talk_share; student_words_per_min; dead_air_per_min;
                  median_response_latency_s: float|None; tutor_speech_rate_wpm; loudness_lufs; clipping_ratio;
                  snr_db_est; pitch_std_semitones_tutor: float|None; energy_cv_tutor: float|None;
                  dropouts: list[tuple[float,float]]
class VideoFacts: states: list[FrameState]; tutor_name: str|None; branded_ratio_tutor_speaking: float|None
                  board_share_ratio; lesson_video_ratio; camera_off_total_ratio;
                  camera_off_while_tutor_speaks_s: float; first5_branded: bool|None; last5_branded: bool|None
                  talking_head_luma_mean: float|None; n_frames: int
class TranscriptFacts: detections: dict[str, Detection]   # keys: greeting_opening, rapport, self_intro_new_joiners,
                       # individual_interaction, homework, closing, chess_engagement, praise_density
                       distinct_responders: int; question_reply_rate: float|None; question_count: int
class DimensionScore: id: DimId; name: str; group: Literal["teaching","presentation"]; score: float|None  # None = not assessed
                      weight: float; confidence: float; evidence: list[Evidence]; sources: list[str]; note: str|None=None
class Report: job_id: str; created_at: str; media: MediaInfo; language: str; language_confidence: float
              grade: FinalGrade; overall_score: float|None; confidence: float; coverage: float; summary: str
              dimensions: list[DimensionScore]; strengths: list[Observation]; improvements: list[Observation]
              technical_flags: list[str]; not_assessed: list[str]; downweighted_signals: list[str]
              degraded: bool; timings_s: dict[str,float]; versions: dict[str,str]
# classeval/config.py
def load_settings(path: Path | str = "config.yaml") -> Settings   # Settings is a pydantic model mirroring config.yaml; env overrides: ACCESS_CODE (required, no default), HF_TOKEN, RECORDINGS_DIR
```
`config.yaml` gains: `weights` (the 12 values), `groups`, `cutoffs` (`poor_lt: 45, average_lt: 62, good_lt: 78`), `limits` (`max_upload_bytes: 2684354560, max_duration_s: 10800, min_duration_s: 600, chunk_bytes: 8388608, job_ttl_hours: 24`), `sampling` (`frame_fps: 0.2, frame_width: 640, keyframes: 24`), `lexicon` (greeting, rapport, self-intro, homework, closing word lists in English), `scoring` (normalisation ranges from Task 11's table), `paths` (`jobs_dir: jobs, uploads_dir: uploads`).

- [ ] **Step 1: Write failing tests.** `test_report_roundtrip` (minimal Report → `model_dump_json` → `model_validate_json` equals original); `test_grade_literals_reject_unknown` (`grade="Great"` → `ValidationError`); `test_quote_over_15_words_rejected` and `test_measure_may_exceed_15_words`; `test_dimension_score_none_means_not_assessed`; `test_weights_sum_100_and_ids_r1_r12` (from loaded config); `test_groups_split_75_25`; `test_access_code_required` (unset env → clear error); `test_env_override` (`ACCESS_CODE=abc` visible in settings; secrets never appear in `repr(settings)`).
- [ ] **Step 2: Run** `uv run pytest tests/test_schema.py tests/test_config.py -q` → FAIL.
- [ ] **Step 3: Implement** `schema.py` exactly as above and `config.py` (`SecretStr` for `access_code`/`hf_token`).
- [ ] **Step 4: Run** the same command + `uv run ruff check .` → PASS.
- [ ] **Step 5: Commit** `feat: report/pipeline contracts and settings`, tag `task-T1`.

---

### Task 2: Server, upload store, job store, stub worker, UI shell

**Files:** Create `classeval/server/{main,auth,uploads,jobs}.py`, `classeval/server/static/{index.html,app.js,style.css}`, `classeval/worker.py`, `classeval/pipeline/orchestrator.py` (stub), Test `tests/server/test_auth.py`, `test_uploads.py`, `test_jobs.py`, `test_static.py`.

**Interfaces — Produces:**
```python
def create_app(settings: Settings) -> FastAPI
class Job: id: str; status: Literal["queued","running","done","failed"]; stage: str; progress: float; created_at: str
           source: dict; error: dict|None; eta_s: float|None
class JobStore:
    def __init__(self, root: Path)
    def create(self, source: dict) -> Job            # {"type":"upload","upload_id":...}  (upload only; no link/URL input — the brief says upload or link)
    def get(self, job_id: str) -> Job | None
    def update(self, job_id: str, **fields) -> Job
    def set_stage(self, job_id: str, stage: str, progress: float) -> Job
    def claim_next(self) -> Job | None               # atomically queued -> running
    def delete(self, job_id: str) -> bool
    def job_dir(self, job_id: str) -> Path
class UploadStore:
    def __init__(self, root: Path, max_bytes: int, chunk_bytes: int)
    def init(self, size: int, filename: str) -> str                 # raises UploadTooLarge
    def put_chunk(self, upload_id: str, index: int, data: bytes) -> None   # idempotent, any order
    def missing(self, upload_id: str) -> list[int]
    def complete(self, upload_id: str) -> Path                       # raises MissingChunks(list)
def run(job_id: str, store: JobStore, settings: Settings, stages=None) -> None   # orchestrator; STUB now: walks STAGES with progress, writes a schema-valid Report with grade "Insufficient evidence"
```
Routes (all except `/healthz` need `X-Access-Code`): `POST /api/uploads {size, filename}`; `PUT /api/uploads/{id}/chunks/{n}`; `POST /api/uploads/{id}/complete`; `POST /api/jobs {upload_id}`; `GET /api/jobs/{id}`; `GET /api/jobs/{id}/events` (SSE, one event per stage change); `GET /api/jobs/{id}/report.json|report.html`; `DELETE /api/jobs/{id}`. There is **no** job-listing route.

- [ ] **Step 1: Write failing tests.** `test_missing_or_wrong_code_401_on_all_api_routes`; `test_healthz_open`; `test_upload_roundtrip_out_of_order_and_duplicate_chunks` (20 MB random bytes in 8 MiB chunks, order 2,0,1,1; completed file sha256 equals source); `test_complete_with_missing_chunk_409_lists_indices`; `test_init_over_limit_413`; `test_filename_path_traversal_ignored` (`"..\\..\\x.mp4"` never leaves the uploads dir; stored under the id); `test_job_stub_flow_stages_in_order` (create job → `run` → SSE emits every stage of `STAGES` in order; `report.json` validates as `Report`); `test_job_ids_unguessable` (len ≥ 22, `GET /api/jobs` → 404/405); `test_delete_removes_job_dir`; `test_rate_limit_429` (61st request in a minute from one client); `test_static_has_no_external_urls` (`index.html/app.js/style.css` contain no `http://` or `https://`).
- [ ] **Step 2: Run** `uv run pytest tests/server -q` → FAIL.
- [ ] **Step 3: Implement** the interfaces. Job state = one `job.json` per job dir, written atomically (temp file + `os.replace`); no database. `claim_next` takes a file lock so two workers never claim one job. UI: single page — access-code field, drop zone/file picker, chunked upload with progress bar (8 MiB chunks, 3 retries with backoff, resume by asking `missing`), stage list with ETA while polling SSE, result view rendering `report.html` from the API. No external requests.
- [ ] **Step 4: Run** tests + `ruff` → PASS. Manually: `uv run uvicorn classeval.server.main:app --factory --port 8765` + `uv run python -m classeval.worker`, upload a small generated mp4 in the browser, see the stub report.
- [ ] **Step 5: Commit** `feat: upload/job API, stub worker, UI shell`, tag `task-T2`.

---

### Task 3: Ops scripts and the public walking skeleton

**Files:** Create `scripts/run.ps1`, `scripts/funnel.ps1`, `scripts/smoke_public.py`, Test `tests/test_smoke_client.py`.

**Interfaces — Produces:** `scripts/smoke_public.py <base_url> <access_code>` → generates a 30 s synthetic clip (via `tests.fixtures.clips.make_clip` from Task 0), uploads it in chunks, creates a job, waits for `done`, validates the Report, prints `OK <seconds>` or `FAIL <reason>`.

- [ ] **Step 1: Write `tests/test_smoke_client.py`**: starts the app in-process via `httpx.ASGITransport`, runs the smoke client logic against it → `OK`. Run → FAIL.
- [ ] **Step 2: Implement** `smoke_public.py` (importable `run_smoke(base_url, code, transport=None)`), `scripts/run.ps1` (activates env, sets `HF_HOME=F:\models\hf` and `HF_HUB_OFFLINE=1`, starts uvicorn on **127.0.0.1:8765** and the worker, restarts either on crash with a 3 s back-off, logs to `logs\` which is ignored), `scripts/funnel.ps1` (`tailscale funnel --bg 8765`, then prints `tailscale funnel status`; with `-Reset` runs `tailscale funnel reset`; with `-DisableSleep` runs `powercfg /change standby-timeout-ac 0` and `monitor-timeout-ac 0` after printing what it will do).
- [ ] **Step 3: Run** tests → PASS; run `scripts/run.ps1`, then `uv run python scripts/smoke_public.py http://127.0.0.1:8765 <code>` → `OK`.
- [ ] **Step 4: Commit** `feat: run/funnel scripts and public smoke client`, tag `task-T3`.
- [ ] **Step 5 (human checkpoint):** open the one-time link `tailscale funnel` prints to enable Funnel for the tailnet, run `scripts/funnel.ps1`, then from a **phone on mobile data** open the public HTTPS URL, enter the code, upload a small video, see the stub report. Paste the result (OK/FAIL, not the URL or code) to the planner.

---

### Task 4: Ingest

**Files:** Create `classeval/pipeline/ingest.py`, Test `tests/pipeline/test_ingest.py` (uses `tests.fixtures.clips.make_clip`; may pass extra ffmpeg flags through a `extra_args` keyword you add to `make_clip` — keep it backwards compatible).

**Interfaces — Produces:**
```python
class IngestError(Exception): code: Literal["unreadable","no_streams","too_long","too_short_zero","timeout"]; message: str
class Prepared: info: MediaInfo; wav: Path | None; frames: list[FrameRef]
def probe(path: Path) -> MediaInfo
def validate(info: MediaInfo, settings: Settings) -> list[str]     # returns warnings (e.g. "no_video", "no_audio", "short"); raises IngestError for hard failures
def prepare(src: Path, job_dir: Path, settings: Settings) -> Prepared
```
`prepare` timeouts = `0.5 * duration + 60` s per ffmpeg call; frames via `fps=0.2,scale=640:-2`, JPEG `-q:v 3`, `FrameRef.t = index / 0.2`.

- [ ] **Step 1: Write failing tests** (clips from `make_clip`): `test_probe_duration_within_0_1s`; `test_audio_only_has_no_frames` (`has_video False`, `frames == []`, warning `no_video`); `test_video_only_has_no_wav` (`wav is None`, warning `no_audio`); `test_frame_count_matches_fps` (60 s → 12 frames ±1); `test_wav_is_16k_mono_s16` (read header); `test_corrupt_file_raises_unreadable` (random bytes with `.mp4`); `test_too_long_by_mocked_probe` (duration 14 400 → `too_long`); `test_ffmpeg_timeout_maps_to_timeout_error` (monkeypatch command to a sleeper); `test_no_shell_true` (source of `ingest.py` does not contain `shell=True`); `test_variable_frame_rate_clip_ok` (clip made with `-vsync vfr`).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** per interfaces. **Step 4: Run** → PASS. **Step 5: Commit** `feat: ingest (probe/validate/prepare)`, tag `task-T4`.

---

### Task 5: ASR

**Files:** Create `classeval/pipeline/asr.py`, `classeval/eval/vtt.py`, `scripts/asr_bench.py`, `tests/fixtures/speech.py` (`make_speech_wav(text: str, path: Path, voice: str = "David") -> Path` using PowerShell `System.Speech`; also `concat_wavs(paths: list[Path], gap_s: float, out: Path) -> Path`), Test `tests/pipeline/test_asr.py`, `tests/eval/test_vtt.py`.

**Interfaces — Produces:**
```python
def transcribe(wav: Path, settings: Settings, progress: Callable[[float], None] | None = None) -> Transcript
def detect_language(wav: Path, settings: Settings) -> tuple[str, float]     # 5 windows x 30 s spread over the file; (majority language, mean prob)
# classeval/eval/vtt.py
def parse_vtt(path: Path) -> list[tuple[float, float, str | None, str]]    # start, end, speaker, text
def wer(ref: str, hyp: str) -> float
```
Settings: `faster_whisper` model from `models.whisper`, `device="cuda"`, `compute_type="int8_float16"`, `BatchedInferencePipeline(batch_size=8)`, `vad_filter=True`, `word_timestamps=True`, `condition_on_previous_text=False`. Frees GPU memory before returning.

- [ ] **Step 1: Write failing tests.** `test_vtt_parse_and_wer` (a tiny inline VTT string; `wer("a b c","a x c") == 1/3`); `@gpu test_transcribes_synthetic_speech` (SAPI: "Hello everyone, welcome to today's chess class" → lowercased text contains `welcome` and `chess`; every `Word.start < Word.end`); `@gpu test_silence_gives_no_segments`; `@gpu test_tone_gives_at_most_3_words`; `@gpu test_language_detect_english` (`("en", p)` with `p > 0.6`); `@gpu test_vram_released` (`nvidia-smi` used MB after the call ≤ before + 400).
- [ ] **Step 2: Run** `uv run pytest tests/pipeline/test_asr.py tests/eval -q -m "gpu or not gpu"` → FAIL. **Step 3: Implement.** `scripts/asr_bench.py --input <mp4> [--vtt <vtt>]` prints **numbers only**: audio minutes, wall seconds, real-time factor, n_segments, mean `avg_logprob`, language, language confidence, and (with `--vtt`) WER and |our tutor-share − VTT tutor-share| — never any text. **Step 4: Run** → PASS. **Step 5: Commit** `feat: faster-whisper ASR, language id, vtt utils`, tag `task-T5`.

---

### Task 6: Diarization and tutor identification

**Files:** Create `classeval/pipeline/diarize.py`, `scripts/diar_check.py`, Test `tests/pipeline/test_diarize.py`.

**Interfaces — Produces:**
```python
def diarize(wav: Path, settings: Settings) -> Diarization          # ECAPA embeddings + VAD + agglomerative clustering only; method == "ecapa"
def identify_tutor(turns: list[SpeakerTurn]) -> tuple[str | None, float]     # (label, confidence)
def assign_speakers(tr: Transcript, d: Diarization) -> Transcript            # each segment gets the speaker with max time overlap
```
`tutor_confidence = min(1.0, ((t1 - t2) / t1) / 0.15)` where `t1,t2` are the two largest speaker talk times; **one speaker → tutor = that speaker, confidence 0.5**; no speech → `(None, 0.0)`. Both back-ends must load with `HF_HUB_OFFLINE=1` (their models come from the Hub cache; pyannote's pipeline config points at its segmentation and embedding repos, so those must be cached too). **Known blocker found in review:** `pyannote.audio` 3.3.2 passes `use_auth_token=` to `hf_hub_download`, which `huggingface_hub` 2.2.0 removed, so `Pipeline.from_pretrained("<repo id>")` raises `TypeError` online and offline. **Decision (cut for time): do not use pyannote at all; ECAPA is the only back-end and `method` is always `"ecapa"`.** Unused idea, do NOT build: resolve the three cached snapshots with `snapshot_download(repo, local_files_only=True)`, write a patched `config.yaml` into `.cache/pyannote/` whose segmentation and embedding entries are absolute paths to the cached `pytorch_model.bin` files, and load with `Pipeline.from_pretrained(<path to that config.yaml>)` (a local path makes pyannote skip the hub). If that does not work inside the time-box, keep ECAPA as the only back-end, record `method="ecapa"`, and say so in the REPORT.

- [ ] **Step 1: Write failing tests** (SAPI voices David + Zira alternating): `@gpu test_two_voices_found_and_tutor_is_dominant`; `test_identify_tutor_equal_time_low_confidence` (< 0.3); `test_identify_tutor_single_speaker` (confidence == 0.5); `test_identify_tutor_no_turns` (`(None, 0.0)`); `test_assign_speakers_max_overlap` (hand-made segments and turns); `@gpu test_ecapa_fallback_when_no_token` (unset `HF_TOKEN` → `method == "ecapa"`).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** `scripts/diar_check.py --input <mp4> --vtt <vtt>` prints numbers only: our tutor share, VTT tutor share, absolute difference, n_speakers, method. **Step 4: Run** → PASS. **Step 5: Commit** `feat: diarization with ecapa fallback and tutor identification`, tag `task-T6`.

---

### Task 7: Audio features

**Files:** Create `classeval/pipeline/audio_features.py`, Test `tests/pipeline/test_audio_features.py`.

**Interfaces — Produces:** `def compute(wav: Path, tr: Transcript, d: Diarization) -> AudioFacts`. Definitions (pinned): `speech_ratio` = speech seconds / duration; `tutor_talk_share` = tutor speech s / all speech s; `student_words_per_min` = words by non-tutor speakers / minutes; `dead_air_per_min` = count of gaps **> 8 s** between consecutive speech turns / minutes; `median_response_latency_s` = median gap (cap 30 s) from the end of a tutor segment ending in `?` to the next non-tutor segment start (`None` if no such pair); `tutor_speech_rate_wpm`; `loudness_lufs` (pyloudnorm, whole file); `clipping_ratio` = share of samples with `|x| ≥ 0.999`; `snr_db_est` = 20·log10(P90 / P10 of 50 ms-frame RMS); `pitch_std_semitones_tutor` (parselmouth, voiced tutor frames only); `energy_cv_tutor` = std/mean of frame RMS inside tutor speech; `dropouts` = intervals ≥ 2 s with RMS < −70 dBFS.

- [ ] **Step 1: Write failing tests** (synthetic arrays + hand-built Transcript/Diarization): `test_loud_vs_quiet_tone_lufs_ordering`; `test_clipped_square_wave_clipping_ratio_gt_0_1`; `test_dead_air_count` (speech–10 s silence–speech–10 s silence–speech → 2 per (duration/60)); `test_response_latency_median` (known gaps 2 s and 4 s → 3.0); `test_pitch_std_varied_gt_monotone` (glissando vs 150 Hz tone); `test_dropout_detected` (3 s digital silence mid-file); `test_no_student_speech_gives_zero_words_per_min`; `test_none_fields_when_no_tutor`.
- [ ] **Step 2–4:** run (FAIL) → implement → run (PASS). **Step 5: Commit** `feat: audio feature extraction`, tag `task-T7`.

---

### Task 8: Screen-state timeline and video features

**Files:** Create `classeval/pipeline/screen_state.py`, `classeval/pipeline/video_features.py`, `tests/fixtures/frames.py` (`draw_frame(kind: Literal["camera_off_card","branded_head","board_grid","plain_dark","noise"], path: Path) -> Path` with PIL/OpenCV), Test `tests/pipeline/test_screen_state.py`, `test_video_features.py`.

**Interfaces — Produces:**
```python
def classify_frames(frames: list[FrameRef], settings: Settings) -> list[FrameState]
#  order: (1) flat-grey card: mean luma 20–40 and std < 8 over >= 92 % of pixels -> CAMERA_OFF_CARD
#         (2) assets/8x8_logo_left8.png multi-scale cv2.matchTemplate (0.35x-1.6x, score >= 0.6) -> BRANDED_TALKING_HEAD
#         (3) CLIP zero-shot (prompts in config.yaml) -> BOARD_SHARE | LESSON_VIDEO_SHARE | OTHER_TALKING_HEAD | OTHER_SHARE
#         (4) else UNKNOWN. Run CLIP in batches of 32 on cuda; free VRAM afterwards.
def compute(states: list[FrameState], d: Diarization, tr: Transcript, settings: Settings) -> VideoFacts
# CUT for time: no tutor-name reading, no VLM here. VideoFacts.tutor_name stays None.
```
`compute` rules: ratios are over all frames (`board_share_ratio`, `lesson_video_ratio`, `camera_off_total_ratio`); `branded_ratio_tutor_speaking` = branded / (branded + other-talking-head) among frames whose time falls inside a tutor turn (`None` if fewer than 3 such frames); `camera_off_while_tutor_speaks_s` = frames in `CAMERA_OFF_CARD` during tutor turns × frame period; `first5_branded`/`last5_branded` = branded share of talking-head frames in those windows ≥ 0.5 (`None` if no talking-head frames); `talking_head_luma_mean` over talking-head frames only. Camera-off in a **thumbnail** is out of scope (documented limitation).

- [ ] **Step 1: Write failing tests.** `test_flat_grey_card_is_camera_off`; `test_logo_pasted_on_navy_is_branded` (template pasted at two scales); `@gpu test_checkerboard_is_board_share`; `test_unknown_for_noise`; `test_compute_ratios_exact` (hand-made 20 states, known turns → exact ratios); `test_camera_off_seconds_only_when_tutor_speaks`; `test_branded_ratio_none_when_few_frames`; `test_luma_only_from_talking_head`.
- [ ] **Step 2–4:** run (FAIL) → implement → run (PASS). **Step 5: Commit** `feat: screen-state timeline and video features`, tag `task-T8`. Also emit `scripts/run_pipeline.py` hook later (Task 12) to write `states_to_label.csv` (timestamps + predicted state only) for the human check in Task 13.

---

### Task 9: Rule-based transcript detectors

**Files:** Create `classeval/pipeline/transcript_features.py`, Test `tests/pipeline/test_transcript_features.py`.

**Interfaces — Produces:** `def compute(tr: Transcript, d: Diarization, settings: Settings) -> TranscriptFacts`. Detectors (lexicons from `config.yaml`): `greeting_opening` (tutor utterance in first 4 min matches greeting lexicon); `rapport` (first 5 min, tutor asks about the child/day: "how are you", "how was", "what did you"); `self_intro_new_joiners` ("introduce yourself", "new friend", "welcome to", "tell me your name"); `individual_interaction` (`distinct_responders` = distinct non-tutor speakers who answered a tutor question within 6 s; `question_reply_rate` = answered tutor questions / tutor questions); `homework` (matches in the **last 10 min** count double); `closing` (last 5 min: bye/see you/next class/thank you/well done today); `chess_engagement` (student utterances with squares `\b[a-h][1-8]\b`, piece names, "check", "mate", "move" per minute); `praise_density` (reported, **weight 0** — misleading signal per the brief). **Every detector returns `assessed=False, score=None` when `tr.language_confidence < 0.5` or the median `avg_logprob < -1.0`.** Hits are `Evidence(kind="quote")` ≤ 15 words with names masked.

- [ ] **Step 1: Write failing tests** (hand-built Transcript objects, no audio): one positive and one negative per detector; `test_not_assessed_when_language_uncertain` (all detectors `assessed False`); `test_homework_in_last_10_min_counts_double`; `test_distinct_responders_counts_speakers_not_utterances`; `test_quotes_capped_15_words_and_names_masked`; `test_praise_density_present_but_weight_zero_flag`; `test_empty_transcript_no_crash`.
- [ ] **Step 2–4:** run (FAIL) → implement → run (PASS). **Step 5: Commit** `feat: rule-based transcript detectors`, tag `task-T9`.

---

### Task 10: LLM client and checklist extraction

**Files:** Create `classeval/pipeline/llm.py`, `classeval/pipeline/checklist.py`, Test `tests/pipeline/test_llm.py`, `test_checklist.py`.

**Interfaces — Produces:**
```python
class LLMUnavailable(Exception); class LLMBadOutput(Exception)
class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 180, transport=None)
    def chat_json(self, system: str, user: str, schema: dict, images: list[bytes] | None = None) -> dict
        # POST /api/chat, stream=False, format=schema, options={temperature:0, seed:7, num_ctx:16384}; one retry on invalid JSON
    def unload(self) -> None                         # keep_alive=0
class ChecklistItem: dimension: DimId; verdict: Literal["met","partial","not_met","not_observed"]; evidence: list[Evidence]
class ChecklistResult: items: dict[DimId, ChecklistItem]; strengths: list[Observation]; improvements: list[Observation]; degraded: bool
def chunk_transcript(tr: Transcript, d: Diarization, minutes: float = 8.0) -> list[tuple[float, float, str]]   # (t0, t1, "TUTOR: ...\nSTUDENT 1: ...")
def verify_evidence(item: ChecklistItem, tr: Transcript) -> ChecklistItem   # drop quotes with rapidfuzz partial_ratio < 85 within t±30 s; mask names; cap 15 words
def extract(tr: Transcript, d: Diarization, tfacts: TranscriptFacts, llm: OllamaClient | None, settings: Settings) -> ChecklistResult
```
Map step per chunk: system prompt states the domain (online chess lessons for children), the rubric text for R4,R5,R6,R7,R9,R10,R11, and demands JSON `{items:[{dimension,verdict,evidence:[{t,quote}]}],strengths:[...],improvements:[...]}`. Reduce rule: R4 uses only chunks starting in the first 5 min, R11 only the last chunk, R10 only the last two; any `met` with **verified** evidence wins, else `partial`, else `not_met`, else `not_observed`. `extract` with `llm=None` or `LLMUnavailable` builds items from `tfacts` only and sets `degraded=True`.

- [ ] **Step 1: Write failing tests** (fake Ollama via `httpx.MockTransport`): `test_chat_json_sends_schema_and_zero_temperature`; `test_invalid_json_twice_raises_bad_output`; `test_timeout_raises_unavailable`; `test_chunking_60min_gives_7_to_8_chunks_each_under_12k_tokens_estimate`; `test_prompt_is_deterministic` (same input → identical prompt hash); `test_hallucinated_quote_dropped`; `test_names_masked_and_quote_truncated_to_15_words`; `test_reduce_rule_homework_ignores_early_chunks`; `test_fallback_to_rules_sets_degraded`; `test_not_observed_when_transcript_language_uncertain`; `@ollama test_live_roundtrip_schema_valid` (tiny synthetic transcript, real model).
- [ ] **Step 2–4:** run (FAIL) → implement → run (PASS). **Step 5: Commit** `feat: ollama client and checklist extraction`, tag `task-T10`.

---

### Task 11: Scoring and report

**Files:** Create `classeval/pipeline/scoring.py`, `classeval/pipeline/report.py`, Test `tests/pipeline/test_scoring.py`, `test_report.py`.

**Interfaces — Produces:**
```python
class ScoreResult: dimensions: list[DimensionScore]; overall_score: float | None; coverage: float
                   grade: FinalGrade; confidence: float; flags: list[str]
def score(media: MediaInfo, tr: Transcript, d: Diarization, audio: AudioFacts, video: VideoFacts,
          tfacts: TranscriptFacts, checklist: ChecklistResult, settings: Settings) -> ScoreResult
def build_report(job_id: str, media: MediaInfo, tr: Transcript, d: Diarization, audio: AudioFacts, video: VideoFacts,
                 tfacts: TranscriptFacts, checklist: ChecklistResult, result: ScoreResult,
                 timings_s: dict[str, float], versions: dict[str, str], summary: str) -> Report
def summarize(result: ScoreResult, checklist: ChecklistResult) -> str   # deterministic template over scores and verified evidence; NO LLM call (cut for time)
def render_html(report: Report) -> str    # single self-contained page: grade banner, summary, dimension table with bars, strengths/improvements with [mm:ss] chips, flags, not-assessed, down-weighted signals; no external URLs
```
Priors (all in `config.yaml → scoring`, tunable): `norm(x,lo,hi)=clamp((x-lo)/(hi-lo),0,1)` (inverse when `lo>hi`); verdict map `met=100, partial=60, not_met=10, not_observed=None`.
- R1 = `100*norm(branded_ratio_tutor_speaking,0,0.7)` (None → not assessed) · R2 = `100*(1-min(1, camera_off_while_tutor_speaks_s / max(1, 0.25*tutor_speech_s)))`, ×0.8 if `talking_head_luma_mean<60` · R3 = 100 − 20 (LUFS outside −30…−14) − 20 (clipping>0.01) − 20 (SNR<15 dB) − 10 per dropout (cap 40)
- R4, R5, R10 = verdict · R11 = verdict (+10 if `last5_branded`) · R6 = `0.6*verdict + 0.4*100*norm(question_reply_rate,0.2,0.8)` · R7 = `0.5*verdict + 0.25*100*norm(pitch_std,1.5,4.0) + 0.25*100*norm(energy_cv,0.4,1.0)`
- R8 = `100*(0.5*norm(student_words_per_min,5,45) + 0.25*norm(dead_air_per_min,0.8,0.05) + 0.25*norm(distinct_responders,0,5))` · R9 = `0.4*100*norm(board_share_ratio,0.15,0.6) + 0.2*(100 if lesson_video_ratio>0.02 else 0) + 0.4*verdict` · R12 = 100 − 15 per dropout − 30 if `camera_off_total_ratio>0.5` (floor 0).
- Overall = weighted mean over **assessed** dimensions (weights renormalised); `coverage` = assessed weight / 100.
- Gates: `camera_off_while_tutor_speaks_s / tutor_speech_s ≥ 0.5` → grade capped at "Average"; `duration < 600` or `speech_ratio < 0.05` or `coverage < 0.5` → "Insufficient evidence". Confidence = `0.4*coverage + 0.2*asr_conf + 0.2*tutor_conf + 0.2*(1.0 if not degraded else 0.4)`; language banner flag when `language_confidence < 0.5`.

- [ ] **Step 1: Write failing tests.** `test_all_good_facts_grade_excellent_over_85`; `test_all_bad_facts_grade_poor`; `test_camera_off_60pct_caps_at_average`; `test_not_assessed_excluded_and_weights_renormalised` (`coverage` exact); `test_silent_recording_insufficient_evidence` (speech_ratio 0.01); `test_short_recording_insufficient_evidence` (duration 300); `test_non_english_marks_text_dimensions_not_assessed_not_zero`; `test_praise_density_has_zero_weight`; `test_confidence_penalised_when_degraded`; `test_low_tutor_confidence_adds_flag_and_lowers_confidence` (tutor_confidence 0.2 → flag `tutor_unclear`, confidence below the 1.0-case by exactly `0.2*(1-0.2)`); `test_report_validates_and_html_has_grade_and_all_dimension_ids`; `test_html_has_no_external_urls`; `test_summary_names_grade_best_and_weakest_dimension`; `test_report_strengths_improvements_each_have_evidence`.
- [ ] **Step 2–4:** run (FAIL) → implement → run (PASS). **Step 5: Commit** `feat: rubric scoring and report builder`, tag `task-T11`.

---

### Task 12: Orchestrator and real pipeline in the worker

**Files:** Modify `classeval/pipeline/orchestrator.py` (replace stub), `classeval/worker.py`; Create `scripts/run_pipeline.py`, Test `tests/pipeline/test_orchestrator.py`, `tests/test_e2e_synthetic.py`.

**Interfaces — Consumes:** Tasks 4–11 functions. **Produces:**
```python
@dataclass class StageFns: ingest; asr; diarize; audio; video; understand; score; report   # injectable callables, defaults = real functions
def run(job_id: str, store: JobStore, settings: Settings, stages: StageFns | None = None) -> None
# scripts/run_pipeline.py --input <mp4> --out <dir> [--fast]   -> writes stage caches + report.json/html under <dir>; prints one numbers-only line
```
Behaviour: stage markers `work/stages/<name>.done.json` (resume skips done stages); ASR and diarization run concurrently, CV/CLIP frame work runs in a thread pool alongside them; GPU memory is freed between stages and Ollama `unload()` after `understand`; per-stage timeouts; on failure → `status="failed"` with `error={"code","message"}` (never a traceback); `Prepared.wav is None` → skip audio stages and flag `no_audio`; `IngestError` → friendly failure; progress per stage; wall-time `timings_s` recorded; the upload file is deleted after `report.json` is written.

- [ ] **Step 1: Write failing tests** (all stages injected as fakes): `test_stage_order_and_progress_monotonic`; `test_failure_in_diarize_marks_job_failed_with_code_no_traceback`; `test_resume_skips_completed_stages`; `test_gpu_cleanup_called_between_gpu_stages`; `test_no_audio_flagged_and_pipeline_continues`; `test_input_deleted_after_success`; `@e2e test_synthetic_class_end_to_end` (3-min generated "class": two SAPI voices alternating, drawn frames cycling card/branded/board; real models) → valid `Report`, `grade` in the 5 allowed strings, wall time < 8 min, VRAM back to baseline.
- [ ] **Step 2–4:** run (FAIL) → implement → run (PASS; e2e passes). **Step 5: Commit** `feat: real pipeline orchestration`, tag `task-T12`.
- [ ] **Step 6 (human):** run `scripts/run_pipeline.py` over the **ten labelled** recordings only (skip the two demos) in the background (`--recordings-dir`, ignored cache `eval_runs/`) — it prints one numbers-only line per recording. It is a long GPU job; develop other tasks meanwhile.

---

### Task 13: Evaluation harness, calibration, metamorphic tests

**Files:** Create `classeval/eval/metrics.py`, `classeval/eval/thresholds.py`, `scripts/eval_loo.py`, `scripts/make_degraded.py`, Test `tests/eval/test_metrics.py`, `test_thresholds.py`, `tests/test_metamorphic.py`.

**Interfaces — Produces:**
```python
def exact_and_within_one(y_true: list[int], y_pred: list[int]) -> tuple[float, float]
def kendall_tau(a: list[float], b: list[float]) -> float
def confusion(y_true, y_pred, k=4) -> list[list[int]]
def fit_cutoffs(scores: list[float], labels: list[int], soft_le_average: list[float]) -> tuple[float, float, float]
    # grid 0..100 step 1 over c1<c2<c3; maximise exact matches on hard labels, hard constraint: every soft score maps to <= Average;
    # tie-break: within-one, then lowest total |error|, then the middle of the plateau
def loo_predict(scores, labels, soft) -> list[int]       # leave-one-out predictions for the hard-labelled classes
```
`scripts/eval_loo.py --cache eval_runs --truth data/checklist_truth.yaml` infers each class's label from its **folder** (excellent=3, good=2, average=1, weak=0, `quality issue` = soft ≤ Average, demo = unlabelled) and prints **numbers only**: per-class `id, true, score, pred, confidence`; LOO exact, within-one, Kendall τ, confusion matrix; fitted cut-offs (also written to `config.yaml → cutoffs` only with `--write`); per-dimension agreement with `data/checklist_truth.yaml` where truth exists; ASR sanity (WER vs VTT, tutor-share delta). `scripts/make_degraded.py <in> <outdir> <kind>` with kinds `mute5`, `black`, `trunc5min` (ffmpeg only).

- [ ] **Step 1: Write failing tests.** `test_exact_and_within_one`; `test_kendall_perfect_and_reversed`; `test_fit_cutoffs_separable_data_recovers_boundaries`; `test_fit_cutoffs_respects_soft_constraint`; `test_loo_does_not_peek` (held-out label changed → its own prediction unchanged); `@e2e test_metamorphic_mute_lowers_clarity_and_flags_audio`; `@e2e test_metamorphic_black_video_flags_camera`; `@e2e test_metamorphic_truncated_5min_insufficient_evidence`.
- [ ] **Step 2–4:** run (FAIL) → implement → run (PASS). **Step 5: Commit** `feat: evaluation harness and calibration`, tag `task-T13`.
- [ ] **Step 6 (human + planner):** run `eval_loo.py` on the Task-12 cache; the planner reviews numbers; you label ≈ 40 timestamps from `states_to_label.csv` per recording locally (target ≥ 90 % agreement).

---

### Task 14: Security and ops hardening

**Files:** Create `classeval/maintenance.py`; Modify `classeval/server/{main,auth,uploads}.py`, `scripts/run.ps1`; Test `tests/test_maintenance.py`, `tests/server/test_hardening.py`, `tests/test_no_egress.py`, `tests/test_log_scrub.py`.

**Interfaces — Produces:** `def purge_expired(root: Path, ttl_hours: float, now: datetime) -> list[str]` (job and upload ids removed); lock-out after **10 failed access-code attempts per IP per 5 min** (429 with `Retry-After`); disk guard (reject init when free space < 3× upload size); security headers on every response: `Content-Security-Policy: default-src 'self'`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` on reports.

- [ ] **Step 1: Write failing tests.** `test_purge_removes_only_expired`; `test_lockout_after_10_failures_and_expires`; `test_disk_guard_rejects_when_low_space` (monkeypatched `shutil.disk_usage`); `test_security_headers_present`; `test_report_not_cacheable`; `test_no_egress` (fixture patches `socket.socket.connect` to fail any non-127.0.0.1 address; the synthetic e2e pipeline runs with only fake-Ollama on localhost → no violation); `test_log_scrub` (sentinel phrase "zebra-quartz-17" spoken by SAPI and present in the transcript never appears in any file under `logs/` or captured stdout/stderr); `test_worker_restart_resumes_running_job` (job left `running` by a killed worker is re-queued after a lease timeout).
- [ ] **Step 2–4:** run (FAIL) → implement → run (PASS). **Step 5: Commit** `feat: purge, lockout, headers, egress and log-scrub guards`, tag `task-T14`.

---

### Task 15: README, explanation, AI history, release scan

**Files:** Create `README.md`, `docs/explanation-300w.md`, `scripts/release_scan.py`, Test `tests/test_release.py`.

**Interfaces — Produces:** `scripts/release_scan.py` scans the working tree **and** `git log -p` for: media/transcript/PDF extensions, strings matching key patterns (`AIza…`, `hf_…`, `sk-…`, `ghp_…`, private-key blocks), the values in `.env`, e-mail addresses, and any word from `data/names.txt` (git-ignored; one name per line) — exit 1 with counts only (never the matching text).
- [ ] **Step 1: Write failing tests.** `test_explanation_at_most_300_words`; `test_release_scan_flags_planted_secret_in_temp_repo` (temporary git repo with a planted fake key → exit 1); `test_release_scan_clean_on_clean_temp_repo`; `test_readme_mentions_run_steps_privacy_limitations_and_misleading_signals`.
- [ ] **Step 2–4:** run (FAIL) → implement → run (PASS). README covers: what it does, architecture picture, privacy stance (local only, TTL), setup/run (`bootstrap.ps1`, `run.ps1`, `funnel.ps1`), evaluation results (honest, with the LOO numbers), **signals we trusted / distrusted and why**, limitations (Malayalam/Hinglish, thumbnail-only camera state, 10 labelled samples, label noise), and that the access code is supplied in the submission form, never in the repo. **Step 5: Commit** `docs: readme, explanation, release tooling`, tag `task-T15`.
- [ ] **Step 6 (human):** run `release_scan.py` → clean; flip the repo to public; open it **logged out** to confirm; submit the form; re-test the public link from mobile data at T-2 h and T-30 min.

---

## Self-Review (planner)
- **Spec coverage:** brief §1 sandwich → Tasks 4–11; §3.4 R1–R12 → Tasks 8–11 (each id has a source); §3.5 screen states / camera-off / name card → Task 8; §4.5 scoring, gates, confidence → Task 11; §4.6 Funnel → Task 3; §4.7–4.9 server/security → Tasks 2, 14; §4.10 firewall → `GEMINI.md`, Tasks 5/6/12/13 (numbers-only), 14 (log scrub), 15 (release scan); §6 evaluation → Task 13; §7 DoD → Tasks 3, 12–15.
- **Type consistency:** names in Tasks 4–12 are those defined in Task 1's `schema.py` (`MediaInfo`, `FrameRef`, `Transcript`, `Diarization`, `AudioFacts`, `VideoFacts`, `TranscriptFacts`, `DimensionScore`, `Report`) and in each task's *Interfaces* block (`transcribe`, `diarize`, `assign_speakers`, `identify_tutor`, `classify_frames`, `OllamaClient`, `ChecklistResult`, `score`, `build_report`, `run`).
- **Known soft spots (flagged, not hidden):** scoring priors are untuned until Task 13; camera-off inside the thumbnail strip is not detected; Malayalam-dominant classes degrade to audio/video-only scoring; pyannote install on Windows may force the ECAPA fallback.
