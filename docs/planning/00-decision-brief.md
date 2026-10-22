# Decision Brief v0.2 — "Can You Teach AI to Recognise a Great Class?"

Status: DRAFT v0.2 for approval — folds in the user's answers and sample frames (layout, domain, labels, hosting, repo). Nothing is built yet. Authors: Claude (planner / overseer / evaluator). Implementer + tester: Gemini CLI.
Recon date: Thu 2026-10-22 10:00–10:15 IST. Deadline: Fri 2026-10-23 12:00 (per brief; confirm timezone = IST).
This file contains **no recording content, no transcript text, no personal names** (batch IDs only) and is safe to commit.

---

## 1. TL;DR — the recommendation

**Build a local, three-layer "sandwich" system, not a VSA/perceptron classifier and not a single end-to-end video model.**

```
  BREAD (deterministic, auditable)     FILLING (small local LLM/VLM)        BREAD (deterministic)
  ───────────────────────────────  →  ───────────────────────────────  →  ──────────────────────────
  ffmpeg · VAD · ASR · diarization     reads transcript + measured facts     weighted rubric → grade
  audio DSP · face/brightness/blur     returns JSON checklist WITH           gates + confidence
  layout/background/camera-off CV      timestamps & short evidence           calibrated thresholds
                                       + a few keyframe yes/no questions     evidence-linked report
```

- **Everything runs on this machine** (RTX 3060 12 GB, 32 GB RAM). No recording byte goes to any cloud model/API. This is forced by the brief's data rule, not just by cost (§2).
- **The grade is a transparent weighted rubric** whose dimensions come straight from the 8x8 QA team's own feedback notes (`feedbacks.pdf`). Only the 3 grade thresholds are fitted. With 10 labelled recordings, anything "learned" would be noise (§3).
- **Hosting = this PC behind Tailscale Funnel** (already installed), FastAPI + a tiny vanilla-JS page, chunked uploads, access code, auto-purge (§4.5–4.9).
- **VSA is not the core.** Optional stretch only after Definition of Done is met (§4.3).
- **Hard rule for Claude and Gemini:** neither cloud assistant ever sees raw frames, audio, transcripts or per-recording outputs. Tests print numbers only (§4.10).

---

## 2. What the brief actually demands (read the fine print)

| Item | Implication |
|---|---|
| Input: upload **or link**; output: **Poor / Average / Good / Excellent** + *specific, useful* observations | 4-class ordinal output; evidence must cite moments/measurements, not generic praise |
| "Tested with a recording you have not seen" | Must generalise across unseen tutors/levels/languages; the **happy path on an unknown file is the whole exam** |
| Weights: assessment quality **35%**, working app + code quality **25%**, problem-solving & use of AI **30%**, clarity **10%** | 65% is judgement + process, not raw accuracy. Our devlog, eval harness and "signals we distrusted" section are worth real points |
| "Which signals are meaningful, which misleading?" | Include an explicit **misleading-signals** section in README/explanation (we already have evidence, §3.3) |
| **"Keep recordings confidential… process only using services specifically permitted by the organiser, especially if children are identifiable."** | Cloud VLM/ASR APIs are **out by default**. Also applies to *us*: pasting frames/transcripts into Claude or Gemini is a plausible violation, and the **AI interaction logs are a deliverable** that evaluators will read |
| Hosted link must be reachable at test time; repo must be accessible; README without secrets | Machine must stay up after the deadline (unknown window); repo visibility must be decided (§9) |
| Submission via Google Form: app link, repo, AI history (redacted), ≤300-word explanation | Plan the 300 words from day one (devlog feeds it) |

**Optional, parallel, zero-dependency action:** email the organiser asking whether any cloud service (e.g. Gemini API) is "specifically permitted". If yes we may upgrade later; the plan never depends on it.

---

## 3. What the data tells us (aggregate stats only)

### 3.1 Inventory
12 MP4s (51–325 MB; 38–89 min; all 1080p H.264, 25 fps, AAC 48 kHz stereo, **bitrate only 100–965 kb/s** → screen-recorded meeting with mostly static content), each with a platform auto-transcript (`.vtt`, speaker-labelled). Folders: excellent ×2, good ×2, average ×2, weak ×2, "quality issue" ×2, **demo ×2** (unlabelled; English and Malayalam). `feedbacks.pdf` = one human note per labelled class.

### 3.2 Label mapping (my call — confirm)
Excellent→Excellent · Good→Good · Average→Average · Weak→**Poor** · "Known quality issues"→**soft label "≤ Average"**: the user has watched them and is unsure whether the notes are about teaching or video quality (the notes mention both rapport/happiness *and* background/face). Soft labels are used only as an **upper-bound check** in validation, never as an exact fitting target · Demo→unlabelled hold-out (platform transcripts show 2–3 speakers → probably 1-on-1/trial classes; rubric must not punish "no group interaction" there).

### 3.3 Signal probe (platform transcripts as a stand-in; n = 10; Spearman ρ vs grade; hypotheses, not facts)

| id | grade | min | tutor talk share | student words/min | dead-air gaps/min (>8 s) | tutor praise words/min | HW mentions |
|---|---|---|---|---|---|---|---|
| B472 | Excellent | 56 | 0.47 | 61.1 | 0.07 | 0.77 | 3 |
| B491 | Excellent | 53 | 0.63 | 33.2 | 0.25 | 1.23 | 0 |
| B435-Jul | Good | 44 | 0.46 | 30.1 | 0.56 | 0.09 | 0 |
| B552 | Good | 46 | 0.57 | 32.1 | 0.46 | 1.27 | 3 |
| B435-Aug | Average | 38 | 0.57 | 42.9 | 0.42 | 0.29 | 0 |
| B516 | Average | 44 | 0.77 | 14.1 | 0.71 | 0.14 | 7 |
| B482 | Weak | 59 | 0.66 | 8.9 | 0.69 | 0.10 | 0 |
| B558 | Weak | 46 | 0.70 | 18.6 | 0.35 | 1.06 | 0 |
| B392 | Known issue | 60 | 0.74 | 16.1 | 0.46 | 0.66 | 0 |
| B441 | Known issue | 84 | 0.56 | 29.4 | 0.55 | 0.57 | 0 |

- **Looks meaningful:** student speech volume (ρ≈+0.69), tutor talk share (−0.43), dead air (−0.43), homework mentioned (+0.41), closing present (+0.35).
- **Looks misleading:** class length (ρ −0.43: longer ≠ better), raw praise-word count (a Weak class has one of the highest rates), "tutor called a child by name" count (ρ −0.39: weak classes spend time *summoning* kids, which is not rapport), video bitrate (the worst-quality-notes class has the *lowest* bitrate simply because the picture is static), speaker count (depends on who turned on name tags/cameras).
- **Warning (n = 10, 18 features tried):** a few of these correlations will be flukes. This is why weights come from the QA notes' *meaning*, not from fitting.
- **Language risk is real:** the "Malayalam" demo's platform transcript contains **zero Malayalam script** (English ASR run on Malayalam speech), and two classes show near-zero English greeting/question counts. Our ASR/LLM must flag low-confidence language and shift weight to audio/video evidence.
- **Label noise:** the same batch (B435) is both "Good" (July) and "Average" (August) → grade belongs to the *class/tutor*, never the batch. Feedback has date typos. Treat labels as ±1 grade.
- **Domain (my inference from word frequencies — confirm):** the lessons look like chess teaching ("8x8" = the board). Rubric stays subject-agnostic; the LLM prompt gets the domain as context.

### 3.4 The rubric the QA team is *already* using (extracted from `feedbacks.pdf`)

| # | Dimension | Cited in feedback as | Primary evidence source |
|---|---|---|---|
| R1 | Branded 8x8 background kept | "kept/fixed 8*8 background", "no background" | video (template/feature match) |
| R2 | Tutor visible: camera on, face visible, well lit | "camera turned off", "can't see his face", "sit where there is light" | video (tile detection, face, luma) |
| R3 | Audio/video clarity | "clarity issues", "informed about quality issues" | audio DSP + blur/bitrate-aware video |
| R4 | Opening: greeting + rapport **before** content | "good intro", "directly entered class without rapport" | transcript + prosody (first ~5 min) |
| R5 | New joiners given time to self-introduce | "should give time for new joiners" | transcript |
| R6 | Individual interaction: asks each child, includes everyone | "individually asking questions", "consider everyone" | diarization + transcript |
| R7 | Warm, energetic, pleasant delivery | "more energetic and pleasant", "happier" | prosody (pitch/energy variance) + transcript tone |
| R8 | Engagement / participation | (implicit in "one-to-one interaction") | student speech share, response latency, dead air |
| R9 | Content moves: revision, puzzles, topic videos | "can play topic related videos", "puzzles solving" | transcript + screen-share detection |
| R10 | Homework assigned | "given HWs", "should give HWs" | transcript (last 10 min weighted) |
| R11 | Proper conclusion | "better conclusion" | transcript (last ~5 min) |
| R12 | Technical stability | camera toggling, dropouts | audio gaps, black/frozen frames |

**Per-class checklist ground truth** (derived from the notes: which items were praised/flagged per recording) becomes our *second* labelled dataset — finer-grained than the 4 grades and much better for validating each detector.

**Weighting prior (my call, adjustable, *not fitted*):** the user confirms the grade is about **teaching practice, not video quality**, so Teaching = 75 % (R4 10 · R5 5 · R6 15 · R7 12 · R8 15 · R9 8 · R10 6 · R11 4) and Presentation = 25 % (R1 8 · R2 10 · R3 4 · R12 3). Clarity (R3) only counts when it actually impairs the lesson.

### 3.5 What the sample frames show (user-described + 5 frames the user supplied; domain = online chess for kids, confirmed)
- Zoom-style cloud recordings. The **shared screen fills the canvas** (chessboard/puzzle site with timer + leaderboard; a lesson-video player with a "Related Videos" panel; board diagrams). Tutor and students appear as a **tiny active-speaker thumbnail in the top-right** (about 1/6 of the width, partly cropped, name label under it). Sometimes the tutor or a student goes **full-screen**: the tutor on the **8x8 virtual background** (dark navy, white "8 ▮ 8" mark with a checker centre, "Eight Times Eight" wordmark, tagline at the bottom).
- **Camera off** = flat dark-grey frame/tile (≈ #1c1c1c) with the participant's name in white on a near-black label, centred. The tutor's name is shown in the first frames of each recording.
- Layout differs between recordings and within one → detectors must be **state-based, never position-hard-coded**.
- Design consequences:
  1. **No face detector in the core** (user's call, agreed): camera on/off needs only the flat-grey-card test plus the tutor's name card. Lighting of an *on* camera is judged from luma on full-frame talking-head states only. A native-resolution thumbnail crop + face detection is a stretch item if the QA notes about "can't see the face" turn out to matter.
  2. The thumbnail shows the *active speaker*, not necessarily the tutor → "tutor camera off while teaching" = (diarization says the tutor is speaking) ∧ (the shown tile is a camera-off card). **The tutor's name card in the first frames** (user's idea) is read once by the VLM; later full-screen cards are matched to it (fuzzy), so a student's camera-off card is never counted against the tutor.
  3. New per-frame **screen-state timeline**: `BRANDED_TALKING_HEAD · OTHER_TALKING_HEAD · CAMERA_OFF_CARD · BOARD/PUZZLE_SHARE · LESSON_VIDEO_SHARE · OTHER_SHARE` (flat-grey test, logo template match, CLIP zero-shot + simple CV for board/video-player UI). It yields: time on camera with the branded background (R1/R4/R11), share of time on board work vs lesson video (R9), camera-off-while-speaking (R2/R12), and class structure (opening on camera → board work → closing on camera).
  4. Bitrate/blur/low-light measures apply **only to talking-head states**. A static board at 100 kb/s is not a "quality problem".
  5. Chess-aware engagement proxies (cheap, domain-specific): student utterances containing piece names/squares/move notation per minute; tutor question followed within N s by a student reply; per-student turn counts during puzzles.
  6. **Privacy:** frames contain student handles (puzzle leaderboard) and faces → they never leave this machine, and reports show timestamps + text descriptions only (no thumbnails of other people).

---

## 4. Decision log (options → pros/cons → verdict)

### 4.1 Local vs third-party AI
| | Local (chosen) | Cloud APIs (Gemini/GPT video etc.) |
|---|---|---|
| Compliance | ✅ default-compliant | ❌ children identifiable; needs organiser permission |
| Cost | ✅ ~0 | ⚠ per-hour video/audio tokens add up; evaluator abuse risk |
| Quality ceiling | ⚠ 7–8B-class models | ✅ best (native video+audio in one call) |
| Latency | ⚠ ~8–12 min per hour of class (estimate, spike will measure) | ✅ faster |
| Uptime | ❌ depends on this PC + home internet | ✅ |
| Story for "judgement" score | ✅ shows we read the rules and engineered around them | ⚠ risks disqualification |
**Verdict: local.** Mitigate quality gap with structure (rubric + evidence + validation) and uptime gap with ops hardening (§4.9).

### 4.2 Architecture: end-to-end video model vs hybrid
Why **not** "just run Qwen2.5-VL-7B / Qwen3-VL-8B on the video":
1. Weights fit in 12 GB at Q4 (~5–6 GB) — **but an hour of video does not**: 1 fps × 3600 frames × ~100+ tokens/frame ≫ any usable context.
2. These models don't hear. Half the signal (tone, turn-taking, who speaks, dead air) is audio.
3. The local runtime we have (Ollama) takes images, not video files; Windows has no vLLM without WSL.
4. A free-form "rate this class" prompt on an 8B model is unstable and unexplainable; our 35% criterion is *usefulness and specificity*.

Use a VLM only as **sparse eyes**: ~20–30 keyframes, each with narrow yes/no + JSON questions the CV can't answer (e.g. "is a screen-share/board visible?", "is the tutor's face visible and lit?").

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| A. End-to-end VLM | simplest to describe | fails on length, no audio, unstable | ✗ |
| B. Custom VSA/perceptron classifier | matches your research; fun | no data to learn from; features still need ASR/CV anyway; hard to explain | ✗ as core |
| **C. Sandwich hybrid** | uses each tool where strong; explainable; testable per layer; degrades gracefully | more moving parts | ✅ |

### 4.3 Where VSA fits (honest assessment, grounded in your own reports)
- Your NSCK final report itself states: VSA nearest-neighbour quality degrades because random hypervectors **don't encode semantic similarity by construction**; text understanding is regex/TF-IDF-level; **no gradient learning** (xfailed by design).
- Our learning problem is ~10 labelled examples. Any trainable stack (perceptrons, HDC prototypes) reduces to nearest-centroid on hand-made features = no gain over a weighted rubric, at the cost of explainability.
- **Legit niche (stretch, ≤90 min, only after DoD is green):** *case-based explanation*: bind (dimension ⊗ level) role-filler vectors into a "class fingerprint", retrieve the most similar calibrated exemplar classes, and say "closest to B491 on participation, closest to B558 on opening". Pretty, but cosine on the 12-number profile gives the same answer. Judges reward working + explained, not exotic.
- **What we reuse conceptually:** the sandwich idea and evidence-binding discipline. **What we skip:** the NSCK codebase (build/dep risk on a 5-hour clock).

### 4.4 Models (all swappable via config; first spike benchmarks them on our own samples)
| Stage | Primary | Fallback | Notes |
|---|---|---|---|
| ASR | `faster-whisper` **large-v3-turbo**, int8_float16, VAD on, word timestamps | `distil-large-v3` (English only, fastest); `large-v3` for Malayalam-dominant | language-ID on several windows; flag low ASR confidence |
| Diarization | `pyannote/speaker-diarization-3.1` (gated: HF token, accept terms once) | SpeechBrain ECAPA + clustering; tutor = dominant cluster | Windows install is the main friction → time-boxed 30 min |
| Audio DSP | `librosa`/`parselmouth`/`pyloudnorm`: LUFS, SNR proxy, clipping, silence, pitch/energy variance, speech rate | — | prosody → "energetic" proxy |
| Video CV | OpenCV: **screen-state timeline** (flat-grey camera-off card, 8x8 logo template match, board / lesson-video share via CLIP zero-shot ViT-B/32 on sampled frames + simple CV); tutor name read once from the first-frame name card by the VLM; luma/blur on talking-head states only; **no face detector in the core** | — | logo template: I can cut it from the frame you sent (logo region only, no face), stored as a repo asset |
| Text LLM + VLM | **Qwen3-VL-8B** via Ollama (one model for text + keyframes, Q4_K_M ≈ 6 GB) *if available in our Ollama* | `qwen2.5vl:7b` + `qwen2.5:7b-instruct`; (cached HF Qwen2.5-7B is bf16 → too big) | JSON-schema constrained output; `keep_alive=0` to free VRAM between stages |
GPU is used sequentially (ASR ∥ diarization → unload → LLM/VLM); CV runs on CPU in parallel. Models and work dirs go on **F:** (98 GB free; C: has 44 GB). Set `OLLAMA_MODELS` accordingly.

### 4.5 Scoring method
- 12 rubric dimensions (§3.4), each 0–100 with **evidence list** (timestamp + ≤15-word quote or numeric measure; names masked).
- Overall = weighted sum with the prior weights in §3.4 (Teaching 75 / Presentation 25) → **ordinal mapping with 3 thresholds fitted only on the 8 hard-labelled classes** (LOO-validated). The 2 soft-labelled "known issue" classes just have to land ≤ Average; whether we say Poor or Average for them is reported, not scored.
- **Gates** that mirror the QA notes: tutor camera off/face not visible for >X% → cap at Average; effectively no audio/no student speech → cap at Average + "cannot assess" flag; recording <10 min or non-class (no speech) → "Insufficient evidence", not a fake grade.
- **Confidence** from ASR language confidence, diarization quality, coverage of detectors, LLM–rule disagreement.
- Detectors are **dual-path where possible** (regex/rule AND LLM extraction); disagreement lowers confidence and is displayed.

### 4.6 Hosting / tunnel
| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Tailscale Funnel** (installed v1.102.4) | free; stable `*.ts.net` HTTPS URL; as far as I know TLS terminates on *our* machine so the relay can't read uploads (**verify**); no small body limit I know of | currently `NoState` (needs login); Funnel must be enabled in the tailnet admin; undocumented bandwidth throttling | ✅ primary |
| Cloudflare quick/named tunnel | easy | `cloudflared` not installed; free plan **100 MB** request limit (files are 51–325 MB → forced chunking); TLS terminates at Cloudflare = third party sees video bytes; random URL per run | fallback only |
| Rent a GPU/VM | always-on | cost; data on third-party host; breaks the data rule | ✗ |
All designs use **chunked, resumable uploads (≤ 8 MB chunks)** so any tunnel and flaky home upstream work. *Measure home upload/download speed now; it bounds evaluator wait.*

### 4.7 UI/backend stack
| | FastAPI + vanilla JS (chosen) | Gradio | Streamlit |
|---|---|---|---|
| Large uploads | ✅ chunked, resumable, real progress | ⚠ single POST | ⚠ 200 MB default, websocket upload |
| Job model / queue / progress | ✅ explicit (SSE) | ✅ built-in queue | ⚠ reruns, session-bound |
| Security control (access code, TTL purge, no CDN) | ✅ full | ⚠ partial | ⚠ partial |
| Dev speed | ⚠ +≈45 min | ✅ | ✅ |
| Code-quality score | ✅ | ⚠ | ⚠ |
FastAPI/uvicorn already installed. UI has **no external CDN/fonts/scripts** (privacy + offline).

### 4.8 Input modes
Upload (MP4/MKV/MOV/WEBM, audio-only allowed with "no video analysis" flag) **and** link (direct URL; Google Drive link via `gdown`-style fetch; Zoom/Meet share links will not work → say so in UI). Server limits: ≤ 2.5 GB, ≤ 3 h; probe with `ffprobe` before accepting; friendly errors for corrupt/no-audio/no-video.

### 4.9 Security, privacy, operations (it is a public URL on a home GPU)
- Access code (shared with evaluators in the form text/README) + per-IP rate limit; random 128-bit job IDs; no job listing.
- Upload deleted right after processing; derived data + report purged after **24 h** (configurable); "Delete now" button; no content in logs; no analytics; no outbound network at runtime (test enforces it).
- Evidence quotes: tutor speech only, ≤ 15 words, **child names masked** (use "student A").
- `ffmpeg`/`ffprobe` invoked without shell, with timeouts and resource caps; disk-quota guard; single GPU worker + visible queue; stuck-job watchdog.
- Ops: `scripts/run.ps1` (server + worker + funnel), auto-restart, **disable sleep** (power plan is "Balanced" now), health page, tested from a phone on mobile data. A **static demo page** (synthetic/redacted example report, no frames, no real names) in the README as fallback if the PC is offline when evaluators click.

### 4.10 Claude ↔ Gemini workflow and the data firewall
- **Claude:** specs, task cards with acceptance tests, review of every diff, eval/calibration analysis, README/explanation. **Gemini via Antigravity (paste-in workflow, user's choice):** the user pastes one wave prompt at a time, Antigravity implements and tests, the user pastes its `REPORT` block back, and Claude verifies by reading the diff and re-running the tests itself. Antigravity is a cloud agent, so the **workspace is the repo folder only** and `GEMINI.md` carries the data firewall. Every prompt is saved under `docs/ai-history/prompts/` (part of the AI-history deliverable).
- `GEMINI.md` in repo will state: never open/cat/print anything under `class recordings/`, transcripts, frames, `data/`, `uploads/`; scripts that touch real recordings write only to ignored dirs and print **numbers only**; unit tests use **synthetic fixtures** (ffmpeg `lavfi`/tone/noise generated clips) so CI-style tests are shareable.
- Visual detectors need someone to *look at frames*: **that is you, locally** (15-min task, §9). Cloud assistants don't.
- Git: branch per card, small commits, no secrets; **`.gitignore` + local pre-commit guard already installed and tested** (blocks `.env`, media, transcripts, PDFs, models, DBs, >3 MB files, key-looking strings — even with `git add -f`).
- AI-history deliverable: raw exports go to `ai_history_raw/` (git-ignored); redacted copies to `docs/ai-history/`; a running `docs/devlog.md` (decision → why → result) feeds the 300-word explanation.

---

## 5. Target architecture

```
Browser (Funnel HTTPS) ──chunked upload / link──▶ FastAPI ──▶ job dir (ignored) ──▶ queue (1 GPU worker)
                                                                  │
   ┌──────────── Stage 0 ingest: ffprobe, validate, extract 16 kHz mono wav, 0.2 fps 640p frames
   ├─ Stage 1a (GPU) ASR turbo + VAD ──┐
   ├─ Stage 1b (GPU) diarization ──────┼─▶ diarized transcript (+confidence, language)
   ├─ Stage 1c (CPU) audio DSP ────────┤
   ├─ Stage 1d video CV + screen-state timeline ─┘─▶ measured facts (R1,R2,R3,R9,R12)
   ├─ Stage 2 (GPU) LLM checklist extraction (map-reduce over ~8-min chunks) + VLM keyframe Qs (R4–R7,R9–R11)
   ├─ Stage 3 rubric scoring + gates + calibrated thresholds + confidence
   └─ Stage 4 report builder (HTML + JSON) ──▶ SSE progress ──▶ result page ──▶ TTL purge
```
Time estimate for a 60-min class (to be **measured** in the first spike): ingest ~1 min, ASR ~3 min, diarization ~2 min (parallel), LLM map-reduce ~3–4 min, VLM ~2–3 min → **≈ 8–12 min wall**. Progress UI shows stage + ETA so waiting is not mistaken for a hang.

---

## 6. Evaluation & quality-analysis mechanisms (beyond "does it run")
1. **Grade validation:** leave-one-out over the 8 hard-labelled classes (+ the "≤ Average" constraint on the 2 soft-labelled ones); report exact, within-one, Kendall τ, confusion matrix; fit only the 3 thresholds.
2. **Checklist validation:** per-dimension detector vs the per-class checklist derived from the QA notes.
3. **ASR sanity:** WER of our ASR vs platform VTT on English classes (sanity bar, not truth — the VTT itself is noisy); **diarization sanity:** our tutor talk-share vs VTT tutor share (|Δ| ≤ 0.10).
4. **Metamorphic tests (ffmpeg-degraded copies):** mute 5 min → audio flag + score drops; add noise → clarity drops; black out video → R2/R1 flagged; downscale to 240p → clarity drops; truncate to 5 min → "insufficient evidence"; swap in a *different* class's audio → R-scores change sensibly. Proves the detectors react to the thing they claim to measure.
5. **Stability:** same file twice → same grade (seeded, temperature 0, schema-constrained); ±1 chunk boundary jitter → no grade flip.
6. **Held-out sanity:** the 2 demo recordings (unlabelled) → results are plausible to *you*, and no crash/odd grade for a different class type.
7. **Human spot-check by you:** 5 random evidence items per class: "is this timestamp/quote true?" Target ≥ 90%.
8. **Screen-state classifier check:** the pipeline writes a CSV of ~40 sampled timestamps with its predicted state per recording; you label them locally in a few minutes (no frames leave the machine); target ≥ 90 % agreement, and camera-off-while-speaking detections ≥ 80 % precision.

---

## 7. Definition of Done — "working end to end"
1. **Public path:** from a phone on mobile data, open the HTTPS link, enter the access code, upload an unseen MP4 (≥ 300 MB, ~60 min) **or** paste a link, watch staged progress, get the result **≤ 12 min**.
2. **Output contract:** grade ∈ {Poor, Average, Good, Excellent} (or "Insufficient evidence"), confidence, 1-paragraph summary, 12 dimension scores with evidence, **≥ 3 strengths and ≥ 3 improvements each tied to a timestamp/measure**, technical flags, "what we could not assess", signals we down-weighted; downloadable JSON + print-friendly page.
3. **Accuracy bar (reported honestly whatever it is):** LOO on the 8 hard-labelled classes: within-one ≥ 7/8, exact ≥ 5/8, both soft-labelled classes ≤ Average, **zero Excellent↔Poor inversions**, Kendall τ ≥ 0.5; checklist detectors ≥ 75% agreement where ground truth exists.
4. **Robustness:** all §6.4 metamorphic tests pass; bad inputs (non-video, corrupt, no audio, no video, >limits) give friendly errors; killing the worker mid-job → job resumes or fails cleanly; GPU memory returns to baseline after each job.
5. **Privacy/security:** zero third-party network calls at runtime (test); TTL purge verified; no content in logs; secrets/recordings scan of repo and git history clean; access code + rate limit working.
6. **Ops:** one-command start; auto-restart; sleep disabled; 3 sequential + 2 queued concurrent uploads succeed (soak).
7. **Submission pack:** README (setup, run, architecture, limitations), ≤ 300-word explanation, redacted AI history, repo accessible to evaluators, Google Form submitted, link re-tested from outside at T-2 h and T-30 min.
**Cut-line if behind at hour 3:** drop VLM keyframes (keep CV), drop pyannote (use ECAPA/heuristics), drop link-input (keep upload). Never drop: walking skeleton, evidence-linked output, eval harness, privacy/ops items.

---

## 8. Risks and fallbacks
| Risk | Likelihood | Impact | Fallback |
|---|---|---|---|
| pyannote install/gating friction on Windows | M | M | ECAPA clustering; 30-min time-box |
| Funnel not enabled / throttled | M | H | Cloudflare quick tunnel + chunking; test on hour 1 |
| Evaluator's recording is Malayalam/Hinglish-heavy | M | H | language-ID → large-v3 `ml`, low-confidence banner, shift weight to audio/video |
| PC offline at evaluation time | M | H | keep-awake, UPS?/power check, static demo page, tell evaluators the window |
| LLM JSON drift / hallucinated evidence | M | M | schema-constrained decoding, quote-must-exist-in-transcript verifier, rule cross-check |
| 12 GB VRAM contention | L | M | strict sequential GPU stages, unload, KV-cache quantisation |
| Overfitting thresholds to 10 samples | H | M | prior-driven weights, LOO, report honestly |
| Confidentiality slip via AI logs | L | H | data firewall, numbers-only prints, redaction pass before submission |
| Time sink in "interesting" work (VSA, fancy UI) | H | H | stretch list is gated behind DoD |

---

## 9. Needs from you — status after your 2026-10-22 replies
- ✅ **Layout + domain:** received and folded into §3.5 (chess; screen-share with tiny top-right thumbnail; camera-off = grey card). Further frames are optional — **please crop out student faces and puzzle-leaderboard handles** before sending any.
- ✅ **Tailscale:** `Running`, HTTPS certs enabled, you are tailnet admin. ⚠ **Funnel is not enabled on the tailnet** (verified 2026-10-22: `tailscale funnel` printed a one-time enable link and was left unconfigured). You open that link once; then `scripts/funnel.ps1` (Task 3) does the rest.
- ✅ **Uptime:** PC stays on through the evaluation window. Still to do on our side: disable sleep (plan is "Balanced"), auto-start after reboot, speed test.
- ✅ **Labels:** weak = Poor; "known quality issues" handled as soft label "≤ Average" (§3.2). ⚠ Still unknown: what the two demo recordings are (trial classes?).
- ✅ **Repo:** private now; **made public once everything works without credentials or recordings** (user's decision). Gate = `scripts/release_scan.py` clean on the working tree *and* full git history, then flip, then open logged-out. The access code goes in the submission form, never in the repo. Before flipping, review `docs/planning/` for per-class tables/batch IDs and decide whether to keep or aggregate them.
- ✅ **Workflow:** Antigravity paste-in (see §4.10). ⚠ Still open: what the two demo recordings are (assumed trial classes).
- ✅ **Go:** treated as given (the user asked for the prompts); data stance (no recording content to any cloud model) stands unless the user says otherwise.

Original list, for reference:
1. **Approve the data stance:** no recording content to any cloud model (incl. me and Gemini). (Recommended: yes.) Optionally email the organiser.
2. **Look at 3–4 frames per category yourself** (15 min) and tell me: layout (gallery vs speaker view; where the tutor tile sits), what the 8x8 background looks like and where, whether camera-off shows black/avatar/initials. Provide one cropped 8x8 logo/background image (no children in it).
3. **Tailscale:** log in on this PC and enable Funnel (status is `NoState` now); confirm the PC can stay on through the evaluation window; run a speed test for upload/download.
4. **Confirm label mapping (§3.2)** and what the two demos are; confirm 8x8 = chess (my inference).
5. **Repo visibility at submission:** public (after secrets scan) or private + evaluator invites? Form will need an openable link.
6. **Gemini CLI:** confirm it is authenticated here (v0.49.0 present) and which model tier you want it on.

## 10. Proposed clock (effort ≈ 5 h; wall-clock buffer overnight)
| Block | Output |
|---|---|
| H0 (30 min) | approvals, spec, `GEMINI.md`, task cards; env + model downloads started |
| H1 | **walking skeleton**: upload → job → stub report **live on the public link**; ASR+diarization spike with measured speed/WER on 2 recordings |
| H2 | audio/video/transcript detectors; LLM checklist extraction |
| H3 | scoring, calibration, LOO harness, metamorphic tests; **cut-line check** |
| H4 | report UI, security hardening, ops scripts |
| H5 | E2E on all 12 + demos, soak, README, 300 words, AI history, submit; final outside-network test Friday morning |

---

## 11. Builder self-check (reading this as the Gemini implementer)
**Answered by the conversation/project:** domain = chess; screen layout and camera-off look (§3.5); hosting = Tailscale Funnel; repo = private `shiva2321/video_class_eval` (no commits yet); Windows 11, Python 3.12, RTX 3060 12 GB, F: has 98 GB free.

**Minor — my calls, builder may refine without asking:**
- Env: `uv` venv at `.venv` (ignored); pinned CUDA-12.4 torch wheel; models under `F:\models` (set `OLLAMA_MODELS`, HF cache there); config in `config.yaml` + `.env` (`ACCESS_CODE`, `HF_TOKEN`), `.env.example` committed.
- API (all behind `X-Access-Code` header/cookie): `POST /api/uploads` → id; `PUT /api/uploads/{id}/chunks/{n}` (≤ 8 MB); `POST /api/uploads/{id}/complete`; `POST /api/jobs {upload_id | url}`; `GET /api/jobs/{id}` + SSE `/api/jobs/{id}/events`; `GET /api/jobs/{id}/report.json|html`; `DELETE /api/jobs/{id}`.
- Job state = one JSON file per job dir, written atomically (no DB); one GPU worker process; resumable stage markers.
- Sampling: 0.2 fps at 640 px wide for CV/CLIP; native-res crops only for the top-right strip; VLM keyframes = 24 (6 in the first 5 min, 6 in the last 5 min, 12 spread by scene change).
- Provisional grade cut-offs until calibration: Poor < 45, Average 45–62, Good 62–78, Excellent ≥ 78 (placeholders, replaced by LOO fit).
- Tutor = dominant diarized speaker (assumption; flagged low-confidence when the top two speakers are within 15 % talk time).
- Report JSON schema is written as the first task card so every stage targets it.

**Theirs (only the user can answer; changes what gets built):** (1) explicit go on the design and data stance; (2) how evaluators get repo access; (3) Gemini CLI auth/tier; (4) what the demo recordings are.
