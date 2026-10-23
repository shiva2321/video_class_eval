# 8x8 Class Quality Assessor

Give it a recording of an online class; get a grade (**Poor / Average / Good / Excellent**, or **Insufficient evidence**) and an
explanation made of specific, checkable observations: what was measured, what the tutor said (short quotes, names masked), and
where in the recording it happened. Built for the Humaniti AI / 8x8 challenge "Can You Teach AI to Recognise a Great Class?".

It runs **entirely on your own computer**. No recording, frame, audio clip or transcript is sent to any cloud service: the
challenge says the recordings are confidential and the children in them are identifiable. After the one-time model download it
works offline.

## What you need

| | |
|---|---|
| Operating system | Windows 11 (developed and tested here), Linux or macOS (see "What has been tested") |
| Graphics card | NVIDIA with about 12 GB of memory recommended. It also runs on the CPU alone, several times slower |
| Disk | about 25 GB free: packages 5 GB, speech models 5 GB, language model 6 GB, working space |
| Tools to install first | [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 itself), [ffmpeg](https://ffmpeg.org), [Ollama](https://ollama.com) |
| Internet | once, for setup; the analysis itself never needs it |

## Set up (about 30 minutes, mostly downloads)

1. Install the three tools. For example:
   - Windows: `winget install --id astral-sh.uv -e`, `winget install --id Gyan.FFmpeg -e`, and the Ollama installer from ollama.com
   - macOS: `brew install uv ffmpeg`, and the Ollama app
   - Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`, `sudo apt install ffmpeg`, `curl -fsSL https://ollama.com/install.sh | sh`
2. Start Ollama (the app, or `ollama serve` in a terminal).
3. In the repository folder run **one command**:

```
python scripts/install.py
```

It checks the tools, builds the environment from `uv.lock` (exact package versions), writes `.env` with a random access code,
downloads the speech, voice and image models, pulls the local language model (`qwen3-vl:8b-instruct`), and finishes with a
**self-test**: a generated clip goes through the real pipeline with the real models. When it prints `SELFTEST OK` the
installation works. `--english-only` skips the 3 GB multilingual speech model; the other languages then cannot be transcribed.

Something went wrong? Every step prints what failed and what to do. The steps can be run alone: `uv sync --frozen`,
`uv run python scripts/download_models.py`, `ollama pull qwen3-vl:8b-instruct`, `uv run python scripts/selftest.py`.

## Use it

**Analyse a recording from the command line**

```
uv run python scripts/run_pipeline.py --input path/to/recording.mp4 --out results_folder
```

The report is written to `results_folder/report.html` (open it in a browser) and `report.json`.

**Or use the web page**

```
uv run python scripts/run.py
```

Open http://127.0.0.1:8765, enter the access code from `.env`, choose a recording and watch the progress; the report appears
on the same page. One job runs at a time, others wait. The page is only reachable from your own computer unless you pass
`--host 0.0.0.0`.

Recordings of 10 minutes to 3 hours work (a shorter one gives "Insufficient evidence"). Settings are in `config.yaml`; settings
that belong to one computer (for example another model folder) go in an untracked `config.local.yaml` merged over it.

### Share it from your own computer (optional)

The page can be reached from other devices without any server or cloud account, using [Tailscale Funnel](https://tailscale.com/kb/1223/funnel)
(free; the analysis still happens on your computer and the recording only travels, encrypted, between the browser and your computer):

```
uv run python scripts/run.py        # keep this running
tailscale funnel --bg 8765          # prints the public https address; "tailscale funnel --https=443 off" switches it off
```

Anyone who opens that address needs the access code from `.env`. A link ending in `#code=` followed by the code fills it in
automatically (the part after `#` is never sent to a server and is removed from the address bar), so one link is enough to hand over.
Anyone who has such a link can upload recordings to your computer: share it only with people you trust and switch the funnel off
afterwards. The computer must stay awake and `scripts/run.py` running while others use it.

## What the report contains

A grade and a confidence; a **"why this grade"** section (a short explanation written from the measured facts, and a table of the
points each dimension cost); twelve rubric dimensions, each with a score or "not assessed" and the evidence behind it; strengths and
improvements; "class at a glance" (who spoke how much, minute by minute; what was on screen; what the class covered); "questions and
answers" (what the tutor asked, whether and how fast students answered, what the tutor did next); technical flags. The same data
is in `report.json`.

## How it works

The language model never decides the grade. It reads the transcript and answers a fixed checklist; everything else is measured,
and a transparent weighted rubric turns the facts into a score.

```
recording -> ffmpeg: audio, one frame per 5 s, the tutor's small video tile every 10 s (one pass over the video)
          -> speech: faster-whisper large-v3-turbo (other languages: large-v3), language detected first
          -> who spoke: ECAPA voice embeddings, clustering, tutor = dominant voice
          -> audio facts: loudness, clipping, noise, talk shares, pauses, response time, pitch and energy variation
          -> screen timeline: CLIP zero-shot (board / lesson video / face), logo template match (8x8 background), camera-off card
          -> the tutor on screen: a local vision model looks at the tutor's video while they speak (on camera? face clearly visible? light?)
          -> questions and answers: tutor questions, student replies, tutor feedback (timing and counts)
          -> local language model (Ollama): checklist verdicts + reason + verbatim tutor quote, per ~8 minute chunk
          -> every quote is verified in the transcript, names are masked -> weighted rubric -> grade, confidence, flags
          -> the local model writes "why this grade" from the finished report's facts only; any sentence with a number or name
             that is not a fact is removed -> report
```

| Dimension | Weight | Measured from |
|---|---|---|
| R1 8x8 background | 8 | logo match in the full-frame tutor video and in the tutor's small video tile |
| R2 Tutor camera and lighting | 10 | local vision model: how often the tutor's face is clearly visible while they speak, and the light |
| R3 Audio clarity | 4 | loudness, clipping, noise |
| R4 Opening and rapport | 10 | model verdict on the opening words (60 %) + how soon the tutor started speaking (40 %) |
| R5 New-joiner introductions | 5 | model verdict |
| R6 Individual interaction | 15 | question reply rate, distinct children answering, model verdict (mean over the class) |
| R7 Warmth and energy | 12 | model verdict (mean over the class), pitch and energy variation |
| R8 Student participation | 15 | student talk pace, dead air, number of children heard |
| R9 Content moves | 8 | share of time on shared material, use of a lesson video, model verdict (mean over the class) |
| R10 Homework | 6 | model verdict (last two chunks) |
| R11 Conclusion | 4 | model verdict (last chunk) |
| R12 Technical stability | 3 | camera off for most of the recording |

The twelve dimensions come from the tutors' own quality-review notes. A dimension that cannot be measured is shown as **not
assessed** and its weight is removed, never guessed. Boundaries on the 0-100 score are in `config.yaml` (`cutoffs`).

Safeguards on the grade: too little evidence gives "Insufficient evidence"; the top grade needs at least 80 % of the rubric
assessed; if the model could not analyse the class at all the grade is capped at Good; a tutor camera that is off for half the
time they speak caps the grade at Average.

## Privacy

- Local only: the app talks to `127.0.0.1` (Ollama) and nothing else when it analyses; a test fails if anything tries to leave the
  machine. Models are downloaded once by `scripts/install.py`.
- The extracted frames and audio (which show and carry the children) are deleted at the end of every run, successful or not; in
  the web page the upload and all working files are deleted when a job ends, and the finished report after 24 hours.
- The report quotes **only the tutor**, at most 15 words, each quote verified against the transcript, names masked. What children
  said is measured (timing, length), never quoted. A test checks that a phrase spoken by a child never reaches logs or stored files.
- Logs contain stage names, durations and error class names, never speech.
- The repository contains no recordings, frames, transcripts or feedback documents (`scripts/release_scan.py` checks the working tree
  and the whole git history for secrets, media, pictures, audio, e-mail addresses and listed names).

## Speed

Measured on the development PC (RTX 3060 12 GB, Windows 11), full pipeline from scratch (`results/timings.json`): a recording of
56.0 minutes took 177.8 s and one of 44.1 minutes took 160.2 s, about 18.9 and 16.5 minutes of recording per minute of processing,
plus the short report stage. The language-model checklist is the largest part (71.45 s and 63.64 s), then speech recognition
(40.3 s and 33.83 s), the video stage with the vision model (26.84 s and 31.92 s) and frame extraction (23.66 s and 19.33 s).

Several transcript chunks and vision questions are sent to the local model at once. That helps when Ollama serves several requests
in parallel (it does by default on the development PC); otherwise they queue and nothing is lost. Without an NVIDIA GPU everything
still works but is much slower: speech, voice and image models run on the CPU, and the language model is the slowest part (minutes
per chunk), so allow hours rather than minutes for a long class. Memory is bounded (audio and embeddings are processed in chunks,
GPU memory is capped, a watchdog stops a runaway stage), so a three-hour recording takes longer but does not exhaust memory.

## Evaluation (read this before trusting the grade)

Ten recordings with a quality folder were available: eight with a clear label (2 excellent, 2 good, 2 average, 2 weak) and
two "quality issue" classes that should land at Average or below. Labels come from folder names; per-dimension truth comes from
the tutors' feedback notes (`results/eval_*.json`, produced by `scripts/eval_loo.py`). With four grades, guessing is right 0.25
of the time, and eight classes cannot tell a good system from a lucky one: read every number below as indicative only.

With the configured boundaries (weak is read as Poor):

| | |
|---|---|
| exact grade match on the 8 labelled classes | 0.75 |
| within one grade | 0.875 |
| rank correlation of score with label (Kendall tau) | 0.7715 |
| "quality issue" classes at Average or below | 2 of 2 |

These are **optimistic**. The boundaries are the rounded midpoints between the mean scores of the labelled grades, and the start-time
part of R4, the thresholds of the tutor-visibility measure, the averaging of whole-class behaviours and several prompt changes were
all chosen after looking at the same ten classes. Fitting the boundaries leave-one-out, as the evaluation script can, gives exact
0.375 and within-one 0.875: above guessing, but on eight classes that is not proof. The model's own sampling noise is small but real:
the same class scored with three different seeds moved by at most 3.0 points (`results/seed_noise.json`), enough to cross a boundary.

What improved for a reason, not by tuning: letting a vision model look at the tutor raised agreement with the tutors' notes on camera
and lighting (R2) from 0 to 0.67 (3 classes), and looking for the 8x8 background in the tutor's small video tile raised R1 to 8 of 8
classes. Three videos from the internet that are not classes (360p, 480p and 720p, 6 to 12 minutes, one in Hindi) were also run
(`results/samples.json`) to check that the pipeline copes with other resolutions, lengths and languages: all three finished without
error, the two under ten minutes were refused as "Insufficient evidence", and the Hindi one was flagged as an unsupported language.

Per-dimension checks against the tutors' notes (a score of 60 or more counts as "present"): R1 background 8 of 8 classes; R10 homework 0.8
(5 classes); R2 camera 0.67 (3); R4 opening 0.5 (6); R6 individual interaction 0.43 (7). **R3 audio clarity and R7 warmth did not
reproduce the issues the tutors flagged (0 of 7 classes together)**: those measures read as fine on classes where the reviewers saw
problems. They carry weight in the score but are the least trustworthy parts of it.

### Signals we trusted, and ones we did not

Trusted: the 8x8 logo match (also in the small tutor tile), whether the tutor's face is on screen while they speak, who spoke
when, how soon the tutor started, question and answer timing, and verbatim tutor quotes that are verified in the transcript.
Not trusted or down-weighted: praise-word counts (easy to game, little signal), audio "dropouts" (normal pauses in meeting
recordings produce many), recording length and bitrate, and any model verdict without a verified quote (it is left out).

## Limitations

- **Language.** Transcript-based checks are validated for English only. Other languages are transcribed and the language-independent
  measures (talk shares, timing, screen use, audio, start time, background) are reported; the text-based dimensions are marked not
  assessed, the grade is capped at Good, and the report says so. Automatic language detection can confuse close languages
  (a Malayalam demo was reported as Tamil).
- **Subject.** Wording is subject neutral; only the screen-state prompts (board, puzzle) are about chess.
- **Speakers.** The tutor's voice sometimes splits into two voices, which understates the tutor's talk time and overstates the
  number of students. Children are not identified by name.
- **Camera.** While something is shared, the tutor is a small tile that the vision model reads; the tile shows the active speaker,
  so when speaker detection is off by a few seconds the tile may show someone else.
- **The written explanation** comes from a small local model. It sees only the report's facts and is cut back to sentences whose
  numbers and names are facts, but it can still phrase a cause too confidently; the table of points lost next to it is plain arithmetic.
- **Homework** shared only in the chat window cannot be heard in the recording.
- **Model noise.** The model's verdicts change a little with its sampling seed; see the evaluation above for how much.
- **Labels are few and noisy.** The tutors' notes are the only ground truth and mix teaching and video-quality remarks.

## What has been tested

- **Windows 11, NVIDIA RTX 3060 12 GB.** A fresh clone with an empty model cache went through `python scripts/install.py` (environment from
  the lock file, speech, voice and image models downloaded, language model present) and printed `SELFTEST OK`. The unit tests
  (`pytest -m "not gpu and not ollama and not e2e"`), the GPU and end-to-end tests, the ten labelled recordings, the two demos and
  three internet videos all ran.
- **Windows, GPU hidden** (`CUDA_VISIBLE_DEVICES=-1`). The self-test also printed `SELFTEST OK`, so speech recognition, voice clustering and
  the image model work on the CPU. The language model still ran on the GPU through Ollama, so a fully CPU-only run was not timed.
- **Linux and macOS: not run.** The code and the lock file are written for them (a torch build per platform, the CPU fallback, test
  speech from SAPI, `say` or espeak, no drive letters), but nobody has run them yet. On Linux with an NVIDIA GPU, faster-whisper needs
  the CUDA libraries; if it complains about a missing cuDNN or cuBLAS library, hide the GPU for the speech step by starting with
  `CUDA_VISIBLE_DEVICES=-1`, or install the system CUDA libraries.
- **Not covered:** recordings in other layouts than a shared screen with the tutor in the top-right corner (the tile position is the
  setting `screen_state.thumb_region`), and classes in a gallery view.

## Repository

`classeval/` application (`pipeline/` stages, `server/` page and API, `eval/` metrics) · `scripts/` setup, run, evaluation and release
tools · `tests/` · `config.yaml` all tunable settings · `docs/` design brief, plan, and the prompts used with AI coding assistants
(`docs/ai-history/`). Work was planned and reviewed with Claude and implemented with Gemini in Antigravity under written rules
(`GEMINI.md`); the prompts and the review notes are the interaction history.
