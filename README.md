# 8x8 Class Quality Assessor

Upload a recording of an online class; get a grade (**Poor / Average / Good / Excellent**, or **Insufficient evidence**) and an
explanation made of specific, checkable observations: what was measured, what the tutor said (short quotes, names masked), and
where in the recording it happened. Built for the Humaniti AI / 8x8 challenge "Can You Teach AI to Recognise a Great Class?".

Everything runs **on one Windows PC with a consumer GPU**. No recording, frame, audio clip or transcript is ever sent to a cloud
service: the challenge says recordings are confidential and the children in them are identifiable.

## What you get

- A web page (works on a phone): enter the access code, choose a recording, watch progress, read the report.
- The report: grade and confidence; twelve rubric dimensions with a score or "not assessed" and the evidence behind each;
  strengths and improvements; a "class at a glance" section (who spoke how much, minute by minute; what was on screen; what
  the class covered); a "questions and answers" section (what the tutor asked, whether and how fast students answered, what the
  tutor did next); technical flags. The same data is available as JSON (`/api/jobs/<id>/report.json`).
- A grade is withheld ("Insufficient evidence") when the recording is too short, silent, or too little of the rubric can be assessed.

## How it works

The language model never decides the grade. It reads the transcript and answers a fixed checklist; everything else is measured,
and a transparent weighted rubric turns the facts into a score.

```
upload -> ffmpeg: audio + one frame per 5 s + top-right tutor tile every 20 s
       -> speech: faster-whisper large-v3-turbo (other languages: large-v3), language detected first
       -> who spoke: ECAPA voice embeddings, clustering, tutor = dominant voice
       -> audio facts: loudness, clipping, noise, talk shares, pauses, response time, pitch/energy variation
       -> screen timeline: CLIP zero-shot (board / lesson video / face), logo template match (8x8 background), camera-off card
       -> questions and answers: tutor questions, student replies, tutor feedback (timing and counts)
       -> local LLM (Ollama, qwen3-vl:8b-instruct): checklist verdicts + reason + verbatim tutor quotes, per ~8 minute chunk
       -> verify every quote in the transcript, mask names -> weighted rubric -> grade, confidence, flags -> report
```

| Dimension | Weight | Measured from |
|---|---|---|
| R1 8x8 background | 8 | logo match in the full-frame tutor video and in the tutor's small video tile |
| R2 Tutor camera and lighting | 10 | camera-off time while the tutor speaks, brightness |
| R3 Audio clarity | 4 | loudness, clipping, noise |
| R4 Opening and rapport | 10 | model verdict on the opening words (60 %) + how soon the tutor started speaking (40 %) |
| R5 New-joiner introductions | 5 | model verdict |
| R6 Individual interaction | 15 | question reply rate, distinct children answering, model verdict |
| R7 Warmth and energy | 12 | model verdict, pitch and energy variation |
| R8 Student participation | 15 | student talk pace, dead air, number of children heard |
| R9 Content moves | 8 | share of time on shared material, use of a lesson video, model verdict |
| R10 Homework | 6 | model verdict (last two chunks) |
| R11 Conclusion | 4 | model verdict (last chunk) |
| R12 Technical stability | 3 | camera off for most of the recording |

The twelve dimensions come from the tutors' own quality-review notes. A dimension that cannot be measured is shown as **not
assessed** and its weight is removed, never guessed. Boundaries on the 0-100 score are in `config.yaml` (`cutoffs`).

Safeguards on the grade: too little evidence gives "Insufficient evidence"; the top grade needs at least 80 % of the rubric
assessed; if the model could not analyse the class at all the grade is capped at Good; a tutor camera that is off for half the
time they speak caps the grade at Average.

## Privacy

- Local only: the app talks to `127.0.0.1` (Ollama) and nothing else at run time; a test fails if anything tries to leave the
  machine. Models are downloaded once by `scripts/bootstrap.ps1`.
- Access code on every request; wrong-code lock-out and rate limiting; ids are validated against path tricks; security headers;
  uploads and working files are deleted when a job ends, finished jobs after 24 hours.
- The report quotes **only the tutor**, at most 15 words, each quote verified against the transcript, names masked. What children
  said is measured (timing, length), never quoted. A test checks that a phrase spoken by a child never reaches logs or stored files.
- Logs contain stage names, durations and error class names, never speech.
- The access code is given to the evaluators in the submission form. It is not in this repository, and neither are recordings,
  transcripts or feedback documents (`scripts/release_scan.py` checks the working tree and the whole git history).

## Run it

Needs Windows 11, an NVIDIA GPU with 12 GB, Python 3.12 with [uv](https://docs.astral.sh/uv/), ffmpeg, [Ollama](https://ollama.com)
and, for a public link, [Tailscale](https://tailscale.com).

```powershell
cd video_class_eval
copy .env.example .env          # then set ACCESS_CODE (at least 10 characters)
.\scripts\bootstrap.ps1         # environment, models, Ollama model
.\scripts\run.ps1               # server on 127.0.0.1:8765 and the worker; restarts either if it dies
.\scripts\funnel.ps1            # public https link through Tailscale Funnel (-DisableSleep keeps the PC awake)
```

Open `http://127.0.0.1:8765` (or the Funnel address), enter the code, choose a recording. One job runs at a time; others queue.
Command line, no server: `.venv\Scripts\python.exe scripts\run_pipeline.py --input <file> --out <folder>`.
Tests: `.venv\Scripts\python.exe -m pytest -m "not gpu and not ollama and not e2e"`.

## Speed

Measured on the development PC (RTX 3060 12 GB), full pipeline from scratch (`results/timings.json`): a recording of 56.0 minutes
took 240.4 s and one of 44.1 minutes took 191.1 s, about 14.0 and 13.8 minutes of recording per minute of processing. The
language-model checklist (130.05 s and 104.64 s) and speech recognition (43.05 s and 35.72 s) dominate. Memory is bounded
(audio and embeddings are processed in chunks, GPU memory is capped, a watchdog stops a runaway stage), so a three-hour recording
takes longer but does not exhaust memory.

## Evaluation (read this before trusting the grade)

Ten recordings with a quality folder were available: eight with a clear label (2 excellent, 2 good, 2 average, 2 weak) and
two "quality issue" classes that should land at Average or below. Labels come from folder names; per-dimension truth comes from
the tutors' feedback notes (`results/eval_*.json`, produced by `scripts/eval_loo.py`).

With the configured boundaries (weak is read as Poor):

| | |
|---|---|
| exact grade match on the 8 labelled classes | 0.25 |
| within one grade | 0.875 |
| rank correlation of score with label (Kendall tau) | 0.5107 |
| "quality issue" classes at Average or below | 2 of 2 |

These are **optimistic**: the boundaries and one rubric rule (the start-time part of R4) were chosen after looking at the same ten
classes. Fitting the boundaries leave-one-out, as the evaluation script can, gives exact 0.25 and within-one 0.625. Ten classes
cannot support a precise fit; treat the grade as a coarse signal and the evidence as the real product.

Per-dimension checks against the tutors' notes (a score of 60 or more counts as "present"): R1 background agrees on 8 of 8 classes;
R4 opening 0.67 (6 classes), R10 homework 0.67 (6), R6 individual interaction 0.57 (7). **R2 camera/lighting, R3 audio clarity and R7
warmth did not reproduce the issues the tutors flagged (0 of 10)**: those measures read as fine on classes where the reviewers saw
problems. They carry weight in the score but are the least trustworthy parts of it.

### Signals we trusted, and ones we did not

Trusted: the 8x8 logo match (also in the small tutor tile), who spoke when, how soon the tutor started, question and answer
timing, and verbatim tutor quotes that are verified in the transcript.
Not trusted or down-weighted: praise-word counts (easy to game, little signal), audio "dropouts" (normal pauses in meeting
recordings produce many), recording length and bitrate, and any model verdict without a verified quote (it falls back to
"partial" or "not assessed").

## Limitations

- **Language.** Transcript-based checks are validated for English only. Other languages are transcribed and the language-independent
  measures (talk shares, timing, screen use, audio, start time, background) are reported; the text-based dimensions are marked not
  assessed, the grade is capped at Good, and the report says so. Automatic language detection can confuse close languages
  (a Malayalam demo was reported as Tamil).
- **Subject.** Wording is subject neutral and nothing is chess specific except the screen-state prompts (board, puzzle).
- **Speakers.** The tutor's voice sometimes splits into two voices, which understates the tutor's talk time and overstates the
  number of students. Children are not identified by name.
- **Camera.** The tutor's video tile is checked for the 8x8 background; a camera-off card inside the small tile is not detected.
- **Homework** shared only in the chat window cannot be heard in the recording.
- **Labels are few and noisy.** See above. The tutors' notes are the only ground truth and mix teaching and video-quality remarks.
- **Hosting.** The public link works only while this PC is on and the worker is running.

## Repository

`classeval/` application (`pipeline/` stages, `server/` API and UI, `eval/` metrics) · `scripts/` ops, evaluation and release tools ·
`tests/` · `config.yaml` all tunable settings · `docs/` design brief, plan, and the prompts used with AI coding assistants
(`docs/ai-history/`). Work was planned and reviewed with Claude and implemented with Gemini in Antigravity under written rules
(`GEMINI.md`); the prompts and the review notes are the interaction history.
