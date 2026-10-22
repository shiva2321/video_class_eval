# 8x8 Class Quality Assessor

Upload a recording of an online class; get a grade (**Poor / Average / Good / Excellent**, or **Insufficient evidence**) and an
explanation made of specific, checkable observations: what was measured, what the tutor said (short quotes, names masked), and
where in the recording it happened. Built for the Humaniti AI / 8x8 challenge "Can You Teach AI to Recognise a Great Class?".

Everything runs **on one Windows PC with a consumer GPU**. No recording, frame, audio clip or transcript is ever sent to a cloud
service: the challenge says recordings are confidential and the children in them are identifiable.

## What you get

- A web page (works on a phone): enter the access code, choose a recording, watch progress, read the report.
- The report: grade and confidence; a **"why this grade"** section (a short explanation written from the measured facts, and a
  table of the points each dimension cost); twelve rubric dimensions with a score or "not assessed" and the evidence behind each;
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
       -> the tutor on screen: a local vision model looks at the tutor's video while they speak (on camera? face clearly visible? light?)
       -> questions and answers: tutor questions, student replies, tutor feedback (timing and counts)
       -> local LLM (Ollama, qwen3-vl:8b-instruct): checklist verdicts + reason + verbatim tutor quotes, per ~8 minute chunk
       -> verify every quote in the transcript, mask names -> weighted rubric -> grade, confidence, flags
       -> the local model writes "why this grade" from the finished report's facts only; any sentence with a number or name that is
          not a fact is removed -> report
```

| Dimension | Weight | Measured from |
|---|---|---|
| R1 8x8 background | 8 | logo match in the full-frame tutor video and in the tutor's small video tile |
| R2 Tutor camera and lighting | 10 | local vision model: how often the tutor's face is clearly visible while they speak, and the light |
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
.\scripts\bootstrap.ps1         # environment, models (into models\), Ollama model
.\scripts\run.ps1               # server on 127.0.0.1:8765 and the worker; restarts either if it dies
.\scripts\funnel.ps1            # public https link through Tailscale Funnel (-DisableSleep keeps the PC awake)
```

Machine-specific settings (for example another model folder) go in an untracked `config.local.yaml` that is merged over
`config.yaml`. On another computer nothing else needs editing: `bootstrap.ps1` needs internet once, and the pipeline then works
offline. Systems other than Windows are untested.

Open `http://127.0.0.1:8765` (or the Funnel address), enter the code, choose a recording. One job runs at a time; others queue.
Command line, no server: `.venv\Scripts\python.exe scripts\run_pipeline.py --input <file> --out <folder>`.
Tests: `.venv\Scripts\python.exe -m pytest -m "not gpu and not ollama and not e2e"`.

## Speed

Measured on the development PC (RTX 3060 12 GB), full pipeline from scratch (`results/timings.json`): a recording of 56.0 minutes
took 302.2 s and one of 44.1 minutes took 232.0 s, about 11.1 and 11.4 minutes of recording per minute of processing, plus the
short report stage. The language-model checklist (126.38 s and 107.89 s) is the largest part, then the video stage with the
vision model (70.86 s and 58.95 s) and speech recognition (43.86 s and 35.52 s). Memory is bounded (audio and embeddings are
processed in chunks, GPU memory is capped, a watchdog stops a runaway stage), so a three-hour recording takes longer but does
not exhaust memory.

## Evaluation (read this before trusting the grade)

Ten recordings with a quality folder were available: eight with a clear label (2 excellent, 2 good, 2 average, 2 weak) and
two "quality issue" classes that should land at Average or below. Labels come from folder names; per-dimension truth comes from
the tutors' feedback notes (`results/eval_*.json`, produced by `scripts/eval_loo.py`). With four grades, guessing is right 0.25
of the time, and eight classes cannot tell a good system from a lucky one: read every number below as indicative only.

With the configured boundaries (weak is read as Poor):

| | |
|---|---|
| exact grade match on the 8 labelled classes | 0.5 |
| within one grade | 0.875 |
| rank correlation of score with label (Kendall tau) | 0.5893 |
| "quality issue" classes at Average or below | 2 of 2 |

These are **optimistic**: the boundaries, the start-time part of R4 and the thresholds of the tutor-visibility measure were chosen
after looking at the same ten classes. Fitting the boundaries leave-one-out, as the evaluation script can, gives exact 0.25 and
within-one 0.875, so the exact-match figure mostly reflects where the boundaries were put. What improved for a reason, not by
tuning: letting a vision model look at the tutor raised agreement with the tutors' notes on camera and lighting (R2) from 0 to
0.67 (3 classes), and looking for the 8x8 background in the tutor's small video tile raised R1 to 8 of 8 classes.

Per-dimension checks against the tutors' notes (a score of 60 or more counts as "present"): R1 background 8 of 8 classes; R2 camera 0.67
(3 classes); R4 opening 0.67 (6), R10 homework 0.67 (6), R6 individual interaction 0.57 (7). **R3 audio clarity and R7 warmth did not
reproduce the issues the tutors flagged (0 of 7 classes together)**: those measures read as fine on classes where the reviewers saw
problems. They carry weight in the score but are the least trustworthy parts of it.

### Signals we trusted, and ones we did not

Trusted: the 8x8 logo match (also in the small tutor tile), whether the tutor's face is on screen while they speak, who spoke
when, how soon the tutor started, question and answer timing, and verbatim tutor quotes that are verified in the transcript.
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
- **Camera.** While something is shared, the tutor is a small tile that the vision model reads; the tile shows the active speaker,
  so when speaker detection is off by a few seconds the tile may show someone else.
- **The written explanation** comes from a small local model. It sees only the report's facts and is cut back to sentences whose
  numbers and names are facts, but it can still phrase a cause too confidently; the table of points lost next to it is plain arithmetic.
- **Homework** shared only in the chat window cannot be heard in the recording.
- **Labels are few and noisy.** See above. The tutors' notes are the only ground truth and mix teaching and video-quality remarks.
- **Hosting.** The public link works only while this PC is on and the worker is running.

## Repository

`classeval/` application (`pipeline/` stages, `server/` API and UI, `eval/` metrics) · `scripts/` ops, evaluation and release tools ·
`tests/` · `config.yaml` all tunable settings · `docs/` design brief, plan, and the prompts used with AI coding assistants
(`docs/ai-history/`). Work was planned and reviewed with Claude and implemented with Gemini in Antigravity under written rules
(`GEMINI.md`); the prompts and the review notes are the interaction history.
