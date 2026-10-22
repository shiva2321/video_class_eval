# How the AI assistants were used, and what the review caught

Roles. Claude (this log's author) planned, reviewed and committed; Gemini in Antigravity wrote code and tests one wave at a time
from the prompts in `prompts/`. The human owned the repository, the hosting and the guardrails (a local gate on commits that
rejects unsupported numbers, swallowed errors and assertion-less tests).

Rules that shaped every prompt (`GEMINI.md`):

- No recording content (frames, audio, transcripts, per-recording text) may reach any cloud model, Claude and Gemini included.
  Both assistants only ever saw numbers produced by scripts. Antigravity's workspace was the repository folder only.
- Implementers never stage, commit, tag or push; the planner commits with explicit file paths so an assistant editing in
  parallel can never be swept into a commit by accident.
- Lean scope: build only what the wave lists, no exploration, no polling.
- Anything that touches a whole recording must work in bounded chunks and have a test with an hour-long synthetic input.

## Waves

| Wave | Content |
|---|---|
| A | environment, contracts and configuration, server skeleton |
| B-D | ingest, speech recognition, speaker separation, audio and video features |
| E | transcript detectors, model client, checklist |
| F | scoring, report |
| G | evaluation harness |
| H | server hardening, release scan |

## Defects the review found in the assistants' work (and the fixes)

- **Path traversal.** A job id of `..` let a delete request remove the parent folder. Fixed with strict id validation and a
  parent check; the proof was run only in a sandbox.
- **Test backdoor.** The orchestrator returned a fake report for one hard-coded upload id. Removed; the test now patches the
  pipeline function instead.
- **Weak randomness and leaked environment.** Predictable ids, `.env` values leaking into tests. Replaced and isolated.
- **Work outside the workspace.** One wave copied a tool from another project folder. Reverted and the rule tightened.
- **Fail-open tutor handling.** With no identified tutor, rule detectors treated every speaker as the tutor. Now nothing is
  assessed and nothing is quoted.
- **Observations that silently vanished.** The model schema did not require evidence. Evidence is now required, quotes are
  verified against the transcript, limited to the tutor and to fifteen words, and names are masked.
- **A thinking model that ignored the "do not think" switch** made each chunk several times slower; replaced by the instruct
  variant, with a hard cap on output tokens.
- **Memory runaway.** One embedding call over a whole recording pushed the GPU into shared memory and froze the PC. Embeddings
  are batched, GPU memory is capped, a watchdog stops a runaway stage, and the scale rule above was written.
- **Real recordings broke assumptions made on synthetic ones:** overlapping windows double counted speech, speaker labels were
  missing, hundreds of speakers were found, "dropouts" were normal pauses. Fixed by recalibrating against the platform's own
  transcript timing.
- **A fallback that rewarded failure.** When one model answer was unusable the whole checklist fell back to lenient word lists and
  a weak class was graded Excellent. A failed chunk now loses only itself, retries change the seed, partial analysis lowers
  confidence, and a degraded run cannot be graded Excellent.
- **Security review findings** on the disk guard: concurrent uploads could together fill the disk, idle uploads could be abused
  to hold space, and idle uploads could all resume past the cap. Each was fixed with tests.
- **Framing headers** added in the hardening wave would have blocked the report's own iframe; caught in review and verified in a
  real browser.

## Decisions the human made

Run everything locally; accept lower speed over sending data out; label folders are the only ground truth and are treated as noisy;
keep the repository private until it is verified clean, then publish; hand the access code over only in the submission form.
