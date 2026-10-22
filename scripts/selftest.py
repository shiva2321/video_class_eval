"""Check an installation end to end: a generated clip goes through the real pipeline with the real models.

    uv run python scripts/selftest.py

The clip is made here (ffmpeg test pattern, plus speech from this machine's text-to-speech engine when it has one), so no
recording is needed. A short synthetic clip is not a class, so the result of interest is not the grade but that every stage ran:
speech recognition, voice clustering, image and vision models, the local language model, scoring and the report.
Exit code 0 means the installation works.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SENTENCES = [
    "Hello everyone, welcome to our chess class today.",
    "Today we will look at the knight and how it moves on the board.",
    "Can you tell me which square the knight can jump to from here?",
    "Very good. Now let us practise with a short puzzle together.",
    "For homework, please solve three puzzles before the next class. Goodbye everyone.",
]


def fail(message: str) -> int:
    print(f"SELFTEST FAILED: {message}", flush=True)
    return 1


def make_test_clip(path: Path) -> str:
    """A 70 s clip: test-pattern video and speech (or a tone when the machine has no speech engine). Returns the audio kind."""
    import subprocess

    from tests.fixtures.clips import make_clip
    from tests.fixtures.speech import concat_wavs, make_speech_wav, speech_backend

    if speech_backend() is None:
        make_clip(path, seconds=70.0)
        return "tone"
    tmp = path.parent
    parts = [make_speech_wav(sentence, tmp / f"s{i}.wav") for i, sentence in enumerate(SENTENCES * 3)]
    speech = concat_wavs(parts, gap_s=1.5, out=tmp / "speech.wav")
    silent = make_clip(tmp / "silent.mp4", seconds=70.0, with_audio=False)
    cmd = ["ffmpeg", "-y", "-i", str(silent), "-i", str(speech), "-c:v", "copy", "-c:a", "aac", "-t", "70", str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120.0)
    return "speech"


def main() -> int:
    import torch

    from classeval.config import load_settings
    from classeval.pipeline import ingest, orchestrator
    from classeval.pipeline.llm import OllamaClient
    from classeval.schema import Report

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"device: GPU {props.name}, {props.total_memory / 2**30:.0f} GB", flush=True)
    else:
        print("device: CPU only (works, but a full class takes much longer)", flush=True)
    try:
        settings = load_settings("config.yaml", env_file=".env")
    except (ValueError, FileNotFoundError) as exc:
        return fail(f"settings: {exc} -> run scripts/install.py first")
    client = OllamaClient.from_settings(settings)
    try:
        client.chat_json("Answer with JSON.", 'Reply {"ok": true}.', {"type": "object", "properties": {"ok": {"type": "boolean"}}})
    except Exception as exc:  # noqa: BLE001 - report the class, then stop
        return fail(f"the local language model did not answer ({type(exc).__name__}); start Ollama and pull {settings.models.llm}")

    settings.limits.min_duration_s = 30.0  # the clip is 70 s; real recordings need 10 minutes
    with tempfile.TemporaryDirectory(prefix="classeval_selftest_") as tmp:
        tmp_path = Path(tmp)
        clip = tmp_path / "selftest.mp4"
        kind = make_test_clip(clip)
        print(f"clip: {ingest.probe(clip).duration_s:.0f} s with {kind} audio", flush=True)
        started = time.monotonic()
        try:
            rep = orchestrator.execute_pipeline(clip, tmp_path / "job", settings, job_id="selftest")
        except Exception as exc:  # noqa: BLE001
            return fail(f"the pipeline stopped with {type(exc).__name__}")
        leftovers = [p for p in (tmp_path / "job").rglob("*") if p.suffix.lower() in {".jpg", ".wav", ".npz"}]
        if leftovers:
            return fail("frames or audio were left behind")
        Report.model_validate(json.loads(rep.model_dump_json()))
        stages = ", ".join(f"{k} {v:.0f}s" for k, v in rep.timings_s.items())
        print(f"pipeline finished in {time.monotonic() - started:.0f} s ({stages})", flush=True)
        print(f"result: grade {rep.grade}, language {rep.language}, rubric coverage {rep.coverage:.0%}", flush=True)
    print("SELFTEST OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
