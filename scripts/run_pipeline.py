import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Offline mode
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from classeval.config import load_settings
from classeval.guard import limit_gpu_memory, start_memory_watchdog
from classeval.pipeline import orchestrator
from classeval.schema import Transcript


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full class evaluation pipeline on a single file.")
    parser.add_argument("--input", required=True, type=Path, help="Path to input media file")
    parser.add_argument("--out", required=True, type=Path, help="Directory to store outputs and reports")
    parser.add_argument("--fast", action="store_true", help="Fast execution mode")
    args = parser.parse_args()

    # A runaway stage must fail fast, never freeze the PC.
    limit_gpu_memory()
    start_memory_watchdog()

    input_path = args.input.resolve()
    if not input_path.exists():
        sys.exit(1)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        settings = load_settings("config.yaml", env_file=".env")
    except Exception:  # noqa: BLE001
        settings = load_settings("config.yaml", env_file=None)

    t_start = time.monotonic()
    try:
        rep = orchestrator.execute_pipeline(
            input_path=input_path,
            job_dir=out_dir,
            settings=settings,
            job_id=out_dir.name or "cli_run",
        )
    except Exception as exc:  # noqa: BLE001
        # Class name only: messages can contain recording text. Re-run is cheap, finished stages are cached.
        print(f"pipeline failed: {type(exc).__name__}", file=sys.stderr)
        sys.exit(1)

    wall_s = round(time.monotonic() - t_start, 2)

    # Read segment count from cached ASR output
    n_segments = 0
    asr_cache = out_dir / "work" / "stages" / "asr.done.json"
    if asr_cache.exists():
        try:
            tr = Transcript.model_validate_json(asr_cache.read_text(encoding="utf-8"))
            n_segments = len(tr.segments)
        except (ValueError, KeyError, OSError):
            pass

    score_val = f"{rep.overall_score:.1f}" if rep.overall_score is not None else "None"
    stage_names = ["ingest", "asr", "diarize", "audio", "video", "understand", "score", "report"]
    stage_durations = ":".join(f"{rep.timings_s.get(st, 0.0):.1f}" for st in stage_names)

    # Exactly one line of numbers/aggregates only:
    # grade, overall_score, coverage, confidence, wall seconds, seconds per stage, number of transcript segments
    print(f"{rep.grade}\t{score_val}\t{rep.coverage:.2f}\t{rep.confidence:.2f}\t{wall_s:.2f}\t{stage_durations}\t{n_segments}")


if __name__ == "__main__":
    main()
