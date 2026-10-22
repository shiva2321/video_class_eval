import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from classeval.config import load_settings
from classeval.eval.vtt import parse_vtt, wer
from classeval.pipeline import asr, ingest


def run_benchmark(input_path: Path, vtt_path: Path | None = None) -> dict[str, float | int | str]:
    input_path = Path(input_path).resolve()
    settings = load_settings("config.yaml", env_file=".env")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_job = Path(tmp_dir)
        if input_path.suffix.lower() == ".wav":
            wav_path = input_path
            info = ingest.probe(input_path)
        else:
            prepared = ingest.prepare(input_path, tmp_job, settings)
            wav_path = prepared.wav
            info = prepared.info

        if not wav_path or not wav_path.exists():
            raise RuntimeError("No audio available to benchmark")

        audio_s = info.duration_s
        audio_minutes = round(audio_s / 60.0, 2)

        start_time = time.monotonic()
        transcript = asr.transcribe(wav_path, settings)
        wall_s = round(time.monotonic() - start_time, 2)
        rtf = round(wall_s / audio_s, 3) if audio_s > 0 else 0.0

        n_segments = len(transcript.segments)
        logprobs = [s.avg_logprob for s in transcript.segments if s.avg_logprob is not None]
        mean_logprob = round(sum(logprobs) / len(logprobs), 3) if logprobs else 0.0

        metrics: dict[str, float | int | str] = {
            "audio_minutes": audio_minutes,
            "wall_seconds": wall_s,
            "real_time_factor": rtf,
            "n_segments": n_segments,
            "mean_avg_logprob": mean_logprob,
            "language": transcript.language,
            "language_confidence": transcript.language_confidence,
        }

        if vtt_path and Path(vtt_path).exists():
            vtt_cues = parse_vtt(Path(vtt_path))
            ref_text = " ".join(c[3] for c in vtt_cues)
            hyp_text = " ".join(s.text for s in transcript.segments)
            metrics["wer"] = round(wer(ref_text, hyp_text), 3)

        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR Benchmark Runner")
    parser.add_argument("--input", required=True, help="Input media file (MP4 or WAV)")
    parser.add_argument("--vtt", required=False, default=None, help="Optional reference VTT file")
    args = parser.parse_args()

    try:
        metrics = run_benchmark(Path(args.input), Path(args.vtt) if args.vtt else None)
        print(json.dumps(metrics, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
