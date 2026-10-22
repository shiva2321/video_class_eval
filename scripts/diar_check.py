import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from classeval.config import load_settings
from classeval.eval.vtt import parse_vtt
from classeval.pipeline import diarize, ingest


def run_diar_check(input_path: Path, vtt_path: Path) -> dict[str, float | int | str]:
    input_path = Path(input_path).resolve()
    vtt_path = Path(vtt_path).resolve()
    settings = load_settings("config.yaml", env_file=".env")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_job = Path(tmp_dir)
        if input_path.suffix.lower() == ".wav":
            wav_path = input_path
        else:
            prepared = ingest.prepare(input_path, tmp_job, settings)
            wav_path = prepared.wav

        if not wav_path or not wav_path.exists():
            raise RuntimeError("No audio available for diarization")

        diar_result = diarize.diarize(wav_path, settings)

        # Compute our tutor talk share
        our_spk_times: dict[str, float] = defaultdict(float)
        for t in diar_result.turns:
            our_spk_times[t.speaker] += max(0.0, float(t.end - t.start))

        our_total = sum(our_spk_times.values())
        our_tutor_time = our_spk_times.get(diar_result.tutor or "", 0.0)
        our_tutor_share = round(our_tutor_time / our_total, 3) if our_total > 0 else 0.0

        # Compute VTT tutor share (dominant speaker in VTT)
        vtt_cues = parse_vtt(vtt_path)
        vtt_spk_times: dict[str, float] = defaultdict(float)
        for start, end, spk, _ in vtt_cues:
            if spk:
                vtt_spk_times[spk] += max(0.0, float(end - start))

        vtt_total = sum(vtt_spk_times.values())
        if vtt_spk_times:
            vtt_top_speaker = max(vtt_spk_times.items(), key=lambda x: x[1])[0]
            vtt_tutor_time = vtt_spk_times[vtt_top_speaker]
            vtt_tutor_share = round(vtt_tutor_time / vtt_total, 3) if vtt_total > 0 else 0.0
        else:
            vtt_tutor_share = 0.0

        diff = round(abs(our_tutor_share - vtt_tutor_share), 3)

        return {
            "our_tutor_share": our_tutor_share,
            "vtt_tutor_share": vtt_tutor_share,
            "absolute_difference": diff,
            "n_speakers": diar_result.n_speakers,
            "method": diar_result.method,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diarization Evaluation Check")
    parser.add_argument("--input", required=True, help="Input media file (MP4 or WAV)")
    parser.add_argument("--vtt", required=True, help="Reference VTT file")
    args = parser.parse_args()

    try:
        metrics = run_diar_check(Path(args.input), Path(args.vtt))
        print(json.dumps(metrics, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
