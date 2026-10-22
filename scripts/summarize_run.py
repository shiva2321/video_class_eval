"""Print a NUMBERS-ONLY diagnostic summary of one pipeline run directory (written by scripts/run_pipeline.py).

Never prints transcript text, quotes, notes, names or file names: safe to paste into a chat.
Usage: python scripts/summarize_run.py <run_dir>
"""

import json
import sys
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(run_dir: Path) -> int:
    stages = run_dir / "work" / "stages"
    report = _load(run_dir / "report.json")
    if report is None:
        print("no report.json in run directory")
        return 1

    asr = _load(stages / "asr.done.json")
    if asr:
        segs = asr["segments"]
        words = sum(len(s["text"].split()) for s in segs)
        lps = [s["avg_logprob"] for s in segs if s.get("avg_logprob") is not None]
        mean_lp = sum(lps) / len(lps) if lps else float("nan")
        print(f"asr: language={asr['language']} conf={asr['language_confidence']:.2f} segments={len(segs)} "
              f"words={words} mean_logprob={mean_lp:.2f}")

    diar = _load(stages / "diarize.done.json")
    if diar:
        d = diar["diarization"]
        print(f"diarization: method={d['method']} speakers={d['n_speakers']} tutor_conf={d['tutor_confidence']:.2f}")

    a = _load(stages / "audio.done.json")
    if a:
        print(f"audio: speech_ratio={a['speech_ratio']:.2f} tutor_share={a['tutor_talk_share']:.2f} "
              f"student_wpm={a['student_words_per_min']:.1f} dead_air_pm={a['dead_air_per_min']:.2f} "
              f"latency_s={a['median_response_latency_s']} dropouts={len(a['dropouts'])} lufs={a['loudness_lufs']:.1f} "
              f"clip={a['clipping_ratio']:.4f} snr={a['snr_db_est']:.1f} pitch_std={a['pitch_std_semitones_tutor']} "
              f"energy_cv={a['energy_cv_tutor']}")

    v = _load(stages / "video.done.json")
    if v:
        counts: dict[str, int] = {}
        for st in v["states"]:
            counts[st["state"]] = counts.get(st["state"], 0) + 1
        print(f"video: frames={v['n_frames']} states={counts} branded_tutor_speaking={v['branded_ratio_tutor_speaking']} "
              f"board={v['board_share_ratio']:.2f} lesson_video={v['lesson_video_ratio']:.2f} "
              f"cam_off_total={v['camera_off_total_ratio']:.2f} cam_off_tutor_s={v['camera_off_while_tutor_speaks_s']:.0f} "
              f"first5={v['first5_branded']} last5={v['last5_branded']} luma={v['talking_head_luma_mean']}")

    print(f"report: grade={report['grade']} score={report['overall_score']} coverage={report['coverage']} "
          f"confidence={report['confidence']} flags={report['technical_flags']} not_assessed={report['not_assessed']} "
          f"strengths={len(report['strengths'])} improvements={len(report['improvements'])} degraded={report['degraded']}")
    print(f"timings_s: {report['timings_s']}")
    for dim in report["dimensions"]:
        print(f"  {dim['id']:3s} {dim['name'][:36]:36s} score={dim['score']} weight={dim['weight']} evidence_items={len(dim['evidence'])}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
