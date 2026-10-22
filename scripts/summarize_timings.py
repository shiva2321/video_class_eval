"""Write the per-stage processing times of full pipeline runs to a JSON file (numbers only).

Usage: python scripts/summarize_timings.py <out.json> <run_dir> [<run_dir> ...]
Each run_dir holds the report.json of a run that executed every stage (no cached stages).
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from classeval.schema import Report


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    out, runs = Path(argv[1]), [Path(p) for p in argv[2:]]
    rows = []
    for run in runs:
        rep = Report.model_validate_json((run / "report.json").read_text(encoding="utf-8"))
        total = round(sum(rep.timings_s.values()), 1)
        duration = rep.media.duration_s
        rows.append(
            {
                "recording_minutes": round(duration / 60.0, 1),
                "stages_s": rep.timings_s,
                "total_s": total,
                "minutes_of_recording_per_minute_of_processing": round(duration / total, 1) if total else None,
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    for row in rows:
        print(row["recording_minutes"], row["total_s"], row["minutes_of_recording_per_minute_of_processing"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
