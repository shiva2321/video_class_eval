import subprocess
import sys
from pathlib import Path


def _get_ffmpeg() -> str:
    return "ffmpeg"


def _get_ffprobe() -> str:
    return "ffprobe"


def probe_duration(path: Path) -> float:
    """Probe media duration in seconds via ffprobe."""
    ffprobe_bin = _get_ffprobe()
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr}")
    return float(res.stdout.strip())


def make_degraded(input_path: Path, out_dir: Path, kind: str) -> Path:
    """Create a degraded or changed copy: mute5, black, trunc5min, scale720, scale480 or first20."""
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{input_path.stem}_{kind}.mp4"
    ffmpeg_bin = _get_ffmpeg()

    if kind == "mute5":
        dur = probe_duration(input_path)
        if dur <= 300.0:
            af_filter = "volume=0"
        else:
            t0 = (dur - 300.0) / 2.0
            t1 = t0 + 300.0
            af_filter = f"volume=enable='between(t,{t0:.2f},{t1:.2f})':volume=0"

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            "copy",
            "-af",
            af_filter,
            "-c:a",
            "aac",
            str(out_path),
        ]

    elif kind == "black":
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(input_path),
            "-vf",
            "drawbox=color=black@1.0:t=fill",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(out_path),
        ]

    elif kind == "trunc5min":
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(input_path),
            "-t",
            "300",
            "-c",
            "copy",
            str(out_path),
        ]

    elif kind in ("scale720", "scale480"):
        height = 720 if kind == "scale720" else 480
        cmd = [
            ffmpeg_bin, "-y", "-i", str(input_path), "-vf", f"scale=-2:{height}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p", "-c:a", "copy", str(out_path),
        ]  # fmt: skip

    elif kind == "first20":
        cmd = [ffmpeg_bin, "-y", "-i", str(input_path), "-t", "1200", "-c", "copy", str(out_path)]

    else:
        raise ValueError(f"Unknown degradation kind: {kind}")

    timeout_s = 180.0
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    if res.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"FFmpeg degradation failed for kind '{kind}': {res.stderr}")

    return out_path


def make_long(inputs: list[Path], out_dir: Path, name: str = "long_concat") -> Path:
    """Join recordings with identical codecs into one long recording (tests length scaling without new data)."""
    import tempfile

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "list.txt"
        lines = [f"file '{Path(p).resolve().as_posix()}'" for p in inputs]
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cmd = [_get_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800.0, check=False)
    if res.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"FFmpeg concat failed: {res.stderr}")
    return out_path


def main() -> None:
    if len(sys.argv) >= 5 and sys.argv[1] == "concat":
        try:
            print(f"OK {make_long([Path(p) for p in sys.argv[3:]], Path(sys.argv[2])).name}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {exc}")
            sys.exit(1)
        return

    if len(sys.argv) < 4:
        print("Usage: python scripts/make_degraded.py <input> <out_dir> <kind>")
        print("       python scripts/make_degraded.py concat <out_dir> <recording> <recording> ...")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    out_directory = Path(sys.argv[2])
    degradation_kind = sys.argv[3]

    try:
        res_file = make_degraded(input_file, out_directory, degradation_kind)
        print(f"OK {res_file.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
