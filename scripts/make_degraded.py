import shutil
import subprocess
import sys
from pathlib import Path


def _get_ffmpeg() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    candidate = Path(r"F:\tools\ffmpeg\bin\ffmpeg.exe")
    if candidate.exists():
        return str(candidate)
    return "ffmpeg"


def _get_ffprobe() -> str:
    if shutil.which("ffprobe"):
        return "ffprobe"
    candidate = Path(r"F:\tools\ffmpeg\bin\ffprobe.exe")
    if candidate.exists():
        return str(candidate)
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
    """Create a degraded media file (mute5, black, or trunc5min)."""
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

    else:
        raise ValueError(f"Unknown degradation kind: {kind}")

    timeout_s = 180.0
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    if res.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"FFmpeg degradation failed for kind '{kind}': {res.stderr}")

    return out_path


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: python scripts/make_degraded.py <input> <out_dir> <kind>")
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
