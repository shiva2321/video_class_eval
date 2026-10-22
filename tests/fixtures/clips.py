import subprocess
from pathlib import Path


def make_clip(
    path: Path,
    seconds: float,
    with_audio: bool = True,
    with_video: bool = True,
    size: tuple[int, int] = (1280, 720),
    extra_args: list[str] | None = None,
) -> Path:
    """Generate a synthetic test clip using ffmpeg lavfi filters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y"]
    w, h = size

    if with_video:
        cmd.extend(["-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=25"])
    if with_audio:
        cmd.extend(["-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=16000"])

    cmd.extend(["-t", str(seconds)])

    if with_video and not with_audio:
        cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    elif with_audio and not with_video:
        if path.suffix.lower() == ".wav":
            cmd.extend(["-c:a", "pcm_s16le"])
        else:
            cmd.extend(["-c:a", "aac"])
    elif with_video and with_audio:
        cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"])

    if extra_args:
        cmd.extend(extra_args)

    cmd.append(str(path))

    timeout = max(30.0, float(seconds) * 2.0 + 10.0)
    res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (code {res.returncode}): {res.stderr}")

    return path
