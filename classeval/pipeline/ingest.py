import json
import os
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from classeval.config import Settings
from classeval.schema import FrameRef, MediaInfo

IngestErrorCode = Literal["unreadable", "no_streams", "too_long", "too_short_zero", "timeout"]


def _ensure_ffmpeg_path() -> None:
    """Put the folder named by FFMPEG_DIR (optional) in front of PATH, for a machine where ffmpeg is not on PATH."""
    extra = os.environ.get("FFMPEG_DIR")
    if extra and Path(extra).is_dir() and extra not in os.environ.get("PATH", ""):
        os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")


class IngestError(Exception):
    """Raised when probing or preparing media fails."""

    def __init__(self, code: IngestErrorCode, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class Prepared(BaseModel):
    info: MediaInfo
    wav: Path | None
    frames: list[FrameRef]
    tiles: list[FrameRef] = Field(default_factory=list)  # the tutor's top-right tile at native resolution


def probe(path: Path) -> MediaInfo:
    """Probe media file streams and container information via ffprobe."""
    _ensure_ffmpeg_path()
    path = Path(path).resolve()
    if not path.exists():
        raise IngestError(code="unreadable", message=f"File {path} does not exist")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]

    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30.0)
    except subprocess.TimeoutExpired as exc:
        raise IngestError(code="timeout", message="ffprobe timed out") from exc

    if res.returncode != 0:
        raise IngestError(code="unreadable", message=f"ffprobe failed: {res.stderr.strip()}")

    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        raise IngestError(code="unreadable", message="Failed to parse ffprobe JSON output") from exc

    streams = data.get("streams", [])
    if not streams:
        raise IngestError(code="no_streams", message="No streams found in media")

    fmt = data.get("format", {})
    size_bytes = int(fmt.get("size", path.stat().st_size))
    container = fmt.get("format_name", path.suffix.lstrip("."))

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    has_video = len(video_streams) > 0
    has_audio = len(audio_streams) > 0

    duration_s = 0.0
    if fmt.get("duration"):
        try:
            duration_s = float(fmt["duration"])
        except ValueError:
            pass

    if duration_s <= 0.0:
        for s in streams:
            if s.get("duration"):
                try:
                    duration_s = max(duration_s, float(s["duration"]))
                except ValueError:
                    pass

    width = None
    height = None
    fps = None
    video_kbps = None

    if has_video:
        vs = video_streams[0]
        width = vs.get("width")
        height = vs.get("height")
        r_frame_rate = vs.get("r_frame_rate", "")
        if "/" in r_frame_rate:
            num, den = r_frame_rate.split("/", 1)
            try:
                den_f = float(den)
                if den_f > 0:
                    fps = round(float(num) / den_f, 2)
            except ValueError:
                pass
        bit_rate = vs.get("bit_rate") or fmt.get("bit_rate")
        if bit_rate:
            try:
                video_kbps = round(float(bit_rate) / 1000.0, 1)
            except ValueError:
                pass

    return MediaInfo(
        duration_s=round(duration_s, 3),
        has_audio=has_audio,
        has_video=has_video,
        width=width,
        height=height,
        fps=fps,
        video_kbps=video_kbps,
        container=container,
        size_bytes=size_bytes,
    )


def validate(info: MediaInfo, settings: Settings) -> list[str]:
    """Validate media constraints; returns warnings or raises IngestError."""
    if info.duration_s <= 0.0:
        raise IngestError(code="too_short_zero", message="Duration is zero or negative")

    if info.duration_s > settings.limits.max_duration_s:
        raise IngestError(
            code="too_long",
            message=f"Duration {info.duration_s}s exceeds maximum {settings.limits.max_duration_s}s",
        )

    warnings: list[str] = []
    if not info.has_video:
        warnings.append("no_video")
    if not info.has_audio:
        warnings.append("no_audio")
    if info.duration_s < settings.limits.min_duration_s:
        warnings.append("short")

    return warnings


def prepare(src: Path, job_dir: Path, settings: Settings) -> Prepared:
    """Extract 16kHz mono audio and sampled frame sequence into job_dir."""
    _ensure_ffmpeg_path()
    src = Path(src).resolve()
    job_dir = Path(job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)

    info = probe(src)
    validate(info, settings)

    timeout = 0.5 * info.duration_s + 60.0

    wav_path: Path | None = None
    if info.has_audio:
        target_wav = job_dir / "audio_16k.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(target_wav),
        ]
        try:
            res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
            if res.returncode != 0:
                raise IngestError(
                    code="unreadable", message=f"Audio extraction failed: {res.stderr.strip()}"
                )
            wav_path = target_wav
        except subprocess.TimeoutExpired as exc:
            raise IngestError(
                code="timeout", message="Audio extraction timed out"
            ) from exc

    frames: list[FrameRef] = []
    tiles: list[FrameRef] = []
    if info.has_video:
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        tiles_dir = job_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        fps = settings.sampling.frame_fps
        width = settings.sampling.frame_width
        cfg = settings.screen_state
        x, y, w, h = cfg.thumb_region
        tile_filter = f"fps=1/{cfg.tile_every_s:g},crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}"

        # One decode of the video feeds both outputs: the sampled whole frames and the tutor's tile.
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-an",
            "-vf",
            f"fps={fps},scale={width}:-2",
            "-q:v",
            "3",
            str(frames_dir / "frame_%06d.jpg"),
            "-an",
            "-vf",
            tile_filter,
            "-q:v",
            "3",
            str(tiles_dir / "tile_%05d.jpg"),
        ]
        try:
            res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
            if res.returncode != 0:
                raise IngestError(
                    code="unreadable", message=f"Frame extraction failed: {res.stderr.strip()}"
                )
        except subprocess.TimeoutExpired as exc:
            raise IngestError(
                code="timeout", message="Frame extraction timed out"
            ) from exc

        jpg_files = sorted(frames_dir.glob("frame_*.jpg"))
        for idx, f_path in enumerate(jpg_files):
            t = float(idx) / fps
            frames.append(FrameRef(t=round(t, 3), path=f_path))
        for idx, t_path in enumerate(sorted(tiles_dir.glob("tile_*.jpg"))):
            tiles.append(FrameRef(t=round(idx * cfg.tile_every_s, 3), path=t_path))

    return Prepared(info=info, wav=wav_path, frames=frames, tiles=tiles)
