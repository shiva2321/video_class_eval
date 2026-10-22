import shutil
import subprocess
import sys
import tempfile
import uuid
import wave
from pathlib import Path


def speech_backend() -> str | None:
    """The text-to-speech engine of this machine: Windows SAPI, macOS `say` or Linux espeak-ng / espeak."""
    if sys.platform == "win32":
        return "sapi"
    if sys.platform == "darwin" and shutil.which("say"):
        return "say"
    for name in ("espeak-ng", "espeak"):
        if shutil.which(name):
            return name
    return None


def _convert_to_16k_mono(src: Path, dst: Path) -> None:
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(dst)],
        check=False, capture_output=True, text=True, timeout=30.0,
    )  # fmt: skip
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {res.stderr.strip()}")


def _speak_with_command(backend: str, text: str, path: Path) -> Path:
    suffix = ".aiff" if backend == "say" else ".wav"
    tmp = Path(tempfile.gettempdir()) / f"synth_{uuid.uuid4().hex}{suffix}"
    cmd = ["say", "-o", str(tmp), text] if backend == "say" else [backend, "-w", str(tmp), text]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0, check=False)
        if res.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"Speech synthesis with {backend} failed: {res.stderr.strip()}")
        _convert_to_16k_mono(tmp, path)
        return path
    finally:
        tmp.unlink(missing_ok=True)


def make_speech_wav(text: str, path: Path, voice: str = "David") -> Path:
    """Generate speech as a 16 kHz mono WAV with this machine's text-to-speech engine (the test is skipped without one)."""
    backend = speech_backend()
    if backend is None:
        import pytest

        pytest.skip("no text-to-speech engine here (Windows SAPI, macOS say, or espeak-ng)")
    if backend != "sapi":
        return _speak_with_command(backend, text, Path(path).resolve())
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    uid = uuid.uuid4().hex
    tmp_wav = Path(tempfile.gettempdir()) / f"synth_{uid}.wav"
    tmp_ps1 = Path(tempfile.gettempdir()) / f"synth_{uid}.ps1"

    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$found = $null
foreach ($v in $synth.GetInstalledVoices()) {{
    if ($v.VoiceInfo.Name -like "*{voice}*") {{
        $found = $v.VoiceInfo.Name
        break
    }}
}}
if ($found) {{
    $synth.SelectVoice($found)
}}
$synth.SetOutputToWaveFile("{tmp_wav!s}")
$synth.Speak("{text}")
$synth.Dispose()
"""
    tmp_ps1.write_text(ps_script, encoding="utf-8")

    try:
        res = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(tmp_ps1),
            ],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        if res.returncode != 0 or not tmp_wav.exists() or tmp_wav.stat().st_size == 0:
            raise RuntimeError(f"Speech synthesis failed: {res.stderr.strip()}")

        # Standardize to 16kHz mono PCM
        convert_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp_wav),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(path),
        ]
        res_conv = subprocess.run(
            convert_cmd, check=False, capture_output=True, text=True, timeout=30.0
        )
        if res_conv.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {res_conv.stderr.strip()}")

        return path
    finally:
        tmp_wav.unlink(missing_ok=True)
        tmp_ps1.unlink(missing_ok=True)


def concat_wavs(paths: list[Path], gap_s: float, out: Path) -> Path:
    """Concatenate multiple WAV files with an optional silence gap (in seconds)."""
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not paths:
        raise ValueError("No WAV files provided to concat_wavs")

    with wave.open(str(paths[0]), "rb") as wf_first:
        params = wf_first.getparams()
        nchannels = wf_first.getnchannels()
        sampwidth = wf_first.getsampwidth()
        framerate = wf_first.getframerate()

    silence_frames = int(framerate * gap_s)
    silence_bytes = b"\x00" * (silence_frames * nchannels * sampwidth)

    with wave.open(str(out), "wb") as wf_out:
        wf_out.setparams(params)
        for idx, p in enumerate(paths):
            if idx > 0 and silence_bytes:
                wf_out.writeframes(silence_bytes)
            with wave.open(str(p), "rb") as wf_in:
                wf_out.writeframes(wf_in.readframes(wf_in.getnframes()))

    return out
