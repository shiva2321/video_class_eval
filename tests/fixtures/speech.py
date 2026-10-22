import subprocess
import tempfile
import uuid
import wave
from pathlib import Path


def make_speech_wav(text: str, path: Path, voice: str = "David") -> Path:
    """Generate speech WAV using Windows SAPI (System.Speech) and convert to 16kHz mono."""
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
        import shutil

        ffmpeg_bin = "ffmpeg"
        if not shutil.which("ffmpeg"):
            candidate = Path(r"F:\tools\ffmpeg\bin\ffmpeg.exe")
            if candidate.exists():
                ffmpeg_bin = str(candidate)

        convert_cmd = [
            ffmpeg_bin,
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
