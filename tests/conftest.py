import os
from pathlib import Path

# Ensure ffmpeg bin is in PATH for all test runs
ffmpeg_bin = Path(r"F:\tools\ffmpeg\bin")
if ffmpeg_bin.exists() and str(ffmpeg_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(ffmpeg_bin) + os.pathsep + os.environ.get("PATH", "")
