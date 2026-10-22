import os
from pathlib import Path

# A machine where ffmpeg is not on PATH can name its folder in FFMPEG_DIR.
extra = os.environ.get("FFMPEG_DIR")
if extra and Path(extra).is_dir() and extra not in os.environ.get("PATH", ""):
    os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")
