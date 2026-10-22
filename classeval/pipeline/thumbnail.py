"""The tutor's video tile: while a board or video is shared, Zoom shows the tutor in a small top-right tile.

The full-frame checks cannot see that tile, so this module samples it at native resolution and looks for the
8x8 backdrop in it. Crops live only in a scratch directory that is removed before returning.
"""

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from classeval.config import Settings
from classeval.pipeline.ingest import _ensure_ffmpeg_path

logger = logging.getLogger("classeval.thumbnail")

LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "8x8_logo_left8.png"
FFMPEG_TIMEOUT_S = 3600.0


def best_logo_score(gray: np.ndarray, template: np.ndarray, scales: list[float]) -> float:
    """Highest normalised correlation of the logo template at any of the scales."""
    best = 0.0
    for scale in scales:
        w, h = int(template.shape[1] * scale), int(template.shape[0] * scale)
        if 8 < w < gray.shape[1] and 8 < h < gray.shape[0]:
            resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
            best = max(best, float(cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED).max()))
    return best


def _extract_crops(video: Path, out_dir: Path, settings: Settings) -> None:
    _ensure_ffmpeg_path()
    cfg = settings.screen_state
    x, y, w, h = cfg.thumb_region
    vf = f"fps=1/{cfg.thumb_every_s:g},crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}"
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-an", "-vf", vf, "-q:v", "3", str(out_dir / "t_%05d.jpg")]
    subprocess.run(cmd, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S)


def branded_share(video: Path, work_dir: Path, settings: Settings) -> tuple[int, float | None]:
    """(number of sampled moments, share of them where the 8x8 backdrop shows in the top-right tile)."""
    template_bgr = cv2.imread(str(LOGO_PATH))
    if template_bgr is None:
        return 0, None
    template = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    cfg = settings.screen_state
    crops_dir = work_dir / "thumbs"
    shutil.rmtree(crops_dir, ignore_errors=True)
    crops_dir.mkdir(parents=True)
    n = hits = 0
    try:
        _extract_crops(video, crops_dir, settings)
        for crop in sorted(crops_dir.glob("t_*.jpg")):  # one crop in memory at a time
            image = cv2.imread(str(crop), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            n += 1
            hits += best_logo_score(image, template, cfg.thumb_scales) >= cfg.thumb_logo_threshold
    except (subprocess.SubprocessError, OSError) as exc:  # best effort: the full-frame checks still stand
        logger.warning("tile sampling failed: %s", type(exc).__name__)
        return 0, None
    finally:
        shutil.rmtree(crops_dir, ignore_errors=True)
    return n, (hits / n if n else None)
