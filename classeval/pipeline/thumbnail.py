"""The tutor's video tile: while a board or video is shared, Zoom shows the tutor in a small top-right tile.

The full-frame checks cannot see that tile, so this module samples it at native resolution and looks for the
8x8 backdrop in it. Crops live only in a scratch directory that is removed before returning.
"""

from pathlib import Path

import cv2
import numpy as np

from classeval.config import Settings
from classeval.schema import FrameRef

LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "8x8_logo_left8.png"


def best_logo_score(gray: np.ndarray, template: np.ndarray, scales: list[float]) -> float:
    """Highest normalised correlation of the logo template at any of the scales."""
    best = 0.0
    for scale in scales:
        w, h = int(template.shape[1] * scale), int(template.shape[0] * scale)
        if 8 < w < gray.shape[1] and 8 < h < gray.shape[0]:
            resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
            best = max(best, float(cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED).max()))
    return best


def branded_share(tiles: list[FrameRef], settings: Settings) -> tuple[int, float | None]:
    """(number of sampled moments, share of them where the 8x8 backdrop shows in the tutor's tile).

    The tiles were cut out during ingest; every `thumb_every_s` seconds one is looked at, one image in memory at a time."""
    template_bgr = cv2.imread(str(LOGO_PATH))
    if template_bgr is None or not tiles:
        return 0, None
    template = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    cfg = settings.screen_state
    stride = max(1, round(cfg.thumb_every_s / cfg.tile_every_s))
    n = hits = 0
    for tile in tiles[::stride]:
        image = cv2.imread(str(tile.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        n += 1
        hits += best_logo_score(image, template, cfg.thumb_scales) >= cfg.thumb_logo_threshold
    return n, (hits / n if n else None)
