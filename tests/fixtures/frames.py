from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

FrameKind = Literal["camera_off_card", "branded_head", "board_grid", "plain_dark", "noise"]


def draw_frame(
    kind: FrameKind,
    path: Path,
    size: tuple[int, int] = (1280, 720),
    scale: float = 0.5,
) -> Path:
    """Generate a synthetic video frame of a specific semantic category."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size

    if kind == "camera_off_card":
        # Flat grey card: mean luma 20-40, std < 8
        arr = np.full((h, w, 3), 30, dtype=np.uint8)
        # Small noise std ~ 2
        noise = np.random.normal(0, 2, (h, w, 3)).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 22, 38).astype(np.uint8)
        img = Image.fromarray(arr)

    elif kind == "branded_head":
        # Navy blue background + 8x8 logo template
        arr = np.full((h, w, 3), (15, 25, 45), dtype=np.uint8)
        img = Image.fromarray(arr)
        logo_path = Path("assets/8x8_logo_left8.png")
        if logo_path.exists():
            logo = Image.open(logo_path).convert("RGB")
            lw = int(logo.width * scale)
            lh = int(logo.height * scale)
            if lw > 0 and lh > 0 and lw <= w and lh <= h:
                logo_resized = logo.resize((lw, lh), Image.Resampling.LANCZOS)
                img.paste(logo_resized, (20, 20))

    elif kind == "board_grid":
        # 8x8 checkerboard chess board
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        board_size = min(h - 40, w - 40)
        sq_size = board_size // 8
        top = (h - sq_size * 8) // 2
        left = (w - sq_size * 8) // 2

        # Green felt and cream squares
        color_dark = (118, 150, 86)
        color_light = (238, 238, 210)

        for r in range(8):
            for c in range(8):
                color = color_light if (r + c) % 2 == 0 else color_dark
                y1 = top + r * sq_size
                y2 = y1 + sq_size
                x1 = left + c * sq_size
                x2 = x1 + sq_size
                arr[y1:y2, x1:x2] = color
        img = Image.fromarray(arr)

    elif kind == "plain_dark":
        arr = np.full((h, w, 3), 10, dtype=np.uint8)
        img = Image.fromarray(arr)

    elif kind == "noise":
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        img = Image.fromarray(arr)

    else:
        raise ValueError(f"Unknown frame kind: {kind}")

    img.save(str(path))
    return path
