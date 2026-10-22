"""Screen-state classification timeline.

Identifies screen states:
- CAMERA_OFF_CARD: flat grey background card
- BRANDED_TALKING_HEAD: video feed showing the 8x8 chess logo
- BOARD_SHARE, LESSON_VIDEO_SHARE, OTHER_TALKING_HEAD, OTHER_SHARE: zero-shot CLIP
- UNKNOWN: noise or low-confidence visual frames

Limitation: Camera-off inside a small thumbnail strip is out of scope.
"""

import gc
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from classeval.config import Settings
from classeval.schema import FrameRef, FrameState, ScreenState


def _is_flat_grey_card(bgr: np.ndarray, gray: np.ndarray, settings: Settings) -> bool:
    """Check if the frame is a uniform flat-grey camera-off card."""
    cfg = settings.screen_state
    lo, hi = cfg.flat_card_luma_range
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Flat grey must be neutral in color (not saturated/colored)
    color_diff = float(np.mean(np.abs(bgr[:, :, 0].astype(float) - bgr[:, :, 1].astype(float)))) + float(
        np.mean(np.abs(bgr[:, :, 1].astype(float) - bgr[:, :, 2].astype(float)))
    )
    if color_diff > 8.0:
        return False

    # Mean luma between lo and hi, std < max
    if lo <= mean_val <= hi and std_val <= cfg.flat_card_std_max:
        # Check pixel share within bounds
        in_range = (gray >= lo - 5) & (gray <= hi + 5)
        share = float(np.mean(in_range))
        return share >= cfg.flat_card_pixel_share
    return False


def _match_logo(gray: np.ndarray, template_gray: np.ndarray, settings: Settings) -> float:
    """Multi-scale template matching of 8x8 logo template."""
    cfg = settings.screen_state
    h_f, w_f = gray.shape[:2]
    h_t, w_t = template_gray.shape[:2]
    max_score = 0.0

    for scale in cfg.logo_scales:
        new_w = int(w_t * scale)
        new_h = int(h_t * scale)
        if 0 < new_w < w_f and 0 < new_h < h_f:
            resized_tpl = cv2.resize(
                template_gray, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            res = cv2.matchTemplate(gray, resized_tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > max_score:
                max_score = float(val)

    return max_score


def classify_frames(frames: list[FrameRef], settings: Settings) -> list[FrameState]:
    """Classify video frames into screen states using heuristic rules and zero-shot CLIP."""
    if not frames:
        return []

    states: list[FrameState | None] = [None] * len(frames)
    clip_indices: list[int] = []

    # Load 8x8 logo template
    logo_path = Path(__file__).resolve().parents[2] / "assets" / "8x8_logo_left8.png"
    template_gray = None
    if logo_path.exists():
        template_bgr = cv2.imread(str(logo_path))
        if template_bgr is not None:
            template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # Pass 1 & 2: Flat grey card and logo template matching
    loaded_images: dict[int, np.ndarray] = {}
    for idx, fr in enumerate(frames):
        bgr = cv2.imread(str(fr.path))
        if bgr is None:
            states[idx] = FrameState(t=fr.t, state="UNKNOWN", confidence=0.0)
            continue

        loaded_images[idx] = bgr
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1. Flat grey card
        if _is_flat_grey_card(bgr, gray, settings):
            states[idx] = FrameState(t=fr.t, state="CAMERA_OFF_CARD", confidence=0.95)
            continue

        # 2. Branded talking head
        if template_gray is not None:
            score = _match_logo(gray, template_gray, settings)
            if score >= settings.screen_state.logo_match_threshold:
                states[idx] = FrameState(
                    t=fr.t, state="BRANDED_TALKING_HEAD", confidence=round(score, 3)
                )
                continue

        # 3. High-entropy noise detection
        if float(np.mean(np.abs(np.diff(gray.astype(float), axis=1)))) > 45.0:
            states[idx] = FrameState(t=fr.t, state="UNKNOWN", confidence=0.9)
            continue

        clip_indices.append(idx)

    # Pass 3: Zero-shot CLIP for remaining frames
    if clip_indices:
        from transformers import CLIPModel, CLIPProcessor

        clip_id = settings.models.clip
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = CLIPModel.from_pretrained(clip_id).to(device)
        processor = CLIPProcessor.from_pretrained(clip_id)

        try:
            # Prepare CLIP prompt mapping
            categories: list[ScreenState] = []
            prompt_texts: list[str] = []
            for cat, prompts in settings.screen_state.clip_prompts.items():
                for p in prompts:
                    categories.append(cat)  # type: ignore[arg-type]
                    prompt_texts.append(p)

            batch_size = settings.screen_state.clip_batch_size
            for i in range(0, len(clip_indices), batch_size):
                b_indices = clip_indices[i : i + batch_size]
                pil_imgs = []
                for b_idx in b_indices:
                    rgb = cv2.cvtColor(loaded_images[b_idx], cv2.COLOR_BGR2RGB)
                    pil_imgs.append(Image.fromarray(rgb))

                inputs = processor(
                    text=prompt_texts,
                    images=pil_imgs,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image  # [B, n_prompts]
                    probs_all = torch.softmax(logits_per_image, dim=1).cpu().numpy()

                for b_sub, b_idx in enumerate(b_indices):
                    # Aggregate probabilities by category
                    cat_probs: dict[ScreenState, float] = {}
                    for p_idx, cat in enumerate(categories):
                        cat_probs[cat] = cat_probs.get(cat, 0.0) + float(probs_all[b_sub, p_idx])

                    best_cat, best_prob = max(cat_probs.items(), key=lambda x: x[1])
                    # If highest probability is low (< 0.35), classify as UNKNOWN
                    if best_prob < 0.35:
                        assigned_cat: ScreenState = "UNKNOWN"
                    else:
                        assigned_cat = best_cat

                    states[b_idx] = FrameState(
                        t=frames[b_idx].t,
                        state=assigned_cat,
                        confidence=round(best_prob, 3),
                    )
        finally:
            del model, processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result: list[FrameState] = []
    for idx, s in enumerate(states):
        if s is not None:
            result.append(s)
        else:
            result.append(FrameState(t=frames[idx].t, state="UNKNOWN", confidence=0.0))

    return result
