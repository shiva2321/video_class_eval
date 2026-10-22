from pathlib import Path

import cv2
import numpy as np

from classeval.config import Settings
from classeval.schema import Diarization, FrameRef, FrameState, Transcript, VideoFacts


def compute(
    states: list[FrameState],
    d: Diarization,
    tr: Transcript,
    settings: Settings,
    frames: list[FrameRef] | None = None,
) -> VideoFacts:
    """Aggregate frame state classifications into global video timeline features."""
    n_frames = len(states)
    if n_frames == 0:
        return VideoFacts(
            states=[],
            tutor_name=None,
            branded_ratio_tutor_speaking=None,
            board_share_ratio=0.0,
            lesson_video_ratio=0.0,
            camera_off_total_ratio=0.0,
            camera_off_while_tutor_speaks_s=0.0,
            first5_branded=None,
            last5_branded=None,
            talking_head_luma_mean=None,
            n_frames=0,
        )

    frame_period = 1.0 / settings.sampling.frame_fps if settings.sampling.frame_fps > 0 else 5.0

    # Counts across all frames
    board_count = sum(1 for s in states if s.state == "BOARD_SHARE")
    lesson_video_count = sum(1 for s in states if s.state == "LESSON_VIDEO_SHARE")
    camera_off_count = sum(1 for s in states if s.state == "CAMERA_OFF_CARD")

    board_share_ratio = round(float(board_count) / float(n_frames), 3)
    lesson_video_ratio = round(float(lesson_video_count) / float(n_frames), 3)
    camera_off_total_ratio = round(float(camera_off_count) / float(n_frames), 3)

    # Tutor turns
    tutor_turns = [t for t in d.turns if d.tutor and t.speaker == d.tutor]

    # Frames occurring while tutor speaks
    tutor_frames = [
        s for s in states if any(t.start <= s.t <= t.end for t in tutor_turns)
    ]

    # Branded ratio while tutor speaks
    branded_tutor = sum(1 for s in tutor_frames if s.state == "BRANDED_TALKING_HEAD")
    other_th_tutor = sum(1 for s in tutor_frames if s.state == "OTHER_TALKING_HEAD")
    total_th_tutor = branded_tutor + other_th_tutor

    if total_th_tutor >= 3:
        branded_ratio_tutor_speaking: float | None = round(
            float(branded_tutor) / float(total_th_tutor), 3
        )
    else:
        branded_ratio_tutor_speaking = None

    # Camera off while tutor speaks (in seconds)
    cam_off_tutor_count = sum(1 for s in tutor_frames if s.state == "CAMERA_OFF_CARD")
    camera_off_while_tutor_speaks_s = round(float(cam_off_tutor_count) * frame_period, 1)

    # First 5 and last 5 minutes branded presence
    max_t = max(s.t for s in states)
    first5_states = [s for s in states if s.t <= 300.0]
    last5_states = [s for s in states if s.t >= max(0.0, max_t - 300.0)]

    def _calc_window_branded(w_states: list[FrameState]) -> bool | None:
        b_count = sum(1 for s in w_states if s.state == "BRANDED_TALKING_HEAD")
        o_count = sum(1 for s in w_states if s.state == "OTHER_TALKING_HEAD")
        tot = b_count + o_count
        if tot == 0:
            return None
        return (float(b_count) / float(tot)) >= 0.5

    first5_branded = _calc_window_branded(first5_states)
    last5_branded = _calc_window_branded(last5_states)

    # Talking head luma mean
    talking_head_luma_mean: float | None = None
    if frames:
        th_lumas: list[float] = []
        frame_map = {fr.t: fr for fr in frames}
        for s in states:
            if s.state in ("BRANDED_TALKING_HEAD", "OTHER_TALKING_HEAD") and s.t in frame_map:
                fr = frame_map[s.t]
                if Path(fr.path).exists():
                    img = cv2.imread(str(fr.path))
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        th_lumas.append(float(np.mean(gray)))
        if th_lumas:
            talking_head_luma_mean = round(float(np.mean(th_lumas)), 2)

    return VideoFacts(
        states=states,
        tutor_name=None,  # Cut for time per plan
        branded_ratio_tutor_speaking=branded_ratio_tutor_speaking,
        board_share_ratio=board_share_ratio,
        lesson_video_ratio=lesson_video_ratio,
        camera_off_total_ratio=camera_off_total_ratio,
        camera_off_while_tutor_speaks_s=camera_off_while_tutor_speaks_s,
        first5_branded=first5_branded,
        last5_branded=last5_branded,
        talking_head_luma_mean=talking_head_luma_mean,
        n_frames=n_frames,
    )
