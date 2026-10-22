import math
from pathlib import Path

import numpy as np
import parselmouth
import pyloudnorm as pyln
import soundfile as sf

from classeval.schema import AudioFacts, Diarization, Transcript

# Plan Task 7 pinned constants
DEAD_AIR_GAP_S = 8.0  # Plan: count of gaps > 8 s between consecutive speech turns
DROPOUT_MIN_S = 2.0  # Plan: intervals >= 2 s with RMS < -70 dBFS
DROPOUT_THRESHOLD_DBFS = -70.0  # Plan: digital dropout threshold RMS < -70 dBFS
FRAME_LEN_S = 0.05  # Plan: 50 ms frames for RMS, SNR and dropout analysis
CLIPPING_SAMPLE_THRESHOLD = 0.999  # Plan: share of samples with |x| >= 0.999
RESPONSE_LATENCY_CAP_S = 30.0  # Plan: cap 30 s from tutor question to response


def compute(wav: Path, tr: Transcript, d: Diarization) -> AudioFacts:
    """Extract acoustic, speech turn, and interaction features from audio and transcript."""
    wav = Path(wav).resolve()
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration_s = float(len(audio)) / float(sr) if sr > 0 else 0.0
    duration_min = max(duration_s / 60.0, 1e-6)

    # 1. Loudness LUFS
    try:
        meter = pyln.Meter(sr)
        loudness_lufs = float(meter.integrated_loudness(audio))
        if math.isinf(loudness_lufs) or math.isnan(loudness_lufs):
            loudness_lufs = -70.0
    except Exception:  # noqa: BLE001
        loudness_lufs = -70.0
    loudness_lufs = round(loudness_lufs, 2)

    # 2. Clipping ratio
    if len(audio) > 0:
        clipping_ratio = round(float(np.mean(np.abs(audio) >= CLIPPING_SAMPLE_THRESHOLD)), 4)
    else:
        clipping_ratio = 0.0

    # 3. 50ms Frame RMS analysis
    frame_samples = max(1, int(sr * FRAME_LEN_S))
    n_frames = len(audio) // frame_samples
    if n_frames > 0:
        reshaped = audio[: n_frames * frame_samples].reshape(n_frames, frame_samples)
        frame_rms = np.sqrt(np.mean(reshaped**2, axis=1))
    else:
        frame_rms = np.array([0.0], dtype=np.float32)

    # 4. SNR estimation (P90 / P10 of 50ms RMS)
    pos_rms = frame_rms[frame_rms > 0]
    if len(pos_rms) >= 10:
        p90 = float(np.percentile(pos_rms, 90))
        p10 = float(np.percentile(pos_rms, 10))
        snr_db_est = round(20.0 * math.log10(max(p90, 1e-9) / max(p10, 1e-9)), 2)
    else:
        snr_db_est = 0.0

    # 5. Dropouts: intervals >= 2s with RMS < -70 dBFS
    dropout_rms_thresh = 10.0 ** (DROPOUT_THRESHOLD_DBFS / 20.0)
    dropouts: list[tuple[float, float]] = []
    in_dropout = False
    start_frame = 0

    for idx, rms in enumerate(frame_rms):
        if rms < dropout_rms_thresh:
            if not in_dropout:
                in_dropout = True
                start_frame = idx
        else:
            if in_dropout:
                in_dropout = False
                dur = (idx - start_frame) * FRAME_LEN_S
                if dur >= DROPOUT_MIN_S:
                    dropouts.append(
                        (round(start_frame * FRAME_LEN_S, 3), round(idx * FRAME_LEN_S, 3))
                    )
    if in_dropout:
        dur = (len(frame_rms) - start_frame) * FRAME_LEN_S
        if dur >= DROPOUT_MIN_S:
            dropouts.append(
                (round(start_frame * FRAME_LEN_S, 3), round(len(frame_rms) * FRAME_LEN_S, 3))
            )

    # 6. Turn timing features
    sorted_turns = sorted(d.turns, key=lambda t: t.start)
    total_speech_s = sum(max(0.0, float(t.end - t.start)) for t in sorted_turns)
    speech_ratio = round(min(1.0, total_speech_s / max(duration_s, 1e-6)), 3)

    tutor_turns = [t for t in sorted_turns if d.tutor and t.speaker == d.tutor]
    tutor_speech_s = sum(max(0.0, float(t.end - t.start)) for t in tutor_turns)
    tutor_talk_share = (
        round(tutor_speech_s / total_speech_s, 3) if total_speech_s > 0.0 and d.tutor else 0.0
    )

    # Dead air per min: gaps > 8s between consecutive speech turns
    dead_air_count = 0
    for i in range(len(sorted_turns) - 1):
        gap = sorted_turns[i + 1].start - sorted_turns[i].end
        if gap > DEAD_AIR_GAP_S:
            dead_air_count += 1
    dead_air_per_min = round(float(dead_air_count) / duration_min, 3)

    # Student words per min
    student_word_count = 0
    tutor_word_count = 0
    for seg in tr.segments:
        words = seg.words if seg.words else [w for w in seg.text.split() if w]
        count = len(words)
        if d.tutor and seg.speaker == d.tutor:
            tutor_word_count += count
        elif seg.speaker is not None:
            student_word_count += count
    student_words_per_min = round(float(student_word_count) / duration_min, 2)

    tutor_min = max(tutor_speech_s / 60.0, 1e-6)
    tutor_speech_rate_wpm = (
        round(float(tutor_word_count) / tutor_min, 1) if tutor_speech_s > 0 else 0.0
    )

    # 7. Response latency: median gap from tutor '?' segment to next student segment
    latencies: list[float] = []
    for idx, seg in enumerate(tr.segments):
        if d.tutor and seg.speaker == d.tutor and seg.text.strip().endswith("?"):
            for next_seg in tr.segments[idx + 1 :]:
                if next_seg.speaker != d.tutor:
                    gap = next_seg.start - seg.end
                    if 0.0 <= gap <= RESPONSE_LATENCY_CAP_S:
                        latencies.append(gap)
                    break
    median_latency = round(float(np.median(latencies)), 2) if latencies else None

    # 8. Tutor pitch variability (semitones std) & energy CV
    pitch_std_semitones: float | None = None
    energy_cv: float | None = None

    if d.tutor and tutor_speech_s >= 1.0:
        try:
            sound = parselmouth.Sound(audio, sampling_frequency=sr)
            pitch = sound.to_pitch()
            pitch_vals = []
            for t_val in pitch.xs():
                if any(t.start <= t_val <= t.end for t in tutor_turns):
                    val = pitch.get_value_at_time(t_val)
                    if val is not None and not np.isnan(val) and val > 50.0:
                        pitch_vals.append(val)

            if len(pitch_vals) >= 10:
                semitones = 12.0 * np.log2(np.array(pitch_vals) / 440.0)
                pitch_std_semitones = round(float(np.std(semitones)), 2)
        except Exception:  # noqa: BLE001
            pitch_std_semitones = None

        # Energy CV over tutor frames
        tutor_frame_rms = []
        for idx, rms in enumerate(frame_rms):
            t_mid = (idx + 0.5) * FRAME_LEN_S
            if any(t.start <= t_mid <= t.end for t in tutor_turns) and rms > 0.0:
                tutor_frame_rms.append(rms)

        if len(tutor_frame_rms) >= 10:
            mean_e = float(np.mean(tutor_frame_rms))
            std_e = float(np.std(tutor_frame_rms))
            if mean_e > 0.0:
                energy_cv = round(std_e / mean_e, 3)

    return AudioFacts(
        duration_s=round(duration_s, 2),
        speech_ratio=speech_ratio,
        tutor_talk_share=tutor_talk_share,
        student_words_per_min=student_words_per_min,
        dead_air_per_min=dead_air_per_min,
        median_response_latency_s=median_latency,
        tutor_speech_rate_wpm=tutor_speech_rate_wpm,
        loudness_lufs=loudness_lufs,
        clipping_ratio=clipping_ratio,
        snr_db_est=snr_db_est,
        pitch_std_semitones_tutor=pitch_std_semitones,
        energy_cv_tutor=energy_cv,
        dropouts=dropouts,
    )
