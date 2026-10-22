from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from classeval.pipeline import audio_features
from classeval.schema import Diarization, Segment, SpeakerTurn, Transcript, Word


@pytest.fixture
def make_wav(tmp_path):
    def _create(audio: np.ndarray, sr: int = 16000, name: str = "test.wav") -> Path:
        p = tmp_path / name
        sf.write(str(p), audio.astype(np.float32), sr)
        return p

    return _create


def test_loud_vs_quiet_tone_lufs_ordering(make_wav):
    sr = 16000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    quiet = (np.sin(2 * np.pi * 440 * t) * 0.05).astype(np.float32)
    loud = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)

    p_quiet = make_wav(quiet, sr=sr, name="quiet.wav")
    p_loud = make_wav(loud, sr=sr, name="loud.wav")

    tr = Transcript(language="en", language_confidence=1.0, asr_model="w", segments=[])
    d = Diarization(turns=[], tutor=None, tutor_confidence=0.0, method="ecapa", n_speakers=0)

    facts_quiet = audio_features.compute(p_quiet, tr, d)
    facts_loud = audio_features.compute(p_loud, tr, d)

    assert facts_quiet.loudness_lufs < facts_loud.loudness_lufs


def test_clipped_square_wave_clipping_ratio_gt_0_1(make_wav):
    sr = 16000
    # Square wave at amplitude 1.0 (all samples at +1.0 or -1.0)
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    sq = np.sign(np.sin(2 * np.pi * 100 * t)).astype(np.float32)

    p_clipped = make_wav(sq, sr=sr, name="clipped.wav")
    tr = Transcript(language="en", language_confidence=1.0, asr_model="w", segments=[])
    d = Diarization(turns=[], tutor=None, tutor_confidence=0.0, method="ecapa", n_speakers=0)

    facts = audio_features.compute(p_clipped, tr, d)
    assert facts.clipping_ratio > 0.1


def test_dead_air_count(make_wav):
    # speech (2s) - 10s silence - speech (2s) - 10s silence - speech (2s) -> total 26s
    # Two gaps > 8s -> 2 dead air occurrences. Duration = 26s = 26/60 min.
    sr = 16000
    dur = 26.0
    audio = np.zeros(int(sr * dur), dtype=np.float32)
    p_wav = make_wav(audio, sr=sr, name="dead_air.wav")

    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker="TUTOR"),
        SpeakerTurn(start=12.0, end=14.0, speaker="TUTOR"),
        SpeakerTurn(start=24.0, end=26.0, speaker="TUTOR"),
    ]
    d = Diarization(turns=turns, tutor="TUTOR", tutor_confidence=0.9, method="ecapa", n_speakers=1)
    tr = Transcript(language="en", language_confidence=1.0, asr_model="w", segments=[])

    facts = audio_features.compute(p_wav, tr, d)
    expected_rate = 2.0 / (dur / 60.0)
    assert pytest.approx(facts.dead_air_per_min, 0.05) == expected_rate


def test_response_latency_median(make_wav):
    sr = 16000
    audio = np.zeros(sr * 30, dtype=np.float32)
    p_wav = make_wav(audio, sr=sr, name="latency.wav")

    # Tutor asks question ending at 2.0, student answers at 4.0 (gap 2.0)
    # Tutor asks question ending at 10.0, student answers at 14.0 (gap 4.0)
    segments = [
        Segment(start=0.0, end=2.0, text="Do you see the bishop?", speaker="TUTOR"),
        Segment(start=4.0, end=6.0, text="Yes", speaker="STUDENT"),
        Segment(start=8.0, end=10.0, text="Can you move it?", speaker="TUTOR"),
        Segment(start=14.0, end=16.0, text="Yes I can", speaker="STUDENT"),
    ]
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker="TUTOR"),
        SpeakerTurn(start=4.0, end=6.0, speaker="STUDENT"),
        SpeakerTurn(start=8.0, end=10.0, speaker="TUTOR"),
        SpeakerTurn(start=14.0, end=16.0, speaker="STUDENT"),
    ]
    tr = Transcript(language="en", language_confidence=1.0, asr_model="w", segments=segments)
    d = Diarization(turns=turns, tutor="TUTOR", tutor_confidence=0.9, method="ecapa", n_speakers=2)

    facts = audio_features.compute(p_wav, tr, d)
    # Median of [2.0, 4.0] = 3.0
    assert facts.median_response_latency_s == pytest.approx(3.0, 0.01)


def test_pitch_std_varied_gt_monotone(make_wav):
    sr = 16000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    # Monotone 150 Hz
    monotone = (np.sin(2 * np.pi * 150 * t) * 0.5).astype(np.float32)
    # Glissando 120 Hz -> 300 Hz
    f_inst = np.linspace(120, 300, sr * 3)
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    varied = (np.sin(phase) * 0.5).astype(np.float32)

    p_mono = make_wav(monotone, sr=sr, name="mono.wav")
    p_var = make_wav(varied, sr=sr, name="var.wav")

    turns = [SpeakerTurn(start=0.0, end=3.0, speaker="TUTOR")]
    d = Diarization(turns=turns, tutor="TUTOR", tutor_confidence=0.9, method="ecapa", n_speakers=1)
    tr = Transcript(language="en", language_confidence=1.0, asr_model="w", segments=[])

    facts_mono = audio_features.compute(p_mono, tr, d)
    facts_var = audio_features.compute(p_var, tr, d)

    assert facts_mono.pitch_std_semitones_tutor is not None
    assert facts_var.pitch_std_semitones_tutor is not None
    assert facts_var.pitch_std_semitones_tutor > facts_mono.pitch_std_semitones_tutor


def test_dropout_detected(make_wav):
    sr = 16000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    # Tone (3s) + Digital silence (3s) + Tone (3s)
    silence = np.zeros(sr * 3, dtype=np.float32)
    full = np.concatenate([tone, silence, tone])

    p_wav = make_wav(full, sr=sr, name="dropout.wav")
    tr = Transcript(language="en", language_confidence=1.0, asr_model="w", segments=[])
    d = Diarization(turns=[], tutor=None, tutor_confidence=0.0, method="ecapa", n_speakers=0)

    facts = audio_features.compute(p_wav, tr, d)
    assert len(facts.dropouts) >= 1
    t_start, t_end = facts.dropouts[0]
    assert 2.8 <= t_start <= 3.2
    assert 5.8 <= t_end <= 6.2


def test_no_student_speech_gives_zero_words_per_min(make_wav):
    sr = 16000
    audio = np.zeros(sr * 60, dtype=np.float32)
    p_wav = make_wav(audio, sr=sr, name="no_student.wav")

    segments = [
        Segment(
            start=0.0,
            end=10.0,
            text="Hello class",
            speaker="TUTOR",
            words=[
                Word(start=0.0, end=5.0, text="Hello", prob=1.0),
                Word(start=5.0, end=10.0, text="class", prob=1.0),
            ],
        )
    ]
    turns = [SpeakerTurn(start=0.0, end=10.0, speaker="TUTOR")]
    tr = Transcript(language="en", language_confidence=1.0, asr_model="w", segments=segments)
    d = Diarization(turns=turns, tutor="TUTOR", tutor_confidence=0.9, method="ecapa", n_speakers=1)

    facts = audio_features.compute(p_wav, tr, d)
    assert facts.student_words_per_min == 0.0


def test_none_fields_when_no_tutor(make_wav):
    sr = 16000
    audio = np.zeros(sr * 10, dtype=np.float32)
    p_wav = make_wav(audio, sr=sr, name="no_tutor.wav")

    tr = Transcript(language="en", language_confidence=1.0, asr_model="w", segments=[])
    d = Diarization(turns=[], tutor=None, tutor_confidence=0.0, method="ecapa", n_speakers=0)

    facts = audio_features.compute(p_wav, tr, d)
    assert facts.tutor_talk_share == 0.0
    assert facts.pitch_std_semitones_tutor is None
    assert facts.energy_cv_tutor is None
