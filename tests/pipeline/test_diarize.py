
import pytest

from classeval.config import load_settings
from classeval.pipeline import diarize
from classeval.schema import Diarization, Segment, SpeakerTurn, Transcript
from tests.fixtures.speech import concat_wavs, make_speech_wav


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test_diar_token_123")
    return load_settings("config.yaml", env_file=None)


def test_identify_tutor_no_turns():
    tutor, conf = diarize.identify_tutor([])
    assert tutor is None
    assert conf == 0.0


def test_identify_tutor_single_speaker():
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=12.0, end=20.0, speaker="SPEAKER_00"),
    ]
    tutor, conf = diarize.identify_tutor(turns)
    assert tutor == "SPEAKER_00"
    assert conf == 0.5


def test_identify_tutor_equal_time_low_confidence():
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=10.0, end=20.0, speaker="SPEAKER_01"),
    ]
    _tutor, conf = diarize.identify_tutor(turns)
    assert conf < 0.3


def test_assign_speakers_max_overlap():
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker="TUTOR"),
        SpeakerTurn(start=10.0, end=15.0, speaker="STUDENT"),
    ]
    d = Diarization(
        turns=turns,
        tutor="TUTOR",
        tutor_confidence=0.8,
        method="ecapa",
        n_speakers=2,
    )
    segments = [
        Segment(start=1.0, end=8.0, text="Hello everyone"),  # inside TUTOR turn
        Segment(start=10.5, end=14.5, text="Yes sir"),       # inside STUDENT turn
        Segment(start=8.0, end=12.0, text="Mixed turn"),     # 2s in TUTOR, 2s in STUDENT
    ]
    tr = Transcript(
        language="en",
        language_confidence=0.99,
        asr_model="whisper",
        segments=segments,
    )

    assigned = diarize.assign_speakers(tr, d)
    assert assigned.segments[0].speaker == "TUTOR"
    assert assigned.segments[1].speaker == "STUDENT"
    assert assigned.segments[2].speaker in ("TUTOR", "STUDENT")


@pytest.mark.gpu
def test_ecapa_fallback_when_no_token(tmp_path, monkeypatch, settings):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    wav = tmp_path / "test_ecapa.wav"
    try:
        make_speech_wav("Hello class, welcome to today's lesson", wav, voice="David")
    except RuntimeError as exc:
        pytest.skip(f"SAPI voice unavailable: {exc}")

    d = diarize.diarize(wav, settings)
    assert d.method == "ecapa"
    assert d.n_speakers >= 1


@pytest.mark.gpu
def test_two_voices_found_and_tutor_is_dominant(tmp_path, settings):
    w1 = tmp_path / "voice_david_long.wav"
    w2 = tmp_path / "voice_zira_short.wav"
    w3 = tmp_path / "voice_david_long2.wav"

    try:
        make_speech_wav(
            "Welcome everyone. Today we are exploring opening pawn structures and how to develop your knights before your bishops.",
            w1,
            voice="David",
        )
        make_speech_wav("Yes teacher, I understand.", w2, voice="Zira")
        make_speech_wav(
            "Good. Now look closely at the e4 square and see how white controls the center of the board with great force.",
            w3,
            voice="David",
        )
    except RuntimeError as exc:
        pytest.skip(f"SAPI voice unavailable: {exc}")

    combined = tmp_path / "alternating.wav"
    concat_wavs([w1, w2, w3], gap_s=0.5, out=combined)

    d = diarize.diarize(combined, settings)
    assert d.method == "ecapa"
    assert d.n_speakers >= 1
    assert d.tutor is not None


def test_embeddings_are_computed_in_bounded_batches(monkeypatch, tmp_path):
    """A one-hour class has thousands of windows; they must never go through the model in one giant batch."""
    import numpy as np
    import soundfile as sf
    import torch

    from classeval.config import load_settings
    from classeval.pipeline import diarize as diarize_mod

    sizes: list[int] = []

    class FakeClassifier:
        def encode_batch(self, batch):
            sizes.append(int(batch.shape[0]))
            return torch.randn(batch.shape[0], 1, 16)

    from speechbrain.inference import classifiers

    monkeypatch.setattr(classifiers.EncoderClassifier, "from_hparams", lambda *a, **k: FakeClassifier())
    monkeypatch.setenv("ACCESS_CODE", "test_diarize_token")
    settings = load_settings("config.yaml", env_file=None)

    sr = 16000
    t = np.arange(sr * 300) / sr  # 300 s of a loud tone: about 400 windows
    sf.write(str(tmp_path / "long.wav"), (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    diarize_mod.diarize(tmp_path / "long.wav", settings)

    assert len(sizes) > 1 and max(sizes) <= diarize_mod.EMBED_BATCH
