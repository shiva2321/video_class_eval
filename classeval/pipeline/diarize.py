import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering

from classeval.config import Settings
from classeval.schema import Diarization, SpeakerTurn, Transcript


def identify_tutor(turns: list[SpeakerTurn]) -> tuple[str | None, float]:
    """Identify tutor as the dominant speaker by speaking time with relative confidence."""
    if not turns:
        return None, 0.0

    talk_times: dict[str, float] = defaultdict(float)
    for turn in turns:
        dur = max(0.0, float(turn.end - turn.start))
        talk_times[turn.speaker] += dur

    if not talk_times:
        return None, 0.0

    sorted_spk = sorted(talk_times.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_spk) == 1:
        return sorted_spk[0][0], 0.5

    t1 = sorted_spk[0][1]
    t2 = sorted_spk[1][1]
    tutor = sorted_spk[0][0]

    if t1 <= 0.0:
        return tutor, 0.0

    conf = min(1.0, ((t1 - t2) / t1) / 0.15)
    return tutor, round(conf, 3)


def assign_speakers(tr: Transcript, d: Diarization) -> Transcript:
    """Assign each transcript segment the speaker with maximum overlapping time."""
    updated_segments = []
    for seg in tr.segments:
        speaker_overlap: dict[str, float] = defaultdict(float)
        for turn in d.turns:
            start = max(seg.start, turn.start)
            end = min(seg.end, turn.end)
            overlap = max(0.0, end - start)
            if overlap > 0.0:
                speaker_overlap[turn.speaker] += overlap

        assigned_speaker = d.tutor
        if speaker_overlap:
            assigned_speaker = max(speaker_overlap.items(), key=lambda x: x[1])[0]

        updated_seg = seg.model_copy(update={"speaker": assigned_speaker})
        updated_segments.append(updated_seg)

    return tr.model_copy(update={"segments": updated_segments})


EMBED_BATCH = 64  # windows per ECAPA forward pass


def diarize(wav: Path, settings: Settings) -> Diarization:
    """Perform speaker diarization using ECAPA-TDNN embeddings and agglomerative clustering."""
    wav = Path(wav).resolve()
    from speechbrain.inference.classifiers import EncoderClassifier

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ecapa_source = settings.models.ecapa
    savedir = ".cache/ecapa"

    classifier = EncoderClassifier.from_hparams(
        source=ecapa_source,
        run_opts={"device": device},
        savedir=savedir,
    )

    try:
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        total_samples = len(audio)
        if total_samples == 0:
            return Diarization(turns=[], tutor=None, tutor_confidence=0.0, method="ecapa", n_speakers=0)

        # Sliding window over audio: 1.5s window, 0.75s step
        win_size = int(1.5 * sr)
        step_size = int(0.75 * sr)

        windows: list[tuple[float, float, np.ndarray]] = []
        # Silence threshold: RMS > -45 dBFS
        min_rms = 10 ** (-45.0 / 20.0)

        pos = 0
        while pos + win_size <= total_samples:
            chunk = audio[pos : pos + win_size]
            rms = np.sqrt(np.mean(chunk**2))
            if rms >= min_rms:
                t_start = float(pos) / float(sr)
                t_end = float(pos + win_size) / float(sr)
                windows.append((t_start, t_end, chunk))
            pos += step_size

        if not windows:
            return Diarization(turns=[], tutor=None, tutor_confidence=0.0, method="ecapa", n_speakers=0)

        # Extract ECAPA embeddings in small batches: a one-hour class has thousands of windows and a single
        # batch of all of them exhausts GPU memory.
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(windows), EMBED_BATCH):
                batch = torch.tensor(
                    np.stack([w[2] for w in windows[i : i + EMBED_BATCH]]), dtype=torch.float32, device=device
                )
                parts.append(classifier.encode_batch(batch).squeeze(1).cpu().numpy())  # [B, 192]
        embeddings = np.concatenate(parts, axis=0)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        if len(embeddings) == 1:
            turns = [SpeakerTurn(start=windows[0][0], end=windows[0][1], speaker="SPEAKER_00")]
            tutor, conf = identify_tutor(turns)
            return Diarization(
                turns=turns,
                tutor=tutor,
                tutor_confidence=conf,
                method="ecapa",
                n_speakers=1,
            )

        # Cluster embeddings with AgglomerativeClustering
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.55,
            metric="cosine",
            linkage="average",
        )
        labels = clusterer.fit_predict(embeddings)

        # Group adjacent windows with same label into turns
        turns: list[SpeakerTurn] = []
        cur_label = labels[0]
        cur_start = windows[0][0]
        cur_end = windows[0][1]

        for i in range(1, len(windows)):
            lbl = labels[i]
            w_start, w_end, _ = windows[i]
            if lbl == cur_label and w_start <= cur_end + 0.5:
                cur_end = max(cur_end, w_end)
            else:
                turns.append(
                    SpeakerTurn(
                        start=round(cur_start, 3),
                        end=round(cur_end, 3),
                        speaker=f"SPEAKER_{cur_label:02d}",
                    )
                )
                cur_label = lbl
                cur_start = w_start
                cur_end = w_end

        turns.append(
            SpeakerTurn(
                start=round(cur_start, 3),
                end=round(cur_end, 3),
                speaker=f"SPEAKER_{cur_label:02d}",
            )
        )

        unique_speakers = len({t.speaker for t in turns})
        tutor, conf = identify_tutor(turns)

        return Diarization(
            turns=turns,
            tutor=tutor,
            tutor_confidence=conf,
            method="ecapa",
            n_speakers=unique_speakers,
        )
    finally:
        del classifier
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
