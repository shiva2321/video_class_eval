import bisect
import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering

from classeval.config import Settings
from classeval.schema import Diarization, Segment, SpeakerTurn, Transcript, Word

EMBED_BATCH = 64  # windows per ECAPA forward pass: a class has thousands of windows, never embed them all at once
WINDOW_S = 1.5
STEP_S = 0.75
SILENCE_RMS = 10 ** (-45.0 / 20.0)  # windows quieter than -45 dBFS carry no speaker information


def identify_tutor(turns: list[SpeakerTurn]) -> tuple[str | None, float]:
    """Identify the tutor as the dominant speaker by speaking time, with a confidence.

    Confidence needs both a clear lead over the runner-up and a real share of all speech: in a class the tutor
    talks a large part of the time, so a "dominant" speaker with a tiny share means the clustering failed.
    """
    if not turns:
        return None, 0.0

    talk_times: dict[str, float] = defaultdict(float)
    for turn in turns:
        talk_times[turn.speaker] += max(0.0, float(turn.end - turn.start))

    sorted_spk = sorted(talk_times.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_spk) == 1:
        return sorted_spk[0][0], 0.5

    t1, t2 = sorted_spk[0][1], sorted_spk[1][1]
    tutor = sorted_spk[0][0]
    if t1 <= 0.0:
        return tutor, 0.0

    share = t1 / sum(talk_times.values())
    conf = min(1.0, ((t1 - t2) / t1) / 0.15) * min(1.0, share / 0.3)
    return tutor, round(conf, 3)


def _speaker_at(t: float, starts: list[float], turns: list[SpeakerTurn]) -> str | None:
    """Speaker of the turn covering time t (turns sorted by start), or of a turn within one second."""
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0 and turns[i].start <= t <= turns[i].end:
        return turns[i].speaker
    best, best_dist = None, 1.0
    for j in (i, i + 1):
        if 0 <= j < len(turns):
            dist = max(turns[j].start - t, t - turns[j].end, 0.0)
            if dist < best_dist:
                best, best_dist = turns[j].speaker, dist
    return best


def _majority_speaker(seg: Segment, d: Diarization) -> str | None:
    overlap: dict[str, float] = defaultdict(float)
    for turn in d.turns:
        amount = max(0.0, min(seg.end, turn.end) - max(seg.start, turn.start))
        if amount > 0.0:
            overlap[turn.speaker] += amount
    return max(overlap.items(), key=lambda x: x[1])[0] if overlap else d.tutor


def assign_speakers(tr: Transcript, d: Diarization, max_gap_s: float = 1.0) -> Transcript:
    """Give every utterance a speaker.

    Whisper's segments can be tens of seconds long and mix several speakers, so segments that carry word
    timestamps are re-cut: each word takes the speaker of the turn it falls in, and consecutive words of the same
    speaker (without a long pause) form one utterance. Segments without words keep the majority speaker.
    """
    starts = [t.start for t in d.turns]
    out: list[Segment] = []
    for seg in tr.segments:
        if not seg.words:
            out.append(seg.model_copy(update={"speaker": _majority_speaker(seg, d)}))
            continue

        groups: list[tuple[str | None, list[Word]]] = []
        cur_spk: str | None = None
        cur: list[Word] = []
        for w in seg.words:
            spk = _speaker_at((w.start + w.end) / 2.0, starts, d.turns) or cur_spk or d.tutor
            if cur and (spk != cur_spk or w.start - cur[-1].end > max_gap_s):
                groups.append((cur_spk, cur))
                cur = []
            cur_spk = spk
            cur.append(w)
        if cur:
            groups.append((cur_spk, cur))

        # A one- or two-word flicker between two stretches of the same speaker is noise: absorb it.
        for k in range(1, len(groups) - 1):
            spk, words = groups[k]
            if len(words) <= 2 and groups[k - 1][0] == groups[k + 1][0] != spk:
                groups[k] = (groups[k - 1][0], words)

        merged: list[tuple[str | None, list[Word]]] = []
        for spk, words in groups:
            if merged and merged[-1][0] == spk and words[0].start - merged[-1][1][-1].end <= max_gap_s:
                merged[-1][1].extend(words)
            else:
                merged.append((spk, list(words)))

        for spk, words in merged:
            out.append(
                Segment(
                    start=words[0].start,
                    end=words[-1].end,
                    text=" ".join(w.text.strip() for w in words).strip(),
                    speaker=spk,
                    avg_logprob=seg.avg_logprob,
                    words=words,
                )
            )
    return tr.model_copy(update={"segments": out})


def embed_windows(wav: Path, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """ECAPA embeddings of the voiced 1.5 s windows of a recording: (window start seconds, unit-length embeddings)."""
    from speechbrain.inference.classifiers import EncoderClassifier

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source=settings.models.ecapa,
        run_opts={"device": device},
        savedir=".cache/ecapa",
    )
    try:
        audio, sr = sf.read(str(Path(wav).resolve()), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        win, step = int(WINDOW_S * sr), int(STEP_S * sr)
        starts = [
            p
            for p in range(0, len(audio) - win + 1, step)
            if float(np.sqrt(np.mean(audio[p : p + win] ** 2))) >= SILENCE_RMS
        ]
        if not starts:
            return np.zeros(0), np.zeros((0, 0), dtype=np.float32)

        parts: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(starts), EMBED_BATCH):
                batch = torch.tensor(
                    np.stack([audio[p : p + win] for p in starts[i : i + EMBED_BATCH]]),
                    dtype=torch.float32,
                    device=device,
                )
                parts.append(classifier.encode_batch(batch).squeeze(1).cpu().numpy())  # [B, 192]
        emb = np.concatenate(parts, axis=0)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.asarray(starts, dtype=np.float64) / sr, emb / norms
    finally:
        del classifier
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _merge_small_clusters(embeddings: np.ndarray, labels: np.ndarray, min_share: float) -> np.ndarray:
    """Fold clusters smaller than `min_share` of all windows into the nearest large cluster (by centroid)."""
    ids, counts = np.unique(labels, return_counts=True)
    big = ids[counts >= max(1, int(min_share * len(labels)))]
    if len(big) == 0:
        big = ids[[int(np.argmax(counts))]]
    if len(big) == len(ids):
        return labels
    centroids = np.stack([embeddings[labels == i].mean(axis=0) for i in big])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    small = ~np.isin(labels, big)
    merged = labels.copy()
    merged[small] = big[np.argmax(embeddings[small] @ centroids.T, axis=1)]
    return merged


def _smooth_labels(labels: np.ndarray, radius: int) -> np.ndarray:
    """Majority vote over +-radius neighbouring windows: removes single-window speaker flicker."""
    if radius <= 0 or len(labels) < 3:
        return labels
    _, compact = np.unique(labels, return_inverse=True)
    out = compact.copy()
    n = len(compact)
    for i in range(n):
        out[i] = np.bincount(compact[max(0, i - radius) : min(n, i + radius + 1)]).argmax()
    return out


def cluster_windows(embeddings: np.ndarray, threshold: float, min_share: float, smooth: int) -> np.ndarray:
    """Speaker label per window: average-linkage clustering, tiny clusters merged, flicker smoothed."""
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average"
    ).fit_predict(embeddings)
    return _smooth_labels(_merge_small_clusters(embeddings, labels, min_share), smooth)


def turns_from_labels(starts: np.ndarray, labels: np.ndarray) -> list[SpeakerTurn]:
    """Group consecutive windows with the same label into speaker turns.

    Windows overlap by half, so each window only "owns" its central STEP_S seconds; turns then tile the timeline
    without overlap and summed turn durations are real seconds of speech (not double-counted).
    """
    pad = (WINDOW_S - STEP_S) / 2.0
    _, compact = np.unique(labels, return_inverse=True)
    turns: list[SpeakerTurn] = []
    cur = int(compact[0])
    cur_start, cur_end = float(starts[0]) + pad, float(starts[0]) + pad + STEP_S
    for s, lab in zip(starts[1:], compact[1:], strict=True):
        w_start, w_end = float(s) + pad, float(s) + pad + STEP_S
        if int(lab) == cur and w_start <= cur_end + 0.5:
            cur_end = max(cur_end, w_end)
        else:
            turns.append(SpeakerTurn(start=round(cur_start, 3), end=round(cur_end, 3), speaker=f"SPEAKER_{cur:02d}"))
            cur, cur_start, cur_end = int(lab), w_start, w_end
    turns.append(SpeakerTurn(start=round(cur_start, 3), end=round(cur_end, 3), speaker=f"SPEAKER_{cur:02d}"))
    return turns


def diarize(wav: Path, settings: Settings) -> Diarization:
    """Speaker diarization with ECAPA-TDNN embeddings and agglomerative clustering."""
    starts, embeddings = embed_windows(wav, settings)
    if len(starts) == 0:
        return Diarization(turns=[], tutor=None, tutor_confidence=0.0, method="ecapa", n_speakers=0)

    cfg = settings.diarization
    labels = cluster_windows(embeddings, cfg.distance_threshold, cfg.min_cluster_share, cfg.smooth_windows)
    turns = turns_from_labels(starts, labels)
    tutor, conf = identify_tutor(turns)
    return Diarization(
        turns=turns,
        tutor=tutor,
        tutor_confidence=conf,
        method="ecapa",
        n_speakers=len({t.speaker for t in turns}),
    )
