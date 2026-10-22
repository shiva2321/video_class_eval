import gc
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import soundfile as sf
import torch

from classeval.config import Settings
from classeval.schema import Segment, Transcript, Word


def _ensure_cuda_dlls() -> None:
    """Ensure Windows finds cuDNN/cuBLAS DLLs in torch\\lib."""
    if os.name == "nt":
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
            except (OSError, AttributeError):
                pass
            if torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")


def _load_model(settings: Settings):
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    _ensure_cuda_dlls()
    whisper_dir = str(settings.models.whisper_dir)
    model_name = settings.models.whisper
    local_dir = Path(whisper_dir) / model_name
    model_path = str(local_dir) if (local_dir / "model.bin").exists() else model_name

    model = WhisperModel(
        model_path,
        device="cuda",
        compute_type="int8_float16",
        download_root=whisper_dir,
    )
    batched = BatchedInferencePipeline(model=model)
    return model, batched


def detect_language(wav: Path, settings: Settings) -> tuple[str, float]:
    """Detect language over up to 5 30-second windows across the WAV file."""
    wav = Path(wav).resolve()
    from faster_whisper import WhisperModel

    _ensure_cuda_dlls()
    whisper_dir = str(settings.models.whisper_dir)
    model_name = settings.models.whisper
    local_dir = Path(whisper_dir) / model_name
    model_path = str(local_dir) if (local_dir / "model.bin").exists() else model_name

    model = WhisperModel(
        model_path,
        device="cuda",
        compute_type="int8_float16",
        download_root=whisper_dir,
    )

    try:
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        duration_s = float(len(audio)) / float(sr)
        window_samples = int(30.0 * sr)

        if duration_s <= 30.0 or len(audio) <= window_samples:
            lang, prob, _ = model.detect_language(audio)
            return lang, round(float(prob), 3)

        # 5 evenly-spaced 30s windows
        step = (len(audio) - window_samples) / 4.0
        results: list[tuple[str, float]] = []
        for i in range(5):
            start = int(i * step)
            chunk = audio[start : start + window_samples]
            lang, prob, _ = model.detect_language(chunk)
            results.append((lang, float(prob)))

        # Majority vote
        counts = Counter(r[0] for r in results)
        majority_lang = counts.most_common(1)[0][0]
        matching_probs = [r[1] for r in results if r[0] == majority_lang]
        mean_prob = sum(matching_probs) / float(len(matching_probs))

        return majority_lang, round(mean_prob, 3)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def transcribe(
    wav: Path,
    settings: Settings,
    progress: Callable[[float], None] | None = None,
) -> Transcript:
    """Transcribe 16kHz mono audio using faster-whisper with BatchedInferencePipeline on GPU."""
    wav = Path(wav).resolve()
    audio, _sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    model, batched = _load_model(settings)

    try:
        segments_gen, info = batched.transcribe(
            audio,
            batch_size=8,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        segments: list[Segment] = []
        for seg in segments_gen:
            words: list[Word] = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        Word(
                            start=round(float(w.start), 3),
                            end=round(float(w.end), 3),
                            text=w.word.strip(),
                            prob=round(float(w.probability), 3),
                        )
                    )

            avg_lp = round(float(seg.avg_logprob), 3) if seg.avg_logprob is not None else None
            segments.append(
                Segment(
                    start=round(float(seg.start), 3),
                    end=round(float(seg.end), 3),
                    text=seg.text.strip(),
                    avg_logprob=avg_lp,
                    words=words,
                )
            )

            if progress and info.duration > 0:
                progress(min(1.0, float(seg.end) / float(info.duration)))

        return Transcript(
            language=info.language,
            language_confidence=round(float(info.language_probability), 3),
            asr_model=settings.models.whisper,
            segments=segments,
        )
    finally:
        del batched, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
