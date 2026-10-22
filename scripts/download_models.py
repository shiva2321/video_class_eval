"""Download the models the pipeline needs, once, so that it can then run completely offline.

    uv run python scripts/download_models.py [--english-only]

- faster-whisper large-v3-turbo (English speech) and large-v3 (other languages) into models.whisper_dir
- the CLIP image model and the ECAPA voice model into the Hugging Face cache / .cache/ecapa
The local language model is pulled separately by Ollama (scripts/install.py does that).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classeval.config import load_config_dict

TURBO_REPO = "dropbox-dash/faster-whisper-large-v3-turbo"
MULTILINGUAL_REPO = "Systran/faster-whisper-large-v3"


def folder_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024) if path.exists() else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the models used by the pipeline.")
    parser.add_argument("--english-only", action="store_true", help="skip the multilingual speech model (3 GB)")
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    cfg = load_config_dict(ROOT / "config.yaml")
    models = cfg.get("models", {})
    whisper_dir = Path(models.get("whisper_dir", "models/whisper"))
    if not whisper_dir.is_absolute():
        whisper_dir = ROOT / whisper_dir
    whisper_dir.mkdir(parents=True, exist_ok=True)

    wanted = [(models.get("whisper", "large-v3-turbo"), TURBO_REPO)]
    if not args.english_only:
        wanted.append((models.get("whisper_multilingual", "faster-whisper-large-v3"), MULTILINGUAL_REPO))
    for folder, repo in wanted:
        target = whisper_dir / folder
        print(f"speech model {repo} -> {target}", flush=True)
        snapshot_download(repo, local_dir=str(target))
        print(f"  done, {folder_mb(target):.0f} MB", flush=True)

    clip = models.get("clip", "openai/clip-vit-base-patch32")
    print(f"image model {clip} -> Hugging Face cache", flush=True)
    snapshot_download(clip)

    ecapa = models.get("ecapa", "speechbrain/spkrec-ecapa-voxceleb")
    print(f"voice model {ecapa} -> {ROOT / '.cache' / 'ecapa'}", flush=True)
    from speechbrain.inference.classifiers import EncoderClassifier

    EncoderClassifier.from_hparams(source=ecapa, savedir=str(ROOT / ".cache" / "ecapa"), run_opts={"device": "cpu"})
    print("all models are in place", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
