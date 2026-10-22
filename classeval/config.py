import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr


class ModelsConfig(BaseModel):
    whisper: str = "large-v3-turbo"
    whisper_dir: Path = Path("models/whisper")
    clip: str = "openai/clip-vit-base-patch32"
    ecapa: str = "speechbrain/spkrec-ecapa-voxceleb"
    whisper_multilingual: str = "faster-whisper-large-v3"
    pyannote: str = "pyannote/speaker-diarization-3.1"
    llm: str = "qwen3-vl:8b"


class OllamaConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"


class GroupsConfig(BaseModel):
    presentation: list[str]
    teaching: list[str]


class CutoffsConfig(BaseModel):
    poor_lt: float = 65.0
    average_lt: float = 75.0
    good_lt: float = 80.0


class LimitsConfig(BaseModel):
    max_upload_bytes: int = 2684354560
    max_duration_s: float = 10800.0
    min_duration_s: float = 600.0
    chunk_bytes: int = 8388608
    job_ttl_hours: float = 24.0
    job_lease_s: float = 1800.0  # a running job untouched for this long is presumed orphaned by a dead worker


class SamplingConfig(BaseModel):
    frame_fps: float = 0.2
    frame_width: int = 640
    keyframes: int = 24


class LexiconConfig(BaseModel):
    greeting: list[str] = Field(default_factory=list)
    rapport: list[str] = Field(default_factory=list)
    self_intro: list[str] = Field(default_factory=list)
    homework: list[str] = Field(default_factory=list)
    closing: list[str] = Field(default_factory=list)


class PathsConfig(BaseModel):
    jobs_dir: Path = Path("jobs")
    uploads_dir: Path = Path("uploads")


class AsrConfig(BaseModel):
    device: str = "cuda"
    compute_type: str = "int8_float16"
    batch_size: int = 8
    beam_size: int = 5
    vad_filter: bool = True
    language_windows: int = 5
    language_window_s: float = 30.0


class DiarizationConfig(BaseModel):
    prefer: Literal["pyannote", "ecapa"] = "ecapa"  # pyannote is not used: it breaks with the pinned huggingface-hub
    ecapa_window_s: float = 1.5
    ecapa_hop_s: float = 0.75
    ecapa_merge_gap_s: float = 0.5
    distance_threshold: float = 0.8
    min_cluster_share: float = 0.03
    smooth_windows: int = 0


class ScreenStateConfig(BaseModel):
    flat_card_luma_range: tuple[float, float] = (20.0, 40.0)
    flat_card_std_max: float = 8.0
    flat_card_pixel_share: float = 0.92
    logo_match_threshold: float = 0.6
    logo_scales: list[float] = Field(default_factory=lambda: [0.35, 0.5, 0.7, 1.0, 1.4, 1.6])
    clip_batch_size: int = 32
    clip_prompts: dict[str, list[str]] = Field(default_factory=dict)
    # The tutor's tile in the top-right corner while something is shared: (x, y, width, height) as frame fractions.
    thumb_region: tuple[float, float, float, float] = (0.68, 0.0, 0.32, 0.42)
    thumb_every_s: float = 20.0
    thumb_scales: list[float] = Field(default_factory=lambda: [0.08, 0.1, 0.13, 0.17, 0.22, 0.28, 0.35, 0.45])
    thumb_logo_threshold: float = 0.56  # best score without the backdrop stays below ~0.52, with it above ~0.6
    thumb_present_share: float = 0.15  # the backdrop counts as shown when it appears in this share of the moments
    thumb_min_crops: int = 5
    visual_samples: int = 16  # tutor-speaking moments shown to the vision model
    visual_min_samples: int = 4


class LanguagesConfig(BaseModel):
    # Transcript-based checks (word lists, model judgements, quotes) are validated for these languages only.
    full_support: list[str] = Field(default_factory=lambda: ["en"])


class Settings(BaseModel):
    models: ModelsConfig
    ollama: OllamaConfig
    weights: dict[str, float]
    groups: GroupsConfig
    cutoffs: CutoffsConfig
    limits: LimitsConfig
    sampling: SamplingConfig
    lexicon: LexiconConfig
    scoring: dict[str, list[float]]
    paths: PathsConfig
    asr: AsrConfig = Field(default_factory=AsrConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    screen_state: ScreenStateConfig = Field(default_factory=ScreenStateConfig)
    languages: LanguagesConfig = Field(default_factory=LanguagesConfig)

    access_code: SecretStr
    hf_token: SecretStr | None = None
    recordings_dir: Path | None = None


def _merge(base: dict, override: dict) -> dict:
    """Recursive dictionary merge: values of `override` win, nested dictionaries are merged key by key."""
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_settings(path: Path | str = "config.yaml", env_file: Path | str | None = ".env") -> Settings:
    if env_file is not None:
        dotenv_path = Path(env_file)
        if dotenv_path.is_file():
            load_dotenv(dotenv_path=dotenv_path, override=False)

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Machine-specific overrides (model folder, ports) live in an untracked config.local.yaml next to config.yaml.
    local_path = cfg_path.with_name("config.local.yaml")
    if local_path.is_file():
        with open(local_path, "r", encoding="utf-8") as f:
            data = _merge(data, yaml.safe_load(f) or {})

    raw_access_code = os.environ.get("ACCESS_CODE")
    if not raw_access_code or not raw_access_code.strip():
        raise ValueError("ACCESS_CODE environment variable is required")

    access_code = raw_access_code.strip()
    if access_code == "change_this_to_a_secure_token":
        raise ValueError("ACCESS_CODE cannot be placeholder value ('change_this_to_a_secure_token')")

    if len(access_code) < 10:
        raise ValueError("ACCESS_CODE must be at least 10 characters")

    data["access_code"] = access_code

    raw_hf_token = os.environ.get("HF_TOKEN")
    if raw_hf_token and raw_hf_token.strip():
        data["hf_token"] = raw_hf_token.strip()

    raw_recordings_dir = os.environ.get("RECORDINGS_DIR")
    if raw_recordings_dir and raw_recordings_dir.strip():
        data["recordings_dir"] = Path(raw_recordings_dir.strip())

    return Settings(**data)
