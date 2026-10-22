import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr


class ModelsConfig(BaseModel):
    whisper: str = "large-v3-turbo"
    whisper_dir: Path = Path(r"F:\models\whisper")
    clip: str = "openai/clip-vit-base-patch32"
    ecapa: str = "speechbrain/spkrec-ecapa-voxceleb"
    pyannote: str = "pyannote/speaker-diarization-3.1"
    llm: str = "qwen3-vl:8b"


class OllamaConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"


class GroupsConfig(BaseModel):
    presentation: list[str]
    teaching: list[str]


class CutoffsConfig(BaseModel):
    poor_lt: float = 45.0
    average_lt: float = 62.0
    good_lt: float = 78.0


class LimitsConfig(BaseModel):
    max_upload_bytes: int = 2684354560
    max_duration_s: float = 10800.0
    min_duration_s: float = 600.0
    chunk_bytes: int = 8388608
    job_ttl_hours: float = 24.0


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

    access_code: SecretStr
    hf_token: SecretStr | None = None
    recordings_dir: Path | None = None


def load_settings(path: Path | str = "config.yaml", env_file: Path | str | None = ".env") -> Settings:
    if env_file is not None:
        dotenv_path = Path(env_file)
        if dotenv_path.is_file():
            load_dotenv(dotenv_path=dotenv_path, override=True)

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_access_code = os.environ.get("ACCESS_CODE")
    if not raw_access_code or not raw_access_code.strip():
        raise ValueError("ACCESS_CODE environment variable is required")

    access_code = raw_access_code.strip()
    if access_code == "change_this_to_a_secure_token":
        raise ValueError("ACCESS_CODE cannot be placeholder value ('change_this_to_a_secure_token')")

    data["access_code"] = access_code

    raw_hf_token = os.environ.get("HF_TOKEN")
    if raw_hf_token and raw_hf_token.strip():
        data["hf_token"] = raw_hf_token.strip()

    raw_recordings_dir = os.environ.get("RECORDINGS_DIR")
    if raw_recordings_dir and raw_recordings_dir.strip():
        data["recordings_dir"] = Path(raw_recordings_dir.strip())

    return Settings(**data)
