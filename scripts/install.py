#!/usr/bin/env python3
"""One-command setup of the Class Quality Assessor on Windows, Linux or macOS.

    python scripts/install.py                 # everything
    python scripts/install.py --english-only  # skip the 3 GB multilingual speech model
    python scripts/install.py --skip-selftest # install only

It uses only the Python standard library so it can run before the project environment exists. Steps: check the tools that
cannot be installed from here (uv, ffmpeg, Ollama), create the environment from the lock file, create .env with a random
access code, download the models, pull the local language model, and run a self-test that pushes a generated clip through
the real pipeline. Every step prints what it did and, if it fails, what to do about it.
"""

import argparse
import json
import platform
import secrets
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://127.0.0.1:11434"
PLACEHOLDER = "change_this_to_a_secure_token"

INSTALL_HINTS = {
    "uv": {
        "win32": "winget install --id astral-sh.uv -e    (or: pip install uv)",
        "darwin": "brew install uv    (or: pip install uv)",
        "linux": "curl -LsSf https://astral.sh/uv/install.sh | sh    (or: pip install uv)",
    },
    "ffmpeg": {
        "win32": "winget install --id Gyan.FFmpeg -e    (then open a new terminal)",
        "darwin": "brew install ffmpeg",
        "linux": "sudo apt install ffmpeg    (or your distribution's package manager)",
    },
    "ollama": {
        "win32": "download the installer from https://ollama.com/download and start it",
        "darwin": "download the app from https://ollama.com/download and start it",
        "linux": "curl -fsSL https://ollama.com/install.sh | sh    (then: ollama serve)",
    },
}


def hint(tool: str) -> str:
    table = INSTALL_HINTS[tool]
    return table.get(sys.platform if sys.platform in table else "linux", table["linux"])


def say(status: str, text: str) -> None:
    print(f"[{status:^6}] {text}", flush=True)


def run(cmd: list[str], **kwargs) -> int:
    print("         $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False, **kwargs).returncode


def check_tools() -> dict[str, bool]:
    found = {name: shutil.which(name) is not None for name in ("uv", "ffmpeg", "ffprobe")}
    for name, ok in found.items():
        say("ok" if ok else "MISSING", f"{name}" + ("" if ok else f" -> {hint('ffmpeg' if name == 'ffprobe' else name)}"))
    return found


def ensure_env_file(root: Path = ROOT) -> str:
    """Create .env from .env.example with a random access code; never overwrite a real value. Returns what happened."""
    env_path, example = root / ".env", root / ".env.example"
    if not env_path.exists():
        text = example.read_text(encoding="utf-8") if example.exists() else "ACCESS_CODE=\n"
        env_path.write_text(text, encoding="utf-8")
        created = True
    else:
        created = False
    lines = env_path.read_text(encoding="utf-8").splitlines()
    out, has_code = [], False
    for line in lines:
        if line.strip().startswith("ACCESS_CODE="):
            value = line.split("=", 1)[1].strip()
            if not value or value == PLACEHOLDER or len(value) < 10:
                line = f"ACCESS_CODE={secrets.token_urlsafe(12)}"
                has_code = "generated"
            else:
                has_code = "kept"
        out.append(line)
    if not has_code:
        out.append(f"ACCESS_CODE={secrets.token_urlsafe(12)}")
        has_code = "generated"
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return ("created .env, " if created else "found .env, ") + f"access code {has_code}"


def ollama_models() -> list[str] | None:
    """Names of the models the running Ollama server has, or None when it is not reachable."""
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5) as resp:
            return [m["name"] for m in json.load(resp).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return None


def llm_name() -> str:
    """The model tag from config.yaml / config.local.yaml, read without importing the project (it may not be installed yet)."""
    name = "qwen3-vl:8b-instruct"
    for fname in ("config.yaml", "config.local.yaml"):
        path = ROOT / fname
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("llm:"):
                    name = line.split(":", 1)[1].strip().strip("\"'")
    return name


def setup_ollama() -> bool:
    if shutil.which("ollama") is None:
        say("MISSING", f"ollama -> {hint('ollama')}")
        return False
    models = ollama_models()
    if models is None:
        say("FAIL", "the Ollama server is not running -> start it (the Ollama app, or `ollama serve`) and run this again")
        return False
    name = llm_name()
    if any(m == name or m.split(":")[0] == name for m in models):
        say("ok", f"Ollama model {name} is already there")
        return True
    say("..", f"pulling {name} (about 6 GB)")
    if run(["ollama", "pull", name]) != 0:
        say("FAIL", f"could not pull {name}")
        return False
    say("ok", f"Ollama model {name} pulled")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up the Class Quality Assessor.")
    parser.add_argument("--english-only", action="store_true", help="skip the multilingual speech model (3 GB)")
    parser.add_argument("--skip-models", action="store_true", help="do not download speech, voice and image models")
    parser.add_argument("--skip-ollama", action="store_true", help="do not check or pull the Ollama model")
    parser.add_argument("--skip-selftest", action="store_true", help="do not run the self-test at the end")
    args = parser.parse_args(argv)

    say("info", f"{platform.system()} {platform.release()}, Python {platform.python_version()}, folder {ROOT}")
    tools = check_tools()
    if not (tools["uv"] and tools["ffmpeg"] and tools["ffprobe"]):
        say("STOP", "install the missing tools above, open a new terminal and run this again")
        return 2

    say("..", "creating the environment from uv.lock (downloads Python 3.12 and about 5 GB of packages the first time)")
    if run(["uv", "sync", "--frozen"]) != 0:
        say("FAIL", "uv sync failed; the messages above say why (usually network or disk space)")
        return 3
    say("ok", "environment ready in .venv")
    say("ok", ensure_env_file())

    if not args.skip_models:
        cmd = ["uv", "run", "--frozen", "python", "scripts/download_models.py"] + (["--english-only"] if args.english_only else [])
        say("..", "downloading speech, voice and image models (about 2 GB; 5 GB with the multilingual model)")
        if run(cmd) != 0:
            say("FAIL", "model download failed; run `uv run python scripts/download_models.py` again after fixing the cause")
            return 4
    if not args.skip_ollama and not setup_ollama():
        return 5

    if not args.skip_selftest:
        say("..", "self-test: a generated clip goes through the real pipeline")
        if run(["uv", "run", "--frozen", "python", "scripts/selftest.py"]) != 0:
            say("FAIL", "the self-test failed; see the output above")
            return 6
    say("DONE", "ready. Start with:  uv run python scripts/run.py   (then open http://127.0.0.1:8765)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
