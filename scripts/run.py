"""Start the local web page and the worker that does the analysis, and keep both running.

    uv run python scripts/run.py [--port 8765] [--host 127.0.0.1]

Open http://127.0.0.1:8765, enter the access code from .env, choose a recording and watch the progress. One job runs at a
time. The page and the worker are restarted if either stops. Ctrl+C stops both. Output goes to logs/.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def start(args: list[str], log: Path) -> subprocess.Popen:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    out = open(log.with_suffix(".log"), "ab")  # noqa: SIM115 - kept open for the life of the child
    err = open(log.with_suffix(".err.log"), "ab")  # noqa: SIM115
    return subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env, stdout=out, stderr=err)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the web page and the worker.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 only on a network you trust")
    args = parser.parse_args(argv)

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    commands = {
        "server": ["-m", "uvicorn", "classeval.server.main:create_app", "--factory", "--host", args.host, "--port", str(args.port)],
        "worker": ["-m", "classeval.worker"],
    }
    procs = {name: start(cmd, logs / name) for name, cmd in commands.items()}
    print(f"Open http://{args.host}:{args.port}  (logs in {logs}; Ctrl+C stops)", flush=True)
    try:
        while True:
            time.sleep(2)
            for name, proc in procs.items():
                if proc.poll() is not None:
                    print(f"{name} stopped with code {proc.returncode}; restarting in 3 s", flush=True)
                    time.sleep(3)
                    procs[name] = start(commands[name], logs / name)
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
