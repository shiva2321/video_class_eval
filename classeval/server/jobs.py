import json
import os
import re
import secrets
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "done", "failed"]

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


class InvalidId(ValueError):
    """An id that came from the outside is not a well-formed token (path-traversal guard)."""


class Job(BaseModel):
    id: str
    status: JobStatus = "queued"
    stage: str = "queued"
    progress: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source: dict[str, Any]
    error: dict[str, Any] | None = None
    eta_s: float | None = None


class _FileLock:
    """Cross-platform advisory file lock using msvcrt on Windows and fcntl on Unix."""

    def __init__(self, lock_path: Path, timeout: float = 10.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> Self:
        start = time.monotonic()
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    self._fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT)
                    # Windows locking locks bytes; ensure file is at least 1 byte
                    if os.fstat(self._fd).st_size == 0:
                        os.write(self._fd, b"L")
                    os.lseek(self._fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    self._fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT)
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except (OSError, BlockingIOError):
                if self._fd is not None:
                    try:
                        os.close(self._fd)
                    except OSError:
                        pass
                    self._fd = None
                if time.monotonic() - start > self.timeout:
                    raise TimeoutError(f"Could not acquire lock on {self.lock_path}") from None
                time.sleep(0.05)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._fd is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    try:
                        os.lseek(self._fd, 0, os.SEEK_SET)
                        msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl

                    try:
                        fcntl.flock(self._fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


class JobStore:
    """Store for managing jobs on the local filesystem."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root / ".claim.lock"

    def job_dir(self, job_id: str) -> Path:
        if not _ID_RE.fullmatch(job_id):
            raise InvalidId("invalid job id")
        path = (self.root / job_id).resolve()
        if path.parent != self.root:
            raise InvalidId("invalid job id")
        return path

    def _write_job(self, job: Job) -> None:
        j_dir = self.job_dir(job.id)
        j_dir.mkdir(parents=True, exist_ok=True)
        target = j_dir / "job.json"
        tmp = j_dir / "job.json.tmp"
        tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(target)

    def create(self, source: dict[str, Any]) -> Job:
        job_id = secrets.token_urlsafe(18)
        job = Job(id=job_id, source=source)
        self._write_job(job)
        return job

    def get(self, job_id: str) -> Job | None:
        try:
            target = self.job_dir(job_id) / "job.json"
        except InvalidId:
            return None
        if not target.exists():
            return None
        try:
            return Job.model_validate_json(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return None

    def update(self, job_id: str, **fields: Any) -> Job:
        job = self.get(job_id)
        if not job:
            raise KeyError(f"Job {job_id} not found")
        for k, v in fields.items():
            if hasattr(job, k):
                setattr(job, k, v)
        self._write_job(job)
        return job

    def set_stage(self, job_id: str, stage: str, progress: float) -> Job:
        return self.update(job_id, stage=stage, progress=progress)

    def claim_next(self) -> Job | None:
        with _FileLock(self._lock_path):
            queued_jobs: list[Job] = []
            for item in self.root.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    job = self.get(item.name)
                    if job and job.status == "queued":
                        queued_jobs.append(job)

            if not queued_jobs:
                return None

            queued_jobs.sort(key=lambda j: j.created_at)
            next_job = queued_jobs[0]
            next_job.status = "running"
            next_job.stage = "ingest"
            self._write_job(next_job)
            return next_job

    def delete(self, job_id: str) -> bool:
        try:
            j_dir = self.job_dir(job_id)
        except InvalidId:
            return False
        if not j_dir.exists():
            return False
        shutil.rmtree(j_dir, ignore_errors=True)
        return True
