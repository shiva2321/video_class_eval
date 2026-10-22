import logging
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from classeval.server.jobs import ID_RE, Job, JobStore

logger = logging.getLogger("classeval.maintenance")


def _newest_mtime(directory: Path) -> float:
    """Return the maximum mtime across a directory and all its files."""
    try:
        newest = directory.stat().st_mtime
    except OSError:
        return 0.0

    try:
        for entry in directory.rglob("*"):
            try:
                m = entry.stat().st_mtime
                newest = max(newest, m)
            except OSError:
                pass
    except OSError:
        pass
    return newest


def purge_expired(
    jobs_dir: Path | str,
    uploads_dir: Path | str,
    ttl_hours: float,
    now: float | datetime | None = None,
) -> list[str]:
    """Remove job and upload dirs whose newest file mtime is older than ttl_hours before now.

    NEVER remove a job whose status is 'running' unless it is also older than ttl_hours (a dead worker).
    Ignore non-id directory names and dot-files. Never raise on a single bad directory.
    """
    if now is None:
        now_ts = time.time()
    elif isinstance(now, datetime):
        now_ts = now.timestamp()
    else:
        now_ts = float(now)

    cutoff_ts = now_ts - (ttl_hours * 3600.0)
    removed_ids: list[str] = []

    for base_path, is_job_store in ((Path(jobs_dir), True), (Path(uploads_dir), False)):
        if not base_path.exists():
            continue
        try:
            entries = list(base_path.iterdir())
        except OSError as exc:
            logger.warning("Could not list directory %s: %s", base_path, exc)
            continue

        for item in entries:
            if not item.is_dir() or item.name.startswith("."):
                continue
            if not ID_RE.fullmatch(item.name):
                continue

            try:
                newest = _newest_mtime(item)
            except OSError as exc:
                logger.debug("Failed stat on %s: %s", item, exc)
                continue

            # If newest file is newer than cutoff, keep it
            if newest >= cutoff_ts:
                continue

            # If it is in jobs_dir, inspect job status
            if is_job_store:
                job_file = item / "job.json"
                if job_file.exists():
                    try:
                        job = Job.model_validate_json(job_file.read_text(encoding="utf-8"))
                        # If running, only remove if older than ttl_hours
                        # Since newest < cutoff_ts, all files are older than ttl_hours (dead worker)
                        if job.status == "running" and newest >= cutoff_ts:
                            continue
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Corrupted job.json in %s: %s", item, exc)

            try:
                shutil.rmtree(item, ignore_errors=False)
                removed_ids.append(item.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to remove directory %s: %s", item, exc)

    return removed_ids


def requeue_stale(
    store: JobStore,
    lease_s: float,
    now: float | datetime | None = None,
) -> list[str]:
    """Reset jobs with status 'running' whose updated_at is older than lease_s back to 'queued'."""
    if now is None:
        now_dt = datetime.now(UTC)
    elif isinstance(now, datetime):
        now_dt = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    else:
        now_dt = datetime.fromtimestamp(float(now), tz=UTC)

    requeued: list[str] = []
    if not store.root.exists():
        return requeued

    try:
        entries = list(store.root.iterdir())
    except OSError as exc:
        logger.warning("Could not list job store %s: %s", store.root, exc)
        return requeued

    for item in entries:
        if not item.is_dir() or item.name.startswith("."):
            continue
        if not ID_RE.fullmatch(item.name):
            continue

        try:
            job = store.get(item.name)
            if not job or job.status != "running":
                continue

            updated_dt = datetime.fromisoformat(job.updated_at)
            if updated_dt.tzinfo is None:
                updated_dt = updated_dt.replace(tzinfo=UTC)

            age_s = (now_dt - updated_dt).total_seconds()
            if age_s > lease_s:
                store.update(job.id, status="queued", stage="queued", progress=0.0, error=None)
                requeued.append(job.id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed checking stale job %s: %s", item.name, exc)

    return requeued
