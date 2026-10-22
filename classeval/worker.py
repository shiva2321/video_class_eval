import logging
import os
import sys
import time

# Worker sets offline environment variables before importing any model code
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from classeval.config import Settings, load_settings
from classeval.guard import limit_gpu_memory, start_memory_watchdog
from classeval.maintenance import purge_expired, requeue_stale
from classeval.pipeline import orchestrator
from classeval.pipeline.orchestrator import StageFns
from classeval.server.jobs import JobStore

logger = logging.getLogger("classeval.worker")


def run_worker(
    settings: Settings,
    store: JobStore,
    poll_interval: float = 0.5,
    once: bool = False,
    stages: StageFns | None = None,
    purge_interval_s: float = 1800.0,
) -> None:
    """Worker polling loop claiming jobs and executing the pipeline."""
    try:
        requeue_stale(store, settings.limits.job_lease_s)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error requeuing stale jobs: %s", exc)

    try:
        purge_expired(
            settings.paths.jobs_dir,
            settings.paths.uploads_dir,
            settings.limits.job_ttl_hours,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error during initial worker purge: %s", exc)

    last_purge = time.monotonic()

    while True:
        try:
            now_mono = time.monotonic()
            if now_mono - last_purge >= purge_interval_s:
                try:
                    purge_expired(
                        settings.paths.jobs_dir,
                        settings.paths.uploads_dir,
                        settings.limits.job_ttl_hours,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Periodic purge failed: %s", exc)
                last_purge = now_mono

            job = store.claim_next()
            if job:
                logger.info("Claimed job %s", job.id)
                try:
                    orchestrator.run(job.id, store, settings, stages=stages)
                    logger.info("Finished job %s", job.id)
                except Exception as exc:  # noqa: BLE001 - last resort: the worker loop must survive any job failure
                    logger.error("Unexpected error executing a job: %s", type(exc).__name__)
                    curr_job = store.get(job.id)
                    if curr_job and curr_job.status != "failed":
                        code = getattr(exc, "code", "worker_error")
                        store.update(
                            job.id,
                            status="failed",
                            error={"code": str(code), "message": orchestrator.FRIENDLY_FAILURE},
                        )
            if once:
                break
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Worker interrupted by user.")
            break


def main() -> None:
    limit_gpu_memory()  # a runaway stage must fail fast, never freeze the PC
    start_memory_watchdog()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        settings = load_settings("config.yaml", env_file=".env")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load settings: %s", exc)
        sys.exit(1)

    store = JobStore(settings.paths.jobs_dir)
    try:
        requeue_stale(store, settings.limits.job_lease_s)
        purge_expired(
            settings.paths.jobs_dir,
            settings.paths.uploads_dir,
            settings.limits.job_ttl_hours,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Worker startup maintenance error: %s", exc)

    logger.info("Starting worker loop on %s...", store.root)
    run_worker(settings, store)


if __name__ == "__main__":
    main()
