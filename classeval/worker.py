import logging
import sys
import time

from classeval.config import Settings, load_settings
from classeval.pipeline import orchestrator
from classeval.server.jobs import JobStore

logger = logging.getLogger("classeval.worker")


def run_worker(
    settings: Settings,
    store: JobStore,
    poll_interval: float = 0.5,
    once: bool = False,
) -> None:
    """Worker polling loop claiming jobs and executing the pipeline."""
    while True:
        try:
            job = store.claim_next()
            if job:
                logger.info("Claimed job %s", job.id)
                try:
                    orchestrator.run(job.id, store, settings)
                    logger.info("Finished job %s", job.id)
                except Exception as exc:
                    logger.exception("Error executing job %s", job.id)
                    store.update(job.id, status="failed", error={"message": str(exc)})
            if once:
                break
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Worker interrupted by user.")
            break


def main() -> None:
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
    logger.info("Starting worker loop on %s...", store.root)
    run_worker(settings, store)


if __name__ == "__main__":
    main()
