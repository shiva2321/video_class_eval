import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from classeval.maintenance import purge_expired, requeue_stale
from classeval.server.jobs import Job, JobStore


def test_old_job_json_without_updated_at_loads():
    # Legacy job.json without updated_at field must still load and default updated_at to created_at
    legacy_json = """{
      "id": "legacy_job_12345678",
      "status": "done",
      "stage": "done",
      "progress": 1.0,
      "created_at": "2026-01-01T12:00:00+00:00",
      "source": {"type": "upload"}
    }"""
    job = Job.model_validate_json(legacy_json)
    assert job.updated_at == "2026-01-01T12:00:00+00:00"
    assert job.id == "legacy_job_12345678"


def test_purge_removes_only_expired(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    uploads_dir = tmp_path / "uploads"
    jobs_dir.mkdir()
    uploads_dir.mkdir()

    now = time.time()
    ttl_hours = 24.0
    cutoff = now - (ttl_hours * 3600.0)

    # 1. Old done job -> should be removed
    old_done = jobs_dir / "old_done_job_123456"
    old_done.mkdir()
    f1 = old_done / "job.json"
    f1.write_text('{"id": "old_done_job_123456", "status": "done", "source": {}}', encoding="utf-8")
    os.utime(old_done, (cutoff - 100, cutoff - 100))
    os.utime(f1, (cutoff - 100, cutoff - 100))

    # 2. Fresh done job -> should be kept
    fresh_done = jobs_dir / "fresh_done_job_12345"
    fresh_done.mkdir()
    f2 = fresh_done / "job.json"
    f2.write_text('{"id": "fresh_done_job_12345", "status": "done", "source": {}}', encoding="utf-8")
    os.utime(fresh_done, (now - 100, now - 100))
    os.utime(f2, (now - 100, now - 100))

    # 3. Running new job -> kept
    running_new = jobs_dir / "running_new_job_1234"
    running_new.mkdir()
    f3 = running_new / "job.json"
    f3.write_text('{"id": "running_new_job_1234", "status": "running", "source": {}}', encoding="utf-8")
    os.utime(running_new, (now - 50, now - 50))
    os.utime(f3, (now - 50, now - 50))

    # 4. Running old job (dead worker, older than ttl) -> removed
    running_old = jobs_dir / "running_old_job_1234"
    running_old.mkdir()
    f4 = running_old / "job.json"
    f4.write_text('{"id": "running_old_job_1234", "status": "running", "source": {}}', encoding="utf-8")
    os.utime(running_old, (cutoff - 500, cutoff - 500))
    os.utime(f4, (cutoff - 500, cutoff - 500))

    # 5. Dot-file e.g. .claim.lock -> ignored
    lock_file = jobs_dir / ".claim.lock"
    lock_file.write_text("lock", encoding="utf-8")
    os.utime(lock_file, (cutoff - 1000, cutoff - 1000))

    # 6. Non-id dir name e.g. "invalid_short" -> ignored
    invalid_dir = jobs_dir / "short"
    invalid_dir.mkdir()
    os.utime(invalid_dir, (cutoff - 1000, cutoff - 1000))

    # 7. Old upload dir -> removed
    old_up = uploads_dir / "old_upload_dir_12345"
    old_up.mkdir()
    u1 = old_up / "upload.bin"
    u1.write_bytes(b"data")
    os.utime(old_up, (cutoff - 200, cutoff - 200))
    os.utime(u1, (cutoff - 200, cutoff - 200))

    # 8. Fresh upload dir -> kept
    fresh_up = uploads_dir / "fresh_upload_dir_123"
    fresh_up.mkdir()
    u2 = fresh_up / "upload.bin"
    u2.write_bytes(b"data")
    os.utime(fresh_up, (now - 50, now - 50))
    os.utime(u2, (now - 50, now - 50))

    removed = purge_expired(jobs_dir, uploads_dir, ttl_hours=ttl_hours, now=now)

    assert "old_done_job_123456" in removed
    assert "running_old_job_1234" in removed
    assert "old_upload_dir_12345" in removed

    assert not old_done.exists()
    assert not running_old.exists()
    assert not old_up.exists()

    assert fresh_done.exists()
    assert running_new.exists()
    assert fresh_up.exists()
    assert lock_file.exists()
    assert invalid_dir.exists()


def test_requeue_stale(tmp_path: Path):
    store = JobStore(tmp_path / "jobs")
    lease_s = 1800.0  # 30 min
    now_dt = datetime.now(UTC)

    # 1. Stale running job -> should be requeued
    job_stale = store.create(source={"type": "upload"})
    stale_time = (now_dt - timedelta(seconds=2000)).isoformat()
    # Write stale updated_at directly
    j_dir = store.job_dir(job_stale.id)
    raw_stale = job_stale.model_copy(update={"status": "running", "stage": "asr", "progress": 0.5, "updated_at": stale_time})
    (j_dir / "job.json").write_text(raw_stale.model_dump_json(indent=2), encoding="utf-8")

    # 2. Fresh running job -> kept running
    job_fresh = store.create(source={"type": "upload"})
    fresh_time = (now_dt - timedelta(seconds=100)).isoformat()
    j_fresh_dir = store.job_dir(job_fresh.id)
    raw_fresh = job_fresh.model_copy(update={"status": "running", "stage": "asr", "progress": 0.3, "updated_at": fresh_time})
    (j_fresh_dir / "job.json").write_text(raw_fresh.model_dump_json(indent=2), encoding="utf-8")

    # 3. Old done job -> stays done
    job_done = store.create(source={"type": "upload"})
    raw_done = job_done.model_copy(update={"status": "done", "stage": "done", "progress": 1.0, "updated_at": stale_time})
    (store.job_dir(job_done.id) / "job.json").write_text(raw_done.model_dump_json(indent=2), encoding="utf-8")

    # 4. Old failed job -> stays failed
    job_failed = store.create(source={"type": "upload"})
    raw_failed = job_failed.model_copy(update={"status": "failed", "stage": "error", "progress": 0.2, "updated_at": stale_time})
    (store.job_dir(job_failed.id) / "job.json").write_text(raw_failed.model_dump_json(indent=2), encoding="utf-8")

    requeued = requeue_stale(store, lease_s=lease_s, now=now_dt)

    assert requeued == [job_stale.id]

    reloaded_stale = store.get(job_stale.id)
    assert reloaded_stale.status == "queued"
    assert reloaded_stale.stage == "queued"
    assert reloaded_stale.progress == 0.0
    assert reloaded_stale.error is None

    reloaded_fresh = store.get(job_fresh.id)
    assert reloaded_fresh.status == "running"
    assert reloaded_fresh.stage == "asr"

    reloaded_done = store.get(job_done.id)
    assert reloaded_done.status == "done"

    reloaded_failed = store.get(job_failed.id)
    assert reloaded_failed.status == "failed"
