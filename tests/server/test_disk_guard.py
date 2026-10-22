from collections import namedtuple

import pytest

from classeval.server import uploads
from classeval.server.uploads import InsufficientDiskSpace, UploadStore

Usage = namedtuple("Usage", ["total", "used", "free"])
GB = 1024**3


def _store(tmp_path):
    return UploadStore(tmp_path / "uploads", max_bytes=10 * GB, chunk_bytes=8 * 1024 * 1024)


def _free(monkeypatch, free_bytes):
    monkeypatch.setattr(uploads.shutil, "disk_usage", lambda _path: Usage(100 * GB, 0, free_bytes))


def test_a_single_upload_needs_three_times_its_size_free(monkeypatch, tmp_path):
    store = _store(tmp_path)
    _free(monkeypatch, 5 * GB)
    with pytest.raises(InsufficientDiskSpace):
        store.init(size=2 * GB, filename="a.mp4")
    assert store.init(size=1 * GB, filename="a.mp4")


def test_uploads_in_progress_claim_their_space(monkeypatch, tmp_path):
    store = _store(tmp_path)
    _free(monkeypatch, 10 * GB)
    store.init(size=2 * GB, filename="a.mp4")  # claims 6 GB of the 10 GB
    with pytest.raises(InsufficientDiskSpace):
        store.init(size=2 * GB, filename="b.mp4")  # would need 6 GB more, only 4 GB unclaimed
    assert store.init(size=1 * GB, filename="c.mp4")


def test_stored_chunks_reduce_what_an_upload_still_claims(monkeypatch, tmp_path):
    store = _store(tmp_path)
    _free(monkeypatch, 10 * GB)
    upload_id = store.init(size=3 * GB, filename="a.mp4")
    (store.root / upload_id / "chunk_0.part").write_bytes(b"x" * 1024)
    assert 9 * GB - 1024 <= store._unfinished_claim() <= 9 * GB


def test_a_completed_upload_no_longer_claims_space(monkeypatch, tmp_path):
    store = _store(tmp_path)
    _free(monkeypatch, 100 * GB)
    upload_id = store.init(size=4, filename="a.mp4")
    store.put_chunk(upload_id, 0, b"1234")
    store.complete(upload_id)
    assert store._unfinished_claim() == 0


def test_a_chunk_is_refused_when_the_disk_is_nearly_full(monkeypatch, tmp_path):
    store = _store(tmp_path)
    _free(monkeypatch, 100 * GB)
    upload_id = store.init(size=4, filename="a.mp4")
    _free(monkeypatch, 3)
    with pytest.raises(InsufficientDiskSpace):
        store.put_chunk(upload_id, 0, b"1234")


def test_completing_needs_room_for_the_joined_file(monkeypatch, tmp_path):
    store = _store(tmp_path)
    _free(monkeypatch, 100 * GB)
    upload_id = store.init(size=4, filename="a.mp4")
    store.put_chunk(upload_id, 0, b"1234")
    _free(monkeypatch, 2)
    with pytest.raises(InsufficientDiskSpace):
        store.complete(upload_id)


def _make_idle(store, upload_id):
    import os
    import time

    old = time.time() - uploads.IDLE_CLAIM_S - 60
    os.utime(store.root / upload_id / "meta.json", (old, old))


def test_an_idle_upload_stops_claiming_space_and_resumes_when_there_is_room(monkeypatch, tmp_path):
    store = _store(tmp_path)
    _free(monkeypatch, 10 * GB)
    upload_id = store.init(size=2 * GB, filename="a.mp4")
    assert store._unfinished_claim() == 6 * GB
    _make_idle(store, upload_id)
    assert store._unfinished_claim() == 0
    store.put_chunk(upload_id, 0, b"1234")  # nothing else is active, so it fits and becomes active again
    assert store._unfinished_claim() > 0


def test_an_idle_upload_cannot_resume_next_to_others_that_fill_the_disk(monkeypatch, tmp_path):
    store = _store(tmp_path)
    _free(monkeypatch, 10 * GB)
    idle_id = store.init(size=2 * GB, filename="a.mp4")  # would need 6 GB
    _make_idle(store, idle_id)
    store.init(size=2 * GB, filename="b.mp4")  # active, claims 6 GB of the 10 GB
    with pytest.raises(InsufficientDiskSpace):
        store.put_chunk(idle_id, 0, b"1234")


def test_idle_uploads_cannot_all_come_back_past_the_cap(monkeypatch, tmp_path):
    store = _store(tmp_path)
    _free(monkeypatch, 1000 * GB)
    extra = 3
    idle = []
    for _ in range(uploads.MAX_OPEN_UPLOADS + extra):  # idle uploads do not count, so init lets them all in
        upload_id = store.init(size=1024, filename="a.mp4")
        _make_idle(store, upload_id)
        idle.append(upload_id)
    refused = 0
    for upload_id in idle:
        try:
            store.put_chunk(upload_id, 0, b"1234")
        except InsufficientDiskSpace:
            refused += 1
    assert refused == extra and len(store._open_uploads()) == uploads.MAX_OPEN_UPLOADS


def test_the_number_of_active_unfinished_uploads_is_capped(monkeypatch, tmp_path):
    store = _store(tmp_path)
    _free(monkeypatch, 1000 * GB)
    for _ in range(uploads.MAX_OPEN_UPLOADS):
        store.init(size=1024, filename="a.mp4")
    with pytest.raises(InsufficientDiskSpace):
        store.init(size=1024, filename="a.mp4")
