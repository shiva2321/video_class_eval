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
