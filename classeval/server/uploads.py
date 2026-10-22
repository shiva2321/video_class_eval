import json
import re
import secrets
import shutil
from pathlib import Path

from fastapi import HTTPException, status

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
# The client's file name is display metadata only and is never used as an on-disk name.
FINAL_NAME = "upload.bin"
_TMP_NAME = "upload.tmp"


class InsufficientDiskSpace(HTTPException):
    """Raised when free disk space is less than 3x the declared upload size."""

    def __init__(self, detail: str = "Not enough free disk space on the server."):
        super().__init__(status_code=status.HTTP_507_INSUFFICIENT_STORAGE, detail=detail)


class UploadClosed(Exception):
    """Raised when a chunk is sent for an upload that has already been completed."""


class UploadTooLarge(Exception):
    """Raised when an initialized upload exceeds max_bytes."""


class MissingChunks(Exception):
    """Raised when attempting to complete an upload with missing chunks."""

    def __init__(self, indices: list[int]):
        super().__init__(f"Missing chunks: {indices}")
        self.indices = indices


class BadChunk(ValueError):
    """Chunk index out of range, or chunk larger than the configured chunk size."""

    def __init__(self, message: str, too_large: bool = False):
        super().__init__(message)
        self.too_large = too_large


class UploadStore:
    """Store for managing chunked file uploads."""

    def __init__(self, root: Path, max_bytes: int, chunk_bytes: int):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.chunk_bytes = chunk_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, upload_id: str) -> Path:
        # Ids come from the outside: only well-formed tokens may become a path (path-traversal guard).
        if not _ID_RE.fullmatch(upload_id):
            raise FileNotFoundError("Upload not found")
        up_dir = self.root / upload_id
        if not up_dir.is_dir():
            raise FileNotFoundError("Upload not found")
        return up_dir

    def _unfinished_claim(self) -> int:
        """Disk that uploads already in progress still need (3x their size, minus what they have stored), so many
        concurrent uploads cannot each pass the free-space check on their own and then fill the disk together."""
        claim = 0
        for meta_path in self.root.glob("*/meta.json"):
            up_dir = meta_path.parent
            if (up_dir / FINAL_NAME).exists():
                continue
            try:
                declared = int(json.loads(meta_path.read_text(encoding="utf-8"))["size"])
                stored = sum(p.stat().st_size for p in up_dir.glob("chunk_*.part"))
            except (OSError, ValueError, KeyError):
                continue  # an unreadable upload folder claims nothing; the purge job removes it later
            claim += max(0, 3 * declared - stored)
        return claim

    def init(self, size: int, filename: str) -> str:
        if size > self.max_bytes:
            raise UploadTooLarge(f"Size {size} exceeds limit of {self.max_bytes} bytes")

        if shutil.disk_usage(self.root).free - self._unfinished_claim() < 3 * size:
            raise InsufficientDiskSpace("Not enough free disk space on the server.")

        upload_id = secrets.token_urlsafe(16)
        up_dir = self.root / upload_id
        up_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = Path(filename.replace("\\", "/")).name or "recording.mp4"
        total_chunks = max(1, (size + self.chunk_bytes - 1) // self.chunk_bytes) if size > 0 else 1

        meta = {
            "size": size,
            "filename": safe_filename,
            "chunk_bytes": self.chunk_bytes,
            "total_chunks": total_chunks,
        }
        (up_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return upload_id

    def _meta(self, upload_id: str) -> dict:
        meta_path = self._dir(upload_id) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError("Upload not found")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def put_chunk(self, upload_id: str, index: int, data: bytes) -> None:
        up_dir = self._dir(upload_id)
        if (up_dir / FINAL_NAME).exists():
            raise UploadClosed("Upload already completed")
        meta = self._meta(upload_id)
        if not 0 <= index < meta["total_chunks"]:
            raise BadChunk("Chunk index out of range")
        if len(data) > self.chunk_bytes:
            raise BadChunk("Chunk too large", too_large=True)

        chunk_path = up_dir / f"chunk_{index}.part"
        tmp_path = up_dir / f"chunk_{index}.tmp"
        if shutil.disk_usage(self.root).free < 2 * len(data):
            raise InsufficientDiskSpace("Not enough free disk space on the server.")
        tmp_path.write_bytes(data)
        tmp_path.replace(chunk_path)

    def missing(self, upload_id: str) -> list[int]:
        meta = self._meta(upload_id)
        up_dir = self._dir(upload_id)
        return [i for i in range(meta["total_chunks"]) if not (up_dir / f"chunk_{i}.part").exists()]

    def complete(self, upload_id: str) -> Path:
        up_dir = self._dir(upload_id)
        target_path = up_dir / FINAL_NAME
        if target_path.exists():  # idempotent: completing twice returns the same file
            return target_path.resolve()

        missing_indices = self.missing(upload_id)
        if missing_indices:
            raise MissingChunks(missing_indices)

        meta = self._meta(upload_id)
        total_chunks = meta["total_chunks"]
        if shutil.disk_usage(self.root).free < int(meta["size"]):  # joining the chunks needs room for a second copy
            raise InsufficientDiskSpace("Not enough free disk space on the server.")
        tmp_target = up_dir / _TMP_NAME

        with open(tmp_target, "wb") as outfile:
            for i in range(total_chunks):
                with open(up_dir / f"chunk_{i}.part", "rb") as infile:
                    while chunk := infile.read(65536):
                        outfile.write(chunk)

        tmp_target.replace(target_path)

        for i in range(total_chunks):
            (up_dir / f"chunk_{i}.part").unlink(missing_ok=True)

        return target_path.resolve()

    def final_path(self, upload_id: str) -> Path | None:
        """Completed file of an upload, or None if the id is invalid or the upload is not complete."""
        try:
            path = self._dir(upload_id) / FINAL_NAME
        except FileNotFoundError:
            return None
        return path.resolve() if path.is_file() else None
