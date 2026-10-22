import asyncio
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from classeval.schema import Report
from tests.fixtures.clips import make_clip


async def _async_run_smoke(
    base_url: str,
    access_code: str,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 60.0,
) -> tuple[bool, str]:
    start_time = time.monotonic()
    headers = {"X-Access-Code": access_code}

    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_path = Path(tmp_dir) / "smoke_30s.mp4"
        try:
            make_clip(clip_path, seconds=30.0)
        except Exception as exc:  # noqa: BLE001
            return False, f"FAIL ffmpeg synthetic clip generation failed: {exc}"

        clip_bytes = clip_path.read_bytes()
        total_size = len(clip_bytes)
        chunk_size = 8 * 1024 * 1024

        client_kwargs: dict[str, Any] = {"base_url": base_url, "timeout": timeout}
        if transport is not None:
            client_kwargs["transport"] = transport

        async with httpx.AsyncClient(**client_kwargs) as client:
            # 1. Init upload
            init_resp = await client.post(
                "/api/uploads",
                json={"size": total_size, "filename": clip_path.name},
                headers=headers,
            )
            if init_resp.status_code != 200:
                return False, f"FAIL init upload failed with status {init_resp.status_code}: {init_resp.text}"
            upload_id = init_resp.json().get("upload_id")
            if not upload_id:
                return False, "FAIL init upload returned no upload_id"

            # 2. Upload chunks
            total_chunks = max(1, (total_size + chunk_size - 1) // chunk_size)
            for idx in range(total_chunks):
                chunk = clip_bytes[idx * chunk_size : (idx + 1) * chunk_size]
                chunk_resp = await client.put(
                    f"/api/uploads/{upload_id}/chunks/{idx}",
                    content=chunk,
                    headers={"Content-Type": "application/octet-stream", **headers},
                )
                if chunk_resp.status_code != 200:
                    return (
                        False,
                        f"FAIL upload chunk {idx} failed with status {chunk_resp.status_code}: {chunk_resp.text}",
                    )

            # 3. Complete upload
            comp_resp = await client.post(f"/api/uploads/{upload_id}/complete", headers=headers)
            if comp_resp.status_code != 200:
                return False, f"FAIL complete upload failed with status {comp_resp.status_code}: {comp_resp.text}"

            # 4. Create job
            job_resp = await client.post(
                "/api/jobs",
                json={"type": "upload", "upload_id": upload_id},
                headers=headers,
            )
            if job_resp.status_code != 200:
                return False, f"FAIL create job failed with status {job_resp.status_code}: {job_resp.text}"
            job_id = job_resp.json().get("job_id")
            if not job_id:
                return False, "FAIL create job returned no job_id"

            # 5. Wait for done
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                poll_resp = await client.get(f"/api/jobs/{job_id}", headers=headers)
                if poll_resp.status_code != 200:
                    return False, f"FAIL poll job failed with status {poll_resp.status_code}: {poll_resp.text}"
                job_data = poll_resp.json()
                status = job_data.get("status")
                if status == "done":
                    break
                if status == "failed":
                    err = job_data.get("error")
                    return False, f"FAIL job execution failed: {err}"
                await asyncio.sleep(0.2)
            else:
                return False, f"FAIL job timed out after {timeout}s"

            # 6. Validate report
            rep_resp = await client.get(f"/api/jobs/{job_id}/report.json", headers=headers)
            if rep_resp.status_code != 200:
                return False, f"FAIL get report.json failed with status {rep_resp.status_code}: {rep_resp.text}"
            try:
                Report.model_validate(rep_resp.json())
            except Exception as exc:  # noqa: BLE001
                return False, f"FAIL report validation error: {exc}"

    elapsed = round(time.monotonic() - start_time, 2)
    return True, f"OK {elapsed}s"


def run_smoke(
    base_url: str,
    access_code: str,
    transport: Any = None,
    timeout: float = 60.0,
) -> tuple[bool, str]:
    """Synchronous entry point for the smoke test."""
    return asyncio.run(
        _async_run_smoke(base_url, access_code, transport=transport, timeout=timeout)
    )


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/smoke_public.py <base_url> <access_code>")
        sys.exit(1)

    base_url = sys.argv[1]
    access_code = sys.argv[2]
    ok, msg = run_smoke(base_url, access_code)
    print(msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
