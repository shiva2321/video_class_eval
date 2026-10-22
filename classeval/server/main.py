import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from classeval.config import Settings
from classeval.server.auth import RateLimiter, get_auth_dependency
from classeval.server.jobs import InvalidId, JobStore
from classeval.server.uploads import (
    BadChunk,
    MissingChunks,
    UploadStore,
    UploadTooLarge,
)


class InitUploadRequest(BaseModel):
    size: int
    filename: str


class CreateJobRequest(BaseModel):
    type: str = "upload"
    upload_id: str = Field(pattern=r"^[A-Za-z0-9_-]{16,64}$")


def create_app(settings: Settings, limiter: RateLimiter | None = None) -> FastAPI:
    """FastAPI application factory for the Class Quality Assessor server."""
    app = FastAPI(title="8x8 Class Quality Assessor")

    # Per-app limiter. The default is generous enough for 8 MiB chunked uploads of a multi-GB file.
    active_limiter = limiter if limiter is not None else RateLimiter(max_requests=600, window_s=60.0)
    auth_dep = get_auth_dependency(settings, limiter=active_limiter)

    @app.exception_handler(InvalidId)
    async def _invalid_id(_request: Request, _exc: InvalidId) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    upload_store = UploadStore(
        root=settings.paths.uploads_dir,
        max_bytes=settings.limits.max_upload_bytes,
        chunk_bytes=settings.limits.chunk_bytes,
    )
    job_store = JobStore(root=settings.paths.jobs_dir)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=FileResponse)
    async def index() -> FileResponse:
        index_file = static_dir / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(index_file, media_type="text/html")

    # Upload routes
    @app.post("/api/uploads")
    async def init_upload(
        req: InitUploadRequest,
        _auth: bool = Depends(auth_dep),
    ) -> dict[str, str]:
        try:
            upload_id = upload_store.init(size=req.size, filename=req.filename)
            return {"upload_id": upload_id}
        except UploadTooLarge as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(exc),
            ) from exc

    @app.put("/api/uploads/{upload_id}/chunks/{chunk_index}")
    async def put_chunk(
        upload_id: str,
        chunk_index: int,
        request: Request,
        _auth: bool = Depends(auth_dep),
    ) -> dict[str, str]:
        limit = settings.limits.chunk_bytes
        buf = bytearray()
        async for part in request.stream():
            buf += part
            if len(buf) > limit:
                raise HTTPException(status_code=413, detail="Chunk too large")
        try:
            upload_store.put_chunk(upload_id=upload_id, index=chunk_index, data=bytes(buf))
            return {"status": "ok"}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Upload not found") from exc
        except BadChunk as exc:
            raise HTTPException(status_code=413 if exc.too_large else 400, detail=str(exc)) from exc

    @app.post("/api/uploads/{upload_id}/complete")
    async def complete_upload(
        upload_id: str,
        _auth: bool = Depends(auth_dep),
    ) -> Response:
        try:
            upload_store.complete(upload_id)
            return JSONResponse({"status": "ok"})
        except MissingChunks as exc:
            return JSONResponse(status_code=409, content={"missing": exc.indices})
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Upload not found") from exc

    # Job routes
    @app.post("/api/jobs")
    async def create_job(
        req: CreateJobRequest,
        _auth: bool = Depends(auth_dep),
    ) -> dict[str, Any]:
        job = job_store.create(source=req.model_dump())
        return {"job_id": job.id, "status": job.status}

    @app.get("/api/jobs/{job_id}")
    async def get_job(
        job_id: str,
        _auth: bool = Depends(auth_dep),
    ) -> dict[str, Any]:
        job = job_store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.model_dump()

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(
        job_id: str,
        request: Request,
        _auth: bool = Depends(auth_dep),
    ) -> EventSourceResponse:
        job = job_store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        async def event_generator():
            last_state = None
            while True:
                if await request.is_disconnected():
                    break
                current_job = job_store.get(job_id)
                if not current_job:
                    break

                state_key = (current_job.status, current_job.stage, round(current_job.progress, 3))
                if state_key != last_state:
                    last_state = state_key
                    event_data = {
                        "job_id": current_job.id,
                        "status": current_job.status,
                        "stage": current_job.stage,
                        "progress": current_job.progress,
                        "eta_s": current_job.eta_s,
                        "error": current_job.error,
                    }
                    yield {"event": "stage", "data": json.dumps(event_data)}
                    if current_job.status in ("done", "failed"):
                        break
                await asyncio.sleep(0.1)

        return EventSourceResponse(event_generator())

    @app.get("/api/jobs/{job_id}/report.json")
    async def get_report_json(
        job_id: str,
        _auth: bool = Depends(auth_dep),
    ) -> Response:
        rep_file = job_store.job_dir(job_id) / "report.json"
        if not rep_file.exists():
            raise HTTPException(status_code=404, detail="Report not ready")
        return Response(
            content=rep_file.read_text(encoding="utf-8"),
            media_type="application/json",
        )

    @app.get("/api/jobs/{job_id}/report.html")
    async def get_report_html(
        job_id: str,
        _auth: bool = Depends(auth_dep),
    ) -> HTMLResponse:
        rep_file = job_store.job_dir(job_id) / "report.html"
        if not rep_file.exists():
            raise HTTPException(status_code=404, detail="Report not ready")
        return HTMLResponse(content=rep_file.read_text(encoding="utf-8"))

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(
        job_id: str,
        _auth: bool = Depends(auth_dep),
    ) -> dict[str, str]:
        deleted = job_store.delete(job_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"status": "ok"}

    return app
