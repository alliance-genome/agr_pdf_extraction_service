"""FastAPI proxy for the PDF Extraction GPU service."""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, Header, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, Response

from app.auth import CognitoAuth
from app.config import settings
from app.ec2_manager import EC2Manager
from app.job_queue import JobQueue, QueueFullError
from app.state_machine import InstanceState, LifecycleManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Singletons (initialized once) ---
ec2_mgr = EC2Manager()
lifecycle = LifecycleManager(ec2_mgr)
cognito_auth = CognitoAuth()
job_queue = JobQueue(max_size=settings.MAX_QUEUED_JOBS)
# Tracks queued proxy process IDs -> backend process IDs after replay.
proxy_to_backend_process: dict[str, str] = {}
# Tracks jobs actively being replayed to avoid transient 404 during queue drain.
replay_inflight_jobs: set[str] = set()
# Tracks replay submission failures so clients can fail fast instead of hanging in queued state.
replay_submission_errors: dict[str, str] = {}


def _queued_response(process_id: str) -> JSONResponse:
    """Standard queued response payload for startup buffering."""
    _stage = "ec2_starting" if lifecycle.state in (InstanceState.STARTING, InstanceState.STOPPED) else "queued"
    _display = "Spinning up GPU instance" if lifecycle.state in (InstanceState.STARTING, InstanceState.STOPPED) else "Job queued"
    return JSONResponse(
        status_code=202,
        content={
            "process_id": process_id,
            "status": "queued",
            "state": lifecycle.state.value,
            "message": "EC2 is starting. Job queued. Poll GET /api/v1/extract/{process_id} for status.",
            "retry_after": 30,
            "progress": {
                "stage": _stage,
                "stage_display": _display,
                "stages_completed": [],
                "stages_pending": [],
                "stages_total": 0,
                "stages_done": 0,
                "percent": 0,
            },
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Kick off EC2 state sync on boot without blocking container readiness."""
    sync_task = asyncio.create_task(lifecycle.sync_state_from_ec2())
    logger.info("Proxy startup: scheduled EC2 state sync task")
    try:
        yield
    finally:
        if not sync_task.done():
            sync_task.cancel()
            with suppress(asyncio.CancelledError):
                await sync_task


app = FastAPI(title="PDFX Proxy", version="1.0.0", lifespan=lifespan)


# --- Auth helper ---

def _require_auth(authorization: str | None) -> dict:
    return cognito_auth.validate_token(authorization or "")


# --- Health (no auth) ---

@app.get("/api/v1/health")
async def health():
    ec2_state = lifecycle.state.value
    result = {"proxy": "ok", "ec2": ec2_state}
    if lifecycle.state in (InstanceState.READY, InstanceState.BUSY) and lifecycle.private_ip:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{lifecycle.ec2_base_url}/api/v1/health")
                result["gpu_healthy"] = resp.status_code == 200
        except Exception:
            result["gpu_healthy"] = False
    return result


# --- Status (auth required) ---

@app.get("/api/v1/status")
async def status(authorization: str = Header(None)):
    _require_auth(authorization)
    lifecycle.touch()
    return {
        "state": lifecycle.state.value,
        "idle_minutes": round(lifecycle.idle_seconds / 60, 1),
        "active_jobs": lifecycle.active_jobs,
        "queued_jobs": job_queue.size,
    }


# --- Wake (auth required) ---

@app.post("/api/v1/wake")
async def wake(authorization: str = Header(None)):
    _require_auth(authorization)
    lifecycle.touch()
    await lifecycle.ensure_running()
    return {"state": lifecycle.state.value}


# --- Extract: submit PDF (auth required, auto-wake) ---

@app.post("/api/v1/extract")
async def submit_extraction(
    file: UploadFile = File(...),
    methods: str = Form("grobid,docling,marker"),
    merge: str = Form("true"),
    clear_cache: str = Form("false"),
    clear_cache_scope: str = Form(default=None),
    reference_curie: str = Form(default=None),
    mod_abbreviation: str = Form(default=None),
    authorization: str = Header(None),
):
    _require_auth(authorization)
    lifecycle.touch()

    process_id = str(uuid.uuid4())
    pdf_data = await file.read()
    filename = file.filename or "upload.pdf"

    form_fields = {
        "methods": methods,
        "merge": merge,
        "clear_cache": clear_cache,
    }
    if clear_cache_scope is not None:
        form_fields["clear_cache_scope"] = clear_cache_scope
    if reference_curie is not None:
        form_fields["reference_curie"] = reference_curie
    if mod_abbreviation is not None:
        form_fields["mod_abbreviation"] = mod_abbreviation

    # If EC2 is ready, forward immediately.
    # If forwarding fails (e.g., instance is transitioning), queue and recover.
    if lifecycle.state in (InstanceState.READY, InstanceState.BUSY):
        try:
            return await _forward_extraction(process_id, pdf_data, filename, form_fields)
        except HTTPException as exc:
            if exc.status_code != 502:
                raise
            logger.warning(
                "Immediate forward failed for process %s; queueing for startup replay instead.",
                process_id,
            )
            try:
                await lifecycle.sync_state_from_ec2()
            except Exception:
                logger.debug("sync_state_from_ec2 failed during forward recovery", exc_info=True)
            try:
                job_queue.enqueue(process_id, pdf_data, form_fields, filename)
            except QueueFullError:
                raise HTTPException(status_code=429, detail="Too many queued jobs. Try again later.")
            await lifecycle.ensure_running()
            asyncio.create_task(_replay_when_ready())
            return _queued_response(process_id)

    # Otherwise, queue the job and ensure EC2 is starting
    try:
        job_queue.enqueue(process_id, pdf_data, form_fields, filename)
    except QueueFullError:
        raise HTTPException(status_code=429, detail="Too many queued jobs. Try again later.")

    await lifecycle.ensure_running()

    # Start background task to replay queued jobs once EC2 is ready
    asyncio.create_task(_replay_when_ready())

    return _queued_response(process_id)


# --- Extract: poll status (auth required, proxied to EC2) ---

@app.get("/api/v1/extract/{process_id}")
async def get_extraction_status(process_id: str, authorization: str = Header(None)):
    _require_auth(authorization)
    lifecycle.touch()

    # If the job is still queued locally, report that
    if job_queue.has_job(process_id):
        if lifecycle.state == InstanceState.STARTING:
            _stage, _display = "ec2_starting", "Spinning up GPU instance"
        elif lifecycle.state in (InstanceState.READY, InstanceState.BUSY):
            _stage, _display = "queued", "Job queued, waiting for backend"
        else:
            _stage, _display = "ec2_starting", "Job queued, waiting for GPU instance"

        return {
            "process_id": process_id,
            "status": "queued",
            "state": lifecycle.state.value,
            "message": _display,
            "progress": {
                "stage": _stage,
                "stage_display": _display,
                "stages_completed": [],
                "stages_pending": [],
                "stages_total": 0,
                "stages_done": 0,
                "percent": 0,
            },
        }

    if process_id in replay_inflight_jobs:
        _display = "Submitting queued job to backend"
        return {
            "process_id": process_id,
            "status": "queued",
            "state": lifecycle.state.value,
            "message": _display,
            "progress": {
                "stage": "queued",
                "stage_display": _display,
                "stages_completed": [],
                "stages_pending": [],
                "stages_total": 0,
                "stages_done": 0,
                "percent": 0,
            },
        }

    if process_id in replay_submission_errors:
        error_message = replay_submission_errors[process_id]
        return {
            "process_id": process_id,
            "status": "failed",
            "state": lifecycle.state.value,
            "error": error_message,
            "message": "Failed to submit queued job to backend.",
            "progress": {
                "stage": "failed",
                "stage_display": "Queued job submission failed",
                "stages_completed": [],
                "stages_pending": [],
                "stages_total": 0,
                "stages_done": 0,
                "percent": 0,
            },
        }

    # Forward to EC2
    if lifecycle.state not in (InstanceState.READY, InstanceState.BUSY):
        raise HTTPException(status_code=503, detail="EC2 is not running")

    try:
        backend_process_id = proxy_to_backend_process.get(process_id, process_id)
        async with httpx.AsyncClient(timeout=settings.FORWARD_TIMEOUT_SECONDS) as client:
            resp = await client.get(f"{lifecycle.ec2_base_url}/api/v1/extract/{backend_process_id}")
            payload = resp.json()
            if isinstance(payload, dict) and backend_process_id != process_id:
                # Keep caller-visible process_id stable when queue replay rewrites backend IDs.
                payload["process_id"] = process_id
            return JSONResponse(status_code=resp.status_code, content=payload)
    except Exception as exc:
        logger.error("Failed to proxy status request: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to reach EC2 backend")


# --- Extract: download artifact output (auth required, proxied to EC2) ---

@app.get("/api/v1/extract/{process_id}/download/{method}")
async def download_extraction_output(
    process_id: str,
    method: str,
    authorization: str = Header(None),
):
    _require_auth(authorization)
    lifecycle.touch()

    if lifecycle.state not in (InstanceState.READY, InstanceState.BUSY):
        raise HTTPException(status_code=503, detail="EC2 is not running")

    backend_process_id = proxy_to_backend_process.get(process_id, process_id)
    download_url = f"{lifecycle.ec2_base_url}/api/v1/extract/{backend_process_id}/download/{method}"
    try:
        async with httpx.AsyncClient(
            timeout=settings.FORWARD_TIMEOUT_SECONDS,
            follow_redirects=True,
        ) as client:
            upstream = await client.get(download_url)

        content_type = upstream.headers.get("content-type", "application/octet-stream")
        content_disposition = upstream.headers.get("content-disposition")
        response_headers = {"content-type": content_type}
        if content_disposition:
            response_headers["content-disposition"] = content_disposition

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=response_headers,
        )
    except Exception as exc:
        logger.error("Failed to proxy download request for %s/%s: %s", process_id, method, exc)
        raise HTTPException(status_code=502, detail="Failed to reach EC2 backend")


# --- Proxy helpers ---

async def _forward_extraction(process_id: str, pdf_data: bytes, filename: str, form_fields: dict):
    """Forward a PDF extraction request to the EC2 backend."""
    lifecycle.job_started()
    try:
        async with httpx.AsyncClient(timeout=settings.FORWARD_TIMEOUT_SECONDS) as client:
            files = {"file": (filename, pdf_data, "application/pdf")}
            resp = await client.post(
                f"{lifecycle.ec2_base_url}/api/v1/extract",
                data=form_fields,
                files=files,
            )
            payload = resp.json()
            if isinstance(payload, dict):
                backend_process_id = str(payload.get("process_id", "")).strip()
                if backend_process_id:
                    proxy_to_backend_process[process_id] = backend_process_id
            return JSONResponse(status_code=resp.status_code, content=payload)
    except Exception as exc:
        logger.error("Failed to forward extraction to EC2: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to reach EC2 backend")
    finally:
        lifecycle.job_finished()


async def _replay_when_ready():
    """Wait for EC2 to become ready, then replay all queued jobs."""
    deadline = asyncio.get_event_loop().time() + settings.STARTUP_TIMEOUT_MINUTES * 60
    while asyncio.get_event_loop().time() < deadline:
        if lifecycle.state == InstanceState.READY:
            break
        if lifecycle.state == InstanceState.STOPPED:
            logger.error("EC2 startup failed. Dropping %d queued jobs.", job_queue.size)
            job_queue.drain()
            return
        await asyncio.sleep(5)
    else:
        logger.error("Timed out waiting for EC2. Dropping %d queued jobs.", job_queue.size)
        job_queue.drain()
        return

    # Replay all queued jobs
    jobs = job_queue.drain()
    logger.info("Replaying %d queued jobs to EC2", len(jobs))
    for job in jobs:
        replay_inflight_jobs.add(job.job_id)
        try:
            await _forward_extraction(job.job_id, job.pdf_data, job.filename, job.form_fields)
            replay_submission_errors.pop(job.job_id, None)
        except Exception as exc:
            logger.error("Failed to replay job %s: %s", job.job_id, exc)
            replay_submission_errors[job.job_id] = str(exc)
        finally:
            replay_inflight_jobs.discard(job.job_id)
