"""FastAPI proxy for the PDF Extraction GPU service."""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Sync EC2 state on proxy boot."""
    await lifecycle.sync_state_from_ec2()
    logger.info("Proxy started. EC2 state: %s", lifecycle.state)
    yield


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

    # If EC2 is ready, forward immediately
    if lifecycle.state in (InstanceState.READY, InstanceState.BUSY):
        return await _forward_extraction(process_id, pdf_data, filename, form_fields)

    # Otherwise, queue the job and ensure EC2 is starting
    try:
        job_queue.enqueue(process_id, pdf_data, form_fields, filename)
    except QueueFullError:
        raise HTTPException(status_code=429, detail="Too many queued jobs. Try again later.")

    await lifecycle.ensure_running()

    # Start background task to replay queued jobs once EC2 is ready
    asyncio.create_task(_replay_when_ready())

    return JSONResponse(
        status_code=202,
        content={
            "process_id": process_id,
            "state": lifecycle.state.value,
            "message": "EC2 is starting. Job queued. Poll GET /api/v1/extract/{process_id} for status.",
            "retry_after": 30,
        },
    )


# --- Extract: poll status (auth required, proxied to EC2) ---

@app.get("/api/v1/extract/{process_id}")
async def get_extraction_status(process_id: str, authorization: str = Header(None)):
    _require_auth(authorization)
    lifecycle.touch()

    # If the job is still queued locally, report that
    if job_queue.has_job(process_id):
        return {
            "process_id": process_id,
            "status": "queued",
            "state": lifecycle.state.value,
            "message": "Waiting for EC2 to start.",
        }

    # Forward to EC2
    if lifecycle.state not in (InstanceState.READY, InstanceState.BUSY):
        raise HTTPException(status_code=503, detail="EC2 is not running")

    try:
        async with httpx.AsyncClient(timeout=settings.FORWARD_TIMEOUT_SECONDS) as client:
            resp = await client.get(f"{lifecycle.ec2_base_url}/api/v1/extract/{process_id}")
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as exc:
        logger.error("Failed to proxy status request: %s", exc)
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
            return JSONResponse(status_code=resp.status_code, content=resp.json())
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
        try:
            await _forward_extraction(job.job_id, job.pdf_data, job.filename, job.form_fields)
        except Exception as exc:
            logger.error("Failed to replay job %s: %s", job.job_id, exc)
