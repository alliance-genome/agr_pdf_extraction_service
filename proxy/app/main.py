"""FastAPI proxy for the PDF Extraction GPU service."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Body, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.auth import CognitoAuth
from app.config import settings
from app.ec2_manager import EC2Manager
from app.job_queue import BaseJobQueue, QueueFullError, QueuedJob, build_job_queue
from app.state_machine import InstanceState, LifecycleManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Singletons (initialized once) ---
ec2_mgr = EC2Manager()
lifecycle = LifecycleManager(ec2_mgr)
cognito_auth = CognitoAuth()
job_queue: BaseJobQueue = build_job_queue(max_size=settings.MAX_QUEUED_JOBS)

# Tracks queued proxy process IDs -> backend process IDs after replay.
proxy_to_backend_process: dict[str, str] = {}
# Tracks jobs actively being replayed to avoid transient 404 during queue drain.
replay_inflight_jobs: set[str] = set()
# Tracks terminal submission failures (replay/forward/reconciler) for fast-fail polling.
replay_submission_errors: dict[str, str] = {}

# Cached source payload for replay/reconciliation.
job_payload_cache: dict[str, QueuedJob] = {}
# Pending cancel requests keyed by proxy-visible process_id.
pending_cancel_requests: dict[str, str] = {}
# Local terminal cancel records for jobs cancelled before backend handoff.
cancelled_jobs: dict[str, str] = {}

# Background tasks.
replay_task: asyncio.Task | None = None
reconciler_task: asyncio.Task | None = None
canary_task: asyncio.Task | None = None


ACTIVE_JOB_STATUSES = {"queued", "pending", "started", "running", "progress", "warming_up", "cancel_requested"}
TERMINAL_JOB_STATUSES = {
    "complete",
    "completed",
    "succeeded",
    "success",
    "failed",
    "failure",
    "error",
    "cancelled",
    "canceled",
}
MAX_LOCAL_CANCELLED_JOBS = max(200, settings.MAX_QUEUED_JOBS * 20)


@dataclass
class JobTracker:
    process_id: str
    status: str = "queued"
    first_seen_at: float = 0.0
    last_seen_at: float = 0.0
    last_progress_signature: str = ""
    last_progress_at: float = 0.0
    requeue_attempted: bool = False


job_trackers: dict[str, JobTracker] = {}

# Lightweight canary state for observability.
canary_state: dict[str, Any] = {
    "enabled": settings.CANARY_INTERVAL_SECONDS > 0,
    "last_checked": None,
    "last_ok": None,
    "last_error": None,
    "consecutive_failures": 0,
}


def _ensure_tracker(process_id: str) -> JobTracker:
    now = time.time()
    tracker = job_trackers.get(process_id)
    if tracker is None:
        tracker = JobTracker(process_id=process_id, first_seen_at=now, last_seen_at=now, last_progress_at=now)
        job_trackers[process_id] = tracker
    return tracker


def _record_job_event(process_id: str, event: str, *, reason: str | None = None) -> None:
    tracker = _ensure_tracker(process_id)
    tracker.status = event
    tracker.last_seen_at = time.time()
    if event in TERMINAL_JOB_STATUSES:
        tracker.last_progress_at = tracker.last_seen_at
    if reason:
        logger.info("job=%s event=%s reason=%s", process_id, event, reason)
    else:
        logger.info("job=%s event=%s", process_id, event)


def _mark_job_failed(process_id: str, reason: str) -> None:
    replay_submission_errors[process_id] = reason
    _record_job_event(process_id, "failed", reason=reason)


def _clear_terminal_state(process_id: str) -> None:
    replay_submission_errors.pop(process_id, None)
    job_payload_cache.pop(process_id, None)
    pending_cancel_requests.pop(process_id, None)


def _drop_submission_state(process_id: str) -> None:
    """Drop cached job data for submissions that never entered durable processing."""
    _clear_terminal_state(process_id)
    replay_inflight_jobs.discard(process_id)
    proxy_to_backend_process.pop(process_id, None)
    job_trackers.pop(process_id, None)


def _remove_queued_job(process_id: str) -> bool:
    """Remove one queued job by ID without draining unrelated jobs."""
    return job_queue.remove_job(process_id)


def _record_job_cancelled(process_id: str, reason: str) -> None:
    # Keep a bounded terminal cache for locally-cancelled IDs so polling can
    # report stable terminal state without unbounded growth.
    if process_id not in cancelled_jobs and len(cancelled_jobs) >= MAX_LOCAL_CANCELLED_JOBS:
        oldest_process_id = next(iter(cancelled_jobs))
        cancelled_jobs.pop(oldest_process_id, None)
    cancelled_jobs[process_id] = reason
    replay_submission_errors.pop(process_id, None)
    _record_job_event(process_id, "cancelled", reason=reason)


def _active_backend_jobs() -> int:
    count = 0
    for process_id, tracker in job_trackers.items():
        if process_id in replay_submission_errors:
            continue
        if tracker.status in ACTIVE_JOB_STATUSES:
            count += 1
    return count


def _oldest_pending_age_seconds() -> float:
    now = time.time()
    ages: list[float] = []
    queue_age = job_queue.oldest_age_seconds()
    if queue_age > 0:
        ages.append(queue_age)

    for process_id, tracker in job_trackers.items():
        if process_id in replay_submission_errors:
            continue
        if tracker.status in ACTIVE_JOB_STATUSES:
            ages.append(max(0.0, now - tracker.last_progress_at))

    return max(ages) if ages else 0.0


def _can_stop_ec2() -> bool:
    if replay_inflight_jobs:
        return False
    if job_queue.size > 0:
        return False
    if _active_backend_jobs() > 0:
        return False
    return True


lifecycle.set_stop_guard(_can_stop_ec2)


def _progress_payload(stage: str, stage_display: str, percent: int = 0) -> dict[str, Any]:
    return {
        "stage": stage,
        "stage_display": stage_display,
        "stages_completed": [],
        "stages_pending": [],
        "stages_total": 0,
        "stages_done": 0,
        "percent": int(max(0, min(100, percent))),
    }


def _queued_response(process_id: str, status_value: str = "queued") -> JSONResponse:
    """Standard queued/warming response payload for startup buffering."""
    is_starting = lifecycle.state in (InstanceState.STARTING, InstanceState.STOPPED)
    if is_starting:
        stage = "ec2_starting"
        display = "Spinning up GPU instance"
        message = "GPU worker is waking up. Job queued. Poll GET /api/v1/extract/{process_id} for status."
    else:
        stage = "queued"
        display = "Job queued"
        message = "Job queued. Poll GET /api/v1/extract/{process_id} for status."

    return JSONResponse(
        status_code=202,
        content={
            "process_id": process_id,
            "status": status_value,
            "state": lifecycle.state.value,
            "message": message,
            "retry_after": 30,
            "progress": _progress_payload(stage, display, 0),
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Kick off EC2 state sync and reconciliation loops on boot."""
    global reconciler_task, canary_task

    sync_task = asyncio.create_task(lifecycle.sync_state_from_ec2())
    reconciler_task = asyncio.create_task(_reconciler_loop())
    if settings.CANARY_INTERVAL_SECONDS > 0:
        canary_task = asyncio.create_task(_canary_loop())

    logger.info("Proxy startup: scheduled EC2 state sync and maintenance tasks")
    try:
        yield
    finally:
        for task in (sync_task, reconciler_task, canary_task):
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


app = FastAPI(title="PDFX Proxy", version="1.1.0", lifespan=lifespan)


# --- Auth helper ---

def _require_auth(authorization: str | None) -> dict:
    return cognito_auth.validate_token(authorization or "")


# --- Health (no auth) ---

@app.get("/api/v1/health")
async def health():
    ec2_state = lifecycle.state.value
    result: dict[str, Any] = {
        "proxy": "ok",
        "status": "degraded",
        "ec2": ec2_state,
        "queue_depth": job_queue.size,
        "queue_durable": job_queue.durable,
        "active_jobs": lifecycle.active_jobs,
        "active_backend_jobs": _active_backend_jobs(),
    }

    if lifecycle.state not in (InstanceState.READY, InstanceState.BUSY):
        result["reason"] = "worker_sleeping_or_starting"
        return result

    if not lifecycle.private_ip:
        result["reason"] = "missing_worker_ip"
        return result

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{lifecycle.ec2_base_url}/api/v1/health")
        result["gpu_healthy"] = resp.status_code == 200

        payload: dict[str, Any] | None = None
        try:
            payload = resp.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            result["gpu_status"] = payload.get("status")
            checks = payload.get("checks")
            if isinstance(checks, dict):
                result["gpu_workers"] = checks.get("workers")
                result["gpu_redis"] = checks.get("redis")
                result["gpu_grobid"] = checks.get("grobid")

        result["status"] = "healthy" if resp.status_code == 200 else "degraded"
        if resp.status_code != 200:
            result["reason"] = f"downstream_health_status_{resp.status_code}"
            logger.error("Proxy health returned degraded; downstream health status=%s", resp.status_code)
    except Exception as exc:  # pragma: no cover - defensive log path
        result["gpu_healthy"] = False
        result["status"] = "degraded"
        result["reason"] = "downstream_unreachable"
        result["error"] = str(exc)
        logger.error("Proxy health reported degraded: downstream unreachable: %s", exc)

    return result


@app.get("/api/v1/health/deep")
async def health_deep():
    """Deep health check: proxy auth validation + downstream status round trip."""
    token = settings.HEALTHCHECK_BEARER_TOKEN or settings.CANARY_BEARER_TOKEN

    # Auth contract check:
    # - If probe token configured, validate it directly.
    # - Otherwise verify auth guard rejects missing header (expected 401).
    auth_valid = False
    auth_mode = "guard_rejects_missing_header"
    auth_error = None
    if token:
        auth_mode = "token_validation"
        try:
            _require_auth(f"Bearer {token}")
            auth_valid = True
        except HTTPException as exc:
            auth_error = str(exc.detail)
    else:
        try:
            _require_auth(None)
            auth_error = "Auth guard accepted empty header unexpectedly"
        except HTTPException as exc:
            if exc.status_code == 401:
                auth_valid = True
            else:
                auth_error = f"Unexpected auth guard status: {exc.status_code}"

    downstream_ok = False
    downstream_status = None
    downstream_error = None

    if lifecycle.state in (InstanceState.READY, InstanceState.BUSY) and lifecycle.private_ip:
        probe_process_id = f"health-probe-{uuid.uuid4()}"
        try:
            request_headers = {"Authorization": f"Bearer {token}"} if token else {}
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{lifecycle.ec2_base_url}/api/v1/extract/{probe_process_id}",
                    headers=request_headers,
                )
            downstream_status = resp.status_code
            # With token: expect normal handled status.
            # Without token: expect auth rejection but still confirms downstream reachability.
            if token:
                downstream_ok = resp.status_code in {200, 404, 422}
            else:
                # Some backend environments are reachable without auth on private network.
                # For token-less deep checks, any non-5xx downstream response confirms roundtrip.
                downstream_ok = resp.status_code < 500
        except Exception as exc:  # pragma: no cover - defensive log path
            downstream_error = str(exc)
    else:
        downstream_error = "worker_not_ready"

    deep_status = "healthy" if auth_valid and downstream_ok else "degraded"
    if deep_status != "healthy":
        logger.error(
            "Deep health degraded auth_valid=%s downstream_ok=%s status=%s error=%s",
            auth_valid,
            downstream_ok,
            downstream_status,
            downstream_error or auth_error,
        )

    return {
        "status": deep_status,
        "ec2": lifecycle.state.value,
        "auth_check_mode": auth_mode,
        "proxy_auth_valid": auth_valid,
        "proxy_auth_error": auth_error,
        "downstream_roundtrip_ok": downstream_ok,
        "downstream_status_code": downstream_status,
        "downstream_error": downstream_error,
        "queue_depth": job_queue.size,
        "queue_durable": job_queue.durable,
    }


@app.get("/api/v1/metrics")
async def metrics():
    """Operational metrics for alerting dashboards."""
    return {
        "queue_depth": job_queue.size,
        "queue_durable": job_queue.durable,
        "oldest_pending_age_seconds": round(_oldest_pending_age_seconds(), 2),
        "replay_failure_count": len(replay_submission_errors),
        "replay_inflight_count": len(replay_inflight_jobs),
        "active_backend_jobs": _active_backend_jobs(),
        "ec2_stop_events_total": lifecycle.stop_events_total,
        "ec2_stop_blocked_total": lifecycle.stop_blocked_total,
        "canary": canary_state,
    }


# --- Status (auth required) ---

@app.get("/api/v1/status")
async def status(authorization: str = Header(None)):
    _require_auth(authorization)
    lifecycle.touch()

    warming = False
    if lifecycle.state == InstanceState.STOPPED:
        await lifecycle.ensure_running()
        warming = True

    return {
        "state": lifecycle.state.value,
        "warming_up": warming or lifecycle.state == InstanceState.STARTING,
        "idle_minutes": round(lifecycle.idle_seconds / 60, 1),
        "ready_minutes": round(lifecycle.ready_seconds / 60, 1),
        "active_jobs": lifecycle.active_jobs,
        "active_backend_jobs": _active_backend_jobs(),
        "queued_jobs": job_queue.size,
        "queue_durable": job_queue.durable,
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

    queued_job = QueuedJob(
        job_id=process_id,
        pdf_data=pdf_data,
        form_fields=form_fields,
        filename=filename,
        authorization=authorization,
    )
    job_payload_cache[process_id] = queued_job
    _record_job_event(process_id, "queued")

    # If EC2 is ready, forward immediately.
    # If forwarding fails due transient startup/transport issue, queue and recover.
    if lifecycle.state in (InstanceState.READY, InstanceState.BUSY):
        try:
            return await _forward_extraction(
                process_id,
                pdf_data,
                filename,
                form_fields,
                authorization=authorization,
            )
        except HTTPException as exc:
            if exc.status_code in {401, 403, 422}:
                _drop_submission_state(process_id)
                raise
            logger.warning(
                "Immediate forward failed for process %s; queueing for replay. detail=%s",
                process_id,
                exc.detail,
            )
            try:
                await lifecycle.sync_state_from_ec2()
            except Exception:
                logger.debug("sync_state_from_ec2 failed during forward recovery", exc_info=True)
            try:
                job_queue.enqueue(
                    process_id,
                    pdf_data,
                    form_fields,
                    filename,
                    authorization=authorization,
                )
            except QueueFullError:
                _drop_submission_state(process_id)
                raise HTTPException(status_code=429, detail="Too many queued jobs. Try again later.")
            await lifecycle.ensure_running()
            _ensure_replay_task()
            return _queued_response(process_id, status_value="queued")

    # Otherwise, queue the job and ensure EC2 is starting.
    try:
        job_queue.enqueue(
            process_id,
            pdf_data,
            form_fields,
            filename,
            authorization=authorization,
        )
    except QueueFullError:
        _drop_submission_state(process_id)
        raise HTTPException(status_code=429, detail="Too many queued jobs. Try again later.")

    await lifecycle.ensure_running()
    _ensure_replay_task()

    return _queued_response(process_id, status_value="warming_up")


# --- Extract: poll status (auth required, proxied to EC2) ---

@app.get("/api/v1/extract/{process_id}")
async def get_extraction_status(process_id: str, authorization: str = Header(None)):
    _require_auth(authorization)
    lifecycle.touch()

    if process_id in cancelled_jobs:
        reason = cancelled_jobs[process_id]
        return {
            "process_id": process_id,
            "status": "cancelled",
            "state": lifecycle.state.value,
            "message": reason,
            "progress": _progress_payload("cancelled", "Job cancelled", 0),
        }

    # If the job is still queued locally, report that.
    if job_queue.has_job(process_id):
        if lifecycle.state == InstanceState.STARTING:
            stage, display = "ec2_starting", "Spinning up GPU instance"
            status_value = "queued"
        elif lifecycle.state in (InstanceState.READY, InstanceState.BUSY):
            stage, display = "queued", "Job queued, waiting for backend"
            status_value = "queued"
        else:
            stage, display = "ec2_starting", "Job queued, waiting for GPU instance"
            status_value = "queued"

        _record_job_event(process_id, status_value)
        return {
            "process_id": process_id,
            "status": status_value,
            "state": lifecycle.state.value,
            "message": display,
            "progress": _progress_payload(stage, display, 0),
        }

    if process_id in replay_inflight_jobs:
        if process_id in pending_cancel_requests:
            display = "Cancellation requested while submitting queued job"
            status_value = "cancel_requested"
        else:
            display = "Submitting queued job to backend"
            status_value = "queued"
        _record_job_event(process_id, status_value)
        return {
            "process_id": process_id,
            "status": status_value,
            "state": lifecycle.state.value,
            "message": display,
            "progress": _progress_payload("queued", display, 0),
        }

    if process_id in replay_submission_errors:
        error_message = replay_submission_errors[process_id]
        return {
            "process_id": process_id,
            "status": "failed",
            "state": lifecycle.state.value,
            "error": error_message,
            "message": "Failed to submit queued job to backend.",
            "progress": _progress_payload("failed", "Queued job submission failed", 0),
        }

    # On status checks while asleep/stopped, wake instance and return warming state.
    if lifecycle.state not in (InstanceState.READY, InstanceState.BUSY):
        await lifecycle.ensure_running()
        _record_job_event(process_id, "warming_up")
        return {
            "process_id": process_id,
            "status": "warming_up",
            "state": lifecycle.state.value,
            "message": "Worker is starting. Please retry status shortly.",
            "progress": _progress_payload("ec2_starting", "Spinning up GPU instance", 0),
        }

    # Forward to EC2.
    try:
        backend_process_id = proxy_to_backend_process.get(process_id, process_id)
        headers = {"Authorization": authorization} if authorization else {}
        async with httpx.AsyncClient(timeout=settings.FORWARD_TIMEOUT_SECONDS) as client:
            status_url = f"{lifecycle.ec2_base_url}/api/v1/extract/{backend_process_id}"
            if headers:
                try:
                    resp = await client.get(status_url, headers=headers)
                except TypeError:
                    resp = await client.get(status_url)
            else:
                resp = await client.get(status_url)

        payload = _coerce_json_payload(resp)
        if isinstance(payload, dict):
            if backend_process_id != process_id:
                # Keep caller-visible process_id stable when replay rewrites backend IDs.
                payload["process_id"] = process_id
            _update_tracker_from_payload(process_id, payload)
            _mark_terminal_cleanup_if_needed(process_id, payload)

        return JSONResponse(status_code=resp.status_code, content=payload)
    except Exception as exc:
        logger.error("Failed to proxy status request: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to reach EC2 backend")


async def _forward_cancel_to_backend(
    process_id: str,
    *,
    authorization: str | None,
    reason: str,
) -> JSONResponse:
    backend_process_id = proxy_to_backend_process.get(process_id, process_id)
    cancel_url = f"{lifecycle.ec2_base_url}/api/v1/extract/{backend_process_id}/cancel"

    async with httpx.AsyncClient(timeout=settings.FORWARD_TIMEOUT_SECONDS) as client:
        if authorization:
            try:
                upstream = await client.post(
                    cancel_url,
                    headers={"Authorization": authorization},
                    json={"reason": reason},
                )
            except TypeError:
                upstream = await client.post(cancel_url, json={"reason": reason})
        else:
            upstream = await client.post(cancel_url, json={"reason": reason})

    payload = _coerce_json_payload(upstream)
    if isinstance(payload, dict):
        if backend_process_id != process_id:
            payload["process_id"] = process_id
        _update_tracker_from_payload(process_id, payload)
        _mark_terminal_cleanup_if_needed(process_id, payload)
        status_value = str(payload.get("status", "")).strip().lower()
        if status_value in {"cancelled", "canceled"}:
            _record_job_cancelled(process_id, str(payload.get("message") or reason))
        elif status_value == "cancel_requested":
            pending_cancel_requests[process_id] = reason
            _record_job_event(process_id, "cancel_requested", reason=reason)

    return JSONResponse(status_code=upstream.status_code, content=payload)


@app.post("/api/v1/extract/{process_id}/cancel")
async def cancel_extraction(
    process_id: str,
    cancellation: dict[str, Any] | None = Body(default=None),
    authorization: str = Header(None),
):
    _require_auth(authorization)
    lifecycle.touch()
    reason = str((cancellation or {}).get("reason", "")).strip() or "Cancelled by user request"

    if process_id in cancelled_jobs:
        return JSONResponse(
            status_code=409,
            content={
                "process_id": process_id,
                "status": "cancelled",
                "message": cancelled_jobs[process_id],
            },
        )

    if _remove_queued_job(process_id):
        _clear_terminal_state(process_id)
        replay_inflight_jobs.discard(process_id)
        proxy_to_backend_process.pop(process_id, None)
        _record_job_cancelled(process_id, reason)
        return JSONResponse(
            status_code=202,
            content={
                "process_id": process_id,
                "status": "cancelled",
                "message": reason,
            },
        )

    if process_id in replay_inflight_jobs:
        pending_cancel_requests[process_id] = reason
        _record_job_event(process_id, "cancel_requested", reason=reason)
        return JSONResponse(
            status_code=202,
            content={
                "process_id": process_id,
                "status": "cancel_requested",
                "message": reason,
            },
        )

    if lifecycle.state not in (InstanceState.READY, InstanceState.BUSY):
        return JSONResponse(
            status_code=503,
            content={
                "process_id": process_id,
                "status": "unavailable",
                "message": "EC2 is not running; unable to forward cancellation request.",
            },
        )

    try:
        return await _forward_cancel_to_backend(
            process_id,
            authorization=authorization,
            reason=reason,
        )
    except Exception as exc:
        logger.error("Failed to proxy cancel request for %s: %s", process_id, exc)
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
            if authorization:
                try:
                    upstream = await client.get(download_url, headers={"Authorization": authorization})
                except TypeError:
                    upstream = await client.get(download_url)
            else:
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


def _coerce_json_payload(resp: httpx.Response) -> dict[str, Any]:
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            return payload
        return {"detail": payload}
    except ValueError:
        text = resp.text.strip()
        return {"detail": text or f"Non-JSON response (status={resp.status_code})"}


async def _forward_extraction(
    process_id: str,
    pdf_data: bytes,
    filename: str,
    form_fields: dict,
    *,
    authorization: str | None,
) -> JSONResponse:
    """Forward a PDF extraction request to the EC2 backend."""
    lifecycle.job_started()
    try:
        async with httpx.AsyncClient(timeout=settings.FORWARD_TIMEOUT_SECONDS) as client:
            files = {"file": (filename, pdf_data, "application/pdf")}
            headers = {"Authorization": authorization} if authorization else None
            resp = await client.post(
                f"{lifecycle.ec2_base_url}/api/v1/extract",
                data=form_fields,
                files=files,
                headers=headers,
            )

        payload = _coerce_json_payload(resp)

        if resp.status_code >= 400:
            detail = payload.get("detail") or f"HTTP {resp.status_code}"
            raise HTTPException(
                status_code=502 if resp.status_code >= 500 else resp.status_code,
                detail=f"Backend extraction submit failed ({resp.status_code}): {detail}",
            )

        backend_process_id = str(payload.get("process_id", "")).strip()
        if backend_process_id:
            proxy_to_backend_process[process_id] = backend_process_id

        _record_job_event(process_id, "accepted_by_backend")
        replay_submission_errors.pop(process_id, None)

        return JSONResponse(status_code=resp.status_code, content=payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to forward extraction to EC2: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to reach EC2 backend")
    finally:
        lifecycle.job_finished()


def _ensure_replay_task() -> None:
    global replay_task
    if replay_task and not replay_task.done():
        return
    replay_task = asyncio.create_task(_replay_when_ready())


async def _replay_when_ready():
    """Wait for EC2 to become ready, then replay all queued jobs."""
    deadline = asyncio.get_event_loop().time() + settings.STARTUP_TIMEOUT_MINUTES * 60
    while asyncio.get_event_loop().time() < deadline:
        if lifecycle.state in (InstanceState.READY, InstanceState.BUSY):
            break
        if lifecycle.state == InstanceState.STOPPED:
            jobs = job_queue.drain()
            for job in jobs:
                _mark_job_failed(job.job_id, "EC2 startup failed before queued replay.")
            logger.error("EC2 startup failed. Marked %d queued jobs as failed.", len(jobs))
            return
        await asyncio.sleep(5)
    else:
        jobs = job_queue.drain()
        for job in jobs:
            _mark_job_failed(job.job_id, "EC2 startup timed out before queued replay.")
        logger.error("Timed out waiting for EC2. Marked %d queued jobs as failed.", len(jobs))
        return

    jobs = job_queue.drain()
    logger.info("Replaying %d queued jobs to EC2", len(jobs))
    for job in jobs:
        replay_inflight_jobs.add(job.job_id)

    for job in jobs:
        _record_job_event(job.job_id, "replayed")
        try:
            await _forward_extraction(
                job.job_id,
                job.pdf_data,
                job.filename,
                job.form_fields,
                authorization=job.authorization,
            )
            if job.job_id in pending_cancel_requests:
                reason = pending_cancel_requests.pop(job.job_id)
                try:
                    await _forward_cancel_to_backend(
                        job.job_id,
                        authorization=job.authorization,
                        reason=reason,
                    )
                except Exception as cancel_exc:
                    logger.error("Failed to enforce pending cancellation for %s: %s", job.job_id, cancel_exc)
                    pending_cancel_requests[job.job_id] = reason
        except Exception as exc:
            logger.error("Failed to replay job %s: %s", job.job_id, exc)
            _mark_job_failed(job.job_id, str(exc))
        finally:
            replay_inflight_jobs.discard(job.job_id)


async def _reconciler_loop():
    """Close stale pending/running records and optionally requeue once."""
    interval = max(15, settings.RECONCILER_INTERVAL_SECONDS)
    stale_after = max(1, settings.STUCK_PENDING_MINUTES) * 60

    while True:
        await asyncio.sleep(interval)
        now = time.time()

        for process_id, tracker in list(job_trackers.items()):
            if process_id in replay_submission_errors:
                continue
            if tracker.status not in ACTIVE_JOB_STATUSES:
                continue

            age = now - tracker.last_progress_at
            if age < stale_after:
                continue

            if settings.RECONCILER_REQUEUE_ONCE and not tracker.requeue_attempted and process_id in job_payload_cache:
                tracker.requeue_attempted = True
                job = job_payload_cache[process_id]
                logger.warning("Reconciler requeueing stale job %s after %.0fs", process_id, age)
                try:
                    if lifecycle.state in (InstanceState.READY, InstanceState.BUSY):
                        await _forward_extraction(
                            process_id,
                            job.pdf_data,
                            job.filename,
                            job.form_fields,
                            authorization=job.authorization,
                        )
                    else:
                        job_queue.enqueue(
                            process_id,
                            job.pdf_data,
                            job.form_fields,
                            job.filename,
                            authorization=job.authorization,
                        )
                        await lifecycle.ensure_running()
                        _ensure_replay_task()
                        _record_job_event(process_id, "queued")
                    continue
                except Exception as exc:
                    _mark_job_failed(process_id, f"Requeue attempt failed: {exc}")
                    continue

            _mark_job_failed(
                process_id,
                "Progress monitoring timeout in proxy. "
                "The worker may still be processing; backend extraction timeout is separate.",
            )


async def _canary_loop():
    """Periodic canary round-trip against downstream status endpoint."""
    interval = max(30, settings.CANARY_INTERVAL_SECONDS)
    token = settings.CANARY_BEARER_TOKEN or settings.HEALTHCHECK_BEARER_TOKEN

    while True:
        await asyncio.sleep(interval)
        canary_state["last_checked"] = time.time()

        if lifecycle.state not in (InstanceState.READY, InstanceState.BUSY) or not lifecycle.private_ip:
            canary_state["last_ok"] = False
            canary_state["last_error"] = "worker_not_ready"
            continue

        if not token:
            canary_state["last_ok"] = False
            canary_state["last_error"] = "probe_token_not_configured"
            continue

        process_id = f"canary-{uuid.uuid4()}"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{lifecycle.ec2_base_url}/api/v1/extract/{process_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
            ok = resp.status_code in {200, 404, 422}
            canary_state["last_ok"] = ok
            if ok:
                canary_state["last_error"] = None
                canary_state["consecutive_failures"] = 0
            else:
                canary_state["last_error"] = f"unexpected_status_{resp.status_code}"
                canary_state["consecutive_failures"] += 1
                logger.error("Canary failed: status=%s", resp.status_code)
        except Exception as exc:  # pragma: no cover - defensive log path
            canary_state["last_ok"] = False
            canary_state["last_error"] = str(exc)
            canary_state["consecutive_failures"] += 1
            logger.error("Canary failed: %s", exc)


def _update_tracker_from_payload(process_id: str, payload: dict[str, Any]) -> None:
    tracker = _ensure_tracker(process_id)
    status = str(payload.get("status", "")).strip().lower() or tracker.status

    progress = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
    stage = str(progress.get("stage", "")).strip().lower()
    percent = progress.get("percent")
    signature = f"{status}|{stage}|{percent}"

    tracker.status = status
    tracker.last_seen_at = time.time()
    if signature != tracker.last_progress_signature:
        tracker.last_progress_signature = signature
        tracker.last_progress_at = tracker.last_seen_at


def _mark_terminal_cleanup_if_needed(process_id: str, payload: dict[str, Any]) -> None:
    status = str(payload.get("status", "")).strip().lower()
    if status in {"complete", "completed", "succeeded", "success"}:
        _record_job_event(process_id, "complete")
        cancelled_jobs.pop(process_id, None)
        _clear_terminal_state(process_id)
    elif status in {"cancelled", "canceled"}:
        _record_job_cancelled(process_id, str(payload.get("message") or "Cancelled by user request"))
        _clear_terminal_state(process_id)
    elif status in {"failed", "failure", "error"}:
        detail = payload.get("error") or payload.get("detail") or "Backend reported failure"
        _mark_job_failed(process_id, str(detail))
        cancelled_jobs.pop(process_id, None)
        _clear_terminal_state(process_id)
