"""
REST API v1 for the PDF Extraction Service.

Endpoints:
    GET  /api/v1/health                                - Service health check
    GET  /api/v1/extractions                           - List all extraction runs
    POST /api/v1/extract                               - Submit a PDF extraction job (async)
    GET  /api/v1/extract/<process_id>                  - Poll job status / retrieve results
    POST /api/v1/extract/<process_id>/cancel           - Cancel an active extraction job
    GET  /api/v1/extract/<process_id>/download/<method> - Download full extraction output
    GET  /api/v1/extract/<process_id>/images            - List extracted images
    GET  /api/v1/extract/<process_id>/images/<filename> - Download a single image
    GET  /api/v1/extract/<process_id>/logs              - Pre-signed URL for NDJSON audit log
    GET  /api/v1/extract/<process_id>/artifacts         - Artifact S3 keys from extraction_run
    GET  /api/v1/extract/<process_id>/artifacts/urls    - Pre-signed URLs for all artifact keys
"""

import os
import uuid
import logging
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename

from app.models import ExtractionRun, get_session
from app.services.audit_logger import build_s3_client

logger = logging.getLogger(__name__)

api = Blueprint("api", __name__, url_prefix="/api/v1")

VALID_METHODS = {"grobid", "docling", "marker"}
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf"}


def _parse_clear_cache_scope(form):
    """Parse and validate cache-clear options from multipart form data.

    Delegates to celery_app._normalize_clear_cache_scope for the single
    source of truth on alias resolution and validation.
    """
    from celery_app import _normalize_clear_cache_scope

    clear_cache = form.get("clear_cache", "false").lower() in ("true", "1", "on")
    clear_cache_scope_raw = form.get("clear_cache_scope")
    clear_cache_scope = _normalize_clear_cache_scope(
        clear_cache_scope=clear_cache_scope_raw,
        clear_cache=clear_cache,
    )
    return clear_cache, clear_cache_scope


def _to_iso(dt):
    if not dt:
        return None
    return dt.isoformat().replace("+00:00", "Z")


def _map_db_status(status):
    if status == "queued":
        return "pending"
    if status == "running":
        return "started"
    if status == "succeeded":
        return "complete"
    return status


def _get_db_session():
    try:
        return get_session()
    except Exception as exc:
        logger.warning("Database unavailable for API request: %s", exc)
        return None


def _insert_queued_run(process_id, reference_curie, mod_abbreviation):
    db_session = _get_db_session()
    if not db_session:
        return

    try:
        existing = db_session.get(ExtractionRun, process_id)
        if existing is None:
            db_session.add(ExtractionRun(
                process_id=process_id,
                reference_curie=reference_curie,
                mod_abbreviation=mod_abbreviation,
                config_version=current_app.config.get("EXTRACTION_CONFIG_VERSION"),
                status="queued",
            ))
            db_session.commit()
    except Exception as exc:
        logger.warning("Failed to insert queued extraction_run row for %s: %s", process_id, exc)
        try:
            db_session.rollback()
        except Exception:
            pass
    finally:
        try:
            db_session.close()
        except Exception:
            pass


def _get_run_by_process_id(process_id):
    db_session = _get_db_session()
    if not db_session:
        return None

    try:
        run = db_session.get(ExtractionRun, process_id)
        if run is None:
            return None

        return {
            "process_id": run.process_id,
            "reference_curie": run.reference_curie,
            "mod_abbreviation": run.mod_abbreviation,
            "source_pdf_md5": run.source_pdf_md5,
            "config_version": run.config_version,
            "status": run.status,
            "started_at": _to_iso(run.started_at),
            "ended_at": _to_iso(run.ended_at),
            "error_code": run.error_code,
            "error_message": run.error_message,
            "artifacts_json": run.artifacts_json,
            "consensus_metrics_json": run.consensus_metrics_json,
            "log_s3_key": run.log_s3_key,
            "llm_usage_json": run.llm_usage_json,
            "llm_cost_usd": float(run.llm_cost_usd) if run.llm_cost_usd is not None else None,
        }
    except Exception as exc:
        logger.warning("Failed to query extraction_run for %s: %s", process_id, exc)
        return None
    finally:
        try:
            db_session.close()
        except Exception:
            pass


def _collect_artifact_keys(value, path="$"):
    """Collect S3 keys from artifact JSON, filtering out non-S3 strings like filenames."""
    keys = []
    if isinstance(value, str):
        # Only treat as S3 key if it looks like one (contains '/' suggesting a path prefix)
        if "/" in value:
            keys.append((path, value))
        return keys
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.extend(_collect_artifact_keys(nested, f"{path}.{key}"))
        return keys
    if isinstance(value, list):
        for idx, nested in enumerate(value):
            keys.extend(_collect_artifact_keys(nested, f"{path}[{idx}]"))
    return keys


def _s3_redirect_for_artifact(s3_key):
    """Generate a pre-signed URL for an S3 key and return a redirect response."""
    from flask import redirect

    s3_client = build_s3_client(current_app.config)
    if not s3_client:
        return None

    bucket = current_app.config.get("AUDIT_S3_BUCKET")
    if not bucket:
        return None

    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": s3_key},
            ExpiresIn=3600,
        )
        return redirect(url, code=302)
    except Exception as exc:
        logger.warning("Failed to generate S3 redirect for key %s: %s", s3_key, exc)
        return None


def _find_artifact_s3_key(artifacts_json, method):
    """Look up an S3 key for a given extraction method in artifacts_json."""
    if not artifacts_json or not isinstance(artifacts_json, dict):
        return None
    return artifacts_json.get(method)


def _find_image_s3_key(artifacts_json, filename):
    """Look up an S3 key for a given image filename in artifacts_json."""
    if not artifacts_json or not isinstance(artifacts_json, dict):
        return None
    images = artifacts_json.get("images")
    if not isinstance(images, list):
        return None
    for img in images:
        if isinstance(img, dict) and img.get("filename") == filename:
            return img.get("s3_key")
    return None


def _mark_run_cancelled(process_id, reason):
    """Best-effort mark extraction_run row as cancelled."""
    db_session = _get_db_session()
    if not db_session:
        return

    try:
        run = db_session.get(ExtractionRun, process_id)
        if run is None:
            return

        if run.status in {"succeeded", "failed", "cancelled"}:
            return

        run.status = "cancelled"
        run.ended_at = datetime.now(timezone.utc)
        run.error_code = "cancelled"
        run.error_message = reason
        db_session.commit()
    except Exception as exc:
        logger.warning("Failed to mark extraction_run cancelled for %s: %s", process_id, exc)
        try:
            db_session.rollback()
        except Exception:
            pass
    finally:
        try:
            db_session.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@api.route("/health", methods=["GET"])
def health():
    """Return service health including GROBID and Redis connectivity."""
    checks = {"service": "ok"}

    # GROBID
    try:
        import requests as req
        grobid_url = current_app.config["GROBID_URL"]
        resp = req.get(f"{grobid_url}/api/isalive", timeout=3)
        checks["grobid"] = "ok" if resp.status_code == 200 else "degraded"
    except Exception:
        checks["grobid"] = "unavailable"

    # Redis
    try:
        import redis
        r = redis.from_url(current_app.config["CELERY_BROKER_URL"])
        r.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unavailable"

    # Celery workers
    try:
        from celery_app import celery
        inspector = celery.control.inspect(timeout=2)
        # Count live worker nodes, not active tasks.
        # `active()` can be empty when workers are idle.
        ping = inspector.ping() if inspector else None
        checks["workers"] = len(ping) if ping else 0
    except Exception:
        checks["workers"] = "unknown"

    overall = "ok"
    if checks["grobid"] != "ok" or checks["redis"] != "ok":
        overall = "degraded"
    if checks["workers"] == "unknown":
        overall = "degraded"
    if checks["redis"] == "unavailable":
        overall = "unhealthy"
    if isinstance(checks["workers"], int) and checks["workers"] <= 0:
        overall = "unhealthy"

    status_code = 200 if overall != "unhealthy" else 503
    return jsonify({"status": overall, "checks": checks}), status_code


# ---------------------------------------------------------------------------
# Config introspection (for benchmark verification)
# ---------------------------------------------------------------------------

@api.route("/config", methods=["GET"])
def get_config():
    """Return current LLM model and reasoning config for verification.

    This endpoint lets operators confirm that per-call-type model overrides
    are actually loaded into the running process (e.g., after container
    restart vs. recreate).
    """
    from config import Config
    return jsonify({
        "llm_model_default": Config.LLM_MODEL,
        "llm_reasoning_default": Config.LLM_REASONING_EFFORT,
        "per_call_type": {
            "zone_resolution": {
                "model": Config.LLM_MODEL_ZONE_RESOLUTION,
                "reasoning": Config.LLM_REASONING_ZONE_RESOLUTION or Config.LLM_REASONING_EFFORT,
            },
            "conflict_batch": {
                "model": Config.LLM_MODEL_CONFLICT_BATCH,
                "reasoning": Config.LLM_REASONING_CONFLICT_BATCH or Config.LLM_REASONING_EFFORT,
            },
            "general_rescue": {
                "model": Config.LLM_MODEL_GENERAL_RESCUE,
                "reasoning": Config.LLM_REASONING_GENERAL_RESCUE or Config.LLM_REASONING_EFFORT,
            },
            "numeric_rescue": {
                "model": Config.LLM_MODEL_NUMERIC_RESCUE,
                "reasoning": Config.LLM_REASONING_NUMERIC_RESCUE or Config.LLM_REASONING_EFFORT,
            },
            "header_hierarchy": {
                "model": Config.HIERARCHY_LLM_MODEL,
                "reasoning": Config.HIERARCHY_LLM_REASONING,
            },
        },
    })


# ---------------------------------------------------------------------------
# List extraction runs
# ---------------------------------------------------------------------------

@api.route("/extractions", methods=["GET"])
def list_extractions():
    """List extraction runs with optional filtering and pagination.

    Query parameters:
        status:           Filter by status (queued, running, succeeded, failed)
        reference_curie:  Filter by reference curie
        mod_abbreviation: Filter by MOD abbreviation
        limit:            Max results (default 50, max 200)
        offset:           Skip first N results (default 0)
    """
    db_session = _get_db_session()
    if not db_session:
        return jsonify({"error": "Database unavailable"}), 503

    try:
        query = db_session.query(ExtractionRun).order_by(
            ExtractionRun.started_at.desc().nullsfirst()
        )

        status_filter = request.args.get("status")
        if status_filter:
            if status_filter == "pending":
                status_filter = "queued"
            elif status_filter == "started":
                status_filter = "running"
            elif status_filter == "complete":
                status_filter = "succeeded"
            query = query.filter(ExtractionRun.status == status_filter)

        curie_filter = request.args.get("reference_curie")
        if curie_filter:
            query = query.filter(ExtractionRun.reference_curie == curie_filter)

        mod_filter = request.args.get("mod_abbreviation")
        if mod_filter:
            query = query.filter(ExtractionRun.mod_abbreviation == mod_filter)

        total = query.count()

        limit = min(int(request.args.get("limit", 50)), 200)
        offset = int(request.args.get("offset", 0))
        runs = query.offset(offset).limit(limit).all()

        items = []
        for run in runs:
            items.append({
                "process_id": run.process_id,
                "status": _map_db_status(run.status),
                "reference_curie": run.reference_curie,
                "mod_abbreviation": run.mod_abbreviation,
                "source_pdf_md5": run.source_pdf_md5,
                "config_version": run.config_version,
                "started_at": _to_iso(run.started_at),
                "ended_at": _to_iso(run.ended_at),
                "error_code": run.error_code,
                "has_artifacts": bool(run.artifacts_json),
                "has_log": bool(run.log_s3_key),
            })

        return jsonify({
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": items,
        }), 200

    except Exception as exc:
        logger.warning("Failed to list extraction runs: %s", exc)
        return jsonify({"error": "Failed to query extraction runs"}), 500
    finally:
        try:
            db_session.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Submit extraction job
# ---------------------------------------------------------------------------

@api.route("/extract", methods=["POST"])
def submit_extraction():
    """
    Submit a PDF for extraction.

    Accepts multipart/form-data with:
        file:             PDF file (required)
        methods:          Comma-separated extractor names, e.g. "grobid,docling,marker"
                          (default: all three)
        merge:            "true" to run LLM merge (default: false)
        clear_cache:      "true" to clear any existing cached results for this PDF
                          before processing (default: false)
        clear_cache_scope: Cache clear mode. One of:
                          "none" (default), "merge", "extraction", "all".
                          If omitted, clear_cache=true maps to "all".
        reference_curie:  Optional curie from upstream system
        mod_abbreviation: Optional MOD abbreviation from upstream system

    Returns 202 with a process_id for polling.
    """
    # --- Validate file --------------------------------------------------------
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send a PDF as 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    # --- Parse parameters -----------------------------------------------------
    methods_param = request.form.get("methods", "grobid,docling,marker")
    methods = [m.strip().lower() for m in methods_param.split(",")]
    invalid = set(methods) - VALID_METHODS
    if invalid:
        return jsonify({"error": f"Invalid methods: {', '.join(invalid)}. "
                        f"Valid: {', '.join(sorted(VALID_METHODS))}"}), 400

    merge = request.form.get("merge", "true").lower() in ("true", "1", "on")
    try:
        clear_cache, clear_cache_scope = _parse_clear_cache_scope(request.form)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    reference_curie = request.form.get("reference_curie")
    mod_abbreviation = request.form.get("mod_abbreviation")

    # --- Save uploaded file ---------------------------------------------------
    filename = secure_filename(file.filename)
    process_id = str(uuid.uuid4())
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_dir, exist_ok=True)
    pdf_path = os.path.join(upload_dir, f"{process_id}_{filename}")
    file.save(pdf_path)

    # --- Enqueue Celery task FIRST --------------------------------------------
    # Enqueue before DB insert so we never create an orphan "queued" row
    # that has no corresponding Celery task.
    from celery_app import extract_pdf

    try:
        task = extract_pdf.apply_async(
            args=[pdf_path, methods],
            kwargs={
                "merge": merge,
                "clear_cache": clear_cache,
                "clear_cache_scope": clear_cache_scope,
                "process_id": process_id,
                "reference_curie": reference_curie,
                "mod_abbreviation": mod_abbreviation,
            },
            task_id=process_id,
        )
    except Exception as exc:
        logger.error("Failed to enqueue extraction task: %s", exc)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        return jsonify({"error": "Failed to enqueue extraction task. Is Redis available?"}), 503

    # Best effort: create queued DB row after successful enqueue.
    _insert_queued_run(process_id, reference_curie, mod_abbreviation)

    logger.info(
        "Queued extraction %s for %s (methods=%s, merge=%s, clear_cache_scope=%s, reference_curie=%s, mod=%s)",
        process_id,
        filename,
        methods,
        merge,
        clear_cache_scope,
        reference_curie,
        mod_abbreviation,
    )

    return jsonify({
        "process_id": process_id,
        "status": "queued",
        "methods": methods,
        "merge": merge,
        "clear_cache_scope": clear_cache_scope,
        "reference_curie": reference_curie,
        "mod_abbreviation": mod_abbreviation,
    }), 202


# ---------------------------------------------------------------------------
# Poll job status
# ---------------------------------------------------------------------------

@api.route("/extract/<process_id>", methods=["GET"])
def get_extraction_status(process_id):
    """
    Poll an extraction run by process_id.

    Primary source is the extraction_run DB table. Falls back to Celery
    result backend if the DB row hasn't been created yet (brief window
    between enqueue and worker startup, or if DB was temporarily unavailable).
    """
    run = _get_run_by_process_id(process_id)
    if run is not None:
        response = {
            "process_id": run["process_id"],
            "status": _map_db_status(run["status"]),
            "reference_curie": run["reference_curie"],
            "mod_abbreviation": run["mod_abbreviation"],
            "source_pdf_md5": run["source_pdf_md5"],
            "started_at": run["started_at"],
            "ended_at": run["ended_at"],
            "log_s3_key": run["log_s3_key"],
            "artifacts_json": run["artifacts_json"],
            "consensus_metrics_json": run["consensus_metrics_json"],
            "llm_cost_usd": run["llm_cost_usd"],
            "llm_usage_json": run["llm_usage_json"],
        }
        if run["error_code"] or run["error_message"]:
            response["error_code"] = run["error_code"]
            response["error"] = run["error_message"]

        # For non-terminal DB states, reconcile with Celery truth to avoid stale "pending".
        if run["status"] in {"queued", "running"}:
            try:
                from celery_app import celery
                result = celery.AsyncResult(process_id)
                if result.state == "SUCCESS":
                    response["status"] = "complete"
                    if isinstance(result.result, dict):
                        response["reference_curie"] = result.result.get("reference_curie") or response["reference_curie"]
                        response["started_at"] = result.result.get("started_at") or response["started_at"]
                        response["ended_at"] = result.result.get("ended_at") or response["ended_at"]
                        response["log_s3_key"] = result.result.get("log_s3_key") or response["log_s3_key"]
                        response["artifacts_json"] = result.result.get("artifacts_json") or response["artifacts_json"]
                        response["llm_usage_json"] = result.result.get("llm_usage_json") or response["llm_usage_json"]
                        response["llm_cost_usd"] = result.result.get("llm_cost_usd") or response["llm_cost_usd"]
                    return jsonify(response), 200

                if result.state == "FAILURE":
                    response["status"] = "failed"
                    response["error"] = str(result.info)
                    return jsonify(response), 500
                if result.state == "REVOKED":
                    response["status"] = "cancelled"
                    response["error"] = run.get("error_message") or "Extraction cancelled"
                    return jsonify(response), 200

                if result.state == "STARTED":
                    response["status"] = "started"

                if result.state == "PROGRESS" and result.info:
                    info = result.info
                    response["status"] = "progress"
                    response["progress"] = {
                        "stage": info.get("stage", info.get("step", "unknown")),
                        "stage_display": info.get("stage_display", ""),
                        "stages_completed": info.get("stages_completed", []),
                        "stages_pending": info.get("stages_pending", []),
                        "stages_total": info.get("stages_total", 0),
                        "stages_done": info.get("stages_done", 0),
                        "percent": info.get("percent", 0),
                    }
            except Exception as exc:
                logger.debug("Could not fetch Celery state for %s: %s", process_id, exc)

        if run["status"] == "failed":
            return jsonify(response), 500

        return jsonify(response), 200

    # Celery fallback: DB row may not exist yet if the worker hasn't
    # started or if the DB was temporarily unavailable at enqueue time.
    from celery_app import celery

    result = celery.AsyncResult(process_id)

    if result.state == "PENDING":
        return jsonify({"process_id": process_id, "status": "pending"}), 200

    if result.state == "STARTED":
        return jsonify({"process_id": process_id, "status": "started"}), 200

    if result.state == "PROGRESS":
        info = result.info or {}
        return jsonify({
            "process_id": process_id,
            "status": "progress",
            "progress": {
                "stage": info.get("stage", info.get("step", "unknown")),
                "stage_display": info.get("stage_display", ""),
                "stages_completed": info.get("stages_completed", []),
                "stages_pending": info.get("stages_pending", []),
                "stages_total": info.get("stages_total", 0),
                "stages_done": info.get("stages_done", 0),
                "percent": info.get("percent", 0),
            },
            "detail": info,
        }), 200

    if result.state == "SUCCESS":
        payload = {
            "process_id": process_id,
            "status": "complete",
            "result": result.result,
        }
        if isinstance(result.result, dict):
            payload["reference_curie"] = result.result.get("reference_curie")
            payload["started_at"] = result.result.get("started_at")
            payload["ended_at"] = result.result.get("ended_at")
            payload["log_s3_key"] = result.result.get("log_s3_key")
        return jsonify(payload), 200

    if result.state == "FAILURE":
        return jsonify({
            "process_id": process_id,
            "status": "failed",
            "error": str(result.info),
        }), 500
    if result.state == "REVOKED":
        return jsonify({
            "process_id": process_id,
            "status": "cancelled",
            "message": "Extraction cancelled",
        }), 200

    return jsonify({
        "process_id": process_id,
        "status": result.state.lower(),
    }), 200


@api.route("/extract/<process_id>/cancel", methods=["POST"])
def cancel_extraction(process_id):
    """Cancel an active extraction run by process_id."""
    reason = (request.get_json(silent=True) or {}).get("reason") or "Cancelled by user request"
    run = _get_run_by_process_id(process_id)
    if run and run["status"] in {"succeeded", "failed", "cancelled"}:
        return jsonify({
            "process_id": process_id,
            "status": _map_db_status(run["status"]),
            "message": f"Job already terminal ({run['status']})",
        }), 409

    from celery_app import celery

    try:
        celery.control.revoke(process_id, terminate=True, signal="SIGTERM")
    except Exception as exc:
        logger.warning("Failed to revoke extraction task %s: %s", process_id, exc)

    _mark_run_cancelled(process_id, reason)

    return jsonify({
        "process_id": process_id,
        "status": "cancelled",
        "message": "Cancellation requested",
    }), 202


# ---------------------------------------------------------------------------
# Audit log and artifacts
# ---------------------------------------------------------------------------

@api.route("/extract/<process_id>/logs", methods=["GET"])
def get_extraction_log_url(process_id):
    """Return a pre-signed URL for the NDJSON audit log in S3."""
    run = _get_run_by_process_id(process_id)
    if run is None:
        return jsonify({"error": "process_id not found"}), 404

    log_s3_key = run.get("log_s3_key")
    if not log_s3_key:
        return jsonify({"error": "No audit log available for this process"}), 404

    s3_client = build_s3_client(current_app.config)
    if not s3_client:
        return jsonify({"error": "S3 client unavailable (check AWS credentials)"}), 503

    bucket = current_app.config.get("AUDIT_S3_BUCKET")
    expires_in = 3600

    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": log_s3_key},
            ExpiresIn=expires_in,
        )
        return jsonify({"log_url": url, "expires_in": expires_in}), 200
    except Exception as exc:
        logger.warning("Failed to generate pre-signed URL for process_id=%s: %s", process_id, exc)
        return jsonify({"error": "Unable to generate log URL"}), 500


@api.route("/extract/<process_id>/artifacts", methods=["GET"])
def get_extraction_artifacts(process_id):
    """Return artifacts_json recorded for an extraction process."""
    run = _get_run_by_process_id(process_id)
    if run is None:
        return jsonify({"error": "process_id not found"}), 404

    return jsonify({
        "process_id": process_id,
        "artifacts_json": run.get("artifacts_json") or {},
    }), 200


@api.route("/extract/<process_id>/artifacts/urls", methods=["GET"])
def get_extraction_artifact_urls(process_id):
    """Return pre-signed S3 URLs for every artifact key stored on the run."""
    run = _get_run_by_process_id(process_id)
    if run is None:
        return jsonify({"error": "process_id not found"}), 404

    artifacts = run.get("artifacts_json") or {}
    key_entries = _collect_artifact_keys(artifacts)
    if not key_entries:
        return jsonify({"error": "No artifacts available for this process"}), 404

    s3_client = build_s3_client(current_app.config)
    if not s3_client:
        return jsonify({"error": "S3 client unavailable (check AWS credentials)"}), 503

    bucket = current_app.config.get("AUDIT_S3_BUCKET")
    expires_in = 3600
    urls = []

    for name, s3_key in key_entries:
        try:
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": s3_key},
                ExpiresIn=expires_in,
            )
            urls.append({"name": name, "s3_key": s3_key, "url": url})
        except Exception as exc:
            logger.warning("Failed to generate artifact URL for process_id=%s key=%s: %s", process_id, s3_key, exc)
            urls.append({"name": name, "s3_key": s3_key, "error": "Unable to generate URL"})

    return jsonify({
        "process_id": process_id,
        "expires_in": expires_in,
        "artifact_urls": urls,
    }), 200


# ---------------------------------------------------------------------------
# Download full extraction output
# ---------------------------------------------------------------------------

@api.route("/extract/<process_id>/download/<method>", methods=["GET"])
def download_result(process_id, method):
    """
    Download the full markdown output for a completed extraction.

    The method can be 'grobid', 'docling', 'marker', or 'merged'.
    Tries local cache first, falls back to S3 artifact if cache is missing.
    """
    if method not in (*VALID_METHODS, "merged", "audit"):
        return jsonify({"error": f"Invalid method: {method}"}), 400

    from celery_app import celery

    # Try local cache via Celery result
    result = celery.AsyncResult(process_id)
    if result.state == "SUCCESS":
        data = result.result
        download_paths = data.get("download_paths", {})
        file_hash = data.get("file_hash", "")

        if method == "merged":
            filepath = data.get("merged_cache_path")
            if not filepath:
                version = current_app.config["EXTRACTION_CONFIG_VERSION"]
                methods_used = data.get("methods_used", [])
                cache_key = f"v{version}_{file_hash}_{'_'.join(sorted(methods_used))}"
                filepath = os.path.join(current_app.config["CACHE_FOLDER"], f"{cache_key}_merged.md")
        elif method == "audit":
            filepath = download_paths.get("audit")
        else:
            filepath = download_paths.get(method)

        if filepath and os.path.exists(filepath):
            if method == "audit":
                download_name = f"{file_hash}_audit.json"
                mimetype = "application/json"
            else:
                download_name = f"{file_hash}_{method}.md"
                mimetype = "text/markdown"
            return send_file(
                filepath,
                as_attachment=True,
                download_name=download_name,
                mimetype=mimetype,
            )

    # Fallback: look up S3 artifact from DB
    run = _get_run_by_process_id(process_id)
    if run is None:
        return jsonify({"error": "process_id not found"}), 404

    if run["status"] not in ("succeeded", "complete"):
        return jsonify({"error": "Job not complete yet.", "status": run["status"]}), 409

    s3_key = _find_artifact_s3_key(run.get("artifacts_json"), method)
    if s3_key:
        resp = _s3_redirect_for_artifact(s3_key)
        if resp:
            return resp

    return jsonify({"error": f"Output file not found for method: {method}"}), 404


# ---------------------------------------------------------------------------
# Image endpoints
# ---------------------------------------------------------------------------

@api.route("/extract/<process_id>/images", methods=["GET"])
def list_job_images(process_id):
    """List extracted images for a completed job.

    Tries Celery result first, falls back to DB artifacts_json.
    """
    from celery_app import celery

    result = celery.AsyncResult(process_id)
    if result.state == "SUCCESS":
        data = result.result
        images = data.get("images", [])
        return jsonify({"process_id": process_id, "images": images}), 200

    # Fallback: DB artifacts
    run = _get_run_by_process_id(process_id)
    if run is None:
        return jsonify({"error": "process_id not found"}), 404

    if run["status"] not in ("succeeded", "complete"):
        return jsonify({"error": "Job not complete yet.", "status": run["status"]}), 409

    artifacts = run.get("artifacts_json") or {}
    images = artifacts.get("images", [])
    return jsonify({"process_id": process_id, "images": images, "source": "s3"}), 200


@api.route("/extract/<process_id>/images/<filename>", methods=["GET"])
def download_job_image(process_id, filename):
    """Download a single extracted image for a completed job.

    Tries local cache first, falls back to S3 artifact.
    """
    from celery_app import celery
    from werkzeug.utils import safe_join
    from app.utils import get_images_dir

    # Try local cache via Celery result
    result = celery.AsyncResult(process_id)
    if result.state == "SUCCESS":
        data = result.result
        file_hash = data.get("file_hash", "")
        images_dir = get_images_dir(file_hash)
        filepath = safe_join(images_dir, filename)
        if filepath and os.path.isfile(filepath):
            if os.path.realpath(filepath).startswith(os.path.realpath(images_dir)):
                import mimetypes
                mime = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
                return send_file(filepath, mimetype=mime)

    # Fallback: S3 artifact from DB
    run = _get_run_by_process_id(process_id)
    if run is None:
        return jsonify({"error": "process_id not found"}), 404

    if run["status"] not in ("succeeded", "complete"):
        return jsonify({"error": "Job not complete yet.", "status": run["status"]}), 409

    s3_key = _find_image_s3_key(run.get("artifacts_json"), filename)
    if s3_key:
        resp = _s3_redirect_for_artifact(s3_key)
        if resp:
            return resp

    return jsonify({"error": "Image not found"}), 404
