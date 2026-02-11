"""
REST API v1 for the PDF Extraction Service.

Endpoints:
    GET  /api/v1/health              - Service health check
    POST /api/v1/extract             - Submit a PDF extraction job (async)
    GET  /api/v1/extract/<job_id>    - Poll job status / retrieve results
    GET  /api/v1/extract/<job_id>/download/<method>  - Download full extraction output
"""

import os
import uuid
import logging

from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

api = Blueprint("api", __name__, url_prefix="/api/v1")

VALID_METHODS = {"grobid", "docling", "marker"}


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
        active = inspector.active()
        checks["workers"] = len(active) if active else 0
    except Exception:
        checks["workers"] = "unknown"

    overall = "ok"
    if checks["grobid"] != "ok" or checks["redis"] != "ok":
        overall = "degraded"
    if checks["redis"] == "unavailable":
        overall = "unhealthy"

    status_code = 200 if overall != "unhealthy" else 503
    return jsonify({"status": overall, "checks": checks}), status_code


# ---------------------------------------------------------------------------
# Submit extraction job
# ---------------------------------------------------------------------------

@api.route("/extract", methods=["POST"])
def submit_extraction():
    """
    Submit a PDF for extraction.

    Accepts multipart/form-data with:
        file:    PDF file (required)
        methods: Comma-separated extractor names, e.g. "grobid,docling,marker"
                 (default: all three)
        merge:   "true" to run LLM merge (default: false)

    Returns 202 with a job_id for polling.
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

    merge = request.form.get("merge", "false").lower() in ("true", "1", "on")

    # --- Save uploaded file ---------------------------------------------------
    filename = secure_filename(file.filename)
    job_id = str(uuid.uuid4())
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_dir, exist_ok=True)
    # Prefix with job_id to avoid filename collisions
    pdf_path = os.path.join(upload_dir, f"{job_id}_{filename}")
    file.save(pdf_path)

    # --- Enqueue Celery task --------------------------------------------------
    from celery_app import extract_pdf

    task = extract_pdf.apply_async(
        args=[pdf_path, methods],
        kwargs={"merge": merge},
        task_id=job_id,
    )

    logger.info("Queued extraction job %s for %s (methods=%s, merge=%s)",
                job_id, filename, methods, merge)

    return jsonify({
        "job_id": task.id,
        "status": "queued",
        "methods": methods,
        "merge": merge,
    }), 202


# ---------------------------------------------------------------------------
# Poll job status
# ---------------------------------------------------------------------------

@api.route("/extract/<job_id>", methods=["GET"])
def get_extraction_status(job_id):
    """
    Poll an extraction job by its ID.

    Returns:
        - 200 with results when complete
        - 200 with progress info when still running
        - 404 if job not found
        - 500 if the job failed
    """
    from celery_app import celery

    result = celery.AsyncResult(job_id)

    if result.state == "PENDING":
        return jsonify({"job_id": job_id, "status": "pending"}), 200

    if result.state == "STARTED":
        return jsonify({"job_id": job_id, "status": "started"}), 200

    if result.state == "PROGRESS":
        return jsonify({
            "job_id": job_id,
            "status": "progress",
            "detail": result.info,
        }), 200

    if result.state == "SUCCESS":
        return jsonify({
            "job_id": job_id,
            "status": "complete",
            "result": result.result,
        }), 200

    if result.state == "FAILURE":
        return jsonify({
            "job_id": job_id,
            "status": "failed",
            "error": str(result.info),
        }), 500

    # Catch-all for other states (RETRY, REVOKED, etc.)
    return jsonify({
        "job_id": job_id,
        "status": result.state.lower(),
    }), 200


# ---------------------------------------------------------------------------
# Download full extraction output
# ---------------------------------------------------------------------------

@api.route("/extract/<job_id>/download/<method>", methods=["GET"])
def download_result(job_id, method):
    """
    Download the full markdown output for a completed extraction.

    The method can be 'grobid', 'docling', 'marker', or 'merged'.
    """
    if method not in (*VALID_METHODS, "merged"):
        return jsonify({"error": f"Invalid method: {method}"}), 400

    from celery_app import celery

    result = celery.AsyncResult(job_id)
    if result.state != "SUCCESS":
        return jsonify({"error": "Job not complete yet.", "status": result.state.lower()}), 409

    data = result.result
    download_paths = data.get("download_paths", {})
    file_hash = data.get("file_hash", "")

    if method == "merged":
        # Merged path isn't in download_paths; reconstruct it
        version = current_app.config["EXTRACTION_CONFIG_VERSION"]
        methods_used = data.get("methods_used", [])
        cache_key = f"v{version}_{file_hash}_{'_'.join(sorted(methods_used))}"
        filepath = os.path.join(current_app.config["CACHE_FOLDER"], f"{cache_key}_merged.md")
    else:
        filepath = download_paths.get(method)

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": f"Output file not found for method: {method}"}), 404

    return send_file(
        filepath,
        as_attachment=True,
        download_name=f"{file_hash}_{method}.md",
        mimetype="text/markdown",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf"}
