import os
import uuid
import time
import logging
from datetime import datetime, timezone

from celery import Celery

from config import Config
from app.models import ExtractionRun, get_session
from app.services.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------
celery = Celery(
    "pdf_extraction",
    broker=Config.CELERY_BROKER_URL,
    backend=Config.CELERY_RESULT_BACKEND,
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=86400,  # 24 hours
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # one job at a time per worker process
    task_soft_time_limit=1800,     # 30 min soft limit (raises SoftTimeLimitExceeded)
    task_time_limit=2100,          # 35 min hard kill
)


# ---------------------------------------------------------------------------
# Helpers (standalone -- no Flask app context needed)
# ---------------------------------------------------------------------------

def _now_utc():
    return datetime.now(timezone.utc)


def _safe_close_session(session):
    if not session:
        return
    try:
        session.close()
    except Exception:
        pass


def _get_file_hash(file_path):
    """MD5 hash of a file."""
    import hashlib
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_path(file_hash, method):
    """Cache path includes config version so config changes invalidate old outputs."""
    version = Config.EXTRACTION_CONFIG_VERSION
    return os.path.join(Config.CACHE_FOLDER, f"v{version}_{file_hash}_{method}.md")


def _get_images_dir(file_hash):
    """Compute images directory for a file hash (standalone, no Flask context)."""
    marker_cache_path = _cached_path(file_hash, "marker")
    stem = os.path.splitext(os.path.basename(marker_cache_path))[0]
    return os.path.join(Config.CACHE_FOLDER, "images", stem)


def _list_images(file_hash):
    """List image files for a file_hash (standalone, no Flask context)."""
    images_dir = _get_images_dir(file_hash)
    if not os.path.isdir(images_dir):
        return []
    result = []
    for f in sorted(os.listdir(images_dir)):
        fpath = os.path.join(images_dir, f)
        if os.path.isfile(fpath) and f.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            result.append({"filename": f, "size_bytes": os.path.getsize(fpath)})
    return result


def _rewrite_image_paths(markdown, file_hash):
    """Rewrite image paths at response time (standalone, no Flask context)."""
    import re
    from urllib.parse import quote
    images_dir = _get_images_dir(file_hash)

    def _replace(match):
        alt, filename = match.group(1), match.group(2)
        basename = os.path.basename(filename)
        if os.path.exists(os.path.join(images_dir, basename)):
            return f"![{alt}](/download/{file_hash}/images/{quote(basename)})"
        return match.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _replace, markdown)


def _is_cached(file_hash, method):
    return os.path.exists(_cached_path(file_hash, method))


def _get_db_session():
    try:
        return get_session()
    except Exception as exc:
        logger.warning("Database unavailable for extraction tracking: %s", exc)
        return None


def _upsert_extraction_run(
    db_session,
    process_id,
    reference_curie=None,
    mod_abbreviation=None,
    source_pdf_md5=None,
    source_referencefile_id=None,
    config_version=None,
    status=None,
    started_at=None,
    ended_at=None,
    error_code=None,
    error_message=None,
    artifacts_json=None,
    log_s3_key=None,
):
    if not db_session:
        return None

    run = db_session.get(ExtractionRun, process_id)
    if run is None:
        run = ExtractionRun(process_id=process_id)
        db_session.add(run)

    if reference_curie is not None:
        run.reference_curie = reference_curie
    if mod_abbreviation is not None:
        run.mod_abbreviation = mod_abbreviation
    if source_pdf_md5 is not None:
        run.source_pdf_md5 = source_pdf_md5
    if source_referencefile_id is not None:
        run.source_referencefile_id = source_referencefile_id
    if config_version is not None:
        run.config_version = config_version
    if status is not None:
        run.status = status
    if started_at is not None:
        run.started_at = started_at
    if ended_at is not None:
        run.ended_at = ended_at
    if error_code is not None:
        run.error_code = error_code
    if error_message is not None:
        run.error_message = error_message
    if artifacts_json is not None:
        run.artifacts_json = artifacts_json
    if log_s3_key is not None:
        run.log_s3_key = log_s3_key

    db_session.commit()
    return run


def _safe_upsert_extraction_run(db_session, **kwargs):
    if not db_session:
        return False

    try:
        _upsert_extraction_run(db_session, **kwargs)
        return True
    except Exception as exc:
        logger.warning("Failed to write extraction_run row for %s: %s", kwargs.get("process_id"), exc)
        try:
            db_session.rollback()
        except Exception:
            pass
        return False


def _safe_log_event(audit_logger, stage, status, **kwargs):
    if not audit_logger:
        return

    try:
        audit_logger.log(stage, status, **kwargs)
    except Exception as exc:
        logger.warning("Failed to append audit event stage=%s status=%s: %s", stage, status, exc)


def _upload_artifacts(audit_logger, result, merge, pdf_path=None):
    if not audit_logger:
        return {}

    artifacts = {}
    download_paths = result.get("download_paths", {})
    file_hash = result.get("file_hash")

    if pdf_path and os.path.exists(pdf_path):
        try:
            with open(pdf_path, "rb") as f:
                key = audit_logger.upload_artifact("source.pdf", f.read(), subdir="inputs")
            if key:
                artifacts["source_pdf"] = key
        except Exception as exc:
            logger.warning("Failed to upload source PDF artifact: %s", exc)

    for method, path in download_paths.items():
        if not path or not os.path.exists(path):
            continue

        try:
            with open(path, "rb") as f:
                key = audit_logger.upload_artifact(f"{method}.md", f.read())
            if key:
                artifacts[method] = key
        except Exception as exc:
            logger.warning("Failed to upload artifact for method %s: %s", method, exc)

    if merge:
        merged_path = result.get("merged_cache_path") or _cached_path(result.get("file_hash"), "merged")
        if merged_path and os.path.exists(merged_path):
            try:
                with open(merged_path, "rb") as f:
                    key = audit_logger.upload_artifact("merged.md", f.read())
                if key:
                    artifacts["merged"] = key
            except Exception as exc:
                logger.warning("Failed to upload merged artifact: %s", exc)

    images = []
    if file_hash:
        images_dir = _get_images_dir(file_hash)
        for image in result.get("images", []):
            filename = image.get("filename")
            if not filename:
                continue

            image_path = os.path.join(images_dir, filename)
            if not os.path.exists(image_path):
                continue

            try:
                with open(image_path, "rb") as f:
                    key = audit_logger.upload_artifact(filename, f.read(), subdir="images")
                if key:
                    images.append({
                        "filename": filename,
                        "s3_key": key,
                        "size_bytes": image.get("size_bytes"),
                    })
            except Exception as exc:
                logger.warning("Failed to upload image artifact %s: %s", filename, exc)

    if images:
        artifacts["images"] = images

    return artifacts


# ---------------------------------------------------------------------------
# Extraction task
# ---------------------------------------------------------------------------

@celery.task(bind=True, name="extract_pdf")
def extract_pdf(
    self,
    pdf_path,
    methods,
    merge=False,
    process_id=None,
    reference_curie=None,
    mod_abbreviation=None,
    source_referencefile_id=None,
):
    """
    Run PDF extraction in the background.

    Args:
        pdf_path: Absolute path to the uploaded PDF on disk.
        methods: List of extractor names, e.g. ["grobid", "docling", "marker"].
        merge: Whether to run the LLM merge step after extraction.

    Returns:
        dict with status, file_hash, per-method outputs, and optional merged output.
    """
    process_id = str(process_id or uuid.uuid4())
    db_session = _get_db_session()
    audit_logger = None

    os.makedirs(Config.CACHE_FOLDER, exist_ok=True)

    file_hash = _get_file_hash(pdf_path)

    try:
        audit_logger = AuditLogger(process_id, Config)
    except Exception as exc:
        logger.warning("Failed to initialize audit logger for process_id=%s: %s", process_id, exc)

    _safe_log_event(audit_logger, "run", "queued", detail="Task accepted")

    started_at = _now_utc()
    total_start = time.monotonic()

    # Update DB row to running (row was created by the API handler).
    # Uses upsert so it still works if the API-side insert failed.
    if db_session:
        ok = _safe_upsert_extraction_run(
            db_session,
            process_id=process_id,
            reference_curie=reference_curie,
            mod_abbreviation=mod_abbreviation,
            source_pdf_md5=file_hash,
            source_referencefile_id=source_referencefile_id,
            config_version=Config.EXTRACTION_CONFIG_VERSION,
            status="running",
            started_at=started_at,
        )
        if not ok:
            _safe_close_session(db_session)
            db_session = None

    _safe_log_event(audit_logger, "run", "running")

    try:
        result = _run_extraction(
            self,
            pdf_path,
            methods,
            merge,
            file_hash=file_hash,
            audit_logger=audit_logger,
        )

        artifacts_json = _upload_artifacts(audit_logger, result, merge, pdf_path=pdf_path)

        total_duration = round(time.monotonic() - total_start, 3)
        _safe_log_event(audit_logger, "finalize", "succeeded", total_duration_s=total_duration)

        log_s3_key = audit_logger.get_log_s3_key() if audit_logger else None
        ended_at = _now_utc()

        if db_session:
            _safe_upsert_extraction_run(
                db_session,
                process_id=process_id,
                status="succeeded",
                ended_at=ended_at,
                artifacts_json=artifacts_json,
                log_s3_key=log_s3_key,
            )

        result["process_id"] = process_id
        result["reference_curie"] = reference_curie
        result["mod_abbreviation"] = mod_abbreviation
        result["started_at"] = started_at.isoformat().replace("+00:00", "Z")
        result["ended_at"] = ended_at.isoformat().replace("+00:00", "Z")
        result["log_s3_key"] = log_s3_key
        result["artifacts_json"] = artifacts_json

        return result

    except Exception as exc:
        total_duration = round(time.monotonic() - total_start, 3)
        _safe_log_event(
            audit_logger,
            "finalize",
            "failed",
            error_code=exc.__class__.__name__,
            detail=str(exc),
            total_duration_s=total_duration,
        )

        if db_session:
            _safe_upsert_extraction_run(
                db_session,
                process_id=process_id,
                status="failed",
                ended_at=_now_utc(),
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                log_s3_key=audit_logger.get_log_s3_key() if audit_logger else None,
            )

        raise

    finally:
        try:
            if audit_logger:
                audit_logger.flush()
                if db_session:
                    _safe_upsert_extraction_run(
                        db_session,
                        process_id=process_id,
                        log_s3_key=audit_logger.get_log_s3_key(),
                    )
        except Exception as exc:
            logger.warning("Failed during audit flush for process_id=%s: %s", process_id, exc)

        _safe_close_session(db_session)

        # Cleanup uploaded PDF (cached outputs are kept)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


def _run_extraction(self, pdf_path, methods, merge, file_hash=None, audit_logger=None):
    """Inner extraction logic, separated so caller can wrap with finally."""

    from app.services.grobid_service import Grobid
    from app.services.docling_service import Docling
    from app.services.marker_service import Marker
    from app.services.llm_service import LLM

    if file_hash is None:
        file_hash = _get_file_hash(pdf_path)

    extractions = {}
    methods_used = []
    cached_methods = []
    total_steps = len(methods) + (1 if merge else 0)
    current_step = 0

    # --- Run each extractor ---------------------------------------------------

    if "grobid" in methods:
        output_path = _cached_path(file_hash, "grobid")
        stage = "extract_grobid"
        if _is_cached(file_hash, "grobid"):
            cached_methods.append("grobid")
            _safe_log_event(audit_logger, stage, "cache_hit", detail="Using cached grobid output")
        else:
            self.update_state(state="PROGRESS", meta={
                "step": "grobid", "current": current_step, "total": total_steps
            })
            logger.info("Running GROBID extraction...")
            grobid = Grobid(
                base_url=Config.GROBID_URL,
                timeout=Config.GROBID_REQUEST_TIMEOUT,
                include_coordinates=Config.GROBID_INCLUDE_COORDINATES,
                include_raw_citations=Config.GROBID_INCLUDE_RAW_CITATIONS,
            )
            started = time.monotonic()
            _safe_log_event(audit_logger, stage, "started")
            try:
                grobid.extract(pdf_path, output_path)
                _safe_log_event(
                    audit_logger,
                    stage,
                    "completed",
                    duration_s=round(time.monotonic() - started, 3),
                )
            except Exception as exc:
                _safe_log_event(
                    audit_logger,
                    stage,
                    "failed",
                    detail=str(exc),
                    duration_s=round(time.monotonic() - started, 3),
                )
                raise
        with open(output_path, "r", encoding="utf-8") as f:
            extractions["grobid"] = f.read()
        methods_used.append("grobid")
        current_step += 1

    if "docling" in methods:
        output_path = _cached_path(file_hash, "docling")
        stage = "extract_docling"
        if _is_cached(file_hash, "docling"):
            cached_methods.append("docling")
            _safe_log_event(audit_logger, stage, "cache_hit", detail="Using cached docling output")
        else:
            self.update_state(state="PROGRESS", meta={
                "step": "docling", "current": current_step, "total": total_steps
            })
            logger.info("Running Docling extraction...")
            docling = Docling(device=Config.DOCLING_DEVICE)
            started = time.monotonic()
            _safe_log_event(audit_logger, stage, "started")
            try:
                docling.extract(pdf_path, output_path)
                _safe_log_event(
                    audit_logger,
                    stage,
                    "completed",
                    duration_s=round(time.monotonic() - started, 3),
                )
            except Exception as exc:
                _safe_log_event(
                    audit_logger,
                    stage,
                    "failed",
                    detail=str(exc),
                    duration_s=round(time.monotonic() - started, 3),
                )
                raise
        with open(output_path, "r", encoding="utf-8") as f:
            extractions["docling"] = f.read()
        methods_used.append("docling")
        current_step += 1

    if "marker" in methods:
        output_path = _cached_path(file_hash, "marker")
        stage = "extract_marker"
        if _is_cached(file_hash, "marker"):
            cached_methods.append("marker")
            _safe_log_event(audit_logger, stage, "cache_hit", detail="Using cached marker output")
        else:
            self.update_state(state="PROGRESS", meta={
                "step": "marker", "current": current_step, "total": total_steps
            })
            logger.info("Running Marker extraction...")
            marker = Marker(
                device=Config.MARKER_DEVICE,
                extract_images=Config.MARKER_EXTRACT_IMAGES,
            )
            started = time.monotonic()
            _safe_log_event(audit_logger, stage, "started")
            try:
                marker.extract(pdf_path, output_path)
                _safe_log_event(
                    audit_logger,
                    stage,
                    "completed",
                    duration_s=round(time.monotonic() - started, 3),
                )
            except Exception as exc:
                _safe_log_event(
                    audit_logger,
                    stage,
                    "failed",
                    detail=str(exc),
                    duration_s=round(time.monotonic() - started, 3),
                )
                raise
        with open(output_path, "r", encoding="utf-8") as f:
            extractions["marker"] = f.read()
        methods_used.append("marker")
        current_step += 1

    # --- Optional LLM merge ---------------------------------------------------

    merged_md = None
    merged_cache_path = None
    consensus_metrics = None
    if merge and extractions:
        if not Config.OPENAI_API_KEY:
            raise ValueError("merge=true but OPENAI_API_KEY is not set")

        version = Config.EXTRACTION_CONFIG_VERSION
        cache_key = f"v{version}_{file_hash}_{'_'.join(sorted(methods))}"
        merged_cache_path = os.path.join(Config.CACHE_FOLDER, f"{cache_key}_merged.md")

        stage = "llm_merge"
        if os.path.exists(merged_cache_path):
            cached_methods.append("merged")
            _safe_log_event(audit_logger, stage, "cache_hit", detail="Using cached merged output")
            with open(merged_cache_path, "r", encoding="utf-8") as f:
                merged_md = f.read()
        else:
            self.update_state(state="PROGRESS", meta={
                "step": "llm_merge", "current": current_step, "total": total_steps
            })

            llm = LLM(
                api_key=Config.OPENAI_API_KEY,
                model=Config.LLM_MODEL,
                max_tokens=Config.LLM_MAX_TOKENS,
            )

            grobid_text = extractions.get("grobid", "")
            docling_text = extractions.get("docling", "")
            marker_text = extractions.get("marker", "")

            started = time.monotonic()
            _safe_log_event(audit_logger, stage, "started")
            try:
                # Try consensus pipeline if enabled and all 3 extractors present
                if (Config.CONSENSUS_ENABLED
                        and grobid_text and docling_text and marker_text):
                    from app.services.consensus_service import merge_with_consensus
                    logger.info("Attempting consensus merge...")
                    try:
                        consensus_md, consensus_metrics = merge_with_consensus(
                            grobid_text, docling_text, marker_text, llm,
                        )
                        if consensus_md is not None:
                            merged_md = consensus_md
                            logger.info("Consensus merge succeeded: %s", consensus_metrics)
                        else:
                            logger.info("Consensus fallback triggered: %s", consensus_metrics)
                    except Exception as e:
                        logger.warning("Consensus pipeline error, falling back to full-LLM merge: %s", e)
                        consensus_metrics = {"fallback_triggered": True, "fallback_reason": "pipeline_error"}

                # Fallback to full-LLM merge
                if merged_md is None:
                    logger.info("Merging outputs with full-LLM merge...")
                    merged_md = llm.extract(grobid_text, docling_text, marker_text)

                with open(merged_cache_path, "w", encoding="utf-8") as f:
                    f.write(merged_md)
                with open(_cached_path(file_hash, "merged"), "w", encoding="utf-8") as f:
                    f.write(merged_md)

                _safe_log_event(
                    audit_logger,
                    stage,
                    "completed",
                    duration_s=round(time.monotonic() - started, 3),
                )
            except Exception as exc:
                _safe_log_event(
                    audit_logger,
                    stage,
                    "failed",
                    detail=str(exc),
                    duration_s=round(time.monotonic() - started, 3),
                )
                raise

    # Rewrite image paths in merged output preview at response time
    merged_preview = merged_md
    if merged_preview:
        merged_preview = _rewrite_image_paths(merged_preview, file_hash)
        if len(merged_preview) > 1000:
            merged_preview = merged_preview[:1000] + "..."

    images = _list_images(file_hash)

    result = {
        "status": "success",
        "file_hash": file_hash,
        "methods_used": methods_used,
        "cached_methods": cached_methods,
        "extractions": {m: extractions[m][:500] + "..." if len(extractions[m]) > 500 else extractions[m]
                        for m in extractions},
        "merged_output": merged_preview,
        "merged_cache_path": merged_cache_path,
        "download_paths": {m: _cached_path(file_hash, m) for m in methods_used},
        "images": images,
        "has_images": bool(images),
    }

    if consensus_metrics is not None:
        result["consensus_metrics"] = consensus_metrics

    return result
