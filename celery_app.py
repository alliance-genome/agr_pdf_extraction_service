import os
import uuid
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from celery import Celery
from celery.signals import setup_logging as celery_setup_logging, worker_process_init

from config import Config
from app.models import ExtractionRun, get_session
from app.services.audit_logger import AuditLogger
from app.logging_config import setup_logging, MergingLoggerAdapter

logger = logging.getLogger(__name__)


@celery_setup_logging.connect
def _configure_celery_logging(**kwargs):
    """Take over Celery's logging so ALL messages go through our GELF handler.

    Connecting to this signal prevents Celery from setting up its own logging.
    This ensures the GELF 'host' field is 'pdfx-worker' from the very first message.
    """
    setup_logging(component="worker")


@worker_process_init.connect
def _init_worker_process(**kwargs):
    """Configure torch threading and GPU once per forked worker process."""
    try:
        import torch
        num_threads = int(os.environ.get("OMP_NUM_THREADS", 8))
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 4))

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_bytes = getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)
            vram_gb = vram_bytes / (1024 ** 3) if vram_bytes else 0
            logger.info(
                "Worker process initialized: GPU=%s (%.1fGB VRAM), CUDA=%s, "
                "torch threads=%d, interop=%d",
                gpu_name, vram_gb, torch.version.cuda,
                torch.get_num_threads(), torch.get_num_interop_threads(),
            )
        else:
            logger.info(
                "Worker process initialized: CPU-only, torch threads=%d, interop=%d",
                torch.get_num_threads(), torch.get_num_interop_threads(),
            )
    except Exception as exc:
        logger.warning("Failed to configure torch/GPU: %s", exc)

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
    task_default_queue="default",  # explicit: workers listen on -Q default
    worker_prefetch_multiplier=1,  # one job at a time per worker process
    task_soft_time_limit=1800,     # 30 min soft limit (GPU: seconds per paper; CPU fallback: 30+ min)
    task_time_limit=2100,          # 35 min hard kill
)


# ---------------------------------------------------------------------------
# GPU snapshot helper
# ---------------------------------------------------------------------------

def _gpu_snapshot():
    """Return GPU memory stats or empty dict if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            used = total - free
            return {
                "_gpu_mem_used_gb": round(used / (1024 ** 3), 2),
                "_gpu_mem_total_gb": round(total / (1024 ** 3), 2),
            }
    except Exception:
        pass
    return {}


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
    file_hash = None
    db_session = _get_db_session()
    audit_logger = None

    # Per-job adapter — merges identity fields with per-event fields for GELF
    adapter = MergingLoggerAdapter(logger, {
        "_process_id": process_id,
        "_reference_curie": reference_curie or "",
        "_mod_abbreviation": mod_abbreviation or "",
        "_file_hash": "",
    })

    os.makedirs(Config.CACHE_FOLDER, exist_ok=True)

    try:
        file_hash = _get_file_hash(pdf_path)
        adapter.extra["_file_hash"] = file_hash
    except FileNotFoundError:
        adapter.error(
            "PDF file not found",
            extra={"_event": "job_failed", "_error_code": "file_not_found"},
        )
        if db_session:
            _safe_upsert_extraction_run(db_session, process_id=process_id,
                                        status="failed", error_message=f"PDF file not found: {pdf_path}")
            _safe_close_session(db_session)
        return {"status": "failed", "error": f"PDF file not found: {pdf_path}", "process_id": process_id}

    try:
        audit_logger = AuditLogger(process_id, Config)
    except Exception as exc:
        adapter.warning("Failed to initialize audit logger: %s", exc)

    _safe_log_event(audit_logger, "run", "queued", detail="Task accepted")

    adapter.info(
        "Job accepted: methods=%s, merge=%s",
        methods, merge,
        extra={
            "_event": "job_accepted",
            "_stage": "init",
        },
    )

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
            adapter=adapter,
        )

        artifacts_json = _upload_artifacts(audit_logger, result, merge, pdf_path=pdf_path)

        total_duration = round(time.monotonic() - total_start, 3)
        _safe_log_event(audit_logger, "finalize", "succeeded", total_duration_s=total_duration)

        adapter.info(
            "Job complete in %.1fs", total_duration,
            extra={"_event": "job_complete", "_duration_s": total_duration},
        )

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

        error_code = "timeout" if "SoftTimeLimitExceeded" in type(exc).__name__ else type(exc).__name__
        adapter.error(
            "Job failed: %s", exc,
            extra={
                "_event": "job_failed",
                "_error_code": error_code,
                "_duration_s": total_duration,
            },
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


def _run_single_extractor(method, pdf_path, file_hash, config, audit_logger, adapter=None):
    """Run one extractor and return (method, output_text, cached_flag).

    This is a standalone function so it can be dispatched to a ThreadPoolExecutor.
    GROBID is HTTP-bound (talks to a separate container) while Docling/Marker are
    CPU-bound, so running GROBID concurrently with the others saves wall-clock time.
    """
    log = adapter or logger
    output_path = _cached_path(file_hash, method)
    stage = f"extract_{method}"

    if _is_cached(file_hash, method):
        _safe_log_event(audit_logger, stage, "cache_hit", detail=f"Using cached {method} output")
        log.info(
            "Cache hit for %s", method,
            extra={"_event": "extractor_cache_hit", "_extractor": method, "_cached": True},
        )
        with open(output_path, "r", encoding="utf-8") as f:
            return method, f.read(), True

    started = time.monotonic()
    _safe_log_event(audit_logger, stage, "started")

    log.info(
        "Starting %s extraction", method,
        extra={"_event": "extractor_start", "_extractor": method, **_gpu_snapshot()},
    )

    try:
        if method == "grobid":
            from app.services.grobid_service import Grobid
            extractor = Grobid(
                base_url=config.GROBID_URL,
                timeout=config.GROBID_REQUEST_TIMEOUT,
                include_coordinates=config.GROBID_INCLUDE_COORDINATES,
                include_raw_citations=config.GROBID_INCLUDE_RAW_CITATIONS,
            )
        elif method == "docling":
            from app.services.docling_service import Docling
            extractor = Docling(device=config.DOCLING_DEVICE)
        elif method == "marker":
            from app.services.marker_service import Marker
            extractor = Marker(
                device=config.MARKER_DEVICE,
                extract_images=config.MARKER_EXTRACT_IMAGES,
            )
        else:
            raise ValueError(f"Unknown extraction method: {method}")

        extractor.extract(pdf_path, output_path)

        duration_s = round(time.monotonic() - started, 3)
        _safe_log_event(
            audit_logger, stage, "completed",
            duration_s=duration_s,
        )
        log.info(
            "%s extraction complete in %.1fs", method, duration_s,
            extra={
                "_event": "extractor_complete",
                "_extractor": method,
                "_duration_s": duration_s,
                "_cached": False,
                "_status": "success",
                **_gpu_snapshot(),
            },
        )
    except Exception as exc:
        duration_s = round(time.monotonic() - started, 3)
        _safe_log_event(
            audit_logger, stage, "failed",
            detail=str(exc),
            duration_s=duration_s,
        )
        log.error(
            "%s extraction failed: %s", method, exc,
            extra={
                "_event": "extractor_complete",
                "_extractor": method,
                "_duration_s": duration_s,
                "_status": "failed",
                "_error_code": type(exc).__name__,
            },
        )
        raise

    with open(output_path, "r", encoding="utf-8") as f:
        return method, f.read(), False


def _run_extraction(self, pdf_path, methods, merge, file_hash=None, audit_logger=None, adapter=None):
    """Inner extraction logic, separated so caller can wrap with finally."""

    from app.services.llm_service import LLM

    log = adapter or logger

    if file_hash is None:
        file_hash = _get_file_hash(pdf_path)

    extractions = {}
    methods_used = []
    cached_methods = []
    total_steps = len(methods) + (1 if merge else 0)

    # --- Run extractors (parallel where possible) -----------------------------
    # GROBID is HTTP-bound (separate container), Docling/Marker are CPU-bound.
    # Running GROBID in a thread overlapping with CPU extractors saves wall time.

    non_cached = [m for m in methods if not _is_cached(file_hash, m)]
    has_grobid_work = "grobid" in non_cached
    cpu_methods = [m for m in non_cached if m != "grobid"]
    use_parallel = has_grobid_work and len(cpu_methods) > 0

    self.update_state(state="PROGRESS", meta={
        "step": "extracting", "current": 0, "total": total_steps,
        "parallel": use_parallel,
    })

    if use_parallel:
        # Run GROBID in a thread while CPU extractors run sequentially in main thread
        log.info("Running GROBID in parallel with %s", cpu_methods)
        errors = []
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="grobid") as pool:
            grobid_future = pool.submit(
                _run_single_extractor, "grobid", pdf_path, file_hash, Config, audit_logger, adapter,
            )

            # Run CPU extractors sequentially in the main thread
            for method in cpu_methods:
                try:
                    m, text, was_cached = _run_single_extractor(
                        method, pdf_path, file_hash, Config, audit_logger, adapter,
                    )
                    extractions[m] = text
                    methods_used.append(m)
                    if was_cached:
                        cached_methods.append(m)
                except Exception as exc:
                    errors.append((method, exc))

            # Collect GROBID result
            try:
                m, text, was_cached = grobid_future.result(timeout=Config.GROBID_REQUEST_TIMEOUT + 60)
                extractions[m] = text
                methods_used.append(m)
                if was_cached:
                    cached_methods.append(m)
            except Exception as exc:
                errors.append(("grobid", exc))

        # Also pick up any cached-only methods not in non_cached
        for method in methods:
            if method not in methods_used and _is_cached(file_hash, method):
                m, text, _ = _run_single_extractor(
                    method, pdf_path, file_hash, Config, audit_logger, adapter,
                )
                extractions[m] = text
                methods_used.append(m)
                cached_methods.append(m)

        if errors:
            # Raise the first error (same behavior as sequential)
            raise errors[0][1]
    else:
        # Sequential fallback (no GROBID work, or only one method)
        for method in methods:
            m, text, was_cached = _run_single_extractor(
                method, pdf_path, file_hash, Config, audit_logger, adapter,
            )
            extractions[m] = text
            methods_used.append(m)
            if was_cached:
                cached_methods.append(m)

    # --- Optional LLM merge ---------------------------------------------------

    merged_md = None
    merged_cache_path = None
    audit_cache_path = None
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
            possible_audit = os.path.join(Config.CACHE_FOLDER, f"{cache_key}_audit.json")
            if os.path.exists(possible_audit):
                audit_cache_path = possible_audit
        else:
            self.update_state(state="PROGRESS", meta={
                "step": "llm_merge", "current": len(methods_used), "total": total_steps
            })

            llm = LLM(
                api_key=Config.OPENAI_API_KEY,
                model=Config.LLM_MODEL,
                reasoning_effort=Config.LLM_REASONING_EFFORT,
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
                    log.info("Attempting consensus merge...")
                    try:
                        consensus_md, consensus_metrics, consensus_audit = merge_with_consensus(
                            grobid_text, docling_text, marker_text, llm,
                        )
                        if consensus_md is not None:
                            merged_md = consensus_md
                            if consensus_audit:
                                audit_cache_path = os.path.join(
                                    Config.CACHE_FOLDER, f"{cache_key}_audit.json",
                                )
                                with open(audit_cache_path, "w", encoding="utf-8") as f:
                                    json.dump(consensus_audit, f, indent=2, ensure_ascii=False)
                            log.info(
                                "Consensus merge succeeded",
                                extra={
                                    "_event": "consensus_classify_summary",
                                    "_conflict_count": consensus_metrics.get("conflict", 0),
                                    "_agree_exact": consensus_metrics.get("agree_exact", 0),
                                    "_agree_near": consensus_metrics.get("agree_near", 0),
                                    "_gap": consensus_metrics.get("gap", 0),
                                    "_conflict_ratio": consensus_metrics.get("conflict_ratio", 0.0),
                                    "_alignment_confidence": consensus_metrics.get("alignment_confidence", 0.0),
                                    "_tokens_saved": consensus_metrics.get("tokens_saved_estimate", 0),
                                },
                            )
                        else:
                            log.info(
                                "Consensus fallback triggered: %s",
                                consensus_metrics.get("fallback_reason"),
                                extra={
                                    "_event": "consensus_fallback",
                                    "_fallback_reason": consensus_metrics.get("fallback_reason", ""),
                                    "_conflict_ratio": consensus_metrics.get("conflict_ratio", 0.0),
                                    "_alignment_confidence": consensus_metrics.get("alignment_confidence", 0.0),
                                },
                            )
                    except Exception as e:
                        log.warning("Consensus pipeline error, falling back to full-LLM merge: %s", e)
                        consensus_metrics = {"fallback_triggered": True, "fallback_reason": "pipeline_error"}

                # Fallback to full-LLM merge
                if merged_md is None:
                    log.info("Merging outputs with full-LLM merge...")
                    merged_md = llm.extract(grobid_text, docling_text, marker_text)

                    # Apply header hierarchy resolution to full-LLM output too
                    if Config.CONSENSUS_HIERARCHY_ENABLED and merged_md:
                        try:
                            from app.services.consensus_service import resolve_header_hierarchy
                            merged_md = resolve_header_hierarchy(merged_md, llm)
                        except Exception as e:
                            log.warning("Header hierarchy resolution failed on fallback path: %s", e)

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

    if audit_cache_path:
        result["download_paths"]["audit"] = audit_cache_path

    return result
