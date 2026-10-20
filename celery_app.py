import os
import time
import json
import logging
from datetime import datetime, timezone
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from celery import Celery
from celery.signals import setup_logging as celery_setup_logging, worker_process_init

from config import Config
from app.error_utils import summarize_error_message
from app.image_metadata import (
    IMAGE_MANIFEST_FILENAME,
    apply_text_image_review,
    copy_image_metadata,
    list_image_directory,
    list_manifest_images,
    strip_text_image_review,
    write_image_manifest,
)

from app.logging_config import setup_logging
from app.services.model_policy import validate_runtime_model_policy
from app.services.merge_artifact import (
    load_merge_bundle,
    persist_merge_bundle,
)
from app.services.source_contracts import SourceArtifact
from app.services.native_extractor_artifact import (
    has_valid_native_extractor_artifact,
    load_native_extractor_artifact,
    native_artifact_path,
    native_manifest_path,
    sha256_file,
)
from app.services.page_coverage import page_coverage_sidecar_path
from app.services.page_resolution import resolve_page_batches
from app.services.page_provenance import (
    bound_page_resolution_ranges,
    build_page_resolution_batches,
    canonical_json_line_sha256,
    finalize_merged_page_provenance_bytes,
    load_source_page_provenance,
    merged_page_provenance_path,
    project_merged_page_ranges,
    source_page_provenance_path,
)

logger = logging.getLogger(__name__)
_torch_worker_configured = False
EXTRACTOR_METHODS = ("grobid", "docling", "marker")


class EmptyExtractorOutputError(ValueError):
    """Typed failure for an extractor artifact with no usable text."""

    def __init__(self, method, *, cached=False):
        self.method = method
        self.cached = bool(cached)
        origin = "cached" if self.cached else "new"
        super().__init__(f"{method} returned an empty {origin} artifact")


def _require_usable_extractor_text(method, text, *, cached=False, output_path=None):
    if isinstance(text, str) and text.strip():
        return text
    if output_path:
        try:
            os.remove(output_path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Unable to remove unusable %s extractor cache artifact", method)
    raise EmptyExtractorOutputError(method, cached=cached)


def _remove_marker_ready_file():
    ready_file = getattr(Config, "MARKER_READY_FILE", "")
    if not ready_file:
        return
    try:
        os.remove(ready_file)
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.warning("Failed to remove Marker readiness file %s: %s", ready_file, exc)


def _write_marker_ready_file(payload):
    ready_file = getattr(Config, "MARKER_READY_FILE", "")
    if not ready_file:
        return
    payload = dict(payload)
    payload.update({
        "pid": os.getpid(),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    ready_dir = os.path.dirname(ready_file)
    if ready_dir:
        os.makedirs(ready_dir, exist_ok=True)
    temp_path = f"{ready_file}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    os.replace(temp_path, ready_file)


def _marker_preload_enabled():
    mode = getattr(Config, "WORKER_PRELOAD_MARKER_MODELS", "off")
    return mode in {"1", "true", "yes", "on", "always", "auto"}


def _preload_marker_models_for_worker():
    if not _marker_preload_enabled():
        return

    _configure_torch_for_worker_process()
    _remove_marker_ready_file()
    try:
        from app.services.marker_service import preload_marker_models

        payload = preload_marker_models(
            device=Config.MARKER_DEVICE,
            extract_images=Config.WORKER_PRELOAD_MARKER_EXTRACT_IMAGES,
            disable_links=True,
        )
        if str(Config.MARKER_DEVICE).lower() != "cpu" and payload.get("device") != "cuda":
            raise RuntimeError("Marker preload resolved to CPU; CUDA is not available")
        _write_marker_ready_file(payload)
        logger.info("Marker worker readiness file written: %s", Config.MARKER_READY_FILE)
    except Exception as exc:
        _remove_marker_ready_file()
        logger.exception("Marker model preload failed: %s", exc)
        if Config.WORKER_PRELOAD_MARKER_REQUIRED:
            raise


def _configure_torch_for_worker_process():
    """Configure torch threading/GPU once in the process that will run tasks."""
    global _torch_worker_configured
    if _torch_worker_configured:
        return

    try:
        import torch
        num_threads = int(os.environ.get("OMP_NUM_THREADS", 8))
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 4))
        _torch_worker_configured = True

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
    validate_runtime_model_policy()
    _configure_torch_for_worker_process()

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------
celery = Celery(
    "pdf_extraction",
    broker=Config.CELERY_BROKER_URL,
    backend=Config.CELERY_RESULT_BACKEND,
)

celery.conf.update(
    imports=("app.pipeline_tasks",),
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=86400,  # 24 hours
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=Config.TASK_REJECT_ON_WORKER_LOST,
    task_default_queue="pdfx.unrouted",
    worker_prefetch_multiplier=1,  # one job at a time per worker process
    task_soft_time_limit=Config.TASK_SOFT_TIME_LIMIT_SECONDS,
    task_time_limit=Config.TASK_HARD_TIME_LIMIT_SECONDS,
)


from celery.signals import worker_ready


@worker_ready.connect
def _pipeline_worker_ready(**kwargs):
    role = os.environ.get("PDFX_WORKER_ROLE")
    if role in {"extraction", "finalization"}:
        from app.pipeline_health import start
        start(role)


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
    manifest_images = list_manifest_images(images_dir)
    if manifest_images is not None:
        return manifest_images
    return list_image_directory(images_dir)


def _has_image_extraction_manifest(file_hash):
    return os.path.exists(os.path.join(_get_images_dir(file_hash), IMAGE_MANIFEST_FILENAME))


def _is_cached(file_hash, method):
    return os.path.exists(_cached_path(file_hash, method))


def _is_extractor_cached(file_hash, method, extract_images=False):
    output_path = _cached_path(file_hash, method)
    if not os.path.exists(output_path):
        return False
    if not has_valid_native_extractor_artifact(
        source=method,
        output_filename=output_path,
    ):
        return False
    if method == "marker" and extract_images:
        return _has_image_extraction_manifest(file_hash)
    return True


VALID_CACHE_CLEAR_SCOPES = {"none", "merge", "extraction", "all"}


def _normalize_clear_cache_scope(clear_cache_scope=None, clear_cache=False):
    """Normalize legacy/new cache-clear inputs to a single scope value."""
    aliases = {
        "": "",
        "false": "none",
        "0": "none",
        "off": "none",
        "none": "none",
        "true": "all",
        "1": "all",
        "on": "all",
        "all": "all",
        "full": "all",
        "merge": "merge",
        "extraction": "extraction",
        "extract": "extraction",
        "extraction_and_merge": "extraction",  # "extraction" scope already includes merge artifacts
    }

    scope = ""
    if clear_cache_scope is not None:
        scope = str(clear_cache_scope).strip().lower()
    scope = aliases.get(scope, scope)

    if not scope:
        return "all" if clear_cache else "none"
    if clear_cache and scope == "none":
        return "all"
    if scope not in VALID_CACHE_CLEAR_SCOPES:
        raise ValueError(
            f"Invalid clear_cache_scope '{clear_cache_scope}'. "
            f"Valid values: {', '.join(sorted(VALID_CACHE_CLEAR_SCOPES))}"
        )
    return scope


def _clear_cached_outputs(file_hash, clear_cache_scope):
    """Clear cached artifacts for a file hash according to scope."""
    import glob as _glob
    import shutil

    version = Config.EXTRACTION_CONFIG_VERSION
    prefix = os.path.join(Config.CACHE_FOLDER, f"v{version}_{file_hash}_")

    patterns = []
    if clear_cache_scope == "all":
        patterns = [f"{prefix}*"]
    else:
        if clear_cache_scope in {"merge", "extraction"}:
            patterns.extend([
                f"{prefix}merged.md",                  # vX_<hash>_merged.md
                f"{prefix}*_merged.md",               # vX_<hash>_<methods>_merged.md
                f"{prefix}*_merged.md.*",             # manifest + page provenance
                f"{prefix}*_consensus_metrics.json",
                f"{prefix}*_audit.json",
            ])
        if clear_cache_scope == "extraction":
            patterns.extend([
                f"{prefix}grobid.md",
                f"{prefix}docling.md",
                f"{prefix}marker.md",
                f"{prefix}grobid.md.page-coverage.json",
                f"{prefix}docling.md.page-coverage.json",
                f"{prefix}marker.md.page-coverage.json",
                f"{prefix}grobid.md.page-provenance.json",
                f"{prefix}docling.md.page-provenance.json",
                f"{prefix}marker.md.page-provenance.json",
                f"{prefix}grobid.md.native.*",
                f"{prefix}docling.md.native.*",
                f"{prefix}marker.md.native.*",
            ])

    removed_files = 0
    seen = set()
    for pattern in patterns:
        for cached_file in _glob.glob(pattern):
            if cached_file in seen:
                continue
            seen.add(cached_file)
            if not os.path.isfile(cached_file):
                continue
            try:
                os.remove(cached_file)
                removed_files += 1
            except OSError as exc:
                logger.warning("Failed to remove cached file %s: %s", cached_file, exc)

    removed_images_dir = 0
    if clear_cache_scope in {"extraction", "all"}:
        images_dir = _get_images_dir(file_hash)
        if os.path.isdir(images_dir):
            try:
                shutil.rmtree(images_dir)
                removed_images_dir = 1
            except OSError as exc:
                logger.warning("Failed to remove images dir %s: %s", images_dir, exc)

    return {
        "scope": clear_cache_scope,
        "files_removed": removed_files,
        "images_dirs_removed": removed_images_dir,
        "removed_total": removed_files + removed_images_dir,
    }


def _safe_log_event(audit_logger, stage, status, **kwargs):
    if not audit_logger:
        return

    try:
        audit_logger.log(stage, status, **kwargs)
    except Exception as exc:
        logger.warning("Failed to append audit event stage=%s status=%s: %s", stage, status, exc)


def _upload_artifacts(audit_logger, result, merge, pdf_path=None):
    if not audit_logger:
        if merge:
            raise RuntimeError(
                "Merged output cannot be completed without durable artifact storage"
            )
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
        if method == "page_provenance":
            continue

        try:
            with open(path, "rb") as f:
                key = audit_logger.upload_artifact(f"{method}.md", f.read())
            if key:
                artifacts[method] = key
        except Exception as exc:
            logger.warning("Failed to upload artifact for method %s: %s", method, exc)

        if method not in EXTRACTOR_METHODS:
            continue
        try:
            manifest, native_bytes = load_native_extractor_artifact(
                source=method,
                output_filename=path,
            )
            native_key = audit_logger.upload_artifact(
                manifest["native_filename"],
                native_bytes,
                subdir=f"native/{method}",
            )
            manifest_key = audit_logger.upload_artifact(
                native_manifest_path(path).name,
                native_manifest_path(path).read_bytes(),
                subdir=f"native/{method}",
            )
            page_provenance_key = audit_logger.upload_artifact(
                manifest["page_provenance_filename"],
                source_page_provenance_path(path).read_bytes(),
                subdir=f"native/{method}",
            )
            if native_key and manifest_key and page_provenance_key:
                native_receipt = {
                    "artifact": native_key,
                    "manifest": manifest_key,
                    "page_provenance": page_provenance_key,
                    "page_provenance_sha256": manifest[
                        "page_provenance_sha256"
                    ],
                    "sha256": manifest["native_sha256"],
                    "media_type": manifest["native_media_type"],
                }
                if manifest.get("native_style_filename"):
                    from app.services.native_extractor_artifact import native_style_artifact_path
                    style_key = audit_logger.upload_artifact(
                        manifest["native_style_filename"], native_style_artifact_path(path).read_bytes(),
                        subdir=f"native/{method}",
                    )
                    if not style_key:
                        raise RuntimeError("Failed to upload required native style receipt")
                    native_receipt["style"] = style_key
                artifacts.setdefault("native", {})[method] = native_receipt
        except Exception as exc:
            logger.warning(
                "Failed to upload native artifact for method %s: %s",
                method,
                type(exc).__name__,
            )

    if merge:
        merged_path = result.get("merged_cache_path") or _cached_path(result.get("file_hash"), "merged")
        if not merged_path or not os.path.exists(merged_path):
            raise RuntimeError("Committed merged output is unavailable for durable upload")
        try:
            with open(merged_path, "rb") as f:
                key = audit_logger.upload_artifact("merged.md", f.read())
        except Exception as exc:
            raise RuntimeError("Failed to upload durable merged output") from exc
        if not key:
            raise RuntimeError("Failed to upload durable merged output")
        artifacts["merged"] = key
        page_path = merged_page_provenance_path(merged_path)
        try:
            page_key = audit_logger.upload_artifact(
                "page_provenance.json",
                page_path.read_bytes(),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to upload durable merged page provenance"
            ) from exc
        if not page_key:
            raise RuntimeError("Failed to upload durable merged page provenance")
        artifacts["page_provenance"] = page_key
        if not artifacts.get("audit"):
            raise RuntimeError("Failed to upload durable merged audit")

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
                    key = audit_logger.upload_artifact(
                        filename,
                        f.read(),
                        subdir="images",
                        tags={
                            "pdfx-artifact-type": "extracted-image",
                            "pdfx-retention": "temporary",
                            "pdfx-retention-ttl-seconds": Config.IMAGE_RETENTION_TTL_SECONDS,
                        },
                    )
                if key:
                    image_artifact = {
                        "filename": filename,
                        "s3_key": key,
                        "size_bytes": image.get("size_bytes"),
                    }
                    images.append(copy_image_metadata(image, image_artifact))
            except Exception as exc:
                logger.warning("Failed to upload image artifact %s: %s", filename, exc)

    if images:
        artifacts["images"] = images

    return artifacts


def _build_llm():
    from app.services.llm_service import LLM

    return LLM(
        api_key=Config.OPENAI_API_KEY,
        openai_timeout_seconds=Config.LLM_OPENAI_TIMEOUT_SECONDS,
        openai_max_retries=Config.LLM_OPENAI_MAX_RETRIES,
    )


def _build_merge_llm_or_none(log=None):
    """Build the selector client; baseline delivery does not require a model."""

    log = log or logger
    if not Config.OPENAI_API_KEY:
        return None
    try:
        return _build_llm()
    except Exception as exc:
        log.warning(
            "Selector unavailable; continuing with deterministic/baseline merge: %s",
            type(exc).__name__,
        )
        return None


def _compute_llm_accounting(llm, log=None):
    """Return cost plus usage without ever discarding observed token usage."""

    log = log or logger
    usage_summary = llm.usage.summary()
    if usage_summary["total_tokens"] <= 0:
        return None, None
    try:
        from app.services.llm_service import compute_cost

        return compute_cost(usage_summary, Config.LLM_PRICING)
    except Exception as exc:
        log.error("Failed to compute LLM cost; preserving raw usage: %s", exc)
        return None, usage_summary


def _record_llm_cost_alert(cost_usd, usage, log=None):
    """Emit a structured warning without changing or truncating the job."""

    log = log or logger
    threshold = Config.LLM_COST_ALERT_USD_PER_JOB
    triggered = cost_usd is not None and cost_usd >= threshold
    if isinstance(usage, dict):
        usage["cost_alert_threshold_usd"] = threshold
        usage["cost_alert_triggered"] = triggered
    if triggered:
        log.warning(
            "LLM per-job cost alert: $%.4f reached the $%.4f threshold",
            cost_usd,
            threshold,
            extra={
                "_event": "llm_cost_alert",
                "_llm_cost_usd": cost_usd,
                "_llm_cost_alert_threshold_usd": threshold,
                "_llm_total_tokens": usage.get("total_tokens", 0),
            },
        )
    return triggered


def _review_images_with_text_context(images, llm, log=None):
    """Apply optional text-only LLM review to image manifest entries."""
    reviewed_images = []
    log = log or logger
    for image in images:
        reviewed = dict(image)
        try:
            response = llm.review_image_context(reviewed)
            if hasattr(response, "model_dump"):
                review_payload = response.model_dump()
            else:
                review_payload = response.dict()
            reviewed = apply_text_image_review(
                reviewed,
                review_payload,
                model=Config.IMAGE_TEXT_REVIEW_MODEL,
            )
        except Exception as exc:
            reviewed["image_reviewed"] = False
            reviewed["image_review_method"] = "llm_text"
            reviewed["image_review_model"] = Config.IMAGE_TEXT_REVIEW_MODEL
            reviewed["image_review_error"] = f"{type(exc).__name__}: {exc}"
            log.warning(
                "Image text review failed for %s: %s",
                reviewed.get("filename"),
                exc,
                extra={
                    "_event": "image_text_review_failed",
                    "_image_filename": reviewed.get("filename"),
                    "_error_code": type(exc).__name__,
                },
            )
        reviewed_images.append(reviewed)
    return reviewed_images


def _resolve_image_review_flags(extract_images=False, review_images=None):
    """Resolve image extraction/review flags for API and direct Celery callers."""
    extract_images = bool(extract_images)
    if review_images is None:
        review_images = extract_images
    else:
        review_images = bool(review_images)
        if review_images:
            extract_images = True
    return extract_images, review_images


# ---------------------------------------------------------------------------
# Extraction task
# ---------------------------------------------------------------------------


def _run_single_extractor(
    method,
    pdf_path,
    file_hash,
    config,
    audit_logger,
    adapter=None,
    extract_images=False,
    force_full_page_ocr=False,
):
    """Run one extractor and return (method, output_text, cached_flag).

    This is a standalone function so it can be dispatched to a ThreadPoolExecutor.
    GROBID is HTTP-bound (talks to a separate container) while Docling/Marker are
    CPU-bound, so running GROBID concurrently with the others saves wall-clock time.
    """
    log = adapter or logger
    output_path = _cached_path(file_hash, method)
    stage = f"extract_{method}"

    extract_images = bool(extract_images)

    if not force_full_page_ocr and _is_extractor_cached(
        file_hash, method, extract_images=extract_images
    ):
        _safe_log_event(audit_logger, stage, "cache_hit", detail=f"Using cached {method} output")
        log.info(
            "Cache hit for %s", method,
            extra={"_event": "extractor_cache_hit", "_extractor": method, "_cached": True},
        )
        with open(output_path, "r", encoding="utf-8") as f:
            text = f.read()
        return method, _require_usable_extractor_text(
            method,
            text,
            cached=True,
            output_path=output_path,
        ), True

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
            docling_options = {"device": config.DOCLING_DEVICE}
            if force_full_page_ocr:
                docling_options["force_full_page_ocr"] = True
            extractor = Docling(**docling_options)
        elif method == "marker":
            from app.services.marker_service import Marker
            extractor = Marker(
                device=config.MARKER_DEVICE,
                extract_images=extract_images,
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
        error_message = summarize_error_message(exc)
        _safe_log_event(
            audit_logger, stage, "failed",
            detail=error_message,
            duration_s=duration_s,
        )
        log.error(
            "%s extraction failed: %s", method, error_message,
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
        text = f.read()
    return method, _require_usable_extractor_text(
        method,
        text,
        cached=False,
        output_path=output_path,
    ), False


def _is_transient_extractor_failure(exc):
    """Classify the narrow failures eligible for one immediate route retry."""

    if isinstance(exc, EmptyExtractorOutputError):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    try:
        import requests

        if isinstance(
            exc,
            (requests.exceptions.Timeout, requests.exceptions.ConnectionError),
        ):
            return True
    except ImportError:
        pass
    message = str(exc).casefold()
    return any(
        token in message
        for token in (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection refused",
            "service unavailable",
            "worker_not_ready",
            "http 502",
            "http 503",
            "http 504",
        )
    )


def _discard_failed_extractor_cache(file_hash, method):
    """Remove one failed generation so recovery cannot replay partial bytes."""

    output_path = _cached_path(file_hash, method)
    paths = (
        output_path,
        native_manifest_path(output_path),
        native_artifact_path(output_path, method),
        page_coverage_sidecar_path(output_path),
        source_page_provenance_path(output_path),
    )
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(
                "Cannot remove failed %s artifact %s; skipping its recovery route: %s",
                method,
                path,
                type(exc).__name__,
            )
            return False
    return True


def _recovery_time_available(job_started_monotonic):
    elapsed = time.monotonic() - job_started_monotonic
    remaining = Config.TASK_SOFT_TIME_LIMIT_SECONDS - elapsed
    return remaining > Config.EXTRACTION_FINALIZATION_RESERVE_SECONDS


def _run_extraction(
    self,
    pdf_path,
    methods,
    merge,
    file_hash=None,
    audit_logger=None,
    adapter=None,
    run_log_path=None,
    extract_images=False,
    review_images=None,
    _shared=None,
    job_started_monotonic=None,
):
    """Direct-call convenience using the same stages as the durable pipeline."""
    file_hash = file_hash or _get_file_hash(pdf_path)
    if job_started_monotonic is None:
        job_started_monotonic = time.monotonic()
    options = dict(file_hash=file_hash, audit_logger=audit_logger, adapter=adapter,
                   run_log_path=run_log_path, extract_images=extract_images,
                   review_images=review_images, _shared=_shared,
                   job_started_monotonic=job_started_monotonic)
    source_state = _extract_sources(self, pdf_path, methods, merge, **options)
    return _finalize_sources(self, pdf_path, methods, merge, source_state=source_state, **options)


def _extract_sources(
    self,
    pdf_path,
    methods,
    merge,
    file_hash=None,
    audit_logger=None,
    adapter=None,
    run_log_path=None,
    extract_images=False,
    review_images=None,
    _shared=None,
    job_started_monotonic=None,
):
    """Inner extraction logic, separated so caller can wrap with finally."""

    log = adapter or logger
    extract_images, review_images = _resolve_image_review_flags(extract_images, review_images)

    if file_hash is None:
        file_hash = _get_file_hash(pdf_path)

    extractions = {}
    methods_used = []
    cached_methods = []
    extractor_failures = {}
    extraction_errors = {}
    retried_extractors = []
    emergency_ocr_used = False
    if job_started_monotonic is None:
        job_started_monotonic = time.monotonic()

    # --- Stage tracking for granular progress reporting -----------------------
    all_stages = (
        ["initializing"]
        + list(methods)
        + (["llm_merge"] if merge else [])
        + (["image_review"] if review_images else [])
        + ["finalizing"]
    )
    completed_stages = []
    failed_stages = []

    def _record_extractor_failure(method, exc):
        error_message = summarize_error_message(exc)[:1000]
        extractor_failures[method] = {
            "error_code": type(exc).__name__,
            "message": error_message,
        }
        extraction_errors[method] = exc
        if method not in failed_stages:
            failed_stages.append(method)
        if method not in completed_stages:
            completed_stages.append(method)

    def _record_extractor_success(method, text, was_cached):
        text = _require_usable_extractor_text(
            method,
            text,
            cached=was_cached,
            output_path=_cached_path(file_hash, method) if was_cached else None,
        )
        extractions[method] = text
        extractor_failures.pop(method, None)
        extraction_errors.pop(method, None)
        failed_stages[:] = [item for item in failed_stages if item != method]
        if method not in methods_used:
            methods_used.append(method)
        if was_cached and method not in cached_methods:
            cached_methods.append(method)
        if method not in completed_stages:
            completed_stages.append(method)

    # Determine parallel execution strategy
    non_cached = [
        m for m in methods
        if not _is_extractor_cached(file_hash, m, extract_images=extract_images)
    ]
    has_grobid_work = "grobid" in non_cached
    cpu_methods = [m for m in non_cached if m != "grobid"]
    use_parallel = has_grobid_work and len(cpu_methods) > 0

    def _emit_progress(current_stage, display_text):
        pending = [s for s in all_stages if s not in completed_stages and s != current_stage]
        self.update_state(state="PROGRESS", meta={
            "step": current_stage,
            "stage": current_stage,
            "stage_display": display_text,
            "stages_completed": list(completed_stages),
            "stages_failed": list(failed_stages),
            "stages_pending": pending,
            "stages_total": len(all_stages),
            "stages_done": len(completed_stages),
            "percent": round(len(completed_stages) / len(all_stages) * 100),
            "parallel": use_parallel,
        })

    _emit_progress("initializing", "Initializing extraction job")
    completed_stages.append("initializing")

    # --- Run extractors (parallel where possible) -----------------------------
    # GROBID is HTTP-bound (separate container), Docling/Marker are CPU-bound.
    # Running GROBID in a thread overlapping with CPU extractors saves wall time.

    if use_parallel:
        # Run GROBID in a thread while CPU extractors run sequentially in main thread
        _emit_progress("grobid", "Running extractions (parallel)")
        log.info("Running GROBID in parallel with %s", cpu_methods)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="grobid") as pool:
            grobid_future = pool.submit(
                _run_single_extractor, "grobid", pdf_path, file_hash, Config, audit_logger, adapter, extract_images,
            )

            # Run CPU extractors sequentially in the main thread
            for method in cpu_methods:
                try:
                    _emit_progress(method, f"Running {method.upper()} extraction")
                    m, text, was_cached = _run_single_extractor(
                        method, pdf_path, file_hash, Config, audit_logger, adapter, extract_images,
                    )
                    _record_extractor_success(m, text, was_cached)
                except Exception as exc:
                    _record_extractor_failure(method, exc)

            # Collect GROBID result
            try:
                m, text, was_cached = grobid_future.result(timeout=Config.GROBID_REQUEST_TIMEOUT + 60)
                _record_extractor_success(m, text, was_cached)
            except Exception as exc:
                _record_extractor_failure("grobid", exc)

        # Also pick up any cached-only methods not in non_cached
        for method in methods:
            if method not in methods_used and _is_extractor_cached(file_hash, method, extract_images=extract_images):
                try:
                    m, text, _ = _run_single_extractor(
                        method, pdf_path, file_hash, Config, audit_logger, adapter, extract_images,
                    )
                    _record_extractor_success(m, text, True)
                except Exception as exc:
                    _record_extractor_failure(method, exc)
    else:
        # Sequential fallback (no GROBID work, or only one method)
        for method in methods:
            try:
                _emit_progress(method, f"Running {method.upper()} extraction")
                m, text, was_cached = _run_single_extractor(
                    method, pdf_path, file_hash, Config, audit_logger, adapter, extract_images,
                )
                _record_extractor_success(m, text, was_cached)
            except Exception as exc:
                _record_extractor_failure(method, exc)

    for method in sorted(extractor_failures):
        failure = extraction_errors[method]
        if not _is_transient_extractor_failure(failure):
            continue
        if not _recovery_time_available(job_started_monotonic):
            _safe_log_event(
                audit_logger,
                f"extract_{method}",
                "retry_skipped",
                detail="finalization reserve reached",
            )
            break
        if not _discard_failed_extractor_cache(file_hash, method):
            _safe_log_event(
                audit_logger,
                f"extract_{method}",
                "retry_skipped",
                detail="failed artifact could not be removed",
            )
            continue
        retried_extractors.append(method)
        _safe_log_event(audit_logger, f"extract_{method}", "retry_started")
        try:
            m, text, was_cached = _run_single_extractor(
                method,
                pdf_path,
                file_hash,
                Config,
                audit_logger,
                adapter,
                extract_images,
            )
            _record_extractor_success(m, text, was_cached)
            _safe_log_event(audit_logger, f"extract_{method}", "retry_completed")
        except Exception as exc:
            _record_extractor_failure(method, exc)

    if not extractions and (methods or merge) and _recovery_time_available(
        job_started_monotonic
    ):
        emergency_ocr_used = _discard_failed_extractor_cache(file_hash, "docling")
    if not extractions and emergency_ocr_used:
        _safe_log_event(audit_logger, "extract_docling", "emergency_ocr_started")
        try:
            m, text, was_cached = _run_single_extractor(
                "docling",
                pdf_path,
                file_hash,
                Config,
                audit_logger,
                adapter,
                False,
                force_full_page_ocr=True,
            )
            _record_extractor_success(m, text, was_cached)
            _safe_log_event(audit_logger, "extract_docling", "emergency_ocr_completed")
        except Exception as exc:
            _record_extractor_failure("docling", exc)

    if not extractions and (methods or merge):
        if extraction_errors:
            raise next(iter(extraction_errors.values()))
        raise ValueError("No extractor produced a usable artifact")
    if extractor_failures:
        from pdfx_observability import capture_failure
        for failure in extraction_errors.values():
            capture_failure(failure, stage="extraction")
        log.warning(
            "Continuing with %d completed extractor(s); failed methods=%s",
            len(extractions),
            sorted(extractor_failures),
            extra={
                "_event": "extractor_partial_success",
                "_available_extractors": sorted(extractions),
                "_failed_extractors": sorted(extractor_failures),
            },
        )

    return {
        "extractions": extractions,
        "methods_used": methods_used,
        "cached_methods": cached_methods,
        "extractor_failures": extractor_failures,
        "retried_extractors": retried_extractors,
        "emergency_ocr_used": emergency_ocr_used,
        "all_stages": all_stages,
        "completed_stages": completed_stages,
        "failed_stages": failed_stages,
        "use_parallel": use_parallel,
    }


def _finalize_sources(
    self,
    pdf_path,
    methods,
    merge,
    file_hash=None,
    audit_logger=None,
    adapter=None,
    run_log_path=None,
    extract_images=False,
    review_images=None,
    _shared=None,
    source_state=None,
    job_started_monotonic=None,
):
    """Finalize an explicit extraction snapshot; never invoke extractors."""
    if source_state is None:
        raise ValueError("Finalization requires an extraction checkpoint")
    log = adapter or logger
    extract_images, review_images = _resolve_image_review_flags(extract_images, review_images)
    extractions = source_state["extractions"]
    methods_used = source_state["methods_used"]
    cached_methods = source_state["cached_methods"]
    extractor_failures = source_state["extractor_failures"]
    retried_extractors = source_state["retried_extractors"]
    emergency_ocr_used = source_state["emergency_ocr_used"]
    all_stages = source_state["all_stages"]
    completed_stages = source_state["completed_stages"]
    failed_stages = source_state["failed_stages"]
    use_parallel = source_state["use_parallel"]

    def _emit_progress(current_stage, display_text):
        pending = [s for s in all_stages if s not in completed_stages and s != current_stage]
        self.update_state(state="PROGRESS", meta={
            "step": current_stage,
            "stage": current_stage,
            "stage_display": display_text,
            "stages_completed": list(completed_stages),
            "stages_failed": list(failed_stages),
            "stages_pending": pending,
            "stages_total": len(all_stages),
            "stages_done": len(completed_stages),
            "percent": round(len(completed_stages) / len(all_stages) * 100),
            "parallel": use_parallel,
        })

    # --- Optional LLM merge ---------------------------------------------------

    merged_md = None
    merged_cache_path = None
    audit_cache_path = None
    consensus_metrics = None
    llm = None
    if merge and extractions:
        version = Config.EXTRACTION_CONFIG_VERSION
        # Bind merge cache paths to the exact source set that completed. A retry
        # that restores a missing extractor must not reuse a partial-source merge.
        cache_key = f"v{version}_{file_hash}_{'_'.join(sorted(extractions))}"
        merged_cache_path = os.path.join(Config.CACHE_FOLDER, f"{cache_key}_merged.md")

        metrics_cache_path = os.path.join(Config.CACHE_FOLDER, f"{cache_key}_consensus_metrics.json")
        possible_audit = os.path.join(Config.CACHE_FOLDER, f"{cache_key}_audit.json")
        merged_alias_path = _cached_path(file_hash, "merged")
        source_artifacts = {
            source: SourceArtifact.from_text(source, extractions[source])
            for source in EXTRACTOR_METHODS
            if source in extractions
        }

        # Native receipts and skeleton identities are inputs to the merge cache,
        # not merely diagnostics produced after a miss.  Rebuild their compact
        # identities before looking at an existing bundle so a changed native
        # artifact can never reuse a merge produced from an older outline.
        from app.services.document_skeleton import (
            build_document_skeleton,
            load_runtime_native_structures,
        )
        pdf_sha256 = sha256_file(pdf_path)
        runtime_output_paths = {
            source: _cached_path(file_hash, source)
            for source in source_artifacts
        }
        runtime_native_structures, runtime_native_structure_failures = (
            load_runtime_native_structures(
                source_artifacts,
                runtime_output_paths,
                expected_pdf_sha256=pdf_sha256,
            )
        )
        runtime_source_page_maps = {}
        runtime_source_page_map_sha256 = {}
        for source, output_path in sorted(runtime_output_paths.items()):
            manifest, _native_bytes = load_native_extractor_artifact(
                source=source,
                output_filename=output_path,
                expected_pdf_sha256=pdf_sha256,
            )
            runtime_source_page_maps[source] = load_source_page_provenance(
                output_path,
                manifest=manifest,
            )
            runtime_source_page_map_sha256[source] = manifest[
                "page_provenance_sha256"
            ]
        # Persistence replay must use skeletons built from the same native inputs
        # that merge_source_artifacts receives below.
        runtime_skeletons = {}
        for source, artifact in sorted(source_artifacts.items()):
            try:
                runtime_skeletons[source] = build_document_skeleton(
                    artifact,
                    runtime_native_structures.get(source),
                )
            except Exception as exc:
                runtime_native_structure_failures[source] = type(exc).__name__
                runtime_skeletons[source] = build_document_skeleton(artifact, None)
        runtime_native_receipts = {
            source: native.receipt_digest
            for source, native in sorted(runtime_native_structures.items())
        }
        runtime_skeleton_ids = {
            source: skeleton.skeleton_id
            for source, skeleton in sorted(runtime_skeletons.items())
        }
        runtime_skeleton_projection_ids = {
            source: skeleton.projection_id
            for source, skeleton in sorted(runtime_skeletons.items())
        }

        stage = "llm_merge"
        cache_is_valid = False
        if os.path.exists(merged_cache_path):
            try:
                merged_md, consensus_metrics, consensus_audit = load_merge_bundle(
                    merged_path=merged_cache_path,
                    metrics_path=metrics_cache_path,
                    audit_path=possible_audit,
                    artifacts=source_artifacts,
                    expected_contract_id=Config.MERGE_CONTRACT_ID,
                    expected_native_structure_receipt_digests=(
                        runtime_native_receipts
                    ),
                    expected_skeleton_candidate_ids=runtime_skeleton_ids,
                    expected_skeleton_candidate_projection_ids=(
                        runtime_skeleton_projection_ids
                    ),
                    expected_pdf_sha256=pdf_sha256,
                    expected_source_page_map_sha256=(
                        runtime_source_page_map_sha256
                    ),
                )
                audit_cache_path = possible_audit
                cache_is_valid = True
            except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
                log.warning(
                    "Ignoring incomplete or inconsistent merge bundle: %s",
                    type(exc).__name__,
                )

        if cache_is_valid:
            cached_methods.append("merged")
            _safe_log_event(audit_logger, stage, "cache_hit", detail="Using cached merged output")
            completed_stages.append("llm_merge")
        else:
            _emit_progress("llm_merge", "Merging extraction outputs with LLM")

            llm = _build_merge_llm_or_none(log)
            if _shared is not None:
                _shared["llm"] = llm

            grobid_text = extractions.get("grobid", "")
            docling_text = extractions.get("docling", "")
            marker_text = extractions.get("marker", "")

            started = time.monotonic()
            _safe_log_event(audit_logger, stage, "started")
            active_merge_phase = None
            phase_started = started
            try:
                if not source_artifacts:
                    raise ValueError("Merge requires at least one extractor artifact")

                from app.services.merge_service import (
                    completion_evidence_for_runtime_artifacts,
                    merge_finished_extractor_outputs,
                )
                runtime_completion_evidence = completion_evidence_for_runtime_artifacts(
                    source_artifacts,
                    pdf_path=pdf_path,
                    output_paths=runtime_output_paths,
                )
                log.info("Attempting source-backed merge...")
                active_merge_phase = "text_merge"
                phase_started = time.monotonic()
                _safe_log_event(audit_logger, active_merge_phase, "started")
                consensus_md, consensus_metrics, consensus_audit = (
                    merge_finished_extractor_outputs(
                        grobid_text,
                        docling_text,
                        marker_text,
                        llm,
                        completion_evidence=runtime_completion_evidence,
                        native_structures=runtime_native_structures,
                        native_structure_failures=(
                            runtime_native_structure_failures
                        ),
                        benchmark_mode=Config.PDFX_BENCHMARK_MODE,
                    )
                )
                if consensus_metrics is not None:
                    audit_cache_path = possible_audit
                if consensus_metrics is not None and Config.PDFX_BENCHMARK_MODE:
                    consensus_metrics["runtime_execution"] = {
                        "extractor_failures": {
                            method: {
                                "error_code": failure.get("error_code")
                            }
                            for method, failure in sorted(extractor_failures.items())
                        },
                        "retried_extractors": list(retried_extractors),
                        "emergency_ocr_used": emergency_ocr_used,
                        "cached_methods": list(cached_methods),
                        "merge_compute_duration_s": round(
                            time.monotonic() - started, 3
                        ),
                    }

                if consensus_md is not None:
                    merged_md = consensus_md
                    log.info(
                        "Source-backed merge succeeded",
                        extra={
                            "_event": "source_merge_summary",
                            "_candidate_region_count": consensus_metrics.get(
                                "candidate_region_count", 0
                            ),
                            "_deterministic_region_count": consensus_metrics.get(
                                "region_decision_counts", {}
                            ).get("deterministic", 0),
                            "_model_selected_region_count": consensus_metrics.get(
                                "region_decision_counts", {}
                            ).get("model_selected", 0),
                            "_unresolved_region_count": consensus_metrics.get(
                                "unresolved_region_count", 0
                            ),
                            "_merge_quality": consensus_metrics.get("merge_quality"),
                            "_qualification_outcome": consensus_metrics.get(
                                "qualification_outcome"
                            ),
                        },
                    )
                else:
                    reason = (consensus_metrics or {}).get("failure_reason", "unknown")
                    raise ValueError(f"Consensus pipeline failed: {reason}")

                text_merge_duration = round(time.monotonic() - phase_started, 3)
                _safe_log_event(
                    audit_logger, "text_merge", "completed",
                    duration_s=text_merge_duration,
                )
                active_merge_phase = "page_resolution"
                phase_started = time.monotonic()
                _safe_log_event(audit_logger, active_merge_phase, "started")
                merged_bytes = consensus_md.encode("utf-8")
                projected_page_ranges = project_merged_page_ranges(
                    merged_bytes=merged_bytes,
                    audit=consensus_audit or [],
                    source_maps=runtime_source_page_maps,
                    page_candidate_regions=consensus_metrics.get(
                        "page_candidate_regions", []
                    ),
                    transformation_events=consensus_metrics.get(
                        "document_skeleton_transformations", []
                    ),
                )
                projected_page_ranges = bound_page_resolution_ranges(
                    merged_bytes=merged_bytes,
                    projected_ranges=projected_page_ranges,
                    max_text_bytes=(
                        Config.PAGE_PROVENANCE_LLM_EVIDENCE_BYTES_PER_RANGE
                    ),
                )
                page_batches = build_page_resolution_batches(
                    merged_bytes=merged_bytes,
                    projected_ranges=projected_page_ranges,
                    source_bytes={
                        source: artifact.raw_utf8
                        for source, artifact in source_artifacts.items()
                    },
                    max_ranges_per_batch=(
                        Config.PAGE_PROVENANCE_LLM_MAX_RANGES_PER_BATCH
                    ),
                    context_bytes=Config.PAGE_PROVENANCE_LLM_CONTEXT_BYTES,
                    evidence_bytes_per_range=(
                        Config.PAGE_PROVENANCE_LLM_EVIDENCE_BYTES_PER_RANGE
                    ),
                )
                page_choices = {}
                page_llm_receipts = []
                if llm is not None:
                    page_choices, page_llm_receipts = resolve_page_batches(
                        page_batches, llm.resolve_page_choices,
                        concurrency=Config.PAGE_PROVENANCE_LLM_CONCURRENCY,
                        may_submit=lambda: _recovery_time_available(job_started_monotonic),
                    )
                    llm.selection_call_traces.extend(page_llm_receipts)
                source_page_counts = {
                    source_map["expected_page_count"]
                    for source_map in runtime_source_page_maps.values()
                }
                if len(source_page_counts) != 1:
                    raise ValueError("source page maps disagree on PDF page count")
                page_provenance_bytes, page_summary = (
                    finalize_merged_page_provenance_bytes(
                        pdf_sha256=pdf_sha256,
                        expected_page_count=next(iter(source_page_counts)),
                        merged_bytes=merged_bytes,
                        audit_sha256=canonical_json_line_sha256(
                            list(consensus_audit or [])
                        ),
                        merge_contract_id=Config.MERGE_CONTRACT_ID,
                        source_map_sha256=runtime_source_page_map_sha256,
                        projected_ranges=projected_page_ranges,
                        llm_choices=page_choices,
                        llm_receipts=page_llm_receipts,
                    )
                )
                consensus_metrics["page_provenance"] = {
                    **page_summary,
                    "llm_batch_count": len(page_llm_receipts),
                    "llm_outcomes": dict(
                        sorted(
                            Counter(
                                receipt.get("outcome", "unknown")
                                for receipt in page_llm_receipts
                            ).items()
                        )
                    ),
                }
                page_resolution_duration = round(time.monotonic() - phase_started, 3)
                phase_metrics = {
                    "text_merge_duration_s": text_merge_duration,
                    "page_resolution_duration_s": page_resolution_duration,
                    "page_resolution_batch_count": len(page_batches),
                    "page_resolution_completed_batch_count": len(page_llm_receipts),
                    "page_resolution_unsubmitted_batch_count": len(page_batches) - len(page_llm_receipts),
                    "page_resolution_concurrency": Config.PAGE_PROVENANCE_LLM_CONCURRENCY,
                    "page_resolution_outcomes": consensus_metrics["page_provenance"]["llm_outcomes"],
                }
                consensus_metrics["merge_phase_timings"] = phase_metrics
                _safe_log_event(
                    audit_logger, "page_resolution", "completed",
                    duration_s=page_resolution_duration, **phase_metrics,
                )
                log.info(
                    "Merge phase timings: text_merge=%.3fs page_resolution=%.3fs "
                    "page_batches=%d/%d concurrency=%d outcomes=%s",
                    text_merge_duration, page_resolution_duration,
                    len(page_llm_receipts), len(page_batches),
                    Config.PAGE_PROVENANCE_LLM_CONCURRENCY,
                    phase_metrics["page_resolution_outcomes"],
                    extra={"_event": "merge_phase_timings", **{
                        f"_{key}": value for key, value in phase_metrics.items()
                    }},
                )
                active_merge_phase = None
                if Config.PDFX_BENCHMARK_MODE:
                    consensus_metrics["runtime_execution"]["stage_events"] = (
                        audit_logger.timing_events()
                        if audit_logger is not None
                        and hasattr(audit_logger, "timing_events")
                        else []
                    )
                if llm is not None:
                    consensus_metrics["llm_usage"] = llm.usage.summary()

                persist_merge_bundle(
                    merged_path=merged_cache_path,
                    metrics_path=metrics_cache_path,
                    audit_path=audit_cache_path,
                    text=merged_md,
                    metrics=consensus_metrics,
                    audit=consensus_audit or [],
                    artifacts=source_artifacts,
                    skeletons=runtime_skeletons,
                    expected_contract_id=Config.MERGE_CONTRACT_ID,
                    page_provenance_bytes=page_provenance_bytes,
                    pdf_sha256=pdf_sha256,
                    source_page_map_sha256=runtime_source_page_map_sha256,
                    alias_path=merged_alias_path,
                )

                merge_duration = round(time.monotonic() - started, 3)
                _safe_log_event(
                    audit_logger,
                    stage,
                    "completed",
                    duration_s=merge_duration,
                )
                completed_stages.append("llm_merge")
            except Exception as exc:
                if active_merge_phase is not None:
                    _safe_log_event(
                        audit_logger, active_merge_phase, "failed",
                        error_code=type(exc).__name__,
                        duration_s=round(time.monotonic() - phase_started, 3),
                    )
                _safe_log_event(
                    audit_logger,
                    stage,
                    "failed",
                    detail=str(exc),
                    duration_s=round(time.monotonic() - started, 3),
                )
                raise

    # --- Optional text-only image review -------------------------------------

    images = _list_images(file_hash) if extract_images else []
    if review_images:
        stage = "image_review"
        if not images:
            _safe_log_event(audit_logger, stage, "skipped", detail="No extracted images to review")
            completed_stages.append("image_review")
        else:
            if not Config.OPENAI_API_KEY:
                raise ValueError("review_images=true but OPENAI_API_KEY is not set")
            if llm is None:
                llm = _build_llm()
                if _shared is not None:
                    _shared["llm"] = llm

            _emit_progress("image_review", "Reviewing extracted image context with LLM")
            started = time.monotonic()
            _safe_log_event(audit_logger, stage, "started", image_count=len(images))
            images = _review_images_with_text_context(images, llm, log=log)
            try:
                write_image_manifest(_get_images_dir(file_hash), images)
            except Exception as exc:
                log.warning("Failed to update image manifest after text review: %s", exc)
            _safe_log_event(
                audit_logger,
                stage,
                "completed",
                duration_s=round(time.monotonic() - started, 3),
                image_count=len(images),
            )
            completed_stages.append("image_review")
    elif images:
        images = [strip_text_image_review(image) for image in images]

    # --- Compute LLM cost tracking -------------------------------------------
    llm_cost_usd = None
    llm_usage_json = None
    if llm is not None:
        llm_cost_usd, llm_usage_json = _compute_llm_accounting(llm, log=log)
        if llm_usage_json is not None:
            _record_llm_cost_alert(llm_cost_usd, llm_usage_json, log=log)
            usage_summary = llm.usage.summary()
            if llm_cost_usd is not None:
                log.info(
                    "LLM cost tracking: $%.4f (%d tokens)",
                    llm_cost_usd, usage_summary["total_tokens"],
                    extra={
                        "_event": "llm_cost_computed",
                        "_llm_cost_usd": llm_cost_usd,
                        "_llm_total_tokens": usage_summary["total_tokens"],
                    },
                )

    merged_preview = merged_md
    if merged_preview:
        if len(merged_preview) > 1000:
            merged_preview = merged_preview[:1000] + "..."

    _emit_progress("finalizing", "Uploading artifacts and finalizing")

    result = {
        "status": "success",
        "extraction_status": (
            "partial_success" if extractor_failures else "complete_success"
        ),
        "file_hash": file_hash,
        "methods_used": methods_used,
        "available_extractors": sorted(extractions),
        "failed_extractors": sorted(extractor_failures),
        "extractor_failures": extractor_failures,
        "retried_extractors": retried_extractors,
        "emergency_ocr_used": emergency_ocr_used,
        "cached_methods": cached_methods,
        "extractions": {m: extractions[m][:500] + "..." if len(extractions[m]) > 500 else extractions[m]
                        for m in extractions},
        "merged_output": merged_preview,
        "merged_cache_path": merged_cache_path,
        "download_paths": {m: _cached_path(file_hash, m) for m in methods_used},
        "run_log_path": run_log_path,
        "extract_images": extract_images,
        "review_images": review_images,
        "images": images,
        "has_images": bool(images),
    }

    if consensus_metrics is not None:
        result["consensus_metrics"] = consensus_metrics
    if merged_cache_path:
        result["merge_contract_id"] = Config.MERGE_CONTRACT_ID
        result["merge_metrics_path"] = metrics_cache_path
        result["merge_audit_path"] = audit_cache_path
        result["page_provenance_cache_path"] = str(
            merged_page_provenance_path(merged_cache_path)
        )
        result["native_structure_receipt_digests"] = runtime_native_receipts
        result["document_skeleton_candidate_ids"] = runtime_skeleton_ids
        result["document_skeleton_candidate_projection_ids"] = (
            runtime_skeleton_projection_ids
        )
        result["pdf_sha256"] = pdf_sha256
        result["source_page_map_sha256"] = runtime_source_page_map_sha256

    if llm_usage_json is not None:
        result["llm_usage_json"] = llm_usage_json
    if llm_cost_usd is not None:
        result["llm_cost_usd"] = llm_cost_usd

    if audit_cache_path:
        result["download_paths"]["audit"] = audit_cache_path
    if merged_cache_path:
        result["download_paths"]["page_provenance"] = str(
            merged_page_provenance_path(merged_cache_path)
        )

    return result
