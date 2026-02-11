import os
import logging
from celery import Celery
from config import Config

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


def _is_cached(file_hash, method):
    return os.path.exists(_cached_path(file_hash, method))


# ---------------------------------------------------------------------------
# Extraction task
# ---------------------------------------------------------------------------

@celery.task(bind=True, name="extract_pdf")
def extract_pdf(self, pdf_path, methods, merge=False):
    """
    Run PDF extraction in the background.

    Args:
        pdf_path: Absolute path to the uploaded PDF on disk.
        methods: List of extractor names, e.g. ["grobid", "docling", "marker"].
        merge: Whether to run the LLM merge step after extraction.

    Returns:
        dict with status, file_hash, per-method outputs, and optional merged output.
    """
    from app.services.grobid_service import Grobid
    from app.services.docling_service import Docling
    from app.services.marker_service import Marker
    from app.services.llm_service import LLM

    os.makedirs(Config.CACHE_FOLDER, exist_ok=True)

    file_hash = _get_file_hash(pdf_path)
    extractions = {}
    methods_used = []
    cached_methods = []
    total_steps = len(methods) + (1 if merge else 0)
    current_step = 0

    # --- Run each extractor ---------------------------------------------------

    if "grobid" in methods:
        output_path = _cached_path(file_hash, "grobid")
        if _is_cached(file_hash, "grobid"):
            cached_methods.append("grobid")
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
            grobid.extract(pdf_path, output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            extractions["grobid"] = f.read()
        methods_used.append("grobid")
        current_step += 1

    if "docling" in methods:
        output_path = _cached_path(file_hash, "docling")
        if _is_cached(file_hash, "docling"):
            cached_methods.append("docling")
        else:
            self.update_state(state="PROGRESS", meta={
                "step": "docling", "current": current_step, "total": total_steps
            })
            logger.info("Running Docling extraction...")
            docling = Docling(device=Config.DOCLING_DEVICE)
            docling.extract(pdf_path, output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            extractions["docling"] = f.read()
        methods_used.append("docling")
        current_step += 1

    if "marker" in methods:
        output_path = _cached_path(file_hash, "marker")
        if _is_cached(file_hash, "marker"):
            cached_methods.append("marker")
        else:
            self.update_state(state="PROGRESS", meta={
                "step": "marker", "current": current_step, "total": total_steps
            })
            logger.info("Running Marker extraction...")
            marker = Marker(
                device=Config.MARKER_DEVICE,
                extract_images=Config.MARKER_EXTRACT_IMAGES,
            )
            marker.extract(pdf_path, output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            extractions["marker"] = f.read()
        methods_used.append("marker")
        current_step += 1

    # --- Optional LLM merge ---------------------------------------------------

    merged_md = None
    if merge and extractions:
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError("merge=true but ANTHROPIC_API_KEY is not set")

        version = Config.EXTRACTION_CONFIG_VERSION
        cache_key = f"v{version}_{file_hash}_{'_'.join(sorted(methods))}"
        merged_cache_path = os.path.join(Config.CACHE_FOLDER, f"{cache_key}_merged.md")

        if os.path.exists(merged_cache_path):
            cached_methods.append("merged")
            with open(merged_cache_path, "r", encoding="utf-8") as f:
                merged_md = f.read()
        else:
            self.update_state(state="PROGRESS", meta={
                "step": "llm_merge", "current": current_step, "total": total_steps
            })
            logger.info("Merging outputs with LLM...")
            llm = LLM(
                api_key=Config.ANTHROPIC_API_KEY,
                model=Config.LLM_MODEL,
                max_tokens=Config.LLM_MAX_TOKENS,
            )
            merged_md = llm.extract(
                extractions.get("grobid", ""),
                extractions.get("docling", ""),
                extractions.get("marker", ""),
            )
            with open(merged_cache_path, "w", encoding="utf-8") as f:
                f.write(merged_md)
            with open(_cached_path(file_hash, "merged"), "w", encoding="utf-8") as f:
                f.write(merged_md)

    # --- Cleanup uploaded PDF (cached outputs are kept) -----------------------
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    return {
        "status": "success",
        "file_hash": file_hash,
        "methods_used": methods_used,
        "cached_methods": cached_methods,
        "extractions": {m: extractions[m][:500] + "..." if len(extractions[m]) > 500 else extractions[m]
                        for m in extractions},
        "merged_output": merged_md[:1000] + "..." if merged_md and len(merged_md) > 1000 else merged_md,
        "download_paths": {m: _cached_path(file_hash, m) for m in methods_used},
    }
