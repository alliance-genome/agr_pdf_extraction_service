import os


class Config:
    # ---- Upload limits -------------------------------------------------------
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 100 * 1024 * 1024))  # 100MB

    # ---- Storage paths -------------------------------------------------------
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploaded_pdfs"))
    CACHE_FOLDER = os.environ.get("CACHE_FOLDER", os.path.join(os.getcwd(), "extraction_cache"))
    ALLOWED_EXTENSIONS = {"pdf"}

    # ---- Cache versioning ----------------------------------------------------
    # Bump this when extractor config changes to invalidate old cached outputs.
    EXTRACTION_CONFIG_VERSION = os.environ.get("EXTRACTION_CONFIG_VERSION", "4")

    # ---- Celery / Redis ------------------------------------------------------
    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # ---- Database --------------------------------------------------------
    DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://pdfx:pdfx@localhost:5432/pdfx")

    # ---- Audit Trail (S3) -----------------------------------------------
    AUDIT_S3_BUCKET = os.environ.get("AUDIT_S3_BUCKET", "agr-pdf-extraction-benchmark")
    AUDIT_S3_PREFIX = os.environ.get("AUDIT_S3_PREFIX", "pdfx/audit")

    # ---- AWS (for S3 access from off-AWS deployment) --------------------
    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    # ---- GROBID --------------------------------------------------------------
    GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")
    GROBID_REQUEST_TIMEOUT = int(os.environ.get("GROBID_REQUEST_TIMEOUT", 120))
    GROBID_INCLUDE_COORDINATES = os.environ.get("GROBID_INCLUDE_COORDINATES", "false").lower() == "true"
    GROBID_INCLUDE_RAW_CITATIONS = os.environ.get("GROBID_INCLUDE_RAW_CITATIONS", "false").lower() == "true"

    # ---- Docling -------------------------------------------------------------
    DOCLING_DEVICE = os.environ.get("DOCLING_DEVICE", "cpu")

    # ---- Marker --------------------------------------------------------------
    MARKER_DEVICE = os.environ.get("MARKER_DEVICE", "cpu")
    MARKER_EXTRACT_IMAGES = os.environ.get("MARKER_EXTRACT_IMAGES", "true").lower() == "true"

    # ---- LLM (merge) ---------------------------------------------------------
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.2")
    LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "medium")
    LLM_CONFLICT_BATCH_SIZE = int(os.environ.get("LLM_CONFLICT_BATCH_SIZE", 10))
    LLM_CONFLICT_MAX_WORKERS = int(os.environ.get("LLM_CONFLICT_MAX_WORKERS", 4))
    LLM_CONFLICT_RETRY_ROUNDS = int(os.environ.get("LLM_CONFLICT_RETRY_ROUNDS", 2))

    # ---- Per-call-type model defaults ----------------------------------------
    LLM_MODEL_ZONE_RESOLUTION = os.environ.get("LLM_MODEL_ZONE_RESOLUTION", "gpt-5-mini")
    LLM_MODEL_FULL_MERGE = os.environ.get("LLM_MODEL_FULL_MERGE", "gpt-5.2")
    LLM_MODEL_RESCUE = os.environ.get("LLM_MODEL_RESCUE", "gpt-5-mini")
    LLM_MODEL_CONFLICT_BATCH = os.environ.get("LLM_MODEL_CONFLICT_BATCH", "gpt-5.2")

    # ---- Zone resolution escalation threshold --------------------------------
    ZONE_ESCALATION_THRESHOLD = int(os.environ.get("ZONE_ESCALATION_THRESHOLD", 20000))
    ZONE_ESCALATION_MODEL = os.environ.get("ZONE_ESCALATION_MODEL", "gpt-5.2")

    # ---- LLM pricing (USD per 1M tokens) ------------------------------------
    LLM_PRICING = {
        "gpt-5.2": {"input": 1.75, "output": 14.00, "cached_input": 0.175},
        "gpt-5-mini": {"input": 0.25, "output": 2.00, "cached_input": 0.025},
        "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    }

    # ---- Consensus pipeline --------------------------------------------------
    CONSENSUS_ENABLED = os.environ.get("CONSENSUS_ENABLED", "true").lower() == "true"
    CONSENSUS_NEAR_THRESHOLD = float(os.environ.get("CONSENSUS_NEAR_THRESHOLD", 0.92))
    CONSENSUS_LEVENSHTEIN_THRESHOLD = float(os.environ.get("CONSENSUS_LEVENSHTEIN_THRESHOLD", 0.90))
    CONSENSUS_CONFLICT_RATIO_FALLBACK = float(os.environ.get("CONSENSUS_CONFLICT_RATIO_FALLBACK", 0.4))
    CONSENSUS_CONFLICT_RATIO_TEXTUAL_FALLBACK = float(
        os.environ.get("CONSENSUS_CONFLICT_RATIO_TEXTUAL_FALLBACK", 0.4),
    )
    CONSENSUS_CONFLICT_RATIO_STRUCTURED_FALLBACK = float(
        os.environ.get("CONSENSUS_CONFLICT_RATIO_STRUCTURED_FALLBACK", 0.85),
    )
    CONSENSUS_LOCALIZED_CONFLICT_SPAN_MAX = float(
        os.environ.get("CONSENSUS_LOCALIZED_CONFLICT_SPAN_MAX", 0.35),
    )
    CONSENSUS_LOCALIZED_CONFLICT_RELIEF = float(
        os.environ.get("CONSENSUS_LOCALIZED_CONFLICT_RELIEF", 0.15),
    )
    CONSENSUS_LOCALIZED_CONFLICT_MAX_BLOCKS = int(
        os.environ.get("CONSENSUS_LOCALIZED_CONFLICT_MAX_BLOCKS", 25),
    )
    CONSENSUS_LAYERED_ENABLED = os.environ.get("CONSENSUS_LAYERED_ENABLED", "true").lower() == "true"
    CONSENSUS_LAYERED_MEDIUM_SIM_THRESHOLD = float(
        os.environ.get("CONSENSUS_LAYERED_MEDIUM_SIM_THRESHOLD", 0.60),
    )
    CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = float(os.environ.get("CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK", 0.5))
    CONSENSUS_ALWAYS_ESCALATE_TABLES = os.environ.get("CONSENSUS_ALWAYS_ESCALATE_TABLES", "true").lower() == "true"
    CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES = os.environ.get("CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES", "true").lower() == "true"

    # ---- Conflict zone grouping -----------------------------------------------
    CONSENSUS_ZONE_FLANKING_COUNT = int(os.environ.get("CONSENSUS_ZONE_FLANKING_COUNT", 2))

    # ---- Header hierarchy resolution -----------------------------------------
    CONSENSUS_HIERARCHY_ENABLED = os.environ.get("CONSENSUS_HIERARCHY_ENABLED", "true").lower() == "true"
    HIERARCHY_LLM_MODEL = os.environ.get("HIERARCHY_LLM_MODEL", "gpt-5.2")
    HIERARCHY_LLM_REASONING = os.environ.get("HIERARCHY_LLM_REASONING", "medium")
