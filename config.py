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
    # Bucket name: read from SSM Parameter Store (/pdfx/audit-s3-bucket),
    # with AUDIT_S3_BUCKET env var as fallback for local dev.
    AUDIT_S3_BUCKET = os.environ.get("AUDIT_S3_BUCKET", "")
    AUDIT_S3_PREFIX = os.environ.get("AUDIT_S3_PREFIX", "pdfx/audit")
    AUDIT_S3_BUCKET_SSM_PARAM = os.environ.get("AUDIT_S3_BUCKET_SSM_PARAM", "/pdfx/audit-s3-bucket")
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
    LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4")
    LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "medium")
    LLM_CONFLICT_BATCH_SIZE = int(os.environ.get("LLM_CONFLICT_BATCH_SIZE", 500))
    LLM_CONFLICT_MAX_WORKERS = int(os.environ.get("LLM_CONFLICT_MAX_WORKERS", 100))
    LLM_CONFLICT_RETRY_ROUNDS = int(os.environ.get("LLM_CONFLICT_RETRY_ROUNDS", 2))

    # ---- Per-call-type model + reasoning defaults ----------------------------
    # Each call type can override both model and reasoning effort individually.
    # Falls back to LLM_MODEL / LLM_REASONING_EFFORT when not set.
    LLM_MODEL_ZONE_RESOLUTION = os.environ.get("LLM_MODEL_ZONE_RESOLUTION", "gpt-5.4")
    LLM_REASONING_ZONE_RESOLUTION = os.environ.get("LLM_REASONING_ZONE_RESOLUTION", "medium")
    LLM_MODEL_GENERAL_RESCUE = os.environ.get("LLM_MODEL_GENERAL_RESCUE", "gpt-5.4")
    LLM_REASONING_GENERAL_RESCUE = os.environ.get("LLM_REASONING_GENERAL_RESCUE", "medium")
    LLM_MODEL_NUMERIC_RESCUE = os.environ.get("LLM_MODEL_NUMERIC_RESCUE", "gpt-5.4")
    LLM_REASONING_NUMERIC_RESCUE = os.environ.get("LLM_REASONING_NUMERIC_RESCUE", "medium")
    LLM_MODEL_CONFLICT_BATCH = os.environ.get("LLM_MODEL_CONFLICT_BATCH", "gpt-5.4")
    LLM_REASONING_CONFLICT_BATCH = os.environ.get("LLM_REASONING_CONFLICT_BATCH", "medium")

    # ---- LLM pricing (USD per 1M tokens) ------------------------------------
    LLM_PRICING = {
        "gpt-5.2": {"input": 1.75, "output": 14.00, "cached_input": 0.175},
        "gpt-5.4": {"input": 2.50, "output": 15.00, "cached_input": 0.25},
        "gpt-5.4-mini": {"input": 0.75, "output": 4.50, "cached_input": 0.075},
        "gpt-5-mini": {"input": 0.25, "output": 2.00, "cached_input": 0.025},
        "gpt-5-nano": {"input": 0.05, "output": 0.40, "cached_input": 0.005},
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
    CONSENSUS_MEDIAN_SOURCE_MAX_MICRO_CONFLICTS = int(
        os.environ.get("CONSENSUS_MEDIAN_SOURCE_MAX_MICRO_CONFLICTS", 20),
    )
    CONSENSUS_ALIGNMENT_CONFIDENCE_MIN = float(os.environ.get("CONSENSUS_ALIGNMENT_CONFIDENCE_MIN", 0.5))
    CONSENSUS_ALWAYS_ESCALATE_TABLES = os.environ.get("CONSENSUS_ALWAYS_ESCALATE_TABLES", "true").lower() == "true"
    # If true, AGREE_NEAR is disabled for any block containing numbers; such blocks
    # become CONFLICT unless they qualify for AGREE_EXACT. This is the safest option
    # for scientific PDFs but can increase LLM usage.
    CONSENSUS_STRICT_NUMERIC_NEAR = os.environ.get("CONSENSUS_STRICT_NUMERIC_NEAR", "true").lower() == "true"
    CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES = os.environ.get("CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES", "true").lower() == "true"
    CONSENSUS_ALIGNMENT_ANCHOR_PARTITIONING_ENABLED = (
        os.environ.get("CONSENSUS_ALIGNMENT_ANCHOR_PARTITIONING_ENABLED", "true").lower() == "true"
    )
    CONSENSUS_ALIGNMENT_ANCHOR_MIN_SCORE = float(
        os.environ.get("CONSENSUS_ALIGNMENT_ANCHOR_MIN_SCORE", 0.72),
    )
    CONSENSUS_ALIGNMENT_ANCHOR_INCLUDE_STRUCTURAL = (
        os.environ.get("CONSENSUS_ALIGNMENT_ANCHOR_INCLUDE_STRUCTURAL", "false").lower() == "true"
    )
    CONSENSUS_ALIGNMENT_ANCHOR_MAX_HEADING_LEVEL = int(
        os.environ.get("CONSENSUS_ALIGNMENT_ANCHOR_MAX_HEADING_LEVEL", 2),
    )
    CONSENSUS_ALIGNMENT_AMBIGUITY_DELTA = float(
        os.environ.get("CONSENSUS_ALIGNMENT_AMBIGUITY_DELTA", 0.03),
    )
    CONSENSUS_ALIGNMENT_SEMANTIC_RERANK_ENABLED = (
        os.environ.get("CONSENSUS_ALIGNMENT_SEMANTIC_RERANK_ENABLED", "true").lower() == "true"
    )
    CONSENSUS_ALIGNMENT_SEMANTIC_MARGIN = float(
        os.environ.get("CONSENSUS_ALIGNMENT_SEMANTIC_MARGIN", 0.02),
    )
    CONSENSUS_ALIGNMENT_LLM_TIEBREAK_ENABLED = (
        os.environ.get("CONSENSUS_ALIGNMENT_LLM_TIEBREAK_ENABLED", "true").lower() == "true"
    )

    # ---- Micro-conflict extraction --------------------------------------------
    MICRO_CONFLICT_CONTEXT_CAP = int(os.environ.get("MICRO_CONFLICT_CONTEXT_CAP", 30))
    MICRO_CONFLICT_HIGH_DIVERGENCE_RATIO_THRESHOLD = float(
        os.environ.get("MICRO_CONFLICT_HIGH_DIVERGENCE_RATIO_THRESHOLD", 0.40),
    )
    MICRO_CONFLICT_HIGH_DIVERGENCE_SPAN_THRESHOLD = int(
        os.environ.get("MICRO_CONFLICT_HIGH_DIVERGENCE_SPAN_THRESHOLD", 12),
    )
    MICRO_CONFLICT_COALESCE_GAP = int(os.environ.get("MICRO_CONFLICT_COALESCE_GAP", 8))
    MICRO_CONFLICT_HIGH_DIVERGENCE_MIN_TOKENS = int(
        os.environ.get("MICRO_CONFLICT_HIGH_DIVERGENCE_MIN_TOKENS", 10),
    )

    # ---- Header hierarchy resolution -----------------------------------------
    CONSENSUS_HIERARCHY_ENABLED = os.environ.get("CONSENSUS_HIERARCHY_ENABLED", "true").lower() == "true"
    HIERARCHY_LLM_MODEL = os.environ.get("HIERARCHY_LLM_MODEL", "gpt-5.4")
    HIERARCHY_LLM_REASONING = os.environ.get("HIERARCHY_LLM_REASONING", "medium")
