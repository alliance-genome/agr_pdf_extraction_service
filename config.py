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
    EXTRACTION_CONFIG_VERSION = os.environ.get("EXTRACTION_CONFIG_VERSION", "3")

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

    # ---- Consensus pipeline --------------------------------------------------
    CONSENSUS_ENABLED = os.environ.get("CONSENSUS_ENABLED", "true").lower() == "true"
    CONSENSUS_NEAR_THRESHOLD = float(os.environ.get("CONSENSUS_NEAR_THRESHOLD", 0.92))
    CONSENSUS_LEVENSHTEIN_THRESHOLD = float(os.environ.get("CONSENSUS_LEVENSHTEIN_THRESHOLD", 0.90))
    CONSENSUS_CONFLICT_RATIO_FALLBACK = float(os.environ.get("CONSENSUS_CONFLICT_RATIO_FALLBACK", 0.4))
    CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = float(os.environ.get("CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK", 0.5))
    CONSENSUS_ALWAYS_ESCALATE_TABLES = os.environ.get("CONSENSUS_ALWAYS_ESCALATE_TABLES", "true").lower() == "true"
    CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES = os.environ.get("CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES", "true").lower() == "true"

    # ---- Header hierarchy resolution -----------------------------------------
    CONSENSUS_HIERARCHY_ENABLED = os.environ.get("CONSENSUS_HIERARCHY_ENABLED", "true").lower() == "true"
    HIERARCHY_LLM_MODEL = os.environ.get("HIERARCHY_LLM_MODEL", "gpt-5.2")
    HIERARCHY_LLM_REASONING = os.environ.get("HIERARCHY_LLM_REASONING", "medium")
