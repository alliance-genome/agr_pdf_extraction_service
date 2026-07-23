import os


class Config:
    TASK_SOFT_TIME_LIMIT_SECONDS = int(
        os.environ.get("TASK_SOFT_TIME_LIMIT_SECONDS", 1800)
    )
    TASK_HARD_TIME_LIMIT_SECONDS = int(
        os.environ.get("TASK_HARD_TIME_LIMIT_SECONDS", 2100)
    )
    EXTRACTION_FINALIZATION_RESERVE_SECONDS = int(
        os.environ.get("EXTRACTION_FINALIZATION_RESERVE_SECONDS", 300)
    )
    # ---- Upload limits -------------------------------------------------------
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 500 * 1024 * 1024))  # 500 MiB

    # ---- Storage paths -------------------------------------------------------
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploaded_pdfs"))
    CACHE_FOLDER = os.environ.get("CACHE_FOLDER", os.path.join(os.getcwd(), "extraction_cache"))
    ALLOWED_EXTENSIONS = {"pdf"}

    # ---- Cache versioning ----------------------------------------------------
    # Bump this when extractor config changes to invalidate old cached outputs.
    EXTRACTION_CONFIG_VERSION = os.environ.get("EXTRACTION_CONFIG_VERSION", "6")
    MERGE_CONTRACT_ID = "pdfx-native-skeleton-selection"

    # ---- Celery / Redis ------------------------------------------------------
    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    HEALTH_BUSY_RUN_MAX_AGE_SECONDS = int(os.environ.get("HEALTH_BUSY_RUN_MAX_AGE_SECONDS", 6 * 60 * 60))

    # ---- Database --------------------------------------------------------
    DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://pdfx:pdfx@localhost:5432/pdfx")

    # ---- Audit Trail (S3) -----------------------------------------------
    # Bucket name: read from SSM Parameter Store (/pdfx/audit-s3-bucket),
    # with AUDIT_S3_BUCKET env var as fallback for local dev.
    AUDIT_S3_BUCKET = os.environ.get("AUDIT_S3_BUCKET", "")
    AUDIT_S3_PREFIX = os.environ.get("AUDIT_S3_PREFIX", "pdfx/audit")
    AUDIT_S3_BUCKET_SSM_PARAM = os.environ.get("AUDIT_S3_BUCKET_SSM_PARAM", "/pdfx/audit-s3-bucket")
    AUDIT_FLUSH_ON_EVENT = os.environ.get("AUDIT_FLUSH_ON_EVENT", "false").lower() == "true"
    IMAGE_URL_TTL_SECONDS = int(os.environ.get("IMAGE_URL_TTL_SECONDS", 3600))
    IMAGE_RETENTION_TTL_SECONDS = int(os.environ.get("IMAGE_RETENTION_TTL_SECONDS", 7 * 24 * 60 * 60))
    AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    # ---- GROBID --------------------------------------------------------------
    GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")
    GROBID_REQUEST_TIMEOUT = int(os.environ.get("GROBID_REQUEST_TIMEOUT", 120))
    GROBID_INCLUDE_COORDINATES = os.environ.get("GROBID_INCLUDE_COORDINATES", "true").lower() == "true"
    GROBID_INCLUDE_RAW_CITATIONS = os.environ.get("GROBID_INCLUDE_RAW_CITATIONS", "false").lower() == "true"

    # ---- Docling -------------------------------------------------------------
    DOCLING_DEVICE = os.environ.get("DOCLING_DEVICE", "cpu")

    # ---- Marker --------------------------------------------------------------
    MARKER_DEVICE = os.environ.get("MARKER_DEVICE", "cpu")
    MARKER_READY_FILE = os.environ.get(
        "PDFX_MARKER_READY_FILE",
        os.path.join(CACHE_FOLDER, "marker_worker_ready.json"),
    )
    HEALTH_REQUIRE_MARKER_READY = os.environ.get("PDFX_HEALTH_REQUIRE_MARKER_READY", "false").lower() == "true"
    WORKER_PRELOAD_MARKER_MODELS = os.environ.get("PDFX_WORKER_PRELOAD_MARKER_MODELS", "off").strip().lower()
    WORKER_PRELOAD_MARKER_REQUIRED = os.environ.get("PDFX_WORKER_PRELOAD_MARKER_REQUIRED", "false").lower() == "true"
    WORKER_PRELOAD_MARKER_EXTRACT_IMAGES = (
        os.environ.get("PDFX_WORKER_PRELOAD_MARKER_EXTRACT_IMAGES", "true").lower() == "true"
    )

    # ---- LLM (merge) ---------------------------------------------------------
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    SOURCE_SELECTION_MODEL = os.environ.get("SOURCE_SELECTION_MODEL", "gpt-5.6-terra")
    SOURCE_SELECTION_REASONING = os.environ.get("SOURCE_SELECTION_REASONING", "medium")
    HARD_SELECTION_MODEL = os.environ.get("HARD_SELECTION_MODEL", "gpt-5.6-sol")
    HARD_SELECTION_REASONING = os.environ.get("HARD_SELECTION_REASONING", "high")
    LLM_OPENAI_TIMEOUT_SECONDS = float(os.environ.get("LLM_OPENAI_TIMEOUT_SECONDS", 180))
    LLM_OPENAI_MAX_RETRIES = int(os.environ.get("LLM_OPENAI_MAX_RETRIES", 1))
    LLM_COST_ALERT_USD_PER_JOB = float(
        os.environ.get("LLM_COST_ALERT_USD_PER_JOB", 2.0)
    )

    IMAGE_TEXT_REVIEW_MODEL = os.environ.get("IMAGE_TEXT_REVIEW_MODEL", "gpt-5.6-luna")
    IMAGE_TEXT_REVIEW_REASONING = os.environ.get("IMAGE_TEXT_REVIEW_REASONING", "medium")

    # ---- LLM pricing (USD per 1M tokens) ------------------------------------
    LLM_PRICING = {
        "gpt-5.6-sol": {"input": 5.00, "output": 30.00, "cached_input": 0.50},
        "gpt-5.6-terra": {"input": 2.50, "output": 15.00, "cached_input": 0.25},
        "gpt-5.6-luna": {"input": 1.00, "output": 6.00, "cached_input": 0.10},
    }

    # ---- Source-backed merge -------------------------------------------------
    PDFX_BENCHMARK_MODE = (
        os.environ.get("PDFX_BENCHMARK_MODE", "false").lower() == "true"
    )
