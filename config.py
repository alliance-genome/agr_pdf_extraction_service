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
    EXTRACTION_CONFIG_VERSION = os.environ.get("EXTRACTION_CONFIG_VERSION", "1")

    # ---- Celery / Redis ------------------------------------------------------
    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # ---- GROBID --------------------------------------------------------------
    GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")
    GROBID_REQUEST_TIMEOUT = int(os.environ.get("GROBID_REQUEST_TIMEOUT", 120))
    GROBID_INCLUDE_COORDINATES = os.environ.get("GROBID_INCLUDE_COORDINATES", "false").lower() == "true"
    GROBID_INCLUDE_RAW_CITATIONS = os.environ.get("GROBID_INCLUDE_RAW_CITATIONS", "false").lower() == "true"

    # ---- Docling -------------------------------------------------------------
    DOCLING_DEVICE = os.environ.get("DOCLING_DEVICE", "cpu")

    # ---- Marker --------------------------------------------------------------
    MARKER_DEVICE = os.environ.get("MARKER_DEVICE", "cpu")
    MARKER_EXTRACT_IMAGES = os.environ.get("MARKER_EXTRACT_IMAGES", "false").lower() == "true"

    # ---- LLM (merge) ---------------------------------------------------------
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
    LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", 16000))
