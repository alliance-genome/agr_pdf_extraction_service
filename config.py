import os

class Config:
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploaded_pdfs')
    CACHE_FOLDER = os.path.join(os.getcwd(), 'extraction_cache')
    ALLOWED_EXTENSIONS = {'pdf'}
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your-key-here")
    GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")
    LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
    
