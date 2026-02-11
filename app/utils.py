import os
import hashlib
from flask import current_app as app


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def get_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_cached_path(file_hash, method):
    """Cache path includes config version so config changes invalidate old outputs."""
    version = app.config["EXTRACTION_CONFIG_VERSION"]
    return os.path.join(app.config["CACHE_FOLDER"], f"v{version}_{file_hash}_{method}.md")


def is_extraction_cached(file_hash, method):
    cached_path = get_cached_path(file_hash, method)
    return os.path.exists(cached_path)
