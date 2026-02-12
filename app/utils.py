import os
import re
import hashlib
from urllib.parse import quote

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


def get_images_dir(file_hash):
    """Compute the images directory for a given file hash. Works for both fresh and cached."""
    marker_cache_path = get_cached_path(file_hash, "marker")
    stem = os.path.splitext(os.path.basename(marker_cache_path))[0]
    return os.path.join(app.config["CACHE_FOLDER"], "images", stem)


def list_images(file_hash):
    """List image files for a file_hash. Returns list of {"filename": ..., "size_bytes": ...}."""
    images_dir = get_images_dir(file_hash)
    if not os.path.isdir(images_dir):
        return []
    result = []
    for f in sorted(os.listdir(images_dir)):
        fpath = os.path.join(images_dir, f)
        if os.path.isfile(fpath) and f.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            result.append({"filename": f, "size_bytes": os.path.getsize(fpath)})
    return result


def rewrite_image_paths(markdown, file_hash):
    """Rewrite ![alt](filename) to ![alt](/download/{hash}/images/{filename}) at response time."""
    images_dir = get_images_dir(file_hash)

    def _replace(match):
        alt, filename = match.group(1), match.group(2)
        basename = os.path.basename(filename)
        if os.path.exists(os.path.join(images_dir, basename)):
            return f"![{alt}](/download/{file_hash}/images/{quote(basename)})"
        return match.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _replace, markdown)
