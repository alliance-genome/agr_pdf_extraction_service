import os
import hashlib

from flask import current_app as app

from app.image_metadata import (
    IMAGE_MANIFEST_FILENAME,
    list_image_directory,
    list_manifest_images,
)


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
    manifest_images = list_manifest_images(images_dir)
    if manifest_images is not None:
        return manifest_images
    return list_image_directory(images_dir)


def has_image_extraction_manifest(file_hash):
    return os.path.exists(os.path.join(get_images_dir(file_hash), IMAGE_MANIFEST_FILENAME))
