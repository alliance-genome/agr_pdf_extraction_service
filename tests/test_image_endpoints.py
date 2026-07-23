"""Tests for image download endpoints and path traversal defense."""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = tempfile.mkdtemp()
    app.config["CACHE_FOLDER"] = tempfile.mkdtemp()
    app.config["OPENAI_API_KEY"] = "dummy"
    app.config["LLM_MODEL"] = "dummy"
    app.config["EXTRACTION_CONFIG_VERSION"] = "1"
    app.config["GROBID_REQUEST_TIMEOUT"] = 120
    app.config["GROBID_INCLUDE_COORDINATES"] = False
    app.config["GROBID_INCLUDE_RAW_CITATIONS"] = False
    app.config["DOCLING_DEVICE"] = "cpu"
    app.config["MARKER_DEVICE"] = "cpu"
    app.config["CONSENSUS_NEAR_THRESHOLD"] = 0.92
    app.config["CONSENSUS_LEVENSHTEIN_THRESHOLD"] = 0.90
    app.config["CONSENSUS_CONFLICT_RATIO_FALLBACK"] = 0.4
    app.config["CONSENSUS_ALIGNMENT_CONFIDENCE_MIN"] = 0.5
    app.config["CONSENSUS_ALWAYS_ESCALATE_TABLES"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def app_with_images(client):
    """Set up a cache folder with a test image."""
    app = client.application
    cache_folder = app.config["CACHE_FOLDER"]
    file_hash = "testhash123"
    version = app.config["EXTRACTION_CONFIG_VERSION"]

    # Create images directory matching the naming convention
    images_dir = os.path.join(cache_folder, "images", f"v{version}_{file_hash}_marker")
    os.makedirs(images_dir, exist_ok=True)

    # Write a fake PNG image
    img_path = os.path.join(images_dir, "figure_001.png")
    with open(img_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # minimal PNG-like content

    return {
        "file_hash": file_hash,
        "images_dir": images_dir,
        "image_filename": "figure_001.png",
        "image_path": img_path,
    }


class TestImageDownloadEndpoint:
    @patch("app.api._get_run_by_process_id", return_value=None)
    @patch("celery_app.celery.AsyncResult")
    def test_image_download_valid(self, async_result, _run, client, app_with_images):
        """A completed process ID serves its locally cached image."""
        fh = app_with_images["file_hash"]
        fn = app_with_images["image_filename"]
        async_result.return_value = MagicMock(
            state="SUCCESS",
            result={"file_hash": fh},
        )
        response = client.get(f"/api/v1/extract/process-1/images/{fn}")
        assert response.status_code == 200
        assert response.content_type == "image/png"
        assert len(response.data) > 0

    @patch("app.api._get_run_by_process_id", return_value=None)
    @patch("celery_app.celery.AsyncResult")
    def test_image_download_not_found(self, async_result, _run, client, app_with_images):
        """Non-existent image should return 404."""
        fh = app_with_images["file_hash"]
        async_result.return_value = MagicMock(
            state="SUCCESS",
            result={"file_hash": fh},
        )
        response = client.get("/api/v1/extract/process-1/images/nonexistent.png")
        assert response.status_code == 404

    def test_image_download_traversal_blocked(self, client, app_with_images):
        """Path traversal attempts should be blocked."""
        response = client.get("/api/v1/extract/process-1/images/../../etc/passwd")
        assert response.status_code in (400, 404)

    def test_image_download_traversal_encoded(self, client, app_with_images):
        """URL-encoded path traversal should also be blocked."""
        response = client.get(
            "/api/v1/extract/process-1/images/..%2F..%2Fetc%2Fpasswd"
        )
        assert response.status_code in (400, 404)


class TestImageListFromCachedJob:
    def test_list_images_finds_images_for_hash(self, client, app_with_images):
        """list_images should find images for a given file hash."""
        from app.utils import list_images

        with client.application.app_context():
            images = list_images(app_with_images["file_hash"])
            assert len(images) == 1
            assert images[0]["filename"] == "figure_001.png"
            assert images[0]["size_bytes"] > 0

    def test_list_images_empty_for_unknown_hash(self, client):
        """list_images should return empty list for unknown hash."""
        from app.utils import list_images

        with client.application.app_context():
            images = list_images("unknown_hash_xyz")
            assert images == []
