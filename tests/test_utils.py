import os
import tempfile
import pytest
from flask import Flask

from app.utils import (
    allowed_file, get_file_hash, get_cached_path, is_extraction_cached,
    get_images_dir, has_image_extraction_manifest, list_images,
)

@pytest.fixture
def app_context():
    app = Flask(__name__)
    app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
    app.config['CACHE_FOLDER'] = tempfile.mkdtemp()
    app.config['EXTRACTION_CONFIG_VERSION'] = '1'
    with app.app_context():
        yield app

def test_allowed_file_pdf(app_context):
    assert allowed_file("document.pdf") is True

def test_allowed_file_wrong_ext(app_context):
    assert allowed_file("document.txt") is False

def test_get_file_hash(tmp_path, app_context):
    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(b"hello world")
    hash1 = get_file_hash(str(file_path))
    file_path.write_bytes(b"hello world")
    hash2 = get_file_hash(str(file_path))
    assert hash1 == hash2
    file_path.write_bytes(b"something else")
    hash3 = get_file_hash(str(file_path))
    assert hash1 != hash3

def test_get_cached_path(app_context):
    path = get_cached_path("abc123", "grobid")
    assert path.endswith("v1_abc123_grobid.md")
    assert app_context.config['CACHE_FOLDER'] in path

def test_is_extraction_cached(app_context):
    file_hash = "abc123"
    method = "grobid"
    # Should not exist yet
    assert not is_extraction_cached(file_hash, method)
    # Create the file with the versioned name
    cache_file = os.path.join(app_context.config['CACHE_FOLDER'], f"v1_{file_hash}_{method}.md")
    with open(cache_file, "w") as f:
        f.write("test")
    assert is_extraction_cached(file_hash, method)


# ---------------------------------------------------------------------------
# Image utility tests
# ---------------------------------------------------------------------------

def test_get_images_dir_deterministic(app_context):
    """Same hash always gives same directory."""
    dir1 = get_images_dir("abc123")
    dir2 = get_images_dir("abc123")
    assert dir1 == dir2
    assert "images" in dir1
    assert "v1_abc123_marker" in dir1


def test_get_images_dir_different_hashes(app_context):
    """Different hashes produce different directories."""
    dir1 = get_images_dir("abc123")
    dir2 = get_images_dir("def456")
    assert dir1 != dir2


def test_list_images_empty_dir(app_context):
    """No images directory should return empty list."""
    result = list_images("nonexistent_hash")
    assert result == []


def test_list_images_with_images(app_context):
    """Should list image files sorted by name."""
    file_hash = "abc123"
    images_dir = get_images_dir(file_hash)
    os.makedirs(images_dir, exist_ok=True)

    # Create test image files
    for name in ["fig_002.png", "fig_001.png", "fig_003.jpg"]:
        with open(os.path.join(images_dir, name), "wb") as f:
            f.write(b"\x89PNG")  # minimal content

    result = list_images(file_hash)
    assert len(result) == 3
    assert result[0]["filename"] == "fig_001.png"
    assert result[1]["filename"] == "fig_002.png"
    assert result[2]["filename"] == "fig_003.jpg"
    assert all("size_bytes" in r for r in result)


def test_list_images_uses_manifest_metadata(app_context):
    file_hash = "manifestimages"
    images_dir = get_images_dir(file_hash)
    os.makedirs(images_dir, exist_ok=True)

    with open(os.path.join(images_dir, "_page_1_Figure_3.jpeg"), "wb") as f:
        f.write(b"\x89PNG")
    with open(os.path.join(images_dir, ".pdfx-images.json"), "w", encoding="utf-8") as f:
        f.write(
            '{"images":[{"filename":"_page_1_Figure_3.jpeg",'
            '"figure_label":"Figure 2","figure_number":"2","alt_text":"Figure 2"}]}'
        )

    result = list_images(file_hash)
    assert len(result) == 1
    expected = {
        "filename": "_page_1_Figure_3.jpeg",
        "size_bytes": 4,
        "page_index": 1,
        "marker_image_type": "Figure",
        "marker_image_index": 3,
        "block_id": None,
        "group_id": None,
        "bbox": None,
        "polygon": None,
        "image_width": None,
        "image_height": None,
        "is_likely_figure": True,
        "diagnostic_flags": ["no_caption"],
        "alt_text": "Figure 2",
        "caption_text": None,
        "figure_label": "Figure 2",
        "figure_number": "2",
    }
    for key, value in expected.items():
        assert result[0][key] == value
    assert result[0]["figure_decision_source"] == "heuristic"
    assert result[0]["heuristic_is_likely_figure"] is True


def test_list_images_supports_legacy_manifest_strings(app_context):
    file_hash = "legacymanifest"
    images_dir = get_images_dir(file_hash)
    os.makedirs(images_dir, exist_ok=True)

    with open(os.path.join(images_dir, "_page_0_Picture_1.jpeg"), "wb") as f:
        f.write(b"\x89PNG")
    with open(os.path.join(images_dir, ".pdfx-images.json"), "w", encoding="utf-8") as f:
        f.write('{"images":["_page_0_Picture_1.jpeg"]}')

    result = list_images(file_hash)
    assert result[0]["filename"] == "_page_0_Picture_1.jpeg"
    assert result[0]["marker_image_type"] == "Picture"
    assert result[0]["figure_number"] is None


def test_list_images_honors_empty_manifest(app_context):
    file_hash = "emptymanifest"
    images_dir = get_images_dir(file_hash)
    os.makedirs(images_dir, exist_ok=True)

    with open(os.path.join(images_dir, "stale.png"), "wb") as f:
        f.write(b"\x89PNG")
    with open(os.path.join(images_dir, ".pdfx-images.json"), "w", encoding="utf-8") as f:
        f.write('{"images":[]}')

    assert list_images(file_hash) == []


def test_list_images_filters_non_images(app_context):
    """Non-image files (.txt, .md) should be excluded."""
    file_hash = "filtertest"
    images_dir = get_images_dir(file_hash)
    os.makedirs(images_dir, exist_ok=True)

    with open(os.path.join(images_dir, "figure.png"), "wb") as f:
        f.write(b"\x89PNG")
    with open(os.path.join(images_dir, "notes.txt"), "w") as f:
        f.write("not an image")
    with open(os.path.join(images_dir, "readme.md"), "w") as f:
        f.write("not an image")

    result = list_images(file_hash)
    assert len(result) == 1
    assert result[0]["filename"] == "figure.png"


def test_has_image_extraction_manifest(app_context):
    file_hash = "manifesttest"
    images_dir = get_images_dir(file_hash)
    os.makedirs(images_dir, exist_ok=True)

    assert has_image_extraction_manifest(file_hash) is False

    with open(os.path.join(images_dir, ".pdfx-images.json"), "w", encoding="utf-8") as f:
        f.write('{"images":[]}')

    assert has_image_extraction_manifest(file_hash) is True
