import os
import tempfile
import pytest
from flask import Flask

from app.utils import (
    allowed_file, get_file_hash, get_cached_path, is_extraction_cached,
    get_images_dir, list_images, rewrite_image_paths,
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


def test_rewrite_image_paths_rewrites_existing(app_context):
    """Image references to existing files should be rewritten."""
    file_hash = "rewritetest"
    images_dir = get_images_dir(file_hash)
    os.makedirs(images_dir, exist_ok=True)

    with open(os.path.join(images_dir, "fig1.png"), "wb") as f:
        f.write(b"\x89PNG")

    md = "Some text.\n\n![Figure 1](fig1.png)\n\nMore text."
    result = rewrite_image_paths(md, file_hash)
    assert f"![Figure 1](/download/{file_hash}/images/fig1.png)" in result
    assert "More text." in result


def test_rewrite_image_paths_leaves_missing(app_context):
    """Image references to non-existing files should be left unchanged."""
    file_hash = "missingtest"
    md = "![Figure 1](nonexistent.png)"
    result = rewrite_image_paths(md, file_hash)
    assert result == md


def test_rewrite_image_paths_no_images(app_context):
    """Markdown with no image references should be unchanged."""
    file_hash = "noimgtest"
    md = "# Title\n\nJust text, no images."
    result = rewrite_image_paths(md, file_hash)
    assert result == md


def test_rewrite_image_paths_strips_subdirectory(app_context):
    """Image references with subdirectory paths should be rewritten using basename only."""
    file_hash = "subdirtest"
    images_dir = get_images_dir(file_hash)
    os.makedirs(images_dir, exist_ok=True)

    with open(os.path.join(images_dir, "fig1.png"), "wb") as f:
        f.write(b"\x89PNG")

    md = "![Alt](images/some_dir/fig1.png)"
    result = rewrite_image_paths(md, file_hash)
    assert f"![Alt](/download/{file_hash}/images/fig1.png)" in result
