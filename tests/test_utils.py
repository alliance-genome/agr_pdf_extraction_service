import os
import tempfile
import pytest
from flask import Flask

from app.utils import allowed_file, get_file_hash, get_cached_path, is_extraction_cached

@pytest.fixture
def app_context():
    app = Flask(__name__)
    app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
    app.config['CACHE_FOLDER'] = tempfile.mkdtemp()
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
    assert path.endswith("abc123_grobid.md")
    assert app_context.config['CACHE_FOLDER'] in path

def test_is_extraction_cached(app_context, tmp_path):
    file_hash = "abc123"
    method = "grobid"
    cache_file = os.path.join(app_context.config['CACHE_FOLDER'], f"{file_hash}_{method}.md")
    # Should not exist yet
    assert not is_extraction_cached(file_hash, method)
    # Create the file
    with open(cache_file, "w") as f:
        f.write("test")