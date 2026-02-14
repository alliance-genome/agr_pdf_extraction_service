import io
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
    app.config['CACHE_FOLDER'] = tempfile.mkdtemp()
    app.config['OPENAI_API_KEY'] = "dummy"
    app.config['LLM_MODEL'] = "dummy"
    app.config['EXTRACTION_CONFIG_VERSION'] = "1"
    app.config['GROBID_REQUEST_TIMEOUT'] = 120
    app.config['GROBID_INCLUDE_COORDINATES'] = False
    app.config['GROBID_INCLUDE_RAW_CITATIONS'] = False
    app.config['DOCLING_DEVICE'] = "cpu"
    app.config['MARKER_DEVICE'] = "cpu"
    app.config['MARKER_EXTRACT_IMAGES'] = True
    app.config['CONSENSUS_ENABLED'] = True
    app.config['CONSENSUS_NEAR_THRESHOLD'] = 0.92
    app.config['CONSENSUS_LEVENSHTEIN_THRESHOLD'] = 0.90
    app.config['CONSENSUS_CONFLICT_RATIO_FALLBACK'] = 0.4
    app.config['CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK'] = 0.5
    app.config['CONSENSUS_ALWAYS_ESCALATE_TABLES'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"<html" in response.data or b"<!DOCTYPE html" in response.data

def test_process_pdf_no_file(client):
    response = client.post('/process', data={})
    assert response.status_code == 400
    assert b'No file provided' in response.data

def test_process_pdf_empty_file(client):
    data = {'file': (io.BytesIO(b''), '')}
    response = client.post('/process', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert b'No file selected' in response.data

def test_process_pdf_invalid_filetype(client):
    data = {
        'file': (io.BytesIO(b'dummy'), 'test.txt'),
        'methods': ['grobid']
    }
    response = client.post('/process', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert b'Only PDF files are allowed' in response.data

def test_process_pdf_no_methods(client):
    data = {
        'file': (io.BytesIO(b'%PDF-1.4'), 'test.pdf')
    }
    response = client.post('/process', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert b'Please select at least one extraction method' in response.data

@patch('app.server.Grobid')
@patch('app.server.Docling')
@patch('app.server.Marker')
@patch('app.server.LLM')
@patch('app.server.get_file_hash', return_value='dummyhash')
@patch('app.server.get_cached_path', side_effect=lambda h, m: os.path.join(tempfile.gettempdir(), f"v1_{h}_{m}.md"))
@patch('app.server.is_extraction_cached', return_value=False)
@patch('app.server.list_images', return_value=[])
def test_process_pdf_grobid(
    mock_list_images, mock_is_cached, mock_get_cached_path, mock_get_file_hash,
    mock_llm, mock_marker, mock_docling, mock_grobid, client
):
    # Mock Grobid extract and output
    grobid_instance = MagicMock()
    grobid_instance.extract.side_effect = lambda pdf, out: open(out, 'w').write("grobid output")
    mock_grobid.return_value = grobid_instance

    data = {
        'file': (io.BytesIO(b'%PDF-1.4'), 'test.pdf'),
        'methods': ['grobid']
    }
    response = client.post('/process', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert "GROBID" in json_data["methods_used"]

@patch('app.server.merge_with_consensus', return_value=(None, {"fallback_triggered": True}, []))
@patch('app.server.Grobid')
@patch('app.server.Docling')
@patch('app.server.Marker')
@patch('app.server.LLM')
@patch('app.server.get_file_hash', return_value='dummyhash')
@patch('app.server.get_cached_path', side_effect=lambda h, m: os.path.join(tempfile.gettempdir(), f"v1_{h}_{m}.md"))
@patch('app.server.is_extraction_cached', return_value=False)
@patch('app.server.list_images', return_value=[])
@patch('app.server.rewrite_image_paths', side_effect=lambda md, fh: md)
def test_process_pdf_merge(
    mock_rewrite, mock_list_images,
    mock_is_cached, mock_get_cached_path, mock_get_file_hash,
    mock_llm, mock_marker, mock_docling, mock_grobid, mock_consensus, client
):
    # Mock all extract methods and LLM
    grobid_instance = MagicMock()
    grobid_instance.extract.side_effect = lambda pdf, out: open(out, 'w').write("grobid output")
    mock_grobid.return_value = grobid_instance

    docling_instance = MagicMock()
    docling_instance.extract.side_effect = lambda pdf, out: open(out, 'w').write("docling output")
    mock_docling.return_value = docling_instance

    marker_instance = MagicMock()
    marker_instance.extract.side_effect = lambda pdf, out: open(out, 'w').write("marker output")
    mock_marker.return_value = marker_instance

    llm_instance = MagicMock()
    llm_instance.extract.return_value = "merged output"
    mock_llm.return_value = llm_instance

    data = {
        'file': (io.BytesIO(b'%PDF-1.4'), 'test.pdf'),
        'methods': ['grobid', 'docling', 'marker'],
        'merge': 'on'
    }
    response = client.post('/process', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b'merged output' in response.data

def test_download_extraction_invalid_method(client):
    response = client.get('/download/dummyhash/invalid')
    assert response.status_code == 400
    assert b'Invalid method' in response.data

@patch('app.server.get_cached_path', return_value='/tmp/dummy_grobid.md')
def test_download_extraction_file_not_found(mock_get_cached_path, client):
    response = client.get('/download/dummyhash/grobid')
    assert response.status_code == 404
    assert b'File not found' in response.data

@patch('app.server.get_cached_path', return_value='/tmp/dummy_grobid.md')
@patch('os.path.exists', return_value=True)
@patch('app.server.send_file')
def test_download_extraction_success(mock_send_file, mock_exists, mock_get_cached_path, client):
    mock_send_file.return_value = "sent"
    response = client.get('/download/dummyhash/grobid')
    assert response.data == b"sent" or response.status_code == 200
