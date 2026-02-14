import tempfile

import pytest
import yaml

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = tempfile.mkdtemp()
    app.config["CACHE_FOLDER"] = tempfile.mkdtemp()

    with app.test_client() as test_client:
        yield test_client


def test_openapi_yaml_available(client):
    response = client.get("/openapi.yaml")
    assert response.status_code == 200
    assert b"openapi: 3.0.3" in response.data
    assert b"/extract/{process_id}" in response.data


def test_openapi_includes_all_v1_paths(client):
    response = client.get("/openapi.yaml")
    assert response.status_code == 200

    spec = yaml.safe_load(response.data)
    expected_paths = {
        "/health",
        "/extractions",
        "/extract",
        "/extract/{process_id}",
        "/extract/{process_id}/download/{method}",
        "/extract/{process_id}/images",
        "/extract/{process_id}/images/{filename}",
        "/extract/{process_id}/logs",
        "/extract/{process_id}/artifacts",
        "/extract/{process_id}/artifacts/urls",
    }
    assert expected_paths.issubset(set(spec["paths"].keys()))


def test_openapi_status_schema_includes_celery_success_result(client):
    response = client.get("/openapi.yaml")
    assert response.status_code == 200

    spec = yaml.safe_load(response.data)
    status_schema = spec["components"]["schemas"]["ExtractionStatusResponse"]
    properties = status_schema["properties"]
    assert "result" in properties
    assert properties["result"]["type"] == "object"


def test_swagger_docs_page_available(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert b"swagger-ui" in response.data
    assert b"/openapi.yaml" in response.data
