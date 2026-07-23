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
        "/extract/{process_id}/cancel",
        "/extract/{process_id}/download/{method}",
        "/extract/{process_id}/images",
        "/extract/{process_id}/images/urls",
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
    assert "extract_images" in properties
    assert properties["extract_images"]["type"] == "boolean"
    assert "review_images" in properties
    assert properties["review_images"]["type"] == "boolean"


def test_openapi_extract_request_includes_clear_cache_scope(client):
    response = client.get("/openapi.yaml")
    assert response.status_code == 200

    spec = yaml.safe_load(response.data)
    extract_schema = (
        spec["paths"]["/extract"]["post"]["requestBody"]["content"]
        ["multipart/form-data"]["schema"]
    )
    properties = extract_schema["properties"]

    assert "clear_cache_scope" in properties
    assert properties["clear_cache_scope"]["type"] == "string"
    assert set(properties["clear_cache_scope"]["enum"]) == {
        "none", "merge", "extraction", "all"
    }


def test_openapi_extract_request_includes_extract_images(client):
    response = client.get("/openapi.yaml")
    assert response.status_code == 200

    spec = yaml.safe_load(response.data)
    extract_schema = (
        spec["paths"]["/extract"]["post"]["requestBody"]["content"]
        ["multipart/form-data"]["schema"]
    )
    properties = extract_schema["properties"]

    assert "extract_images" in properties
    assert properties["extract_images"]["type"] == "boolean"
    assert properties["extract_images"]["default"] is False
    assert "review_images" in properties
    assert properties["review_images"]["type"] == "boolean"
    assert properties["review_images"]["default"] is True


def test_openapi_image_schemas_include_figure_metadata(client):
    response = client.get("/openapi.yaml")
    assert response.status_code == 200

    spec = yaml.safe_load(response.data)
    image_metadata = spec["components"]["schemas"]["ImageMetadata"]["properties"]
    image_url_entry = spec["components"]["schemas"]["ImageUrlEntry"]["properties"]

    for properties in (image_metadata, image_url_entry):
        assert properties["page_index"]["nullable"] is True
        assert properties["marker_image_type"]["nullable"] is True
        assert properties["marker_image_index"]["nullable"] is True
        assert properties["block_id"]["nullable"] is True
        assert properties["group_id"]["nullable"] is True
        assert properties["bbox"]["nullable"] is True
        assert properties["polygon"]["nullable"] is True
        assert properties["image_width"]["nullable"] is True
        assert properties["image_height"]["nullable"] is True
        assert properties["is_likely_figure"]["nullable"] is True
        assert properties["diagnostic_flags"]["type"] == "array"
        assert properties["caption_text"]["nullable"] is True
        assert properties["nearby_text"]["nullable"] is True
        assert properties["figure_label"]["nullable"] is True
        assert properties["figure_number"]["nullable"] is True
        assert properties["figure_decision_source"]["nullable"] is True
        assert properties["image_review_classification"]["nullable"] is True
        assert properties["image_review_reason"]["nullable"] is True


def test_openapi_cancel_response_enum_is_precise(client):
    response = client.get("/openapi.yaml")
    assert response.status_code == 200

    spec = yaml.safe_load(response.data)
    cancel_operation = spec["paths"]["/extract/{process_id}/cancel"]["post"]
    assert set(cancel_operation["responses"].keys()) == {"202", "409"}
    assert "requestBody" in cancel_operation
    reason_schema = cancel_operation["requestBody"]["content"]["application/json"]["schema"]["properties"]["reason"]
    assert reason_schema["type"] == "string"

    cancel_schema = spec["components"]["schemas"]["CancelExtractionResponse"]
    enum_values = set(cancel_schema["properties"]["status"]["enum"])
    assert enum_values == {"cancelled", "complete", "failed"}


def test_config_endpoint_reports_only_release_approved_5_6_models(client):
    response = client.get("/api/v1/config")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["merge_contract_id"] == "pdfx-native-skeleton-selection"
    assert payload["resolved_runtime_models"] == {
        "source_selection": {"model": "gpt-5.6-terra", "reasoning_effort": "medium"},
        "hard_selection": {"model": "gpt-5.6-sol", "reasoning_effort": "high"},
        "image_text_review": {"model": "gpt-5.6-luna", "reasoning_effort": "medium"},
    }
    assert all(
        role["model"].startswith("gpt-5.6-")
        for role in payload["resolved_runtime_models"].values()
    )


def test_swagger_docs_page_available(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert b"swagger-ui" in response.data
    assert b"/openapi.yaml" in response.data
