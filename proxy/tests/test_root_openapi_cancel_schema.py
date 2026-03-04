"""Contract tests for root OpenAPI cancel schema."""

from pathlib import Path

import yaml


def _load_openapi() -> dict:
    root = Path(__file__).resolve().parents[2]
    spec_path = root / "app" / "openapi.yaml"
    return yaml.safe_load(spec_path.read_text(encoding="utf-8"))


def test_cancel_endpoint_contract_in_root_openapi():
    spec = _load_openapi()
    cancel_post = spec["paths"]["/extract/{process_id}/cancel"]["post"]
    assert set(cancel_post["responses"].keys()) == {"202", "409"}
    reason_schema = cancel_post["requestBody"]["content"]["application/json"]["schema"]["properties"]["reason"]
    assert reason_schema["type"] == "string"


def test_cancel_response_status_enum_is_precise_in_root_openapi():
    spec = _load_openapi()
    cancel_schema = spec["components"]["schemas"]["CancelExtractionResponse"]
    assert set(cancel_schema["properties"]["status"]["enum"]) == {"cancelled", "complete", "failed"}
