import json
import re
from unittest.mock import MagicMock, patch

from app.services.audit_logger import AuditLogger


class DummyConfig:
    AUDIT_S3_BUCKET = "test-bucket"
    AUDIT_S3_PREFIX = "pdfx/audit"
    AWS_ACCESS_KEY_ID = "test-key"
    AWS_SECRET_ACCESS_KEY = "test-secret"
    AWS_DEFAULT_REGION = "us-east-1"


@patch("app.services.audit_logger.boto3.client")
def test_flush_writes_ndjson(mock_boto_client):
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    audit = AuditLogger("process-123", DummyConfig)
    audit.log("extract_grobid", "started")
    audit.log("extract_grobid", "completed", duration_s=1.23)
    audit.flush()

    assert mock_s3.put_object.call_count == 1
    kwargs = mock_s3.put_object.call_args.kwargs

    assert kwargs["Bucket"] == "test-bucket"
    assert kwargs["ContentType"] == "application/x-ndjson"
    assert re.match(r"pdfx/audit/\d{4}/\d{2}/\d{2}/process-123\.ndjson", kwargs["Key"])

    lines = kwargs["Body"].decode("utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["stage"] == "extract_grobid"
    assert first["status"] == "started"
    assert "ts" in first

    assert second["stage"] == "extract_grobid"
    assert second["status"] == "completed"
    assert second["duration_s"] == 1.23


@patch("app.services.audit_logger.boto3.client")
def test_upload_artifact_uses_process_prefix(mock_boto_client):
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    audit = AuditLogger("process-456", DummyConfig)
    key = audit.upload_artifact("grobid.md", "hello")

    assert key is not None
    assert key.endswith("/process-456/grobid.md")

    kwargs = mock_s3.put_object.call_args.kwargs
    assert kwargs["Bucket"] == "test-bucket"
    assert kwargs["Key"] == key
    assert kwargs["Body"] == b"hello"


@patch("app.services.audit_logger.boto3.client")
def test_upload_artifact_supports_subdir(mock_boto_client):
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    audit = AuditLogger("process-654", DummyConfig)
    key = audit.upload_artifact("figure1.png", b"img-bytes", subdir="images/raw")

    assert key is not None
    assert key.endswith("/process-654/images/raw/figure1.png")

    kwargs = mock_s3.put_object.call_args.kwargs
    assert kwargs["Bucket"] == "test-bucket"
    assert kwargs["Key"] == key
    assert kwargs["Body"] == b"img-bytes"


@patch("app.services.audit_logger.boto3.client")
def test_context_manager_flushes_on_exit(mock_boto_client):
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    with AuditLogger("process-789", DummyConfig) as audit:
        audit.log("finalize", "succeeded")

    assert mock_s3.put_object.call_count == 1
