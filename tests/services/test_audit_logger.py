import json
import re
from unittest.mock import MagicMock, patch

from app.services.audit_logger import AuditLogger


class DummyConfig:
    AUDIT_S3_BUCKET = "test-bucket"
    AUDIT_S3_PREFIX = "pdfx/audit"
    AUDIT_S3_BUCKET_SSM_PARAM = ""
    AUDIT_FLUSH_ON_EVENT = False
    AWS_DEFAULT_REGION = "us-east-1"


class FlushOnEventConfig(DummyConfig):
    AUDIT_FLUSH_ON_EVENT = True


@patch("app.services.audit_logger.build_s3_client")
def test_flush_writes_ndjson(mock_build_s3):
    mock_s3 = MagicMock()
    mock_build_s3.return_value = mock_s3

    audit = AuditLogger("process-123", DummyConfig)
    audit.log("extract_grobid", "started")
    audit.log("extract_grobid", "completed", duration_s=1.23)
    audit.flush()

    assert mock_s3.put_object.call_count == 1
    kwargs = mock_s3.put_object.call_args.kwargs

    assert kwargs["Bucket"] == "test-bucket"
    assert kwargs["ContentType"] == "application/x-ndjson"
    assert re.match(r"pdfx/audit/\d{4}/\d{2}/\d{2}/process-123/attempt-0\.ndjson", kwargs["Key"])

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

    assert audit.timing_events() == [
        {
            "ts": first["ts"],
            "stage": "extract_grobid",
            "status": "started",
        },
        {
            "ts": second["ts"],
            "stage": "extract_grobid",
            "status": "completed",
            "duration_s": 1.23,
        },
    ]


@patch("app.services.audit_logger.build_s3_client")
def test_flush_on_event_updates_ndjson_after_each_event(mock_build_s3):
    mock_s3 = MagicMock()
    mock_build_s3.return_value = mock_s3

    audit = AuditLogger("process-flush-on-event", FlushOnEventConfig)
    audit.log("run", "queued")
    audit.log("run", "running")

    assert mock_s3.put_object.call_count == 2
    first_body = mock_s3.put_object.call_args_list[0].kwargs["Body"].decode("utf-8")
    second_body = mock_s3.put_object.call_args_list[1].kwargs["Body"].decode("utf-8")

    assert len(first_body.strip().splitlines()) == 1
    assert len(second_body.strip().splitlines()) == 2
    assert '"status":"running"' in second_body


@patch("app.services.audit_logger.build_s3_client")
def test_attempt_id_makes_log_key_unique_for_redelivery(mock_build_s3):
    mock_s3 = MagicMock()
    mock_build_s3.return_value = mock_s3

    first_attempt = AuditLogger("process-retry", DummyConfig, attempt_id="attempt-first")
    second_attempt = AuditLogger("process-retry", DummyConfig, attempt_id="attempt-second")

    assert first_attempt.get_log_s3_key().endswith("/process-retry/attempt-first.ndjson")
    assert second_attempt.get_log_s3_key().endswith("/process-retry/attempt-second.ndjson")
    assert first_attempt.get_log_s3_key() != second_attempt.get_log_s3_key()


@patch("app.services.audit_logger.build_s3_client")
def test_upload_artifact_uses_process_prefix(mock_build_s3):
    mock_s3 = MagicMock()
    mock_build_s3.return_value = mock_s3

    audit = AuditLogger("process-456", DummyConfig)
    key = audit.upload_artifact("grobid.md", "hello")

    assert key is not None
    assert key.endswith("/process-456/grobid.md")

    kwargs = mock_s3.put_object.call_args.kwargs
    assert kwargs["Bucket"] == "test-bucket"
    assert kwargs["Key"] == key
    assert kwargs["Body"] == b"hello"


@patch("app.services.audit_logger.build_s3_client")
def test_upload_artifact_supports_subdir(mock_build_s3):
    mock_s3 = MagicMock()
    mock_build_s3.return_value = mock_s3

    audit = AuditLogger("process-654", DummyConfig)
    key = audit.upload_artifact("figure1.png", b"img-bytes", subdir="images/raw")

    assert key is not None
    assert key.endswith("/process-654/images/raw/figure1.png")

    kwargs = mock_s3.put_object.call_args.kwargs
    assert kwargs["Bucket"] == "test-bucket"
    assert kwargs["Key"] == key
    assert kwargs["Body"] == b"img-bytes"


@patch("app.services.audit_logger.build_s3_client")
def test_upload_artifact_supports_tags(mock_build_s3):
    mock_s3 = MagicMock()
    mock_build_s3.return_value = mock_s3

    audit = AuditLogger("process-654", DummyConfig)
    key = audit.upload_artifact(
        "figure1.png",
        b"img-bytes",
        subdir="images",
        tags={
            "pdfx-artifact-type": "extracted-image",
            "pdfx-retention": "temporary",
        },
    )

    assert key is not None
    kwargs = mock_s3.put_object.call_args.kwargs
    assert kwargs["Tagging"] == "pdfx-artifact-type=extracted-image&pdfx-retention=temporary"


@patch("app.services.audit_logger.build_s3_client")
def test_context_manager_flushes_on_exit(mock_build_s3):
    mock_s3 = MagicMock()
    mock_build_s3.return_value = mock_s3

    with AuditLogger("process-789", DummyConfig) as audit:
        audit.log("finalize", "succeeded")

    assert mock_s3.put_object.call_count == 1
