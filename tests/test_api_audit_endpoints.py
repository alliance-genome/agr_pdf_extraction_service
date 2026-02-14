import io
import tempfile
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import create_app
from app.models import Base, ExtractionRun, get_engine, get_session, reset_db_engine


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_db_engine()
    Base.metadata.create_all(bind=get_engine())

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
    app.config["MARKER_EXTRACT_IMAGES"] = True
    app.config["CONSENSUS_ENABLED"] = True
    app.config["CONSENSUS_NEAR_THRESHOLD"] = 0.92
    app.config["CONSENSUS_LEVENSHTEIN_THRESHOLD"] = 0.90
    app.config["CONSENSUS_CONFLICT_RATIO_FALLBACK"] = 0.4
    app.config["CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK"] = 0.5
    app.config["CONSENSUS_ALWAYS_ESCALATE_TABLES"] = True
    app.config["AUDIT_S3_BUCKET"] = "test-bucket"
    app.config["AUDIT_S3_PREFIX"] = "pdfx/audit"
    app.config["AWS_ACCESS_KEY_ID"] = "test-key"
    app.config["AWS_SECRET_ACCESS_KEY"] = "test-secret"
    app.config["AWS_DEFAULT_REGION"] = "us-east-1"

    with app.test_client() as test_client:
        yield test_client

    reset_db_engine()


@patch("celery_app.extract_pdf.apply_async")
def test_submit_extraction_returns_process_id_and_creates_db_row(mock_apply_async, client):
    mock_apply_async.return_value = SimpleNamespace(id="process-id-placeholder")

    response = client.post(
        "/api/v1/extract",
        data={
            "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf"),
            "methods": "grobid,docling",
            "reference_curie": "PMID:12345",
            "mod_abbreviation": "RGD",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    payload = response.get_json()

    assert "process_id" in payload
    assert "job_id" not in payload
    assert payload["reference_curie"] == "PMID:12345"
    assert payload["mod_abbreviation"] == "RGD"

    # process_id is used as the Celery task_id
    call_kwargs = mock_apply_async.call_args.kwargs
    assert call_kwargs["task_id"] == payload["process_id"]
    assert call_kwargs["kwargs"]["process_id"] == payload["process_id"]
    assert call_kwargs["kwargs"]["reference_curie"] == "PMID:12345"
    assert call_kwargs["kwargs"]["mod_abbreviation"] == "RGD"

    session = get_session()
    run = session.get(ExtractionRun, payload["process_id"])
    assert run is not None
    assert run.status == "queued"
    assert run.reference_curie == "PMID:12345"
    assert run.mod_abbreviation == "RGD"
    session.close()


@patch("celery_app.extract_pdf.apply_async")
def test_submit_extraction_returns_503_when_enqueue_fails(mock_apply_async, client):
    mock_apply_async.side_effect = ConnectionError("Redis unavailable")

    response = client.post(
        "/api/v1/extract",
        data={"file": (io.BytesIO(b"%PDF-1.4"), "test.pdf")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 503
    payload = response.get_json()
    assert "enqueue" in payload["error"].lower() or "redis" in payload["error"].lower()

    # No orphan DB row should exist
    session = get_session()
    assert session.query(ExtractionRun).count() == 0
    session.close()


def test_get_extraction_status_prefers_db(client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="succeeded",
        reference_curie="PMID:99999",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        log_s3_key="pdfx/audit/2026/02/11/test.ndjson",
        artifacts_json={"grobid": "pdfx/audit/2026/02/11/x/grobid.md"},
    ))
    session.commit()
    session.close()

    with patch("celery_app.celery.AsyncResult") as mock_async:
        response = client.get(f"/api/v1/extract/{process_id}")

    assert response.status_code == 200
    payload = response.get_json()

    assert payload["process_id"] == process_id
    assert payload["status"] == "complete"
    assert payload["reference_curie"] == "PMID:99999"
    assert payload["log_s3_key"] == "pdfx/audit/2026/02/11/test.ndjson"
    assert "grobid" in payload["artifacts_json"]
    assert mock_async.called is False


def test_get_extraction_status_falls_back_to_celery_when_db_row_missing(client):
    """If the DB row hasn't been created yet (e.g., worker hasn't started),
    the status endpoint falls back to checking Celery state."""
    mock_result = MagicMock()
    mock_result.state = "PENDING"
    unknown_id = str(uuid.uuid4())

    with patch("celery_app.celery.AsyncResult", return_value=mock_result):
        response = client.get(f"/api/v1/extract/{unknown_id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["process_id"] == unknown_id
    assert payload["status"] == "pending"


@patch("celery_app.extract_pdf.apply_async")
@patch("app.api._get_db_session", return_value=None)
def test_submit_extraction_works_when_db_unavailable(mock_db_session, mock_apply_async, client):
    """Core extraction should still enqueue even if the DB is completely unavailable."""
    mock_apply_async.return_value = SimpleNamespace(id="process-id-placeholder")

    response = client.post(
        "/api/v1/extract",
        data={"file": (io.BytesIO(b"%PDF-1.4"), "test.pdf")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert "process_id" in payload
    mock_apply_async.assert_called_once()


@patch("app.api.build_s3_client")
def test_get_log_url_returns_presigned_url(mock_build_s3_client, client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="succeeded",
        log_s3_key="pdfx/audit/2026/02/11/test-log.ndjson",
    ))
    session.commit()
    session.close()

    mock_s3 = MagicMock()
    mock_s3.generate_presigned_url.return_value = "https://example.com/presigned"
    mock_build_s3_client.return_value = mock_s3

    response = client.get(f"/api/v1/extract/{process_id}/logs")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["log_url"] == "https://example.com/presigned"
    assert payload["expires_in"] == 3600


def test_get_artifacts_returns_artifacts_json(client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="succeeded",
        artifacts_json={
            "grobid": "pdfx/audit/2026/02/11/x/grobid.md",
            "docling": "pdfx/audit/2026/02/11/x/docling.md",
        },
    ))
    session.commit()
    session.close()

    response = client.get(f"/api/v1/extract/{process_id}/artifacts")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["process_id"] == process_id
    assert "grobid" in payload["artifacts_json"]
    assert "docling" in payload["artifacts_json"]


@patch("app.api.build_s3_client")
def test_get_artifact_urls_returns_presigned_urls_for_nested_artifacts(mock_build_s3_client, client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="succeeded",
        artifacts_json={
            "grobid": "pdfx/audit/2026/02/11/x/grobid.md",
            "source_pdf": "pdfx/audit/2026/02/11/x/inputs/source.pdf",
            "images": [
                {
                    "filename": "fig1.png",
                    "s3_key": "pdfx/audit/2026/02/11/x/images/fig1.png",
                }
            ],
        },
    ))
    session.commit()
    session.close()

    mock_s3 = MagicMock()
    mock_s3.generate_presigned_url.side_effect = lambda *_args, **kwargs: (
        f"https://example.com/{kwargs['Params']['Key']}"
    )
    mock_build_s3_client.return_value = mock_s3

    response = client.get(f"/api/v1/extract/{process_id}/artifacts/urls")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["process_id"] == process_id
    assert payload["expires_in"] == 3600

    urls = payload["artifact_urls"]
    keys = [item["s3_key"] for item in urls]
    assert "pdfx/audit/2026/02/11/x/grobid.md" in keys
    assert "pdfx/audit/2026/02/11/x/inputs/source.pdf" in keys
    assert "pdfx/audit/2026/02/11/x/images/fig1.png" in keys
