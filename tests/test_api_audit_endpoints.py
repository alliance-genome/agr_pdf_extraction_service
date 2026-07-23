import io
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import create_app
from app.models import Base, ExtractionRun, get_engine, get_session, reset_db_engine


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("AUDIT_S3_BUCKET", "test-bucket")
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
    app.config["CONSENSUS_NEAR_THRESHOLD"] = 0.92
    app.config["CONSENSUS_LEVENSHTEIN_THRESHOLD"] = 0.90
    app.config["CONSENSUS_CONFLICT_RATIO_FALLBACK"] = 0.4
    app.config["CONSENSUS_ALIGNMENT_CONFIDENCE_MIN"] = 0.5
    app.config["CONSENSUS_ALWAYS_ESCALATE_TABLES"] = True
    app.config["AUDIT_S3_BUCKET"] = "test-bucket"
    app.config["AUDIT_S3_PREFIX"] = "pdfx/audit"
    app.config["AUDIT_S3_BUCKET_SSM_PARAM"] = ""
    app.config["IMAGE_URL_TTL_SECONDS"] = 3600
    app.config["IMAGE_RETENTION_TTL_SECONDS"] = 604800
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
    assert call_kwargs["kwargs"]["extract_images"] is False
    assert call_kwargs["kwargs"]["review_images"] is False
    assert payload["extract_images"] is False
    assert payload["review_images"] is False

    session = get_session()
    run = session.get(ExtractionRun, payload["process_id"])
    assert run is not None
    assert run.status == "queued"
    assert run.reference_curie == "PMID:12345"
    assert run.mod_abbreviation == "RGD"
    assert run.extract_images is False
    assert run.review_images is False
    session.close()


@patch("celery_app.extract_pdf.apply_async")
def test_submit_extraction_honors_client_supplied_process_id(mock_apply_async, client):
    requested_process_id = str(uuid.uuid4())
    mock_apply_async.return_value = SimpleNamespace(id=requested_process_id)

    response = client.post(
        "/api/v1/extract",
        data={
            "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf"),
            "process_id": requested_process_id,
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["process_id"] == requested_process_id

    call_kwargs = mock_apply_async.call_args.kwargs
    assert call_kwargs["task_id"] == requested_process_id
    assert call_kwargs["kwargs"]["process_id"] == requested_process_id

    session = get_session()
    run = session.get(ExtractionRun, requested_process_id)
    assert run is not None
    assert run.status == "queued"
    session.close()


def test_submit_extraction_rejects_invalid_client_process_id(client):
    response = client.post(
        "/api/v1/extract",
        data={
            "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf"),
            "process_id": "../not-a-uuid",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "Invalid process_id" in response.get_json()["error"]


@patch("celery_app.extract_pdf.apply_async")
def test_submit_extraction_passes_extract_images_to_celery(mock_apply_async, client):
    mock_apply_async.return_value = SimpleNamespace(id="process-id-placeholder")

    response = client.post(
        "/api/v1/extract",
        data={
            "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf"),
            "extract_images": "true",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    payload = response.get_json()
    call_kwargs = mock_apply_async.call_args.kwargs
    assert call_kwargs["kwargs"]["extract_images"] is True
    assert call_kwargs["kwargs"]["review_images"] is True
    assert payload["extract_images"] is True
    assert payload["review_images"] is True

    session = get_session()
    run = session.get(ExtractionRun, payload["process_id"])
    assert run.extract_images is True
    assert run.review_images is True
    session.close()


@patch("celery_app.extract_pdf.apply_async")
def test_submit_extraction_can_disable_default_image_review(mock_apply_async, client):
    mock_apply_async.return_value = SimpleNamespace(id="process-id-placeholder")

    response = client.post(
        "/api/v1/extract",
        data={
            "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf"),
            "extract_images": "true",
            "review_images": "false",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    payload = response.get_json()
    call_kwargs = mock_apply_async.call_args.kwargs
    assert call_kwargs["kwargs"]["extract_images"] is True
    assert call_kwargs["kwargs"]["review_images"] is False
    assert payload["extract_images"] is True
    assert payload["review_images"] is False

    session = get_session()
    run = session.get(ExtractionRun, payload["process_id"])
    assert run.extract_images is True
    assert run.review_images is False
    session.close()


@patch("celery_app.extract_pdf.apply_async")
def test_submit_extraction_passes_clear_cache_to_celery(mock_apply_async, client):
    mock_apply_async.return_value = SimpleNamespace(id="process-id-placeholder")

    response = client.post(
        "/api/v1/extract",
        data={
            "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf"),
            "clear_cache": "true",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    call_kwargs = mock_apply_async.call_args.kwargs
    assert call_kwargs["kwargs"]["clear_cache"] is True
    assert call_kwargs["kwargs"]["clear_cache_scope"] == "all"


@patch("celery_app.extract_pdf.apply_async")
def test_submit_extraction_passes_clear_cache_scope_to_celery(mock_apply_async, client):
    mock_apply_async.return_value = SimpleNamespace(id="process-id-placeholder")

    response = client.post(
        "/api/v1/extract",
        data={
            "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf"),
            "clear_cache_scope": "merge",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    payload = response.get_json()
    call_kwargs = mock_apply_async.call_args.kwargs
    assert call_kwargs["kwargs"]["clear_cache"] is False
    assert call_kwargs["kwargs"]["clear_cache_scope"] == "merge"
    assert payload["clear_cache_scope"] == "merge"


@patch("celery_app.extract_pdf.apply_async")
def test_submit_extraction_rejects_invalid_clear_cache_scope(mock_apply_async, client):
    response = client.post(
        "/api/v1/extract",
        data={
            "file": (io.BytesIO(b"%PDF-1.4"), "test.pdf"),
            "clear_cache_scope": "banana",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert "clear_cache_scope" in payload["error"]
    assert mock_apply_async.called is False


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
        extract_images=True,
        reference_curie="PMID:99999",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        log_s3_key="pdfx/audit/2026/02/11/test.ndjson",
        artifacts_json={
            "grobid": "pdfx/audit/2026/02/11/x/grobid.md",
            "images": [
                {"filename": "fig1.png", "s3_key": "pdfx/audit/2026/02/11/x/images/fig1.png"},
                {"filename": "fig2.png", "s3_key": "pdfx/audit/2026/02/11/x/images/fig2.png"},
            ],
        },
        consensus_metrics_json={
            "total_blocks": 100,
            "agree_exact": 90,
            "degradation_metrics": {"quality_score": 0.995, "quality_grade": "A"},
        },
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
    assert payload["extract_images"] is True
    assert payload["log_s3_key"] == "pdfx/audit/2026/02/11/test.ndjson"
    assert "grobid" in payload["artifacts_json"]
    assert payload["available_extractors"] == ["grobid"]
    assert payload["image_count"] == 2
    assert payload["consensus_metrics_json"]["total_blocks"] == 100
    assert payload["consensus_metrics_json"]["degradation_metrics"]["quality_score"] == 0.995
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


def test_health_reports_busy_solo_worker_as_busy(client):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    ))
    session.commit()
    session.close()

    grobid_response = MagicMock()
    grobid_response.status_code = 200
    redis_client = MagicMock()
    redis_client.llen.return_value = 2
    redis_client.hlen.return_value = 1
    redis_client.zcard.return_value = 1

    with (
        patch("requests.get", return_value=grobid_response),
        patch("redis.from_url", return_value=redis_client),
        patch("celery_app.celery.control.inspect") as mock_inspect,
    ):
        inspector = MagicMock()
        inspector.ping.return_value = None
        mock_inspect.return_value = inspector

        response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "busy"
    assert payload["checks"]["workers"] == 0
    assert payload["checks"]["active_runs"] == 1
    assert payload["checks"]["fresh_active_runs"] == 1
    assert payload["checks"]["queued_runs"] == 0
    assert payload["checks"]["broker_queued"] == 2
    assert payload["checks"]["broker_unacked"] == 1
    assert payload["checks"]["worker_state"] == "busy"


def test_health_requires_marker_ready_file_when_configured(client, tmp_path):
    client.application.config["HEALTH_REQUIRE_MARKER_READY"] = True
    client.application.config["MARKER_READY_FILE"] = str(tmp_path / "marker_worker_ready.json")

    grobid_response = MagicMock()
    grobid_response.status_code = 200
    redis_client = MagicMock()
    redis_client.llen.return_value = 0
    redis_client.hlen.return_value = 0
    redis_client.zcard.return_value = 0

    with (
        patch("requests.get", return_value=grobid_response),
        patch("redis.from_url", return_value=redis_client),
        patch("celery_app.celery.control.inspect") as mock_inspect,
    ):
        inspector = MagicMock()
        inspector.ping.return_value = {"worker@example": {"ok": "pong"}}
        mock_inspect.return_value = inspector

        response = client.get("/api/v1/health")

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["status"] == "unhealthy"
    assert payload["checks"]["marker_models"] == "loading"
    assert payload["checks"]["marker_models_error"] == "ready_file_missing"


def test_health_accepts_marker_ready_file_when_configured(client, tmp_path):
    ready_file = tmp_path / "marker_worker_ready.json"
    ready_file.write_text('{"device": "cuda"}', encoding="utf-8")
    client.application.config["HEALTH_REQUIRE_MARKER_READY"] = True
    client.application.config["MARKER_READY_FILE"] = str(ready_file)

    grobid_response = MagicMock()
    grobid_response.status_code = 200
    redis_client = MagicMock()
    redis_client.llen.return_value = 0
    redis_client.hlen.return_value = 0
    redis_client.zcard.return_value = 0

    with (
        patch("requests.get", return_value=grobid_response),
        patch("redis.from_url", return_value=redis_client),
        patch("celery_app.celery.control.inspect") as mock_inspect,
    ):
        inspector = MagicMock()
        inspector.ping.return_value = {"worker@example": {"ok": "pong"}}
        mock_inspect.return_value = inspector

        response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["checks"]["marker_models"] == "ok"


def test_health_is_unhealthy_when_extraction_run_table_missing(client):
    Base.metadata.drop_all(bind=get_engine())

    grobid_response = MagicMock()
    grobid_response.status_code = 200
    redis_client = MagicMock()
    redis_client.llen.return_value = 0
    redis_client.hlen.return_value = 0
    redis_client.zcard.return_value = 0

    with (
        patch("requests.get", return_value=grobid_response),
        patch("redis.from_url", return_value=redis_client),
        patch("celery_app.celery.control.inspect") as mock_inspect,
    ):
        inspector = MagicMock()
        inspector.ping.return_value = {"worker@example": {"ok": "pong"}}
        mock_inspect.return_value = inspector

        response = client.get("/api/v1/health")

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["status"] == "unhealthy"
    assert payload["checks"]["database"] == "unavailable"
    assert payload["checks"]["active_runs"] == "unknown"
    assert payload["checks"]["fresh_active_runs"] == "unknown"
    assert payload["checks"]["queued_runs"] == "unknown"


def test_health_stays_unhealthy_for_stale_running_row_without_unacked_task(client):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    ))
    session.commit()
    session.close()

    grobid_response = MagicMock()
    grobid_response.status_code = 200
    redis_client = MagicMock()
    redis_client.llen.return_value = 0
    redis_client.hlen.return_value = 0
    redis_client.zcard.return_value = 0

    with (
        patch("requests.get", return_value=grobid_response),
        patch("redis.from_url", return_value=redis_client),
        patch("celery_app.celery.control.inspect") as mock_inspect,
    ):
        inspector = MagicMock()
        inspector.ping.return_value = None
        mock_inspect.return_value = inspector

        response = client.get("/api/v1/health")

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["status"] == "unhealthy"
    assert payload["checks"]["active_runs"] == 1
    assert payload["checks"]["fresh_active_runs"] == 1
    assert payload["checks"]["broker_unacked"] == 0
    assert "worker_state" not in payload["checks"]


def test_health_stays_unhealthy_for_old_running_row_with_unacked_task(client):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="running",
        started_at=datetime.now(timezone.utc) - timedelta(seconds=21601),
    ))
    session.commit()
    session.close()

    grobid_response = MagicMock()
    grobid_response.status_code = 200
    redis_client = MagicMock()
    redis_client.llen.return_value = 0
    redis_client.hlen.return_value = 1
    redis_client.zcard.return_value = 1

    with (
        patch("requests.get", return_value=grobid_response),
        patch("redis.from_url", return_value=redis_client),
        patch("celery_app.celery.control.inspect") as mock_inspect,
    ):
        inspector = MagicMock()
        inspector.ping.return_value = None
        mock_inspect.return_value = inspector

        response = client.get("/api/v1/health")

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["status"] == "unhealthy"
    assert payload["checks"]["active_runs"] == 1
    assert payload["checks"]["fresh_active_runs"] == 0
    assert payload["checks"]["broker_unacked"] == 1
    assert "worker_state" not in payload["checks"]


def test_health_stays_unhealthy_when_redis_unavailable_with_active_run(client):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    ))
    session.commit()
    session.close()

    grobid_response = MagicMock()
    grobid_response.status_code = 200

    with (
        patch("requests.get", return_value=grobid_response),
        patch("redis.from_url", side_effect=RuntimeError("redis down")),
        patch("celery_app.celery.control.inspect") as mock_inspect,
    ):
        inspector = MagicMock()
        inspector.ping.return_value = None
        mock_inspect.return_value = inspector

        response = client.get("/api/v1/health")

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["status"] == "unhealthy"
    assert payload["checks"]["redis"] == "unavailable"
    assert payload["checks"]["active_runs"] == 1
    assert "worker_state" not in payload["checks"]


def test_get_extraction_status_uses_celery_success_when_db_row_is_stale_queued(client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="queued",
        reference_curie="PMID:11111",
    ))
    session.commit()
    session.close()

    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.result = {
        "reference_curie": "PMID:11111",
        "started_at": "2026-02-25T12:00:00Z",
        "ended_at": "2026-02-25T12:10:00Z",
        "log_s3_key": "pdfx/audit/2026/02/25/test.ndjson",
        "artifacts_json": {
            "merged": "pdfx/audit/2026/02/25/x/merged.md",
            "images": [
                {"filename": "fig1.png", "s3_key": "pdfx/audit/2026/02/25/x/images/fig1.png"},
            ],
        },
        "extract_images": True,
        "llm_usage_json": {"total_tokens": 123},
        "llm_cost_usd": 0.42,
        "available_extractors": ["docling"],
    }

    with patch("celery_app.celery.AsyncResult", return_value=mock_result):
        response = client.get(f"/api/v1/extract/{process_id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["process_id"] == process_id
    assert payload["status"] == "complete"
    assert payload["started_at"] == "2026-02-25T12:00:00Z"
    assert payload["ended_at"] == "2026-02-25T12:10:00Z"
    assert payload["log_s3_key"] == "pdfx/audit/2026/02/25/test.ndjson"
    assert payload["artifacts_json"]["merged"].endswith("merged.md")
    assert payload["extract_images"] is True
    assert payload["image_count"] == 1
    assert payload["llm_usage_json"]["total_tokens"] == 123
    assert payload["llm_cost_usd"] == 0.42
    assert payload["available_extractors"] == ["docling"]


def test_get_extraction_status_returns_failed_job_as_poll_result(client):
    process_id = str(uuid.uuid4())
    long_error = "cuda exploded " + ("x" * 5000)

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="failed",
        error_code="ConversionError",
        error_message=long_error,
    ))
    session.commit()
    session.close()

    response = client.get(f"/api/v1/extract/{process_id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["process_id"] == process_id
    assert payload["status"] == "failed"
    assert payload["error_code"] == "ConversionError"
    assert payload["error"].startswith("cuda exploded")
    assert "[truncated " in payload["error"]
    assert len(payload["error"]) < len(long_error)


def test_get_extraction_status_maps_celery_failure_to_failed_poll_result(client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    ))
    session.commit()
    session.close()

    mock_result = MagicMock()
    mock_result.state = "FAILURE"
    mock_result.info = RuntimeError("docling failed " + ("y" * 5000))

    with patch("celery_app.celery.AsyncResult", return_value=mock_result):
        response = client.get(f"/api/v1/extract/{process_id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["process_id"] == process_id
    assert payload["status"] == "failed"
    assert payload["error"].startswith("docling failed")
    assert "[truncated " in payload["error"]


def test_get_extraction_status_fallback_maps_celery_failure_to_failed_poll_result(client):
    process_id = str(uuid.uuid4())

    mock_result = MagicMock()
    mock_result.state = "FAILURE"
    mock_result.info = RuntimeError("worker failed")

    with patch("celery_app.celery.AsyncResult", return_value=mock_result):
        response = client.get(f"/api/v1/extract/{process_id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["process_id"] == process_id
    assert payload["status"] == "failed"
    assert payload["error"] == "worker failed"


@patch("celery_app.celery.control.revoke")
def test_cancel_extraction_marks_running_job_cancelled(mock_revoke, client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    ))
    session.commit()
    session.close()

    response = client.post(
        f"/api/v1/extract/{process_id}/cancel",
        json={"reason": "User cancelled"},
    )
    assert response.status_code == 202
    payload = response.get_json()
    assert payload["process_id"] == process_id
    assert payload["status"] == "cancelled"

    mock_revoke.assert_called_once()

    session = get_session()
    run = session.get(ExtractionRun, process_id)
    assert run is not None
    assert run.status == "cancelled"
    assert run.error_code == "cancelled"
    assert run.error_message == "User cancelled"
    assert run.ended_at is not None
    session.close()


@patch("celery_app.celery.control.revoke")
def test_cancel_extraction_returns_409_for_terminal_job(mock_revoke, client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="succeeded",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
    ))
    session.commit()
    session.close()

    response = client.post(f"/api/v1/extract/{process_id}/cancel")
    assert response.status_code == 409
    payload = response.get_json()
    assert payload["status"] == "complete"
    assert "terminal" in payload["message"].lower()
    mock_revoke.assert_not_called()


def test_get_extraction_status_maps_revoked_to_cancelled(client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    ))
    session.commit()
    session.close()

    mock_result = MagicMock()
    mock_result.state = "REVOKED"
    mock_result.info = None

    with patch("celery_app.celery.AsyncResult", return_value=mock_result):
        response = client.get(f"/api/v1/extract/{process_id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["process_id"] == process_id
    assert payload["status"] == "cancelled"


def test_merged_download_uses_verified_bundle_bytes(client, tmp_path):
    process_id = str(uuid.uuid4())
    source_paths = {}
    for source in ("grobid", "docling", "marker"):
        path = tmp_path / f"{source}.md"
        path.write_text(f"# Title\n\n{source} output.", encoding="utf-8")
        source_paths[source] = str(path)
    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.result = {
        "merge_contract_id": "pdfx-native-skeleton-selection",
        "merged_cache_path": str(tmp_path / "merged.md"),
        "merge_metrics_path": str(tmp_path / "metrics.json"),
        "merge_audit_path": str(tmp_path / "audit.json"),
        "available_extractors": ["grobid", "docling", "marker"],
        "native_structure_receipt_digests": {},
        "document_skeleton_candidate_ids": {
            "grobid": "a" * 64,
            "docling": "b" * 64,
            "marker": "c" * 64,
        },
        "document_skeleton_candidate_projection_ids": {
            "grobid": "d" * 64,
            "docling": "e" * 64,
            "marker": "f" * 64,
        },
        "download_paths": source_paths,
        "file_hash": "merged-hash",
    }

    with (
        patch("celery_app.celery.AsyncResult", return_value=mock_result),
        patch(
            "app.api.load_merge_bundle",
            return_value=("# Title\n\nVerified merge.", {}, []),
        ) as mock_load,
    ):
        response = client.get(f"/api/v1/extract/{process_id}/download/merged")

    assert response.status_code == 200
    assert response.data == b"# Title\n\nVerified merge."
    mock_load.assert_called_once()


def test_merge_download_uses_durable_artifact_when_local_metadata_is_incomplete(
    client, tmp_path
):
    process_id = str(uuid.uuid4())
    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.result = {
        "merged_cache_path": str(tmp_path / "merged.md"),
        "download_paths": {},
        "artifacts_json": {
            "merged": "pdfx/audit/2026/07/23/process/merged.md",
        },
        "file_hash": "merge-hash",
    }

    with (
        patch("celery_app.celery.AsyncResult", return_value=mock_result),
        patch(
            "app.api._s3_redirect_for_artifact",
            return_value=("durable merged output", 200),
        ) as mock_redirect,
    ):
        response = client.get(f"/api/v1/extract/{process_id}/download/merged")

    assert response.status_code == 200
    assert response.data == b"durable merged output"
    mock_redirect.assert_called_once_with(
        "pdfx/audit/2026/07/23/process/merged.md"
    )


def test_completed_merge_without_local_or_durable_artifact_is_internal_error(
    client, tmp_path
):
    process_id = str(uuid.uuid4())
    source = tmp_path / "grobid.md"
    source.write_text("# Title\n\nSource.", encoding="utf-8")
    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.result = {
        "merge_contract_id": "pdfx-native-skeleton-selection",
        "merged_cache_path": str(tmp_path / "merged.md"),
        "merge_metrics_path": str(tmp_path / "metrics.json"),
        "merge_audit_path": str(tmp_path / "audit.json"),
        "native_structure_receipt_digests": {},
        "document_skeleton_candidate_ids": {"grobid": "a" * 64},
        "document_skeleton_candidate_projection_ids": {"grobid": "b" * 64},
        "download_paths": {"grobid": str(source)},
        "file_hash": "merge-hash",
    }

    with patch("celery_app.celery.AsyncResult", return_value=mock_result):
        response = client.get(f"/api/v1/extract/{process_id}/download/merged")

    assert response.status_code == 500
    assert response.get_json()["error"] == (
        "Completed merge artifact is internally unavailable"
    )


def test_completed_celery_merge_does_not_use_stale_nonterminal_db_status(
    client, tmp_path
):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    ))
    session.commit()
    session.close()

    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.result = {
        "merged_cache_path": str(tmp_path / "merged.md"),
        "download_paths": {},
        "artifacts_json": {},
        "file_hash": "merge-hash",
    }

    with patch("celery_app.celery.AsyncResult", return_value=mock_result):
        response = client.get(f"/api/v1/extract/{process_id}/download/merged")

    assert response.status_code == 500
    assert response.get_json()["error"] == (
        "Completed merge artifact is internally unavailable"
    )


def test_completed_celery_merge_uses_durable_artifact_from_stale_db_row(
    client, tmp_path
):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="running",
        started_at=datetime.now(timezone.utc),
        artifacts_json={
            "merged": "pdfx/audit/2026/07/23/process/merged.md",
        },
    ))
    session.commit()
    session.close()

    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.result = {
        "merged_cache_path": str(tmp_path / "merged.md"),
        "download_paths": {},
        "artifacts_json": {},
        "file_hash": "merge-hash",
    }

    with (
        patch("celery_app.celery.AsyncResult", return_value=mock_result),
        patch(
            "app.api._s3_redirect_for_artifact",
            return_value=("durable merged output", 200),
        ) as mock_redirect,
    ):
        response = client.get(f"/api/v1/extract/{process_id}/download/merged")

    assert response.status_code == 200
    assert response.data == b"durable merged output"
    mock_redirect.assert_called_once_with(
        "pdfx/audit/2026/07/23/process/merged.md"
    )


def test_completed_nonmerge_job_has_no_merged_artifact(client):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="succeeded",
        consensus_metrics_json=None,
    ))
    session.commit()
    session.close()

    mock_result = MagicMock()
    mock_result.state = "PENDING"
    with patch("celery_app.celery.AsyncResult", return_value=mock_result):
        response = client.get(f"/api/v1/extract/{process_id}/download/merged")

    assert response.status_code == 404
    assert response.get_json()["error"] == (
        "Output file not found for method: merged"
    )


def test_completed_db_merge_without_durable_artifact_is_internal_error(client):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="succeeded",
        consensus_metrics_json={},
    ))
    session.commit()
    session.close()

    mock_result = MagicMock()
    mock_result.state = "PENDING"
    with patch("celery_app.celery.AsyncResult", return_value=mock_result):
        response = client.get(f"/api/v1/extract/{process_id}/download/merged")

    assert response.status_code == 500
    assert response.get_json()["error"] == (
        "Completed merge artifact is internally unavailable"
    )


def test_merged_download_uses_durable_db_artifact_after_celery_result_expires(
    client
):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="succeeded",
        artifacts_json={
            "merged": "pdfx/audit/2026/07/23/process/merged.md",
        },
    ))
    session.commit()
    session.close()

    mock_result = MagicMock()
    mock_result.state = "PENDING"
    with (
        patch("celery_app.celery.AsyncResult", return_value=mock_result),
        patch(
            "app.api._s3_redirect_for_artifact",
            return_value=("durable merged output", 200),
        ) as mock_redirect,
    ):
        response = client.get(f"/api/v1/extract/{process_id}/download/merged")

    assert response.status_code == 200
    assert response.data == b"durable merged output"
    mock_redirect.assert_called_once_with(
        "pdfx/audit/2026/07/23/process/merged.md"
    )


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


@patch("app.api.build_s3_client")
def test_get_image_urls_returns_presigned_manifest(mock_build_s3_client, client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="succeeded",
        artifacts_json={
            "images": [
                {
                    "filename": "fig1.png",
                    "s3_key": "pdfx/audit/2026/02/11/x/images/fig1.png",
                    "size_bytes": 123,
                    "page_index": 1,
                    "marker_image_type": "Figure",
                    "marker_image_index": 3,
                    "block_id": "/page/1/Figure/3",
                    "group_id": "/page/1/FigureGroup/7",
                    "bbox": [10.0, 20.0, 200.0, 300.0],
                    "caption_text": "Figure 2. Test caption.",
                    "nearby_text": "Figure 2. Test caption.",
                    "figure_label": "Figure 2",
                    "figure_number": "2",
                    "figure_decision_source": "llm_text",
                    "image_reviewed": True,
                    "image_review_method": "llm_text",
                    "image_review_classification": "scientific_figure",
                    "image_review_is_scientific_figure": True,
                    "image_review_confidence": 0.98,
                    "image_review_reason": "Caption explicitly identifies a figure.",
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

    response = client.get(f"/api/v1/extract/{process_id}/images/urls")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["process_id"] == process_id
    assert payload["manifest_ttl_seconds"] == 3600
    assert payload["image_retention_ttl_seconds"] == 604800
    assert payload["images"][0]["filename"] == "fig1.png"
    assert payload["images"][0]["url"].endswith("/pdfx/audit/2026/02/11/x/images/fig1.png")
    assert payload["images"][0]["expires_at"].endswith("Z")
    assert payload["images"][0]["page_index"] == 1
    assert payload["images"][0]["marker_image_type"] == "Figure"
    assert payload["images"][0]["marker_image_index"] == 3
    assert payload["images"][0]["block_id"] == "/page/1/Figure/3"
    assert payload["images"][0]["group_id"] == "/page/1/FigureGroup/7"
    assert payload["images"][0]["bbox"] == [10.0, 20.0, 200.0, 300.0]
    assert payload["images"][0]["caption_text"] == "Figure 2. Test caption."
    assert payload["images"][0]["nearby_text"] == "Figure 2. Test caption."
    assert payload["images"][0]["figure_label"] == "Figure 2"
    assert payload["images"][0]["figure_number"] == "2"
    assert payload["images"][0]["figure_decision_source"] == "llm_text"
    assert payload["images"][0]["image_reviewed"] is True
    assert payload["images"][0]["image_review_classification"] == "scientific_figure"
    assert payload["images"][0]["image_review_is_scientific_figure"] is True


@patch("app.api.build_s3_client", return_value=None)
def test_get_image_urls_returns_503_when_s3_unavailable(_mock_build_s3_client, client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="succeeded",
        artifacts_json={
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

    response = client.get(f"/api/v1/extract/{process_id}/images/urls")
    assert response.status_code == 503


@patch("app.api.build_s3_client")
def test_get_image_urls_records_per_image_presign_failure(mock_build_s3_client, client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="succeeded",
        artifacts_json={
            "images": [
                {
                    "filename": "fig1.png",
                    "s3_key": "pdfx/audit/2026/02/11/x/images/fig1.png",
                    "size_bytes": 123,
                }
            ],
        },
    ))
    session.commit()
    session.close()

    mock_s3 = MagicMock()
    mock_s3.generate_presigned_url.side_effect = RuntimeError("boom")
    mock_build_s3_client.return_value = mock_s3

    response = client.get(f"/api/v1/extract/{process_id}/images/urls")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["images"][0]["filename"] == "fig1.png"
    assert payload["images"][0]["error"] == "Unable to generate URL"
    assert "url" not in payload["images"][0]


def test_get_image_urls_404s_for_unknown_process(client):
    response = client.get(f"/api/v1/extract/{uuid.uuid4()}/images/urls")
    assert response.status_code == 404


def test_get_image_urls_409s_for_incomplete_job(client):
    process_id = str(uuid.uuid4())

    session = get_session()
    session.add(ExtractionRun(
        process_id=process_id,
        status="running",
    ))
    session.commit()
    session.close()

    response = client.get(f"/api/v1/extract/{process_id}/images/urls")
    assert response.status_code == 409
