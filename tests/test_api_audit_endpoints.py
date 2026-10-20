import io
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
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


@pytest.fixture(autouse=True)
def durable_store(monkeypatch):
    from app.pipeline_storage import digest
    from pathlib import Path
    store = MagicMock()
    store.put_file.side_effect = lambda path, prefix: {'key':prefix+'/source', 'sha256':digest(path), 'size':Path(path).stat().st_size}
    monkeypatch.setattr('app.pipeline_storage.PipelineStore', lambda: store)
    monkeypatch.setattr('app.pipeline_health.fresh', lambda role: True)
    return store


def submit(client, process_id=None, **options):
    data={'file':(io.BytesIO(b'%PDF-1.4'), 'test.pdf'), **options}
    if process_id: data['process_id']=process_id
    return client.post('/api/v1/extract', data=data, content_type='multipart/form-data')


def test_submission_commits_durable_source_and_options_without_broker(client):
    from app.pipeline import snapshot
    with patch('celery_app.celery.send_task', side_effect=ConnectionError('Redis unavailable')) as publish:
        response=submit(client, reference_curie='PMID:12345', mod_abbreviation='RGD', methods='grobid,docling')
    assert response.status_code==202
    pid=response.get_json()['process_id']
    job=snapshot(pid)
    assert job['state']=='ready'
    assert job['source']['size']==8
    assert job['options']['methods']==['grobid','docling']
    assert job['options']['reference_curie']=='PMID:12345'
    assert job['options']['mod_abbreviation']=='RGD'
    assert job['options']['extract_images'] is False
    assert job['options']['review_images'] is False
    publish.assert_not_called()
    with get_session() as db: assert db.get(ExtractionRun,pid).status=='queued'


def test_duplicate_submission_reuses_durable_identity(client,durable_store):
    from app.pipeline import snapshot
    pid=str(uuid.uuid4())
    assert submit(client,pid).status_code==202
    original=snapshot(pid)
    response=submit(client,pid)
    assert response.status_code==202
    assert response.get_json()['process_id']==pid
    assert snapshot(pid)['source']==original['source']
    assert durable_store.put_file.call_count==1


def test_concurrent_duplicate_submissions_have_one_durable_source(tmp_path,monkeypatch,durable_store):
    monkeypatch.setenv('DATABASE_URL',f'sqlite:///{tmp_path / "submissions.db"}')
    monkeypatch.setenv('AUDIT_S3_BUCKET','test-bucket')
    reset_db_engine();Base.metadata.create_all(get_engine())
    app=create_app();app.config.update(TESTING=True,UPLOAD_FOLDER=str(tmp_path/'uploads'))
    pid=str(uuid.uuid4())
    def post(_):
        with app.test_client() as client:return submit(client,pid).status_code
    with ThreadPoolExecutor(2) as pool: statuses=list(pool.map(post,range(2)))
    assert statuses==[202,202]
    assert durable_store.put_file.call_count==1
    reset_db_engine()


def test_terminal_duplicate_returns_existing_failure_without_upload(client,durable_store):
    pid=str(uuid.uuid4())
    with get_session() as db:
        db.add(ExtractionRun(process_id=pid,status='failed',error_message='existing failure'))
        db.commit()
    response=submit(client,pid)
    assert response.status_code==200
    assert response.get_json()['error']=='existing failure'
    durable_store.put_file.assert_not_called()


def test_storage_failure_can_be_retried_with_same_identity(client,durable_store):
    from app.pipeline import snapshot
    pid=str(uuid.uuid4())
    successful_upload=durable_store.put_file.side_effect
    durable_store.put_file.side_effect=ConnectionError('S3 unavailable')
    assert submit(client,pid).status_code==503
    assert snapshot(pid) is None
    durable_store.put_file.side_effect=successful_upload
    assert submit(client,pid).status_code==202
    assert snapshot(pid)['state']=='ready'


def test_stale_unaccepted_submission_can_be_recovered(client,monkeypatch):
    from app.pipeline import snapshot
    pid=str(uuid.uuid4())
    monkeypatch.setattr('app.api.Config.SUBMISSION_CLAIM_STALE_SECONDS',30)
    with get_session() as db:
        db.add(ExtractionRun(process_id=pid,status='submitting',started_at=datetime.now(timezone.utc)-timedelta(seconds=31)))
        db.commit()
    assert submit(client,pid).status_code==202
    assert snapshot(pid)['state']=='ready'


@pytest.mark.parametrize('options, expected', [
    ({'extract_images':'true'}, (True,True)),
    ({'extract_images':'true','review_images':'false'}, (True,False)),
    ({}, (False,False)),
])
def test_image_flags_are_durable(client, options, expected):
    from app.pipeline import snapshot
    response=submit(client,**options)
    assert response.status_code==202
    opts=snapshot(response.get_json()['process_id'])['options']
    assert (opts['extract_images'],opts['review_images'])==expected


@pytest.mark.parametrize('options, expected', [({'clear_cache':'true'},'all'),({'clear_cache_scope':'merge'},'merge')])
def test_cache_controls_are_accepted_with_fresh_isolated_workspaces(client, options, expected):
    response=submit(client,**options)
    assert response.status_code==202
    assert response.get_json()['clear_cache_scope']==expected


@pytest.mark.parametrize('options', [{'process_id':'../invalid'}, {'clear_cache_scope':'banana'}, {'methods':'unknown'}])
def test_invalid_submission_does_not_upload(client,durable_store,options):
    assert submit(client,**options).status_code==400
    durable_store.put_file.assert_not_called()


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


def test_unknown_process_id_is_not_mistaken_for_pending_work(client):
    unknown_id=str(uuid.uuid4())
    with patch("celery_app.celery.AsyncResult") as result:
        response=client.get(f"/api/v1/extract/{unknown_id}")
    assert response.status_code==404
    assert response.get_json()["error"]=="Extraction not found"
    result.assert_not_called()


def test_health_reports_busy_solo_worker_from_process_heartbeats(client):
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
    assert payload["status"] == "ok"
    assert payload["checks"]["workers"] == 2
    assert payload["checks"]["active_runs"] == 1
    assert payload["checks"]["fresh_active_runs"] == 1
    assert payload["checks"]["queued_runs"] == 0
    assert payload["checks"]["broker_queued"] == 4
    assert payload["checks"]["broker_unacked"] == 1
    assert payload["checks"]["finalization_worker"] == "ok"
    assert payload["checks"]["pipeline_dispatcher"] == "ok"


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


def test_health_stays_unhealthy_for_stale_running_row_without_unacked_task(client, monkeypatch):
    monkeypatch.setattr("app.pipeline_health.fresh", lambda role: False)
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


def test_health_stays_unhealthy_for_old_running_row_with_unacked_task(client, monkeypatch):
    monkeypatch.setattr("app.pipeline_health.fresh", lambda role: False)
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


def test_get_extraction_status_does_not_complete_queued_run_from_celery_success(client):
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
    assert payload["status"] == "pending"
    assert payload["started_at"] is None
    assert payload["artifacts_json"] is None
    assert payload["available_extractors"] == []


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


def test_get_extraction_status_does_not_fail_active_db_run_from_celery_failure(client):
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
    assert payload["status"] == "started"
    assert "error" not in payload


def test_status_database_outage_is_not_replaced_by_celery_truth(client):
    with patch("app.api._get_run_by_process_id",return_value=None), patch("app.api._check_db_ready",return_value=("unavailable","database down")), patch("celery_app.celery.AsyncResult") as result:
        response=client.get(f"/api/v1/extract/{uuid.uuid4()}")
    assert response.status_code==503
    result.assert_not_called()


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

    mock_revoke.assert_not_called()

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


@patch("celery_app.celery.control.revoke")
@patch("app.api._mark_run_cancelled", return_value=False)
@patch("app.api._get_run_by_process_id")
def test_cancel_race_returns_first_committed_terminal_state(mock_get_run, _mock_cancel, mock_revoke, client):
    process_id = str(uuid.uuid4())
    mock_get_run.side_effect = [
        {"process_id": process_id, "status": "running"},
        {"process_id": process_id, "status": "succeeded"},
    ]

    response = client.post(f"/api/v1/extract/{process_id}/cancel")

    assert response.status_code == 409
    assert response.get_json()["status"] == "complete"
    mock_revoke.assert_not_called()


@patch("celery_app.celery.control.revoke")
@patch("app.api._mark_run_cancelled", return_value=False)
@patch("app.api._get_run_by_process_id")
def test_cancel_write_failure_remains_nonterminal(mock_get_run, _mock_cancel, mock_revoke, client):
    process_id = str(uuid.uuid4())
    running = {"process_id": process_id, "status": "running"}
    mock_get_run.side_effect = [running, running]

    response = client.post(f"/api/v1/extract/{process_id}/cancel")

    assert response.status_code == 202
    assert response.get_json() == {
        "process_id": process_id,
        "status": "cancel_requested",
        "message": "Cancellation requested; authoritative terminal state is pending.",
    }
    mock_revoke.assert_not_called()


def test_get_extraction_status_requires_db_cancellation_even_if_task_revoked(client):
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
    assert payload["status"] == "started"


@pytest.mark.parametrize("status", ["running", "queued", "cancelled", "failed"])
@pytest.mark.parametrize("suffix", ["download/merged", "images", "images/figure.png"])
def test_artifacts_never_trust_celery_success_for_unfinished_or_failed_job(client, status, suffix):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(ExtractionRun(process_id=process_id, status=status, artifacts_json={"merged": "old-generation/merged.md"}))
    session.commit()
    session.close()
    with patch("celery_app.celery.AsyncResult", side_effect=AssertionError("Artifact lookup must not consult Redis")):
        response = client.get(f"/api/v1/extract/{process_id}/{suffix}")
    assert response.status_code == 409


@pytest.mark.parametrize("method", ["grobid", "docling", "marker", "merged", "audit", "page_provenance"])
def test_downloads_use_only_winning_database_artifacts_without_redis(client, method):
    process_id = str(uuid.uuid4())
    key = f"winning-generation/{method}"
    session = get_session()
    session.add(ExtractionRun(process_id=process_id, status="succeeded", artifacts_json={method: key}))
    session.commit()
    session.close()
    with (
        patch("celery_app.celery.AsyncResult", side_effect=AssertionError("Redis unavailable")),
        patch("app.api._s3_redirect_for_artifact", return_value=("durable output", 200)) as redirect,
    ):
        response = client.get(f"/api/v1/extract/{process_id}/download/{method}")
    assert response.status_code == 200
    redirect.assert_called_once_with(key)


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


@patch("app.pipeline.accept")
@patch("app.api._get_db_session", return_value=None)
def test_submit_extraction_fails_closed_when_idempotency_db_is_unavailable(mock_db_session, mock_apply_async, client):
    """Publishing without the RDS ownership claim would permit duplicates."""
    mock_apply_async.return_value = SimpleNamespace(id="process-id-placeholder")

    response = client.post(
        "/api/v1/extract",
        data={"file": (io.BytesIO(b"%PDF-1.4"), "test.pdf")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 503
    mock_apply_async.assert_not_called()


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
            "page_provenance": (
                "pdfx/audit/2026/02/11/x/page_provenance.json"
            ),
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
    assert "pdfx/audit/2026/02/11/x/page_provenance.json" in keys
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


def test_health_requires_finalizer_even_with_empty_gpu_queue(client,monkeypatch):
    monkeypatch.setattr('app.pipeline_health.fresh',lambda role:role!='finalization')
    redis=MagicMock();redis.llen.return_value=0;redis.hlen.return_value=0;redis.zcard.return_value=0
    with patch('requests.get',return_value=SimpleNamespace(status_code=200)),patch('redis.from_url',return_value=redis):
        response=client.get('/api/v1/health')
    assert response.status_code==503
    assert response.get_json()['checks']['finalization_worker']=='unavailable'


def test_health_counts_durable_jobs_despite_stale_broker_deliveries(client):
    pid = submit(client).get_json()['process_id']
    redis = MagicMock()
    redis.llen.return_value = 3
    redis.hlen.return_value = 2
    redis.zcard.return_value = 2
    with patch('requests.get', return_value=SimpleNamespace(status_code=200)), patch('redis.from_url', return_value=redis):
        assert client.get('/api/v1/health').get_json()['checks']['pipeline_active_jobs'] == 1
        from app.pipeline import cancel
        cancel(pid, 'finished lab work')
        checks = client.get('/api/v1/health').get_json()['checks']
    assert checks['database'] == 'ok'
    assert checks['pipeline_active_jobs'] == 0
    assert checks['broker_unacked'] == 2
    assert checks['broker_queued'] == 6
