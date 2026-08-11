import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
from celery.exceptions import Ignore, Retry

import celery_app
from app.models import Base, ExtractionRun, get_engine, get_session, reset_db_engine


@pytest.fixture
def ownership_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'ownership.db'}")
    reset_db_engine()
    Base.metadata.create_all(bind=get_engine())
    yield
    reset_db_engine()


def test_concurrent_duplicate_delivery_does_not_run_with_live_owner(ownership_db, monkeypatch):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(ExtractionRun(process_id=process_id, status="queued"))
    session.commit()
    session.close()

    entered = threading.Event()
    release = threading.Event()
    executions = []

    def _impl(_task, *_args, **kwargs):
        executions.append(kwargs["process_id"])
        entered.set()
        release.wait(timeout=5)
        return {"process_id": process_id, "status": "complete"}

    monkeypatch.setattr(celery_app, "_extract_pdf_impl", _impl)
    arguments = ("/tmp/unused.pdf", ["grobid"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(celery_app.extract_pdf.run, *arguments, process_id=process_id)
        assert entered.wait(timeout=2)
        duplicate = pool.submit(celery_app.extract_pdf.run, *arguments, process_id=process_id)
        with pytest.raises(Ignore):
            duplicate.result(timeout=2)
        release.set()
        first_result = first.result(timeout=2)

    assert executions == [process_id]
    assert first_result["status"] == "complete"


def test_terminal_delivery_is_noop_cleans_retained_input_and_preserves_rds_state(
    ownership_db,
    monkeypatch,
    tmp_path,
):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(
        ExtractionRun(
            process_id=process_id,
            status="failed",
            error_code="ValueError",
            error_message="deterministic markup provenance is invalid",
        )
    )
    session.commit()
    session.close()
    monkeypatch.setattr(
        celery_app,
        "_extract_pdf_impl",
        lambda *_args, **_kwargs: pytest.fail("terminal delivery must not execute"),
    )
    retained_pdf = tmp_path / "retained-terminal-input.pdf"
    retained_pdf.write_bytes(b"%PDF retained after ambiguous terminal verification")

    with pytest.raises(Ignore):
        celery_app.extract_pdf.run(str(retained_pdf), ["grobid"], process_id=process_id)

    assert retained_pdf.exists() is False
    session = get_session()
    run = session.get(ExtractionRun, process_id)
    assert run.status == "failed"
    assert run.error_message == "deterministic markup provenance is invalid"
    session.close()


def test_stale_running_row_is_recoverable_after_live_owner_releases(ownership_db, monkeypatch):
    process_id = str(uuid.uuid4())
    session = get_session()
    session.add(ExtractionRun(process_id=process_id, status="running"))
    session.commit()
    session.close()
    calls = []

    def _impl(_task, *_args, **kwargs):
        calls.append(kwargs["process_id"])
        return {"process_id": process_id, "status": "complete"}

    monkeypatch.setattr(celery_app, "_extract_pdf_impl", _impl)

    result = celery_app.extract_pdf.run("/tmp/replayed.pdf", ["grobid"], process_id=process_id)

    assert calls == [process_id]
    assert result["status"] == "complete"


def test_live_duplicate_is_ignored_without_celery_success_state(ownership_db, monkeypatch):
    process_id = str(uuid.uuid4())

    @contextmanager
    def _not_owner(_process_id):
        yield False

    monkeypatch.setattr(celery_app, "_live_attempt_claim", _not_owner)

    result = celery_app.extract_pdf.apply(
        args=("/tmp/unused.pdf", ["grobid"]),
        kwargs={"process_id": process_id},
        task_id=process_id,
        throw=False,
    )

    assert result.state == "IGNORED"
    assert result.successful() is False


def test_worker_loss_redelivery_is_explicitly_enabled():
    assert celery_app.celery.conf.task_acks_late is True
    assert celery_app.celery.conf.task_reject_on_worker_lost is True


def test_postgres_live_attempt_lock_uses_autocommit(monkeypatch):
    class _Result:
        @staticmethod
        def scalar():
            return True

    class _Connection:
        def __init__(self):
            self.options = {}
            self.queries = []
            self.closed = False

        def execution_options(self, **kwargs):
            self.options.update(kwargs)
            return self

        def execute(self, query, params):
            self.queries.append((str(query), params))
            return _Result()

        def close(self):
            self.closed = True

    connection = _Connection()

    class _Engine:
        dialect = type("Dialect", (), {"name": "postgresql"})()

        @staticmethod
        def connect():
            return connection

    monkeypatch.setattr(celery_app, "get_engine", lambda: _Engine())

    with celery_app._live_attempt_claim("job-autocommit") as acquired:
        assert acquired is True

    assert connection.options == {"isolation_level": "AUTOCOMMIT"}
    assert "pg_try_advisory_lock" in connection.queries[0][0]
    assert "pg_advisory_unlock" in connection.queries[1][0]
    assert connection.closed is True


def test_status_database_outage_uses_delayed_task_retry(monkeypatch):
    monkeypatch.setattr(
        celery_app,
        "get_engine",
        lambda: (_ for _ in ()).throw(OSError("RDS unavailable")),
    )
    monkeypatch.setattr(celery_app.Config, "TERMINAL_STATE_RETRY_DELAY_SECONDS", 9)
    celery_app.extract_pdf.push_request(retries=0, called_directly=False, is_eager=True)
    try:
        with pytest.raises(Retry) as retry:
            celery_app.extract_pdf.run(
                "/tmp/retained-during-rds-outage.pdf",
                ["grobid"],
                process_id="00000000-0000-0000-0000-000000000011",
            )
    finally:
        celery_app.extract_pdf.pop_request()

    assert retry.value.when == 9
    assert isinstance(retry.value.exc, celery_app.StatusDatabaseUnavailable)


def test_terminal_persistence_retry_remains_available_after_default_retry_limit():
    assert celery_app.extract_pdf.max_retries is None
    celery_app.extract_pdf.push_request(retries=4, called_directly=False, is_eager=True)
    try:
        with pytest.raises(Retry):
            celery_app._retry_terminal_state_persistence(
                celery_app.extract_pdf,
                "00000000-0000-0000-0000-000000000009",
            )
    finally:
        celery_app.extract_pdf.pop_request()


def test_terminal_write_failure_retries_without_deleting_stable_input(tmp_path, monkeypatch):
    process_id = str(uuid.uuid4())
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF terminal retry")
    cache_path = tmp_path / "cache"
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(cache_path))
    monkeypatch.setattr(celery_app, "_get_file_hash", lambda _path: "abc123")
    monkeypatch.setattr(celery_app, "_safe_log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(celery_app, "_safe_upsert_extraction_run", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        celery_app,
        "_run_extraction",
        lambda *_args, **_kwargs: {
            "consensus_metrics": None,
            "llm_usage_json": None,
            "llm_cost_usd": None,
        },
    )
    monkeypatch.setattr(celery_app, "_upload_artifacts", lambda *_args, **_kwargs: {})
    terminal_writes = []

    def _fail_terminal_write(session, **kwargs):
        terminal_writes.append(kwargs)
        return False, session

    monkeypatch.setattr(celery_app, "_persist_terminal_extraction_run", _fail_terminal_write)

    class _AuditLogger:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def get_log_s3_key():
            return "pdfx/audit/retry.ndjson"

        @staticmethod
        def flush():
            return None

    monkeypatch.setattr(celery_app, "AuditLogger", _AuditLogger)
    retry_kwargs = {}

    class _Task:
        @staticmethod
        def retry(**kwargs):
            retry_kwargs.update(kwargs)
            raise Retry()

    with pytest.raises(Retry):
        celery_app._extract_pdf_impl(
            _Task(),
            str(pdf_path),
            ["grobid"],
            process_id=process_id,
        )

    assert terminal_writes[0]["status"] == "succeeded"
    assert retry_kwargs["max_retries"] is None
    assert pdf_path.exists()
