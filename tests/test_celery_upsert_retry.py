import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from celery.exceptions import Retry
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

import celery_app
from app.models import Base, ExtractionRun, update_extraction_run_if_nonterminal


class _DummySession:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def run_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def test_upsert_running_cannot_reset_terminal_fields(run_session):
    run = ExtractionRun(
        process_id="00000000-0000-0000-0000-000000000003",
        status="failed",
        ended_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        error_code="worker_lost",
        error_message="old failure",
        artifacts_json={"merged": "old"},
        consensus_metrics_json={"old": True},
        log_s3_key="pdfx/audit/old.ndjson",
        llm_usage_json={"total_tokens": 123},
        llm_cost_usd=1.23,
    )
    session = run_session
    session.add(run)
    session.commit()
    started_at = datetime(2026, 6, 12, tzinfo=timezone.utc)

    celery_app._upsert_extraction_run(
        session,
        process_id=run.process_id,
        status="running",
        started_at=started_at,
    )

    assert run.status == "failed"
    assert run.started_at is None
    assert run.ended_at == datetime(2026, 6, 1)
    assert run.error_code == "worker_lost"
    assert run.error_message == "old failure"
    assert run.artifacts_json == {"merged": "old"}
    assert run.consensus_metrics_json == {"old": True}
    assert run.log_s3_key == "pdfx/audit/old.ndjson"
    assert run.llm_usage_json == {"total_tokens": 123}
    assert float(run.llm_cost_usd) == 1.23


def test_upsert_running_cannot_replace_terminal_log_key(run_session):
    run = ExtractionRun(
        process_id="00000000-0000-0000-0000-000000000005",
        status="failed",
        ended_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        error_code="worker_lost",
        error_message="old failure",
        log_s3_key="pdfx/audit/old.ndjson",
    )
    session = run_session
    session.add(run)
    session.commit()

    celery_app._upsert_extraction_run(
        session,
        process_id=run.process_id,
        status="running",
        log_s3_key="pdfx/audit/2026/06/12/new.ndjson",
    )

    assert run.status == "failed"
    assert run.ended_at == datetime(2026, 6, 1)
    assert run.error_code == "worker_lost"
    assert run.error_message == "old failure"
    assert run.log_s3_key == "pdfx/audit/old.ndjson"


def test_statusless_update_cannot_mutate_terminal_fields(run_session):
    run = ExtractionRun(
        process_id="00000000-0000-0000-0000-000000000006",
        status="cancelled",
        ended_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        error_code="cancelled",
        error_message="user cancelled",
        log_s3_key="pdfx/audit/original.ndjson",
    )
    session = run_session
    session.add(run)
    session.commit()

    celery_app._upsert_extraction_run(
        session,
        process_id=run.process_id,
        log_s3_key="pdfx/audit/late-worker.ndjson",
    )

    assert run.status == "cancelled"
    assert run.error_message == "user cancelled"
    assert run.log_s3_key == "pdfx/audit/original.ndjson"


def test_empty_existing_upsert_is_a_noop(run_session):
    run = ExtractionRun(
        process_id="00000000-0000-0000-0000-000000000010",
        status="running",
    )
    run_session.add(run)
    run_session.commit()

    result = celery_app._upsert_extraction_run(run_session, process_id=run.process_id)

    assert result is run
    assert run.status == "running"


def test_upsert_success_clears_stale_error_fields(run_session):
    run = ExtractionRun(
        process_id="00000000-0000-0000-0000-000000000004",
        status="running",
        error_code="cancelled",
        error_message="old cancellation",
    )
    session = run_session
    session.add(run)
    session.commit()

    celery_app._upsert_extraction_run(
        session,
        process_id=run.process_id,
        status="succeeded",
        artifacts_json={"merged": "new"},
    )

    assert run.status == "succeeded"
    assert run.error_code is None
    assert run.error_message is None
    assert run.artifacts_json == {"merged": "new"}


def test_two_sessions_allow_only_one_first_terminal_transition(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'terminal-race.db'}",
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    process_id = "00000000-0000-0000-0000-000000000007"
    seed = factory()
    seed.add(ExtractionRun(process_id=process_id, status="running"))
    seed.commit()
    seed.close()
    barrier = threading.Barrier(2)

    def _transition(status):
        session = factory()
        try:
            barrier.wait(timeout=2)
            updated, current = update_extraction_run_if_nonterminal(
                session,
                process_id,
                {
                    ExtractionRun.status: status,
                    ExtractionRun.ended_at: datetime.now(timezone.utc),
                    ExtractionRun.error_code: "cancelled" if status == "cancelled" else None,
                },
            )
            return updated, current.status
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_transition, ("succeeded", "cancelled")))

    assert sum(1 for updated, _status in results if updated) == 1
    final = factory()
    assert final.get(ExtractionRun, process_id).status in {"succeeded", "cancelled"}
    final.close()
    engine.dispose()


def test_safe_upsert_retries_once_on_unique_violation(monkeypatch):
    calls = {"count": 0}
    session = _DummySession()

    class _UniqueViolation:
        pgcode = "23505"

        def __str__(self):
            return "duplicate key value violates unique constraint"

    def _fake_upsert(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise IntegrityError("duplicate", params=None, orig=_UniqueViolation())
        return None

    monkeypatch.setattr(celery_app, "_upsert_extraction_run", _fake_upsert)

    ok = celery_app._safe_upsert_extraction_run(
        session,
        process_id="00000000-0000-0000-0000-000000000001",
        status="running",
    )

    assert ok is True
    assert calls["count"] == 2
    assert session.rollbacks == 1


def test_safe_upsert_returns_false_on_non_unique_failure(monkeypatch):
    calls = {"count": 0}
    session = _DummySession()

    def _fake_upsert(*args, **kwargs):
        calls["count"] += 1
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(celery_app, "_upsert_extraction_run", _fake_upsert)

    ok = celery_app._safe_upsert_extraction_run(
        session,
        process_id="00000000-0000-0000-0000-000000000002",
        status="running",
    )

    assert ok is False
    assert calls["count"] == 1
    assert session.rollbacks == 1


def test_terminal_write_failure_retries_and_later_repairs_authoritative_row(run_session, monkeypatch):
    process_id = "00000000-0000-0000-0000-000000000008"
    run_session.add(ExtractionRun(process_id=process_id, status="running"))
    run_session.commit()
    real_safe_upsert = celery_app._safe_upsert_extraction_run
    monkeypatch.setattr(celery_app, "_safe_upsert_extraction_run", lambda *_args, **_kwargs: False)

    persisted, _session = celery_app._persist_terminal_extraction_run(
        run_session,
        process_id=process_id,
        status="succeeded",
        ended_at=datetime.now(timezone.utc),
    )

    assert persisted is False
    run_session.expire_all()
    assert run_session.get(ExtractionRun, process_id).status == "running"

    retry_kwargs = {}

    class _Task:
        @staticmethod
        def retry(**kwargs):
            retry_kwargs.update(kwargs)
            raise Retry()

    with pytest.raises(Retry):
        celery_app._retry_terminal_state_persistence(_Task(), process_id)
    assert retry_kwargs["max_retries"] is None
    assert retry_kwargs["countdown"] == celery_app.Config.TERMINAL_STATE_RETRY_DELAY_SECONDS

    monkeypatch.setattr(celery_app, "_safe_upsert_extraction_run", real_safe_upsert)
    persisted, _session = celery_app._persist_terminal_extraction_run(
        run_session,
        process_id=process_id,
        status="succeeded",
        ended_at=datetime.now(timezone.utc),
    )

    assert persisted is True
    run_session.expire_all()
    assert run_session.get(ExtractionRun, process_id).status == "succeeded"
