from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError

import celery_app
from app.models import ExtractionRun


class _DummySession:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


class _RunSession:
    def __init__(self, run):
        self.run = run
        self.commits = 0

    def get(self, model, process_id):
        assert model is ExtractionRun
        assert process_id == self.run.process_id
        return self.run

    def add(self, run):
        self.run = run

    def commit(self):
        self.commits += 1


def test_upsert_running_clears_stale_terminal_fields():
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
    session = _RunSession(run)
    started_at = datetime(2026, 6, 12, tzinfo=timezone.utc)

    celery_app._upsert_extraction_run(
        session,
        process_id=run.process_id,
        status="running",
        started_at=started_at,
    )

    assert run.status == "running"
    assert run.started_at == started_at
    assert run.ended_at is None
    assert run.error_code is None
    assert run.error_message is None
    assert run.artifacts_json is None
    assert run.consensus_metrics_json is None
    assert run.log_s3_key is None
    assert run.llm_usage_json is None
    assert run.llm_cost_usd is None
    assert session.commits == 1


def test_upsert_running_persists_new_log_key_after_clearing_stale_fields():
    run = ExtractionRun(
        process_id="00000000-0000-0000-0000-000000000005",
        status="failed",
        ended_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        error_code="worker_lost",
        error_message="old failure",
        log_s3_key="pdfx/audit/old.ndjson",
    )
    session = _RunSession(run)

    celery_app._upsert_extraction_run(
        session,
        process_id=run.process_id,
        status="running",
        log_s3_key="pdfx/audit/2026/06/12/new.ndjson",
    )

    assert run.status == "running"
    assert run.ended_at is None
    assert run.error_code is None
    assert run.error_message is None
    assert run.log_s3_key == "pdfx/audit/2026/06/12/new.ndjson"
    assert session.commits == 1


def test_upsert_success_clears_stale_error_fields():
    run = ExtractionRun(
        process_id="00000000-0000-0000-0000-000000000004",
        status="running",
        error_code="cancelled",
        error_message="old cancellation",
    )
    session = _RunSession(run)

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
    assert session.commits == 1


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
