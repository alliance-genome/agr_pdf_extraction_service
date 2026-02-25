from sqlalchemy.exc import IntegrityError

import celery_app


class _DummySession:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


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
