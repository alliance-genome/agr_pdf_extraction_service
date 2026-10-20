"""Production task notifications cannot bypass durable ownership."""
from contextlib import contextmanager
from celery.exceptions import Ignore
from app import pipeline
from app.pipeline_tasks import finalize_stage, extract_stage
from celery_app import celery
import pytest


@contextmanager
def owned_lock(*args):
    yield object()


def test_unclaimed_finalizer_notification_is_ignored_not_success(monkeypatch):
    monkeypatch.setattr(pipeline, 'live_lock', owned_lock)
    monkeypatch.setattr(pipeline, 'claim', lambda *args: None)
    result = finalize_stage.apply(args=['00000000-0000-0000-0000-000000000001', 2], throw=False)
    assert result.state == 'IGNORED'
    assert isinstance(result.result, Ignore)


def test_no_monolithic_task_registered_and_extraction_has_no_public_result():
    assert 'extract_pdf' not in celery.tasks
    assert extract_stage.ignore_result is True
    assert celery.conf.task_acks_late is True
    assert celery.conf.task_reject_on_worker_lost is True


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

    monkeypatch.setattr(pipeline, "get_engine", lambda: _Engine())

    with pipeline.live_lock("job-autocommit") as acquired:
        assert acquired is connection

    assert connection.options == {"isolation_level": "AUTOCOMMIT"}
    assert "pg_try_advisory_lock" in connection.queries[0][0]
    assert "pg_advisory_unlock" in connection.queries[1][0]
    assert connection.closed is True
