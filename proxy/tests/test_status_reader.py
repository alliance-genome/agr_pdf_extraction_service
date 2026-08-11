from datetime import datetime, timezone
from decimal import Decimal

from app.status_reader import RDSStatusReader


class _Cursor:
    def __init__(self, row):
        self.row = row
        self.query = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=None):
        self.query = query
        self.params = params

    def fetchone(self):
        return self.row


class _Connection:
    def __init__(self, row):
        self.cursor_instance = _Cursor(row)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self, **_kwargs):
        return self.cursor_instance


def test_reader_returns_authoritative_terminal_shape(monkeypatch):
    row = {
        "process_id": "job-failed",
        "reference_curie": "PMID:1",
        "mod_abbreviation": "MGI",
        "source_pdf_md5": "abc",
        "extract_images": False,
        "review_images": False,
        "status": "failed",
        "started_at": datetime(2026, 8, 3, 16, 0, tzinfo=timezone.utc),
        "ended_at": datetime(2026, 8, 3, 16, 44, tzinfo=timezone.utc),
        "error_code": "ValueError",
        "error_message": "deterministic markup provenance is invalid",
        "artifacts_json": None,
        "log_s3_key": "pdfx/audit/job.ndjson",
        "consensus_metrics_json": None,
        "llm_usage_json": {"total_tokens": 1},
        "llm_cost_usd": Decimal("0.125000"),
    }
    connection = _Connection(row)
    captured = {}

    def _connect(dsn, **kwargs):
        captured["dsn"] = dsn
        captured.update(kwargs)
        return connection

    monkeypatch.setattr("app.status_reader.psycopg2.connect", _connect)
    reader = RDSStatusReader("postgresql+psycopg2://app@db.example/pdfx")

    reachable, payload = reader.lookup("job-failed")

    assert reachable is True
    assert payload["status"] == "failed"
    assert payload["error"] == "deterministic markup provenance is invalid"
    assert payload["started_at"] == "2026-08-03T16:00:00Z"
    assert payload["llm_cost_usd"] == 0.125
    assert payload["image_count"] == 0
    assert payload["available_extractors"] == []
    assert connection.cursor_instance.params == ("job-failed",)
    assert "default_transaction_read_only=on" in captured["options"]
    assert captured["connect_timeout"] > 0


def test_reader_distinguishes_missing_row_from_unreachable(monkeypatch):
    monkeypatch.setattr("app.status_reader.psycopg2.connect", lambda *_args, **_kwargs: _Connection(None))
    reader = RDSStatusReader("postgresql://app@db.example/pdfx")

    assert reader.lookup("unknown") == (True, None)

    def _unreachable(*_args, **_kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr("app.status_reader.psycopg2.connect", _unreachable)
    assert reader.lookup("known") == (False, None)


def test_active_work_lookup_is_shared_and_fail_closed(monkeypatch):
    monkeypatch.setattr(
        "app.status_reader.psycopg2.connect",
        lambda *_args, **_kwargs: _Connection((True,)),
    )
    reader = RDSStatusReader("postgresql://app@db.example/pdfx")

    assert reader.has_active_work() == (True, True)

    monkeypatch.setattr(
        "app.status_reader.psycopg2.connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("network unavailable")),
    )
    assert reader.has_active_work() == (False, None)


def test_direct_status_error_shape_uses_backend_compatible_bound(monkeypatch):
    monkeypatch.setattr("app.status_reader.settings.STATUS_ERROR_MESSAGE_MAX_CHARS", 10)

    payload = RDSStatusReader._to_payload({
        "process_id": "job-long-error",
        "status": "failed",
        "error_code": "ValueError",
        "error_message": "x" * 15,
    })

    assert payload["error"] == "xxxxxxxxxx... [truncated 5 chars]"
