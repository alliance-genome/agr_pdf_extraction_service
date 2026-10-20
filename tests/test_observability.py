import hashlib
import json
import pytest
import sentry_sdk
from sentry_sdk.transport import Transport
from pdfx_observability import initialize, sanitize_event, capture_failure, stage_scope


@pytest.fixture(autouse=True)
def isolated_sentry_settings(monkeypatch):
    for name in (
        "SENTRY_DSN",
        "SENTRY_RUNTIME_SECRET_ID",
        "SENTRY_PROJECT",
        "SENTRY_ALLOW_INSECURE_DSN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SENTRY_ENABLED", "true")


def test_sanitizer_rebuilds_error_and_transaction_without_content():
    sentinel = "PRIVATE-PDF-token-auth-signed-url"
    event = {
        "event_id": "a" * 32,
        "type": "transaction",
        "transaction": sentinel,
        "request": {"data": sentinel, "headers": {"Authorization": sentinel}},
        "user": {"email": sentinel},
        "extra": {"celery-job": sentinel},
        "breadcrumbs": [{"message": sentinel}],
        "logentry": {"message": sentinel},
        "tags": {
            "pdfx.stage": "provider",
            "pdfx.process_hash": "b" * 64,
            "untrusted": sentinel,
        },
        "exception": {
            "values": [
                {
                    "type": "RuntimeError",
                    "value": sentinel,
                    "stacktrace": {
                        "frames": [
                            {
                                "filename": "/private/" + sentinel + "/worker.py",
                                "function": "run",
                                "lineno": 12,
                                "vars": {"pdf": sentinel},
                                "context_line": sentinel,
                            }
                        ]
                    },
                }
            ]
        },
        "spans": [
            {"op": "http.client", "description": sentinel, "data": {"url": sentinel}}
        ],
        "contexts": {
            "trace": {"trace_id": "c" * 32, "data": sentinel},
            "custom": sentinel,
        },
    }
    clean = sanitize_event(event)
    assert sentinel not in json.dumps(clean)
    assert clean["exception"]["values"][0]["type"] == "RuntimeError"
    assert (
        clean["exception"]["values"][0]["stacktrace"]["frames"][0]["filename"]
        == "worker.py"
    )
    assert clean["tags"]["pdfx.process_hash"] == "b" * 64
    assert clean["transaction"] == "pdfx.provider"


def test_disabled_and_invalid_transport(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert initialize("api") is False
    monkeypatch.setenv("SENTRY_DSN", "http://public@example.invalid/10")
    monkeypatch.delenv("SENTRY_ALLOW_INSECURE_DSN", raising=False)
    with pytest.raises(ValueError, match="HTTPS"):
        initialize("worker")


def test_real_sdk_capture_redacts_and_deduplicates(monkeypatch):
    envelopes = []

    class MemoryTransport(Transport):
        def capture_envelope(self, envelope):
            envelopes.append(envelope)

    original_init = sentry_sdk.init

    def init(**kwargs):
        kwargs["transport"] = MemoryTransport
        return original_init(**kwargs)

    monkeypatch.setattr(sentry_sdk, "init", init)
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.invalid/10")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "1")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "lab")
    monkeypatch.setenv("SENTRY_RELEASE", "pdfx@test")
    try:
        assert initialize("worker")
        with stage_scope("job-identity", "finalization", 2):
            try:
                raise RuntimeError("PRIVATE-PDF-content")
            except RuntimeError as exc:
                assert capture_failure(exc, process_id="job-identity", stage="provider")
                assert capture_failure(exc, stage="provider") is None
        events = [
            item.get_event()
            for env in envelopes
            for item in env.items
            if item.type == "event"
        ]
        assert len(events) == 1
        assert (
            events[0]["tags"]["pdfx.process_hash"]
            == hashlib.sha256(b"job-identity").hexdigest()
        )
        assert events[0]["tags"]["pdfx.stage"] == "provider"
        assert events[0]["environment"] == "lab"
        assert "PRIVATE-PDF-content" not in json.dumps(events)
        assert "job-identity" not in json.dumps(events)
    finally:
        sentry_sdk.get_client().close()


def test_secret_project_selection_is_explicit_and_kill_switch_skips_lookup(monkeypatch):
    import boto3

    monkeypatch.delenv("SENTRY_DSN", raising=False)
    monkeypatch.setenv("SENTRY_RUNTIME_SECRET_ID", "runtime-secret")
    monkeypatch.delenv("SENTRY_PROJECT", raising=False)
    calls = []
    monkeypatch.setattr(boto3, "client", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="explicitly"):
        initialize("proxy")
    assert calls == []
    monkeypatch.setenv("SENTRY_ENABLED", "false")
    assert initialize("proxy") is False
    assert calls == []


def test_runtime_secret_selects_only_configured_project(monkeypatch):
    import boto3
    from types import SimpleNamespace

    monkeypatch.delenv("SENTRY_DSN", raising=False)
    monkeypatch.setenv("SENTRY_ENABLED", "true")
    monkeypatch.setenv("SENTRY_RUNTIME_SECRET_ID", "runtime-secret")
    monkeypatch.setenv("SENTRY_PROJECT", "pdfx-dev")
    secret = {
        "projects": {
            "pdfx-dev": {"dsn_private_vpc": "https://dev@example.invalid/10"},
            "pdfx-prod": {"dsn_private_vpc": "https://prod@example.invalid/11"},
        }
    }
    monkeypatch.setattr(
        boto3,
        "client",
        lambda *a, **k: SimpleNamespace(
            get_secret_value=lambda **kw: {"SecretString": json.dumps(secret)}
        ),
    )
    captured = {}
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: captured.update(kw))
    assert initialize("api")
    assert captured["dsn"] == "https://dev@example.invalid/10"
    assert captured["default_integrations"] is False
    assert captured["include_local_variables"] is False
    assert captured["max_request_body_size"] == "never"


def test_reporting_secret_outage_is_bounded_and_does_not_raise(monkeypatch, caplog):
    import boto3
    from botocore.exceptions import EndpointConnectionError

    monkeypatch.setenv("SENTRY_RUNTIME_SECRET_ID", "runtime-secret")
    monkeypatch.setenv("SENTRY_PROJECT", "pdfx-dev")
    configurations = []

    def unavailable(*args, **kwargs):
        configurations.append(kwargs["config"])
        raise EndpointConnectionError(endpoint_url="PRIVATE-SENTINEL")

    monkeypatch.setattr(boto3, "client", unavailable)
    assert initialize("extraction") is False
    assert configurations[0].connect_timeout == 2
    assert configurations[0].read_timeout == 2
    assert configurations[0].retries == {"total_max_attempts": 1}
    assert "reporting disabled" in caplog.text
    assert "PRIVATE-SENTINEL" not in caplog.text


def test_log_context_correlates_sentry_hash_across_internal_threads():
    import logging
    from concurrent.futures import ThreadPoolExecutor
    from app.logging_config import set_job_context, JobFormatter

    formatter = JobFormatter("%(message)s")

    def render():
        return formatter.format(
            logging.LogRecord("worker", logging.INFO, "", 1, "event", (), None)
        )

    try:
        set_job_context("job-identity", "extraction", 3)
        with ThreadPoolExecutor(1) as pool:
            message = pool.submit(render).result()
        assert "process_hash=" + hashlib.sha256(b"job-identity").hexdigest() in message
        assert "stage=extraction generation=3" in message
    finally:
        set_job_context()
    assert render() == "event"


def test_flask_unhandled_exception_is_redacted_and_correlated(monkeypatch):
    from app import create_app

    envelopes = []

    class MemoryTransport(Transport):
        def capture_envelope(self, envelope):
            envelopes.append(envelope)

    original_init = sentry_sdk.init
    monkeypatch.setattr(
        sentry_sdk, "init", lambda **kw: original_init(transport=MemoryTransport, **kw)
    )
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.invalid/10")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0")
    process_id = "00000000-0000-0000-0000-000000000123"
    try:
        app = create_app()
        app.config["PROPAGATE_EXCEPTIONS"] = False

        @app.route("/synthetic/<process_id>")
        def synthetic(process_id):
            raise RuntimeError("PRIVATE-PDF-SENTINEL")

        assert app.test_client().get("/synthetic/" + process_id).status_code == 500
        events = [
            item.get_event()
            for env in envelopes
            for item in env.items
            if item.type == "event"
        ]
        assert len(events) == 1
        assert events[0]["tags"]["pdfx.component"] == "api"
        assert (
            events[0]["tags"]["pdfx.process_hash"]
            == hashlib.sha256(process_id.encode()).hexdigest()
        )
        assert "PRIVATE-PDF-SENTINEL" not in str(events)
    finally:
        sentry_sdk.get_client().close()
        original_init(
            dsn="", default_integrations=False, auto_enabling_integrations=False
        )


@pytest.mark.parametrize(
    "secret",
    [
        "not-json",
        "{}",
        '{"projects":null}',
        '{"projects":{"pdfx-dev":{"dsn_private_vpc":123}}}',
    ],
)
def test_malformed_reporting_secret_disables_reporting_without_content(
    monkeypatch, caplog, secret
):
    import boto3
    from types import SimpleNamespace

    monkeypatch.setenv("SENTRY_RUNTIME_SECRET_ID", "runtime-secret")
    monkeypatch.setenv("SENTRY_PROJECT", "pdfx-dev")
    monkeypatch.setattr(
        boto3,
        "client",
        lambda *a, **kw: SimpleNamespace(
            get_secret_value=lambda **kw: {"SecretString": secret}
        ),
    )
    assert initialize("extraction") is False
    assert "reporting disabled" in caplog.text
    assert secret not in caplog.text
