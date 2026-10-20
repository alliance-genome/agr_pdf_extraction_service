"""Exercise the real SDK middleware, including exception deduplication."""

import hashlib
import sentry_sdk
from sentry_sdk.transport import Transport
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pdfx_observability import initialize


def test_unhandled_proxy_exception_reports_once_with_request_correlation(monkeypatch):
    from app.main import sentry_request_context

    envelopes = []

    class MemoryTransport(Transport):
        def capture_envelope(self, envelope):
            envelopes.append(envelope)

    original_init = sentry_sdk.init
    monkeypatch.setattr(
        sentry_sdk, "init", lambda **kw: original_init(transport=MemoryTransport, **kw)
    )
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.invalid/10")
    monkeypatch.setenv("SENTRY_ENABLED", "true")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0")
    app = FastAPI()
    app.middleware("http")(sentry_request_context)
    process_id = "00000000-0000-0000-0000-000000000123"

    @app.get("/api/v1/extract/{process_id}")
    async def fail(process_id: str):
        raise RuntimeError("PRIVATE-PDF-SENTINEL")

    try:
        initialize("proxy")
        response = TestClient(app, raise_server_exceptions=False).get(
            "/api/v1/extract/" + process_id
        )
        assert response.status_code == 500
        events = [
            item.get_event()
            for env in envelopes
            for item in env.items
            if item.type == "event"
        ]
        assert len(events) == 1
        assert events[0]["tags"]["pdfx.stage"] == "proxy"
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
