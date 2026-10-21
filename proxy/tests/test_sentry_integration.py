"""Exercise the real SDK middleware, including exception deduplication."""

import hashlib
import pytest
import sentry_sdk
from sentry_sdk.transport import Transport
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.testclient import TestClient
from pdfx_observability import initialize


@pytest.mark.parametrize("kind,count", [("unhandled",1),("handled",1),("returned",1),("twice",2),("health",0),("client",0)])
def test_proxy_failures_have_one_capture_owner(monkeypatch, kind, count):
    from app.main import sentry_request_context, report_http_failure

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
    app.add_exception_handler(StarletteHTTPException, report_http_failure)
    process_id = "00000000-0000-0000-0000-000000000123"

    @app.get("/api/v1/extract/{process_id}")
    async def fail(process_id: str):
        if kind in {"handled", "twice", "health"}:
            raise HTTPException(503, "PRIVATE-PDF-SENTINEL")
        if kind == "client":
            raise HTTPException(404, "PRIVATE-PDF-SENTINEL")
        if kind == "returned":
            return JSONResponse({"private": "PRIVATE-PDF-SENTINEL"}, status_code=500)
        raise RuntimeError("PRIVATE-PDF-SENTINEL")

    app.add_api_route("/api/v1/health/{process_id}", fail)

    try:
        initialize("proxy")
        client = TestClient(app, raise_server_exceptions=False)
        path = ("/api/v1/health/" if kind == "health" else "/api/v1/extract/") + process_id
        for _ in range(2 if kind == "twice" else 1):
            response = client.get(path, headers={"Authorization": "Bearer PRIVATE-TOKEN"})
        assert response.status_code == (404 if kind == "client" else 503 if kind in {"handled", "twice", "health"} else 500)
        events = [
            item.get_event()
            for env in envelopes
            for item in env.items
            if item.type == "event"
        ]
        assert len(events) == count
        for event in events:
            assert event["tags"]["pdfx.stage"] == "proxy"
            assert event["tags"]["pdfx.process_hash"] == hashlib.sha256(process_id.encode()).hexdigest()
            if kind != "returned":
                assert event["exception"]["values"][-1]["stacktrace"]["frames"]
        assert "PRIVATE-TOKEN" not in str(events)
        assert process_id not in str(events)
        assert "PRIVATE-PDF-SENTINEL" not in str(events)
    finally:
        sentry_sdk.get_client().close()
        original_init(
            dsn="", default_integrations=False, auto_enabling_integrations=False
        )
