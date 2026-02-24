"""Integration tests for the FastAPI proxy routes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.state_machine import InstanceState


@pytest.fixture(autouse=True)
def _patch_singletons(monkeypatch):
    """Patch module-level singletons before importing app."""
    # Patch EC2Manager to avoid real boto3 calls
    mock_ec2 = MagicMock()
    mock_ec2.get_instance_state.return_value = ("stopped", None)
    monkeypatch.setattr("app.main.ec2_mgr", mock_ec2)

    # Patch CognitoAuth to bypass JWT validation
    mock_auth = MagicMock()
    mock_auth.validate_token.return_value = {"scope": "pdfx-api/extract"}
    monkeypatch.setattr("app.main.cognito_auth", mock_auth)

    # Patch lifecycle manager
    mock_lifecycle = MagicMock()
    mock_lifecycle.state = InstanceState.STOPPED
    mock_lifecycle.idle_seconds = 0.0
    mock_lifecycle.active_jobs = 0
    mock_lifecycle.private_ip = None
    mock_lifecycle.ensure_running = AsyncMock()
    mock_lifecycle.sync_state_from_ec2 = AsyncMock()
    mock_lifecycle.touch = MagicMock()
    monkeypatch.setattr("app.main.lifecycle", mock_lifecycle)

    # Reset job queue
    from app.job_queue import JobQueue
    monkeypatch.setattr("app.main.job_queue", JobQueue(max_size=10))


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client, monkeypatch):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["proxy"] == "ok"

    def test_health_no_auth_required(self, client):
        """Health endpoint should work without Authorization header."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200


class TestStatusEndpoint:
    def test_status_returns_state(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.lifecycle.state = InstanceState.STOPPED
        main_mod.lifecycle.idle_seconds = 120.0
        main_mod.lifecycle.active_jobs = 0
        resp = client.get("/api/v1/status", headers={"Authorization": "Bearer test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "stopped"

    def test_status_requires_auth(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.cognito_auth.validate_token.side_effect = __import__(
            "fastapi", fromlist=["HTTPException"]
        ).HTTPException(status_code=401, detail="Missing auth")
        resp = client.get("/api/v1/status")
        assert resp.status_code == 401


class TestWakeEndpoint:
    def test_wake_triggers_ensure_running(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.lifecycle.state = InstanceState.STARTING
        resp = client.post("/api/v1/wake", headers={"Authorization": "Bearer test"})
        assert resp.status_code == 200
        main_mod.lifecycle.ensure_running.assert_called_once()

    def test_wake_returns_state(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.lifecycle.state = InstanceState.READY
        resp = client.post("/api/v1/wake", headers={"Authorization": "Bearer test"})
        assert resp.status_code == 200
        assert resp.json()["state"] == "ready"


class TestExtractEndpoint:
    def test_extract_queues_when_stopped(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.lifecycle.state = InstanceState.STOPPED

        resp = client.post(
            "/api/v1/extract",
            headers={"Authorization": "Bearer test"},
            files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"methods": "grobid,docling,marker", "merge": "true"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "process_id" in data
        assert data["state"] == "stopped"
        assert data["progress"]["stage"] == "ec2_starting"
        main_mod.lifecycle.ensure_running.assert_called()

    def test_extract_requires_auth(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.cognito_auth.validate_token.side_effect = __import__(
            "fastapi", fromlist=["HTTPException"]
        ).HTTPException(status_code=401, detail="Missing auth")
        resp = client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 401

    def test_extract_requeues_when_immediate_forward_fails(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.lifecycle.state = InstanceState.READY
        main_mod.lifecycle.sync_state_from_ec2 = AsyncMock()

        async def _forward_fail(*args, **kwargs):
            raise HTTPException(status_code=502, detail="Failed to reach EC2 backend")

        monkeypatch.setattr(main_mod, "_forward_extraction", _forward_fail)

        resp = client.post(
            "/api/v1/extract",
            headers={"Authorization": "Bearer test"},
            files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"methods": "grobid,docling,marker", "merge": "true"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "queued"
        assert data["progress"]["stage"] in {"queued", "ec2_starting"}
        main_mod.lifecycle.sync_state_from_ec2.assert_called_once()
        main_mod.lifecycle.ensure_running.assert_called()


class TestExtractStatusEndpoint:
    def test_queued_job_returns_queued(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.job_queue.enqueue("test-123", b"pdf", {})
        main_mod.lifecycle.state = InstanceState.STARTING

        resp = client.get(
            "/api/v1/extract/test-123",
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert "progress" in data
        assert data["progress"]["stage"] == "ec2_starting"

    def test_queued_job_progress_has_correct_shape(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.job_queue.enqueue("shape-test-123", b"pdf", {})
        main_mod.lifecycle.state = InstanceState.STARTING

        resp = client.get(
            "/api/v1/extract/shape-test-123",
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        progress = resp.json()["progress"]
        assert isinstance(progress["stage"], str)
        assert isinstance(progress["stage_display"], str)
        assert isinstance(progress["stages_completed"], list)
        assert isinstance(progress["stages_pending"], list)
        assert isinstance(progress["stages_total"], int)
        assert isinstance(progress["stages_done"], int)
        assert isinstance(progress["percent"], int)
        assert progress["percent"] == 0

    def test_unknown_job_when_ec2_stopped(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.lifecycle.state = InstanceState.STOPPED

        resp = client.get(
            "/api/v1/extract/nonexistent",
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 503

    def test_status_uses_mapped_backend_process_id(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.lifecycle.state = InstanceState.READY
        main_mod.lifecycle.ec2_base_url = "http://172.31.1.100:5000"
        main_mod.proxy_to_backend_process["proxy-123"] = "backend-456"

        captured = {"url": None}

        class _DummyResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"process_id": "backend-456", "status": "running"}

        class _DummyClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url):
                captured["url"] = url
                return _DummyResponse()

        monkeypatch.setattr(main_mod.httpx, "AsyncClient", _DummyClient)

        resp = client.get(
            "/api/v1/extract/proxy-123",
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        assert captured["url"].endswith("/api/v1/extract/backend-456")
        assert resp.json()["process_id"] == "proxy-123"
        assert resp.json()["status"] == "running"

    def test_replay_submission_error_returns_failed(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.lifecycle.state = InstanceState.READY
        main_mod.replay_submission_errors["proxy-fail-1"] = "backend unavailable"

        resp = client.get(
            "/api/v1/extract/proxy-fail-1",
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "backend unavailable" in data["error"]
        assert data["progress"]["stage"] == "failed"


class TestExtractDownloadEndpoint:
    def test_download_returns_503_when_ec2_not_running(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.lifecycle.state = InstanceState.STOPPED

        resp = client.get(
            "/api/v1/extract/proc-1/download/grobid",
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 503

    def test_download_proxies_content_when_ready(self, client, monkeypatch):
        import app.main as main_mod

        main_mod.lifecycle.state = InstanceState.READY
        main_mod.lifecycle.ec2_base_url = "http://172.31.1.100:5000"

        class _DummyResponse:
            status_code = 200
            content = b"# extracted markdown\n"
            headers = {
                "content-type": "text/markdown; charset=utf-8",
                "content-disposition": "attachment; filename=test.md",
            }

        class _DummyClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.last_url = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url):
                self.last_url = url
                return _DummyResponse()

        monkeypatch.setattr(main_mod.httpx, "AsyncClient", _DummyClient)

        resp = client.get(
            "/api/v1/extract/proc-1/download/grobid",
            headers={"Authorization": "Bearer test"},
        )

        assert resp.status_code == 200
        assert resp.text == "# extracted markdown\n"
        assert "text/markdown" in resp.headers["content-type"]
        assert "content-disposition" in resp.headers

    def test_download_uses_mapped_backend_process_id(self, client, monkeypatch):
        import app.main as main_mod

        main_mod.lifecycle.state = InstanceState.READY
        main_mod.lifecycle.ec2_base_url = "http://172.31.1.100:5000"
        main_mod.proxy_to_backend_process["proxy-dl-1"] = "backend-dl-9"

        captured = {"url": None}

        class _DummyResponse:
            status_code = 200
            content = b"ok"
            headers = {"content-type": "text/plain"}

        class _DummyClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url):
                captured["url"] = url
                return _DummyResponse()

        monkeypatch.setattr(main_mod.httpx, "AsyncClient", _DummyClient)

        resp = client.get(
            "/api/v1/extract/proxy-dl-1/download/merged",
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200
        assert captured["url"].endswith("/api/v1/extract/backend-dl-9/download/merged")
