"""Integration tests for the FastAPI proxy routes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

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
        assert resp.json()["status"] == "queued"

    def test_unknown_job_when_ec2_stopped(self, client, monkeypatch):
        import app.main as main_mod
        main_mod.lifecycle.state = InstanceState.STOPPED

        resp = client.get(
            "/api/v1/extract/nonexistent",
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 503
