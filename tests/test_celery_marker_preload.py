import json
import sys
from types import SimpleNamespace

import pytest

import celery_app
from app import worker_main


def test_preload_marker_models_writes_ready_file(monkeypatch, tmp_path):
    ready_file = tmp_path / "marker_ready.json"
    fake_marker_service = SimpleNamespace(
        preload_marker_models=lambda **_: {
            "device": "cuda",
            "device_name": "fake-gpu",
            "dtype": "torch.float16",
            "elapsed_seconds": 1.25,
        },
    )

    monkeypatch.setitem(sys.modules, "app.services.marker_service", fake_marker_service)
    monkeypatch.setattr(celery_app.Config, "WORKER_PRELOAD_MARKER_MODELS", "auto")
    monkeypatch.setattr(celery_app.Config, "WORKER_PRELOAD_MARKER_REQUIRED", True)
    monkeypatch.setattr(celery_app.Config, "WORKER_PRELOAD_MARKER_EXTRACT_IMAGES", True)
    monkeypatch.setattr(celery_app.Config, "MARKER_DEVICE", "auto")
    monkeypatch.setattr(celery_app.Config, "MARKER_READY_FILE", str(ready_file))

    celery_app._preload_marker_models_for_worker()

    payload = json.loads(ready_file.read_text(encoding="utf-8"))
    assert payload["device"] == "cuda"
    assert payload["device_name"] == "fake-gpu"
    assert payload["pid"] > 0
    assert "created_at" in payload


def test_required_marker_preload_removes_stale_ready_file_on_failure(monkeypatch, tmp_path):
    ready_file = tmp_path / "marker_ready.json"
    ready_file.write_text('{"device": "cuda"}', encoding="utf-8")

    def fail_preload(**_):
        raise RuntimeError("no cuda")

    fake_marker_service = SimpleNamespace(preload_marker_models=fail_preload)

    monkeypatch.setitem(sys.modules, "app.services.marker_service", fake_marker_service)
    monkeypatch.setattr(celery_app.Config, "WORKER_PRELOAD_MARKER_MODELS", "auto")
    monkeypatch.setattr(celery_app.Config, "WORKER_PRELOAD_MARKER_REQUIRED", True)
    monkeypatch.setattr(celery_app.Config, "WORKER_PRELOAD_MARKER_EXTRACT_IMAGES", True)
    monkeypatch.setattr(celery_app.Config, "MARKER_DEVICE", "auto")
    monkeypatch.setattr(celery_app.Config, "MARKER_READY_FILE", str(ready_file))

    with pytest.raises(RuntimeError, match="no cuda"):
        celery_app._preload_marker_models_for_worker()

    assert not ready_file.exists()


def test_worker_main_preloads_before_starting_celery(monkeypatch):
    events = []

    class FakeCelery:
        def worker_main(self, argv):
            events.append(("celery", argv))

    monkeypatch.setattr(worker_main, "_configure_torch_for_worker_process", lambda: events.append("torch"))
    monkeypatch.setattr(worker_main, "_preload_marker_models_for_worker", lambda: events.append("preload"))
    monkeypatch.setattr(worker_main, "celery", FakeCelery())

    worker_main.main()

    assert events == [
        "torch",
        "preload",
        ("celery", ["worker", "--loglevel=info", "--pool=solo", "-Q", "default"]),
    ]
