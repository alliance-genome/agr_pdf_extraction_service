from pathlib import Path
from types import SimpleNamespace
from config import Config
import celery_app as processing
from app.pipeline_tasks import workspace


def test_finalization_uses_snapshot_without_running_extractors(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(processing, "_is_extractor_cached", lambda *a, **kw: False)
    monkeypatch.setattr(
        processing,
        "_run_single_extractor",
        lambda *a, **kw: ("docling", "Full source text", False),
    )
    task = SimpleNamespace(update_state=lambda **kwargs: None)
    state = processing._extract_sources(
        task, "unused.pdf", ["docling"], False, file_hash="hash"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU finalization invoked an extractor")

    monkeypatch.setattr(processing, "_run_single_extractor", forbidden)
    result = processing._finalize_sources(
        task, "unused.pdf", ["docling"], False, file_hash="hash", source_state=state
    )
    assert result["status"] == "success"
    assert result["extractions"]["docling"] == "Full source text"


def test_identical_pdf_workspaces_cannot_clear_each_other(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "PIPELINE_WORKSPACE", str(tmp_path))
    original = Config.CACHE_FOLDER
    with workspace("first-job", 1) as first:
        Path(Config.CACHE_FOLDER).mkdir(parents=True)
        first_file = Path(processing._cached_path("same-pdf", "marker"))
        first_file.write_text("first receipt")
        with workspace("second-job", 1) as second:
            Path(Config.CACHE_FOLDER).mkdir(parents=True)
            second_file = Path(processing._cached_path("same-pdf", "marker"))
            second_file.write_text("second receipt")
            assert first_file != second_file
            assert first_file.read_text() == "first receipt"
        assert not second_file.exists()
        assert first_file.read_text() == "first receipt"
    assert not first_file.exists()
    assert Config.CACHE_FOLDER == original


def test_next_attempt_removes_abandoned_workspace_but_preserves_live_lease(
    tmp_path, monkeypatch
):
    import subprocess, sys, os

    monkeypatch.setattr(Config, "PIPELINE_WORKSPACE", str(tmp_path))
    code = "from app.pipeline_tasks import workspace; import os;\nwith workspace('dead-job',1) as p:\n p.mkdir(); (p/'partial').write_text('partial'); os._exit(9)"
    env = dict(
        os.environ, PIPELINE_WORKSPACE=str(tmp_path), AUDIT_S3_BUCKET_SSM_PARAM=""
    )
    result = subprocess.run([sys.executable, "-c", code], env=env)
    assert result.returncode == 9
    assert list(tmp_path.glob("job-dead-job-*"))
    with workspace("live-job", 1) as first:
        first.mkdir()
        (first / "intact").write_text("receipt")
        assert not list(tmp_path.glob("job-dead-job-*"))
        with workspace("next-job", 1):
            assert (first / "intact").read_text() == "receipt"
