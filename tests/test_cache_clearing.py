from pathlib import Path

import pytest

import celery_app


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _cache_file(cache_dir: Path, version: str, file_hash: str, suffix: str) -> Path:
    return cache_dir / f"v{version}_{file_hash}_{suffix}"


def test_clear_merge_scope_keeps_extractor_cache(tmp_path, monkeypatch):
    file_hash = "abc123"
    version = "99"
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", version)

    grobid = _cache_file(tmp_path, version, file_hash, "grobid.md")
    docling = _cache_file(tmp_path, version, file_hash, "docling.md")
    marker = _cache_file(tmp_path, version, file_hash, "marker.md")
    merged = _cache_file(tmp_path, version, file_hash, "merged.md")
    merged_combo = _cache_file(tmp_path, version, file_hash, "docling_grobid_marker_merged.md")
    metrics = _cache_file(tmp_path, version, file_hash, "docling_grobid_marker_consensus_metrics.json")
    audit = _cache_file(tmp_path, version, file_hash, "docling_grobid_marker_audit.json")
    run_log = _cache_file(tmp_path, version, file_hash, "docling_grobid_marker_run.log")
    image = tmp_path / "images" / f"v{version}_{file_hash}_marker" / "page_1.png"

    for path in (grobid, docling, marker, merged, merged_combo, metrics, audit, run_log, image):
        _touch(path)

    result = celery_app._clear_cached_outputs(file_hash, "merge")

    assert result["scope"] == "merge"
    assert grobid.exists()
    assert docling.exists()
    assert marker.exists()
    assert not merged.exists()
    assert not merged_combo.exists()
    assert not metrics.exists()
    assert not audit.exists()
    assert run_log.exists()
    assert image.exists()


def test_clear_extraction_scope_also_clears_merge_cache(tmp_path, monkeypatch):
    file_hash = "def456"
    version = "100"
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", version)

    grobid = _cache_file(tmp_path, version, file_hash, "grobid.md")
    merged_combo = _cache_file(tmp_path, version, file_hash, "docling_grobid_marker_merged.md")
    image = tmp_path / "images" / f"v{version}_{file_hash}_marker" / "page_1.png"
    run_log = _cache_file(tmp_path, version, file_hash, "docling_grobid_marker_run.log")

    for path in (grobid, merged_combo, image, run_log):
        _touch(path)

    result = celery_app._clear_cached_outputs(file_hash, "extraction")

    assert result["scope"] == "extraction"
    assert not grobid.exists()
    assert not merged_combo.exists()
    assert not image.exists()
    assert run_log.exists()


def test_clear_cache_legacy_true_maps_to_all_scope():
    assert celery_app._normalize_clear_cache_scope(clear_cache_scope=None, clear_cache=True) == "all"


def test_clear_all_scope_removes_hash_prefixed_files(tmp_path, monkeypatch):
    file_hash = "ghi789"
    version = "101"
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", version)

    grobid = _cache_file(tmp_path, version, file_hash, "grobid.md")
    run_log = _cache_file(tmp_path, version, file_hash, "docling_grobid_marker_run.log")
    image = tmp_path / "images" / f"v{version}_{file_hash}_marker" / "page_1.png"

    other_hash = _cache_file(tmp_path, version, "another", "grobid.md")

    for path in (grobid, run_log, image, other_hash):
        _touch(path)

    result = celery_app._clear_cached_outputs(file_hash, "all")

    assert result["scope"] == "all"
    assert not grobid.exists()
    assert not run_log.exists()
    assert not image.exists()
    assert other_hash.exists()


def test_normalize_clear_cache_scope_rejects_invalid():
    with pytest.raises(ValueError, match="Invalid clear_cache_scope"):
        celery_app._normalize_clear_cache_scope(clear_cache_scope="banana")
