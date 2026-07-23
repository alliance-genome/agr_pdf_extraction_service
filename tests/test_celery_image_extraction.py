import sys
import time
from types import SimpleNamespace

import pytest

import celery_app
from app.services.native_extractor_artifact import persist_native_extractor_artifact


class _ProgressTask:
    def __init__(self):
        self.updates = []

    def update_state(self, **payload):
        self.updates.append(payload)


def test_cache_without_native_artifact_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")
    cache_path = celery_app._cached_path("cached-empty", "grobid")
    with open(cache_path, "w", encoding="utf-8") as handle:
        handle.write(" \n\t")

    assert not celery_app._is_extractor_cached(
        "cached-empty", "grobid", extract_images=False
    )


def test_extraction_continues_when_one_of_three_methods_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")

    def fake_extract(method, *_args, **_kwargs):
        if method == "docling":
            raise RuntimeError("synthetic docling failure")
        return method, f"# {method}\n\nUsable output.", False

    monkeypatch.setattr(celery_app, "_run_single_extractor", fake_extract)
    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_bytes(b"%PDF-1.7\nfocused worker fixture")

    result = celery_app._run_extraction(
        _ProgressTask(),
        str(input_pdf),
        ["grobid", "docling", "marker"],
        False,
        file_hash="partial-success",
    )

    assert result["status"] == "success"
    assert result["extraction_status"] == "partial_success"
    assert result["available_extractors"] == ["grobid", "marker"]
    assert result["failed_extractors"] == ["docling"]
    assert result["extractor_failures"]["docling"]["error_code"] == "RuntimeError"
    assert result["methods_used"] == ["marker", "grobid"]


def test_candidate_merge_uses_two_valid_sources_when_third_is_whitespace(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")
    monkeypatch.setattr(celery_app.Config, "OPENAI_API_KEY", "")

    valid = (
        "# Source title\n\n"
        "## Abstract\n\n"
        "This complete source-backed abstract contains enough publication words "
        "to satisfy the absolute article floor while exercising partial extractor "
        "delivery without asking a model to generate any document text. The exact "
        "same bytes are available from two successful extraction methods for this "
        "focused worker integration check.\n\n"
        "## Results\n\n"
        "The result remains deterministic, source proven, auditable, and suitable "
        "for persistence in the candidate artifact bundle after one extractor "
        "returns only whitespace.\n\n"
        "## References\n\n"
        "1. Example reference for the focused integration fixture.\n"
    )

    def fake_extract(method, *_args, **_kwargs):
        text = "  \n\t" if method == "docling" else valid
        return method, text, False

    monkeypatch.setattr(celery_app, "_run_single_extractor", fake_extract)
    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_bytes(b"%PDF-1.7\npartial merge fixture")

    result = celery_app._run_extraction(
        _ProgressTask(),
        str(input_pdf),
        ["grobid", "docling", "marker"],
        True,
        file_hash="two-valid-one-empty",
    )

    assert result["status"] == "success"
    assert result["extraction_status"] == "partial_success"
    assert result["available_extractors"] == ["grobid", "marker"]
    assert result["failed_extractors"] == ["docling"]
    assert result["extractor_failures"]["docling"]["error_code"] == (
        "EmptyExtractorOutputError"
    )
    assert result["consensus_metrics"]["source_count"] == 2
    assert result["consensus_metrics"]["available_extractors"] == [
        "grobid",
        "marker",
    ]
    assert result["consensus_metrics"]["missing_extractors"] == ["docling"]


def test_extraction_fails_when_no_requested_method_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")
    monkeypatch.setattr(
        celery_app,
        "_run_single_extractor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("all failed")),
    )

    with pytest.raises(RuntimeError, match="all failed"):
        celery_app._run_extraction(
            _ProgressTask(),
            str(tmp_path / "input.pdf"),
            ["marker"],
            False,
            file_hash="no-success",
        )


def test_transient_extractor_failure_is_retried_once(tmp_path, monkeypatch):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")
    calls = []

    def fake_extract(method, *_args, **_kwargs):
        calls.append(method)
        if len(calls) == 1:
            raise TimeoutError("temporary timeout")
        return method, "# Recovered\n\nUsable source.", False

    monkeypatch.setattr(celery_app, "_run_single_extractor", fake_extract)

    result = celery_app._run_extraction(
        _ProgressTask(),
        str(tmp_path / "input.pdf"),
        ["grobid"],
        False,
        file_hash="retry-success",
    )

    assert calls == ["grobid", "grobid"]
    assert result["retried_extractors"] == ["grobid"]
    assert result["available_extractors"] == ["grobid"]
    assert result["failed_extractors"] == []
    assert result["emergency_ocr_used"] is False


def test_retry_does_not_spend_finalization_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")
    monkeypatch.setattr(celery_app.Config, "TASK_SOFT_TIME_LIMIT_SECONDS", 10)
    monkeypatch.setattr(
        celery_app.Config,
        "EXTRACTION_FINALIZATION_RESERVE_SECONDS",
        5,
    )
    calls = []

    def fail(method, *_args, **_kwargs):
        calls.append(method)
        raise TimeoutError("temporary timeout")

    monkeypatch.setattr(celery_app, "_run_single_extractor", fail)

    with pytest.raises(TimeoutError, match="temporary timeout"):
        celery_app._run_extraction(
            _ProgressTask(),
            str(tmp_path / "input.pdf"),
            ["grobid"],
            False,
            file_hash="reserve",
            job_started_monotonic=time.monotonic() - 6,
        )

    assert calls == ["grobid"]


def test_failed_cache_cleanup_does_not_defeat_partial_delivery(tmp_path, monkeypatch):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")
    calls = []

    def fake_extract(method, *_args, **_kwargs):
        calls.append(method)
        if method == "grobid":
            raise TimeoutError("temporary timeout")
        return method, "# Available\n\nUsable source.", False

    monkeypatch.setattr(celery_app, "_run_single_extractor", fake_extract)
    monkeypatch.setattr(
        celery_app,
        "_discard_failed_extractor_cache",
        lambda *_args: False,
    )

    result = celery_app._run_extraction(
        _ProgressTask(),
        str(tmp_path / "input.pdf"),
        ["grobid", "docling"],
        False,
        file_hash="cleanup-partial",
    )

    assert calls == ["grobid", "docling"]
    assert result["available_extractors"] == ["docling"]
    assert result["failed_extractors"] == ["grobid"]
    assert result["retried_extractors"] == []


def test_all_normal_failures_use_one_forced_full_page_docling_ocr(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")
    calls = []

    def fake_extract(method, *_args, **kwargs):
        calls.append((method, kwargs.get("force_full_page_ocr", False)))
        if method == "docling" and kwargs.get("force_full_page_ocr") is True:
            return method, "# OCR recovery\n\nUsable source.", False
        raise ValueError("permanent normal failure")

    monkeypatch.setattr(celery_app, "_run_single_extractor", fake_extract)

    result = celery_app._run_extraction(
        _ProgressTask(),
        str(tmp_path / "input.pdf"),
        ["marker"],
        False,
        file_hash="emergency-ocr",
    )

    assert calls == [("marker", False), ("docling", True)]
    assert result["available_extractors"] == ["docling"]
    assert result["failed_extractors"] == ["marker"]
    assert result["emergency_ocr_used"] is True


def test_candidate_worker_can_merge_without_openai_credentials(monkeypatch):
    monkeypatch.setattr(celery_app.Config, "OPENAI_API_KEY", "")

    assert celery_app._build_merge_llm_or_none() is None


def test_candidate_worker_can_merge_when_client_construction_fails(monkeypatch):
    monkeypatch.setattr(celery_app.Config, "OPENAI_API_KEY", "present")
    monkeypatch.setattr(
        celery_app,
        "_build_llm",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("client unavailable")),
    )

    assert celery_app._build_merge_llm_or_none() is None


def test_llm_accounting_preserves_raw_usage_when_pricing_fails(monkeypatch):
    summary = {
        "total_prompt_tokens": 10,
        "total_completion_tokens": 5,
        "total_cached_tokens": 0,
        "total_tokens": 15,
        "breakdown": {
            "candidate_selection": {
                "model": "unknown-model",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "calls": 1,
            }
        },
    }
    llm = SimpleNamespace(usage=SimpleNamespace(summary=lambda: summary))
    log = SimpleNamespace(error=lambda *args, **kwargs: None)
    monkeypatch.setattr(celery_app.Config, "LLM_PRICING", {})

    cost, usage = celery_app._compute_llm_accounting(llm, log=log)

    assert cost is None
    assert usage == summary


def test_llm_cost_alert_is_structured_and_never_changes_usage(monkeypatch):
    warnings = []
    log = SimpleNamespace(
        warning=lambda *args, **kwargs: warnings.append((args, kwargs))
    )
    usage = {"total_tokens": 123}
    monkeypatch.setattr(celery_app.Config, "LLM_COST_ALERT_USD_PER_JOB", 0.5)

    triggered = celery_app._record_llm_cost_alert(0.75, usage, log=log)

    assert triggered is True
    assert usage == {
        "total_tokens": 123,
        "cost_alert_threshold_usd": 0.5,
        "cost_alert_triggered": True,
    }
    assert warnings[0][1]["extra"]["_event"] == "llm_cost_alert"


def test_marker_cache_requires_images_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")

    file_hash = "abc123"
    marker_path = celery_app._cached_path(file_hash, "marker")
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write("Marker output")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"pdf fixture")
    persist_native_extractor_artifact(
        source="marker",
        output_filename=marker_path,
        native_bytes=b"{}",
        native_media_type="application/json",
        pdf_path=pdf_path,
        extractor_versions={"marker-pdf": "1.10.2"},
        options={"disable_links": True},
        expected_page_count=1,
        covered_pages=[1],
    )

    assert celery_app._is_extractor_cached(file_hash, "marker", extract_images=False) is True
    assert celery_app._is_extractor_cached(file_hash, "marker", extract_images=True) is False

    images_dir = tmp_path / "images" / "v1_abc123_marker"
    images_dir.mkdir(parents=True)
    (images_dir / "stale.png").write_bytes(b"old")

    assert celery_app._is_extractor_cached(file_hash, "marker", extract_images=True) is False

    (images_dir / celery_app.IMAGE_MANIFEST_FILENAME).write_text('{"images":[]}', encoding="utf-8")
    assert celery_app._is_extractor_cached(file_hash, "marker", extract_images=True) is True


def test_run_single_marker_extractor_uses_request_image_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")
    monkeypatch.setattr(celery_app.Config, "MARKER_DEVICE", "cpu")

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    seen = []

    class FakeMarker:
        def __init__(self, device="cpu", extract_images=False):
            seen.append(extract_images)

        def extract(self, _pdf_path, output_path):
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("Marker output")

    monkeypatch.setitem(
        sys.modules,
        "app.services.marker_service",
        SimpleNamespace(Marker=FakeMarker),
    )

    method, text, was_cached = celery_app._run_single_extractor(
        "marker",
        str(pdf_path),
        "hash-with-images",
        celery_app.Config,
        audit_logger=None,
        extract_images=True,
    )

    assert method == "marker"
    assert text == "Marker output"
    assert was_cached is False
    assert seen == [True]


def test_upload_artifacts_tags_image_objects(tmp_path, monkeypatch):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")
    monkeypatch.setattr(celery_app.Config, "IMAGE_RETENTION_TTL_SECONDS", 604800)

    file_hash = "abc123"
    images_dir = tmp_path / "images" / "v1_abc123_marker"
    images_dir.mkdir(parents=True)
    (images_dir / "fig1.png").write_bytes(b"image")

    calls = []

    class FakeAuditLogger:
        def upload_artifact(self, filename, content, subdir=None, tags=None):
            calls.append({
                "filename": filename,
                "content": content,
                "subdir": subdir,
                "tags": tags,
            })
            return f"pdfx/audit/process/images/{filename}"

    artifacts = celery_app._upload_artifacts(
        FakeAuditLogger(),
        {
            "file_hash": file_hash,
            "download_paths": {},
            "images": [{
                "filename": "fig1.png",
                "size_bytes": 5,
                "figure_label": "Figure 2",
                "figure_number": "2",
                "page_index": 1,
                "marker_image_type": "Figure",
                "marker_image_index": 3,
                "block_id": "/page/1/Figure/3",
                "group_id": "/page/1/FigureGroup/7",
                "caption_text": "Figure 2. Test caption.",
                "nearby_text": "Figure 2. Test caption.",
                "figure_decision_source": "llm_text",
                "image_reviewed": True,
                "image_review_method": "llm_text",
                "image_review_model": "gpt-5.6-luna",
                "image_review_classification": "scientific_figure",
                "image_review_is_scientific_figure": True,
                "image_review_confidence": 0.98,
                "image_review_reason": "Caption explicitly identifies a figure.",
            }],
        },
        merge=False,
    )

    assert artifacts["images"][0]["filename"] == "fig1.png"
    assert artifacts["images"][0]["figure_label"] == "Figure 2"
    assert artifacts["images"][0]["figure_number"] == "2"
    assert artifacts["images"][0]["page_index"] == 1
    assert artifacts["images"][0]["marker_image_type"] == "Figure"
    assert artifacts["images"][0]["marker_image_index"] == 3
    assert artifacts["images"][0]["block_id"] == "/page/1/Figure/3"
    assert artifacts["images"][0]["group_id"] == "/page/1/FigureGroup/7"
    assert artifacts["images"][0]["caption_text"] == "Figure 2. Test caption."
    assert artifacts["images"][0]["nearby_text"] == "Figure 2. Test caption."
    assert artifacts["images"][0]["figure_decision_source"] == "llm_text"
    assert artifacts["images"][0]["image_review_classification"] == "scientific_figure"
    assert calls[0]["subdir"] == "images"
    assert calls[0]["tags"] == {
        "pdfx-artifact-type": "extracted-image",
        "pdfx-retention": "temporary",
        "pdfx-retention-ttl-seconds": 604800,
    }


def test_review_images_with_text_context_applies_llm_decision(monkeypatch):
    monkeypatch.setattr(celery_app.Config, "IMAGE_TEXT_REVIEW_MODEL", "gpt-5.6-luna")

    class FakeReview:
        def model_dump(self):
            return {
                "classification": "publisher_logo",
                "is_scientific_figure": False,
                "confidence": 0.96,
                "figure_label": None,
                "figure_number": None,
                "needs_vision_review": False,
                "reason": "Text context looks like publisher branding.",
            }

    class FakeLLM:
        def review_image_context(self, image):
            assert image["caption_text"] is None
            return FakeReview()

    images = [{
        "filename": "_page_0_Picture_3.jpeg",
        "is_likely_figure": True,
        "heuristic_is_likely_figure": True,
        "caption_text": None,
        "nearby_text": "Taylor & Francis",
    }]

    reviewed = celery_app._review_images_with_text_context(images, FakeLLM())

    assert reviewed[0]["figure_decision_source"] == "llm_text"
    assert reviewed[0]["is_likely_figure"] is False
    assert reviewed[0]["heuristic_is_likely_figure"] is True
    assert reviewed[0]["image_reviewed"] is True
    assert reviewed[0]["image_review_classification"] == "publisher_logo"
    assert reviewed[0]["image_review_reason"] == "Text context looks like publisher branding."


def test_image_review_defaults_to_image_extraction_for_direct_celery_callers():
    assert celery_app._resolve_image_review_flags(extract_images=True, review_images=None) == (True, True)
    assert celery_app._resolve_image_review_flags(extract_images=True, review_images=False) == (True, False)
    assert celery_app._resolve_image_review_flags(extract_images=False, review_images=None) == (False, False)


def test_explicit_direct_image_review_still_extracts_images():
    assert celery_app._resolve_image_review_flags(extract_images=False, review_images=True) == (True, True)


def test_skipping_image_review_strips_cached_llm_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")

    file_hash = "abc123"
    images_dir = tmp_path / "images" / "v1_abc123_marker"
    images_dir.mkdir(parents=True)
    (images_dir / "_page_1_Figure_3.jpeg").write_bytes(b"image")
    (images_dir / celery_app.IMAGE_MANIFEST_FILENAME).write_text(
        """
        {
          "images": [{
            "filename": "_page_1_Figure_3.jpeg",
            "caption_text": "Fig. 2. Test caption.",
            "figure_label": "Fig. 2",
            "figure_number": "2",
            "figure_decision_source": "llm_text",
            "is_likely_figure": true,
            "heuristic_is_likely_figure": true,
            "image_reviewed": true,
            "image_review_method": "llm_text",
            "image_review_model": "gpt-5.6-luna",
            "image_review_classification": "scientific_figure",
            "image_review_is_scientific_figure": true,
            "image_review_confidence": 0.99,
            "image_review_reason": "Caption identifies a figure.",
            "image_review_needs_vision": false
          }]
        }
        """,
        encoding="utf-8",
    )

    class FakeTask:
        def update_state(self, *args, **kwargs):
            pass

    result = celery_app._run_extraction(
        FakeTask(),
        pdf_path=str(tmp_path / "test.pdf"),
        methods=[],
        merge=False,
        file_hash=file_hash,
        extract_images=True,
        review_images=False,
    )

    image = result["images"][0]
    assert image["figure_label"] is None
    assert image["figure_number"] is None
    assert image["figure_decision_source"] == "heuristic"
    assert image["image_reviewed"] is False
    assert image["image_review_method"] is None
    assert image["image_review_classification"] is None
    assert "llm_cost_usd" not in result
