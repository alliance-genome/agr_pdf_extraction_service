import sys
from types import SimpleNamespace

import celery_app


def test_marker_cache_requires_images_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(celery_app.Config, "CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(celery_app.Config, "EXTRACTION_CONFIG_VERSION", "1")

    file_hash = "abc123"
    marker_path = celery_app._cached_path(file_hash, "marker")
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write("Marker output")

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
                "image_review_model": "gpt-5.4-mini",
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
    monkeypatch.setattr(celery_app.Config, "IMAGE_TEXT_REVIEW_MODEL", "gpt-5.4-mini")

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
            "image_review_model": "gpt-5.4-mini",
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
