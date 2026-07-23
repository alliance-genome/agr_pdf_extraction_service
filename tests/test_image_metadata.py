from app.image_metadata import (
    annotate_image_diagnostics,
    apply_text_image_review,
    build_image_manifest_entry,
    extract_image_references,
    metadata_from_filename,
    normalize_image_manifest_entry,
    strip_text_image_review,
)


def test_metadata_from_marker_filename_keeps_internal_index_separate():
    metadata = metadata_from_filename("_page_1_Figure_3.jpeg")

    assert metadata == {
        "page_index": 1,
        "marker_image_type": "Figure",
        "marker_image_index": 3,
    }


def test_extract_image_references_keeps_alt_text_and_nearby_text_as_evidence():
    markdown = (
        "![Figure 1. Overview](_page_0_Figure_0.jpeg)\n\n"
        "![](_page_1_Figure_1.jpeg)\n"
        "Figure 2: Follow-up panel.\n"
    )

    references = extract_image_references(markdown)

    assert references["_page_0_Figure_0.jpeg"] == {
        "alt_text": "Figure 1. Overview",
        "nearby_text": None,
    }
    assert references["_page_1_Figure_1.jpeg"] == {
        "alt_text": None,
        "nearby_text": "\nFigure 2: Follow-up panel.\n",
    }


def test_build_image_manifest_entry_does_not_guess_paper_figure_number_from_filename():
    entry = build_image_manifest_entry("_page_1_Figure_3.jpeg", size_bytes=42)

    assert entry["filename"] == "_page_1_Figure_3.jpeg"
    assert entry["size_bytes"] == 42
    assert entry["marker_image_type"] == "Figure"
    assert entry["marker_image_index"] == 3
    assert entry["figure_label"] is None
    assert entry["figure_number"] is None


def test_build_image_manifest_entry_merges_structured_caption_metadata_as_raw_evidence():
    entry = build_image_manifest_entry(
        "_page_2_Figure_4.jpeg",
        size_bytes=42,
        references={
            "_page_2_Figure_4.jpeg": {
                "structured_metadata": {
                    "block_id": "/page/2/Figure/4",
                    "group_id": "/page/2/FigureGroup/9",
                    "bbox": [10.0, 20.0, 200.0, 300.0],
                    "caption_text": "Figure 5. Sample caption.",
                },
            },
        },
    )

    assert entry["block_id"] == "/page/2/Figure/4"
    assert entry["group_id"] == "/page/2/FigureGroup/9"
    assert entry["bbox"] == [10.0, 20.0, 200.0, 300.0]
    assert entry["caption_text"] == "Figure 5. Sample caption."
    assert entry["figure_label"] is None
    assert entry["figure_number"] is None


def test_annotate_image_diagnostics_flags_tiny_unlabeled_artifacts():
    entry = build_image_manifest_entry("_page_1_Figure_3.jpeg")
    entry["image_width"] = 196
    entry["image_height"] = 38

    annotate_image_diagnostics(entry)

    assert entry["is_likely_figure"] is False
    assert entry["heuristic_is_likely_figure"] is False
    assert entry["figure_decision_source"] == "heuristic"
    assert entry["diagnostic_flags"] == ["small_image", "no_caption", "no_figure_label"]


def test_annotate_image_diagnostics_accepts_captioned_picture():
    entry = build_image_manifest_entry("_page_2_Picture_4.jpeg")
    entry["image_width"] = 400
    entry["image_height"] = 300
    entry["caption_text"] = "Figure 2. A real microscopy panel."
    entry["figure_label"] = "Figure 2"
    entry["figure_number"] = "2"

    annotate_image_diagnostics(entry)

    assert entry["is_likely_figure"] is True
    assert entry["heuristic_is_likely_figure"] is True
    assert entry["figure_decision_source"] == "heuristic"
    assert entry["diagnostic_flags"] == ["marker_picture_type"]


def test_normalize_legacy_string_manifest_entry():
    entry = normalize_image_manifest_entry("_page_0_Picture_1.jpeg")

    assert entry["filename"] == "_page_0_Picture_1.jpeg"
    assert entry["page_index"] == 0
    assert entry["marker_image_type"] == "Picture"
    assert entry["marker_image_index"] == 1


def test_apply_text_image_review_overrides_decision_but_keeps_heuristic():
    entry = build_image_manifest_entry("_page_1_Figure_3.jpeg")
    entry["image_width"] = 400
    entry["image_height"] = 300
    annotate_image_diagnostics(entry)

    reviewed = apply_text_image_review(
        entry,
        {
            "classification": "publisher_logo",
            "is_scientific_figure": False,
            "confidence": 0.94,
            "reason": "Nearby text indicates publisher branding.",
            "needs_vision_review": False,
            "figure_label": None,
            "figure_number": None,
        },
        model="gpt-5.6-luna",
    )

    assert reviewed["heuristic_is_likely_figure"] is True
    assert reviewed["is_likely_figure"] is False
    assert reviewed["figure_decision_source"] == "llm_text"
    assert reviewed["image_reviewed"] is True
    assert reviewed["image_review_classification"] == "publisher_logo"
    assert reviewed["image_review_model"] == "gpt-5.6-luna"


def test_apply_text_image_review_uses_llm_label_and_number_without_reconstruction():
    entry = build_image_manifest_entry("_page_2_Figure_4.jpeg")

    reviewed = apply_text_image_review(
        entry,
        {
            "classification": "scientific_figure",
            "is_scientific_figure": True,
            "confidence": 0.99,
            "reason": "Caption explicitly identifies a figure.",
            "needs_vision_review": False,
            "figure_label": "Figure",
            "figure_number": "2",
        },
        model="gpt-5.6-luna",
    )

    assert reviewed["figure_label"] == "Figure"
    assert reviewed["figure_number"] == "2"


def test_strip_text_image_review_removes_cached_llm_metadata_and_restores_heuristic():
    entry = build_image_manifest_entry("_page_1_Figure_3.jpeg")
    entry["image_width"] = 400
    entry["image_height"] = 300
    entry["caption_text"] = "Fig. 2. Test caption."
    annotate_image_diagnostics(entry)
    reviewed = apply_text_image_review(
        entry,
        {
            "classification": "scientific_figure",
            "is_scientific_figure": True,
            "confidence": 0.99,
            "reason": "Caption explicitly identifies a figure.",
            "needs_vision_review": False,
            "figure_label": "Fig. 2",
            "figure_number": "2",
        },
        model="gpt-5.6-luna",
    )

    stripped = strip_text_image_review(reviewed)

    assert stripped["figure_label"] is None
    assert stripped["figure_number"] is None
    assert stripped["figure_decision_source"] == "heuristic"
    assert stripped["image_reviewed"] is False
    assert stripped["image_review_method"] is None
    assert stripped["image_review_classification"] is None
    assert stripped["image_review_reason"] is None
    assert stripped["heuristic_is_likely_figure"] is True
    assert stripped["is_likely_figure"] is True
