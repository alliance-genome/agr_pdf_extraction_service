import hashlib
import json

import pytest

from app.services.deterministic_markup import DETERMINISTIC_MARKUP_SHAPES
import app.services.page_provenance as page_provenance_module
from app.services.page_provenance import (
    MARKER_PAGE_SEPARATOR,
    bound_page_resolution_ranges,
    build_source_page_provenance_bytes,
    build_page_resolution_batches,
    docling_markdown_with_page_ranges,
    marker_markdown_with_page_ranges,
    partition_source_ranges,
    finalize_merged_page_provenance_bytes,
    project_merged_page_ranges,
    validate_merged_page_provenance_bytes,
    validate_source_page_provenance_bytes,
)


def test_every_emitted_deterministic_markup_shape_has_page_ownership():
    owned_operations = (
        page_provenance_module._FOLLOWING_OWNER_OPERATIONS
        | page_provenance_module._PRECEDING_OWNER_OPERATIONS
        | page_provenance_module._SURROUNDING_OWNER_OPERATIONS
        | {"native_emphasis_projection"}
    )

    assert owned_operations == set(DETERMINISTIC_MARKUP_SHAPES)


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _rebind_record(payload):
    core = {key: value for key, value in payload.items() if key != "record_sha256"}
    payload["record_sha256"] = _sha256(
        json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def test_docling_transition_removal_preserves_markdown_and_utf8_offsets():
    sentinel = "PDFX_DOCLING_PAGE_BOUNDARY_TEST"
    paginated = f"  α page one\n\n{sentinel}\n\n漢 page two  "

    markdown, ranges = docling_markdown_with_page_ranges(
        paginated,
        sentinel=sentinel,
        expected_page_count=2,
        primary_page_order_is_monotonic=True,
    )

    assert markdown == "α page one\n\n\n\n漢 page two"
    encoded = markdown.encode("utf-8")
    assert encoded[ranges[0]["byte_start"] : ranges[0]["byte_end"]].decode() == (
        "α page one\n\n"
    )
    assert encoded[ranges[1]["byte_start"] : ranges[1]["byte_end"]].decode() == (
        "\n\n漢 page two"
    )
    assert [item["page_number"] for item in ranges] == [1, 2]


def test_docling_unsafe_transition_inventory_leaves_all_bytes_residual():
    markdown, ranges = docling_markdown_with_page_ranges(
        "first and third",
        sentinel="MISSING",
        expected_page_count=3,
        primary_page_order_is_monotonic=True,
    )

    assert markdown == "first and third"
    assert ranges == []


def test_docling_non_monotonic_primary_page_order_leaves_all_bytes_residual():
    sentinel = "PDFX_DOCLING_PAGE_BOUNDARY_TEST"
    paginated = f"page one{sentinel}page two and revisited one{sentinel}page three"

    markdown, ranges = docling_markdown_with_page_ranges(
        paginated,
        sentinel=sentinel,
        expected_page_count=3,
        primary_page_order_is_monotonic=False,
    )

    assert markdown == "page onepage two and revisited onepage three"
    assert ranges == []


def test_marker_exact_tokens_are_removed_without_changing_legacy_cleaned_text():
    separator = MARKER_PAGE_SEPARATOR
    paginated = (
        f"{{0}}{separator}\n\nPage one.\n\n"
        f"{{1}}{separator}\n\nPage two.\n\n"
        f"{{2}}{separator}"
    )

    markdown, ranges = marker_markdown_with_page_ranges(
        paginated,
        expected_page_count=3,
    )

    assert markdown == "Page one.\n\nPage two."
    assert [item["page_number"] for item in ranges] == [1, 2]
    assert ranges[0]["byte_start"] == 0
    assert ranges[-1]["byte_end"] == len(markdown.encode("utf-8"))


def test_marker_single_pass_preserves_structured_content_and_blank_page():
    separator = MARKER_PAGE_SEPARATOR
    pages = [
        "# Table\n\n| A | B |\n| - | - |\n| 1 | 2 |",
        "",
        "- item\n\n![figure](image.png) and [link](https://example.org)",
        "Terminal page.",
    ]
    paginated = "".join(
        f"{{{page_id}}}{separator}\n\n{page}"
        + ("\n\n" if page_id < len(pages) - 1 else "")
        for page_id, page in enumerate(pages)
    )

    markdown, ranges = marker_markdown_with_page_ranges(
        paginated,
        expected_page_count=len(pages),
    )

    assert markdown == "\n\n".join(pages).strip()
    assert [item["page_number"] for item in ranges] == [1, 2, 3, 4]
    assert markdown.endswith("Terminal page.")


@pytest.mark.parametrize(
    "paginated",
    [
        f"{{0}}{MARKER_PAGE_SEPARATOR}\n\nbody",
        f"{{0}}{MARKER_PAGE_SEPARATOR}\n\nbody\n\n{{2}}{MARKER_PAGE_SEPARATOR}",
        f"prefix{{0}}{MARKER_PAGE_SEPARATOR}\n\nbody",
    ],
)
def test_marker_rejects_missing_or_misordered_token_inventory(paginated):
    with pytest.raises(ValueError, match="token|leading"):
        marker_markdown_with_page_ranges(paginated, expected_page_count=2)


def test_source_record_is_digest_bound_and_exactly_partitions_markdown():
    markdown = "α one\n\ntwo".encode()
    native = b'{"native":true}'
    pdf_digest = _sha256(b"pdf")
    raw = build_source_page_provenance_bytes(
        source="docling",
        pdf_sha256=pdf_digest,
        native_bytes=native,
        markdown_bytes=markdown,
        expected_page_count=2,
        extractor_versions={"docling": "2.113.0", "docling-core": "2.87.1"},
        options={"page_provenance": "digest_sentinel_v1"},
        evidence_ranges=[
            {
                "byte_start": 0,
                "byte_end": len("α one".encode()),
                "page_number": 1,
                "candidate_pages": [1],
                "method": "direct",
            }
        ],
        residual_reason="unsafe_docling_page_transition",
    )

    payload = validate_source_page_provenance_bytes(
        raw,
        source="docling",
        pdf_sha256=pdf_digest,
        native_sha256=_sha256(native),
        markdown_sha256=_sha256(markdown),
    )
    assert payload["ranges"][0]["page_number"] == 1
    assert payload["ranges"][1]["page_number"] is None
    assert payload["ranges"][0]["byte_start"] == 0
    assert payload["ranges"][-1]["byte_end"] == len(markdown)

    tampered = json.loads(raw)
    tampered["ranges"][-1]["byte_end"] -= 1
    with pytest.raises(ValueError, match="cover Markdown|digest"):
        validate_source_page_provenance_bytes(json.dumps(tampered).encode())


def test_source_ranges_reject_overlap_and_out_of_bounds():
    with pytest.raises(ValueError, match="overlap"):
        partition_source_ranges(
            b"abcdef",
            [
                {"byte_start": 0, "byte_end": 4},
                {"byte_start": 3, "byte_end": 6},
            ],
            residual_reason="missing",
        )


def test_merge_audit_projection_assigns_static_markup_without_changing_bytes():
    source = b"Body"
    source_raw = build_source_page_provenance_bytes(
        source="docling",
        pdf_sha256=_sha256(b"pdf"),
        native_bytes=b"native",
        markdown_bytes=source,
        expected_page_count=3,
        extractor_versions={"docling": "2.113.0"},
        options={"page_provenance": "digest_sentinel_v1"},
        evidence_ranges=[
            {
                "byte_start": 0,
                "byte_end": 4,
                "page_number": 2,
                "candidate_pages": [2],
                "method": "direct",
            }
        ],
        residual_reason="fixture",
    )
    source_map = json.loads(source_raw)
    merged = b"## Body\n"
    audit = [
        {
            "output_byte_start": 0,
            "output_byte_end": 3,
            "source": "deterministic_markup",
            "transformation": "alliance_heading_role_marker",
            "source_byte_start": 0,
            "source_byte_end": 3,
        },
        {
            "output_byte_start": 3,
            "output_byte_end": 7,
            "source": "docling",
            "source_byte_start": 0,
            "source_byte_end": 4,
        },
        {
            "output_byte_start": 7,
            "output_byte_end": 8,
            "source": "deterministic_markup",
            "transformation": "trailing_newline_normalization",
            "source_byte_start": 0,
            "source_byte_end": 1,
        },
    ]
    projected = project_merged_page_ranges(
        merged_bytes=merged,
        audit=audit,
        source_maps={"docling": source_map},
    )
    assert [item["page_number"] for item in projected] == [2, 2, 2]
    assert [item["method"] for item in projected] == [
        "deterministic_owner",
        "direct",
        "deterministic_owner",
    ]

    audit_bytes = json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
    raw, summary = finalize_merged_page_provenance_bytes(
        pdf_sha256=_sha256(b"pdf"),
        expected_page_count=3,
        merged_bytes=merged,
        audit_sha256=_sha256(audit_bytes),
        merge_contract_id="contract-v2",
        source_map_sha256={"docling": _sha256(source_raw)},
        projected_ranges=projected,
    )
    payload = validate_merged_page_provenance_bytes(
        raw,
        pdf_sha256=_sha256(b"pdf"),
        merged_sha256=_sha256(merged),
        merged_size_bytes=len(merged),
        audit_sha256=_sha256(audit_bytes),
        merge_contract_id="contract-v2",
        source_map_sha256={"docling": _sha256(source_raw)},
    )
    assert payload["ranges"][-1]["byte_end"] == len(merged)
    assert all(item["page_number"] == 2 for item in payload["ranges"])
    assert sum(summary["byte_counts_by_method"].values()) == len(merged)


def test_cross_page_heading_and_emphasis_delimiters_inherit_owned_content():
    source = b"BeforeHeadingtermAfter"
    boundaries = (0, 6, 17, len(source))
    evidence_ranges = []
    for page_number, (start, end) in enumerate(
        zip(boundaries, boundaries[1:]), start=1
    ):
        evidence_ranges.append(
            {
                "byte_start": start,
                "byte_end": end,
                "page_number": page_number,
                "candidate_pages": [page_number],
                "method": "direct",
            }
        )
    source_map = json.loads(
        build_source_page_provenance_bytes(
            source="docling",
            pdf_sha256=_sha256(b"pdf"),
            native_bytes=b"native",
            markdown_bytes=source,
            expected_page_count=3,
            extractor_versions={"docling": "fixture"},
            options={},
            evidence_ranges=evidence_ranges,
            residual_reason="fixture",
        )
    )
    merged = b"Before## Heading*term*After"
    audit = [
        {"output_byte_start": 0, "output_byte_end": 6, "source": "docling", "source_byte_start": 0, "source_byte_end": 6},
        {"output_byte_start": 6, "output_byte_end": 9, "source": "deterministic_markup", "transformation": "selected_document_skeleton", "transformation_id": "heading"},
        {"output_byte_start": 9, "output_byte_end": 16, "source": "docling", "source_byte_start": 6, "source_byte_end": 13},
        {"output_byte_start": 16, "output_byte_end": 17, "source": "deterministic_markup", "transformation": "native_emphasis_projection", "transformation_id": "open"},
        {"output_byte_start": 17, "output_byte_end": 21, "source": "docling", "source_byte_start": 13, "source_byte_end": 17},
        {"output_byte_start": 21, "output_byte_end": 22, "source": "deterministic_markup", "transformation": "native_emphasis_projection", "transformation_id": "close"},
        {"output_byte_start": 22, "output_byte_end": 27, "source": "docling", "source_byte_start": 17, "source_byte_end": 22},
    ]
    events = [
        {"operation": "native_emphasis_projection", "audit_span_emitted": True, "boundary": "open", "transformation_id": "open"},
        {"operation": "native_emphasis_projection", "audit_span_emitted": True, "boundary": "close", "transformation_id": "close"},
    ]

    projected = project_merged_page_ranges(
        merged_bytes=merged,
        audit=audit,
        source_maps={"docling": source_map},
        transformation_events=events,
    )

    by_operation = {
        (item.get("operation"), item.get("_transformation_id")): item
        for item in projected
        if item["source"] == "deterministic_markup"
    }
    assert by_operation[("selected_document_skeleton", "heading")]["page_number"] == 2
    assert by_operation[("native_emphasis_projection", "open")]["page_number"] == 2
    assert by_operation[("native_emphasis_projection", "close")]["page_number"] == 2


def test_residual_model_choice_is_bounded_and_invalid_choice_falls_back():
    projected = [
        {
            "byte_start": 0,
            "byte_end": 4,
            "page_number": None,
            "candidate_pages": [3, 2],
            "method": None,
            "source": "grobid",
            "operation": None,
            "region_id": "r1",
            "evidence_digest": None,
            "_publication_text": True,
            "_candidate_votes": {3: 2, 2: 1},
        }
    ]
    range_id = hashlib.sha256(
        json.dumps(
            {
                "byte_end": 4,
                "byte_start": 0,
                "operation": None,
                "source": "grobid",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    common = {
        "pdf_sha256": _sha256(b"pdf"),
        "expected_page_count": 3,
        "merged_bytes": b"text",
        "audit_sha256": _sha256(b"audit"),
        "merge_contract_id": "contract-v2",
        "source_map_sha256": {"grobid": _sha256(b"map")},
        "projected_ranges": projected,
    }
    selected, _summary = finalize_merged_page_provenance_bytes(
        **common,
        llm_choices={range_id: 2},
    )
    fallback, _summary = finalize_merged_page_provenance_bytes(
        **common,
        llm_choices={range_id: 1},
    )
    assert json.loads(selected)["ranges"][0]["method"] == "llm_selected"
    assert json.loads(selected)["ranges"][0]["page_number"] == 2
    assert json.loads(fallback)["ranges"][0]["method"] == "deterministic_fallback"
    assert json.loads(fallback)["ranges"][0]["page_number"] == 3


def test_page_resolution_evidence_is_bounded_and_uses_exact_byte_ranges():
    projected = [
        {
            "byte_start": 2,
            "byte_end": 6,
            "page_number": None,
            "candidate_pages": [2, 3],
            "method": None,
            "source": "grobid",
            "operation": None,
            "_publication_text": True,
            "_candidate_evidence": [
                {
                    "source": "docling",
                    "source_byte_start": 0,
                    "source_byte_end": 20,
                    "pages": [2],
                    "page_intervals": [
                        {
                            "page_number": 2,
                            "source_byte_start": 0,
                            "source_byte_end": 20,
                        }
                    ],
                }
            ],
            "_preceding_anchor": {"page_number": 2, "distance_bytes": 1},
            "_following_anchor": {"page_number": 3, "distance_bytes": 2},
        }
    ]
    batches = build_page_resolution_batches(
        merged_bytes=b"xxtextyy",
        projected_ranges=projected,
        source_bytes={"docling": b"alternative evidence"},
        max_ranges_per_batch=1,
        context_bytes=2,
        evidence_bytes_per_range=5,
    )
    item = batches[0]["ranges"][0]
    assert item["text"] == "text"
    assert item["preceding_context"] == "xx"
    assert item["following_context"] == "yy"
    assert item["alternative_evidence"][0]["excerpt"] == "alter"
    assert item["alternative_evidence"][0]["page_number"] == 2
    assert item["alternative_evidence"][0]["source_byte_start"] == 0
    assert item["alternative_evidence"][0]["source_byte_end"] == 5
    assert item["preceding_anchor"] == {"page_number": 2, "distance_bytes": 1}
    assert item["following_anchor"] == {"page_number": 3, "distance_bytes": 2}
    assert (item["preceding_context_byte_start"], item["preceding_context_byte_end"]) == (0, 2)
    assert (item["following_context_byte_start"], item["following_context_byte_end"]) == (6, 8)
    assert item["page_choices"] == [2, 3]
    with pytest.raises(ValueError, match="out of bounds"):
        partition_source_ranges(
            b"abcdef",
            [{"byte_start": 0, "byte_end": 7}],
            residual_reason="missing",
        )


def test_page_resolution_records_adjusted_utf8_excerpt_bounds():
    projected = [
        {
            "byte_start": 0,
            "byte_end": 1,
            "page_number": None,
            "candidate_pages": [2],
            "method": None,
            "source": "grobid",
            "operation": None,
            "_publication_text": True,
            "_candidate_evidence": [
                {
                    "source": "docling",
                    "page_intervals": [
                        {
                            "page_number": 2,
                            "source_byte_start": 0,
                            "source_byte_end": 4,
                        }
                    ],
                }
            ],
        }
    ]

    batch = build_page_resolution_batches(
        merged_bytes=b"x",
        projected_ranges=projected,
        source_bytes={"docling": "αβ".encode()},
        max_ranges_per_batch=1,
        context_bytes=0,
        evidence_bytes_per_range=3,
    )[0]

    evidence = batch["ranges"][0]["alternative_evidence"][0]
    assert evidence["excerpt"] == "α"
    assert evidence["source_byte_start"] == 0
    assert evidence["source_byte_end"] == len("α".encode())


def test_model_eligible_residual_text_is_split_at_utf8_boundaries():
    merged = "αβγ".encode()
    projected = [
        {
            "byte_start": 0,
            "byte_end": len(merged),
            "page_number": None,
            "candidate_pages": [1, 2],
            "method": None,
            "source": "grobid",
            "operation": None,
            "_publication_text": True,
            "_candidate_evidence": [],
        }
    ]

    bounded = bound_page_resolution_ranges(
        merged_bytes=merged,
        projected_ranges=projected,
        max_text_bytes=3,
    )
    batches = build_page_resolution_batches(
        merged_bytes=merged,
        projected_ranges=bounded,
        source_bytes={},
        max_ranges_per_batch=3,
        context_bytes=0,
        evidence_bytes_per_range=3,
    )

    assert [(item["byte_start"], item["byte_end"]) for item in bounded] == [
        (0, 2),
        (2, 4),
        (4, 6),
    ]
    assert [item["text"] for item in batches[0]["ranges"]] == ["α", "β", "γ"]


@pytest.mark.parametrize(
    "mutation",
    ("reversed", "overlap", "missing", "out_of_bounds", "wrong_summary"),
)
def test_merged_record_rejects_semantically_invalid_rebound_ranges(mutation):
    merged = b"abcd"
    projected = [
        {
            "byte_start": 0,
            "byte_end": 2,
            "page_number": 1,
            "candidate_pages": [1],
            "method": "direct",
            "source": "grobid",
            "operation": None,
            "region_id": None,
            "evidence_digest": _sha256(b"first"),
            "_publication_text": True,
            "_candidate_votes": {},
            "_candidate_evidence": [],
        },
        {
            "byte_start": 2,
            "byte_end": 4,
            "page_number": 2,
            "candidate_pages": [2],
            "method": "direct",
            "source": "docling",
            "operation": None,
            "region_id": None,
            "evidence_digest": _sha256(b"second"),
            "_publication_text": True,
            "_candidate_votes": {},
            "_candidate_evidence": [],
        },
    ]
    pdf_digest = _sha256(b"pdf")
    audit_digest = _sha256(b"audit")
    source_digests = {"grobid": _sha256(b"g"), "docling": _sha256(b"d")}
    raw, _summary = finalize_merged_page_provenance_bytes(
        pdf_sha256=pdf_digest,
        expected_page_count=2,
        merged_bytes=merged,
        audit_sha256=audit_digest,
        merge_contract_id="contract-v2",
        source_map_sha256=source_digests,
        projected_ranges=projected,
    )
    payload = json.loads(raw)
    if mutation == "reversed":
        payload["ranges"].reverse()
    elif mutation == "overlap":
        payload["ranges"][1]["byte_start"] = 1
    elif mutation == "missing":
        payload["ranges"][1]["byte_start"] = 3
    elif mutation == "out_of_bounds":
        payload["ranges"][1]["byte_end"] = 5
    else:
        payload["summary"]["byte_counts_by_method"]["direct"] = 3

    with pytest.raises(ValueError, match="range|cover|summary"):
        validate_merged_page_provenance_bytes(
            _rebind_record(payload),
            pdf_sha256=pdf_digest,
            merged_sha256=_sha256(merged),
            merged_size_bytes=len(merged),
            audit_sha256=audit_digest,
            merge_contract_id="contract-v2",
            source_map_sha256=source_digests,
        )
