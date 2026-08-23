import json
import hashlib

import pytest

from app.services.native_extractor_artifact import (
    has_valid_native_extractor_artifact,
    load_native_extractor_artifact,
    native_artifact_path,
    native_manifest_path,
    native_style_artifact_path,
    persist_native_extractor_artifact,
)
from app.services.native_style import unavailable_native_style_bytes
from app.services.page_provenance import (
    build_source_page_provenance_bytes,
    source_page_provenance_path,
)


def _page_map(pdf, markdown, native, source, versions, options, page_count):
    return build_source_page_provenance_bytes(
        source=source,
        pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        native_bytes=native,
        markdown_bytes=markdown.read_bytes(),
        expected_page_count=page_count,
        extractor_versions=versions,
        options=options,
        evidence_ranges=[],
        residual_reason="fixture",
    )


def test_native_artifact_is_manifest_bound_to_pdf_and_markdown(tmp_path):
    pdf = tmp_path / "paper.pdf"
    markdown = tmp_path / "docling.md"
    pdf.write_bytes(b"pdf bytes")
    markdown.write_text("# Title\n\n*Gene*", encoding="utf-8")

    native = b'{"document":"exact"}'
    versions = {"docling": "2.113.0", "docling-core": "2.87.1"}
    options = {
        "do_ocr": True,
        "generate_parsed_pages": True,
        "native_style_cell_collection": "word_cells",
        "native_style_sidecar": True,
        "page_provenance": "digest_sentinel_v1",
    }
    manifest = persist_native_extractor_artifact(
        source="docling",
        output_filename=markdown,
        native_bytes=native,
        native_media_type="application/json",
        pdf_path=pdf,
        extractor_versions=versions,
        options=options,
        page_provenance_bytes=_page_map(
            pdf, markdown, native, "docling", versions, options, 2
        ),
        expected_page_count=2,
        covered_pages=[1, 2],
        native_style_bytes=unavailable_native_style_bytes("docling", "fixture"),
    )

    loaded, native = load_native_extractor_artifact(
        source="docling",
        output_filename=markdown,
        expected_pdf_sha256=manifest["pdf_sha256"],
    )
    assert loaded == manifest
    assert loaded["page_coverage_status"] == "complete"
    assert native == b'{"document":"exact"}'
    assert has_valid_native_extractor_artifact(
        source="docling", output_filename=markdown
    )

    markdown.write_text("changed", encoding="utf-8")
    assert not has_valid_native_extractor_artifact(
        source="docling", output_filename=markdown
    )


def test_missing_source_page_map_invalidates_native_cache(tmp_path):
    pdf = tmp_path / "paper.pdf"
    markdown = tmp_path / "marker.md"
    pdf.write_bytes(b"pdf bytes")
    markdown.write_text("# Title\n\nBody", encoding="utf-8")
    native = b"{}"
    versions = {"marker-pdf": "1.10.2"}
    options = {
        "disable_links": True,
        "page_provenance": "marker_paginated_v1",
    }
    persist_native_extractor_artifact(
        source="marker",
        output_filename=markdown,
        native_bytes=native,
        native_media_type="application/json",
        pdf_path=pdf,
        extractor_versions=versions,
        options=options,
        page_provenance_bytes=_page_map(
            pdf, markdown, native, "marker", versions, options, 1
        ),
        expected_page_count=1,
        covered_pages=[1],
    )
    source_page_provenance_path(markdown).unlink()

    assert not has_valid_native_extractor_artifact(
        source="marker", output_filename=markdown
    )


def test_native_artifact_records_partial_pages_and_rejects_tampering(tmp_path):
    pdf = tmp_path / "paper.pdf"
    markdown = tmp_path / "marker.md"
    pdf.write_bytes(b"pdf bytes")
    markdown.write_text("content", encoding="utf-8")

    native = b"{}"
    versions = {"marker-pdf": "1.10.2"}
    options = {
        "disable_links": True,
        "page_provenance": "marker_paginated_v1",
    }
    partial = persist_native_extractor_artifact(
        source="marker",
        output_filename=markdown,
        native_bytes=native,
        native_media_type="application/json",
        pdf_path=pdf,
        extractor_versions=versions,
        options=options,
        page_provenance_bytes=_page_map(
            pdf, markdown, native, "marker", versions, options, 2
        ),
        expected_page_count=2,
        covered_pages=[1],
    )
    assert partial["page_coverage_status"] == "partial"

    persist_native_extractor_artifact(
        source="marker",
        output_filename=markdown,
        native_bytes=b"{}",
        native_media_type="application/json",
        pdf_path=pdf,
        extractor_versions={"marker-pdf": "1.10.2"},
        options=options,
        page_provenance_bytes=_page_map(
            pdf, markdown, native, "marker", versions, options, 1
        ),
        expected_page_count=1,
        covered_pages=[1],
    )
    native_artifact_path(markdown, "marker").write_text(
        json.dumps({"tampered": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="size mismatch|digest mismatch"):
        load_native_extractor_artifact(source="marker", output_filename=markdown)


def test_native_style_artifact_digest_tampering_is_rejected(tmp_path):
    pdf = tmp_path / "paper.pdf"
    markdown = tmp_path / "docling.md"
    pdf.write_bytes(b"pdf bytes")
    markdown.write_text("# Title\n\nBody", encoding="utf-8")
    native = b'{"schema_name":"DoclingDocument","texts":[]}'
    versions = {"docling": "2.113.0", "docling-core": "2.87.1"}
    options = {
        "do_ocr": True,
        "generate_parsed_pages": True,
        "native_style_cell_collection": "word_cells",
        "native_style_sidecar": True,
        "page_provenance": "digest_sentinel_v1",
    }
    persist_native_extractor_artifact(
        source="docling",
        output_filename=markdown,
        native_bytes=native,
        native_media_type="application/json",
        pdf_path=pdf,
        extractor_versions=versions,
        options=options,
        page_provenance_bytes=_page_map(
            pdf, markdown, native, "docling", versions, options, 1
        ),
        expected_page_count=1,
        covered_pages=[1],
        native_style_bytes=unavailable_native_style_bytes("docling", "fixture"),
    )
    native_style_artifact_path(markdown).write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="size mismatch|digest mismatch"):
        load_native_extractor_artifact(
            source="docling", output_filename=markdown
        )


def test_grobid_native_manifest_labels_page_qualification_unavailable(tmp_path):
    pdf = tmp_path / "paper.pdf"
    markdown = tmp_path / "grobid.md"
    pdf.write_bytes(b"pdf bytes")
    markdown.write_text("# Title\n\nBody", encoding="utf-8")

    native = b"<TEI><text/></TEI>"
    versions = {
        "grobid": "0.8.2",
        "agr-abc-document-parsers": "1.7.0",
    }
    options = {
        "include_coordinates": True,
        "generate_ids": True,
        "native_style_sidecar": True,
        "page_provenance": "tei_coords_v1",
    }
    manifest = persist_native_extractor_artifact(
        source="grobid",
        output_filename=markdown,
        native_bytes=native,
        native_media_type="application/tei+xml",
        pdf_path=pdf,
        extractor_versions=versions,
        options=options,
        page_provenance_bytes=_page_map(
            pdf, markdown, native, "grobid", versions, options, 2
        ),
        native_style_bytes=unavailable_native_style_bytes("grobid", "fixture"),
    )

    assert manifest["expected_page_count"] is None
    assert manifest["covered_pages"] is None
    assert manifest["page_coverage_status"] == "unavailable"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("contract_version", "old-native-contract", "contract version"),
        ("native_media_type", "text/plain", "media type"),
        ("options", {"do_ocr": False}, "options"),
        (
            "options",
            {
                "do_ocr": True,
                "generate_parsed_pages": True,
                "native_style_cell_collection": "char_cells",
                "native_style_sidecar": True,
            },
            "options",
        ),
        ("pdf_sha256", "0" * 64, "PDF digest"),
        ("extractor_versions", {"docling": "2.113.0"}, "runtime pins"),
        (
            "extractor_versions",
            {"docling": "0.0.0", "docling-core": "2.87.1"},
            "runtime pins",
        ),
    ],
)
def test_native_manifest_rejects_contract_config_pdf_and_version_drift(
    tmp_path, field, value, message
):
    pdf = tmp_path / "paper.pdf"
    markdown = tmp_path / "docling.md"
    pdf.write_bytes(b"exact pdf")
    markdown.write_text("# Title\n\nBody.", encoding="utf-8")
    native = b'{"schema_name":"DoclingDocument","texts":[]}'
    versions = {"docling": "2.113.0", "docling-core": "2.87.1"}
    options = {
        "do_ocr": True,
        "generate_parsed_pages": True,
        "native_style_cell_collection": "word_cells",
        "native_style_sidecar": True,
        "page_provenance": "digest_sentinel_v1",
    }
    manifest = persist_native_extractor_artifact(
        source="docling",
        output_filename=markdown,
        native_bytes=native,
        native_media_type="application/json",
        pdf_path=pdf,
        extractor_versions=versions,
        options=options,
        page_provenance_bytes=_page_map(
            pdf, markdown, native, "docling", versions, options, 1
        ),
        expected_page_count=1,
        covered_pages=[1],
        native_style_bytes=unavailable_native_style_bytes("docling", "fixture"),
    )
    payload = dict(manifest)
    payload[field] = value
    native_manifest_path(markdown).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        load_native_extractor_artifact(
            source="docling",
            output_filename=markdown,
            expected_pdf_sha256=manifest["pdf_sha256"],
        )
