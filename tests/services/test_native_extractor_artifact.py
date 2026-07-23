import json

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


def test_native_artifact_is_manifest_bound_to_pdf_and_markdown(tmp_path):
    pdf = tmp_path / "paper.pdf"
    markdown = tmp_path / "docling.md"
    pdf.write_bytes(b"pdf bytes")
    markdown.write_text("# Title\n\n*Gene*", encoding="utf-8")

    manifest = persist_native_extractor_artifact(
        source="docling",
        output_filename=markdown,
        native_bytes=b'{"document":"exact"}',
        native_media_type="application/json",
        pdf_path=pdf,
        extractor_versions={"docling": "2.113.0", "docling-core": "2.87.1"},
        options={
            "do_ocr": True,
            "generate_parsed_pages": True,
            "native_style_cell_collection": "word_cells",
            "native_style_sidecar": True,
        },
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


def test_native_artifact_records_partial_pages_and_rejects_tampering(tmp_path):
    pdf = tmp_path / "paper.pdf"
    markdown = tmp_path / "marker.md"
    pdf.write_bytes(b"pdf bytes")
    markdown.write_text("content", encoding="utf-8")

    partial = persist_native_extractor_artifact(
        source="marker",
        output_filename=markdown,
        native_bytes=b"{}",
        native_media_type="application/json",
        pdf_path=pdf,
        extractor_versions={"marker-pdf": "1.10.2"},
        options={"disable_links": True},
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
        options={"disable_links": True},
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
    persist_native_extractor_artifact(
        source="docling",
        output_filename=markdown,
        native_bytes=b'{"schema_name":"DoclingDocument","texts":[]}',
        native_media_type="application/json",
        pdf_path=pdf,
        extractor_versions={"docling": "2.113.0", "docling-core": "2.87.1"},
        options={
            "do_ocr": True,
            "generate_parsed_pages": True,
            "native_style_cell_collection": "word_cells",
            "native_style_sidecar": True,
        },
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

    manifest = persist_native_extractor_artifact(
        source="grobid",
        output_filename=markdown,
        native_bytes=b"<TEI><text/></TEI>",
        native_media_type="application/tei+xml",
        pdf_path=pdf,
        extractor_versions={
            "grobid": "0.8.2",
            "agr-abc-document-parsers": "1.6.0",
        },
        options={
            "coordinates": True,
            "generate_ids": True,
            "native_style_sidecar": True,
        },
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
    manifest = persist_native_extractor_artifact(
        source="docling",
        output_filename=markdown,
        native_bytes=b'{"schema_name":"DoclingDocument","texts":[]}',
        native_media_type="application/json",
        pdf_path=pdf,
        extractor_versions={"docling": "2.113.0", "docling-core": "2.87.1"},
        options={
            "do_ocr": True,
            "generate_parsed_pages": True,
            "native_style_cell_collection": "word_cells",
            "native_style_sidecar": True,
        },
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
