import hashlib

import pytest

from app.services.abc_markdown_policy import abc_markdown_report
from app.services.document_skeleton import build_document_skeleton
from app.services.semantic_payload import (
    build_semantic_payload_receipt,
    semantic_payload_reader_report,
)
from app.services.source_contracts import SourceArtifact


def _document():
    return (
        "# Example study title\n\n"
        "## Abstract\n\nGene *dpp* is active.\n\n"
        "## Results\n\nFirst result.\n\nSecond result.\n\n"
        "## References\n\n1. Example reference.\n"
    )


def _receipt(text):
    artifact = SourceArtifact.from_text("grobid", text)
    audit = [{
        "output_byte_start": 0,
        "output_byte_end": len(artifact.raw_utf8),
        "source": "grobid",
        "artifact_digest": artifact.digest,
        "source_byte_start": 0,
        "source_byte_end": len(artifact.raw_utf8),
        "candidate_id": None,
        "region_id": None,
        "decision_method": "baseline_fallback",
    }]
    skeletons = {"grobid": build_document_skeleton(artifact, None)}
    return build_semantic_payload_receipt(
        text,
        audit,
        baseline_source="grobid",
        skeletons=skeletons,
    )


def test_actual_reader_retains_ordered_bound_occurrences_roles_and_italics():
    text = _document()
    receipt = _receipt(text)
    report = semantic_payload_reader_report(
        text,
        receipt,
        validator_report=abc_markdown_report(text),
    )

    assert receipt.output_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert len(receipt.occurrences) == 8
    assert all(item.source_bound for item in receipt.occurrences)
    assert report["reader_payload_retained"] is True
    assert report["protected_italics_retained"] is True
    assert report["mismatch_codes"] == []
    assert report["reader_contract_pass"] is True
    assert report["normalized_token_recall_ppm"] == 1_000_000
    assert report["normalized_token_precision_ppm"] == 1_000_000
    assert "dpp" not in repr(receipt)
    assert "dpp" not in repr(report)


@pytest.mark.parametrize(
    ("mutation", "field", "code", "gating"),
    [
        (
            lambda text: text.replace("\n\nSecond result.", ""),
            "missing_occurrence_ids",
            "reader_occurrence_missing",
            True,
        ),
        (
            lambda text: text.replace(
                "First result.\n\nSecond result.",
                "Second result.\n\nFirst result.",
            ),
            "reordered_occurrence_ids",
            "reader_occurrence_reordered",
            True,
        ),
        (
            lambda text: text.replace("## Results", "### Results"),
            "role_changed_occurrence_ids",
            "reader_occurrence_role_changed",
            True,
        ),
        (
            lambda text: text.replace("*dpp*", "dpp"),
            "formatting_lost_occurrence_ids",
            "reader_occurrence_formatting_lost",
            True,
        ),
    ],
)
def test_reader_reports_exact_occurrence_ids_for_semantic_loss(
    monkeypatch, mutation, field, code, gating
):
    text = _document()
    receipt = _receipt(text)
    monkeypatch.setattr(
        "agr_abc_document_parsers.emit_markdown",
        lambda _document: mutation(text),
    )

    report = semantic_payload_reader_report(
        text,
        receipt,
        validator_report=abc_markdown_report(text),
    )

    assert report[field]
    codes = (
        report["mismatch_codes"]
        if code == "reader_occurrence_formatting_lost"
        else report["diagnostic_codes"]
    )
    assert code in codes
    assert report["reader_contract_pass"] is (not gating)


def test_reader_role_stream_gate_accepts_canonical_table_movement(monkeypatch):
    text = (
        "# Example study title\n\n## Results\n\nBody.\n\n"
        "**Table 1.** Values.\n\n| Gene | Value |\n|---|---|\n| dpp | 1 |\n"
    )
    receipt = _receipt(text)
    canonical = (
        "# Example study title\n\n## Results\n\nBody.\n\n"
        "**Table 1.** Values.\n\n| Gene | Value |\n| --- | --- |\n| dpp | 1 |\n"
    )
    monkeypatch.setattr(
        "agr_abc_document_parsers.emit_markdown",
        lambda _document: canonical,
    )

    report = semantic_payload_reader_report(
        text,
        receipt,
        validator_report=abc_markdown_report(text),
    )

    assert report["role_stream_mismatch_roles"] == []
    assert report["reader_payload_retained"] is True


def test_reader_role_stream_gate_ignores_only_list_ordinal_canonicalization(
    monkeypatch,
):
    text = (
        "# Example study title\n\n## Results\n\n"
        "19. First retained item.\n20. Second retained item.\n"
    )
    receipt = _receipt(text)
    monkeypatch.setattr(
        "agr_abc_document_parsers.emit_markdown",
        lambda _document: text.replace("19.", "1.").replace("20.", "2."),
    )

    report = semantic_payload_reader_report(
        text,
        receipt,
        validator_report=abc_markdown_report(text),
    )

    assert report["role_stream_mismatch_roles"] == []
    assert report["reader_payload_retained"] is True


def test_reader_role_stream_gate_accepts_only_lossless_front_matter_reordering():
    text = (
        "# Title\n\nAuthors One, Authors Two\n\n"
        "Unstructured affiliation.\n\n## Abstract\n\nSummary.\n\n"
        "## Results\n\nBody.\n"
    )
    reordered = text.replace(
        "Unstructured affiliation.\n\n## Abstract\n\nSummary.",
        "## Abstract\n\nSummary.\n\nUnstructured affiliation.",
    )

    from app.services.semantic_payload import _markdown_role_streams

    assert _markdown_role_streams(text) == _markdown_role_streams(reordered)
    dropped = reordered.replace("Unstructured affiliation.\n\n", "")
    assert _markdown_role_streams(text) != _markdown_role_streams(dropped)


def test_reader_role_stream_gate_accepts_secondary_abstract_relocation_only_when_lossless():
    text = (
        "# Title\n\n## Introduction\n\nBody.\n\n"
        "## Author summary\n\nAccessible summary.\n"
    )
    relocated = (
        "# Title\n\n## Author summary\n\nAccessible summary.\n\n"
        "## Introduction\n\nBody.\n"
    )

    from app.services.semantic_payload import _markdown_role_streams

    assert _markdown_role_streams(text) == _markdown_role_streams(relocated)
    assert _markdown_role_streams(text) != _markdown_role_streams(
        relocated.replace("Accessible summary.\n\n", "")
    )


def test_reader_role_stream_gate_ignores_empty_back_heading_but_not_back_content():
    text = (
        "# Title\n\n## Results\n\nBody.\n\n"
        "## Author Contributions\n\n## Competing Interests\n\nNone.\n"
    )
    reader_text = text.replace("## Author Contributions\n\n", "")

    from app.services.semantic_payload import _markdown_role_streams

    assert _markdown_role_streams(text) == _markdown_role_streams(reader_text)
    assert _markdown_role_streams(text) != _markdown_role_streams(
        reader_text.replace("None.\n", "")
    )


@pytest.mark.parametrize(
    "text,reader_text,missing_role",
    [
        (
            "# Title\n\n## References\n\n1. Alpha (2024). One.\n2. Beta (2025). Two.\n",
            "# Title\n\n## References\n\n1. Alpha (2024). One.\n",
            "references",
        ),
        (
            "# Title\n\n## Results\n\n| Gene | Value |\n|---|---|\n| dpp | 1 |\n",
            "# Title\n\n## Results\n\n",
            "tables",
        ),
    ],
)
def test_reader_role_stream_gate_rejects_real_role_payload_loss(
    monkeypatch, text, reader_text, missing_role
):
    receipt = _receipt(text)
    monkeypatch.setattr(
        "agr_abc_document_parsers.emit_markdown",
        lambda _document: reader_text,
    )

    report = semantic_payload_reader_report(
        text,
        receipt,
        validator_report=abc_markdown_report(text),
    )

    assert missing_role in report["role_stream_mismatch_roles"]
    assert report["reader_payload_retained"] is False
