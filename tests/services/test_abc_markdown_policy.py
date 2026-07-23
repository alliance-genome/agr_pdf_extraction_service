from pathlib import Path
import hashlib

from agr_abc_document_parsers import emit_markdown, read_markdown

from app.services import abc_markdown_policy


def _canonical_document() -> str:
    source = (
        "# Example title\n\n"
        "## Abstract\n\n"
        "Gene *dpp* is active.\n\n"
        "## Results\n\n"
        "Result text.\n"
    )
    return emit_markdown(read_markdown(source))


def test_abc_report_accepts_exact_stable_canonical_bytes():
    text = _canonical_document()

    report = abc_markdown_policy.abc_markdown_report(text)

    assert report["parser_version"] == "1.6.0"
    assert report["parser_version_exact"] is True
    assert (
        report["parser_implementation_sha256"]
        == abc_markdown_policy.ABC_PARSER_IMPLEMENTATION_SHA256
    )
    assert report["parser_implementation_exact"] is True
    assert report["validated_output_sha256"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    assert report["valid"] is True
    assert report["error_rule_ids"] == []
    assert report["warning_rule_ids"] == []
    assert report["canonical_round_trip_stable"] is True
    assert report["canonical_bytes_match"] is True
    assert report["validator_clean"] is True
    assert "dpp" not in repr(report)


def test_abc_report_exposes_warning_without_defeating_safe_delivery():
    text = _canonical_document().rstrip("\n")

    report = abc_markdown_policy.abc_markdown_report(text)

    assert report["valid"] is True
    assert "S09" in report["warning_rule_ids"]
    assert report["validator_clean"] is False
    assert abc_markdown_policy.hard_abc_validation_reasons(text) == ()


def test_abc_errors_are_hard_final_validation_failures():
    text = "# First\n\n# Second\n\nText.\n"

    report = abc_markdown_policy.abc_markdown_report(text)
    reasons = abc_markdown_policy.hard_abc_validation_reasons(text)

    assert report["valid"] is False
    assert "S01" in report["error_rule_ids"]
    assert "abc_validation_error:S01" in reasons


def test_runtime_parser_version_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(
        abc_markdown_policy,
        "runtime_abc_parser_version",
        lambda: "1.5.0",
    )

    report = abc_markdown_policy.abc_markdown_report(_canonical_document())
    reasons = abc_markdown_policy.hard_abc_validation_reasons(_canonical_document())

    assert report["parser_version_exact"] is False
    assert "abc_parser_version_mismatch" in reasons


def test_runtime_parser_implementation_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(
        abc_markdown_policy,
        "runtime_abc_parser_implementation_sha256",
        lambda: "0" * 64,
    )

    report = abc_markdown_policy.abc_markdown_report(_canonical_document())
    reasons = abc_markdown_policy.hard_abc_validation_reasons(_canonical_document())

    assert report["parser_version_exact"] is True
    assert report["parser_implementation_exact"] is False
    assert report["validator_clean"] is False
    assert "abc_parser_implementation_mismatch" in reasons


def test_round_trip_is_diagnostic_not_part_of_validator_conformance(monkeypatch):
    monkeypatch.setattr(
        abc_markdown_policy,
        "_round_trip_diagnostics",
        lambda _text: (False, False),
    )

    report = abc_markdown_policy.abc_markdown_report(_canonical_document())
    reasons = abc_markdown_policy.hard_abc_validation_reasons(_canonical_document())

    assert report["validator_clean"] is True
    assert report["canonical_round_trip_stable"] is False
    assert report["canonical_bytes_match"] is False
    assert reasons == ()


def test_runtime_requirement_pins_exact_authoritative_parser_version():
    requirements = (
        (Path(__file__).resolve().parents[2] / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert "agr-abc-document-parsers==1.6.0" in requirements
    assert not any(
        line.startswith("agr-abc-document-parsers")
        and line != "agr-abc-document-parsers==1.6.0"
        for line in requirements
    )
