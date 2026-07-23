"""Runtime policy and content-free evidence for canonical Alliance Markdown."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path


ABC_MARKDOWN_POLICY_VERSION = "alliance-abc-1.6-schema-v4"
ABC_PARSER_DISTRIBUTION = "agr-abc-document-parsers"
ABC_PARSER_VERSION = "1.6.0"
ABC_PARSER_IMPLEMENTATION_SHA256 = (
    "34e5ddfb6b8549648b04c3aa8e7355769dde89c08abf1622d2e08ff102d932cf"
)


def runtime_abc_parser_version() -> str:
    """Return the installed parser distribution version."""

    return metadata.version(ABC_PARSER_DISTRIBUTION)


def runtime_abc_parser_implementation_sha256() -> str:
    """Hash the exact loaded parser sources and packaged schema.

    Version metadata alone does not distinguish a rebuilt or locally modified
    wheel. The digest covers the loaded package's Python sources and normative
    schema document using stable relative paths; caches and installation paths
    are intentionally excluded.
    """

    import agr_abc_document_parsers

    package_root = Path(agr_abc_document_parsers.__file__).resolve().parent
    files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and (path.suffix == ".py" or path.name == "MARKDOWN_SCHEMA.md")
    )
    if not files:
        raise RuntimeError("parser_implementation_files_missing")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(package_root).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{file_digest}  {relative}\n".encode("utf-8"))
    return digest.hexdigest()


def _round_trip_diagnostics(text: str) -> tuple[bool, bool]:
    """Return stability and exact-byte identity without changing *text*."""

    from agr_abc_document_parsers import emit_markdown, read_markdown

    emitted_once = emit_markdown(read_markdown(text))
    emitted_twice = emit_markdown(read_markdown(emitted_once))
    return emitted_once == emitted_twice, text == emitted_once


def abc_markdown_report(text: str) -> dict:
    """Validate exact bytes and report canonical round-trip properties.

    The report intentionally contains rule IDs and counts, never publication
    text or validator messages. ``validator_clean`` means the exact 1.6 parser
    implementation is installed and its authoritative validator has no errors
    or warnings. Parser round-trip stability and emitter byte identity
    remain diagnostics because
    forcing arbitrary extractor Markdown through the emitter can discard
    unsupported content. Callers may still deliver a safe baseline as an explicitly labeled
    last-resort artifact when a source-data warning cannot be repaired without
    inventing content.
    """

    report = {
        "policy_version": ABC_MARKDOWN_POLICY_VERSION,
        "parser_version": None,
        "parser_version_exact": False,
        "parser_implementation_sha256": None,
        "parser_implementation_exact": False,
        "validated_output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "valid": False,
        "error_rule_ids": [],
        "warning_rule_ids": [],
        "canonical_round_trip_stable": False,
        "canonical_bytes_match": False,
        "validator_clean": False,
        "failure_code": None,
    }
    try:
        version = runtime_abc_parser_version()
        report["parser_version"] = version
        report["parser_version_exact"] = version == ABC_PARSER_VERSION
        implementation_digest = runtime_abc_parser_implementation_sha256()
        report["parser_implementation_sha256"] = implementation_digest
        report["parser_implementation_exact"] = (
            implementation_digest == ABC_PARSER_IMPLEMENTATION_SHA256
        )

        from agr_abc_document_parsers import validate_markdown

        validation = validate_markdown(text)
        report["valid"] = bool(validation.valid and not validation.errors)
        report["error_rule_ids"] = sorted(
            {str(issue.rule_id) for issue in validation.errors}
        )
        report["warning_rule_ids"] = sorted(
            {str(issue.rule_id) for issue in validation.warnings}
        )

        (
            report["canonical_round_trip_stable"],
            report["canonical_bytes_match"],
        ) = _round_trip_diagnostics(text)
        report["validator_clean"] = bool(
            report["parser_version_exact"]
            and report["parser_implementation_exact"]
            and report["valid"]
            and not report["error_rule_ids"]
            and not report["warning_rule_ids"]
        )
    except metadata.PackageNotFoundError:
        report["failure_code"] = "parser_distribution_missing"
    except Exception as exc:
        report["failure_code"] = type(exc).__name__
    return report


def hard_abc_validation_reasons(text: str) -> tuple[str, ...]:
    """Return failures that make even last-resort publication unsafe.

    Warnings and a noncanonical-but-stable byte representation remain visible
    in the report and fail release qualification, but do not defeat the
    guaranteed-delivery baseline by themselves.
    """

    report = abc_markdown_report(text)
    reasons = []
    if not report["parser_version_exact"]:
        reasons.append("abc_parser_version_mismatch")
    if not report["parser_implementation_exact"]:
        reasons.append("abc_parser_implementation_mismatch")
    if report["failure_code"] is not None:
        reasons.append(f"abc_policy_failure:{report['failure_code']}")
    if not report["valid"]:
        reasons.extend(
            f"abc_validation_error:{rule_id}" for rule_id in report["error_rule_ids"]
        )
        if not report["error_rule_ids"]:
            reasons.append("abc_validation_failed")
    return tuple(dict.fromkeys(reasons))
