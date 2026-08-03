import hashlib
import json
from pathlib import Path

import pytest

from app.services.source_contracts import SourceArtifact
from app.services.merge_artifact import (
    _validate_positive_style_overlay_receipts,
    _validate_title_selection_receipts,
    bundle_manifest_path,
    load_merge_bundle,
    persist_merge_alias,
    persist_merge_bundle,
    italic_preservation_receipt_error,
    validate_merge_artifacts,
    verify_merge_alias,
)
from app.services.model_policy import resolved_runtime_model_map
from app.services.abc_markdown_policy import abc_markdown_report
from app.services.document_skeleton import build_document_skeleton
from app.services.semantic_payload import (
    build_semantic_payload_receipt,
    semantic_payload_reader_report,
)


CONTRACT_ID = "pdfx-native-skeleton-selection"


def _empty_native_italic_receipt() -> dict:
    return {
        "policy_version": "positive-style-overlay-v1",
        "native_body_emphasis_count": 0,
        "mapped_native_body_emphasis_count": 0,
        "retained_native_body_emphasis_count": 0,
        "excluded_native_body_emphasis_count": 0,
        "native_body_exclusion_reason_counts": {},
        "native_body_evidence_reconciled": True,
        "native_evidence_ready": True,
        "native_evidence_failure_sources": [],
        "all_native_body_italics_retained": True,
        "all_protected_italics_retained": True,
        "protected_claim_ids_sha256": hashlib.sha256(b"").hexdigest(),
        "auxiliary_positive_emphasis_count": 0,
        "auxiliary_positive_outcome_counts": {},
        "canonical_output_emphasis_interval_count": 0,
        "canonical_output_existing_interval_count": 0,
        "canonical_output_new_interval_count": 0,
        "unique_mapped_plain_interval_count": 0,
        "direct_native_donor_retained_count": 0,
        "deterministic_target_count": 0,
        "model_selected_target_count": 0,
        "finite_model_selection_tie_count": 0,
        "numbered_model_eligible_tie_count": 0,
        "model_selection_call_count": 0,
        "model_selection_outcome_counts": {},
        "model_selection_candidate_count_distribution": {},
        "protected_outcome_counts": {},
        "protected_claim_counts_by_source_and_evidence": {"grobid": {}},
        "markdown_only_body_emphasis_count": 0,
        "reference_markdown_emphasis_count": 0,
        "explicit_native_reference_emphasis_count": 0,
        "native_style_emphasis_count": 0,
        "mapped_native_style_emphasis_count": 0,
        "unmapped_native_style_emphasis_count": 0,
        "source_counts": {
            "grobid": {
                "native_body_emphasis_count": 0,
                "mapped_native_body_emphasis_count": 0,
                "retained_native_body_emphasis_count": 0,
                "native_reference_emphasis_count": 0,
                "native_style_emphasis_count": 0,
                "mapped_native_style_emphasis_count": 0,
                "unmapped_native_style_emphasis_count": 0,
                "auxiliary_positive_emphasis_count": 0,
            }
        },
    }


def _case(text: str = "# Title\nBody\n"):
    artifact = SourceArtifact.from_text("grobid", text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    artifacts = {"grobid": artifact}
    abc_report = abc_markdown_report(text)
    skeleton_id = hashlib.sha256(f"skeleton:{digest}".encode()).hexdigest()
    projection_id = hashlib.sha256(f"projection:{digest}".encode()).hexdigest()
    metrics = {
        "merge_contract_id": CONTRACT_ID,
        "failed": False,
        "failure_reason": None,
        "qualification_outcome": "failsafe",
        "qualification_reasons": ["page_coverage_unverified"],
        "repetition_diagnostics": [],
        "available_extractors": ["grobid"],
        "source_artifact_digests": {"grobid": artifact.digest},
        "baseline_source": "grobid",
        "baseline_digest": artifact.digest,
        "output_digest": digest,
        "runtime_models": resolved_runtime_model_map(),
        "document_skeleton_transformations": [],
        "document_skeleton_source": "grobid",
        "document_skeleton_id": skeleton_id,
        "document_skeleton_conflict": False,
        "document_skeleton_resolution": {
            "reason": None,
            "baseline_skeleton_id": skeleton_id,
            "selected_skeleton_id": skeleton_id,
            "delivered_skeleton_id": skeleton_id,
            "outcome": "deterministic_agreement",
            "candidate_count": 1,
        },
        "document_skeleton_candidate_ids": {"grobid": skeleton_id},
        "document_skeleton_candidate_projection_ids": {
            "grobid": projection_id
        },
        "native_structure_receipt_digests": {},
        "native_structure_artifact_digests": {},
        "abc_markdown": abc_report,
        "italic_preservation": _empty_native_italic_receipt(),
        "quality_receipt_status": {
            "native_italics_valid": True,
            "native_italics_error": None,
        },
    }
    audit = [
        {
            "output_byte_start": 0,
            "output_byte_end": len(artifact.raw_utf8),
            "source": "grobid",
            "artifact_digest": artifact.digest,
            "source_byte_start": 0,
            "source_byte_end": len(artifact.raw_utf8),
        }
    ]
    semantic_receipt = build_semantic_payload_receipt(
        text,
        audit,
        baseline_source="grobid",
        skeletons={"grobid": build_document_skeleton(artifact, None)},
    )
    metrics["semantic_payload_receipt"] = semantic_receipt.as_metric()
    metrics["semantic_payload_reader"] = semantic_payload_reader_report(
        text,
        semantic_receipt,
        validator_report=abc_report,
    )
    return artifacts, metrics, audit


def _rebind_output_receipts(
    metrics: dict,
    text: str,
    artifacts: dict,
    audit: list,
) -> None:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    report = abc_markdown_report(text)
    metrics["output_digest"] = digest
    metrics["abc_markdown"] = report
    semantic_receipt = build_semantic_payload_receipt(
        text,
        audit,
        baseline_source="grobid",
        skeletons={
            "grobid": build_document_skeleton(artifacts["grobid"], None)
        },
    )
    metrics["semantic_payload_receipt"] = semantic_receipt.as_metric()
    metrics["semantic_payload_reader"] = semantic_payload_reader_report(
        text,
        semantic_receipt,
        validator_report=report,
    )


def _load_expectations(metrics: dict) -> dict:
    return {
        "expected_native_structure_receipt_digests": metrics[
            "native_structure_receipt_digests"
        ],
        "expected_skeleton_candidate_ids": metrics[
            "document_skeleton_candidate_ids"
        ],
        "expected_skeleton_candidate_projection_ids": metrics[
            "document_skeleton_candidate_projection_ids"
        ],
    }


def test_validation_accepts_exact_source_bytes_and_receipts():
    text = "# Title\nBody\n"
    artifacts, metrics, audit = _case(text)
    assert validate_merge_artifacts(
        text,
        metrics,
        audit,
        artifacts=artifacts,
        expected_contract_id=CONTRACT_ID,
    ) == metrics["output_digest"]


def test_unreconciled_native_italic_receipt_delivers_only_as_bound_failsafe():
    text = "# Title\nBody\n"
    artifacts, metrics, audit = _case(text)
    metrics["italic_preservation"].update(
        native_body_emphasis_count=1,
        mapped_native_body_emphasis_count=1,
    )
    metrics["italic_preservation"]["source_counts"]["grobid"].update(
        native_body_emphasis_count=1,
        mapped_native_body_emphasis_count=1,
    )

    error = italic_preservation_receipt_error(metrics)
    metrics["quality_receipt_status"] = {
        "native_italics_valid": False,
        "native_italics_error": error,
    }
    metrics["qualification_reasons"].append("native_italic_receipt_invalid")

    assert validate_merge_artifacts(
        text,
        metrics,
        audit,
        artifacts=artifacts,
        expected_contract_id=CONTRACT_ID,
    ) == metrics["output_digest"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda metrics, audit: metrics.update(runtime_models={}), "model receipt"),
        (
            lambda metrics, audit: metrics["source_artifact_digests"].update(grobid="0" * 64),
            "source digest",
        ),
        (
            lambda metrics, audit: audit[0].update(source_byte_start=1),
            "source bytes",
        ),
    ],
)
def test_validation_rejects_unbound_receipts_or_provenance(mutation, message):
    text = "# Title\nBody\n"
    artifacts, metrics, audit = _case(text)
    mutation(metrics, audit)
    with pytest.raises(ValueError, match=message):
        validate_merge_artifacts(
            text,
            metrics,
            audit,
            artifacts=artifacts,
            expected_contract_id=CONTRACT_ID,
        )


def test_validation_allows_only_a_deterministic_terminal_newline():
    source_text = "# Title\nBody"
    text = source_text + "\n"
    artifact = SourceArtifact.from_text("grobid", source_text)
    artifacts, metrics, _audit = _case(source_text)
    newline = b"\n"
    audit = [
        {
            "output_byte_start": 0,
            "output_byte_end": len(artifact.raw_utf8),
            "source": "grobid",
            "artifact_digest": artifact.digest,
            "source_byte_start": 0,
            "source_byte_end": len(artifact.raw_utf8),
        },
        {
            "output_byte_start": len(artifact.raw_utf8),
            "output_byte_end": len(artifact.raw_utf8) + 1,
            "source": "deterministic_markup",
            "artifact_digest": hashlib.sha256(newline).hexdigest(),
            "source_byte_start": 0,
            "source_byte_end": 1,
            "transformation": "trailing_newline_normalization",
        },
    ]
    _rebind_output_receipts(metrics, text, artifacts, audit)
    validate_merge_artifacts(
        text,
        metrics,
        audit,
        artifacts=artifacts,
        expected_contract_id=CONTRACT_ID,
    )
    audit[1]["transformation"] = "authored_heading"
    with pytest.raises(ValueError, match="deterministic markup provenance"):
        validate_merge_artifacts(
            text,
            metrics,
            audit,
            artifacts=artifacts,
            expected_contract_id=CONTRACT_ID,
        )


def test_validation_rejects_reusing_the_same_source_occurrence():
    source = "Body\n"
    text = source + source
    artifacts, metrics, _audit = _case(source)
    width = len(source.encode("utf-8"))
    audit = [
        {
            "output_byte_start": index * width,
            "output_byte_end": (index + 1) * width,
            "source": "grobid",
            "artifact_digest": artifacts["grobid"].digest,
            "source_byte_start": 0,
            "source_byte_end": width,
        }
        for index in range(2)
    ]
    _rebind_output_receipts(metrics, text, artifacts, audit)

    with pytest.raises(ValueError, match="reuses a source occurrence"):
        validate_merge_artifacts(
            text,
            metrics,
            audit,
            artifacts=artifacts,
            expected_contract_id=CONTRACT_ID,
        )


def test_validation_rejects_skeletal_qualified_receipts():
    text = "# Title\nBody\n"
    artifacts, metrics, audit = _case(text)
    metrics["qualification_outcome"] = "qualified"
    metrics["qualification_reasons"] = []

    with pytest.raises(ValueError, match="qualified merge receipts"):
        validate_merge_artifacts(
            text,
            metrics,
            audit,
            artifacts=artifacts,
            expected_contract_id=CONTRACT_ID,
        )


def test_validation_rejects_nonempty_repetition_receipt():
    text = "# Title\nBody\n"
    artifacts, metrics, audit = _case(text)
    metrics["repetition_diagnostics"] = [{"kind": "paragraph"}]

    with pytest.raises(ValueError, match="excess repetition"):
        validate_merge_artifacts(
            text,
            metrics,
            audit,
            artifacts=artifacts,
            expected_contract_id=CONTRACT_ID,
        )


def test_validation_rejects_error_bearing_failsafe_receipt():
    text = "# First title\n\n# Second title\n"
    artifacts, metrics, audit = _case(text)
    assert metrics["qualification_outcome"] == "failsafe"
    assert metrics["abc_markdown"]["error_rule_ids"]

    with pytest.raises(ValueError, match="error-bearing"):
        validate_merge_artifacts(
            text,
            metrics,
            audit,
            artifacts=artifacts,
            expected_contract_id=CONTRACT_ID,
        )


def test_manifest_last_bundle_round_trip_and_tamper_detection(tmp_path):
    text = "# Title\nBody\n"
    artifacts, metrics, audit = _case(text)
    merged = tmp_path / "merged.md"
    metric_file = tmp_path / "metrics.json"
    audit_file = tmp_path / "audit.json"
    manifest = persist_merge_bundle(
        merged_path=str(merged),
        metrics_path=str(metric_file),
        audit_path=str(audit_file),
        text=text,
        metrics=metrics,
        audit=audit,
        artifacts=artifacts,
        skeletons={
            source: build_document_skeleton(artifact, None)
            for source, artifact in artifacts.items()
        },
        expected_contract_id=CONTRACT_ID,
    )
    assert manifest == bundle_manifest_path(merged)
    assert load_merge_bundle(
        merged_path=str(merged),
        metrics_path=str(metric_file),
        audit_path=str(audit_file),
        artifacts=artifacts,
        expected_contract_id=CONTRACT_ID,
        **_load_expectations(metrics),
    ) == (text, metrics, audit)

    with pytest.raises(ValueError, match="native skeleton identity"):
        load_merge_bundle(
            merged_path=str(merged),
            metrics_path=str(metric_file),
            audit_path=str(audit_file),
            artifacts=artifacts,
            expected_contract_id=CONTRACT_ID,
            expected_native_structure_receipt_digests={"grobid": "0" * 64},
            expected_skeleton_candidate_ids=metrics[
                "document_skeleton_candidate_ids"
            ],
            expected_skeleton_candidate_projection_ids=metrics[
                "document_skeleton_candidate_projection_ids"
            ],
        )
    with pytest.raises(ValueError, match="native skeleton identity"):
        load_merge_bundle(
            merged_path=str(merged),
            metrics_path=str(metric_file),
            audit_path=str(audit_file),
            artifacts=artifacts,
            expected_contract_id=CONTRACT_ID,
            expected_native_structure_receipt_digests={},
            expected_skeleton_candidate_ids={"grobid": "0" * 64},
            expected_skeleton_candidate_projection_ids=metrics[
                "document_skeleton_candidate_projection_ids"
            ],
        )
    with pytest.raises(ValueError, match="native skeleton identity"):
        load_merge_bundle(
            merged_path=str(merged),
            metrics_path=str(metric_file),
            audit_path=str(audit_file),
            artifacts=artifacts,
            expected_contract_id=CONTRACT_ID,
            expected_native_structure_receipt_digests={},
            expected_skeleton_candidate_ids=metrics[
                "document_skeleton_candidate_ids"
            ],
            expected_skeleton_candidate_projection_ids={"grobid": "0" * 64},
        )
    with pytest.raises(ValueError, match="manifest identity"):
        load_merge_bundle(
            merged_path=str(merged),
            metrics_path=str(metric_file),
            audit_path=str(audit_file),
            artifacts=artifacts,
            expected_contract_id="pdfx-source-selection",
            **_load_expectations(metrics),
        )
    merged.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest digest"):
        load_merge_bundle(
            merged_path=str(merged),
            metrics_path=str(metric_file),
            audit_path=str(audit_file),
            artifacts=artifacts,
            expected_contract_id=CONTRACT_ID,
            **_load_expectations(metrics),
        )


def test_manifest_binds_the_exact_source_generation(tmp_path):
    text = "# Title\nBody\n"
    artifacts, metrics, audit = _case(text)
    merged = tmp_path / "merged.md"
    metric_file = tmp_path / "metrics.json"
    audit_file = tmp_path / "audit.json"
    persist_merge_bundle(
        merged_path=str(merged), metrics_path=str(metric_file),
        audit_path=str(audit_file), text=text, metrics=metrics, audit=audit,
        artifacts=artifacts,
        skeletons={
            source: build_document_skeleton(artifact, None)
            for source, artifact in artifacts.items()
        },
        expected_contract_id=CONTRACT_ID,
    )
    newer = {"grobid": SourceArtifact.from_text("grobid", text + "changed")}
    with pytest.raises(ValueError, match="source digest"):
        load_merge_bundle(
            merged_path=str(merged), metrics_path=str(metric_file),
            audit_path=str(audit_file), artifacts=newer,
            expected_contract_id=CONTRACT_ID,
            **_load_expectations(metrics),
        )


def test_alias_commit_is_digest_checked(tmp_path):
    text = "# Title\nBody\n"
    artifacts, metrics, audit = _case(text)
    merged = tmp_path / "merged.md"
    manifest = persist_merge_bundle(
        merged_path=str(merged), metrics_path=str(tmp_path / "metrics.json"),
        audit_path=str(tmp_path / "audit.json"), text=text, metrics=metrics,
        audit=audit,
        artifacts=artifacts,
        skeletons={
            source: build_document_skeleton(artifact, None)
            for source, artifact in artifacts.items()
        },
        expected_contract_id=CONTRACT_ID,
    )
    alias = tmp_path / "alias.md"
    persist_merge_alias(str(alias), text, metrics, bundle_manifest_path=manifest)
    assert verify_merge_alias(str(alias)) == text.encode()
    commit = json.loads((tmp_path / "alias.md.commit.json").read_text())
    commit["output_sha256"] = "0" * 64
    (tmp_path / "alias.md.commit.json").write_text(json.dumps(commit))
    with pytest.raises(ValueError, match="alias commit"):
        verify_merge_alias(str(alias))


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_alias_commit_requires_its_exact_bundle_manifest(tmp_path, mutation):
    text = "# Title\nBody\n"
    artifacts, metrics, audit = _case(text)
    merged = tmp_path / "merged.md"
    manifest = persist_merge_bundle(
        merged_path=str(merged),
        metrics_path=str(tmp_path / "metrics.json"),
        audit_path=str(tmp_path / "audit.json"),
        text=text,
        metrics=metrics,
        audit=audit,
        artifacts=artifacts,
        skeletons={
            source: build_document_skeleton(artifact, None)
            for source, artifact in artifacts.items()
        },
        expected_contract_id=CONTRACT_ID,
    )
    alias = tmp_path / "alias.md"
    persist_merge_alias(str(alias), text, metrics, bundle_manifest_path=manifest)
    if mutation == "missing":
        Path(manifest).unlink()
    else:
        Path(manifest).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="alias manifest"):
        verify_merge_alias(str(alias))


def test_positive_style_selection_must_match_its_numeric_sol_trace():
    artifact = SourceArtifact.from_text("grobid", "Body\n")
    audit = [{
        "output_byte_start": 0,
        "output_byte_end": len(artifact.raw_utf8),
        "source": "grobid",
        "artifact_digest": artifact.digest,
        "source_byte_start": 0,
        "source_byte_end": len(artifact.raw_utf8),
    }]
    selection_id = "style-selection"
    request_sha256 = "a" * 64
    event = {
        "operation": "native_emphasis_projection",
        "outcome": "supported",
        "positive_style_claim_id": "b" * 64,
        "style_selection_id": selection_id,
        "style_selection_method": "sol_numbered_choice",
        "style_selection_candidate_ids": ["target"],
        "style_selection_candidate_count": 1,
        "style_selection_none_id": "none",
        "style_selected_candidate_id": "target",
        "style_selection_request_sha256": request_sha256,
        "style_selection_response_choice": 1,
        "style_selection_model": "gpt-5.6-sol",
        "style_selection_reasoning_effort": "high",
        "model_selected_target": True,
    }
    trace = {
        "tier": "sol",
        "call_type": "bounded_id_selection_sol",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "outcome": "valid",
        "request": {
            "request_sha256": request_sha256,
            "regions": [{
                "region_id": selection_id,
                "keep_baseline_choice": 0,
                "baseline_candidate_id": "none",
                "candidates": [
                    {"candidate_id": "none"},
                    {"candidate_id": "target"},
                ],
                "path_choices": [{"choice": 1, "candidate_ids": ["target"]}],
            }],
        },
        "response": {
            "request_sha256": request_sha256,
            "decisions": [{"region_id": selection_id, "choice": 1}],
        },
    }
    metrics = {
        "document_skeleton_transformations": [event],
        "model_selection_calls": [trace],
    }

    _validate_positive_style_overlay_receipts(
        artifact.raw_utf8,
        audit,
        metrics,
        {"grobid": artifact},
        None,
    )

    trace["response"]["decisions"][0]["choice"] = 0
    with pytest.raises(ValueError, match="numeric-choice trace is malformed"):
        _validate_positive_style_overlay_receipts(
            artifact.raw_utf8,
            audit,
            metrics,
            {"grobid": artifact},
            None,
        )


def test_title_selection_must_match_its_numeric_sol_trace():
    selection_id = "document_title_choice:" + "b" * 24
    request_sha256 = "a" * 64
    event = {
        "operation": "alliance_model_title_selection",
        "audit_span_emitted": False,
        "title_selection_id": selection_id,
        "title_selection_method": "sol_numbered_choice",
        "title_selection_candidate_ids": ["title-001"],
        "title_selection_candidate_count": 1,
        "title_selection_none_id": "title-none",
        "title_selected_candidate_id": "title-001",
        "title_selection_request_sha256": request_sha256,
        "title_selection_response_choice": 1,
        "title_selection_model": "gpt-5.6-sol",
        "title_selection_reasoning_effort": "high",
        "outcome": "selected",
    }
    trace = {
        "tier": "sol",
        "call_type": "bounded_id_selection_sol",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "outcome": "valid",
        "request": {
            "request_sha256": request_sha256,
            "regions": [{
                "region_id": selection_id,
                "keep_baseline_choice": 0,
                "baseline_candidate_id": "title-none",
                "candidates": [
                    {"candidate_id": "title-none"},
                    {"candidate_id": "title-001"},
                ],
                "path_choices": [
                    {"choice": 1, "candidate_ids": ["title-001"]}
                ],
            }],
        },
        "response": {
            "request_sha256": request_sha256,
            "decisions": [{"region_id": selection_id, "choice": 1}],
        },
    }
    metrics = {
        "document_skeleton_transformations": [event],
        "model_selection_calls": [trace],
    }

    _validate_title_selection_receipts(metrics)

    trace["response"]["decisions"][0]["choice"] = 0
    with pytest.raises(ValueError, match="title numeric-choice trace is malformed"):
        _validate_title_selection_receipts(metrics)
