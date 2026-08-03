"""Compact manifest-last persistence for one completed source-backed merge."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.services.source_contracts import SourceArtifact, SourceName, unsafe_unicode_characters
from app.services.merge_cache_policy import verify_merged_output_digest
from app.services.model_policy import (
    GPT_5_6_SOL,
    MAX_BOUNDED_TARGET_CHOICES,
    resolved_runtime_model_map,
)
from app.services.abc_markdown_policy import (
    ABC_PARSER_IMPLEMENTATION_SHA256,
    ABC_PARSER_VERSION,
)
from app.services.semantic_payload import SEMANTIC_PAYLOAD_CONTRACT_VERSION
from app.services.document_skeleton import (
    DocumentSkeleton,
    project_native_emphasis,
    reconcile_native_emphasis_fallback,
)


MANIFEST_SCHEMA = "pdfx-source-merge-manifest"
ALIAS_SCHEMA = "pdfx-source-merge-alias"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def bundle_manifest_path(merged_path: str | os.PathLike[str]) -> str:
    return f"{merged_path}.manifest.json"


def _atomic_write(path: str | os.PathLike[str], payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_audit(
    output: bytes,
    audit: Sequence[Mapping],
    artifacts: Mapping[SourceName, SourceArtifact],
) -> None:
    cursor = 0
    source_ranges: dict[tuple[str, str], list[tuple[int, int]]] = {}
    deterministic_transformations: list[str] = []
    for entry in audit:
        start = entry.get("output_byte_start")
        end = entry.get("output_byte_end")
        if type(start) is not int or type(end) is not int or start != cursor or end <= start:
            raise ValueError("merge provenance must exactly partition output bytes")
        if end > len(output):
            raise ValueError("merge provenance exceeds output bytes")
        source = entry.get("source")
        source_start = entry.get("source_byte_start")
        source_end = entry.get("source_byte_end")
        if type(source_start) is not int or type(source_end) is not int:
            raise ValueError("merge provenance source range is invalid")
        if source == "deterministic_markup":
            span = output[start:end]
            transformation = entry.get("transformation")
            if transformation == "trailing_newline_normalization":
                expected_span = b"\n"
            elif transformation == "selected_document_skeleton":
                expected_span = (
                    span
                    if 1 <= len(span) <= 6 and set(span) == {ord("#")}
                    else None
                )
            elif transformation == "alliance_table_separator":
                expected_span = (
                    span
                    if re.fullmatch(rb"\n?\|(?:---\|)+\n", span) is not None
                    else None
                )
            elif transformation == "alliance_heading_role_marker":
                expected_span = (
                    span
                    if 1 <= len(span) <= 6 and set(span) == {ord("#")}
                    else None
                )
            elif transformation == "alliance_reference_marker":
                expected_span = (
                    span
                    if re.fullmatch(rb"[1-9]\d*\. ", span) is not None
                    else None
                )
            elif transformation == "alliance_bibliography_heading_insert":
                expected_span = (
                    span
                    if re.fullmatch(rb"\n{0,2}## References\n\n", span) is not None
                    else None
                )
            elif transformation == "alliance_figure_legend_heading_insert":
                expected_span = (
                    span
                    if re.fullmatch(
                        rb"\n{0,2}## Figure Legends\n\n", span
                    )
                    is not None
                    else None
                )
            elif transformation in {
                "alliance_heading_depth",
            }:
                expected_span = (
                    span
                    if 1 <= len(span) <= 6 and set(span) == {ord("#")}
                    else None
                )
            elif transformation in {
                "alliance_figure_label_heading",
            }:
                expected_span = span if span == b"### " else None
            elif transformation == "alliance_table_label_emphasis_marker":
                expected_span = span if span == b"**" else None
            elif transformation in {
                "alliance_figure_label_caption_boundary",
                "alliance_abstract_heading_separator",
                "alliance_table_heading_boundary",
            }:
                expected_span = span if span == b"\n\n" else None
            elif transformation in {
                "alliance_reference_blank_separator",
                "alliance_front_list_block_separator",
            }:
                expected_span = span if span == b"\n" else None
            elif transformation == "alliance_orcid_url_prefix":
                expected_span = (
                    span if span == b"https://orcid.org/" else None
                )
            elif transformation == "alliance_abstract_heading_marker":
                expected_span = span if span == b"## " else None
            elif transformation == "alliance_affiliation_ordinal_marker":
                expected_span = span if span == b"." else None
            elif transformation == "alliance_article_category_marker":
                expected_span = span if span == b"**Categories:** " else None
            elif transformation == "native_emphasis_projection":
                expected_span = span if span == b"*" else None
            else:
                expected_span = None
            if (
                expected_span is None
                or (
                    transformation == "trailing_newline_normalization"
                    and deterministic_transformations.count(transformation) >= 1
                )
                or span != expected_span
                or source_start != 0
                or source_end != len(span)
                or entry.get("artifact_digest") != _sha256(span)
            ):
                raise ValueError("deterministic markup provenance is invalid")
            deterministic_transformations.append(transformation)
        else:
            if source not in artifacts:
                raise ValueError("merge provenance references an unknown source")
            artifact = artifacts[source]
            if entry.get("artifact_digest") != artifact.digest:
                raise ValueError("merge provenance source digest mismatch")
            if not 0 <= source_start < source_end <= len(artifact.raw_utf8):
                raise ValueError("merge provenance source range is invalid")
            if artifact.raw_utf8[source_start:source_end] != output[start:end]:
                raise ValueError("merge output bytes are not the recorded source bytes")
            source_ranges.setdefault((source, artifact.digest), []).append(
                (source_start, source_end)
            )
        cursor = end
    if cursor != len(output):
        raise ValueError("merge provenance does not cover all output bytes")
    for ranges in source_ranges.values():
        ranges.sort()
        if any(
            start < previous_end
            for (_, previous_end), (start, _) in zip(ranges, ranges[1:])
        ):
            raise ValueError("merge provenance reuses a source occurrence")


def _validate_qualified_metrics(metrics: Mapping, *, baseline_source: SourceName) -> None:
    """Fail closed if a claimed qualified receipt omits any release gate."""

    abc_report = metrics.get("abc_markdown")
    semantic_report = metrics.get("semantic_payload_reader")
    italic_report = metrics.get("italic_preservation")
    page_coverage = metrics.get("page_coverage_verified")
    required = (
        isinstance(abc_report, Mapping) and abc_report.get("validator_clean") is True,
        isinstance(semantic_report, Mapping)
        and semantic_report.get("reader_payload_retained") is True,
        isinstance(semantic_report, Mapping)
        and semantic_report.get("protected_italics_retained") is True,
        isinstance(italic_report, Mapping)
        and italic_report.get("all_protected_italics_retained") is True,
        isinstance(italic_report, Mapping)
        and italic_report.get("native_evidence_ready") is True,
        metrics.get("delivery_assurance") == "page_coverage_verified",
        isinstance(page_coverage, Mapping)
        and page_coverage.get(baseline_source) is True,
        metrics.get("unsafe_character_count") == 0,
        metrics.get("final_validation_passed") is True,
        metrics.get("repetition_diagnostics") == [],
        metrics.get("unresolved_region_count") == 0,
        isinstance(metrics.get("quality_receipt_status"), Mapping)
        and metrics["quality_receipt_status"].get("native_italics_valid") is True,
    )
    if not all(required):
        raise ValueError("qualified merge receipts are incomplete or unclean")


def _validate_italic_preservation_receipt(metrics: Mapping) -> None:
    receipt = metrics.get("italic_preservation")
    if not isinstance(receipt, Mapping):
        raise ValueError("native italics receipt is missing")
    integer_fields = (
        "native_body_emphasis_count",
        "mapped_native_body_emphasis_count",
        "retained_native_body_emphasis_count",
        "excluded_native_body_emphasis_count",
        "markdown_only_body_emphasis_count",
        "reference_markdown_emphasis_count",
        "explicit_native_reference_emphasis_count",
        "native_style_emphasis_count",
        "mapped_native_style_emphasis_count",
        "unmapped_native_style_emphasis_count",
        "auxiliary_positive_emphasis_count",
        "canonical_output_emphasis_interval_count",
        "canonical_output_existing_interval_count",
        "canonical_output_new_interval_count",
        "unique_mapped_plain_interval_count",
        "direct_native_donor_retained_count",
        "deterministic_target_count",
        "model_selected_target_count",
        "finite_model_selection_tie_count",
        "numbered_model_eligible_tie_count",
        "model_selection_call_count",
    )
    if receipt.get("policy_version") != "positive-style-overlay-v1" or any(
        type(receipt.get(field)) is not int or receipt[field] < 0
        for field in integer_fields
    ):
        raise ValueError("native italics receipt has invalid counters")
    total = receipt["native_body_emphasis_count"]
    retained = receipt["retained_native_body_emphasis_count"]
    excluded = receipt["excluded_native_body_emphasis_count"]
    reasons = receipt.get("native_body_exclusion_reason_counts")
    auxiliary_outcomes = receipt.get("auxiliary_positive_outcome_counts")
    source_counts = receipt.get("source_counts")
    protected_outcomes = receipt.get("protected_outcome_counts")
    model_selection_outcomes = receipt.get("model_selection_outcome_counts")
    model_selection_candidate_counts = receipt.get(
        "model_selection_candidate_count_distribution"
    )
    protected_by_source_and_evidence = receipt.get(
        "protected_claim_counts_by_source_and_evidence"
    )
    evidence_failure_sources = receipt.get("native_evidence_failure_sources")
    if (
        not isinstance(reasons, Mapping)
        or any(type(value) is not int or value < 0 for value in reasons.values())
        or sum(reasons.values()) != excluded
        or not isinstance(auxiliary_outcomes, Mapping)
        or any(
            type(value) is not int or value < 0
            for value in auxiliary_outcomes.values()
        )
        or sum(auxiliary_outcomes.values())
        != receipt["auxiliary_positive_emphasis_count"]
        or receipt["mapped_native_body_emphasis_count"] > total
        or receipt["native_style_emphasis_count"]
        != receipt["mapped_native_style_emphasis_count"]
        + receipt["unmapped_native_style_emphasis_count"]
        or retained + excluded != total
        or receipt.get("native_body_evidence_reconciled") is not True
        or type(receipt.get("native_evidence_ready")) is not bool
        or not isinstance(evidence_failure_sources, list)
        or any(not isinstance(source, str) for source in evidence_failure_sources)
        or receipt["native_evidence_ready"] is not (not evidence_failure_sources)
        or receipt.get("all_native_body_italics_retained")
        is not (retained == total)
        or receipt.get("all_protected_italics_retained")
        is not receipt.get("all_native_body_italics_retained")
        or not isinstance(protected_outcomes, Mapping)
        or any(
            type(value) is not int or value < 0
            for value in protected_outcomes.values()
        )
        or sum(protected_outcomes.values()) != total
        or protected_outcomes.get("unresolved", 0) != excluded
        or receipt["canonical_output_emphasis_interval_count"]
        != receipt["canonical_output_existing_interval_count"]
        + receipt["canonical_output_new_interval_count"]
        or receipt["deterministic_target_count"]
        + receipt["model_selected_target_count"]
        != receipt["canonical_output_emphasis_interval_count"]
        or receipt["model_selection_call_count"]
        > receipt["finite_model_selection_tie_count"]
        or receipt["numbered_model_eligible_tie_count"]
        > receipt["finite_model_selection_tie_count"]
        or not isinstance(model_selection_outcomes, Mapping)
        or any(
            outcome not in {"selected", "none", "unavailable", "not_run"}
            or type(count) is not int
            or count < 0
            for outcome, count in model_selection_outcomes.items()
        )
        or sum(model_selection_outcomes.values())
        != receipt["finite_model_selection_tie_count"]
        or not isinstance(model_selection_candidate_counts, Mapping)
        or any(
            not isinstance(candidate_count, str)
            or not candidate_count.isdigit()
            or int(candidate_count) < 1
            or type(count) is not int
            or count < 0
            for candidate_count, count in model_selection_candidate_counts.items()
        )
        or sum(model_selection_candidate_counts.values())
        != receipt["finite_model_selection_tie_count"]
        or receipt["direct_native_donor_retained_count"] > retained
        or not isinstance(receipt.get("protected_claim_ids_sha256"), str)
        or len(receipt["protected_claim_ids_sha256"]) != 64
        or not isinstance(source_counts, Mapping)
        or not isinstance(protected_by_source_and_evidence, Mapping)
    ):
        raise ValueError("native italics receipt does not reconcile")
    source_integer_fields = (
        "native_body_emphasis_count",
        "mapped_native_body_emphasis_count",
        "retained_native_body_emphasis_count",
        "native_style_emphasis_count",
        "mapped_native_style_emphasis_count",
        "unmapped_native_style_emphasis_count",
        "auxiliary_positive_emphasis_count",
    )
    if any(
        not isinstance(counts, Mapping)
        or any(
            type(counts.get(field)) is not int or counts[field] < 0
            for field in source_integer_fields
        )
        for counts in source_counts.values()
    ) or (
        sum(counts["native_body_emphasis_count"] for counts in source_counts.values())
        != total
        or sum(
            counts["mapped_native_body_emphasis_count"]
            for counts in source_counts.values()
        )
        != receipt["mapped_native_body_emphasis_count"]
        or sum(
            counts["retained_native_body_emphasis_count"]
            for counts in source_counts.values()
        )
        != retained
        or sum(
            counts["native_style_emphasis_count"]
            for counts in source_counts.values()
        )
        != receipt["native_style_emphasis_count"]
        or sum(
            counts["mapped_native_style_emphasis_count"]
            for counts in source_counts.values()
        )
        != receipt["mapped_native_style_emphasis_count"]
        or sum(
            counts["unmapped_native_style_emphasis_count"]
            for counts in source_counts.values()
        )
        != receipt["unmapped_native_style_emphasis_count"]
        or sum(
            counts["auxiliary_positive_emphasis_count"]
            for counts in source_counts.values()
        )
        != receipt["auxiliary_positive_emphasis_count"]
        or sum(
            sum(evidence_counts.values())
            for evidence_counts in protected_by_source_and_evidence.values()
            if isinstance(evidence_counts, Mapping)
        )
        != total
        or any(
            not isinstance(evidence_counts, Mapping)
            or any(
                evidence not in {"native_document", "native_style"}
                or type(count) is not int
                or count < 0
                for evidence, count in evidence_counts.items()
            )
            for evidence_counts in protected_by_source_and_evidence.values()
        )
    ):
        raise ValueError("native italics source receipt does not reconcile")


def italic_preservation_receipt_error(metrics: Mapping) -> str | None:
    """Return a stable quality-receipt diagnostic without weakening delivery checks."""

    try:
        _validate_italic_preservation_receipt(metrics)
    except ValueError as exc:
        return str(exc)
    return None


def _validate_title_selection_receipts(metrics: Mapping) -> None:
    """Bind every valid title choice to one executable Sol/high trace."""

    transformations = metrics.get("document_skeleton_transformations", ())
    title_events = [
        event
        for event in transformations
        if isinstance(event, Mapping)
        and event.get("operation") == "alliance_model_title_selection"
        and event.get("title_selection_method") == "sol_numbered_choice"
    ]
    model_call_traces = metrics.get("model_selection_calls", [])
    if not isinstance(model_call_traces, list):
        raise ValueError("model-selection trace ledger is missing")
    seen_selection_ids = set()
    for event in title_events:
        selection_id = event.get("title_selection_id")
        request_sha256 = event.get("title_selection_request_sha256")
        selected_id = event.get("title_selected_candidate_id")
        candidate_ids = event.get("title_selection_candidate_ids")
        none_id = event.get("title_selection_none_id")
        response_choice = event.get("title_selection_response_choice")
        if (
            not isinstance(selection_id, str)
            or selection_id in seen_selection_ids
            or not selection_id.startswith("document_title_choice:")
            or not isinstance(request_sha256, str)
            or len(request_sha256) != 64
            or not isinstance(selected_id, str)
            or not isinstance(none_id, str)
            or not isinstance(candidate_ids, list)
            or not candidate_ids
            or len(candidate_ids) > MAX_BOUNDED_TARGET_CHOICES
            or len(candidate_ids) != len(set(candidate_ids))
            or none_id in candidate_ids
            or selected_id not in {none_id, *candidate_ids}
            or type(response_choice) is not int
            or event.get("title_selection_model") != GPT_5_6_SOL
            or event.get("title_selection_reasoning_effort") != "high"
        ):
            raise ValueError("title numeric-choice receipt is incomplete")
        seen_selection_ids.add(selection_id)
        matching_traces = [
            trace
            for trace in model_call_traces
            if isinstance(trace, Mapping)
            and trace.get("call_type") == "bounded_id_selection_sol"
            and trace.get("tier") == "sol"
            and trace.get("model") == GPT_5_6_SOL
            and trace.get("reasoning_effort") == "high"
            and trace.get("outcome") == "valid"
            and isinstance(trace.get("request"), Mapping)
            and trace["request"].get("request_sha256") == request_sha256
        ]
        if len(matching_traces) != 1:
            raise ValueError("title numeric choice has no unique Sol trace")
        trace = matching_traces[0]
        regions = trace["request"].get("regions")
        response = trace.get("response")
        decisions = response.get("decisions") if isinstance(response, Mapping) else None
        if (
            not isinstance(regions, list)
            or len(regions) != 1
            or not isinstance(regions[0], Mapping)
            or regions[0].get("region_id") != selection_id
            or regions[0].get("baseline_candidate_id") != none_id
            or regions[0].get("keep_baseline_choice") != 0
            or not isinstance(decisions, list)
            or len(decisions) != 1
            or decisions[0].get("region_id") != selection_id
            or decisions[0].get("choice") != response_choice
            or response.get("request_sha256") != request_sha256
        ):
            raise ValueError("title numeric-choice trace is malformed")
        trace_candidate_ids = [
            candidate.get("candidate_id")
            for candidate in regions[0].get("candidates", ())
            if isinstance(candidate, Mapping)
        ]
        if (
            len(trace_candidate_ids) != len(candidate_ids) + 1
            or set(trace_candidate_ids) != {none_id, *candidate_ids}
        ):
            raise ValueError("title numeric-choice candidates do not reconcile")
        numbered = {
            path.get("choice"): path.get("candidate_ids")
            for path in regions[0].get("path_choices", ())
            if isinstance(path, Mapping)
        }
        selected_from_trace = none_id
        if response_choice != 0:
            selected_path = numbered.get(response_choice)
            if not isinstance(selected_path, list) or len(selected_path) != 1:
                raise ValueError("title numeric choice is not executable")
            selected_from_trace = selected_path[0]
        if selected_from_trace != selected_id:
            raise ValueError("title numeric choice does not match projection")


def _validate_positive_style_overlay_receipts(
    output: bytes,
    audit: Sequence[Mapping],
    metrics: Mapping,
    artifacts: Mapping[SourceName, SourceArtifact],
    skeletons: Mapping[SourceName, DocumentSkeleton] | None,
) -> None:
    """Replay the one final-target overlay and bind every emitted delimiter."""

    transformations = metrics.get("document_skeleton_transformations", ())
    native_events = [
        dict(event)
        for event in transformations
        if isinstance(event, Mapping)
        and event.get("operation") == "native_emphasis_projection"
    ]
    projection_audit = [
        dict(entry)
        for entry in audit
        if entry.get("transformation") == "native_emphasis_projection"
    ]
    pre_projection = b"".join(
        output[entry["output_byte_start"] : entry["output_byte_end"]]
        for entry in audit
        if entry.get("transformation") != "native_emphasis_projection"
    )
    pre_projection_audit = []
    cursor = 0
    for entry in audit:
        if entry.get("transformation") == "native_emphasis_projection":
            continue
        length = entry["output_byte_end"] - entry["output_byte_start"]
        rewritten = dict(entry)
        rewritten["output_byte_start"] = cursor
        cursor += length
        rewritten["output_byte_end"] = cursor
        pre_projection_audit.append(rewritten)
    pre_digest = _sha256(pre_projection)
    post_digest = _sha256(output)
    selection_records: dict[str, dict] = {}
    for event in native_events:
        selection_id = event.get("style_selection_id")
        if (
            not isinstance(selection_id, str)
            or event.get("style_selection_method") != "sol_numbered_choice"
            or event.get("reason") == "model_selection_unavailable"
        ):
            continue
        record = {
            key: value
            for key, value in event.items()
            if key.startswith("style_selection_")
            or key == "style_selected_candidate_id"
            or key == "model_selected_target"
        }
        previous = selection_records.setdefault(selection_id, record)
        if previous != record:
            raise ValueError("positive style model-selection receipt is inconsistent")
    model_call_traces = metrics.get("model_selection_calls", [])
    if not isinstance(model_call_traces, list):
        raise ValueError("model-selection trace ledger is missing")
    for selection_id, record in selection_records.items():
        request_sha256 = record.get("style_selection_request_sha256")
        selected_id = record.get("style_selected_candidate_id")
        candidate_ids = record.get("style_selection_candidate_ids")
        none_id = record.get("style_selection_none_id")
        response_choice = record.get("style_selection_response_choice")
        if (
            not isinstance(request_sha256, str)
            or len(request_sha256) != 64
            or not isinstance(selected_id, str)
            or not isinstance(none_id, str)
            or not isinstance(candidate_ids, list)
            or not candidate_ids
            or len(candidate_ids) > MAX_BOUNDED_TARGET_CHOICES
            or len(candidate_ids) != len(set(candidate_ids))
            or type(response_choice) is not int
            or record.get("style_selection_model") != GPT_5_6_SOL
            or record.get("style_selection_reasoning_effort") != "high"
        ):
            raise ValueError("positive style numeric-choice receipt is incomplete")
        matching_traces = [
            trace
            for trace in model_call_traces
            if isinstance(trace, Mapping)
            and trace.get("call_type") == "bounded_id_selection_sol"
            and trace.get("tier") == "sol"
            and trace.get("model") == GPT_5_6_SOL
            and trace.get("reasoning_effort") == "high"
            and trace.get("outcome") == "valid"
            and isinstance(trace.get("request"), Mapping)
            and trace["request"].get("request_sha256") == request_sha256
        ]
        if len(matching_traces) != 1:
            raise ValueError("positive style numeric choice has no unique Sol trace")
        trace = matching_traces[0]
        regions = trace["request"].get("regions")
        response = trace.get("response")
        decisions = response.get("decisions") if isinstance(response, Mapping) else None
        if (
            not isinstance(regions, list)
            or len(regions) != 1
            or not isinstance(regions[0], Mapping)
            or regions[0].get("region_id") != selection_id
            or regions[0].get("baseline_candidate_id") != none_id
            or regions[0].get("keep_baseline_choice") != 0
            or not isinstance(decisions, list)
            or len(decisions) != 1
            or decisions[0].get("region_id") != selection_id
            or decisions[0].get("choice") != response_choice
            or response.get("request_sha256") != request_sha256
        ):
            raise ValueError("positive style numeric-choice trace is malformed")
        trace_candidate_ids = [
            candidate.get("candidate_id")
            for candidate in regions[0].get("candidates", ())
            if isinstance(candidate, Mapping)
        ]
        if (
            len(trace_candidate_ids) != len(candidate_ids) + 1
            or set(trace_candidate_ids) != {none_id, *candidate_ids}
        ):
            raise ValueError("positive style numeric-choice candidates do not reconcile")
        numbered = {
            path.get("choice"): path.get("candidate_ids")
            for path in regions[0].get("path_choices", ())
            if isinstance(path, Mapping)
        }
        selected_from_trace = none_id
        if response_choice != 0:
            selected_path = numbered.get(response_choice)
            if not isinstance(selected_path, list) or len(selected_path) != 1:
                raise ValueError("positive style numeric choice is not executable")
            selected_from_trace = selected_path[0]
        if selected_from_trace != selected_id:
            raise ValueError("positive style numeric choice does not match projection")
    fallback_events = [
        event
        for event in native_events
        if event.get("reconciliation_method")
        == "baseline-fallback-style-ledger-v1"
    ]
    if fallback_events:
        if len(fallback_events) != len(native_events) or skeletons is None:
            raise ValueError("fallback style reconciliation receipt is inconsistent")
        reasons = {event.get("reason") for event in fallback_events}
        if len(reasons) != 1 or not all(
            event.get("outcome") == "declined"
            and event.get("audit_span_emitted") is False
            for event in fallback_events
        ):
            raise ValueError("fallback style reconciliation receipt is malformed")
        expected_fallback_events = reconcile_native_emphasis_fallback(
            output.decode("utf-8", errors="strict"),
            skeletons,
            artifacts,
            reason=next(iter(reasons)),
        )
        if expected_fallback_events != native_events:
            raise ValueError("fallback style reconciliation replay failed")
    elif skeletons is not None:
        recorded_style_selections: dict[str, dict] = {}
        for event in native_events:
            selection_id = event.get("style_selection_id")
            if not isinstance(selection_id, str):
                continue
            record = {
                key: value
                for key, value in event.items()
                if key == "unit_pair_ambiguous"
                or key.startswith("style_selection_")
                or key == "style_selected_candidate_id"
                or key == "model_selected_target"
            }
            if (
                event.get("style_selected_candidate_id")
                == event.get("style_selection_none_id")
            ):
                record["reason"] = event.get("reason")
            previous = recorded_style_selections.setdefault(selection_id, record)
            if previous != record:
                raise ValueError("positive style model-selection receipt is inconsistent")
        replayed_text, replayed_audit, replayed_events = project_native_emphasis(
            pre_projection.decode("utf-8", errors="strict"),
            pre_projection_audit,
            skeletons,
            artifacts,
            recorded_style_selections=recorded_style_selections,
        )
        if (
            replayed_text.encode("utf-8") != output
            or replayed_audit != list(audit)
            or replayed_events != native_events
        ):
            raise ValueError("native emphasis complete reconciliation replay failed")

    projected = [
        event
        for event in native_events
        if event.get("outcome") == "projected"
        and event.get("audit_span_emitted") is True
    ]
    projected_by_id: dict[object, list[dict]] = {}
    for event in projected:
        projected_by_id.setdefault(event.get("projection_id"), []).append(event)
    transient = {
        "boundary",
        "delimiter_output_byte_start",
        "delimiter_output_byte_end",
        "transformation_id",
    }
    for projection_id, pair in projected_by_id.items():
        if (
            not isinstance(projection_id, str)
            or len(projection_id) != 64
            or len(pair) != 2
            or {event.get("boundary") for event in pair} != {"open", "close"}
            or len(
                {
                    json.dumps(
                        {key: value for key, value in event.items() if key not in transient},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for event in pair
                }
            )
            != 1
        ):
            raise ValueError("native emphasis projection delimiter pair is invalid")
        for event in pair:
            transformation_id = event.get("transformation_id")
            delimiter_start = event.get("delimiter_output_byte_start")
            delimiter_end = event.get("delimiter_output_byte_end")
            entry = next(
                (
                    item
                    for item in projection_audit
                    if item.get("transformation_id") == transformation_id
                ),
                None,
            )
            hex_fields = (
                "positive_style_claim_id",
                "donor_artifact_digest",
                "donor_spine_occurrence_id",
                "donor_spine_placement_digest",
                "native_emphasis_occurrence_id",
                "target_artifact_digest",
                "target_occurrence_id",
                "target_visible_digest",
                "target_non_emphasis_ast_digest",
                "target_emphasis_occurrence_id",
                "unit_pair_digest",
                "character_alignment_digest",
                "target_serialization_digest",
                "pre_projection_output_sha256",
                "post_projection_output_sha256",
            )
            if (
                event.get("overlay_method")
                != "post-merge-positive-style-overlay-v1"
                or event.get("alignment_method")
                != "final-unit-shared-character-alignment-v1"
                or event.get("mapping_kind") != "final_unit_alignment"
                or event.get("target_source") != "final"
                or event.get("target_artifact_digest") != pre_digest
                or event.get("donor_source") not in artifacts
                or event.get("donor_artifact_digest")
                != artifacts[event["donor_source"]].digest
                or event.get("pre_projection_output_sha256") != pre_digest
                or event.get("post_projection_output_sha256") != post_digest
                or any(
                    not isinstance(event.get(field), str)
                    or len(event[field]) != 64
                    or any(character not in "0123456789abcdef" for character in event[field])
                    for field in hex_fields
                )
                or type(delimiter_start) is not int
                or type(delimiter_end) is not int
                or not 0 <= delimiter_start < delimiter_end <= len(output)
                or output[delimiter_start:delimiter_end] != b"*"
                or entry is None
                or entry.get("output_byte_start") != delimiter_start
                or entry.get("output_byte_end") != delimiter_end
            ):
                raise ValueError("native emphasis projection receipt is malformed")
    if len(projection_audit) != len(projected):
        raise ValueError("native emphasis projection audit does not reconcile")

    terminal_claim_ids = []
    for event in native_events:
        is_terminal = (
            event.get("outcome") in {"supported", "declined"}
            or (
                event.get("outcome") == "projected"
                and event.get("boundary") == "open"
            )
        )
        if not is_terminal:
            continue
        claim_id = event.get("positive_style_claim_id")
        if not isinstance(claim_id, str) or len(claim_id) != 64:
            raise ValueError("positive style terminal claim identity is invalid")
        terminal_claim_ids.append(claim_id)
    if len(terminal_claim_ids) != len(set(terminal_claim_ids)):
        raise ValueError("positive style claim terminal ledger is not unique")


def _validate_primary_abc_receipt(report: object, output_digest: str) -> None:
    """Require exact-parser, zero-error Alliance validity for every outcome."""

    if not isinstance(report, Mapping):
        raise ValueError("Alliance validator receipt is missing")
    required = (
        report.get("validated_output_sha256") == output_digest,
        report.get("parser_version") == ABC_PARSER_VERSION,
        report.get("parser_version_exact") is True,
        report.get("parser_implementation_sha256")
        == ABC_PARSER_IMPLEMENTATION_SHA256,
        report.get("parser_implementation_exact") is True,
        report.get("failure_code") is None,
        report.get("valid") is True,
        report.get("error_rule_ids") == [],
    )
    if not all(required):
        raise ValueError("Alliance validator receipt is invalid or error-bearing")


def _validate_semantic_payload_receipt(
    metrics: Mapping,
    output_digest: str,
    output_byte_count: int,
) -> None:
    receipt = metrics.get("semantic_payload_receipt")
    reader = metrics.get("semantic_payload_reader")
    if not isinstance(receipt, Mapping) or not isinstance(reader, Mapping):
        raise ValueError("semantic payload receipt is missing")
    occurrences = receipt.get("occurrences")
    if not isinstance(occurrences, list):
        raise ValueError("semantic payload occurrence list is invalid")
    core = {
        "contract_version": SEMANTIC_PAYLOAD_CONTRACT_VERSION,
        "output_sha256": output_digest,
        "occurrences": occurrences,
    }
    expected_receipt_digest = _sha256(
        json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    occurrence_ids = [
        item.get("occurrence_id")
        for item in occurrences
        if isinstance(item, Mapping)
    ]
    ranges = [
        (item.get("output_byte_start"), item.get("output_byte_end"))
        for item in occurrences
        if isinstance(item, Mapping)
    ]
    digest_fields = (
        "occurrence_id",
        "content_digest",
        "role_digest",
        "source_span_digest",
    )
    occurrence_shape_valid = bool(occurrences) and all(
        isinstance(item, Mapping)
        and item.get("order") == index
        and item.get("unit_type") in {
            "heading",
            "paragraph",
            "list",
            "table",
            "table_cell",
            "fenced_block",
            "equation",
            "figure_caption",
            "reference",
        }
        and all(
            isinstance(item.get(field), str)
            and len(item[field]) == 64
            and all(character in "0123456789abcdef" for character in item[field])
            for field in digest_fields
        )
        and (
            item.get("native_id_digest") is None
            or (
                isinstance(item.get("native_id_digest"), str)
                and len(item["native_id_digest"]) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in item["native_id_digest"]
                )
            )
        )
        and isinstance(item.get("emphasis_occurrence_ids"), list)
        and all(
            isinstance(identifier, str) and len(identifier) == 64
            for identifier in item["emphasis_occurrence_ids"]
        )
        and isinstance(item.get("source_bound"), bool)
        and item.get("selected_source") in {None, "grobid", "docling", "marker"}
        and type(item.get("output_byte_start")) is int
        and type(item.get("output_byte_end")) is int
        and 0 <= item["output_byte_start"] < item["output_byte_end"] <= output_byte_count
        for index, item in enumerate(occurrences)
    )
    ordered_ranges = occurrence_shape_valid and all(
        previous_end <= start
        for (_previous_start, previous_end), (start, _end) in zip(
            ranges,
            ranges[1:],
        )
    )
    receipt_ids = set(occurrence_ids)
    reader_id_fields = (
        "missing_occurrence_ids",
        "reordered_occurrence_ids",
        "role_changed_occurrence_ids",
        "formatting_lost_occurrence_ids",
        "unbound_occurrence_ids",
    )
    reader_ids_valid = all(
        isinstance(reader.get(field), list)
        and all(identifier in receipt_ids for identifier in reader[field])
        for field in reader_id_fields
    )
    semantic_roles = {
        "body",
        "tables",
        "figures",
        "back_matter",
        "references",
    }
    role_streams_valid = all(
        isinstance(reader.get(field), Mapping)
        and set(reader[field]) == semantic_roles
        and all(
            isinstance(reader[field][role], Mapping)
            and type(reader[field][role].get("token_count")) is int
            and reader[field][role]["token_count"] >= 0
            and isinstance(
                reader[field][role].get("ordered_token_sha256"), str
            )
            and len(reader[field][role]["ordered_token_sha256"]) == 64
            for role in semantic_roles
        )
        for field in ("expected_role_streams", "reader_role_streams")
    )
    role_mismatches = reader.get("role_stream_mismatch_roles")
    role_mismatches_valid = (
        isinstance(role_mismatches, list)
        and len(role_mismatches) == len(set(role_mismatches))
        and all(role in semantic_roles for role in role_mismatches)
    )
    required = (
        receipt.get("contract_version") == SEMANTIC_PAYLOAD_CONTRACT_VERSION,
        receipt.get("output_sha256") == output_digest,
        receipt.get("receipt_sha256") == expected_receipt_digest,
        receipt.get("occurrence_count") == len(occurrences),
        receipt.get("source_bound_occurrence_count")
        == sum(item.get("source_bound") is True for item in occurrences if isinstance(item, Mapping)),
        occurrence_shape_valid,
        len(occurrence_ids) == len(set(occurrence_ids)),
        ordered_ranges,
        reader.get("contract_version") == SEMANTIC_PAYLOAD_CONTRACT_VERSION,
        reader.get("validated_output_sha256") == output_digest,
        reader.get("receipt_sha256") == expected_receipt_digest,
        reader.get("required_occurrence_count") == len(occurrences),
        type(reader.get("reader_occurrence_count")) is int,
        reader.get("reader_occurrence_count", -1) >= 0,
        reader_ids_valid,
        role_streams_valid,
        role_mismatches_valid,
        all(
            type(reader.get(field)) is int and reader[field] >= 0
            for field in (
                "reader_ast_reference_count",
                "reader_ast_table_count",
                "reader_ast_figure_count",
            )
        ),
        isinstance(reader.get("diagnostic_codes"), list)
        and all(isinstance(code, str) for code in reader["diagnostic_codes"]),
        isinstance(reader.get("reader_payload_retained"), bool),
        isinstance(reader.get("protected_italics_retained"), bool),
        isinstance(reader.get("reader_contract_pass"), bool),
        reader.get("reader_contract_pass")
        == (
            reader.get("reader_payload_retained")
            and reader.get("protected_italics_retained")
        ),
        reader.get("reader_payload_retained")
        == (not bool(role_mismatches) and not bool(reader.get("unbound_occurrence_ids"))),
    )
    if not all(required):
        raise ValueError("semantic payload receipt is invalid or unbound")


def validate_merge_artifacts(
    text: str,
    metrics: Mapping,
    audit: Sequence[Mapping],
    *,
    artifacts: Mapping[SourceName, SourceArtifact],
    expected_contract_id: str,
    skeletons: Mapping[SourceName, DocumentSkeleton] | None = None,
) -> str:
    """Validate exact output, source spans, policy receipts, and contract identity."""

    if not text.strip() or unsafe_unicode_characters(text):
        raise ValueError("merge output is empty or contains unsafe Unicode")
    output_digest = verify_merged_output_digest(text, metrics)
    if metrics.get("merge_contract_id") != expected_contract_id:
        raise ValueError("merge contract identity mismatch")
    if metrics.get("failed") is not False or metrics.get("failure_reason") is not None:
        raise ValueError("failed merge cannot be committed")
    outcome = metrics.get("qualification_outcome")
    reasons = metrics.get("qualification_reasons")
    if outcome not in {"qualified", "failsafe"} or not isinstance(reasons, list):
        raise ValueError("merge qualification outcome is invalid")
    if outcome == "qualified" and reasons:
        raise ValueError("qualified merge cannot contain qualification reasons")
    repetition_diagnostics = metrics.get("repetition_diagnostics")
    if not isinstance(repetition_diagnostics, list):
        raise ValueError("merge repetition receipt is invalid")
    if repetition_diagnostics:
        raise ValueError("merge output contains excess repetition")
    sources = sorted(artifacts)
    if metrics.get("available_extractors") != sources:
        raise ValueError("merge source set mismatch")
    expected_source_digests = {
        source: artifacts[source].digest for source in sources
    }
    if metrics.get("source_artifact_digests") != expected_source_digests:
        raise ValueError("merge source digest set mismatch")
    baseline_source = metrics.get("baseline_source")
    if baseline_source not in artifacts:
        raise ValueError("merge baseline source is unavailable")
    if metrics.get("baseline_digest") != artifacts[baseline_source].digest:
        raise ValueError("merge baseline digest mismatch")
    if outcome == "qualified":
        _validate_qualified_metrics(metrics, baseline_source=baseline_source)
    if metrics.get("runtime_models") != resolved_runtime_model_map():
        raise ValueError("merge runtime model receipt mismatch")
    _validate_primary_abc_receipt(metrics.get("abc_markdown"), output_digest)
    _validate_semantic_payload_receipt(
        metrics,
        output_digest,
        len(text.encode("utf-8")),
    )
    italic_receipt_error = italic_preservation_receipt_error(metrics)
    quality_status = metrics.get("quality_receipt_status")
    if not isinstance(quality_status, Mapping):
        raise ValueError("quality receipt status is missing")
    if quality_status.get("native_italics_valid") is True:
        if quality_status.get("native_italics_error") is not None:
            raise ValueError("quality receipt status is inconsistent")
        if italic_receipt_error is not None:
            raise ValueError(italic_receipt_error)
    elif quality_status.get("native_italics_valid") is False:
        if (
            outcome != "failsafe"
            or "native_italic_receipt_invalid" not in reasons
            or quality_status.get("native_italics_error") != italic_receipt_error
            or italic_receipt_error is None
        ):
            raise ValueError("invalid quality receipt is not bound to failsafe delivery")
    else:
        raise ValueError("quality receipt status is invalid")
    skeleton_id = metrics.get("document_skeleton_id")
    skeleton_candidates = metrics.get("document_skeleton_candidate_ids")
    if (
        not isinstance(skeleton_id, str)
        or len(skeleton_id) != 64
        or not isinstance(skeleton_candidates, Mapping)
        or skeleton_id not in skeleton_candidates.values()
        or not isinstance(metrics.get("document_skeleton_candidate_projection_ids"), Mapping)
        or not isinstance(metrics.get("native_structure_receipt_digests"), Mapping)
        or not isinstance(metrics.get("native_structure_artifact_digests"), Mapping)
        or skeleton_candidates.get(metrics.get("document_skeleton_source"))
        != skeleton_id
    ):
        raise ValueError("document skeleton receipt is invalid")
    skeleton_resolution = metrics.get("document_skeleton_resolution")
    if (
        not isinstance(skeleton_resolution, Mapping)
        or skeleton_resolution.get("baseline_skeleton_id")
        not in skeleton_candidates.values()
        or skeleton_resolution.get("selected_skeleton_id")
        not in skeleton_candidates.values()
        or skeleton_resolution.get("delivered_skeleton_id") != skeleton_id
        or skeleton_resolution.get("outcome")
        not in {
            "deterministic_agreement",
            "sol_selected_existing_id",
            "safe_retention_no_model",
            "safe_retention_sol_refusal",
            "safe_retention_sol_timeout",
            "safe_retention_no_valid_sol_selection",
            "safe_retention_invalid_existing_id",
        }
        or (
            metrics.get("document_skeleton_conflict") is True
            and skeleton_resolution.get("reason") != "skeleton_conflict"
        )
        or (
            metrics.get("document_skeleton_conflict") is not True
            and skeleton_resolution.get("reason") is not None
        )
    ):
        raise ValueError("document skeleton resolution receipt is invalid")
    _validate_audit(text.encode("utf-8"), audit, artifacts)
    transformation_events = metrics.get("document_skeleton_transformations")
    permitted_alliance_operations = {
        "selected_document_skeleton",
        "alliance_table_separator",
        "alliance_title_role_order",
        "alliance_title_composite_join",
        "alliance_model_title_selection",
        "alliance_role_container_order",
        "alliance_role_binding_unresolved",
        "alliance_heading_role_marker",
        "alliance_reference_marker",
        "alliance_bibliography_heading_remove",
        "alliance_bibliography_heading_insert",
        "alliance_bibliography_role_order",
        "alliance_figure_legend_role_order",
        "alliance_figure_legend_heading_insert",
        "alliance_figure_label_heading",
        "alliance_figure_label_caption_boundary",
        "alliance_figure_label_bold_remove",
        "alliance_figure_caption_outer_bold_remove",
        "alliance_table_label_emphasis_marker",
        "alliance_table_heading_boundary",
        "alliance_reference_blank_separator",
        "alliance_heading_depth",
        "alliance_orcid_url_prefix",
        "alliance_abstract_heading_marker",
        "alliance_abstract_heading_separator",
        "alliance_affiliation_list_marker",
        "alliance_affiliation_ordinal_marker",
        "alliance_article_category_marker",
        "alliance_front_list_block_separator",
        "native_emphasis_projection",
    }
    if not isinstance(transformation_events, list) or any(
        not isinstance(event, Mapping)
        or event.get("operation") not in permitted_alliance_operations
        for event in transformation_events
    ):
        raise ValueError("Alliance transformation receipt is invalid")
    recorded_alliance_operations = [
        event.get("transformation_id")
        for event in transformation_events
        if event.get("audit_span_emitted", True) is True
    ]
    audited_alliance_operations = [
        entry.get("transformation_id")
        for entry in audit
        if entry.get("transformation")
        in permitted_alliance_operations
    ]
    if (
        any(
            not isinstance(transformation_id, str)
            or len(transformation_id) != 64
            for transformation_id in [
                *recorded_alliance_operations,
                *audited_alliance_operations,
            ]
        )
        or sorted(recorded_alliance_operations)
        != sorted(audited_alliance_operations)
    ):
        raise ValueError("Alliance transformation receipt mismatch")
    _validate_title_selection_receipts(metrics)
    _validate_positive_style_overlay_receipts(
        text.encode("utf-8"),
        audit,
        metrics,
        artifacts,
        skeletons,
    )
    return output_digest


def persist_merge_bundle(
    *,
    merged_path: str,
    metrics_path: str,
    audit_path: str,
    text: str,
    metrics: Mapping,
    audit: Sequence[Mapping],
    artifacts: Mapping[SourceName, SourceArtifact],
    skeletons: Mapping[SourceName, DocumentSkeleton],
    expected_contract_id: str,
    alias_path: str | None = None,
) -> str:
    """Write content first and the visibility manifest last."""

    output_digest = validate_merge_artifacts(
        text,
        metrics,
        audit,
        artifacts=artifacts,
        expected_contract_id=expected_contract_id,
        skeletons=skeletons,
    )
    merged_bytes = text.encode("utf-8")
    metrics_bytes = _json_bytes(dict(metrics))
    audit_bytes = _json_bytes(list(audit))
    _atomic_write(merged_path, merged_bytes)
    _atomic_write(metrics_path, metrics_bytes)
    _atomic_write(audit_path, audit_bytes)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "contract_id": expected_contract_id,
        "output_filename": Path(merged_path).name,
        "output_sha256": output_digest,
        "metrics_filename": Path(metrics_path).name,
        "metrics_sha256": _sha256(metrics_bytes),
        "audit_filename": Path(audit_path).name,
        "audit_sha256": _sha256(audit_bytes),
        "source_artifact_digests": {
            source: artifacts[source].digest for source in sorted(artifacts)
        },
        "native_structure_receipt_digests": dict(
            metrics["native_structure_receipt_digests"]
        ),
        "document_skeleton_candidate_ids": dict(
            metrics["document_skeleton_candidate_ids"]
        ),
        "document_skeleton_candidate_projection_ids": dict(
            metrics["document_skeleton_candidate_projection_ids"]
        ),
        "document_skeleton_id": metrics["document_skeleton_id"],
        "abc_parser_version": metrics["abc_markdown"]["parser_version"],
        "abc_parser_implementation_sha256": metrics["abc_markdown"][
            "parser_implementation_sha256"
        ],
        "semantic_payload_contract_version": SEMANTIC_PAYLOAD_CONTRACT_VERSION,
        "semantic_payload_receipt_sha256": metrics["semantic_payload_receipt"][
            "receipt_sha256"
        ],
    }
    manifest_path = bundle_manifest_path(merged_path)
    _atomic_write(manifest_path, _json_bytes(manifest))
    if alias_path:
        persist_merge_alias(
            alias_path,
            text,
            metrics,
            bundle_manifest_path=manifest_path,
        )
    return manifest_path


def load_merge_bundle(
    *,
    merged_path: str,
    metrics_path: str,
    audit_path: str,
    artifacts: Mapping[SourceName, SourceArtifact],
    expected_contract_id: str,
    expected_native_structure_receipt_digests: Mapping[SourceName, str],
    expected_skeleton_candidate_ids: Mapping[SourceName, str],
    expected_skeleton_candidate_projection_ids: Mapping[SourceName, str],
) -> tuple[str, dict, list]:
    manifest = json.loads(Path(bundle_manifest_path(merged_path)).read_text("utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("contract_id") != expected_contract_id:
        raise ValueError("merge manifest identity mismatch")
    expected_source_digests = {
        source: artifacts[source].digest for source in sorted(artifacts)
    }
    if manifest.get("source_artifact_digests") != expected_source_digests:
        raise ValueError("merge manifest source digest mismatch")
    if (
        manifest.get("native_structure_receipt_digests")
        != dict(expected_native_structure_receipt_digests)
        or manifest.get("document_skeleton_candidate_ids")
        != dict(expected_skeleton_candidate_ids)
        or manifest.get("document_skeleton_candidate_projection_ids")
        != dict(expected_skeleton_candidate_projection_ids)
        or manifest.get("abc_parser_version") != ABC_PARSER_VERSION
        or manifest.get("abc_parser_implementation_sha256")
        != ABC_PARSER_IMPLEMENTATION_SHA256
        or manifest.get("semantic_payload_contract_version")
        != SEMANTIC_PAYLOAD_CONTRACT_VERSION
    ):
        raise ValueError("merge manifest native skeleton identity mismatch")
    merged_bytes = Path(merged_path).read_bytes()
    metrics_bytes = Path(metrics_path).read_bytes()
    audit_bytes = Path(audit_path).read_bytes()
    if (
        manifest.get("output_filename") != Path(merged_path).name
        or manifest.get("metrics_filename") != Path(metrics_path).name
        or manifest.get("audit_filename") != Path(audit_path).name
        or manifest.get("output_sha256") != _sha256(merged_bytes)
        or manifest.get("metrics_sha256") != _sha256(metrics_bytes)
        or manifest.get("audit_sha256") != _sha256(audit_bytes)
    ):
        raise ValueError("merge manifest digest mismatch")
    text = merged_bytes.decode("utf-8")
    metrics = json.loads(metrics_bytes)
    audit = json.loads(audit_bytes)
    if (
        metrics.get("native_structure_receipt_digests")
        != dict(expected_native_structure_receipt_digests)
        or metrics.get("document_skeleton_candidate_ids")
        != dict(expected_skeleton_candidate_ids)
        or metrics.get("document_skeleton_candidate_projection_ids")
        != dict(expected_skeleton_candidate_projection_ids)
        or metrics.get("document_skeleton_id") != manifest.get("document_skeleton_id")
        or metrics.get("semantic_payload_receipt", {}).get("receipt_sha256")
        != manifest.get("semantic_payload_receipt_sha256")
    ):
        raise ValueError("merge metrics native skeleton identity mismatch")
    validate_merge_artifacts(
        text,
        metrics,
        audit,
        artifacts=artifacts,
        expected_contract_id=expected_contract_id,
    )
    return text, metrics, audit


def persist_merge_alias(
    path: str,
    text: str,
    metrics: Mapping,
    *,
    bundle_manifest_path: str,
) -> None:
    output_digest = verify_merged_output_digest(text, metrics)
    manifest_path = Path(bundle_manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("output_sha256") != output_digest
    ):
        raise ValueError("merge alias manifest mismatch")
    manifest_digest = _sha256(manifest_bytes)
    _atomic_write(path, text.encode("utf-8"))
    _atomic_write(
        f"{path}.commit.json",
        _json_bytes(
            {
                "schema": ALIAS_SCHEMA,
                "output_sha256": output_digest,
                "bundle_manifest_sha256": manifest_digest,
                "bundle_manifest_path": str(manifest_path),
            }
        ),
    )


def verify_merge_alias(path: str) -> bytes:
    payload = Path(path).read_bytes()
    commit = json.loads(Path(f"{path}.commit.json").read_text("utf-8"))
    if commit.get("schema") != ALIAS_SCHEMA or commit.get("output_sha256") != _sha256(payload):
        raise ValueError("merge alias commit mismatch")
    manifest_path = commit.get("bundle_manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ValueError("merge alias manifest path is missing")
    try:
        manifest_bytes = Path(manifest_path).read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("merge alias manifest is unavailable") from exc
    if (
        commit.get("bundle_manifest_sha256") != _sha256(manifest_bytes)
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("output_sha256") != _sha256(payload)
    ):
        raise ValueError("merge alias manifest mismatch")
    return payload
