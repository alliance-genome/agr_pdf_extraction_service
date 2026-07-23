"""Source-backed merger that exposes only application-owned choices to models."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Literal, Mapping, TYPE_CHECKING

from config import Config
from app.services.abc_markdown_policy import abc_markdown_report
from app.services.merge_artifact import (
    italic_preservation_receipt_error,
    validate_merge_artifacts,
)
from app.services.deterministic_markdown_repair import normalize_trailing_newline
from app.services.source_contracts import (
    CandidateGraph,
    CandidateStore,
    ConsensusContractError,
    RegionSelectionDecision,
    SelectionDecisionResponse,
    SourceArtifact,
    SourceName,
    unsafe_unicode_characters,
)
from app.services.llm_service import CandidateSelectionFailure
from app.services.model_policy import (
    MAX_BOUNDED_TARGET_CHOICES,
    resolved_runtime_model_map,
)
from app.services.document_skeleton import (
    DocumentSkeleton,
    NativeStructureArtifact,
    SkeletonSelection,
    build_document_skeleton,
    choose_document_skeleton,
    effective_occurrence_region,
    _projection_claim_inventory,
    project_native_emphasis,
    render_document_role_slots,
    reconcile_document_transformations,
    render_document_skeleton,
)
from app.services.semantic_payload import (
    build_semantic_payload_receipt,
    semantic_payload_reader_report,
)
from app.services.source_merge import (
    BaselineCompletionEvidence,
    BaselineRequirements,
    SelectionResolution,
    _inline_italic_profile,
    final_document_safety_reasons,
    merge_with_baseline_failsafe,
    repetition_diagnostics_metric,
    repetition_policy_metric,
    source_repetition_profiles_metric,
    scan_structural_units,
    select_baseline,
    validate_baseline_structure,
)

if TYPE_CHECKING:
    from app.services.llm_service import LLM


SelectionRiskReason = Literal[
    "protected_format_conflict",
    "protected_payload_conflict",
]

_NUMERIC_PAYLOAD_RE = re.compile(
    r"[+\-]?"
    r"(?:\d+(?:[.,:/]\d+)*|\.\d+)"
    r"(?:[eE][+\-]?\d+)?"
    r"[%‰]?"
)
_MINUS_TRANSLATION = str.maketrans(
    {
        "−": "-",
        "–": "-",
        "—": "-",
        "﹣": "-",
    }
)


def _numeric_payload(text: str) -> tuple[str, ...]:
    """Extract a Unicode-normalized, ordered numeric payload from source text."""

    normalized = unicodedata.normalize("NFKC", text).translate(_MINUS_TRANSLATION)
    normalized = "".join(
        str(unicodedata.decimal(character))
        if unicodedata.category(character) == "Nd"
        else character
        for character in normalized
    )
    return tuple(match.group(0) for match in _NUMERIC_PAYLOAD_RE.finditer(normalized))


def _selection_risk_reason(
    candidate_store: CandidateStore,
    region,
) -> SelectionRiskReason | None:
    """Classify the two broad evidence classes that require Sol directly."""

    candidate_ids = tuple(
        dict.fromkeys(
            candidate_id
            for path in region.valid_paths
            for candidate_id in path
        )
    )
    candidates = [
        candidate_store.candidate_metadata(candidate_id)
        for candidate_id in candidate_ids
    ]
    visible_digests = {
        candidate.visible_text_digest
        for candidate in candidates
        if candidate.visible_text_digest is not None
    }
    emphasis_profiles = {
        candidate.emphasis_occurrence_ids
        for candidate in candidates
    }
    if len(visible_digests) == 1 and len(emphasis_profiles) > 1:
        return "protected_format_conflict"
    if any(
        candidate.candidate_type
        in {"equation", "table_row", "table_cell", "citation"}
        for candidate in candidates
    ):
        return "protected_payload_conflict"
    if any(
        _numeric_payload(
            candidate_store.candidate_bytes(candidate.candidate_id).decode(
                "utf-8", errors="strict"
            )
        )
        for candidate in candidates
    ):
        return "protected_payload_conflict"
    return None


@dataclass
class BoundedCandidateSelector:
    """Run Terra normally and Sol once only after a typed Terra failure."""

    llm: "LLM"
    events: list[dict] = field(default_factory=list)

    @staticmethod
    def _subgraph(
        graph: CandidateGraph,
        region_ids: tuple[str, ...],
    ) -> CandidateGraph:
        wanted = set(region_ids)
        return CandidateGraph(
            regions=tuple(
                region for region in graph.regions if region.region_id in wanted
            )
        )

    def __call__(
        self,
        candidate_store: CandidateStore,
        graph: CandidateGraph,
    ) -> SelectionResolution:
        batch_plan = candidate_store.build_selection_request_batches(graph)
        regions_by_id = {region.region_id: region for region in graph.regions}
        decisions: list[RegionSelectionDecision] = []
        used_sol = False

        for request in batch_plan.requests:
            request_region_ids = tuple(
                region.region_id for region in request.regions
            )
            risk_reasons = {
                region_id: _selection_risk_reason(
                    candidate_store,
                    regions_by_id[region_id],
                )
                for region_id in request_region_ids
            }
            high_risk_ids = tuple(
                region_id
                for region_id in request_region_ids
                if risk_reasons[region_id] is not None
            )
            ordinary_ids = tuple(
                region_id
                for region_id in request_region_ids
                if risk_reasons[region_id] is None
            )

            if high_risk_ids:
                sol_graph = self._subgraph(
                    CandidateGraph(
                        regions=tuple(
                            regions_by_id[region_id]
                            for region_id in request_region_ids
                        )
                    ),
                    high_risk_ids,
                )
                try:
                    sol = self.llm.resolve_candidate_selections(
                        candidate_store,
                        sol_graph,
                        use_sol=True,
                        raise_on_failure=True,
                    )
                    decisions.extend(sol.decisions)
                    used_sol = True
                    self.events.append(
                        {
                            "tier": "sol",
                            "outcome": "valid",
                            "region_ids": high_risk_ids,
                            "reasons": {
                                region_id: risk_reasons[region_id]
                                for region_id in high_risk_ids
                            },
                        }
                    )
                except (CandidateSelectionFailure, ValueError, KeyError) as exc:
                    outcome = (
                        self._sol_failure_outcome(exc.reason)
                        if isinstance(exc, CandidateSelectionFailure)
                        else "authorization_failed"
                    )
                    self.events.append(
                        {
                            "tier": "sol",
                            "outcome": outcome,
                            "region_ids": high_risk_ids,
                            "reasons": {
                                region_id: risk_reasons[region_id]
                                for region_id in high_risk_ids
                            },
                        }
                    )

            if not ordinary_ids:
                continue
            region_ids = ordinary_ids
            batch_graph = CandidateGraph(
                regions=tuple(regions_by_id[region_id] for region_id in region_ids)
            )
            try:
                terra = self.llm.resolve_candidate_selections(
                    candidate_store,
                    batch_graph,
                    raise_on_failure=True,
                )
                self.events.append(
                    {
                        "tier": "terra",
                        "outcome": "valid",
                        "region_ids": region_ids,
                        "reasons": {
                            region_id: "ordinary_text_conflict"
                            for region_id in region_ids
                        },
                    }
                )
                decisions.extend(terra.decisions)
                continue
            except CandidateSelectionFailure as exc:
                terra_failure_reason = exc.reason
                self.events.append(
                    {
                        "tier": "terra",
                        "outcome": terra_failure_reason,
                        "region_ids": region_ids,
                        "reasons": {
                            region_id: "ordinary_text_conflict"
                            for region_id in region_ids
                        },
                    }
                )

            sol_graph = self._subgraph(batch_graph, region_ids)
            try:
                sol = self.llm.resolve_candidate_selections(
                    candidate_store,
                    sol_graph,
                    use_sol=True,
                    raise_on_failure=True,
                )
                decisions.extend(sol.decisions)
                used_sol = True
                self.events.append(
                    {
                        "tier": "sol",
                        "outcome": "valid",
                        "region_ids": region_ids,
                        "reasons": {
                            region_id: "terra_failure" for region_id in region_ids
                        },
                    }
                )
            except (CandidateSelectionFailure, ValueError, KeyError) as sol_exc:
                outcome = (
                    self._sol_failure_outcome(sol_exc.reason)
                    if isinstance(sol_exc, CandidateSelectionFailure)
                    else "authorization_failed"
                )
                self.events.append(
                    {
                        "tier": "sol",
                        "outcome": outcome,
                        "region_ids": region_ids,
                        "reasons": {
                            region_id: "terra_failure" for region_id in region_ids
                        },
                    }
                )

        if batch_plan.fallback_region_ids:
            self.events.append(
                {
                    "tier": "none",
                    "outcome": "oversized_baseline_fallback",
                    "region_ids": batch_plan.fallback_region_ids,
                }
            )

        return SelectionResolution(
            response=SelectionDecisionResponse(decisions=tuple(decisions)),
            quality="sol_selected" if used_sol else "terra_selected",
        )

    @staticmethod
    def _sol_failure_outcome(terra_shaped_reason: str) -> str:
        return {
            "terra_refusal": "sol_refusal",
            "terra_timeout": "sol_timeout",
            "no_valid_terra_selection": "no_valid_sol_selection",
        }.get(terra_shaped_reason, "sol_failure")


def completion_evidence_for_finished_artifacts(
    artifacts: Mapping[SourceName, SourceArtifact],
) -> dict[SourceName, BaselineCompletionEvidence]:
    """Bind lifecycle evidence without claiming unavailable PDF page coverage."""

    return {
        source: BaselineCompletionEvidence(
            artifact_digest=artifact.digest,
            extraction_succeeded=True,
            artifact_complete=True,
            completion_basis="synchronous_return",
        )
        for source, artifact in artifacts.items()
    }


def completion_evidence_for_runtime_artifacts(
    artifacts: Mapping[SourceName, SourceArtifact],
    *,
    pdf_path: str,
    output_paths: Mapping[SourceName, str],
) -> dict[SourceName, BaselineCompletionEvidence]:
    """Bind exact artifacts to independently agreeing extractor page counts."""

    from app.services.page_coverage import verified_runtime_page_coverage

    verified = verified_runtime_page_coverage(
        artifacts,
        pdf_path=pdf_path,
        output_paths=output_paths,
    )
    evidence = completion_evidence_for_finished_artifacts(artifacts)
    for source, proof in verified.items():
        artifact = artifacts[source]
        evidence[source] = BaselineCompletionEvidence(
            artifact_digest=artifact.digest,
            extraction_succeeded=True,
            artifact_complete=True,
            expected_page_count=proof["expected_page_count"],
            covered_page_count=proof["covered_page_count"],
            pdf_digest=proof["pdf_digest"],
            coverage_method=proof["coverage_method"],
            page_coverage_digest=proof["page_coverage_digest"],
            completion_basis="page_coverage",
        )
    return evidence


def _completion_evidence_metric(
    completion_evidence: Mapping[SourceName, BaselineCompletionEvidence],
) -> dict[str, dict]:
    return {
        source: {
            "artifact_digest": evidence.artifact_digest,
            "completion_basis": evidence.completion_basis,
            "expected_page_count": evidence.expected_page_count,
            "covered_page_count": evidence.covered_page_count,
            "pdf_digest": evidence.pdf_digest,
            "coverage_method": evidence.coverage_method,
            "page_coverage_digest": evidence.page_coverage_digest,
            "page_coverage_verified": evidence.page_coverage_verified,
        }
        for source, evidence in sorted(completion_evidence.items())
    }


def _delivery_assurance(
    source: SourceName,
    completion_evidence: Mapping[SourceName, BaselineCompletionEvidence],
) -> str:
    evidence = completion_evidence.get(source)
    return (
        "page_coverage_verified"
        if evidence is not None and evidence.page_coverage_verified
        else "unverified_failsafe"
    )


def _structure_assurance(text: str, source: SourceName) -> str:
    artifact = SourceArtifact.from_text(source, text)
    units = scan_structural_units(artifact)
    if validate_baseline_structure(artifact, units, BaselineRequirements()).eligible:
        return "strict_structure_validated"
    terminal_requirements = BaselineRequirements(
        required_heading_groups=BaselineRequirements().required_heading_groups[:-1]
    )
    if validate_baseline_structure(artifact, units, terminal_requirements).eligible:
        return "terminal_heading_relaxed"
    return "single_structural_unit_failsafe"


def _llm_usage_summary(llm: "LLM | None") -> dict:
    return {} if llm is None else llm.usage.summary()


def _model_call_count(usage: Mapping) -> int:
    breakdown = usage.get("breakdown", {})
    if not isinstance(breakdown, Mapping):
        return 0
    return sum(
        entry.get("calls", 0)
        for entry in breakdown.values()
        if isinstance(entry, Mapping)
        and isinstance(entry.get("calls", 0), int)
        and not isinstance(entry.get("calls", 0), bool)
        and entry.get("calls", 0) >= 0
    )


def _positive_style_italic_receipt(
    *,
    skeletons: Mapping[SourceName, DocumentSkeleton],
    artifacts: Mapping[SourceName, SourceArtifact],
    evidence_failures: Mapping[SourceName, str],
    transformations: list[dict],
) -> dict:
    """Reconcile the protected and auxiliary style ledgers after the overlay."""

    inventory = _projection_claim_inventory(skeletons, artifacts)
    claims_by_id = {
        claim.claim_id: (source, claim)
        for source, donors in inventory.donors.items()
        for donor in donors
        for claim in donor.claims
    }
    claims_by_id.update(
        {
            claim.claim_id: (source, claim)
            for source, claims in inventory.unplaced.items()
            for claim, _reason in claims
        }
    )
    terminal_events: dict[str, dict] = {}
    for event in transformations:
        if event.get("operation") != "native_emphasis_projection":
            continue
        claim_id = event.get("positive_style_claim_id")
        if claim_id not in claims_by_id:
            continue
        is_terminal = (
            event.get("outcome") in {"supported", "declined"}
            or (
                event.get("outcome") == "projected"
                and event.get("boundary") == "open"
                and event.get("projection_reconciled") is True
            )
        )
        if not is_terminal:
            continue
        if claim_id in terminal_events:
            raise ConsensusContractError(
                "positive style claim has multiple terminal outcomes"
            )
        terminal_events[claim_id] = event

    protected_ids = set(inventory.protected_claim_ids)
    auxiliary_ids = set(inventory.auxiliary_claim_ids)
    missing = (protected_ids | auxiliary_ids) - set(terminal_events)
    if missing:
        raise ConsensusContractError(
            "positive style claim inventory has missing terminal outcomes"
        )
    protected_retained = {
        claim_id
        for claim_id in protected_ids
        if terminal_events[claim_id].get("outcome") in {"projected", "supported"}
    }
    protected_reasons = Counter(
        str(terminal_events[claim_id].get("reason", "unknown"))
        for claim_id in protected_ids - protected_retained
    )
    auxiliary_outcomes = Counter(
        (
            "retained"
            if terminal_events[claim_id].get("outcome") in {"projected", "supported"}
            else str(terminal_events[claim_id].get("reason", "unknown"))
        )
        for claim_id in auxiliary_ids
    )
    source_counts = {}
    for source in sorted(skeletons):
        source_protected = {
            claim_id
            for claim_id in protected_ids
            if claims_by_id[claim_id][0] == source
        }
        source_retained = source_protected & protected_retained
        skeleton = skeletons[source]
        source_counts[source] = {
            "native_body_emphasis_count": len(source_protected),
            "mapped_native_body_emphasis_count": (
                skeleton.mapped_native_body_emphasis_count
            ),
            "retained_native_body_emphasis_count": len(source_retained),
            "native_reference_emphasis_count": (
                skeleton.native_reference_emphasis_count
            ),
            "native_style_emphasis_count": skeleton.native_style_emphasis_count,
            "mapped_native_style_emphasis_count": (
                skeleton.mapped_native_style_emphasis_count
            ),
            "unmapped_native_style_emphasis_count": (
                skeleton.unmapped_native_style_emphasis_count
            ),
            "auxiliary_positive_emphasis_count": sum(
                claims_by_id[claim_id][0] == source for claim_id in auxiliary_ids
            ),
        }
    canonical_interval_ids = {
        event.get("projection_id") or event.get("canonical_interval_id")
        for event in terminal_events.values()
        if event.get("outcome") in {"projected", "supported"}
    }
    canonical_interval_ids.discard(None)
    existing_interval_ids = {
        event.get("canonical_interval_id")
        for event in terminal_events.values()
        if event.get("reason") == "existing_final_emphasis"
    }
    existing_interval_ids.discard(None)
    new_interval_ids = canonical_interval_ids - existing_interval_ids
    unresolved_interval_ids = {
        event.get("unresolved_output_interval_id")
        for event in terminal_events.values()
        if event.get("outcome") == "declined"
    }
    unresolved_interval_ids.discard(None)
    style_selection_ids = {
        event.get("style_selection_id")
        for event in terminal_events.values()
        if event.get("unit_pair_ambiguous") is True
    }
    style_selection_ids.discard(None)
    model_selected_interval_ids = {
        event.get("projection_id") or event.get("canonical_interval_id")
        for event in terminal_events.values()
        if event.get("model_selected_target") is True
        and event.get("outcome") in {"projected", "supported"}
    }
    model_selected_interval_ids.discard(None)
    style_selection_outcomes = Counter()
    style_selection_candidate_counts = Counter()
    for selection_id in style_selection_ids:
        events = [
            event
            for event in terminal_events.values()
            if event.get("style_selection_id") == selection_id
        ]
        style_selection_candidate_counts[
            int(events[0].get("style_selection_candidate_count", 0))
        ] += 1
        if any(event.get("model_selected_target") is True for event in events):
            outcome = "selected"
        elif any(event.get("reason") == "model_selection_unavailable" for event in events):
            outcome = "unavailable"
        elif any(event.get("reason") == "model_selection_none" for event in events):
            outcome = "none"
        else:
            outcome = "not_run"
        style_selection_outcomes[outcome] += 1
    protected_outcomes = Counter()
    protected_by_source_and_evidence: dict[str, Counter[str]] = {}
    for claim_id in sorted(protected_ids):
        source, claim = claims_by_id[claim_id]
        event = terminal_events[claim_id]
        protected_by_source_and_evidence.setdefault(source, Counter())[
            claim.evidence_kind
        ] += 1
        if event.get("outcome") == "projected":
            protected_outcomes["newly_projected"] += 1
        elif event.get("reason") == "existing_final_emphasis":
            protected_outcomes["already_satisfied"] += 1
        elif event.get("outcome") == "supported":
            protected_outcomes["canonical_support"] += 1
        else:
            protected_outcomes["unresolved"] += 1
    direct_native_donor_retained = sum(
        claim_id in protected_ids
        and event.get("direct_native_donor") is True
        and event.get("outcome") in {"projected", "supported"}
        for claim_id, event in terminal_events.items()
    )
    protected_digest = hashlib.sha256(
        "\0".join(sorted(protected_ids)).encode("utf-8")
    ).hexdigest()
    return {
        "policy_version": "positive-style-overlay-v1",
        "native_body_emphasis_count": len(protected_ids),
        "mapped_native_body_emphasis_count": sum(
            skeleton.mapped_native_body_emphasis_count
            for skeleton in skeletons.values()
        ),
        "retained_native_body_emphasis_count": len(protected_retained),
        "excluded_native_body_emphasis_count": len(protected_ids - protected_retained),
        "native_body_exclusion_reason_counts": dict(sorted(protected_reasons.items())),
        "native_body_evidence_reconciled": (
            len(protected_retained) + sum(protected_reasons.values())
            == len(protected_ids)
        ),
        "native_evidence_ready": not evidence_failures,
        "native_evidence_failure_sources": sorted(evidence_failures),
        "all_native_body_italics_retained": len(protected_retained) == len(protected_ids),
        "protected_claim_ids_sha256": protected_digest,
        "auxiliary_positive_emphasis_count": len(auxiliary_ids),
        "auxiliary_positive_outcome_counts": dict(sorted(auxiliary_outcomes.items())),
        "canonical_output_emphasis_interval_count": len(canonical_interval_ids),
        "canonical_output_existing_interval_count": len(existing_interval_ids),
        "canonical_output_new_interval_count": len(new_interval_ids),
        "unique_mapped_plain_interval_count": len(unresolved_interval_ids),
        "direct_native_donor_retained_count": direct_native_donor_retained,
        "deterministic_target_count": len(
            canonical_interval_ids - model_selected_interval_ids
        ),
        "model_selected_target_count": len(model_selected_interval_ids),
        "finite_model_selection_tie_count": len(style_selection_ids),
        "numbered_model_eligible_tie_count": sum(
            count
            for candidate_count, count in style_selection_candidate_counts.items()
            if 1 <= candidate_count <= MAX_BOUNDED_TARGET_CHOICES
        ),
        "model_selection_call_count": sum(
            event.get("style_selection_method") == "sol_numbered_choice"
            for selection_id in style_selection_ids
            for event in [
                next(
                    item
                    for item in terminal_events.values()
                    if item.get("style_selection_id") == selection_id
                )
            ]
        ),
        "model_selection_outcome_counts": dict(
            sorted(style_selection_outcomes.items())
        ),
        "model_selection_candidate_count_distribution": {
            str(candidate_count): count
            for candidate_count, count in sorted(
                style_selection_candidate_counts.items()
            )
        },
        "protected_outcome_counts": dict(sorted(protected_outcomes.items())),
        "protected_claim_counts_by_source_and_evidence": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(protected_by_source_and_evidence.items())
        },
        "markdown_only_body_emphasis_count": sum(
            claim.evidence_kind == "source_markdown" and not claim.protected
            for _source, claim in claims_by_id.values()
        ),
        "reference_markdown_emphasis_count": sum(
            len(_inline_italic_profile(
                artifacts[source].raw_utf8[
                    occurrence.source_byte_start : occurrence.source_byte_end
                ].decode("utf-8", errors="strict"),
                occurrence.unit_type,
            ).emphasis_occurrence_ids)
            for source, skeleton in skeletons.items()
            for occurrence in skeleton.occurrences
            if effective_occurrence_region(occurrence.region, occurrence.unit_type)
            == "back"
            and _inline_italic_profile(
                artifacts[source].raw_utf8[
                    occurrence.source_byte_start : occurrence.source_byte_end
                ].decode("utf-8", errors="strict"),
                occurrence.unit_type,
            )
            is not None
        ),
        "explicit_native_reference_emphasis_count": sum(
            skeleton.native_reference_emphasis_count
            for skeleton in skeletons.values()
        ),
        "native_style_emphasis_count": sum(
            skeleton.native_style_emphasis_count for skeleton in skeletons.values()
        ),
        "mapped_native_style_emphasis_count": sum(
            skeleton.mapped_native_style_emphasis_count
            for skeleton in skeletons.values()
        ),
        "unmapped_native_style_emphasis_count": sum(
            skeleton.unmapped_native_style_emphasis_count
            for skeleton in skeletons.values()
        ),
        "source_counts": source_counts,
    }


def _qualification(
    *,
    abc_report: Mapping,
    semantic_report: Mapping,
    italic_report: Mapping,
    delivery_assurance: str,
    repetition_diagnostics: list[dict],
    final_safety_reasons: tuple[str, ...],
    structural_resolution_failure: str | None,
    unresolved_region_count: int,
) -> tuple[str, list[str]]:
    """Derive the single qualified/failsafe outcome from final receipts."""

    reasons = []
    if abc_report.get("validator_clean") is not True:
        reasons.append("alliance_validator_not_clean")
    if semantic_report.get("reader_payload_retained") is not True:
        reasons.append("reader_payload_not_retained")
    if semantic_report.get("protected_italics_retained") is not True:
        reasons.append("reader_formatting_not_retained")
    if italic_report.get("all_protected_italics_retained") is not True:
        reasons.append("protected_italics_unresolved")
    if italic_report.get("native_evidence_ready") is not True:
        reasons.append("native_italic_evidence_unavailable")
    if delivery_assurance != "page_coverage_verified":
        reasons.append("page_coverage_unverified")
    if repetition_diagnostics:
        reasons.append("excess_repetition")
    if final_safety_reasons:
        reasons.append("final_safety_validation_failed")
    if structural_resolution_failure is not None:
        reasons.append(structural_resolution_failure)
    if unresolved_region_count:
        reasons.append("unresolved_candidate_regions")
    return ("qualified" if not reasons else "failsafe", reasons)


def _resolve_skeleton_conflict(
    selection: SkeletonSelection,
    skeletons: Mapping[SourceName, DocumentSkeleton],
    *,
    llm: "LLM | None",
) -> tuple[DocumentSkeleton, dict, str | None, list[dict]]:
    """Resolve one material whole-document conflict by an existing ID only."""

    baseline = selection.skeleton
    evidence = {
        entry["skeleton_id"]: {
            "candidate_id": entry["skeleton_id"],
            "source": entry["source"],
            "projection_id": entry["projection_id"],
            "title_proven": entry["title_proven"],
            "payload_byte_count": entry["payload_byte_count"],
            "occurrence_count": entry["occurrence_count"],
            "native_mapped_occurrence_count": entry[
                "native_mapped_occurrence_count"
            ],
            "expected_page_count": entry["expected_page_count"],
            "covered_page_count": entry["covered_page_count"],
            "validator_warning_rule_ids": entry[
                "validator_warning_rule_ids"
            ],
        }
        for entry in selection.trace
        if entry["validator_error_rule_ids"] == []
    }
    receipt = {
        "reason": "skeleton_conflict" if selection.conflict else None,
        "baseline_skeleton_id": baseline.skeleton_id,
        "selected_skeleton_id": baseline.skeleton_id,
        "outcome": "deterministic_agreement",
        "candidate_count": len(evidence),
    }
    if not selection.conflict:
        return baseline, receipt, None, []
    if len({item["projection_id"] for item in evidence.values()}) < 2:
        raise ConsensusContractError(
            "skeleton conflict lacks two executable zero-error projections"
        )
    if llm is None:
        receipt["outcome"] = "safe_retention_no_model"
        return baseline, receipt, "skeleton_conflict_unresolved:no_model", []
    try:
        selected_id = llm.resolve_bounded_id_choice(
            reason="skeleton_conflict",
            baseline_id=baseline.skeleton_id,
            choices=list(evidence.values()),
        )
        selected = next(
            skeleton
            for skeleton in skeletons.values()
            if skeleton.skeleton_id == selected_id
        )
        receipt.update(
            {
                "selected_skeleton_id": selected_id,
                "outcome": "sol_selected_existing_id",
            }
        )
        return selected, receipt, None, [{
            "tier": "sol",
            "outcome": "valid",
            "region_ids": ("skeleton_conflict",),
            "reasons": {"skeleton_conflict": "skeleton_conflict"},
        }]
    except (CandidateSelectionFailure, KeyError, StopIteration, ValueError) as exc:
        outcome = (
            BoundedCandidateSelector._sol_failure_outcome(exc.reason)
            if isinstance(exc, CandidateSelectionFailure)
            else "invalid_existing_id"
        )
        receipt["outcome"] = f"safe_retention_{outcome}"
        return (
            baseline,
            receipt,
            f"skeleton_conflict_unresolved:{outcome}",
            [{
                "tier": "sol",
                "outcome": outcome,
                "region_ids": ("skeleton_conflict",),
                "reasons": {"skeleton_conflict": "skeleton_conflict"},
            }],
        )


def _finalize_output(
    text: str,
    audit: list[dict],
) -> tuple[str, list[dict], list[str]]:
    """Apply the only harmless post-render normalization."""

    text, audit, newline_normalized = normalize_trailing_newline(text, audit)
    warnings = (
        ["deterministic_trailing_newline_normalization"]
        if newline_normalized
        else []
    )
    return text, audit, warnings


def _zero_error_source_delivery(
    artifacts: Mapping[SourceName, SourceArtifact],
    completion_evidence: Mapping[SourceName, BaselineCompletionEvidence],
    skeletons: Mapping[SourceName, object],
    *,
    preferred_source: SourceName,
    rejected_result,
):
    """Return a complete exact-source document whose Alliance errors are zero.

    This terminal path does not synthesize publication text.  It chooses one
    completed source, conservatively removes title semantics by demoting H1
    markers, and accepts the result only after the exact pinned validator says
    it is valid with no errors.
    """

    candidates = sorted(
        (
            source
            for source in artifacts
            if source in skeletons and source in completion_evidence
        ),
        key=lambda source: (
            completion_evidence[source].page_coverage_verified,
            source == preferred_source,
            len(artifacts[source].raw_utf8),
            source,
        ),
        reverse=True,
    )
    requirements = BaselineRequirements(
        minimum_words=1,
        minimum_structural_units=1,
        minimum_non_whitespace_bytes=1,
        require_heading_or_five_units=False,
        required_heading_groups=(),
        require_abc_validation=False,
    )
    for source in candidates:
        artifact = artifacts[source]
        try:
            baseline = select_baseline(
                {source: artifact},
                completion_evidence={source: completion_evidence[source]},
                requirements=requirements,
            )
            candidate_result = merge_with_baseline_failsafe(
                baseline,
                {source: artifact},
            )
            audit = [
                {
                    "output_byte_start": span.output_byte_start,
                    "output_byte_end": span.output_byte_end,
                    "source": span.source,
                    "artifact_digest": span.artifact_digest,
                    "source_byte_start": span.source_byte_start,
                    "source_byte_end": span.source_byte_end,
                    "candidate_id": span.candidate_id,
                    "region_id": span.region_id,
                    "decision_method": "baseline_fallback",
                }
                for span in candidate_result.document.provenance
            ]
            text, audit, transformations = render_document_skeleton(
                candidate_result.document.text,
                audit,
                skeletons[source],
                force_titleless=True,
            )
            text, audit, newline_normalized = normalize_trailing_newline(text, audit)
            report = abc_markdown_report(text)
            if (
                report.get("valid") is not True
                or report.get("error_rule_ids") != []
                or report.get("failure_code") is not None
                or report.get("parser_version_exact") is not True
                or report.get("parser_implementation_exact") is not True
                or repetition_diagnostics_metric(
                    text,
                    tuple(artifacts[item] for item in sorted(artifacts)),
                )
                or final_document_safety_reasons(text)
            ):
                continue
            candidate_result = replace(
                candidate_result,
                unresolved_region_count=len(rejected_result.decision_trace),
                decision_trace=tuple(
                    {
                        **event,
                        "decision_method": "baseline_fallback",
                        "decision_reason": "alliance_error_source_delivery",
                        "selected_choice": 0,
                        "selected_candidate_ids": [event["baseline_candidate_id"]],
                        "replaced_baseline": False,
                    }
                    for event in rejected_result.decision_trace
                ),
                candidate_construction_trace=(
                    rejected_result.candidate_construction_trace
                ),
                candidate_construction_counts=(
                    rejected_result.candidate_construction_counts
                ),
            )
            warnings = [f"alliance_error_source_delivery:{source}"]
            if newline_normalized:
                warnings.append("deterministic_trailing_newline_normalization")
            return (
                candidate_result,
                text,
                audit,
                warnings,
                transformations,
                skeletons[source],
            )
        except Exception:
            continue
    return None


def merge_source_artifacts(
    grobid_md: str,
    docling_md: str,
    marker_md: str,
    llm: "LLM | None",
    *,
    completion_evidence: Mapping[SourceName, BaselineCompletionEvidence],
    native_structures: Mapping[SourceName, NativeStructureArtifact] | None = None,
    native_structure_failures: Mapping[SourceName, str] | None = None,
    baseline_requirements: BaselineRequirements | None = None,
    benchmark_mode: bool = False,
) -> tuple[str | None, dict, list]:
    """Merge exact source spans and expose only candidate IDs to Terra/Sol."""

    source_texts = {
        "grobid": grobid_md,
        "docling": docling_md,
        "marker": marker_md,
    }
    artifacts: dict[SourceName, SourceArtifact] = {
        source: SourceArtifact.from_text(source, text)
        for source, text in source_texts.items()
        if text and text.strip()
    }
    available_extractors = sorted(artifacts)
    missing_extractors = sorted(set(source_texts) - set(artifacts))
    source_metrics = {
        "source_count": len(artifacts),
        "available_extractors": available_extractors,
        "missing_extractors": missing_extractors,
    }
    if not artifacts:
        return (
            None,
            {
                "merge_contract_id": Config.MERGE_CONTRACT_ID,
                "failed": True,
                "failure_reason": "no_usable_extractor",
                **source_metrics,
            },
            [],
        )

    native_structures = dict(native_structures or {})
    native_structure_failures = dict(native_structure_failures or {})
    unexpected_native_sources = set(native_structures) - set(artifacts)
    if unexpected_native_sources:
        raise ConsensusContractError(
            "native structure supplied without matching Markdown: "
            f"{sorted(unexpected_native_sources)!r}"
        )

    relaxed_delivery_baseline_reason = None
    default_heading_groups = BaselineRequirements().required_heading_groups
    try:
        baseline = select_baseline(
            artifacts,
            completion_evidence=completion_evidence,
            requirements=baseline_requirements,
        )
    except Exception as exc:
        if baseline_requirements is not None:
            return (
                None,
                {
                    "merge_contract_id": Config.MERGE_CONTRACT_ID,
                    "failed": True,
                    "failure_reason": f"baseline_unavailable:{type(exc).__name__}",
                    **source_metrics,
                },
                [],
            )
        try:
            try:
                baseline = select_baseline(
                    artifacts,
                    completion_evidence=completion_evidence,
                    requirements=BaselineRequirements(
                        required_heading_groups=default_heading_groups[:-1]
                    ),
                )
                relaxed_delivery_baseline_reason = "terminal_heading_group"
            except Exception:
                final_requirements = BaselineRequirements(
                    minimum_words=1,
                    minimum_structural_units=1,
                    minimum_non_whitespace_bytes=1,
                    require_heading_or_five_units=False,
                    required_heading_groups=(),
                    require_abc_validation=False,
                )
                safe_sources = []
                for source, artifact in artifacts.items():
                    try:
                        safe_sources.append(
                            select_baseline(
                                {source: artifact},
                                completion_evidence={
                                    source: completion_evidence[source]
                                },
                                requirements=final_requirements,
                            )
                        )
                    except (KeyError, ValueError, ConsensusContractError):
                        continue
                if not safe_sources:
                    raise ValueError("no safe source-backed delivery baseline")
                baseline = max(
                    safe_sources,
                    key=lambda item: (
                        len(item.artifact.raw_utf8),
                        len(item.units),
                        item.artifact.source,
                    ),
                )
                relaxed_delivery_baseline_reason = "unverified_largest_safe_source"
        except Exception as baseline_exc:
            return (
                None,
                {
                    "merge_contract_id": Config.MERGE_CONTRACT_ID,
                    "failed": True,
                    "failure_reason": (
                        f"baseline_unavailable:{type(baseline_exc).__name__}"
                    ),
                    **source_metrics,
                },
                [],
            )

    content_baseline_source = baseline.artifact.source
    skeletons = {}
    selectable_skeletons = {}
    skeleton_build_failures = {}
    for source, artifact in sorted(artifacts.items()):
        evidence = completion_evidence.get(source)
        try:
            skeletons[source] = build_document_skeleton(
                artifact,
                native_structures.get(source),
            )
        except Exception as exc:
            skeleton_build_failures[source] = type(exc).__name__
            skeletons[source] = build_document_skeleton(artifact, None)
        if evidence is not None and not evidence.rejection_reasons(artifact):
            selectable_skeletons[source] = skeletons[source]
    native_evidence_failures = {
        source: str(native_structure_failures[source])
        for source in sorted(artifacts)
        if source in native_structure_failures
    }
    for source in sorted(artifacts):
        if source in skeleton_build_failures:
            native_evidence_failures[source] = (
                f"native_structure_invalid:{skeleton_build_failures[source]}"
            )
        elif source not in native_structures:
            native_evidence_failures.setdefault(
                source, "native_structure_unavailable"
            )
    if not selectable_skeletons:
        selectable_skeletons[content_baseline_source] = skeletons[
            content_baseline_source
        ]
    skeleton_selection = choose_document_skeleton(
        selectable_skeletons,
        artifacts,
        preferred_source=content_baseline_source,
    )
    (
        selected_skeleton,
        skeleton_resolution,
        structural_resolution_failure,
        skeleton_selection_events,
    ) = _resolve_skeleton_conflict(
        skeleton_selection,
        skeletons,
        llm=llm,
    )
    skeleton_occurrence_ids = {
        (source, occurrence.unit_id): occurrence.occurrence_id
        for source, skeleton in skeletons.items()
        for occurrence in skeleton.occurrences
    }
    skeleton_slot_keys = {
        (source, occurrence.unit_id): occurrence.slot_key
        for source, skeleton in skeletons.items()
        for occurrence in skeleton.occurrences
    }

    selector = (
        None
        if llm is None
        else BoundedCandidateSelector(llm=llm)
    )
    result = merge_with_baseline_failsafe(
        baseline,
        artifacts,
        selection_resolver=selector,
        skeleton_occurrence_ids=skeleton_occurrence_ids,
        skeleton_slot_keys=skeleton_slot_keys,
    )
    if relaxed_delivery_baseline_reason is not None:
        relaxed_warning = (
            f"relaxed_delivery_baseline:{relaxed_delivery_baseline_reason}"
        )
        result = replace(
            result,
            document=replace(
                result.document,
                provenance=tuple(
                    replace(span, decision_method="baseline_fallback")
                    for span in result.document.provenance
                ),
                rejection_reasons=result.document.rejection_reasons
                + (relaxed_warning,),
            ),
            merge_quality="baseline_fallback",
            warnings=result.warnings + (relaxed_warning,),
        )
    if len(artifacts) == 1:
        single_warning = "single_extractor_fallback"
        result = replace(
            result,
            document=replace(
                result.document,
                provenance=tuple(
                    replace(span, decision_method="baseline_fallback")
                    for span in result.document.provenance
                ),
                rejection_reasons=result.document.rejection_reasons + (single_warning,),
            ),
            merge_quality="baseline_fallback",
            warnings=result.warnings + (single_warning,),
        )

    def result_audit(current_result):
        return [
            {
                "output_byte_start": span.output_byte_start,
                "output_byte_end": span.output_byte_end,
                "source": span.source,
                "artifact_digest": span.artifact_digest,
                "source_byte_start": span.source_byte_start,
                "source_byte_end": span.source_byte_end,
                "candidate_id": span.candidate_id,
                "region_id": span.region_id,
                "decision_method": span.decision_method,
            }
            for span in current_result.document.provenance
        ]

    audit = result_audit(result)
    merged_text = result.document.text
    skeleton_render_warnings = []
    try:
        merged_text, audit, skeleton_transformations = render_document_skeleton(
            merged_text,
            audit,
            selected_skeleton,
            decision_trace=result.decision_trace,
        )
        merged_text, audit, role_transformations = render_document_role_slots(
            merged_text,
            audit,
            selected_skeleton,
            skeletons,
            decision_trace=result.decision_trace,
            artifacts=artifacts,
            title_selection_resolver=(
                None
                if llm is None
                else llm.resolve_bounded_id_choice_with_receipt
            ),
        )
        if any(
            event.get("operation") == "alliance_role_binding_unresolved"
            for event in role_transformations
        ) and structural_resolution_failure is None:
            structural_resolution_failure = "role_binding_unresolved"
        skeleton_transformations = [
            *skeleton_transformations,
            *role_transformations,
        ]
        # A later occurrence-bound marker decision may intentionally supersede
        # an earlier conservative marker edit at the same byte. Receipts describe
        # the final audit partition, so mark only transformations still present
        # in that partition as audit-emitting.
        skeleton_transformations = reconcile_document_transformations(
            audit, skeleton_transformations
        )
    except Exception as exc:
        result = merge_with_baseline_failsafe(
            baseline,
            {baseline.artifact.source: baseline.artifact},
        )
        audit = result_audit(result)
        merged_text, audit, skeleton_transformations = render_document_skeleton(
            result.document.text,
            audit,
            selected_skeleton,
        )
        skeleton_render_warnings.append(
            f"merged_skeleton_render_fallback:{type(exc).__name__}"
        )
    merged_text, audit, finalization_warnings = _finalize_output(
        merged_text,
        audit,
    )
    merged_warnings = [
        *result.warnings,
        *skeleton_render_warnings,
        *finalization_warnings,
    ]
    effective_merge_quality = result.merge_quality

    try:
        merged_text, audit, emphasis_transformations = project_native_emphasis(
            merged_text,
            audit,
            skeletons,
            artifacts,
            style_selection_resolver=(
                None
                if llm is None
                else llm.resolve_bounded_id_choice_with_receipt
            ),
        )
        skeleton_transformations = [
            *skeleton_transformations,
            *emphasis_transformations,
        ]
    except Exception as exc:
        merged_warnings.append(
            f"native_emphasis_projection_declined:{type(exc).__name__}"
        )

    postprocess_repetition = repetition_diagnostics_metric(
        merged_text,
        tuple(artifacts[source] for source in sorted(artifacts)),
    )
    if postprocess_repetition:
        rejected_result = result
        result = merge_with_baseline_failsafe(
            baseline,
            {baseline.artifact.source: baseline.artifact},
        )
        result = replace(
            result,
            unresolved_region_count=len(rejected_result.decision_trace),
            decision_trace=tuple(
                {
                    **event,
                    "decision_method": "baseline_fallback",
                    "decision_reason": "rendered_output_rejected:excess_repetition",
                    "selected_choice": 0,
                    "selected_candidate_ids": [event["baseline_candidate_id"]],
                    "replaced_baseline": False,
                }
                for event in rejected_result.decision_trace
            ),
            candidate_construction_trace=(
                rejected_result.candidate_construction_trace
            ),
            candidate_construction_counts=(
                rejected_result.candidate_construction_counts
            ),
        )
        audit = [
            {
                "output_byte_start": span.output_byte_start,
                "output_byte_end": span.output_byte_end,
                "source": span.source,
                "artifact_digest": span.artifact_digest,
                "source_byte_start": span.source_byte_start,
                "source_byte_end": span.source_byte_end,
                "candidate_id": span.candidate_id,
                "region_id": span.region_id,
                "decision_method": "baseline_fallback",
            }
            for span in result.document.provenance
        ]
        merged_text, audit, finalization_warnings = _finalize_output(
            result.document.text, audit
        )
        merged_warnings = [
            *(
                warning
                for warning in merged_warnings
                if warning != "deterministic_trailing_newline_normalization"
            ),
            "rendered_output_rejected:excess_repetition",
            *finalization_warnings,
        ]
        effective_merge_quality = "baseline_fallback"
        skeleton_transformations = []

    # Alliance errors are categorically forbidden for every persisted outcome,
    # including the complete-source failsafe.  If the selected merge still has
    # an error, choose a complete source and render only conservative heading
    # markers; never ask a model to author replacement publication text.
    precommit_abc_report = abc_markdown_report(merged_text)
    if (
        precommit_abc_report.get("valid") is not True
        or precommit_abc_report.get("error_rule_ids") != []
        or precommit_abc_report.get("failure_code") is not None
    ):
        zero_error_delivery = _zero_error_source_delivery(
            artifacts,
            completion_evidence,
            skeletons,
            preferred_source=result.document.baseline_source,
            rejected_result=result,
        )
        if zero_error_delivery is None:
            raise ConsensusContractError(
                "no exact-source delivery candidate passed Alliance validation"
            )
        (
            result,
            merged_text,
            audit,
            zero_error_warnings,
            skeleton_transformations,
            selected_skeleton,
        ) = zero_error_delivery
        merged_warnings = [
            *result.warnings,
            *zero_error_warnings,
        ]
        effective_merge_quality = "baseline_fallback"

    if (
        structural_resolution_failure is not None
        and structural_resolution_failure not in merged_warnings
    ):
        merged_warnings.append(structural_resolution_failure)

    llm_usage = _llm_usage_summary(llm)
    model_selection_calls = list(
        getattr(llm, "selection_call_traces", ()) if llm is not None else ()
    )
    selection_events = [
        *skeleton_selection_events,
        *([] if selector is None else selector.events),
    ]
    abc_report = abc_markdown_report(merged_text)
    semantic_receipt = build_semantic_payload_receipt(
        merged_text,
        audit,
        baseline_source=result.document.baseline_source,
        skeletons=skeletons,
        decision_trace=result.decision_trace,
    )
    semantic_report = semantic_payload_reader_report(
        merged_text,
        semantic_receipt,
        validator_report=abc_report,
    )
    native_italic_report = _positive_style_italic_receipt(
        skeletons=skeletons,
        artifacts=artifacts,
        evidence_failures=native_evidence_failures,
        transformations=skeleton_transformations,
    )
    projection_declines = Counter(
        str(event.get("reason", "unknown"))
        for event in skeleton_transformations
        if event.get("operation") == "native_emphasis_projection"
        and event.get("phase") == "result"
        and event.get("outcome") == "declined"
    )
    native_emphasis_projection = {
        "policy_version": "post-merge-positive-style-overlay-v1",
        "eligible_occurrence_count": sum(
            event.get("operation") == "native_emphasis_projection"
            and event.get("phase") == "eligibility"
            and event.get("outcome") == "eligible"
            for event in skeleton_transformations
        ),
        "projected_reconciled_occurrence_count": sum(
            event.get("operation") == "native_emphasis_projection"
            and event.get("boundary") == "open"
            and event.get("outcome") == "projected"
            and event.get("projection_reconciled") is True
            for event in skeleton_transformations
        ),
        "supported_occurrence_count": sum(
            event.get("operation") == "native_emphasis_projection"
            and event.get("outcome") == "supported"
            and event.get("support_reconciled") is True
            for event in skeleton_transformations
        ),
        "declined_occurrence_count": sum(projection_declines.values()),
        "decline_reason_counts": dict(sorted(projection_declines.items())),
        "finite_model_selection_tie_count": native_italic_report[
            "finite_model_selection_tie_count"
        ],
        "numbered_model_eligible_tie_count": native_italic_report[
            "numbered_model_eligible_tie_count"
        ],
        "model_selection_call_count": native_italic_report[
            "model_selection_call_count"
        ],
        "model_selected_target_count": native_italic_report[
            "model_selected_target_count"
        ],
        "model_selection_outcome_counts": native_italic_report[
            "model_selection_outcome_counts"
        ],
    }
    italic_report = {
        **native_italic_report,
        "all_protected_italics_retained": native_italic_report[
            "all_native_body_italics_retained"
        ],
        "protected_italic_occurrence_count": native_italic_report[
            "native_body_emphasis_count"
        ],
        "retained_protected_italic_occurrence_count": native_italic_report[
            "retained_native_body_emphasis_count"
        ],
        "lost_protected_italic_occurrence_count": (
            native_italic_report["native_body_emphasis_count"]
            - native_italic_report["retained_native_body_emphasis_count"]
        ),
    }
    delivery_assurance = _delivery_assurance(
        result.document.baseline_source,
        completion_evidence,
    )
    repetition_diagnostics = repetition_diagnostics_metric(
        merged_text,
        tuple(artifacts[source] for source in sorted(artifacts)),
    )
    final_safety_reasons = final_document_safety_reasons(merged_text)
    qualification_outcome, qualification_reasons = _qualification(
        abc_report=abc_report,
        semantic_report=semantic_report,
        italic_report=italic_report,
        delivery_assurance=delivery_assurance,
        repetition_diagnostics=repetition_diagnostics,
        final_safety_reasons=final_safety_reasons,
        structural_resolution_failure=structural_resolution_failure,
        unresolved_region_count=result.unresolved_region_count,
    )
    unresolved_region_reason_counts = dict(
        sorted(
            Counter(
                str(event.get("decision_reason", "unknown"))
                for event in result.decision_trace
                if event.get("decision_method") == "baseline_fallback"
            ).items()
        )
    )
    metrics = {
        "merge_contract_id": Config.MERGE_CONTRACT_ID,
        "failed": False,
        "failure_reason": None,
        "source_artifact_digests": {
            source: artifacts[source].digest for source in sorted(artifacts)
        },
        "baseline_source": result.document.baseline_source,
        "baseline_digest": result.document.baseline_digest,
        "content_baseline_source": content_baseline_source,
        "document_skeleton_source": selected_skeleton.source,
        "document_skeleton_id": selected_skeleton.skeleton_id,
        "document_skeleton_projection_id": selected_skeleton.projection_id,
        "document_skeleton_candidate_ids": {
            source: skeleton.skeleton_id
            for source, skeleton in sorted(skeletons.items())
        },
        "document_skeleton_candidate_projection_ids": {
            source: skeleton.projection_id
            for source, skeleton in sorted(skeletons.items())
        },
        "native_structure_receipt_digests": {
            source: native.receipt_digest
            for source, native in sorted(native_structures.items())
        },
        "native_structure_artifact_digests": {
            source: native.native_digest
            for source, native in sorted(native_structures.items())
        },
        "document_skeleton_native_digest": (
            selected_skeleton.native_artifact_digest
        ),
        "document_skeleton_title_proven": selected_skeleton.title_proven,
        "document_skeleton_conflict": skeleton_selection.conflict,
        "document_skeleton_findings": list(selected_skeleton.findings),
        "document_skeleton_selection_trace": (
            list(skeleton_selection.trace) if benchmark_mode else []
        ),
        "document_skeleton_resolution": {
            **skeleton_resolution,
            "delivered_skeleton_id": selected_skeleton.skeleton_id,
        },
        "document_skeleton_build_failures": skeleton_build_failures,
        "native_structure_load_failures": native_structure_failures,
        "native_italic_evidence_failures": native_evidence_failures,
        "benchmark_mode": bool(benchmark_mode),
        "baseline_selection_trace": (
            list(baseline.selection_trace) if benchmark_mode else []
        ),
        "baseline_selection_relaxation": relaxed_delivery_baseline_reason,
        "output_digest": hashlib.sha256(merged_text.encode("utf-8")).hexdigest(),
        "merge_quality": effective_merge_quality,
        "unresolved_region_count": result.unresolved_region_count,
        "unresolved_region_reason_counts": unresolved_region_reason_counts,
        "replaced_region_count": len(result.document.replaced_region_ids),
        "warnings": merged_warnings,
        "rejection_reasons": list(result.document.rejection_reasons),
        "repetition_diagnostics": repetition_diagnostics,
        "repetition_policy": repetition_policy_metric(),
        "source_repetition_profiles": source_repetition_profiles_metric(artifacts),
        "quarantined_repeated_sources": list(
            result.document.quarantined_repeated_sources
        ),
        "direct_sol_region_count": sum(
            len(event.get("region_ids", ()))
            for event in selection_events
            if isinstance(event, Mapping)
            and event.get("tier") == "sol"
            and "terra_failure" not in set(
                (event.get("reasons") or {}).values()
            )
        ),
        "terra_to_sol_escalation_count": sum(
            len(event.get("region_ids", ()))
            for event in selection_events
            if isinstance(event, Mapping)
            and event.get("tier") == "sol"
            and "terra_failure" in set(
                (event.get("reasons") or {}).values()
            )
        ),
        "model_failure_counts": dict(
            sorted(
                Counter(
                    str(event.get("outcome"))
                    for event in selection_events
                    if isinstance(event, Mapping)
                    and event.get("tier") in {"terra", "sol"}
                    and event.get("outcome") != "valid"
                ).items()
            )
        ),
        "selection_events": selection_events,
        "region_decisions": list(result.decision_trace) if benchmark_mode else [],
        "candidate_alignment_trace": (
            list(result.candidate_construction_trace) if benchmark_mode else []
        ),
        "candidate_construction_counts": dict(
            result.candidate_construction_counts
        ),
        "region_decision_counts": dict(
            sorted(
                Counter(
                    event.get("decision_method", "unknown")
                    for event in result.decision_trace
                ).items()
            )
        ),
        "candidate_region_count": len(result.decision_trace),
        "runtime_models": resolved_runtime_model_map(),
        "llm_usage": llm_usage,
        "model_call_attempts": _model_call_count(llm_usage),
        "model_selection_calls": model_selection_calls,
        "model_calls": _model_call_count(llm_usage),
        "fallback_used": effective_merge_quality == "baseline_fallback",
        "qualification_outcome": qualification_outcome,
        "qualification_reasons": qualification_reasons,
        "final_validation_passed": not bool(final_safety_reasons),
        "unsafe_character_count": len(unsafe_unicode_characters(merged_text)),
        "delivery_assurance": delivery_assurance,
        "structure_assurance": _structure_assurance(
            merged_text, result.document.baseline_source
        ),
        "completion_evidence_basis": {
            source: evidence.completion_basis
            for source, evidence in completion_evidence.items()
        },
        "page_coverage_verified": {
            source: evidence.page_coverage_verified
            for source, evidence in completion_evidence.items()
        },
        "page_coverage_evidence": _completion_evidence_metric(
            completion_evidence
        ),
        "abc_markdown": abc_report,
        "semantic_payload_receipt": semantic_receipt.as_metric(),
        "semantic_payload_reader": semantic_report,
        "italic_preservation": italic_report,
        "native_emphasis_projection": native_emphasis_projection,
        "quality_receipt_status": {
            "native_italics_valid": True,
            "native_italics_error": None,
        },
        "document_skeleton_transformations": skeleton_transformations,
        **source_metrics,
    }
    italic_receipt_error = italic_preservation_receipt_error(metrics)
    if italic_receipt_error is not None:
        metrics["quality_receipt_status"] = {
            "native_italics_valid": False,
            "native_italics_error": italic_receipt_error,
        }
        metrics["qualification_outcome"] = "failsafe"
        if "native_italic_receipt_invalid" not in metrics["qualification_reasons"]:
            metrics["qualification_reasons"].append(
                "native_italic_receipt_invalid"
            )
    validate_merge_artifacts(
        merged_text,
        metrics,
        audit,
        artifacts=artifacts,
        expected_contract_id=Config.MERGE_CONTRACT_ID,
        skeletons=skeletons,
    )
    return merged_text, metrics, audit


def merge_finished_extractor_outputs(
    grobid_md: str,
    docling_md: str,
    marker_md: str,
    llm: "LLM | None",
    *,
    completion_evidence: Mapping[
        SourceName, BaselineCompletionEvidence
    ] | None = None,
    native_structures: Mapping[SourceName, NativeStructureArtifact] | None = None,
    native_structure_failures: Mapping[SourceName, str] | None = None,
    benchmark_mode: bool = False,
) -> tuple[str | None, dict, list]:
    """Merge after all requested extractor calls terminate."""

    artifacts: dict[SourceName, SourceArtifact] = {
        source: SourceArtifact.from_text(source, text)
        for source, text in {
            "grobid": grobid_md,
            "docling": docling_md,
            "marker": marker_md,
        }.items()
        if text and text.strip()
    }
    return merge_source_artifacts(
        grobid_md,
        docling_md,
        marker_md,
        llm,
        completion_evidence=(
            completion_evidence_for_finished_artifacts(artifacts)
            if completion_evidence is None
            else completion_evidence
        ),
        native_structures=native_structures,
        native_structure_failures=native_structure_failures,
        benchmark_mode=benchmark_mode,
    )
