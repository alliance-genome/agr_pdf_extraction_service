import hashlib

import pytest

from app.services.merge_service import completion_evidence_for_finished_artifacts
from app.services.document_skeleton import build_document_skeleton
from app.services.source_contracts import ConsensusContractError, SourceArtifact
from app.services.source_merge import (
    BaselineDocument,
    BaselineRequirements,
    BaselineScore,
    _align_to_baseline,
    _has_nonlocal_scope_spill,
    _preferred_source_backed_italic_candidate,
    build_candidate_merge_plan,
    italic_preservation_metric,
    merge_with_baseline_failsafe,
    repetition_diagnostics_metric,
    scan_structural_units,
    select_baseline,
)


RELAXED = BaselineRequirements(
    minimum_words=1,
    minimum_structural_units=1,
    minimum_non_whitespace_bytes=1,
    require_heading_or_five_units=False,
    required_heading_groups=(),
    require_abc_validation=False,
)


def _artifacts():
    return {
        "grobid": SourceArtifact.from_text("grobid", "# Title\n\nGene dpp is active."),
        "docling": SourceArtifact.from_text("docling", "# Title\n\nGene *dpp* is active."),
        "marker": SourceArtifact.from_text("marker", "# Title\n\nGene dpp is active."),
    }


def test_structural_scan_ranges_reproduce_exact_source_bytes():
    artifact = _artifacts()["docling"]
    units = scan_structural_units(artifact)
    assert [unit.unit_type for unit in units] == ["heading", "paragraph"]
    for unit in units:
        raw = artifact.raw_utf8[unit.byte_start:unit.byte_end]
        assert raw.decode("utf-8")


def test_baseline_requires_digest_bound_completion_evidence():
    artifacts = _artifacts()
    with pytest.raises(ConsensusContractError, match="no clean baseline"):
        select_baseline(artifacts, completion_evidence={}, requirements=RELAXED)
    baseline = select_baseline(
        artifacts,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        requirements=RELAXED,
    )
    assert baseline.artifact.source in artifacts
    assert baseline.artifact.digest == artifacts[baseline.artifact.source].digest


def test_missing_selector_retains_a_complete_source_with_exact_provenance():
    artifacts = {
        source: SourceArtifact.from_text(source, f"# Title\n\nGene {symbol} is active.")
        for source, symbol in (
            ("grobid", "Gg1"),
            ("docling", "Gγ1"),
            ("marker", "G g 1"),
        )
    }
    baseline = select_baseline(
        artifacts,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        requirements=RELAXED,
    )
    result = merge_with_baseline_failsafe(baseline, artifacts)
    assert result.document.raw_utf8 == baseline.artifact.raw_utf8
    assert result.merge_quality == "baseline_fallback"
    assert result.unresolved_region_count >= 1
    assert all(
        baseline.artifact.raw_utf8[span.source_byte_start:span.source_byte_end]
        == result.document.raw_utf8[span.output_byte_start:span.output_byte_end]
        for span in result.document.provenance
    )


def test_union_source_italics_protects_peer_evidence_on_plain_baseline():
    artifacts = _artifacts()

    retained = italic_preservation_metric(
        artifacts["docling"].text,
        artifacts,
        baseline_source="grobid",
    )
    lost = italic_preservation_metric(
        artifacts["grobid"].text,
        artifacts,
        baseline_source="grobid",
    )

    assert retained["policy_version"] == "union-source-italics-v4"
    assert retained["protected_italic_occurrence_count"] == 1
    assert retained["lost_protected_italic_occurrence_count"] == 0
    assert retained["all_protected_italics_retained"] is True
    assert lost["protected_italic_occurrence_count"] == 1
    assert lost["lost_protected_italic_occurrence_count"] == 1
    assert lost["matched_block_formatting_lost_occurrence_count"] == 1
    assert lost["whole_block_missing_protected_italic_occurrence_count"] == 0
    assert lost["all_protected_italics_retained"] is False


def test_union_source_italics_reports_peer_occurrence_missing_from_output():
    baseline = "# Title\n\nBaseline paragraph."
    artifacts = {
        "grobid": SourceArtifact.from_text("grobid", baseline),
        "docling": SourceArtifact.from_text(
            "docling",
            "# Title\n\nBaseline paragraph.\n\nGene *dpp* is absent from baseline.",
        ),
    }

    report = italic_preservation_metric(
        baseline,
        artifacts,
        baseline_source="grobid",
    )

    assert report["protected_italic_occurrence_count"] == 1
    assert report["lost_protected_italic_occurrence_count"] == 1
    assert report["whole_block_missing_protected_italic_occurrence_count"] == 1
    assert report["matched_block_formatting_lost_occurrence_count"] == 0
    assert report["unresolved_italic_block_count"] == 1
    assert report["all_protected_italics_retained"] is False


def test_conflicting_source_emphasis_boundaries_are_ambiguous_not_rewritten():
    artifacts = {
        "grobid": SourceArtifact.from_text("grobid", "# Title\n\nGene *dpp* is active."),
        "docling": SourceArtifact.from_text("docling", "# Title\n\n*Gene dpp* is active."),
        "marker": SourceArtifact.from_text("marker", "# Title\n\nGene dpp is active."),
    }

    report = italic_preservation_metric(
        artifacts["grobid"].text,
        artifacts,
        baseline_source="marker",
    )

    assert report["ambiguous_italic_block_count"] == 1
    assert report["protected_italic_occurrence_count"] == 0
    assert report["all_protected_italics_retained"] is False


@pytest.mark.parametrize(
    "collateral",
    [
        "Gene *dpp* is **active**.",
        "Gene *dpp* is [active](https://example.org).",
        "Gene *dpp* is `active`.",
        "Gene *dpp* is <strong>active</strong>.",
    ],
)
def test_italic_preference_declines_collateral_inline_markup(collateral):
    artifacts = {
        "grobid": SourceArtifact.from_text("grobid", "Gene dpp is active."),
        "docling": SourceArtifact.from_text("docling", "Gene *dpp* is active."),
        "marker": SourceArtifact.from_text("marker", collateral),
    }
    units = [scan_structural_units(artifacts[source])[0] for source in artifacts]

    assert _preferred_source_backed_italic_candidate(
        units,
        ["grobid-candidate", "docling-candidate", "marker-candidate"],
        artifacts,
    ) is None


def test_candidate_request_carries_content_free_formatting_evidence():
    artifacts = _artifacts()
    plan = build_candidate_merge_plan(
        _baseline_document(artifacts["grobid"]),
        artifacts,
    )

    request = plan.store.build_selection_request(plan.graph)
    paragraph_region = next(
        region
        for region in request.regions
        if any(candidate.emphasis_occurrence_ids for candidate in region.candidates)
    )
    italic_candidate = next(
        candidate
        for candidate in paragraph_region.candidates
        if candidate.emphasis_occurrence_ids
    )
    assert italic_candidate.visible_text_digest is not None
    assert italic_candidate.non_emphasis_ast_digest is not None
    assert len(italic_candidate.emphasis_occurrence_ids) == 1


def test_alignment_rejects_peer_body_candidate_from_a_different_skeleton_slot():
    artifacts = {
        "grobid": SourceArtifact.from_text(
            "grobid",
            "# Title\n\n## Methods\n\n### Shared\n\nGene dpp is active.\n",
        ),
        "docling": SourceArtifact.from_text(
            "docling",
            "# Title\n\n## Results\n\n### Shared\n\nGene dpp is inactive.\n",
        ),
    }
    skeletons = {
        source: build_document_skeleton(artifact, None)
        for source, artifact in artifacts.items()
    }
    occurrence_ids = {
        (source, occurrence.unit_id): occurrence.occurrence_id
        for source, skeleton in skeletons.items()
        for occurrence in skeleton.occurrences
    }
    slot_keys = {
        (source, occurrence.unit_id): occurrence.slot_key
        for source, skeleton in skeletons.items()
        for occurrence in skeleton.occurrences
    }
    plan = build_candidate_merge_plan(
        _baseline_document(artifacts["grobid"]),
        artifacts,
        skeleton_occurrence_ids=occurrence_ids,
        skeleton_slot_keys=slot_keys,
    )

    assert all(
        candidate.source != "docling" or candidate.candidate_type != "prose"
        for region in plan.graph.regions
        for candidate_id in {
            candidate_id
            for path in region.valid_paths
            for candidate_id in path
        }
        for candidate in (plan.store.candidate_metadata(candidate_id),)
    )


def test_alignment_admits_peer_body_candidate_with_matching_ancestor_slot():
    artifacts = {
        "grobid": SourceArtifact.from_text(
            "grobid",
            "# Title\n\n## Methods\n\n### Shared\n\nGene dpp is active.\n",
        ),
        "docling": SourceArtifact.from_text(
            "docling",
            "# Title\n\n## Methods\n\n### Shared\n\nGene dpp is inactive.\n",
        ),
    }
    skeletons = {
        source: build_document_skeleton(artifact, None)
        for source, artifact in artifacts.items()
    }
    occurrence_ids = {
        (source, occurrence.unit_id): occurrence.occurrence_id
        for source, skeleton in skeletons.items()
        for occurrence in skeleton.occurrences
    }
    slot_keys = {
        (source, occurrence.unit_id): occurrence.slot_key
        for source, skeleton in skeletons.items()
        for occurrence in skeleton.occurrences
    }

    plan = build_candidate_merge_plan(
        _baseline_document(artifacts["grobid"]),
        artifacts,
        skeleton_occurrence_ids=occurrence_ids,
        skeleton_slot_keys=slot_keys,
    )

    assert plan.construction_counts.get("peer_skeleton_slot_mismatch", 0) == 0
    assert any(
        plan.store.candidate_metadata(candidate_id).source == "docling"
        and plan.store.candidate_metadata(candidate_id).candidate_type == "prose"
        for region in plan.graph.regions
        for path in region.valid_paths
        for candidate_id in path
    )


def test_bounded_composite_alignment_selects_source_backed_italics_across_1_to_n():
    baseline_artifact = SourceArtifact.from_text(
        "docling",
        "# Title\n\n## Results\n\nGene dpp is active and robust.\n",
    )
    marker_artifact = SourceArtifact.from_text(
        "marker",
        "# Title\n\n## Results\n\nGene *dpp* is active\n\nand robust.\n",
    )
    artifacts = {
        "docling": baseline_artifact,
        "marker": marker_artifact,
    }
    skeletons = {
        source: build_document_skeleton(artifact, None)
        for source, artifact in artifacts.items()
    }
    occurrence_ids = {
        (source, occurrence.unit_id): occurrence.occurrence_id
        for source, skeleton in skeletons.items()
        for occurrence in skeleton.occurrences
    }
    slot_keys = {
        (source, occurrence.unit_id): occurrence.slot_key
        for source, skeleton in skeletons.items()
        for occurrence in skeleton.occurrences
    }

    result = merge_with_baseline_failsafe(
        _baseline_document(baseline_artifact),
        artifacts,
        skeleton_occurrence_ids=occurrence_ids,
        skeleton_slot_keys=slot_keys,
    )

    assert "Gene *dpp* is active\n\nand robust." in result.document.text
    assert result.candidate_construction_counts["composite_alignment_region"] == 1
    assert result.candidate_construction_counts[
        "region_composite_source_backed_italic_preference"
    ] == 1


def test_bounded_composite_alignment_selects_source_backed_italics_across_n_to_1():
    baseline_artifact = SourceArtifact.from_text(
        "docling",
        "# Title\n\n## Results\n\nGene dpp is active\n\nand robust.\n",
    )
    marker_artifact = SourceArtifact.from_text(
        "marker",
        "# Title\n\n## Results\n\nGene *dpp* is active and robust.\n",
    )
    artifacts = {"docling": baseline_artifact, "marker": marker_artifact}
    skeletons = {
        source: build_document_skeleton(artifact, None)
        for source, artifact in artifacts.items()
    }

    result = merge_with_baseline_failsafe(
        _baseline_document(baseline_artifact),
        artifacts,
        skeleton_occurrence_ids={
            (source, occurrence.unit_id): occurrence.occurrence_id
            for source, skeleton in skeletons.items()
            for occurrence in skeleton.occurrences
        },
        skeleton_slot_keys={
            (source, occurrence.unit_id): occurrence.slot_key
            for source, skeleton in skeletons.items()
            for occurrence in skeleton.occurrences
        },
    )

    assert "Gene *dpp* is active and robust." in result.document.text
    assert result.candidate_construction_counts[
        "region_composite_source_backed_italic_preference"
    ] == 1


def test_bounded_composite_alignment_selects_source_backed_italics_across_n_to_m():
    baseline_artifact = SourceArtifact.from_text(
        "docling",
        (
            "# Title\n\n## Results\n\nGene dpp is active\n\n"
            "and robust in cells.\n\n## Discussion\n\nDone.\n"
        ),
    )
    marker_artifact = SourceArtifact.from_text(
        "marker",
        (
            "# Title\n\n## Results\n\nGene *dpp*\n\nis active and robust\n\n"
            "in cells.\n\n## Discussion\n\nDone.\n"
        ),
    )
    artifacts = {"docling": baseline_artifact, "marker": marker_artifact}
    skeletons = {
        source: build_document_skeleton(artifact, None)
        for source, artifact in artifacts.items()
    }

    result = merge_with_baseline_failsafe(
        _baseline_document(baseline_artifact),
        artifacts,
        skeleton_occurrence_ids={
            (source, occurrence.unit_id): occurrence.occurrence_id
            for source, skeleton in skeletons.items()
            for occurrence in skeleton.occurrences
        },
        skeleton_slot_keys={
            (source, occurrence.unit_id): occurrence.slot_key
            for source, skeleton in skeletons.items()
            for occurrence in skeleton.occurrences
        },
    )

    assert "Gene *dpp*\n\nis active and robust\n\nin cells." in result.document.text
    assert result.candidate_construction_counts["composite_alignment_region_nm"] == 1
    assert result.candidate_construction_counts[
        "region_composite_source_backed_italic_preference"
    ] == 1


def test_oversized_n_to_m_run_stays_out_of_composite_comparison():
    payload = "a" * 20_010
    baseline_artifact = SourceArtifact.from_text(
        "docling",
        f"# Title\n\n## Results\n\n{payload}\n\n## Discussion\n\nDone.\n",
    )
    marker_artifact = SourceArtifact.from_text(
        "marker",
        (
            f"# Title\n\n## Results\n\n{payload[:10_005]}\n\n"
            f"{payload[10_005:]}\n\n## Discussion\n\nDone.\n"
        ),
    )
    artifacts = {"docling": baseline_artifact, "marker": marker_artifact}
    skeletons = {
        source: build_document_skeleton(artifact, None)
        for source, artifact in artifacts.items()
    }

    plan = build_candidate_merge_plan(
        _baseline_document(baseline_artifact),
        artifacts,
        skeleton_occurrence_ids={
            (source, occurrence.unit_id): occurrence.occurrence_id
            for source, skeleton in skeletons.items()
            for occurrence in skeleton.occurrences
        },
        skeleton_slot_keys={
            (source, occurrence.unit_id): occurrence.slot_key
            for source, skeleton in skeletons.items()
            for occurrence in skeleton.occurrences
        },
    )

    assert plan.construction_counts["composite_oversized_run"] >= 1
    assert plan.construction_counts.get("composite_alignment_region", 0) == 0


@pytest.mark.parametrize(
    "collateral",
    [
        "and **robust**.",
        "and [robust](https://example.org).",
        "and `robust`.",
        "and <strong>robust</strong>.",
    ],
)
def test_composite_italic_preference_routes_collateral_markup_to_model(collateral):
    baseline_artifact = SourceArtifact.from_text(
        "docling",
        "# Title\n\n## Results\n\nGene dpp is active and robust.\n",
    )
    marker_artifact = SourceArtifact.from_text(
        "marker",
        f"# Title\n\n## Results\n\nGene *dpp* is active\n\n{collateral}\n",
    )
    artifacts = {"docling": baseline_artifact, "marker": marker_artifact}
    skeletons = {
        source: build_document_skeleton(artifact, None)
        for source, artifact in artifacts.items()
    }

    plan = build_candidate_merge_plan(
        _baseline_document(baseline_artifact),
        artifacts,
        skeleton_occurrence_ids={
            (source, occurrence.unit_id): occurrence.occurrence_id
            for source, skeleton in skeletons.items()
            for occurrence in skeleton.occurrences
        },
        skeleton_slot_keys={
            (source, occurrence.unit_id): occurrence.slot_key
            for source, skeleton in skeletons.items()
            for occurrence in skeleton.occurrences
        },
    )

    assert plan.construction_counts["composite_alignment_region"] == 1
    assert plan.construction_counts[
        "region_composite_model_selection_required"
    ] == 1
    assert "region-composite-0000" in plan.unresolved_region_ids
    assert (
        "region-composite-0000" not in plan.deterministic_reasons
    )


def test_composite_region_includes_conflicting_one_to_one_italic_peer():
    artifacts = {
        "docling": SourceArtifact.from_text(
            "docling",
            "# Title\n\n## Results\n\nGene dpp is active and robust.\n",
        ),
        "grobid": SourceArtifact.from_text(
            "grobid",
            "# Title\n\n## Results\n\nGene *dpp* is active and robust.\n",
        ),
        "marker": SourceArtifact.from_text(
            "marker",
            "# Title\n\n## Results\n\nGene dpp is active\n\nand *robust*.\n",
        ),
    }
    skeletons = {
        source: build_document_skeleton(artifact, None)
        for source, artifact in artifacts.items()
    }

    plan = build_candidate_merge_plan(
        _baseline_document(artifacts["docling"]),
        artifacts,
        skeleton_occurrence_ids={
            (source, occurrence.unit_id): occurrence.occurrence_id
            for source, skeleton in skeletons.items()
            for occurrence in skeleton.occurrences
        },
        skeleton_slot_keys={
            (source, occurrence.unit_id): occurrence.slot_key
            for source, skeleton in skeletons.items()
            for occurrence in skeleton.occurrences
        },
    )

    region = next(
        region
        for region in plan.selection_graph.regions
        if region.region_id.startswith("region-composite-")
    )
    candidate_sources = {
        plan.store.candidate_metadata(path[0]).source for path in region.valid_paths
    }
    assert candidate_sources == {"docling", "grobid", "marker"}
    assert plan.construction_counts["composite_aligned_peer_included"] == 1
    assert plan.construction_counts[
        "region_composite_model_selection_required"
    ] == 1
    assert region.region_id not in plan.deterministic_reasons


def test_reference_bullets_are_typed_as_reference_candidates():
    artifact = SourceArtifact.from_text(
        "docling",
        "# Title\n\n## References\n\n- Alpha et al. (2024). One.\n- Beta et al. (2025). Two.\n",
    )

    units = scan_structural_units(artifact)

    assert [unit.unit_type for unit in units[-2:]] == ["reference", "reference"]


def test_ordered_methods_steps_remain_lists_outside_references():
    artifact = SourceArtifact.from_text(
        "docling",
        "# Title\n\n## Methods\n\n1. First step.\n2. Second step.\n",
    )

    units = scan_structural_units(artifact)

    assert [unit.unit_type for unit in units[-2:]] == ["list", "list"]


@pytest.mark.parametrize("marker", ["-", "1."])
def test_reference_wrapped_continuation_stays_in_logical_entry(marker):
    second_marker = "-" if marker == "-" else "2."
    artifact = SourceArtifact.from_text(
        "docling",
        (
            "# Title\n\n## References\n\n"
            f"{marker} Alpha et al. (2024). Long title.\n"
            "continued journal details.\n"
            f"{second_marker} Beta et al. (2025). Two.\n"
        ),
    )

    units = [
        unit for unit in scan_structural_units(artifact) if unit.unit_type == "reference"
    ]
    first = artifact.raw_utf8[units[0].byte_start:units[0].byte_end].decode()

    assert len(units) == 2
    assert "continued journal details." in first


def test_bullet_reference_owns_author_year_continuation():
    artifact = SourceArtifact.from_text(
        "docling",
        (
            "# Title\n\n## References\n\n"
            "- Article title continues.\n"
            "Smith et al. (2024). Journal details.\n"
            "- Beta et al. (2025). Two.\n"
        ),
    )

    units = [
        unit for unit in scan_structural_units(artifact) if unit.unit_type == "reference"
    ]
    first = artifact.raw_utf8[units[0].byte_start:units[0].byte_end].decode()

    assert len(units) == 2
    assert "Smith et al. (2024)" in first


def test_numbered_reference_does_not_own_following_unnumbered_author_year_entry():
    artifact = SourceArtifact.from_text(
        "docling",
        (
            "# Title\n\n## References\n\n"
            "1. Alpha et al. (2024). One.\n"
            "Beta et al. (2025). Two.\n"
            "2. Gamma et al. (2026). Three.\n"
        ),
    )

    units = [
        unit for unit in scan_structural_units(artifact) if unit.unit_type == "reference"
    ]
    entries = [
        artifact.raw_utf8[unit.byte_start:unit.byte_end].decode()
        for unit in units
    ]

    assert len(units) == 3
    assert entries[1] == "Beta et al. (2025). Two."


def test_whole_bold_figure_caption_is_structurally_typed_as_figure():
    artifact = SourceArtifact.from_text(
        "marker",
        "# Title\n\n## Results\n\n**Figure 1. A complete caption.**\n",
    )

    units = scan_structural_units(artifact)

    assert units[-1].unit_type == "figure_caption"


def test_bracketed_numeric_reference_markers_split_logical_entries():
    artifact = SourceArtifact.from_text(
        "docling",
        "# Title\n\n## References\n\n[1] Alpha.\n[2] Beta.\n",
    )

    units = [
        unit for unit in scan_structural_units(artifact) if unit.unit_type == "reference"
    ]

    assert len(units) == 2


def test_unnumbered_author_year_references_still_split_entries():
    artifact = SourceArtifact.from_text(
        "docling",
        (
            "# Title\n\n## References\n\n"
            "Alpha et al. (2024). One.\n"
            "Beta et al. (2025). Two.\n"
        ),
    )

    units = [
        unit for unit in scan_structural_units(artifact) if unit.unit_type == "reference"
    ]

    assert len(units) == 2


def test_repetition_detector_reports_only_excess_over_all_sources():
    block = " ".join(f"token{index}" for index in range(20))
    source = SourceArtifact.from_text("grobid", block)
    assert repetition_diagnostics_metric(block, (source,)) == []
    diagnostics = repetition_diagnostics_metric(
        f"{block}\n\n{block}", (source,)
    )
    assert diagnostics
    assert all(item["excess_count"] >= 1 for item in diagnostics)


def _baseline_document(artifact: SourceArtifact) -> BaselineDocument:
    units = scan_structural_units(artifact)
    return BaselineDocument(
        artifact=artifact,
        units=units,
        score=BaselineScore(
            agreement=1.0,
            length_balance=1.0,
            structural_unit_count=len(units),
            heading_count=sum(unit.unit_type == "heading" for unit in units),
            non_whitespace_bytes=sum(
                not chr(byte).isspace() for byte in artifact.raw_utf8
            ),
        ),
    )


def _distinct_paragraphs(count: int, phrase: str) -> str:
    return "\n\n".join(
        " ".join(
            [f"Paragraph {index} {phrase}"]
            + [
                hashlib.sha256(f"{index}-{token}".encode()).hexdigest()[:16]
                for token in range(18)
            ]
        )
        for index in range(count)
    )


def test_alignment_does_not_disable_merging_for_real_article_sized_work():
    baseline_text = _distinct_paragraphs(150, "baseline evidence")
    alternative_text = baseline_text.replace(
        "baseline evidence", "baseline scientific evidence"
    )
    baseline = SourceArtifact.from_text("grobid", baseline_text)
    alternative = SourceArtifact.from_text("docling", alternative_text)

    mapping = _align_to_baseline(
        scan_structural_units(baseline),
        scan_structural_units(alternative),
    )

    assert len(mapping) == 150
    assert [mapping[index].alternative_index for index in sorted(mapping)] == list(
        range(150)
    )


def test_one_oversized_unit_stays_local_instead_of_disabling_alignment():
    oversized = "oversized " * 2_501
    baseline = SourceArtifact.from_text(
        "grobid", f"First unique paragraph.\n\n{oversized}\n\nLast unique paragraph."
    )
    alternative = SourceArtifact.from_text(
        "docling", f"First unique paragraph.\n\n{oversized}\n\nLast unique paragraph."
    )

    mapping = _align_to_baseline(
        scan_structural_units(baseline),
        scan_structural_units(alternative),
    )

    assert set(mapping) == {0, 2}
    assert mapping[0].alternative_index == 0
    assert mapping[2].alternative_index == 2


def test_alignment_can_treat_alternative_unit_type_as_diagnostic():
    baseline = SourceArtifact.from_text(
        "grobid", "Fig. 1. Gene dpp responds to treatment."
    )
    alternative = SourceArtifact.from_text(
        "marker", "Gene dpp responds to treatment."
    )
    baseline_units = scan_structural_units(baseline)
    alternative_units = scan_structural_units(alternative)
    assert baseline_units[0].unit_type == "figure_caption"
    assert alternative_units[0].unit_type == "paragraph"

    assert _align_to_baseline(baseline_units, alternative_units) == {}
    mapping = _align_to_baseline(
        baseline_units,
        alternative_units,
        require_matching_unit_types=False,
    )

    assert mapping[0].alternative_index == 0


def test_normalized_near_agreement_retains_baseline_without_model_region():
    baseline_artifact = SourceArtifact.from_text(
        "grobid", "# Title\n\nGene dpp is active."
    )
    alternative_artifact = SourceArtifact.from_text(
        "docling", "#   Title\n\nGene   dpp is active."
    )

    plan = build_candidate_merge_plan(
        _baseline_document(baseline_artifact),
        {
            "grobid": baseline_artifact,
            "docling": alternative_artifact,
        },
    )

    assert plan.graph.regions == ()
    assert plan.selection_graph.regions == ()
    assert plan.construction_counts["baseline_normalized_near_agreement"] == 2


def test_scope_proof_is_bounded_to_a_local_neighborhood_for_long_documents():
    text = "\n\n".join(f"uniqueunit{index:05d}" for index in range(4_200))
    baseline = SourceArtifact.from_text("grobid", text)
    alternative = SourceArtifact.from_text("docling", text)
    baseline_units = scan_structural_units(baseline)
    alternative_units = scan_structural_units(alternative)
    target_index = 2_100

    assert not _has_nonlocal_scope_spill(
        unit=alternative_units[target_index],
        artifact=alternative,
        baseline_units=baseline_units,
        baseline_artifact=baseline,
        baseline_index=target_index,
    )


def test_large_structural_alignment_exposes_source_backed_candidate_regions():
    baseline_text = _distinct_paragraphs(120, "shared evidence")
    alternative_text = baseline_text.replace(
        "shared evidence", "shared scientific evidence"
    )
    baseline_artifact = SourceArtifact.from_text("grobid", baseline_text)
    alternative_artifact = SourceArtifact.from_text("docling", alternative_text)
    baseline = _baseline_document(baseline_artifact)

    plan = build_candidate_merge_plan(
        baseline,
        {
            "grobid": baseline_artifact,
            "docling": alternative_artifact,
        },
    )

    assert len(plan.graph.regions) == 120
    assert len(plan.selection_graph.regions) == 120
