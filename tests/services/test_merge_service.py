from pathlib import Path
import copy
import hashlib
import json

import pytest

import app.services.merge_service as merge_service_module
from app.services.merge_service import (
    BoundedCandidateSelector,
    completion_evidence_for_finished_artifacts,
    merge_source_artifacts,
    merge_finished_extractor_outputs,
)
from app.services.source_contracts import (
    CandidateGraph,
    CandidateStore,
    RegionCandidateGraph,
    RegionSelectionDecision,
    SelectionDecisionResponse,
    SourceArtifact,
    SourceSpanCandidate,
)
from app.services.llm_service import CandidateSelectionFailure
from app.services.document_skeleton import NativeStructureArtifact, build_document_skeleton
from app.services.merge_artifact import persist_merge_bundle, validate_merge_artifacts
from config import Config
from app.services.page_coverage import (
    PAGE_COVERAGE_METHOD,
    page_coverage_proof_digest,
)
from app.services.source_merge import BaselineCompletionEvidence, BaselineRequirements


FRAGMENT_REQUIREMENTS = BaselineRequirements(
    minimum_words=1,
    minimum_structural_units=1,
    minimum_non_whitespace_bytes=1,
    require_heading_or_five_units=False,
    required_heading_groups=(),
    require_abc_validation=False,
)


class _Usage:
    def summary(self):
        return {"total": {"calls": 0}}


class FakeSelectorLLM:
    def __init__(
        self,
        *,
        terra_failure=None,
        sol_failure=None,
        terra_keep=False,
        skeleton_choice="baseline",
    ):
        self.terra_failure = terra_failure
        self.sol_failure = sol_failure
        self.terra_keep = terra_keep
        self.terra_calls = 0
        self.sol_calls = 0
        self.skeleton_calls = 0
        self.skeleton_choice = skeleton_choice
        self.skeleton_choice_payload = None
        self.usage = _Usage()
        self.selection_call_traces = []

    def resolve_bounded_id_choice(self, *, reason, baseline_id, choices):
        self.skeleton_calls += 1
        self.skeleton_choice_payload = {
            "reason": reason,
            "baseline_id": baseline_id,
            "choices": choices,
        }
        if self.sol_failure:
            raise CandidateSelectionFailure(self.sol_failure)
        if self.skeleton_choice == "alternative":
            return next(
                choice["candidate_id"]
                for choice in choices
                if choice["candidate_id"] != baseline_id
            )
        return baseline_id

    def resolve_bounded_id_choice_with_receipt(
        self, *, reason, baseline_id, choices
    ):
        selected_id = self.resolve_bounded_id_choice(
            reason=reason,
            baseline_id=baseline_id,
            choices=choices,
        )
        ordered_ids = [baseline_id] + sorted(
            choice["candidate_id"]
            for choice in choices
            if choice["candidate_id"] != baseline_id
        )
        response_choice = ordered_ids.index(selected_id)
        request_sha256 = hashlib.sha256(
            f"{reason}\0{baseline_id}\0{'\0'.join(ordered_ids)}".encode()
        ).hexdigest()
        self.selection_call_traces.append({
            "tier": "sol",
            "call_type": "bounded_id_selection_sol",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "outcome": "valid",
            "request": {
                "request_sha256": request_sha256,
                "regions": [{
                    "region_id": reason,
                    "keep_baseline_choice": 0,
                    "baseline_candidate_id": baseline_id,
                    "candidates": [
                        {"candidate_id": candidate_id}
                        for candidate_id in ordered_ids
                    ],
                    "path_choices": [
                        {"choice": index, "candidate_ids": [candidate_id]}
                        for index, candidate_id in enumerate(ordered_ids[1:], 1)
                    ],
                }],
            },
            "response": {
                "request_sha256": request_sha256,
                "decisions": [{
                    "region_id": reason,
                    "choice": response_choice,
                }],
            },
        })
        return {
            "selected_candidate_id": selected_id,
            "request_sha256": request_sha256,
            "response_choice": response_choice,
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
        }

    @staticmethod
    def _choose_alternative(graph):
        decisions = []
        for region in graph.regions:
            path = next(
                path
                for path in region.valid_paths
                if path != (region.baseline_candidate_id,)
            )
            decisions.append(
                RegionSelectionDecision(
                    region_id=region.region_id,
                    action="select_candidates",
                    candidate_ids=path,
                )
            )
        return SelectionDecisionResponse(decisions=tuple(decisions))

    def resolve_candidate_selections(
        self,
        _store,
        graph,
        *,
        use_sol=False,
        raise_on_failure=False,
    ):
        if not use_sol:
            self.terra_calls += 1
            if self.terra_failure:
                raise CandidateSelectionFailure(self.terra_failure)
            if self.terra_keep:
                return SelectionDecisionResponse(
                    decisions=tuple(
                        RegionSelectionDecision(
                            region_id=region.region_id,
                            action="keep_baseline",
                        )
                        for region in graph.regions
                    )
                )
        else:
            self.sol_calls += 1
            if self.sol_failure:
                raise CandidateSelectionFailure(self.sol_failure)
        return self._choose_alternative(graph)


def _artifacts():
    return {
        "grobid": SourceArtifact.from_text("grobid", "# Title\n\nGene Gga is active."),
        "docling": SourceArtifact.from_text(
            "docling", "# Title\n\nGene Gγa is active."
        ),
        "marker": SourceArtifact.from_text(
            "marker", "# Title\n\nGene G g a is active."
        ),
    }


def _merge(llm):
    artifacts = _artifacts()
    return merge_source_artifacts(
        artifacts["grobid"].text,
        artifacts["docling"].text,
        artifacts["marker"].text,
        llm,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )


def _selector_fixture(
    texts,
    *,
    candidate_type="prose",
    emphasis_profiles=None,
):
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source, text in texts.items()
    }
    candidates = {}
    paths = []
    for source, artifact in artifacts.items():
        candidate_id = f"candidate-{source}"
        formatting = (
            {
                "visible_text_digest": "a" * 64,
                "non_emphasis_ast_digest": "b" * 64,
                "emphasis_occurrence_ids": tuple(
                    (emphasis_profiles or {}).get(source, ())
                ),
            }
            if emphasis_profiles is not None
            else {}
        )
        candidates[candidate_id] = SourceSpanCandidate(
            candidate_id=candidate_id,
            occurrence_id=f"occurrence-{source}",
            structural_unit_id="unit-1",
            source=source,
            artifact_digest=artifact.digest,
            byte_start=0,
            byte_end=len(artifact.raw_utf8),
            candidate_type=candidate_type,
            comparison_key=texts[source],
            **formatting,
        )
        paths.append((candidate_id,))
    store = CandidateStore(
        {(source, artifact.digest): artifact for source, artifact in artifacts.items()},
        candidates,
    )
    graph = CandidateGraph(
        regions=(
            RegionCandidateGraph(
                region_id="region-0001",
                baseline_candidate_id="candidate-grobid",
                valid_paths=tuple(paths),
            ),
        )
    )
    return store, graph


def _page_verified_evidence(artifacts):
    pdf_digest = "a" * 64
    return {
        source: BaselineCompletionEvidence(
            artifact_digest=artifact.digest,
            extraction_succeeded=True,
            artifact_complete=True,
            expected_page_count=1,
            covered_page_count=1,
            pdf_digest=pdf_digest,
            coverage_method=PAGE_COVERAGE_METHOD,
            page_coverage_digest=page_coverage_proof_digest(
                source=source,
                artifact_digest=artifact.digest,
                pdf_digest=pdf_digest,
                expected_page_count=1,
                covered_page_count=1,
            ),
            completion_basis="page_coverage",
        )
        for source, artifact in artifacts.items()
    }


def _minimal_native_structures(artifacts):
    title = "Example title"
    return {
        "grobid": NativeStructureArtifact.for_test(
            "grobid",
            artifacts["grobid"],
            (
                '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
                f"<titleStmt><title>{title}</title></titleStmt>"
                "</fileDesc></teiHeader><text><body/></text></TEI>"
            ).encode("utf-8"),
        ),
        "docling": NativeStructureArtifact.for_test(
            "docling",
            artifacts["docling"],
            json.dumps({
                "schema_name": "DoclingDocument",
                "texts": [{
                    "self_ref": "#/texts/0",
                    "label": "title",
                    "text": title,
                }],
            }).encode("utf-8"),
        ),
        "marker": NativeStructureArtifact.for_test(
            "marker",
            artifacts["marker"],
            json.dumps({
                "block_type": "Document",
                "children": [{
                    "id": "/page/0/Title/0",
                    "block_type": "Title",
                    "html": f"<h1>{title}</h1>",
                }],
            }).encode("utf-8"),
        ),
    }


def test_candidate_adapter_uses_selection_only_terra_and_emits_span_audit():
    llm = FakeSelectorLLM()

    merged, metrics, audit = _merge(llm)

    assert merged in {
        "# Title\n\nGene Gga is active.\n",
        "# Title\n\nGene Gγa is active.\n",
        "# Title\n\nGene G g a is active.\n",
    }
    assert metrics["merge_contract_id"] == "pdfx-native-skeleton-selection"
    assert metrics["merge_quality"] == "terra_selected"
    assert metrics["runtime_models"]["source_selection"]["model"] == "gpt-5.6-terra"
    assert metrics["qualification_outcome"] == "failsafe"
    assert "page_coverage_unverified" in metrics["qualification_reasons"]
    assert metrics["repetition_diagnostics"] == []
    assert llm.terra_calls == 1
    assert llm.sol_calls == 0
    assert audit
    assert all("text" not in entry and "chosen_text" not in entry for entry in audit)
    assert any(entry["decision_method"] == "model_selected" for entry in audit)
    assert metrics["candidate_region_count"] == 1
    assert metrics["model_call_attempts"] == 0  # fake selector has no transport trace
    assert metrics["region_decision_counts"] == {"model_selected": 1}
    region = metrics["region_decisions"][0]
    assert region["decision_reason"] == "model_selected_numbered_path"
    assert region["selected_choice"] in {1, 2}
    assert all("text" not in candidate for candidate in region["candidates"])
    assert any(
        candidate["content_sha256"] for candidate in region["candidates"]
    )
    assert any(
        item["outcome"] == "selected"
        for item in metrics["baseline_selection_trace"]
    )


def test_marker_native_body_italics_are_selected_published_and_reconciled():
    grobid = "# Title\n\n## Results\n\nGene dpp is active.\n"
    marker = "# Title\n\n## Results\n\nGene *dpp* is active.\n"
    artifacts = {
        "grobid": SourceArtifact.from_text("grobid", grobid),
        "docling": SourceArtifact.from_text("docling", grobid),
        "marker": SourceArtifact.from_text("marker", marker),
    }
    marker_native = NativeStructureArtifact.for_test(
        "marker",
        artifacts["marker"],
        json.dumps({
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {"id": "/page/0/Title/0", "block_type": "Title", "html": "<h1>Title</h1>"},
                    {"id": "/page/0/SectionHeader/1", "block_type": "SectionHeader", "html": "<h2>Results</h2>"},
                    {"id": "/page/0/Text/2", "block_type": "Text", "html": "<p>Gene <i>dpp</i> is active.</p>"},
                ],
            }],
        }).encode("utf-8"),
    )

    merged, metrics, _audit = merge_source_artifacts(
        grobid,
        grobid,
        marker,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        native_structures={"marker": marker_native},
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    receipt = metrics["italic_preservation"]
    assert "Gene *dpp* is active." in merged
    assert receipt["native_body_emphasis_count"] == 1
    assert receipt["mapped_native_body_emphasis_count"] == 1
    assert receipt["retained_native_body_emphasis_count"] == 1
    assert receipt["native_body_exclusion_reason_counts"] == {}
    assert receipt["policy_version"] == "positive-style-overlay-v1"
    assert len(receipt["protected_claim_ids_sha256"]) == 64
    assert receipt["canonical_output_emphasis_interval_count"] == 1


def test_baseline_retained_native_italics_reconcile_without_replacement():
    marker = "# Title\n\n## Results\n\nGene *dpp* is active.\n"
    artifact = SourceArtifact.from_text("marker", marker)
    marker_native = NativeStructureArtifact.for_test(
        "marker",
        artifact,
        json.dumps({
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {"id": "/page/0/Title/0", "block_type": "Title", "html": "<h1>Title</h1>"},
                    {"id": "/page/0/SectionHeader/1", "block_type": "SectionHeader", "html": "<h2>Results</h2>"},
                    {"id": "/page/0/Text/2", "block_type": "Text", "html": "<p>Gene <em>dpp</em> is active.</p>"},
                ],
            }],
        }).encode("utf-8"),
    )

    merged, metrics, _audit = merge_source_artifacts(
        "",
        "",
        marker,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(
            {"marker": artifact}
        ),
        native_structures={"marker": marker_native},
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    receipt = metrics["italic_preservation"]
    assert merged == marker
    assert receipt["retained_native_body_emphasis_count"] == 1
    assert receipt["canonical_output_emphasis_interval_count"] == 1
    assert receipt["auxiliary_positive_emphasis_count"] == 0
    assert receipt["all_native_body_italics_retained"] is True


def test_repetition_fallback_closes_style_ledger_and_delivers_baseline(
    monkeypatch,
    tmp_path,
):
    marker = "# Title\n\n## Results\n\nGene dpp is active.\n"
    artifact = SourceArtifact.from_text("marker", marker)
    marker_native = NativeStructureArtifact.for_test(
        "marker",
        artifact,
        json.dumps({
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {
                        "id": "/page/0/Title/0",
                        "block_type": "Title",
                        "html": "<h1>Title</h1>",
                    },
                    {
                        "id": "/page/0/SectionHeader/1",
                        "block_type": "SectionHeader",
                        "html": "<h2>Results</h2>",
                    },
                    {
                        "id": "/page/0/Text/2",
                        "block_type": "Text",
                        "html": "<p>Gene <i>dpp</i> is active.</p>",
                    },
                ],
            }],
        }).encode("utf-8"),
    )
    calls = 0

    def repetition_once(_text, _artifacts):
        nonlocal calls
        calls += 1
        return [{"kind": "paragraph"}] if calls == 1 else []

    monkeypatch.setattr(
        merge_service_module,
        "repetition_diagnostics_metric",
        repetition_once,
    )

    merged, metrics, audit = merge_source_artifacts(
        "",
        "",
        marker,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(
            {"marker": artifact}
        ),
        native_structures={"marker": marker_native},
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert merged == marker
    assert metrics["merge_quality"] == "baseline_fallback"
    assert "rendered_output_rejected:excess_repetition" in metrics["warnings"]
    assert metrics["repetition_diagnostics"] == []
    receipt = metrics["italic_preservation"]
    assert receipt["native_body_emphasis_count"] == 1
    assert receipt["retained_native_body_emphasis_count"] == 0
    assert receipt["native_body_exclusion_reason_counts"] == {
        "baseline_fallback_after_repetition": 1
    }
    fallback_events = [
        event
        for event in metrics["document_skeleton_transformations"]
        if event.get("reconciliation_method")
        == "baseline-fallback-style-ledger-v1"
    ]
    assert len(fallback_events) == 1
    assert fallback_events[0]["outcome"] == "declined"
    assert fallback_events[0]["fallback_output_sha256"] == metrics["output_digest"]
    assert not any(
        entry.get("transformation") == "native_emphasis_projection"
        for entry in audit
    )
    skeletons = {"marker": build_document_skeleton(artifact, marker_native)}
    manifest_path = persist_merge_bundle(
        merged_path=str(tmp_path / "merged.md"),
        metrics_path=str(tmp_path / "metrics.json"),
        audit_path=str(tmp_path / "audit.json"),
        text=merged,
        metrics=metrics,
        audit=audit,
        artifacts={"marker": artifact},
        skeletons=skeletons,
        expected_contract_id=Config.MERGE_CONTRACT_ID,
    )
    assert Path(manifest_path).is_file()

    tampered = copy.deepcopy(metrics)
    next(
        event
        for event in tampered["document_skeleton_transformations"]
        if event.get("reconciliation_method")
        == "baseline-fallback-style-ledger-v1"
    )["fallback_output_sha256"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="fallback style reconciliation replay failed",
    ):
        validate_merge_artifacts(
            merged,
            tampered,
            audit,
            artifacts={"marker": artifact},
            expected_contract_id=Config.MERGE_CONTRACT_ID,
            skeletons=skeletons,
        )


def test_explicit_marker_emphasis_projects_onto_one_exact_plain_target():
    marker = "# Title\n\n## Results\n\nGene dpp is active.\n"
    artifact = SourceArtifact.from_text("marker", marker)
    marker_native = NativeStructureArtifact.for_test(
        "marker",
        artifact,
        json.dumps({
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {
                        "id": "/page/0/Title/0",
                        "block_type": "Title",
                        "html": "<h1>Title</h1>",
                    },
                    {
                        "id": "/page/0/SectionHeader/1",
                        "block_type": "SectionHeader",
                        "html": "<h2>Results</h2>",
                    },
                    {
                        "id": "/page/0/Text/2",
                        "block_type": "Text",
                        "html": "<p>Gene <i>dpp</i> is active.</p>",
                    },
                ],
            }],
        }).encode("utf-8"),
    )

    merged, metrics, audit = merge_source_artifacts(
        "",
        "",
        marker,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(
            {"marker": artifact}
        ),
        native_structures={"marker": marker_native},
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert merged == "# Title\n\n## Results\n\nGene *dpp* is active.\n"
    assert metrics["italic_preservation"][
        "retained_native_body_emphasis_count"
    ] == 1
    assert metrics["italic_preservation"][
        "native_body_exclusion_reason_counts"
    ] == {}
    projection_events = [
        event
        for event in metrics["document_skeleton_transformations"]
        if event["operation"] == "native_emphasis_projection"
        and event.get("audit_span_emitted") is True
    ]
    assert [event["boundary"] for event in projection_events] == ["open", "close"]
    assert sum(
        entry.get("transformation") == "native_emphasis_projection"
        for entry in audit
    ) == 2
    assert all(
        event["projection_reconciled"] is True
        and event["native_receipt_digest"] == marker_native.receipt_digest
        and event["native_artifact_digest"] == marker_native.native_digest
        and event["target_artifact_digest"] == artifact.digest
        for event in projection_events
    )
    tampered = copy.deepcopy(metrics)
    for event in tampered["document_skeleton_transformations"]:
        if event.get("operation") == "native_emphasis_projection" and event.get(
            "audit_span_emitted"
        ):
            event["target_emphasis_occurrence_id"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="complete reconciliation replay failed|target identity|identity is invalid",
    ):
        validate_merge_artifacts(
            merged,
            tampered,
            audit,
            artifacts={"marker": artifact},
            expected_contract_id=Config.MERGE_CONTRACT_ID,
            skeletons={"marker": build_document_skeleton(artifact, marker_native)},
        )
    tampered = copy.deepcopy(metrics)
    for event in tampered["document_skeleton_transformations"]:
        if event.get("operation") == "native_emphasis_projection" and event.get(
            "audit_span_emitted"
        ):
            event["anchor_region_digest"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="complete reconciliation replay failed|correspondence replay",
    ):
        validate_merge_artifacts(
            merged,
            tampered,
            audit,
            artifacts={"marker": artifact},
            expected_contract_id=Config.MERGE_CONTRACT_ID,
            skeletons={"marker": build_document_skeleton(artifact, marker_native)},
        )


def test_docling_style_projection_passes_full_replay_and_rejects_digest_tamper():
    text = "# Title\n\n## Results\n\nGene dpp is active.\n"
    artifact = SourceArtifact.from_text("docling", text)
    native_bytes = json.dumps({
        "schema_name": "DoclingDocument",
        "body": {"children": [
            {"$ref": "#/texts/0"},
            {"$ref": "#/texts/1"},
            {"$ref": "#/texts/2"},
        ]},
        "texts": [
            {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
            {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
            {"self_ref": "#/texts/2", "label": "text", "text": "Gene dpp is active."},
        ],
    }).encode("utf-8")
    style_bytes = json.dumps({
        "schema": "pdfx-native-style",
        "contract_version": "native-style-v1",
        "source": "docling",
        "status": "available",
        "pages": [{
            "page_no": 1,
            "lines": [{
                "native_id": "docling-line-1",
                "text": "Gene dpp is active.",
                "italic_spans": [{
                    "start": 5,
                    "end": 8,
                    "styles": ["Times-Italic"],
                }],
            }],
        }],
    }).encode("utf-8")
    docling_native = NativeStructureArtifact.for_test(
        "docling", artifact, native_bytes, style_bytes
    )
    skeleton = build_document_skeleton(artifact, docling_native)

    merged, metrics, audit = merge_source_artifacts(
        "",
        text,
        "",
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(
            {"docling": artifact}
        ),
        native_structures={"docling": docling_native},
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert merged == "# Title\n\n## Results\n\nGene *dpp* is active.\n"
    assert validate_merge_artifacts(
        merged,
        metrics,
        audit,
        artifacts={"docling": artifact},
        expected_contract_id=Config.MERGE_CONTRACT_ID,
        skeletons={"docling": skeleton},
    ) == metrics["output_digest"]

    tampered = copy.deepcopy(metrics)
    for event in tampered["document_skeleton_transformations"]:
        if (
            event.get("operation") == "native_emphasis_projection"
            and event.get("audit_span_emitted") is True
        ):
            event["native_style_digest"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="complete reconciliation replay failed|donor evidence is unbound",
    ):
        validate_merge_artifacts(
            merged,
            tampered,
            audit,
            artifacts={"docling": artifact},
            expected_contract_id=Config.MERGE_CONTRACT_ID,
            skeletons={"docling": skeleton},
        )
    tampered = copy.deepcopy(metrics)
    for event in tampered["document_skeleton_transformations"]:
        if (
            event.get("operation") == "native_emphasis_projection"
            and event.get("audit_span_emitted") is True
        ):
            event["donor_spine_placement_digest"] = "0" * 64
    with pytest.raises(
        ValueError,
        match=(
            "complete reconciliation replay failed|donor evidence is unbound|"
            "identity is invalid"
        ),
    ):
        validate_merge_artifacts(
            merged,
            tampered,
            audit,
            artifacts={"docling": artifact},
            expected_contract_id=Config.MERGE_CONTRACT_ID,
            skeletons={"docling": skeleton},
        )


def test_identical_grobid_docling_style_support_reconciles_without_double_markup():
    text = "# Title\n\n## Results\n\nGene dpp is active.\n"
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("grobid", "docling")
    }
    native_payloads = {
        "grobid": b"<TEI xmlns='http://www.tei-c.org/ns/1.0'><teiHeader><fileDesc><titleStmt><title>Title</title></titleStmt></fileDesc></teiHeader><text><body><div><head>Results</head><p>Gene dpp is active.</p></div></body></text></TEI>",
        "docling": json.dumps({
            "schema_name": "DoclingDocument",
            "body": {"children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"},
            ]},
            "texts": [
                {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
                {"self_ref": "#/texts/2", "label": "text", "text": "Gene dpp is active."},
            ],
        }).encode("utf-8"),
    }
    natives = {}
    for source in artifacts:
        style_bytes = json.dumps({
            "schema": "pdfx-native-style",
            "contract_version": "native-style-v1",
            "source": source,
            "status": "available",
            "pages": [{
                "page_no": 1,
                "lines": [{
                    "native_id": f"{source}-line-1",
                    "text": "Gene dpp is active.",
                    "italic_spans": [{
                        "start": 5,
                        "end": 8,
                        "styles": ["Times-Italic"],
                    }],
                }],
            }],
        }).encode("utf-8")
        natives[source] = NativeStructureArtifact.for_test(
            source,
            artifacts[source],
            native_payloads[source],
            style_bytes,
        )
    skeletons = {
        source: build_document_skeleton(artifacts[source], natives[source])
        for source in artifacts
    }

    merged, metrics, audit = merge_source_artifacts(
        text,
        text,
        "",
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        native_structures=natives,
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert merged == "# Title\n\n## Results\n\nGene *dpp* is active.\n"
    assert sum(
        entry.get("transformation") == "native_emphasis_projection"
        for entry in audit
    ) == 2
    receipt = metrics["italic_preservation"]
    assert receipt["native_body_emphasis_count"] == 2
    assert receipt["retained_native_body_emphasis_count"] == 2
    assert receipt["excluded_native_body_emphasis_count"] == 0
    assert receipt["all_native_body_italics_retained"] is True
    assert {
        source: counts["retained_native_body_emphasis_count"]
        for source, counts in receipt["source_counts"].items()
    } == {"docling": 1, "grobid": 1}
    assert metrics["native_emphasis_projection"][
        "supported_occurrence_count"
    ] == 1
    assert validate_merge_artifacts(
        merged,
        metrics,
        audit,
        artifacts=artifacts,
        expected_contract_id=Config.MERGE_CONTRACT_ID,
        skeletons=skeletons,
    ) == metrics["output_digest"]

    tampered = copy.deepcopy(metrics)
    support_event = next(
        event
        for event in tampered["document_skeleton_transformations"]
        if event.get("outcome") == "supported"
    )
    support_event["supported_projection_id"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="complete reconciliation replay failed|identical support is unbound",
    ):
        validate_merge_artifacts(
            merged,
            tampered,
            audit,
            artifacts=artifacts,
            expected_contract_id=Config.MERGE_CONTRACT_ID,
            skeletons=skeletons,
        )
    tampered = copy.deepcopy(metrics)
    projected_event = next(
        event
        for event in tampered["document_skeleton_transformations"]
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    )
    projected_event["supporting_sources"] = [projected_event["donor_source"]]
    with pytest.raises(
        ValueError,
        match=(
            "complete reconciliation replay failed|delimiter pair is invalid|"
            "identical support is unbound|source support does not reconcile"
        ),
    ):
        validate_merge_artifacts(
            merged,
            tampered,
            audit,
            artifacts=artifacts,
            expected_contract_id=Config.MERGE_CONTRACT_ID,
            skeletons=skeletons,
        )


def test_aggressive_cross_source_conflict_replays_and_rejects_reason_tamper():
    text = "# Title\n\n## Results\n\nGene dpp is active.\n"
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("marker", "docling")
    }
    marker_native = NativeStructureArtifact.for_test(
        "marker",
        artifacts["marker"],
        json.dumps(
            {
                "block_type": "Document",
                "children": [
                    {
                        "block_type": "Page",
                        "children": [
                            {
                                "id": "/page/0/Title/0",
                                "block_type": "Title",
                                "html": "<h1>Title</h1>",
                            },
                            {
                                "id": "/page/0/SectionHeader/1",
                                "block_type": "SectionHeader",
                                "html": "<h2>Results</h2>",
                            },
                            {
                                "id": "/page/0/Text/2",
                                "block_type": "Text",
                                "html": "<p>Gene <i>dpp</i> is active.</p>",
                            },
                        ],
                    }
                ],
            }
        ).encode("utf-8"),
    )
    docling_style = json.dumps(
        {
            "schema": "pdfx-native-style",
            "contract_version": "native-style-v1",
            "source": "docling",
            "status": "available",
            "pages": [
                {
                    "page_no": 1,
                    "lines": [
                        {
                            "native_id": "docling-line-1",
                            "text": "Gene dpp is active.",
                            "italic_spans": [
                                {
                                    "start": 5,
                                    "end": 18,
                                    "styles": ["Times-Italic"],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ).encode("utf-8")
    docling_native = NativeStructureArtifact.for_test(
        "docling",
        artifacts["docling"],
        json.dumps(
            {
                "schema_name": "DoclingDocument",
                "texts": [
                    {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                    {
                        "self_ref": "#/texts/1",
                        "label": "section_header",
                        "text": "Results",
                    },
                    {
                        "self_ref": "#/texts/2",
                        "label": "text",
                        "text": "Gene dpp is active.",
                    },
                ],
            }
        ).encode("utf-8"),
        docling_style,
    )
    natives = {"marker": marker_native, "docling": docling_native}
    skeletons = {
        source: build_document_skeleton(artifacts[source], natives[source])
        for source in artifacts
    }

    merged, metrics, audit = merge_source_artifacts(
        "",
        text,
        text,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        native_structures=natives,
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert merged == text.replace("dpp is active", "*dpp is active*")
    narrower_supports = [
        event
        for event in metrics["document_skeleton_transformations"]
        if event.get("reason") == "canonical_interval_supported"
    ]
    assert {event["donor_source"] for event in narrower_supports} == {"marker"}
    assert validate_merge_artifacts(
        merged,
        metrics,
        audit,
        artifacts=artifacts,
        expected_contract_id=Config.MERGE_CONTRACT_ID,
        skeletons=skeletons,
    ) == metrics["output_digest"]

    tampered = copy.deepcopy(metrics)
    next(
        event
        for event in tampered["document_skeleton_transformations"]
        if event.get("reason") == "canonical_interval_supported"
    )["reason"] = "invented_conflict_reason"
    with pytest.raises(ValueError, match="complete reconciliation replay failed"):
        validate_merge_artifacts(
            merged,
            tampered,
            audit,
            artifacts=artifacts,
            expected_contract_id=Config.MERGE_CONTRACT_ID,
            skeletons=skeletons,
        )


def test_projection_declines_when_delimiters_would_remain_literal_markdown():
    marker = "# Title\n\n## Results\n\nA.B\n"
    artifact = SourceArtifact.from_text("marker", marker)
    marker_native = NativeStructureArtifact.for_test(
        "marker",
        artifact,
        json.dumps({
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {
                        "id": "/page/0/Title/0",
                        "block_type": "Title",
                        "html": "<h1>Title</h1>",
                    },
                    {
                        "id": "/page/0/SectionHeader/1",
                        "block_type": "SectionHeader",
                        "html": "<h2>Results</h2>",
                    },
                    {
                        "id": "/page/0/Text/2",
                        "block_type": "Text",
                        "html": "<p>A<i>.</i>B</p>",
                    },
                ],
            }],
        }).encode("utf-8"),
    )

    merged, metrics, audit = merge_source_artifacts(
        "",
        "",
        marker,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(
            {"marker": artifact}
        ),
        native_structures={"marker": marker_native},
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert merged == marker
    assert not any(
        entry.get("transformation") == "native_emphasis_projection"
        for entry in audit
    )
    assert metrics["native_emphasis_projection"] == {
        "policy_version": "post-merge-positive-style-overlay-v1",
        "eligible_occurrence_count": 1,
        "projected_reconciled_occurrence_count": 0,
        "supported_occurrence_count": 0,
            "declined_occurrence_count": 1,
            "decline_reason_counts": {"character_interval_unmapped": 1},
            "finite_model_selection_tie_count": 0,
            "numbered_model_eligible_tie_count": 0,
            "model_selection_call_count": 0,
            "model_selected_target_count": 0,
            "model_selection_outcome_counts": {},
        }
    assert metrics["italic_preservation"][
        "retained_native_body_emphasis_count"
    ] == 0


def test_invalid_emphasis_does_not_discard_valid_sibling_in_same_paragraph():
    marker = "# Title\n\n## Results\n\nn = 8 and A.B\n"
    artifact = SourceArtifact.from_text("marker", marker)
    marker_native = NativeStructureArtifact.for_test(
        "marker",
        artifact,
        json.dumps({
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {
                        "id": "/page/0/Title/0",
                        "block_type": "Title",
                        "html": "<h1>Title</h1>",
                    },
                    {
                        "id": "/page/0/SectionHeader/1",
                        "block_type": "SectionHeader",
                        "html": "<h2>Results</h2>",
                    },
                    {
                        "id": "/page/0/Text/2",
                        "block_type": "Text",
                        "html": "<p><i>n</i> = 8 and A<i>.</i>B</p>",
                    },
                ],
            }],
        }).encode("utf-8"),
    )

    merged, metrics, _audit_result = merge_source_artifacts(
        "",
        "",
        marker,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(
            {"marker": artifact}
        ),
        native_structures={"marker": marker_native},
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert merged == marker.replace("n =", "*n* =")
    assert metrics["italic_preservation"][
        "retained_native_body_emphasis_count"
    ] == 1


def test_native_claim_is_added_without_disturbing_an_unrelated_markdown_italic():
    marker = "# Title\n\n## Results\n\nGene dpp and *control*.\n"
    artifact = SourceArtifact.from_text("marker", marker)
    native = NativeStructureArtifact.for_test(
        "marker",
        artifact,
        json.dumps({
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {"id": "/page/0/Title/0", "block_type": "Title", "html": "<h1>Title</h1>"},
                    {"id": "/page/0/SectionHeader/1", "block_type": "SectionHeader", "html": "<h2>Results</h2>"},
                    {"id": "/page/0/Text/2", "block_type": "Text", "html": "<p>Gene <i>dpp</i> and control.</p>"},
                ],
            }],
        }).encode("utf-8"),
    )

    merged, metrics, _audit = merge_source_artifacts(
        "",
        "",
        marker,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(
            {"marker": artifact}
        ),
        native_structures={"marker": native},
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    receipt = metrics["italic_preservation"]
    assert merged == "# Title\n\n## Results\n\nGene *dpp* and *control*.\n"
    assert receipt["native_body_emphasis_count"] == 1
    assert receipt["retained_native_body_emphasis_count"] == 1
    assert receipt["markdown_only_body_emphasis_count"] == 1
    assert receipt["native_body_exclusion_reason_counts"] == {}
    assert receipt["all_native_body_italics_retained"] is True


def test_protected_numeric_payload_routes_directly_to_bounded_sol():
    llm = FakeSelectorLLM()
    artifacts = {
        source: SourceArtifact.from_text(
            source,
            f"# Title\n\nThe measured value was {value} nM.",
        )
        for source, value in (
            ("grobid", "10"),
            ("docling", "1.0"),
            ("marker", "100"),
        )
    }

    merged, metrics, _audit = merge_source_artifacts(
        artifacts["grobid"].text,
        artifacts["docling"].text,
        artifacts["marker"].text,
        llm,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert llm.terra_calls == 0
    assert llm.sol_calls == 1
    assert metrics["direct_sol_region_count"] == 1
    assert metrics["terra_to_sol_escalation_count"] == 0
    assert metrics["selection_events"] == [
        {
            "tier": "sol",
            "outcome": "valid",
            "region_ids": ("region-0001",),
            "reasons": {"region-0001": "protected_payload_conflict"},
        }
    ]


def test_ordinary_prose_with_identical_numeric_payload_routes_directly_to_sol():
    llm = FakeSelectorLLM()
    store, graph = _selector_fixture(
        {
            "grobid": "The 10 nM dose activated alpha.",
            "docling": "A 10 nM dose activated alpha.",
            "marker": "The 10 nM treatment activated alpha.",
        }
    )

    resolution = BoundedCandidateSelector(llm=llm)(store, graph)

    assert llm.terra_calls == 0
    assert llm.sol_calls == 1
    assert len(resolution.response.decisions) == 1


def test_equivalent_unicode_numeric_payload_routes_directly_to_sol():
    llm = FakeSelectorLLM()
    store, graph = _selector_fixture(
        {
            "grobid": "The range was 10−20 μm.",
            "docling": "A range of 10-20 μm was observed.",
            "marker": "The observed range was １０–２０ μm.",
        }
    )

    BoundedCandidateSelector(llm=llm)(store, graph)

    assert llm.terra_calls == 0
    assert llm.sol_calls == 1


@pytest.mark.parametrize(
    "candidate_type",
    ["equation", "table_row", "table_cell", "citation"],
)
def test_structured_scientific_payload_routes_directly_to_sol(candidate_type):
    llm = FakeSelectorLLM()
    store, graph = _selector_fixture(
        {
            "grobid": "Alpha payload",
            "docling": "Beta payload",
            "marker": "Gamma payload",
        },
        candidate_type=candidate_type,
    )

    BoundedCandidateSelector(llm=llm)(store, graph)

    assert llm.terra_calls == 0
    assert llm.sol_calls == 1


def test_same_visible_text_with_conflicting_emphasis_routes_directly_to_sol():
    llm = FakeSelectorLLM()
    store, graph = _selector_fixture(
        {
            "grobid": "Gene alpha is active.",
            "docling": "Gene *alpha* is active.",
            "marker": "Gene alpha is active.",
        },
        emphasis_profiles={
            "grobid": (),
            "docling": ("c" * 64,),
            "marker": (),
        },
    )

    selector = BoundedCandidateSelector(llm=llm)
    selector(store, graph)

    assert llm.terra_calls == 0
    assert llm.sol_calls == 1
    assert selector.events[0]["reasons"] == {
        "region-0001": "protected_format_conflict"
    }


def test_all_bounded_high_risk_batches_reach_sol_without_a_job_quota():
    baseline_chunks = [f"A{index};" for index in range(9)]
    alternative_chunks = [f"B{index};" for index in range(9)]
    baseline = SourceArtifact.from_text("grobid", "".join(baseline_chunks))
    alternative = SourceArtifact.from_text("docling", "".join(alternative_chunks))
    candidates = {}
    regions = []
    baseline_offset = 0
    alternative_offset = 0
    for index, (baseline_chunk, alternative_chunk) in enumerate(
        zip(baseline_chunks, alternative_chunks)
    ):
        baseline_id = f"base-{index}"
        alternative_id = f"alternative-{index}"
        candidates[baseline_id] = SourceSpanCandidate(
            candidate_id=baseline_id,
            occurrence_id=f"base-occurrence-{index}",
            structural_unit_id=f"unit-{index}",
            source="grobid",
            artifact_digest=baseline.digest,
            byte_start=baseline_offset,
            byte_end=baseline_offset + len(baseline_chunk.encode()),
            candidate_type="equation",
            comparison_key=baseline_chunk,
        )
        candidates[alternative_id] = SourceSpanCandidate(
            candidate_id=alternative_id,
            occurrence_id=f"alternative-occurrence-{index}",
            structural_unit_id=f"unit-{index}",
            source="docling",
            artifact_digest=alternative.digest,
            byte_start=alternative_offset,
            byte_end=alternative_offset + len(alternative_chunk.encode()),
            candidate_type="equation",
            comparison_key=alternative_chunk,
        )
        regions.append(
            RegionCandidateGraph(
                region_id=f"region-{index}",
                baseline_candidate_id=baseline_id,
                valid_paths=((baseline_id,), (alternative_id,)),
            )
        )
        baseline_offset += len(baseline_chunk.encode())
        alternative_offset += len(alternative_chunk.encode())
    store = CandidateStore(
        {
            ("grobid", baseline.digest): baseline,
            ("docling", alternative.digest): alternative,
        },
        candidates,
    )
    llm = FakeSelectorLLM()

    resolution = BoundedCandidateSelector(llm=llm)(
        store,
        CandidateGraph(regions=tuple(regions)),
    )

    assert llm.sol_calls == 2  # the existing eight-region request bound remains
    assert len(resolution.response.decisions) == 9


def test_candidate_adapter_uses_one_bounded_sol_attempt_after_typed_terra_failure():
    llm = FakeSelectorLLM(terra_failure="terra_timeout")

    _merged, metrics, _audit = _merge(llm)

    assert llm.terra_calls == 1
    assert llm.sol_calls == 1
    assert metrics["merge_quality"] == "sol_selected"
    assert metrics["direct_sol_region_count"] == 0
    assert metrics["terra_to_sol_escalation_count"] == 1
    assert [event["tier"] for event in metrics["selection_events"]] == [
        "terra",
        "sol",
    ]


def test_candidate_adapter_delivers_baseline_when_model_client_is_unavailable():
    merged, metrics, audit = _merge(None)

    assert merged
    assert metrics["merge_quality"] == "baseline_fallback"
    assert metrics["warnings"] == [
        "selection_unavailable",
        "deterministic_trailing_newline_normalization",
    ]
    assert metrics["selection_events"] == []
    assert any(entry["decision_method"] == "baseline_fallback" for entry in audit)


def test_candidate_adapter_accepts_valid_terra_keep_without_sol():
    llm = FakeSelectorLLM(terra_keep=True)

    _merged, metrics, audit = _merge(llm)

    assert llm.terra_calls == 1
    assert llm.sol_calls == 0
    assert metrics["merge_quality"] == "terra_selected"
    assert metrics["unresolved_region_count"] == 0
    assert metrics["region_decision_counts"] == {"model_selected": 1}
    assert metrics["region_decisions"][0]["selected_choice"] == 0
    region_audit = [entry for entry in audit if entry["region_id"] is not None]
    assert region_audit
    assert all(
        entry["decision_method"] == "model_selected" for entry in region_audit
    )
    assert len(metrics["selection_events"]) == 1
    assert metrics["selection_events"][0]["tier"] == "terra"
    assert metrics["selection_events"][0]["outcome"] == "valid"
    assert metrics["selection_events"][0]["reasons"] == {
        "region-0001": "ordinary_text_conflict"
    }


def test_candidate_adapter_labels_sol_timeout_and_retains_baseline():
    llm = FakeSelectorLLM(
        terra_failure="terra_timeout",
        sol_failure="terra_timeout",
    )

    merged, metrics, audit = _merge(llm)

    assert merged
    assert metrics["merge_quality"] == "baseline_fallback"
    assert metrics["selection_events"][-1]["outcome"] == "sol_timeout"
    assert any(entry["decision_method"] == "baseline_fallback" for entry in audit)


def test_page_verified_sol_timeout_is_delivered_only_as_failsafe():
    artifacts = _artifacts()
    llm = FakeSelectorLLM(
        terra_failure="terra_timeout",
        sol_failure="terra_timeout",
    )

    merged, metrics, _audit = merge_source_artifacts(
        artifacts["grobid"].text,
        artifacts["docling"].text,
        artifacts["marker"].text,
        llm,
        completion_evidence=_page_verified_evidence(artifacts),
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert merged
    assert metrics["unresolved_region_count"] == 1
    assert metrics["unresolved_region_reason_counts"] == {
        "unresolved_selection": 1
    }
    assert metrics["qualification_outcome"] == "failsafe"
    assert "unresolved_candidate_regions" in metrics["qualification_reasons"]


def test_candidate_adapter_merges_two_completed_extractors():
    text = "# Title\n\nGene Gγ1 is active."
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("grobid", "marker")
    }
    merged, metrics, audit = merge_source_artifacts(
        text,
        "",
        text,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        baseline_requirements=FRAGMENT_REQUIREMENTS,
    )

    assert merged == text + "\n"
    assert metrics["failed"] is False
    assert metrics["source_count"] == 2
    assert metrics["available_extractors"] == ["grobid", "marker"]
    assert metrics["missing_extractors"] == ["docling"]
    assert audit


def test_candidate_adapter_delivers_one_completed_extractor_as_labeled_fallback():
    text = "# Title\n\nGene *dpp* is active."
    artifact = SourceArtifact.from_text("marker", text)

    merged, metrics, audit = merge_source_artifacts(
        "",
        "",
        text,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(
            {"marker": artifact}
        ),
        baseline_requirements=FRAGMENT_REQUIREMENTS,
    )

    assert merged == text + "\n"
    assert metrics["merge_quality"] == "baseline_fallback"
    assert metrics["source_count"] == 1
    assert metrics["available_extractors"] == ["marker"]
    assert metrics["missing_extractors"] == ["docling", "grobid"]
    assert "single_extractor_fallback" in metrics["warnings"]
    assert {entry["decision_method"] for entry in audit} == {
        "baseline_fallback",
        "deterministic",
    }


def test_candidate_adapter_fails_only_when_no_extractor_is_usable():
    merged, metrics, audit = merge_source_artifacts(
        "", "", "", None, completion_evidence={}
    )

    assert merged is None
    assert metrics["failure_reason"] == "no_usable_extractor"
    assert metrics["source_count"] == 0
    assert audit == []


def test_delivery_baseline_relaxes_only_terminal_heading_group():
    body = " ".join(f"word{index}" for index in range(80))
    grobid = f"# Example title\n\n## Abstract\n\n{body}\n\n## Results\n\n{body}"
    docling = grobid.replace("word10", "word\u0003b310")
    marker = "# Fragment\n\nToo short."

    artifacts = {
        "grobid": SourceArtifact.from_text("grobid", grobid),
        "docling": SourceArtifact.from_text("docling", docling),
        "marker": SourceArtifact.from_text("marker", marker),
    }
    merged, metrics, audit = merge_source_artifacts(
        grobid,
        docling,
        marker,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
    )

    assert merged == grobid + "\n"
    assert metrics["failed"] is False
    assert metrics["baseline_source"] == "grobid"
    assert metrics["merge_quality"] == "baseline_fallback"
    assert metrics["warnings"] == [
        "relaxed_delivery_baseline:terminal_heading_group",
        "deterministic_trailing_newline_normalization",
        "skeleton_conflict_unresolved:no_model",
    ]
    assert metrics["delivery_assurance"] == "unverified_failsafe"
    assert metrics["structure_assurance"] == "terminal_heading_relaxed"
    assert {entry["decision_method"] for entry in audit} == {
        "baseline_fallback",
        "deterministic",
    }


def test_delivery_baseline_does_not_relax_intro_or_body_heading_groups():
    body = " ".join(f"word{index}" for index in range(80))
    flat = f"# Example title\n\n{body}\n\n{body}"
    artifacts = {
        source: SourceArtifact.from_text(source, flat)
        for source in ("grobid", "docling", "marker")
    }

    merged, metrics, _audit = merge_source_artifacts(
        flat,
        flat,
        flat,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
    )

    assert merged == flat + "\n"
    assert metrics["delivery_assurance"] == "unverified_failsafe"
    assert metrics["structure_assurance"] == "single_structural_unit_failsafe"
    assert metrics["warnings"] == [
        "relaxed_delivery_baseline:unverified_largest_safe_source",
        "deterministic_trailing_newline_normalization",
    ]


def test_relaxed_delivery_baseline_still_merges_all_usable_sources():
    body = " ".join(f"word{index}" for index in range(100))
    texts = {
        "grobid": f"# Example title\n\nGene Gg1 is active. {body}",
        "docling": f"# Example title\n\nGene Gγ1 is active. {body}",
        "marker": f"# Example title\n\nGene G g 1 is active. {body}",
    }
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source, text in texts.items()
    }
    llm = FakeSelectorLLM()

    merged, metrics, _audit = merge_source_artifacts(
        texts["grobid"],
        texts["docling"],
        texts["marker"],
        llm,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        benchmark_mode=True,
    )

    assert metrics["baseline_selection_relaxation"] == (
        "unverified_largest_safe_source"
    )
    assert metrics["candidate_region_count"] >= 1
    assert metrics["candidate_construction_counts"][
        "region_model_selection_required"
    ] >= 1
    assert llm.terra_calls == 0
    assert llm.sol_calls == 1


def test_delivery_assurance_requires_selected_baseline_page_coverage():
    body = " ".join(f"word{index}" for index in range(80))
    text = (
        f"# Example title\n\n## Abstract\n\n{body}\n\n"
        f"## Results\n\n{body}\n\n## References\n\n1. Example 2026."
    )
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("grobid", "docling", "marker")
    }
    evidence = {
        source: BaselineCompletionEvidence(
            artifact_digest=artifact.digest,
            extraction_succeeded=True,
            artifact_complete=True,
            expected_page_count=4,
            covered_page_count=4,
            pdf_digest="a" * 64,
            coverage_method=PAGE_COVERAGE_METHOD,
            page_coverage_digest=page_coverage_proof_digest(
                source=source,
                artifact_digest=artifact.digest,
                pdf_digest="a" * 64,
                expected_page_count=4,
                covered_page_count=4,
            ),
            completion_basis="page_coverage",
        )
        for source, artifact in artifacts.items()
    }

    merged, missing_native_metrics, _audit = merge_source_artifacts(
        text, text, text, None, completion_evidence=evidence
    )

    assert merged == text + "\n"
    assert missing_native_metrics["delivery_assurance"] == "page_coverage_verified"
    assert missing_native_metrics["qualification_outcome"] == "failsafe"
    assert "native_italic_evidence_unavailable" in missing_native_metrics[
        "qualification_reasons"
    ]

    merged, metrics, _audit = merge_source_artifacts(
        text,
        text,
        text,
        None,
        completion_evidence=evidence,
        native_structures=_minimal_native_structures(artifacts),
    )

    assert metrics["delivery_assurance"] == "page_coverage_verified"
    assert metrics["structure_assurance"] == "strict_structure_validated"
    assert metrics["qualification_outcome"] == "qualified"
    assert metrics["qualification_reasons"] == []


def test_exact_page_coverage_proof_wins_content_baseline_selection():
    body = " ".join(f"word{index}" for index in range(100))
    grobid = (
        f"# Example scientific title\n\n## Abstract\n\n{body}\n\n"
        f"## Results\n\nGene *dpp* is active. {body}\n\n"
        "## References\n\n1. Example (2026) Reference.\n"
    )
    docling = grobid.replace("# Example", "## Example", 1).replace("*dpp*", "dpp")
    artifacts = {
        "grobid": SourceArtifact.from_text("grobid", grobid),
        "docling": SourceArtifact.from_text("docling", docling),
    }
    evidence = completion_evidence_for_finished_artifacts(artifacts)
    pdf_digest = "a" * 64
    evidence["docling"] = BaselineCompletionEvidence(
        artifact_digest=artifacts["docling"].digest,
        extraction_succeeded=True,
        artifact_complete=True,
        expected_page_count=4,
        covered_page_count=4,
        pdf_digest=pdf_digest,
        coverage_method=PAGE_COVERAGE_METHOD,
        page_coverage_digest=page_coverage_proof_digest(
            source="docling",
            artifact_digest=artifacts["docling"].digest,
            pdf_digest=pdf_digest,
            expected_page_count=4,
            covered_page_count=4,
        ),
        completion_basis="page_coverage",
    )

    merged, metrics, _audit = merge_source_artifacts(
        grobid,
        docling,
        "",
        None,
        completion_evidence=evidence,
    )

    assert metrics["baseline_source"] == "docling"
    assert metrics["delivery_assurance"] == "page_coverage_verified"
    assert metrics["qualification_outcome"] == "failsafe"
    assert "*dpp*" in merged
    assert metrics["italic_preservation"]["all_protected_italics_retained"] is True


def test_page_verified_baseline_retains_complete_content_without_model():
    body = " ".join(f"word{index}" for index in range(100))
    grobid = (
        f"# Example scientific discovery title\n\nPublished: 2026\n\n## Abstract\n\n{body}\n\n"
        f"## Results\n\nGene *dpp* is active. {body}\n\n"
        "## Author Contributions\n\n"
        "## References\n\n1. Example (2026) Reference.\n"
    )
    docling = (
        f"## Example scientific discovery title\n\n## Abstract\n\n{body}\n\n"
        f"## Results\n\nGene dpp is active. {body}\n\n"
        "## Author contributions\n\nAll authors planned and wrote the study.\n\n"
        "## References\n\n1. Example (2026) Reference.\n"
    )
    artifacts = {
        "grobid": SourceArtifact.from_text("grobid", grobid),
        "docling": SourceArtifact.from_text("docling", docling),
    }
    evidence = completion_evidence_for_finished_artifacts(artifacts)
    pdf_digest = "a" * 64
    evidence["docling"] = BaselineCompletionEvidence(
        artifact_digest=artifacts["docling"].digest,
        extraction_succeeded=True,
        artifact_complete=True,
        expected_page_count=4,
        covered_page_count=4,
        pdf_digest=pdf_digest,
        coverage_method=PAGE_COVERAGE_METHOD,
        page_coverage_digest=page_coverage_proof_digest(
            source="docling",
            artifact_digest=artifacts["docling"].digest,
            pdf_digest=pdf_digest,
            expected_page_count=4,
            covered_page_count=4,
        ),
        completion_basis="page_coverage",
    )

    merged, metrics, _audit = merge_source_artifacts(
        grobid,
        docling,
        "",
        None,
        completion_evidence=evidence,
    )

    assert "All authors planned and wrote the study." in merged
    assert metrics["baseline_source"] == "docling"
    assert metrics["abc_markdown"]["error_rule_ids"] == []
    assert metrics["qualification_outcome"] == "failsafe"
    assert metrics["delivery_assurance"] == "page_coverage_verified"


def test_native_skeleton_fixes_metadata_before_title_and_late_body_h1():
    body = " ".join(f"word{index}" for index in range(80))
    texts = {
        "grobid": f"# A scientific article title\n\n## Results\n\n{body}\n",
        "docling": (
            "## REVIEW\n\n## A scientific article title\n\n"
            f"# Results\n\n{body} additional complete source text\n"
        ),
        "marker": f"## A scientific article title\n\n# Results\n\n{body}\n",
    }
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source, text in texts.items()
    }
    native_payloads = {
        "grobid": b"""<TEI><teiHeader><titleStmt><title>A scientific article title</title>
        </titleStmt></teiHeader><text><body><div><head>Results</head></div></body></text></TEI>""",
        "docling": json.dumps({
            "schema_name": "DoclingDocument",
            "body": {
                "children": [
                    {"$ref": "#/texts/0"},
                    {"$ref": "#/texts/1"},
                    {"$ref": "#/texts/2"},
                ]
            },
            "texts": [
                {"self_ref": "#/texts/0", "label": "section_header", "level": 1, "text": "REVIEW"},
                {"self_ref": "#/texts/1", "label": "title", "text": "A scientific article title"},
                {"self_ref": "#/texts/2", "label": "section_header", "level": 1, "text": "Results"},
            ],
        }).encode("utf-8"),
        "marker": json.dumps({
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {"id": "title", "block_type": "Title", "html": "<h1>A scientific article title</h1>"},
                    {"id": "results", "block_type": "SectionHeader", "html": "<h2>Results</h2>"},
                ],
            }],
        }).encode("utf-8"),
    }
    native = {
        source: NativeStructureArtifact.for_test(
            source, artifacts[source], native_payloads[source]
        )
        for source in artifacts
    }

    merged, metrics, _audit = merge_source_artifacts(
        texts["grobid"],
        texts["docling"],
        texts["marker"],
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        native_structures=native,
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert merged.startswith("# A scientific article title\n\nREVIEW")
    assert "\n\n## Results\n" in merged
    assert metrics["document_skeleton_source"] == "docling"
    assert metrics["document_skeleton_title_proven"] is True
    assert metrics["abc_markdown"]["error_rule_ids"] == []


def test_malformed_native_structure_degrades_to_zero_error_source_skeleton():
    text = "# A scientific article title\n\n## Results\n\n" + "word " * 80
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("grobid", "docling", "marker")
    }
    malformed = NativeStructureArtifact.for_test(
        "docling",
        artifacts["docling"],
        json.dumps({"schema_name": "wrong-schema", "texts": []}).encode(),
    )

    merged, metrics, audit = merge_source_artifacts(
        text,
        text,
        text,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        native_structures={"docling": malformed},
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert merged is not None
    assert metrics["document_skeleton_build_failures"]["docling"] == "ValueError"
    assert set(metrics["document_skeleton_candidate_ids"]) == set(artifacts)
    assert metrics["abc_markdown"]["error_rule_ids"] == []
    assert audit


def test_all_malformed_native_structures_still_deliver_complete_markdown():
    text = "# Article title\n\n## Results\n\n" + "word " * 80
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("grobid", "docling", "marker")
    }
    native = {
        source: NativeStructureArtifact.for_test(
            source,
            artifacts[source],
            b"<not-tei" if source == "grobid" else b"{not-json",
        )
        for source in artifacts
    }

    merged, metrics, audit = merge_source_artifacts(
        text,
        text,
        text,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        native_structures=native,
        baseline_requirements=FRAGMENT_REQUIREMENTS,
    )

    assert merged is not None and "word word word" in merged
    assert set(metrics["document_skeleton_build_failures"]) == set(artifacts)
    assert metrics["abc_markdown"]["error_rule_ids"] == []
    assert audit


def test_native_heading_fragment_cannot_replace_complete_content_baseline():
    body = " ".join(f"word{index}" for index in range(120))
    complete = (
        "# Complete scientific article\n\n"
        f"## Abstract\n\n{body}\n\n"
        f"## Results\n\n{body}\n\n"
        "## References\n\n1. Complete source reference.\n"
    )
    fragment = "# Fragment title\n\n## Results\n\nTiny fragment.\n"
    artifacts = {
        "grobid": SourceArtifact.from_text("grobid", complete),
        "docling": SourceArtifact.from_text("docling", fragment),
    }
    docling_native = NativeStructureArtifact.for_test(
        "docling",
        artifacts["docling"],
        json.dumps(
            {
                "schema_name": "DoclingDocument",
                "body": {
                    "children": [
                        {"$ref": "#/texts/0"},
                        {"$ref": "#/texts/1"},
                        {"$ref": "#/texts/2"},
                    ]
                },
                "texts": [
                    {"self_ref": "#/texts/0", "label": "title", "text": "Fragment title"},
                    {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
                    {"self_ref": "#/texts/2", "label": "text", "text": "Tiny fragment."},
                ],
            }
        ).encode(),
    )

    merged, metrics, audit = merge_source_artifacts(
        complete,
        fragment,
        "",
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        native_structures={"docling": docling_native},
        benchmark_mode=True,
    )

    assert metrics["content_baseline_source"] == "grobid"
    assert metrics["baseline_source"] == "grobid"
    assert metrics["document_skeleton_source"] == "grobid"
    assert metrics["document_skeleton_conflict"] is True
    assert len(merged) >= len(complete)
    assert body in merged
    assert "Tiny fragment" not in merged
    assert metrics["abc_markdown"]["error_rule_ids"] == []
    assert audit


def test_skeleton_conflict_uses_one_existing_id_without_replacing_content():
    body = " ".join(f"word{index}" for index in range(120))
    complete = (
        "# Complete scientific article\n\n"
        f"## Abstract\n\n{body}\n\n"
        f"## Results\n\n{body}\n\n"
        "## References\n\n1. Complete source reference.\n"
    )
    fragment = "# Fragment title\n\n## Results\n\nTiny fragment.\n"
    artifacts = {
        "grobid": SourceArtifact.from_text("grobid", complete),
        "docling": SourceArtifact.from_text("docling", fragment),
    }
    native = NativeStructureArtifact.for_test(
        "docling",
        artifacts["docling"],
        json.dumps(
            {
                "schema_name": "DoclingDocument",
                "body": {
                    "children": [
                        {"$ref": "#/texts/0"},
                        {"$ref": "#/texts/1"},
                        {"$ref": "#/texts/2"},
                    ]
                },
                "texts": [
                    {
                        "self_ref": "#/texts/0",
                        "label": "title",
                        "text": "Fragment title",
                    },
                    {
                        "self_ref": "#/texts/1",
                        "label": "section_header",
                        "text": "Results",
                    },
                    {
                        "self_ref": "#/texts/2",
                        "label": "text",
                        "text": "Tiny fragment.",
                    },
                ],
            }
        ).encode(),
    )
    llm = FakeSelectorLLM(skeleton_choice="alternative")

    merged, metrics, _audit = merge_source_artifacts(
        complete,
        fragment,
        "",
        llm,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        native_structures={"docling": native},
        benchmark_mode=True,
    )

    assert llm.skeleton_calls == 1
    assert llm.skeleton_choice_payload["reason"] == "skeleton_conflict"
    assert all(
        "text" not in choice and "display" not in choice
        for choice in llm.skeleton_choice_payload["choices"]
    )
    assert metrics["document_skeleton_resolution"]["outcome"] == (
        "sol_selected_existing_id"
    )
    assert metrics["direct_sol_region_count"] >= 1
    assert metrics["content_baseline_source"] == "grobid"
    assert metrics["baseline_source"] == "grobid"
    assert body in merged
    assert "Tiny fragment" not in merged
    assert metrics["abc_markdown"]["error_rule_ids"] == []


def test_skeleton_sol_failure_retains_safe_content_and_marks_failsafe():
    left = "# Article\n\n## Results\n\n" + "result " * 80
    right = "# Article\n\n## Methods\n\n" + "method " * 80
    artifacts = {
        "grobid": SourceArtifact.from_text("grobid", left),
        "docling": SourceArtifact.from_text("docling", right),
    }
    llm = FakeSelectorLLM(sol_failure="terra_timeout")

    merged, metrics, _audit = merge_source_artifacts(
        left,
        right,
        "",
        llm,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert merged is not None
    assert metrics["document_skeleton_conflict"] is True
    assert metrics["document_skeleton_resolution"]["outcome"] == (
        "safe_retention_sol_timeout"
    )
    assert "skeleton_conflict_unresolved:sol_timeout" in metrics[
        "qualification_reasons"
    ]
    assert metrics["qualification_outcome"] == "failsafe"
    assert metrics["abc_markdown"]["error_rule_ids"] == []


def test_short_safe_source_is_delivered_as_failsafe():
    short = "# Title\n\nToo short."
    artifacts = {
        source: SourceArtifact.from_text(source, short)
        for source in ("grobid", "docling", "marker")
    }

    merged, metrics, audit = merge_source_artifacts(
        short,
        short,
        short,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
    )

    assert merged == short + "\n"
    assert metrics["failed"] is False
    assert metrics["qualification_outcome"] == "failsafe"
    assert "page_coverage_unverified" in metrics["qualification_reasons"]
    assert metrics["final_validation_passed"] is True
    assert audit


def test_all_sources_with_s07_table_error_deliver_through_single_renderer():
    text = "# Title\n\n| Gene | Value |\n| dpp | 12 |\n"
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("grobid", "docling", "marker")
    }

    merged, metrics, audit = merge_source_artifacts(
        text,
        text,
        text,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
        baseline_requirements=FRAGMENT_REQUIREMENTS,
        benchmark_mode=True,
    )

    assert merged is not None
    assert "| Gene | Value |\n|---|---|\n| dpp | 12 |" in merged
    assert metrics["abc_markdown"]["error_rule_ids"] == []
    assert any(
        event["operation"] == "alliance_table_separator"
        for event in metrics["document_skeleton_transformations"]
    )
    assert any(
        entry.get("transformation") == "alliance_table_separator"
        for entry in audit
    )


def test_full_length_no_h1_source_is_delivered_unchanged_as_failsafe():
    body = " ".join(f"word{index}" for index in range(100))
    text = f"## Abstract\n\n{body}\n\n## Results\n\n{body}"
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("grobid", "docling", "marker")
    }

    merged, metrics, audit = merge_source_artifacts(
        text,
        text,
        text,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
    )

    assert merged == text + "\n"
    assert metrics["failed"] is False
    assert metrics["qualification_outcome"] == "failsafe"
    assert "alliance_validator_not_clean" in metrics["qualification_reasons"]
    assert metrics["final_validation_passed"] is True
    assert audit


def test_delivery_uses_largest_safe_source_when_relative_shapes_cross():
    grobid = "A long safe paragraph " + "word " * 140
    docling = "\n".join(f"- item {index}" for index in range(30))
    marker = "\n".join(f"- m {index}" for index in range(20))
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source, text in {
            "grobid": grobid,
            "docling": docling,
            "marker": marker,
        }.items()
    }

    merged, metrics, audit = merge_source_artifacts(
        grobid,
        docling,
        marker,
        None,
        completion_evidence=completion_evidence_for_finished_artifacts(artifacts),
    )

    assert merged == grobid + "\n"
    assert metrics["baseline_source"] == "grobid"
    assert metrics["qualification_outcome"] == "failsafe"
    assert audit


@pytest.mark.parametrize("case_id", [f"case-{index:02d}" for index in range(1, 7)])
def test_saved_cases_report_no_model_italics_and_schema_diagnostics(case_id):
    case_dir = (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "scrum-6297-adjudication-v1"
        / case_id
    )
    if not case_dir.is_dir():
        pytest.skip("ignored local adjudication artifacts are unavailable")
    texts = {
        source: (case_dir / f"evidence-{source}.md").read_text(encoding="utf-8")
        for source in ("grobid", "docling", "marker")
    }

    merged, metrics, audit = merge_finished_extractor_outputs(
        texts["grobid"], texts["docling"], texts["marker"], None
    )

    assert merged is not None and merged.endswith("\n") and not merged.endswith("\n\n")
    assert metrics["abc_markdown"]["parser_version"] == "1.6.0"
    assert metrics["abc_markdown"]["error_rule_ids"] == []
    assert isinstance(metrics["abc_markdown"]["warning_rule_ids"], list)
    assert metrics["abc_markdown"]["validator_clean"] is (
        not metrics["abc_markdown"]["warning_rule_ids"]
    )
    italic_evidence = metrics["italic_preservation"]
    if not italic_evidence["all_native_body_italics_retained"]:
        assert metrics["qualification_outcome"] == "failsafe"
        assert "protected_italics_unresolved" in metrics["qualification_reasons"]
    else:
        assert italic_evidence["retained_native_body_emphasis_count"] == (
            italic_evidence["native_body_emphasis_count"]
        )
    assert italic_evidence["native_body_evidence_reconciled"] is True
    assert (
        italic_evidence["retained_native_body_emphasis_count"]
        + italic_evidence["excluded_native_body_emphasis_count"]
        == italic_evidence["native_body_emphasis_count"]
    )
    assert isinstance(italic_evidence["native_body_exclusion_reason_counts"], dict)
    assert isinstance(italic_evidence["all_protected_italics_retained"], bool)
    assert italic_evidence["policy_version"] == "positive-style-overlay-v1"
    assert isinstance(italic_evidence["auxiliary_positive_outcome_counts"], dict)
    assert len(italic_evidence["protected_claim_ids_sha256"]) == 64
    assert audit
