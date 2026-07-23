import csv
import json

from benchmarking.pdfx_benchmark import SCHEMA_VERSION, analyze_corpus


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_analyze_corpus_emits_browseable_and_machine_readable_evidence(tmp_path):
    case = tmp_path / "paper-01"
    case.mkdir()
    merged = "# Title\n\n## Abstract\n\nGene *dpp* is active.\n"
    (case / "merged.md").write_text(merged, encoding="utf-8")
    for source in ("grobid", "docling", "marker"):
        (case / f"{source}.md").write_text(merged, encoding="utf-8")
    _write_json(
        case / "run.json",
        {
            "status": "complete",
            "process_id": "process-1",
            "pdf_sha256": "a" * 64,
            "pdf_bytes": 1234,
            "elapsed_seconds": 45.6,
        },
    )
    _write_json(
        case / "audit.json",
        [
            {
                "output_byte_start": 0,
                "output_byte_end": len(merged.encode("utf-8")),
                "source": "grobid",
                "decision_method": "model_selected",
                "region_id": "region-0001",
            }
        ],
    )
    metrics = {
        "available_extractors": ["docling", "grobid", "marker"],
        "missing_extractors": [],
        "baseline_source": "grobid",
        "baseline_digest": "b" * 64,
        "baseline_selection_trace": [
            {"source": "grobid", "outcome": "selected"}
        ],
        "baseline_selection_relaxation": None,
        "merge_quality": "terra_selected",
        "qualification_outcome": "qualified",
        "qualification_reasons": [],
        "quality_receipt_status": {
            "native_italics_valid": True,
            "native_italics_error": None,
        },
        "delivery_assurance": "page_coverage_verified",
        "structure_assurance": "strict_structure_validated",
        "candidate_region_count": 1,
        "candidate_alignment_trace": [
            {
                "source": "docling",
                "matched_unit_count": 2,
                "candidate_edge_count": 3,
            }
        ],
        "candidate_construction_counts": {
            "region_model_selection_required": 1,
        },
        "replaced_region_count": 1,
        "unresolved_region_count": 0,
        "unresolved_region_reason_counts": {},
        "region_decisions": [
            {
                "region_id": "region-0001",
                "decision_method": "model_selected",
                "decision_reason": "model_selected_numbered_path",
                "selected_choice": 1,
            }
        ],
        "selection_events": [
            {
                "tier": "terra",
                "outcome": "valid",
                "region_ids": ["region-0001"],
                "reasons": {"region-0001": "ordinary_text_conflict"},
            }
        ],
        "model_selection_calls": [
            {
                "tier": "terra",
                "outcome": "valid",
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 20,
                    "cached_tokens": 10,
                },
                "response": {
                    "request_sha256": "c" * 64,
                    "decisions": [{"region_id": "region-0001", "choice": 1}],
                },
            }
        ],
        "model_calls": 1,
        "model_call_attempts": 1,
        "direct_sol_region_count": 0,
        "terra_to_sol_escalation_count": 0,
        "model_failure_counts": {},
        "llm_usage": {"total_tokens": 100},
        "abc_markdown": {
            "valid": True,
            "validator_clean": True,
            "error_rule_ids": [],
            "warning_rule_ids": [],
        },
        "semantic_payload_reader": {
            "reader_contract_pass": True,
            "mismatch_codes": [],
            "required_occurrence_count": 4,
            "reader_occurrence_count": 4,
            "missing_occurrence_ids": [],
            "reordered_occurrence_ids": [],
            "role_changed_occurrence_ids": [],
            "formatting_lost_occurrence_ids": [],
            "input_normalized_token_count": 10,
            "reader_normalized_token_count": 10,
            "retained_normalized_token_count": 10,
            "normalized_token_recall_ppm": 1_000_000,
            "normalized_token_precision_ppm": 1_000_000,
        },
        "italic_preservation": {
            "policy_version": "native-body-italics-v1",
            "all_protected_italics_retained": True,
            "all_native_body_italics_retained": True,
            "native_body_emphasis_count": 1,
            "mapped_native_body_emphasis_count": 1,
            "retained_native_body_emphasis_count": 1,
            "excluded_native_body_emphasis_count": 0,
            "native_body_exclusion_reason_counts": {},
            "native_body_evidence_reconciled": True,
            "native_evidence_ready": True,
            "native_evidence_failure_sources": [],
            "markdown_only_body_emphasis_count": 0,
            "reference_markdown_emphasis_count": 0,
            "explicit_native_reference_emphasis_count": 0,
            "protected_italic_count": 1,
            "unresolved_occurrence_count": 0,
            "protected_italic_occurrence_count": 1,
            "retained_protected_italic_occurrence_count": 1,
            "lost_protected_italic_occurrence_count": 0,
            "unresolved_italic_block_count": 0,
            "whole_block_missing_protected_italic_occurrence_count": 0,
            "matched_block_formatting_lost_occurrence_count": 0,
            "candidate_offered_italic_occurrence_count": 2,
            "candidate_selected_italic_occurrence_count": 1,
            "candidate_published_selected_italic_occurrence_count": 1,
            "selected_candidate_italics_reconciled": True,
            "candidate_offered_italic_occurrence_counts_by_source": {
                "docling": 1,
                "marker": 1,
            },
            "candidate_selected_italic_occurrence_counts_by_source": {
                "marker": 1,
            },
        },
        "native_emphasis_projection": {
            "policy_version": "native-emphasis-projection-v4",
            "eligible_occurrence_count": 3,
            "projected_reconciled_occurrence_count": 1,
            "declined_occurrence_count": 2,
            "decline_reason_counts": {
                "ambiguous_exact_target": 1,
                "markdown_emphasis_not_realized": 1,
            },
        },
        "document_skeleton_transformations": [
            {"operation": "selected_document_skeleton"}
        ],
        "unsafe_character_count": 0,
        "repetition_diagnostics": [],
        "final_validation_passed": True,
    }
    _write_json(
        case / "status.json",
        {
            "status": "complete",
            "process_id": "process-1",
            "consensus_metrics_json": metrics,
            "llm_cost_usd": 0.01,
            "llm_usage_json": {
                "total_prompt_tokens": 80,
                "total_completion_tokens": 20,
                "total_cached_tokens": 10,
                "total_tokens": 100,
                "estimated_cost_usd": 0.01,
                "cost_alert_threshold_usd": 2.0,
                "cost_alert_triggered": False,
                "breakdown": {
                    "source_path_selection_terra": {
                        "model": "gpt-5.6-terra",
                        "calls": 1,
                        "prompt_tokens": 80,
                        "completion_tokens": 20,
                        "cached_tokens": 10,
                        "cost_usd": 0.01,
                    }
                },
            },
        },
    )

    summary = analyze_corpus(tmp_path)

    assert summary["schema"] == SCHEMA_VERSION
    assert summary["case_count"] == 1
    assert summary["extraction_status_counts"] == {"complete_success": 1}
    assert summary["total_candidate_regions"] == 1
    assert summary["total_alignment_matches"] == 2
    assert summary["total_alignment_candidate_edges"] == 3
    assert summary["candidate_construction_counts"] == {
        "region_model_selection_required": 1
    }
    assert summary["total_terra_calls"] == 1
    assert summary["total_sol_calls"] == 0
    assert summary["total_direct_sol_regions"] == 0
    assert summary["total_terra_to_sol_escalations"] == 0
    assert summary["unresolved_region_reason_counts"] == {}
    assert summary["routing_reason_counts"] == {"ordinary_text_conflict": 1}
    assert summary["selection_outcome_counts"] == {"valid": 1}
    assert summary["model_failure_counts"] == {}
    assert summary["llm_cost"]["total_usd"] == 0.01
    assert summary["llm_cost"]["mean_per_paper_usd"] == 0.01
    assert summary["model_usage_by_model"]["gpt-5.6-terra"]["calls"] == 1
    assert summary["model_usage_by_call_type"]["source_path_selection_terra"][
        "cost_usd"
    ] == 0.01
    assert summary["total_llm_prompt_tokens"] == 80
    assert summary["total_llm_completion_tokens"] == 20
    assert summary["model_calls_reconciled_count"] == 1
    assert summary["trace_usage_reconciled_count"] == 1
    assert summary["usage_totals_reconciled_count"] == 1
    assert summary["cost_reconciled_count"] == 1
    assert summary["total_semantic_required_occurrences"] == 4
    assert summary["total_semantic_retained_occurrences"] == 4
    assert summary["total_protected_italic_occurrences"] == 1
    assert summary["total_retained_protected_italic_occurrences"] == 1
    assert summary["total_semantic_input_normalized_tokens"] == 10
    assert summary["total_semantic_retained_normalized_tokens"] == 10
    assert summary["total_candidate_offered_italic_occurrences"] == 2
    assert summary["total_candidate_selected_italic_occurrences"] == 1
    assert summary["native_evidence_ready_count"] == 1
    assert summary["native_evidence_failure_source_counts"] == {}
    assert summary["native_italics_quality_receipt_valid_count"] == 1
    assert summary["native_italics_quality_receipt_invalid_count"] == 0
    assert summary["total_projection_eligible_occurrences"] == 3
    assert summary["total_projection_reconciled_occurrences"] == 1
    assert summary["total_projection_declined_occurrences"] == 2
    assert summary["projection_decline_reason_counts"] == {
        "ambiguous_exact_target": 1,
        "markdown_emphasis_not_realized": 1,
    }
    assert summary["candidate_offered_italic_occurrence_counts_by_source"] == {
        "docling": 1,
        "marker": 1,
    }
    assert summary["alliance_valid_count"] == 1
    assert summary["alliance_validator_clean_count"] == 1
    assert summary["semantic_reader_pass_count"] == 1
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "cases.jsonl").exists()
    assert (tmp_path / "decision-events.jsonl").exists()
    assert (tmp_path / "benchmark-manifest.json").exists()
    with (tmp_path / "cases.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["case"] == "paper-01"
    events = [
        json.loads(line)
        for line in (tmp_path / "decision-events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert any(event["decision_type"] == "region_choice" for event in events)
    assert any(event["decision_type"] == "numbered_selection" for event in events)

    # Priced usage is the authoritative call ledger. A missing optional trace
    # must remain visible as a trace diagnostic without zeroing routing totals.
    status = json.loads((case / "status.json").read_text(encoding="utf-8"))
    status["consensus_metrics_json"]["model_selection_calls"] = []
    _write_json(case / "status.json", status)
    summary_without_trace = analyze_corpus(tmp_path)
    assert summary_without_trace["total_terra_calls"] == 1
    assert summary_without_trace["total_model_calls"] == 1
    assert summary_without_trace["trace_usage_reconciled_count"] == 0
