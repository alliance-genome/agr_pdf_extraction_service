#!/usr/bin/env python3
"""Run and summarize a read-only-output PDFX benchmark corpus.

The runner submits PDFs to an isolated PDFX endpoint. It never writes to ABC
Literature. Each case retains its source artifacts, final output, status,
content-free decision evidence, and timing history. The analyzer emits JSONL,
CSV, JSON, and Markdown so the same evidence is useful to scripts and people.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import requests


SCHEMA_VERSION = "pdfx-benchmark-v2"
EXTRACTORS = ("grobid", "docling", "marker")
TERMINAL_STATUSES = {"complete", "failed", "cancelled"}
HEADING_RE = re.compile(r"^(#{1,6})\s+", re.MULTILINE)
ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return default


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeError):
        return None


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _counter(values: Iterable[object]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _numeric(value, default=0):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _usage_breakdown(usage: Mapping) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return observed usage grouped by call type and model."""

    by_call_type: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    for call_type, raw in (usage.get("breakdown") or {}).items():
        if not isinstance(raw, Mapping):
            continue
        item = {
            "calls": int(_numeric(raw.get("calls"))),
            "prompt_tokens": int(_numeric(raw.get("prompt_tokens"))),
            "completion_tokens": int(_numeric(raw.get("completion_tokens"))),
            "cached_tokens": int(_numeric(raw.get("cached_tokens"))),
            "cost_usd": round(float(_numeric(raw.get("cost_usd"))), 6),
        }
        item["total_tokens"] = item["prompt_tokens"] + item["completion_tokens"]
        model = str(raw.get("model") or "unknown")
        by_call_type[str(call_type)] = {"model": model, **item}
        target = by_model.setdefault(
            model,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            },
        )
        for key in (
            "calls",
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "total_tokens",
        ):
            target[key] += item[key]
        target["cost_usd"] = round(target["cost_usd"] + item["cost_usd"], 6)
    return dict(sorted(by_call_type.items())), dict(sorted(by_model.items()))


def _sum_grouped_usage(cases: list[dict], field: str) -> dict[str, dict]:
    combined: dict[str, dict] = {}
    for case in cases:
        for name, item in (case.get(field) or {}).items():
            target = combined.setdefault(
                name,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            if field == "model_usage_by_call_type":
                target["model"] = item.get("model")
            for key in (
                "calls",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "total_tokens",
            ):
                target[key] += int(_numeric(item.get(key)))
            target["cost_usd"] = round(
                target["cost_usd"] + float(_numeric(item.get("cost_usd"))), 6
            )
    return dict(sorted(combined.items()))


def _sum_counter_fields(cases: list[dict], field: str) -> dict[str, int]:
    return dict(
        sorted(
            sum(
                (Counter(item.get(field) or {}) for item in cases),
                Counter(),
            ).items()
        )
    )


def _safe_case_name(path: Path, used: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-.") or "paper"
    name = stem
    suffix = 2
    while name in used:
        name = f"{stem}-{suffix}"
        suffix += 1
    used.add(name)
    return name


def _headers(token_env: str | None) -> dict[str, str]:
    if not token_env:
        return {}
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"benchmark token environment variable {token_env!r} is unset")
    return {"Authorization": f"Bearer {token}"}


def _download_artifact(
    session: requests.Session,
    base_url: str,
    process_id: str,
    artifact: str,
    target: Path,
    *,
    timeout_seconds: float,
) -> bool:
    response = session.get(
        f"{base_url}/api/v1/extract/{process_id}/download/{artifact}",
        timeout=timeout_seconds,
        allow_redirects=True,
    )
    if response.status_code != 200:
        return False
    target.write_bytes(response.content)
    return True


def run_case(
    session: requests.Session,
    *,
    base_url: str,
    pdf_path: Path,
    case_name: str,
    output_dir: Path,
    poll_seconds: float,
    timeout_seconds: float,
) -> None:
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=False)
    pdf_bytes = pdf_path.read_bytes()
    input_record = {
        "schema": SCHEMA_VERSION,
        "case": case_name,
        "source_filename": pdf_path.name,
        "pdf_sha256": _sha256_bytes(pdf_bytes),
        "pdf_bytes": len(pdf_bytes),
    }
    _write_json(case_dir / "input.json", input_record)

    started = time.monotonic()
    with pdf_path.open("rb") as handle:
        response = session.post(
            f"{base_url}/api/v1/extract",
            files={"file": (pdf_path.name, handle, "application/pdf")},
            data={
                "methods": ",".join(EXTRACTORS),
                "merge": "true",
                "clear_cache_scope": "all",
                "extract_images": "false",
                "review_images": "false",
            },
            timeout=min(timeout_seconds, 1200),
        )
    try:
        submission = response.json()
    except requests.JSONDecodeError:
        submission = {"raw_response_sha256": _sha256_bytes(response.content)}
    submission["http_status"] = response.status_code
    _write_json(case_dir / "submission.json", submission)
    process_id = submission.get("process_id")
    if response.status_code != 202 or not isinstance(process_id, str):
        _write_json(
            case_dir / "run.json",
            {
                **input_record,
                "status": "submission_failed",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "http_status": response.status_code,
            },
        )
        return

    deadline = time.monotonic() + timeout_seconds
    terminal = None
    history_path = case_dir / "status-history.jsonl"
    while time.monotonic() < deadline:
        status_response = session.get(
            f"{base_url}/api/v1/extract/{process_id}", timeout=60
        )
        if status_response.status_code == 200:
            status = status_response.json()
            _append_jsonl(
                history_path,
                {
                    "observed_at": _utc_now(),
                    "status": status.get("status"),
                    "stage": status.get("stage") or status.get("step"),
                    "percent": status.get("percent"),
                    "stages_completed": status.get("stages_completed"),
                    "stages_failed": status.get("stages_failed"),
                },
            )
            if status.get("status") in TERMINAL_STATUSES:
                terminal = status
                break
        time.sleep(poll_seconds)

    if terminal is None:
        terminal = {"process_id": process_id, "status": "timeout"}
    _write_json(case_dir / "status.json", terminal)
    _write_json(
        case_dir / "run.json",
        {
            **input_record,
            "process_id": process_id,
            "status": terminal.get("status", "unknown"),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )
    if terminal.get("status") != "complete":
        return

    for artifact in (*EXTRACTORS, "merged", "audit"):
        suffix = ".json" if artifact == "audit" else ".md"
        _download_artifact(
            session,
            base_url,
            process_id,
            artifact,
            case_dir / f"{artifact}{suffix}",
            timeout_seconds=min(timeout_seconds, 600),
        )
    artifact_response = session.get(
        f"{base_url}/api/v1/extract/{process_id}/artifacts", timeout=60
    )
    if artifact_response.status_code == 200:
        _write_json(case_dir / "artifacts.json", artifact_response.json())


def _event(case_name: str, sequence: int, stage: str, kind: str, outcome, details) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "case": case_name,
        "sequence": sequence,
        "stage": stage,
        "decision_type": kind,
        "outcome": outcome,
        "details": details,
    }


def analyze_case(case_dir: Path) -> tuple[dict, list[dict]]:
    case_name = case_dir.name
    run = _read_json(case_dir / "run.json", {}) or {}
    status = _read_json(case_dir / "status.json", {}) or {}
    metrics = status.get("consensus_metrics_json") or {}
    audit = _read_json(case_dir / "audit.json", []) or []
    merged = _read_text(case_dir / "merged.md")
    source_text = {
        source: _read_text(case_dir / f"{source}.md") for source in EXTRACTORS
    }
    sequence = 0
    events = []

    def add(stage: str, kind: str, outcome, details) -> None:
        nonlocal sequence
        sequence += 1
        events.append(_event(case_name, sequence, stage, kind, outcome, details))

    failures = status.get("extractor_failures") or {}
    available = metrics.get("available_extractors") or [
        source for source, text in source_text.items() if text is not None
    ]
    missing = metrics.get("missing_extractors") or [
        source for source in EXTRACTORS if source not in available
    ]
    extraction_status = status.get("extraction_status")
    if extraction_status is None and status.get("status") == "complete":
        extraction_status = "partial_success" if missing else "complete_success"
    for source in EXTRACTORS:
        add(
            "extraction",
            "extractor_result",
            "available" if source in available else "failed_or_missing",
            {"source": source, "failure": failures.get(source)},
        )
    for item in metrics.get("baseline_selection_trace", []):
        add("baseline_selection", "baseline_candidate", item.get("outcome"), item)
    if metrics.get("baseline_source"):
        add(
            "baseline_selection",
            "baseline_final",
            metrics.get("baseline_source"),
            {
                "baseline_digest": metrics.get("baseline_digest"),
                "relaxation": metrics.get("baseline_selection_relaxation"),
            },
        )
    for item in metrics.get("candidate_alignment_trace", []):
        add(
            "candidate_construction",
            "alignment_summary",
            "matched" if item.get("matched_unit_count", 0) else "no_matches",
            item,
        )
    for reason, count in sorted(
        (metrics.get("candidate_construction_counts") or {}).items()
    ):
        add(
            "candidate_construction",
            "construction_outcome",
            reason,
            {"count": count},
        )
    for item in metrics.get("region_decisions", []):
        add("candidate_selection", "region_choice", item.get("decision_method"), item)
    for item in metrics.get("selection_events", []):
        add("model_routing", "selector_batch", item.get("outcome"), item)
    for item in metrics.get("model_selection_calls", []):
        add("model_call", "numbered_selection", item.get("outcome"), item)
    runtime_execution = metrics.get("runtime_execution") or {}
    for item in runtime_execution.get("stage_events", []):
        add("runtime", "stage_event", item.get("status"), item)
    for item in audit:
        add("assembly", "output_span", item.get("decision_method"), item)
    for item in metrics.get("document_skeleton_transformations", []):
        add("render", "skeleton_transformation", item.get("operation"), item)

    abc = metrics.get("abc_markdown") or {}
    reader = metrics.get("semantic_payload_reader") or {}
    italic = metrics.get("italic_preservation") or {}
    quality_status = metrics.get("quality_receipt_status") or {}
    projection = metrics.get("native_emphasis_projection") or {}
    title_selection_events = [
        item
        for item in metrics.get("document_skeleton_transformations", [])
        if isinstance(item, Mapping)
        and item.get("operation") == "alliance_model_title_selection"
    ]
    usage = status.get("llm_usage_json") or metrics.get("llm_usage") or {}
    usage_by_call_type, usage_by_model = _usage_breakdown(usage)
    usage_breakdown_calls = sum(
        item["calls"] for item in usage_by_call_type.values()
    )
    usage_breakdown_prompt = sum(
        item["prompt_tokens"] for item in usage_by_call_type.values()
    )
    usage_breakdown_completion = sum(
        item["completion_tokens"] for item in usage_by_call_type.values()
    )
    usage_breakdown_cached = sum(
        item["cached_tokens"] for item in usage_by_call_type.values()
    )
    terra_usage_calls = sum(
        item["calls"]
        for call_type, item in usage_by_call_type.items()
        if call_type.endswith("_terra")
    )
    sol_usage_calls = sum(
        item["calls"]
        for call_type, item in usage_by_call_type.items()
        if call_type.endswith("_sol")
    )
    model_call_traces = metrics.get("model_selection_calls", [])
    trace_usage_call_count = sum(
        isinstance(item, Mapping) and isinstance(item.get("usage"), Mapping)
        for item in model_call_traces
    )
    persisted_cost = status.get("llm_cost_usd")
    usage_cost = usage.get("estimated_cost_usd")
    cost_reconciled = (
        isinstance(persisted_cost, (int, float))
        and isinstance(usage_cost, (int, float))
        and abs(float(persisted_cost) - float(usage_cost)) <= 0.000001
    )
    selection_events = metrics.get("selection_events", [])
    routing_reasons = []
    for item in selection_events:
        if not isinstance(item, Mapping):
            continue
        reasons = item.get("reasons") or {}
        if reasons:
            routing_reasons.extend(reasons.values())
        elif item.get("tier") == "terra":
            # Older immutable benchmark images predate the explicit reason map,
            # but Terra was reachable only for the ordinary-ID partition.
            routing_reasons.extend(
                "ordinary_text_conflict" for _ in item.get("region_ids", ())
            )
    routing_reason_counts = _counter(routing_reasons)
    selection_outcome_counts = _counter(
        item.get("outcome", "unknown")
        for item in selection_events
        if isinstance(item, Mapping)
    )
    model_failure_counts = metrics.get("model_failure_counts") or _counter(
        item.get("outcome", "unknown")
        for item in selection_events
        if isinstance(item, Mapping)
        and item.get("tier") in {"terra", "sol"}
        and item.get("outcome") != "valid"
    )
    checks = (
        ("final_safety", metrics.get("final_validation_passed"), {
            "unsafe_character_count": metrics.get("unsafe_character_count"),
            "repetition_diagnostic_count": len(metrics.get("repetition_diagnostics", [])),
        }),
        ("alliance_validator", abc.get("validator_clean"), {
            "error_rule_ids": abc.get("error_rule_ids", []),
            "warning_rule_ids": abc.get("warning_rule_ids", []),
        }),
        ("semantic_reader", reader.get("reader_contract_pass"), {
            "mismatch_codes": reader.get("mismatch_codes", []),
        }),
        ("italic_preservation", italic.get("all_protected_italics_retained"), {
            "protected_italic_count": italic.get("protected_italic_count"),
            "unresolved_occurrence_count": italic.get("unresolved_occurrence_count"),
        }),
        ("page_coverage", metrics.get("delivery_assurance") == "page_coverage_verified", {
            "delivery_assurance": metrics.get("delivery_assurance"),
        }),
    )
    for name, passed, details in checks:
        add("validation", name, "pass" if passed is True else "not_pass", details)
    add(
        "qualification",
        "final_outcome",
        metrics.get("qualification_outcome"),
        {"reasons": metrics.get("qualification_reasons", [])},
    )

    headings = [] if merged is None else HEADING_RE.findall(merged)
    italic_count = 0 if merged is None else len(ITALIC_RE.findall(merged))
    exact_source_matches = []
    if merged is not None:
        exact_source_matches = [
            source for source, text in source_text.items() if text == merged
        ]
    decision_methods = _counter(
        item.get("decision_method", "unknown")
        for item in audit
        if isinstance(item, Mapping)
    )
    region_methods = _counter(
        item.get("decision_method", "unknown")
        for item in metrics.get("region_decisions", [])
        if isinstance(item, Mapping)
    )
    stage_duration_seconds: dict[str, float] = {}
    for item in runtime_execution.get("stage_events", []):
        stage = item.get("stage")
        duration = item.get("duration_s", item.get("total_duration_s"))
        if isinstance(stage, str) and isinstance(duration, (int, float)):
            stage_duration_seconds[stage] = round(
                stage_duration_seconds.get(stage, 0.0) + float(duration), 3
            )
    merge_compute_duration = runtime_execution.get("merge_compute_duration_s")
    if isinstance(merge_compute_duration, (int, float)):
        stage_duration_seconds["merge_compute"] = float(merge_compute_duration)
    record = {
        "schema": SCHEMA_VERSION,
        "case": case_name,
        "status": status.get("status") or run.get("status") or "unknown",
        "process_id": status.get("process_id") or run.get("process_id"),
        "pdf_sha256": run.get("pdf_sha256") or (_read_json(case_dir / "input.json", {}) or {}).get("pdf_sha256"),
        "pdf_bytes": run.get("pdf_bytes") or (_read_json(case_dir / "input.json", {}) or {}).get("pdf_bytes"),
        "elapsed_seconds": run.get("elapsed_seconds"),
        "extraction_status": extraction_status,
        "available_extractors": available,
        "missing_extractors": missing,
        "failed_extractors": status.get("failed_extractors", []),
        "retried_extractors": runtime_execution.get("retried_extractors", []),
        "emergency_ocr_used": runtime_execution.get("emergency_ocr_used"),
        "cached_methods": runtime_execution.get("cached_methods", []),
        "stage_duration_seconds": stage_duration_seconds,
        "baseline_source": metrics.get("baseline_source"),
        "baseline_selection_relaxation": metrics.get("baseline_selection_relaxation"),
        "merge_quality": metrics.get("merge_quality"),
        "qualification_outcome": metrics.get("qualification_outcome"),
        "qualification_reasons": metrics.get("qualification_reasons", []),
        "native_italics_quality_receipt_valid": quality_status.get(
            "native_italics_valid"
        ),
        "native_italics_quality_receipt_error": quality_status.get(
            "native_italics_error"
        ),
        "delivery_assurance": metrics.get("delivery_assurance"),
        "structure_assurance": metrics.get("structure_assurance"),
        "candidate_region_count": metrics.get("candidate_region_count", len(metrics.get("region_decisions", []))),
        "replaced_region_count": metrics.get("replaced_region_count"),
        "unresolved_region_count": metrics.get("unresolved_region_count"),
        "unresolved_region_reason_counts": metrics.get(
            "unresolved_region_reason_counts"
        ) or _counter(
            item.get("decision_reason", "unknown")
            for item in metrics.get("region_decisions", [])
            if isinstance(item, Mapping)
            and item.get("decision_method") == "baseline_fallback"
        ),
        "region_decision_counts": region_methods,
        "candidate_construction_counts": metrics.get(
            "candidate_construction_counts", {}
        ),
        "alignment_matched_unit_count": sum(
            item.get("matched_unit_count", 0)
            for item in metrics.get("candidate_alignment_trace", [])
        ),
        "alignment_candidate_edge_count": sum(
            item.get("candidate_edge_count", 0)
            for item in metrics.get("candidate_alignment_trace", [])
        ),
        "audit_decision_counts": decision_methods,
        "model_calls": metrics.get("model_calls", 0),
        "model_call_attempts": metrics.get(
            "model_call_attempts", len(metrics.get("model_selection_calls", []))
        ),
        "terra_calls": terra_usage_calls,
        "sol_calls": sol_usage_calls,
        "direct_sol_region_count": metrics.get("direct_sol_region_count", 0),
        "title_selection_call_count": sum(
            item.get("title_selection_method") == "sol_numbered_choice"
            for item in title_selection_events
        ),
        "title_selection_outcome_counts": _counter(
            item.get("outcome", "unknown") for item in title_selection_events
        ),
        "terra_to_sol_escalation_count": metrics.get(
            "terra_to_sol_escalation_count", 0
        ),
        "routing_reason_counts": routing_reason_counts,
        "selection_outcome_counts": selection_outcome_counts,
        "model_failure_counts": model_failure_counts,
        "llm_usage": usage,
        "llm_cost_usd": persisted_cost,
        "llm_prompt_tokens": int(_numeric(usage.get("total_prompt_tokens"))),
        "llm_completion_tokens": int(_numeric(usage.get("total_completion_tokens"))),
        "llm_cached_tokens": int(_numeric(usage.get("total_cached_tokens"))),
        "llm_total_tokens": int(_numeric(usage.get("total_tokens"))),
        "model_usage_by_call_type": usage_by_call_type,
        "model_usage_by_model": usage_by_model,
        "cost_alert_threshold_usd": usage.get("cost_alert_threshold_usd"),
        "cost_alert_triggered": usage.get("cost_alert_triggered", False),
        "usage_breakdown_call_count": usage_breakdown_calls,
        "trace_usage_call_count": trace_usage_call_count,
        "model_calls_reconciled": metrics.get("model_calls", 0)
        == usage_breakdown_calls,
        "trace_usage_reconciled": trace_usage_call_count == usage_breakdown_calls,
        "usage_totals_reconciled": (
            int(_numeric(usage.get("total_prompt_tokens")))
            == usage_breakdown_prompt
            and int(_numeric(usage.get("total_completion_tokens")))
            == usage_breakdown_completion
            and int(_numeric(usage.get("total_cached_tokens")))
            == usage_breakdown_cached
            and int(_numeric(usage.get("total_tokens")))
            == usage_breakdown_prompt + usage_breakdown_completion
        ),
        "cost_reconciled": cost_reconciled,
        "alliance_valid": abc.get("valid"),
        "alliance_warning_free": bool(
            abc.get("valid") is True and not abc.get("warning_rule_ids", [])
        ),
        "alliance_validator_clean": abc.get("validator_clean"),
        "alliance_error_rule_ids": abc.get("error_rule_ids", []),
        "alliance_warning_rule_ids": abc.get("warning_rule_ids", []),
        "semantic_reader_pass": reader.get("reader_contract_pass"),
        "semantic_mismatch_codes": reader.get("mismatch_codes", []),
        "semantic_required_occurrence_count": reader.get(
            "required_occurrence_count", 0
        ),
        "semantic_reader_occurrence_count": reader.get(
            "reader_occurrence_count", 0
        ),
        "semantic_retained_occurrence_count": max(
            0,
            int(_numeric(reader.get("required_occurrence_count")))
            - len(reader.get("missing_occurrence_ids", [])),
        ),
        "semantic_missing_occurrence_count": len(
            reader.get("missing_occurrence_ids", [])
        ),
        "semantic_reordered_occurrence_count": len(
            reader.get("reordered_occurrence_ids", [])
        ),
        "semantic_role_changed_occurrence_count": len(
            reader.get("role_changed_occurrence_ids", [])
        ),
        "semantic_formatting_lost_occurrence_count": len(
            reader.get("formatting_lost_occurrence_ids", [])
        ),
        "semantic_input_normalized_token_count": reader.get(
            "input_normalized_token_count", 0
        ),
        "semantic_reader_normalized_token_count": reader.get(
            "reader_normalized_token_count", 0
        ),
        "semantic_retained_normalized_token_count": reader.get(
            "retained_normalized_token_count", 0
        ),
        "semantic_normalized_token_recall_ppm": reader.get(
            "normalized_token_recall_ppm", 0
        ),
        "semantic_normalized_token_precision_ppm": reader.get(
            "normalized_token_precision_ppm", 0
        ),
        "protected_italics_retained": italic.get("all_protected_italics_retained"),
        "protected_italic_occurrence_count": italic.get(
            "protected_italic_occurrence_count", 0
        ),
        "retained_protected_italic_occurrence_count": italic.get(
            "retained_protected_italic_occurrence_count", 0
        ),
        "lost_protected_italic_occurrence_count": italic.get(
            "lost_protected_italic_occurrence_count", 0
        ),
        "unresolved_italic_block_count": italic.get(
            "unresolved_italic_block_count", 0
        ),
        "whole_block_missing_protected_italic_occurrence_count": italic.get(
            "whole_block_missing_protected_italic_occurrence_count", 0
        ),
        "matched_block_formatting_lost_occurrence_count": italic.get(
            "matched_block_formatting_lost_occurrence_count", 0
        ),
        "candidate_offered_italic_occurrence_count": italic.get(
            "candidate_offered_italic_occurrence_count", 0
        ),
        "candidate_selected_italic_occurrence_count": italic.get(
            "candidate_selected_italic_occurrence_count", 0
        ),
        "candidate_published_selected_italic_occurrence_count": italic.get(
            "candidate_published_selected_italic_occurrence_count", 0
        ),
        "selected_candidate_italics_reconciled": italic.get(
            "selected_candidate_italics_reconciled"
        ),
        "native_body_emphasis_count": italic.get(
            "native_body_emphasis_count", 0
        ),
        "mapped_native_body_emphasis_count": italic.get(
            "mapped_native_body_emphasis_count", 0
        ),
        "retained_native_body_emphasis_count": italic.get(
            "retained_native_body_emphasis_count", 0
        ),
        "excluded_native_body_emphasis_count": italic.get(
            "excluded_native_body_emphasis_count", 0
        ),
        "native_body_evidence_reconciled": italic.get(
            "native_body_evidence_reconciled"
        ),
        "native_evidence_ready": italic.get("native_evidence_ready"),
        "native_evidence_failure_sources": italic.get(
            "native_evidence_failure_sources", []
        ),
        "native_body_exclusion_reason_counts": italic.get(
            "native_body_exclusion_reason_counts", {}
        ),
        "markdown_only_body_emphasis_count": italic.get(
            "markdown_only_body_emphasis_count", 0
        ),
        "reference_markdown_emphasis_count": italic.get(
            "reference_markdown_emphasis_count", 0
        ),
        "explicit_native_reference_emphasis_count": italic.get(
            "explicit_native_reference_emphasis_count", 0
        ),
        "projection_eligible_occurrence_count": projection.get(
            "eligible_occurrence_count", 0
        ),
        "projection_reconciled_occurrence_count": projection.get(
            "projected_reconciled_occurrence_count", 0
        ),
        "projection_identical_supported_occurrence_count": projection.get(
            "identical_supported_occurrence_count", 0
        ),
        "projection_declined_occurrence_count": projection.get(
            "declined_occurrence_count", 0
        ),
        "projection_decline_reason_counts": projection.get(
            "decline_reason_counts", {}
        ),
        "candidate_offered_italic_occurrence_counts_by_source": italic.get(
            "candidate_offered_italic_occurrence_counts_by_source", {}
        ),
        "candidate_selected_italic_occurrence_counts_by_source": italic.get(
            "candidate_selected_italic_occurrence_counts_by_source", {}
        ),
        "skeleton_transformation_count": len(metrics.get("document_skeleton_transformations", [])),
        "skeleton_transformation_counts": _counter(
            item.get("operation", "unknown") for item in metrics.get("document_skeleton_transformations", [])
        ),
        "unsafe_character_count": metrics.get("unsafe_character_count"),
        "repetition_diagnostic_count": len(metrics.get("repetition_diagnostics", [])),
        "final_validation_passed": metrics.get("final_validation_passed"),
        "merged_sha256": None if merged is None else _sha256_bytes(merged.encode("utf-8")),
        "merged_bytes": None if merged is None else len(merged.encode("utf-8")),
        "merged_lines": None if merged is None else len(merged.splitlines()),
        "heading_count": len(headings),
        "h1_count": headings.count("#"),
        "h2_count": headings.count("##"),
        "italic_markup_count": italic_count,
        "exact_source_matches": exact_source_matches,
        "decision_event_count": len(events),
    }
    return record, events


def _sum_field(cases: list[dict], key: str) -> int:
    return sum(value for item in cases if isinstance((value := item.get(key)), int))


def build_summary(cases: list[dict], events: list[dict]) -> dict:
    paper_costs = [
        float(item["llm_cost_usd"])
        for item in cases
        if isinstance(item.get("llm_cost_usd"), (int, float))
    ]
    total_cost = round(sum(paper_costs), 6)
    mean_cost = statistics.fmean(paper_costs) if paper_costs else 0.0
    cost_distribution = {
        "papers_with_cost": len(paper_costs),
        "total_usd": total_cost,
        "mean_per_paper_usd": round(mean_cost, 6),
        "median_per_paper_usd": round(statistics.median(paper_costs), 6)
        if paper_costs
        else 0.0,
        "minimum_per_paper_usd": round(min(paper_costs), 6)
        if paper_costs
        else 0.0,
        "maximum_per_paper_usd": round(max(paper_costs), 6)
        if paper_costs
        else 0.0,
    }
    cost_projections = {
        str(count): round(mean_cost * count, 2)
        for count in (1_000, 10_000, 100_000, 1_000_000)
    }
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "case_count": len(cases),
        "status_counts": _counter(item.get("status") for item in cases),
        "extraction_status_counts": _counter(item.get("extraction_status") for item in cases),
        "baseline_source_counts": _counter(item.get("baseline_source") for item in cases),
        "merge_quality_counts": _counter(item.get("merge_quality") for item in cases),
        "qualification_outcome_counts": _counter(item.get("qualification_outcome") for item in cases),
        "qualification_reason_counts": _counter(
            reason for item in cases for reason in item.get("qualification_reasons", [])
        ),
        "native_italics_quality_receipt_valid_count": sum(
            item.get("native_italics_quality_receipt_valid") is True
            for item in cases
        ),
        "native_italics_quality_receipt_invalid_count": sum(
            item.get("native_italics_quality_receipt_valid") is False
            for item in cases
        ),
        "native_italics_quality_receipt_error_counts": _counter(
            item.get("native_italics_quality_receipt_error")
            for item in cases
            if item.get("native_italics_quality_receipt_error")
        ),
        "alliance_validator_clean_count": sum(item.get("alliance_validator_clean") is True for item in cases),
        "alliance_valid_count": sum(item.get("alliance_valid") is True for item in cases),
        "alliance_warning_free_count": sum(item.get("alliance_warning_free") is True for item in cases),
        "semantic_reader_pass_count": sum(item.get("semantic_reader_pass") is True for item in cases),
        "protected_italics_retained_count": sum(item.get("protected_italics_retained") is True for item in cases),
        "final_validation_passed_count": sum(item.get("final_validation_passed") is True for item in cases),
        "total_candidate_regions": _sum_field(cases, "candidate_region_count"),
        "total_alignment_matches": _sum_field(cases, "alignment_matched_unit_count"),
        "total_alignment_candidate_edges": _sum_field(cases, "alignment_candidate_edge_count"),
        "total_replaced_regions": _sum_field(cases, "replaced_region_count"),
        "total_unresolved_regions": _sum_field(cases, "unresolved_region_count"),
        "unresolved_region_reason_counts": _sum_counter_fields(
            cases, "unresolved_region_reason_counts"
        ),
        "total_model_calls": _sum_field(cases, "model_calls"),
        "total_model_call_attempts": _sum_field(cases, "model_call_attempts"),
        "total_terra_calls": _sum_field(cases, "terra_calls"),
        "total_sol_calls": _sum_field(cases, "sol_calls"),
        "total_direct_sol_regions": _sum_field(cases, "direct_sol_region_count"),
        "total_title_selection_calls": _sum_field(
            cases, "title_selection_call_count"
        ),
        "title_selection_outcome_counts": _sum_counter_fields(
            cases, "title_selection_outcome_counts"
        ),
        "total_terra_to_sol_escalations": _sum_field(
            cases, "terra_to_sol_escalation_count"
        ),
        "total_llm_prompt_tokens": _sum_field(cases, "llm_prompt_tokens"),
        "total_llm_completion_tokens": _sum_field(
            cases, "llm_completion_tokens"
        ),
        "total_llm_cached_tokens": _sum_field(cases, "llm_cached_tokens"),
        "total_llm_tokens": _sum_field(cases, "llm_total_tokens"),
        "llm_cost": cost_distribution,
        "cost_projections_usd": cost_projections,
        "cost_alert_triggered_count": sum(
            item.get("cost_alert_triggered") is True for item in cases
        ),
        "model_usage_by_model": _sum_grouped_usage(
            cases, "model_usage_by_model"
        ),
        "model_usage_by_call_type": _sum_grouped_usage(
            cases, "model_usage_by_call_type"
        ),
        "routing_reason_counts": _sum_counter_fields(
            cases, "routing_reason_counts"
        ),
        "selection_outcome_counts": _sum_counter_fields(
            cases, "selection_outcome_counts"
        ),
        "model_failure_counts": _sum_counter_fields(
            cases, "model_failure_counts"
        ),
        "model_calls_reconciled_count": sum(
            item.get("model_calls_reconciled") is True for item in cases
        ),
        "trace_usage_reconciled_count": sum(
            item.get("trace_usage_reconciled") is True for item in cases
        ),
        "usage_totals_reconciled_count": sum(
            item.get("usage_totals_reconciled") is True for item in cases
        ),
        "cost_reconciled_count": sum(
            item.get("cost_reconciled") is True for item in cases
        ),
        "total_semantic_required_occurrences": _sum_field(
            cases, "semantic_required_occurrence_count"
        ),
        "total_semantic_reader_occurrences": _sum_field(
            cases, "semantic_reader_occurrence_count"
        ),
        "total_semantic_retained_occurrences": _sum_field(
            cases, "semantic_retained_occurrence_count"
        ),
        "total_semantic_missing_occurrences": _sum_field(
            cases, "semantic_missing_occurrence_count"
        ),
        "total_semantic_reordered_occurrences": _sum_field(
            cases, "semantic_reordered_occurrence_count"
        ),
        "total_semantic_role_changed_occurrences": _sum_field(
            cases, "semantic_role_changed_occurrence_count"
        ),
        "total_semantic_formatting_lost_occurrences": _sum_field(
            cases, "semantic_formatting_lost_occurrence_count"
        ),
        "total_semantic_input_normalized_tokens": _sum_field(
            cases, "semantic_input_normalized_token_count"
        ),
        "total_semantic_reader_normalized_tokens": _sum_field(
            cases, "semantic_reader_normalized_token_count"
        ),
        "total_semantic_retained_normalized_tokens": _sum_field(
            cases, "semantic_retained_normalized_token_count"
        ),
        "total_protected_italic_occurrences": _sum_field(
            cases, "protected_italic_occurrence_count"
        ),
        "total_retained_protected_italic_occurrences": _sum_field(
            cases, "retained_protected_italic_occurrence_count"
        ),
        "total_lost_protected_italic_occurrences": _sum_field(
            cases, "lost_protected_italic_occurrence_count"
        ),
        "total_unresolved_italic_blocks": _sum_field(
            cases, "unresolved_italic_block_count"
        ),
        "total_whole_block_missing_protected_italic_occurrences": _sum_field(
            cases, "whole_block_missing_protected_italic_occurrence_count"
        ),
        "total_matched_block_formatting_lost_occurrences": _sum_field(
            cases, "matched_block_formatting_lost_occurrence_count"
        ),
        "total_candidate_offered_italic_occurrences": _sum_field(
            cases, "candidate_offered_italic_occurrence_count"
        ),
        "total_candidate_selected_italic_occurrences": _sum_field(
            cases, "candidate_selected_italic_occurrence_count"
        ),
        "total_candidate_published_selected_italic_occurrences": _sum_field(
            cases, "candidate_published_selected_italic_occurrence_count"
        ),
        "selected_candidate_italics_reconciled_count": sum(
            item.get("selected_candidate_italics_reconciled") is True
            for item in cases
        ),
        "total_native_body_emphasis_occurrences": _sum_field(
            cases, "native_body_emphasis_count"
        ),
        "total_mapped_native_body_emphasis_occurrences": _sum_field(
            cases, "mapped_native_body_emphasis_count"
        ),
        "total_retained_native_body_emphasis_occurrences": _sum_field(
            cases, "retained_native_body_emphasis_count"
        ),
        "total_excluded_native_body_emphasis_occurrences": _sum_field(
            cases, "excluded_native_body_emphasis_count"
        ),
        "native_body_evidence_reconciled_count": sum(
            item.get("native_body_evidence_reconciled") is True for item in cases
        ),
        "native_evidence_ready_count": sum(
            item.get("native_evidence_ready") is True for item in cases
        ),
        "native_evidence_failure_source_counts": _counter(
            source
            for item in cases
            for source in item.get("native_evidence_failure_sources", [])
        ),
        "native_body_exclusion_reason_counts": _sum_counter_fields(
            cases, "native_body_exclusion_reason_counts"
        ),
        "total_markdown_only_body_emphasis_occurrences": _sum_field(
            cases, "markdown_only_body_emphasis_count"
        ),
        "total_reference_markdown_emphasis_occurrences": _sum_field(
            cases, "reference_markdown_emphasis_count"
        ),
        "total_explicit_native_reference_emphasis_occurrences": _sum_field(
            cases, "explicit_native_reference_emphasis_count"
        ),
        "total_projection_eligible_occurrences": _sum_field(
            cases, "projection_eligible_occurrence_count"
        ),
        "total_projection_reconciled_occurrences": _sum_field(
            cases, "projection_reconciled_occurrence_count"
        ),
        "total_projection_identical_supported_occurrences": _sum_field(
            cases, "projection_identical_supported_occurrence_count"
        ),
        "total_projection_declined_occurrences": _sum_field(
            cases, "projection_declined_occurrence_count"
        ),
        "projection_decline_reason_counts": _sum_counter_fields(
            cases, "projection_decline_reason_counts"
        ),
        "candidate_offered_italic_occurrence_counts_by_source": (
            _sum_counter_fields(
                cases, "candidate_offered_italic_occurrence_counts_by_source"
            )
        ),
        "candidate_selected_italic_occurrence_counts_by_source": (
            _sum_counter_fields(
                cases, "candidate_selected_italic_occurrence_counts_by_source"
            )
        ),
        "total_unsafe_characters": _sum_field(cases, "unsafe_character_count"),
        "total_repetition_diagnostics": _sum_field(cases, "repetition_diagnostic_count"),
        "total_decision_events": len(events),
        "decision_type_counts": _counter(item.get("decision_type") for item in events),
        "alliance_error_rule_counts": _counter(
            rule for item in cases for rule in item.get("alliance_error_rule_ids", [])
        ),
        "semantic_mismatch_counts": _counter(
            code for item in cases for code in item.get("semantic_mismatch_codes", [])
        ),
        "candidate_construction_counts": dict(
            sorted(
                sum(
                    (
                        Counter(item.get("candidate_construction_counts", {}))
                        for item in cases
                    ),
                    Counter(),
                ).items()
            )
        ),
    }


def _markdown_table(values: Mapping[str, int]) -> str:
    if not values:
        return "_None._"
    lines = ["| Value | Count |", "|---|---:|"]
    lines.extend(f"| `{key}` | {value} |" for key, value in values.items())
    return "\n".join(lines)


def _markdown_usage_table(values: Mapping[str, Mapping]) -> str:
    if not values:
        return "_None._"
    lines = [
        "| Value | Calls | Prompt | Cached | Completion | Total | Cost (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in values.items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | ${:.6f} |".format(
                key,
                item.get("calls", 0),
                item.get("prompt_tokens", 0),
                item.get("cached_tokens", 0),
                item.get("completion_tokens", 0),
                item.get("total_tokens", 0),
                float(item.get("cost_usd", 0.0)),
            )
        )
    return "\n".join(lines)


def render_summary(summary: dict, cases: list[dict]) -> str:
    count = summary["case_count"]
    token_recall_ppm = (
        1_000_000
        if summary["total_semantic_input_normalized_tokens"] == 0
        else summary["total_semantic_retained_normalized_tokens"]
        * 1_000_000
        // summary["total_semantic_input_normalized_tokens"]
    )
    token_precision_ppm = (
        1_000_000
        if summary["total_semantic_reader_normalized_tokens"] == 0
        else summary["total_semantic_retained_normalized_tokens"]
        * 1_000_000
        // summary["total_semantic_reader_normalized_tokens"]
    )
    lines = [
        "# PDFX benchmark summary",
        "",
        f"Schema: `{summary['schema']}`  ",
        f"Generated: `{summary['generated_at']}`  ",
        f"Cases: **{count}**",
        "",
        "This report measures execution, decision routing, safety, provenance, and",
        "Alliance/semantic-reader contracts. It does not by itself establish that a",
        "scientific-text choice was semantically best; changed regions remain the",
        "target set for blinded review.",
        "",
        "## Headline",
        "",
        f"- Terminally complete: {summary['status_counts'].get('complete', 0)}/{count}",
        f"- Final safety passed: {summary['final_validation_passed_count']}/{count}",
        f"- Alliance valid (zero errors): {summary['alliance_valid_count']}/{count}",
        f"- Alliance warning-free: {summary['alliance_warning_free_count']}/{count}",
        f"- Semantic reader passed: {summary['semantic_reader_pass_count']}/{count}",
        f"- Protected italics retained: {summary['protected_italics_retained_count']}/{count}",
        f"- Alignment matches / candidate edges: {summary['total_alignment_matches']} / {summary['total_alignment_candidate_edges']}",
        f"- Candidate regions: {summary['total_candidate_regions']}",
        f"- Replaced / unresolved regions: {summary['total_replaced_regions']} / {summary['total_unresolved_regions']}",
        f"- Terra / Sol calls: {summary['total_terra_calls']} / {summary['total_sol_calls']}",
        f"- Title-selection calls: {summary['total_title_selection_calls']}",
        f"- Direct Sol regions / Terra-to-Sol escalations: {summary['total_direct_sol_regions']} / {summary['total_terra_to_sol_escalations']}",
        f"- Model attempts / usage-bearing calls: {summary['total_model_call_attempts']} / {summary['total_model_calls']}",
        f"- Observed model cost: ${summary['llm_cost']['total_usd']:.6f}",
        f"- Prompt / cached / completion tokens: {summary['total_llm_prompt_tokens']} / {summary['total_llm_cached_tokens']} / {summary['total_llm_completion_tokens']}",
        f"- Unsafe characters / repetition diagnostics: {summary['total_unsafe_characters']} / {summary['total_repetition_diagnostics']}",
        "",
        "## Model cost and usage",
        "",
        f"- Mean / median per paper: ${summary['llm_cost']['mean_per_paper_usd']:.6f} / ${summary['llm_cost']['median_per_paper_usd']:.6f}",
        f"- Minimum / maximum per paper: ${summary['llm_cost']['minimum_per_paper_usd']:.6f} / ${summary['llm_cost']['maximum_per_paper_usd']:.6f}",
        f"- Nonblocking cost alerts: {summary['cost_alert_triggered_count']}/{count}",
        "",
        "### By model",
        "",
        _markdown_usage_table(summary["model_usage_by_model"]),
        "",
        "### By call type",
        "",
        _markdown_usage_table(summary["model_usage_by_call_type"]),
        "",
        "### Volume projections",
        "",
        "These are linear projections from this ten-paper sample, not guaranteed production costs.",
        "",
        "| Papers | Projected cost (USD) |",
        "|---:|---:|",
        *(
            f"| {int(papers):,} | ${cost:,.2f} |"
            for papers, cost in summary["cost_projections_usd"].items()
        ),
        "",
        "## Routing outcomes",
        "",
        "### Reasons",
        "",
        _markdown_table(summary["routing_reason_counts"]),
        "",
        "### Selection outcomes",
        "",
        _markdown_table(summary["selection_outcome_counts"]),
        "",
        "### Title-selection outcomes",
        "",
        _markdown_table(summary["title_selection_outcome_counts"]),
        "",
        "### Model failures",
        "",
        _markdown_table(summary["model_failure_counts"]),
        "",
        "## Accounting reconciliation",
        "",
        f"- Metric call count matches priced usage: {summary['model_calls_reconciled_count']}/{count}",
        f"- Usage-bearing traces match priced usage: {summary['trace_usage_reconciled_count']}/{count}",
        f"- Token totals match breakdowns: {summary['usage_totals_reconciled_count']}/{count}",
        f"- Persisted cost matches priced usage: {summary['cost_reconciled_count']}/{count}",
        "",
        "## Semantic and italic preservation",
        "",
        f"- Required / retained / missing semantic occurrences: {summary['total_semantic_required_occurrences']} / {summary['total_semantic_retained_occurrences']} / {summary['total_semantic_missing_occurrences']}",
        f"- Reordered / role-changed / formatting-lost occurrences: {summary['total_semantic_reordered_occurrences']} / {summary['total_semantic_role_changed_occurrences']} / {summary['total_semantic_formatting_lost_occurrences']}",
        f"- Normalized tokens input / reader / retained: {summary['total_semantic_input_normalized_tokens']} / {summary['total_semantic_reader_normalized_tokens']} / {summary['total_semantic_retained_normalized_tokens']}",
        f"- Normalized token recall / precision: {token_recall_ppm / 10_000:.2f}% / {token_precision_ppm / 10_000:.2f}%",
        f"- Native body emphasis retained / observed / excluded: {summary['total_retained_native_body_emphasis_occurrences']} / {summary['total_native_body_emphasis_occurrences']} / {summary['total_excluded_native_body_emphasis_occurrences']}",
        f"- Native body evidence reconciled: {summary['native_body_evidence_reconciled_count']}/{count}",
        f"- Native evidence artifacts ready: {summary['native_evidence_ready_count']}/{count}",
        f"- Native italics quality receipts valid / invalid: {summary['native_italics_quality_receipt_valid_count']} / {summary['native_italics_quality_receipt_invalid_count']}",
        f"- Native emphasis projection eligible / reconciled / identical-support / declined: {summary['total_projection_eligible_occurrences']} / {summary['total_projection_reconciled_occurrences']} / {summary['total_projection_identical_supported_occurrences']} / {summary['total_projection_declined_occurrences']}",
        f"- Italic candidate occurrences offered / selected / published: {summary['total_candidate_offered_italic_occurrences']} / {summary['total_candidate_selected_italic_occurrences']} / {summary['total_candidate_published_selected_italic_occurrences']}",
        f"- Selected candidate italics reconciled: {summary['selected_candidate_italics_reconciled_count']}/{count}",
        f"- Markdown-only body / reference Markdown / explicit native-reference emphasis: {summary['total_markdown_only_body_emphasis_occurrences']} / {summary['total_reference_markdown_emphasis_occurrences']} / {summary['total_explicit_native_reference_emphasis_occurrences']}",
        "",
        "### Native body exclusion reasons",
        "",
        _markdown_table(summary["native_body_exclusion_reason_counts"]),
        "",
        "### Native emphasis projection decline reasons",
        "",
        _markdown_table(summary["projection_decline_reason_counts"]),
        "",
        "### Native italics quality receipt errors",
        "",
        _markdown_table(summary["native_italics_quality_receipt_error_counts"]),
        "",
        "### Native evidence failures by source",
        "",
        _markdown_table(summary["native_evidence_failure_source_counts"]),
        "",
        "### Italic candidates by source",
        "",
        "Offered:",
        "",
        _markdown_table(summary["candidate_offered_italic_occurrence_counts_by_source"]),
        "",
        "Selected:",
        "",
        _markdown_table(summary["candidate_selected_italic_occurrence_counts_by_source"]),
        "",
        "## Qualification outcomes",
        "",
        _markdown_table(summary["qualification_outcome_counts"]),
        "",
        "## Qualification reasons",
        "",
        _markdown_table(summary["qualification_reason_counts"]),
        "",
        "## Candidate construction outcomes",
        "",
        _markdown_table(summary["candidate_construction_counts"]),
        "",
        "## Per-paper index",
        "",
        "| Case | Status | Baseline | Merge | Regions | Terra/Sol | Direct/Escalated | Cost | ABC | Reader | Italics |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for item in cases:
        lines.append(
            "| {case} | {status} | {baseline} | {merge} | {regions} | {terra}/{sol} | {direct}/{escalated} | ${cost:.6f} | {abc} | {reader} | {italics} |".format(
                case=item["case"],
                status=item.get("status"),
                baseline=item.get("baseline_source"),
                merge=item.get("merge_quality"),
                regions=item.get("candidate_region_count"),
                terra=item.get("terra_calls", 0),
                sol=item.get("sol_calls", 0),
                direct=item.get("direct_sol_region_count", 0),
                escalated=item.get("terra_to_sol_escalation_count", 0),
                cost=float(item.get("llm_cost_usd") or 0.0),
                abc="pass" if item.get("alliance_validator_clean") is True else "not pass",
                reader="pass" if item.get("semantic_reader_pass") is True else "not pass",
                italics="pass" if item.get("protected_italics_retained") is True else "not pass",
            )
        )
    lines.extend(
        [
            "",
            "## Evidence files",
            "",
            "- `cases.csv`: sortable per-paper headline metrics.",
            "- `cases.jsonl`: complete per-paper machine-readable metrics.",
            "- `decision-events.jsonl`: ordered decision evidence across every paper.",
            "- Each case directory: status, source Markdown, merged Markdown, audit spans, and polling history.",
            "",
        ]
    )
    return "\n".join(lines)


CSV_FIELDS = (
    "case", "status", "elapsed_seconds", "pdf_bytes", "extraction_status",
    "baseline_source", "merge_quality", "qualification_outcome",
    "native_italics_quality_receipt_valid",
    "delivery_assurance", "candidate_region_count", "replaced_region_count",
    "unresolved_region_count", "model_calls", "terra_calls", "sol_calls",
    "title_selection_call_count",
    "model_call_attempts", "direct_sol_region_count",
    "terra_to_sol_escalation_count", "llm_cost_usd", "llm_prompt_tokens",
    "llm_cached_tokens", "llm_completion_tokens", "llm_total_tokens",
    "cost_alert_triggered", "model_calls_reconciled", "trace_usage_reconciled",
    "usage_totals_reconciled", "cost_reconciled",
    "alignment_matched_unit_count", "alignment_candidate_edge_count",
    "alliance_valid", "alliance_warning_free", "alliance_validator_clean",
    "semantic_reader_pass", "semantic_required_occurrence_count",
    "semantic_retained_occurrence_count", "semantic_missing_occurrence_count",
    "semantic_reordered_occurrence_count", "semantic_role_changed_occurrence_count",
    "semantic_formatting_lost_occurrence_count", "protected_italics_retained",
    "semantic_input_normalized_token_count", "semantic_reader_normalized_token_count",
    "semantic_retained_normalized_token_count", "semantic_normalized_token_recall_ppm",
    "semantic_normalized_token_precision_ppm",
    "protected_italic_occurrence_count",
    "retained_protected_italic_occurrence_count",
    "lost_protected_italic_occurrence_count", "unresolved_italic_block_count",
    "whole_block_missing_protected_italic_occurrence_count",
    "matched_block_formatting_lost_occurrence_count",
    "candidate_offered_italic_occurrence_count",
    "candidate_selected_italic_occurrence_count",
    "candidate_published_selected_italic_occurrence_count",
    "selected_candidate_italics_reconciled", "native_body_emphasis_count",
    "mapped_native_body_emphasis_count", "retained_native_body_emphasis_count",
    "excluded_native_body_emphasis_count", "native_body_evidence_reconciled",
    "native_evidence_ready",
    "markdown_only_body_emphasis_count", "reference_markdown_emphasis_count",
    "explicit_native_reference_emphasis_count",
    "projection_eligible_occurrence_count",
    "projection_reconciled_occurrence_count",
    "projection_identical_supported_occurrence_count",
    "projection_declined_occurrence_count",
    "unsafe_character_count",
    "repetition_diagnostic_count", "final_validation_passed", "merged_bytes",
    "heading_count", "italic_markup_count", "decision_event_count",
    "grobid_seconds", "docling_seconds", "marker_seconds", "merge_seconds",
)


def analyze_corpus(root: Path) -> dict:
    case_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and ((path / "run.json").exists() or (path / "status.json").exists())
    )
    cases = []
    events = []
    for case_dir in case_dirs:
        record, case_events = analyze_case(case_dir)
        cases.append(record)
        events.extend(case_events)
    summary = build_summary(cases, events)
    _write_json(root / "summary.json", summary)
    (root / "summary.md").write_text(render_summary(summary, cases), encoding="utf-8")
    with (root / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for item in cases:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    with (root / "decision-events.jsonl").open("w", encoding="utf-8") as handle:
        for item in events:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    with (root / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in cases:
            durations = item.get("stage_duration_seconds", {})
            writer.writerow(
                {
                    **item,
                    "grobid_seconds": durations.get("extract_grobid"),
                    "docling_seconds": durations.get("extract_docling"),
                    "marker_seconds": durations.get("extract_marker"),
                    "merge_seconds": durations.get("merge_compute"),
                }
            )
    manifest = {
        "schema": SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "case_count": len(cases),
        "files": {},
    }
    for name in ("summary.json", "summary.md", "cases.jsonl", "decision-events.jsonl", "cases.csv"):
        payload = (root / name).read_bytes()
        manifest["files"][name] = {
            "sha256": _sha256_bytes(payload),
            "bytes": len(payload),
        }
    _write_json(root / "benchmark-manifest.json", manifest)
    return summary


def _run(args) -> int:
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    pdfs = sorted(args.corpus_dir.glob("*.pdf"))
    if args.limit is not None:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise RuntimeError("benchmark corpus contains no PDF files")
    session = requests.Session()
    session.headers.update(_headers(args.bearer_token_env))
    used: set[str] = set()
    for index, pdf_path in enumerate(pdfs, start=1):
        case_name = _safe_case_name(pdf_path, used)
        print(f"[{index}/{len(pdfs)}] {case_name}", flush=True)
        run_case(
            session,
            base_url=args.base_url.rstrip("/"),
            pdf_path=pdf_path,
            case_name=case_name,
            output_dir=root,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    analyze_corpus(root)
    print(root / "summary.md")
    return 0


def _analyze(args) -> int:
    summary = analyze_corpus(args.results_dir.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="submit a PDF corpus and analyze it")
    run.add_argument("--base-url", required=True)
    run.add_argument("--corpus-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--limit", type=int, default=10)
    run.add_argument("--poll-seconds", type=float, default=10.0)
    run.add_argument("--timeout-seconds", type=float, default=3600.0)
    run.add_argument(
        "--bearer-token-env",
        help="name of an environment variable containing the endpoint bearer token",
    )
    run.set_defaults(handler=_run)
    analyze = subparsers.add_parser("analyze", help="rebuild reports from saved case directories")
    analyze.add_argument("--results-dir", type=Path, required=True)
    analyze.set_defaults(handler=_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    limit = getattr(args, "limit", None)
    if limit is not None and limit < 1:
        raise SystemExit("--limit must be at least 1")
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
