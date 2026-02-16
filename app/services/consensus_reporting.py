"""Metrics and audit reporting helpers for the consensus pipeline."""

from __future__ import annotations

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from app.services.consensus_models import AGREE_EXACT, AGREE_NEAR, CONFLICT, GAP, AlignedTriple, MicroConflictResult
from app.services.consensus_pipeline_steps import (
    _SOURCE_PREFERENCE,
    _extract_citation_keys,
    _extract_numeric_tokens,
    _get_present_blocks,
    _normalize_for_comparison,
    clean_output_md,
)
# ---------------------------------------------------------------------------
# Step 6: METRICS — Compute pipeline statistics
# ---------------------------------------------------------------------------

def compute_metrics(
    triples: list[AlignedTriple],
    alignment_confidence: float,
    failed: bool,
    failure_reason: str | None,
    guard_telemetry: dict | None = None,
    micro_results: dict[str, MicroConflictResult] | None = None,
) -> dict:
    """Compute metrics for the consensus pipeline run."""
    total = len(triples)
    num_exact = sum(1 for t in triples if t.classification == AGREE_EXACT)
    num_near = sum(1 for t in triples if t.classification == AGREE_NEAR)
    num_gap = sum(1 for t in triples if t.classification == GAP)
    num_conflict = sum(1 for t in triples if t.classification == CONFLICT)

    denominator = total - num_gap
    conflict_ratio = num_conflict / denominator if denominator > 0 else 0.0

    metrics = {
        "total_blocks": total,
        "agree_exact": num_exact,
        "agree_near": num_near,
        "gap": num_gap,
        "conflict": num_conflict,
        "conflict_ratio": round(conflict_ratio, 4),
        "alignment_confidence": round(alignment_confidence, 4),
        "failed": failed,
        "failure_reason": failure_reason,
    }

    micro_results = micro_results or {}
    micro_count = len(micro_results)
    if micro_count > 0:
        micro_auto = sum(1 for res in micro_results.values() if not res.conflicts)
        micro_spans = sum(len(res.conflicts) for res in micro_results.values())
        micro_ratio_avg = sum(res.majority_agree_ratio for res in micro_results.values()) / micro_count
        by_block: dict[str, int] = {}
        for res in micro_results.values():
            by_block[res.block_type] = by_block.get(res.block_type, 0) + len(res.conflicts)
    else:
        micro_auto = 0
        micro_spans = 0
        micro_ratio_avg = 0.0
        by_block = {}

    metrics.update({
        "micro_conflict_segments": micro_count,
        "micro_conflict_auto_resolved": micro_auto,
        "micro_conflict_total_spans": micro_spans,
        "micro_conflict_majority_agree_avg": round(micro_ratio_avg, 4),
        "micro_conflict_by_block_type": by_block,
    })

    if guard_telemetry:
        metrics.update({
            "conflict_ratio_textual": guard_telemetry.get("conflict_ratio_textual", 0.0),
            "conflict_ratio_structured": guard_telemetry.get("conflict_ratio_structured", 0.0),
            "conflicts_localized": guard_telemetry.get("conflicts_localized", False),
            "conflict_span_ratio": guard_telemetry.get("conflict_span_ratio", 0.0),
            "adaptive_conflict_ratio_threshold": guard_telemetry.get(
                "adaptive_conflict_ratio_threshold", metrics["conflict_ratio"],
            ),
        })
    return metrics


# ---------------------------------------------------------------------------
# Audit entries — Per-decision log for curator review
# ---------------------------------------------------------------------------


def _build_audit_entries(
    triples: list[AlignedTriple],
    resolved_conflicts: dict[str, str],
    resolution_metadata: dict[str, dict] | None = None,
    near_threshold: float = 0.92,
    levenshtein_threshold: float = 0.90,
) -> list[dict]:
    """Build audit entries for every non-AGREE_EXACT triple.

    Each entry records the classification, what each extractor produced,
    what text was chosen, and classification-specific details so curators
    can review pipeline decisions.
    """
    entries: list[dict] = []

    for idx, triple in enumerate(triples):
        if triple.classification == AGREE_EXACT:
            continue

        blocks = _get_present_blocks(triple)
        if not blocks:
            continue

        block_type = blocks[0].block_type
        sources_present = [b.source for b in blocks]
        extractor_texts = {b.source: b.source_md or b.raw_text for b in blocks}

        entry: dict = {
            "segment_id": triple.segment_id,
            "classification": triple.classification,
            "block_type": block_type,
            "sources_present": sources_present,
            "extractor_texts": extractor_texts,
            "chosen_text": "",
            "chosen_source": "",
            "details": {},
        }

        if triple.classification == AGREE_NEAR:
            resolution_details = (resolution_metadata or {}).get(triple.segment_id, {})
            resolved_text = resolved_conflicts.get(triple.segment_id)

            resolution_method = resolution_details.get("method", "")
            if resolved_text is not None and resolution_method.startswith("llm_near_agree"):
                # LLM-resolved near_agree (zone-based, rescue, or fallback)
                entry["chosen_text"] = resolved_text
                entry["chosen_source"] = resolution_details.get("chosen_source", "llm")
                entry["details"] = {
                    "context_id": resolution_details.get("context_id", ""),
                    "llm_resolved": True,
                    "resolution_method": resolution_method,
                    "resolution_confidence": resolution_details.get("confidence", 0.0),
                    "sources_agreeing": resolution_details.get("sources_agreeing", []),
                }
                if resolution_details.get("numeric_integrity"):
                    entry["details"]["numeric_integrity"] = resolution_details["numeric_integrity"]
                if resolution_details.get("rescue_explanation"):
                    entry["details"]["rescue_explanation"] = resolution_details["rescue_explanation"]
                if resolution_details.get("degraded"):
                    entry["details"]["degraded"] = True
            else:
                # Programmatic near-agree resolution (not in a zone)
                normalized_texts = [_normalize_for_comparison(b.normalized_text) for b in blocks]
                raw_texts = [b.raw_text for b in blocks]
                near_pair = None
                best_score = 0.0

                for i in range(len(blocks)):
                    for j in range(i + 1, len(blocks)):
                        token_ratio = fuzz.token_set_ratio(
                            normalized_texts[i], normalized_texts[j],
                        ) / 100.0
                        max_len = max(len(normalized_texts[i]), len(normalized_texts[j]), 1)
                        lev_dist = Levenshtein.distance(
                            normalized_texts[i], normalized_texts[j],
                        )
                        lev_sim = 1.0 - (lev_dist / max_len)

                        if token_ratio >= near_threshold and lev_sim >= levenshtein_threshold:
                            assem_norm_i = _normalize_for_comparison(raw_texts[i])
                            assem_norm_j = _normalize_for_comparison(raw_texts[j])
                            nums_i = _extract_numeric_tokens(assem_norm_i)
                            nums_j = _extract_numeric_tokens(assem_norm_j)
                            cites_i = _extract_citation_keys(assem_norm_i)
                            cites_j = _extract_citation_keys(assem_norm_j)
                            if nums_i == nums_j and cites_i == cites_j:
                                near_pair = (i, j)
                                best_score = token_ratio
                                break
                    if near_pair is not None:
                        break

                preferred_source = _SOURCE_PREFERENCE.get(block_type, "marker")
                if near_pair is not None:
                    agreeing = [blocks[near_pair[0]], blocks[near_pair[1]]]
                    chosen_source = preferred_source
                    if not any(b.source == preferred_source for b in agreeing):
                        chosen_source = agreeing[0].source
                    entry["details"] = {
                        "agreeing_pair": [agreeing[0].source, agreeing[1].source],
                        "similarity_score": round(best_score, 4),
                        "source_preference_rule": f"{block_type} \u2192 {preferred_source}",
                    }
                else:
                    chosen_source = blocks[0].source

                entry["chosen_text"] = triple.agreed_text or ""
                entry["chosen_source"] = chosen_source

        elif triple.classification == GAP:
            sole_source = blocks[0].source
            deduped = not (triple.agreed_text and triple.agreed_text.strip())
            # GAPs pulled into zones may have been resolved by LLM
            resolved_text = resolved_conflicts.get(triple.segment_id)
            resolution_details = (resolution_metadata or {}).get(triple.segment_id, {})
            if resolved_text is not None:
                entry["chosen_text"] = resolved_text
                entry["chosen_source"] = resolution_details.get("chosen_source", sole_source)
                entry["details"] = {
                    "sole_source": sole_source,
                    "deduped": deduped,
                    "context_id": resolution_details.get("context_id", ""),
                    "llm_resolved": True,
                    "resolution_method": resolution_details.get("method", ""),
                    "resolution_confidence": resolution_details.get("confidence", 0.0),
                }
                if resolution_details.get("numeric_integrity"):
                    entry["details"]["numeric_integrity"] = resolution_details["numeric_integrity"]
            else:
                entry["chosen_text"] = triple.agreed_text or ""
                entry["chosen_source"] = sole_source
                entry["details"] = {
                    "sole_source": sole_source,
                    "deduped": deduped,
                }

        elif triple.classification == CONFLICT:
            resolved_text = resolved_conflicts.get(triple.segment_id)
            resolution_details = (resolution_metadata or {}).get(triple.segment_id, {})

            if resolved_text is not None:
                entry["chosen_text"] = resolved_text
                entry["chosen_source"] = resolution_details.get("chosen_source", resolution_details.get("method", "llm"))
            elif blocks:
                entry["chosen_text"] = clean_output_md(
                    blocks[0].source_md or blocks[0].raw_text,
                )
                entry["chosen_source"] = blocks[0].source

            entry["details"] = {
                "context_id": resolution_details.get("context_id", ""),
                "llm_resolved": resolved_text is not None,
                "resolution_method": resolution_details.get("method", ""),
                "resolution_confidence": resolution_details.get("confidence", 0.0),
                "sources_agreeing": resolution_details.get("sources_agreeing", []),
            }
            if resolution_details.get("numeric_integrity"):
                entry["details"]["numeric_integrity"] = resolution_details["numeric_integrity"]

        entries.append(entry)

    return entries
