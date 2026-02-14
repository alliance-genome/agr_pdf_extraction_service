"""Degradation measurement models for the consensus pipeline.

Computes quality scores, risk indicators, and per-segment degradation
detail from the resolution_metadata dict and audit_entries list that
the consensus pipeline already produces.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Quality tier classification
# ---------------------------------------------------------------------------

# Maps each resolution method to its quality tier.
# High: LLM validated the output (normal zone resolution, rescue success,
#        intentional drop with explanation)
# Medium-high: Deterministic heuristic with reasonable confidence
# Medium: No LLM validation; picked best extractor text blindly
METHOD_QUALITY_TIER: dict[str, str] = {
    "llm_conflict": "high",
    "llm_near_agree": "high",
    "llm_rescue_resolved": "high",
    "llm_conflict_rescue_resolved": "high",
    "llm_near_agree_rescue_resolved": "high",
    "llm_rescue_intentional_drop": "high",
    "llm_conflict_rescue_intentional_drop": "high",
    "llm_near_agree_rescue_intentional_drop": "high",
    "llm_gap": "high",
    "median_source": "high",
    "deterministic_two_source": "medium_high",
    "zone_fallback_best_source": "medium",
    "llm_conflict_fallback_best_source": "medium",
    "llm_near_agree_fallback_best_source": "medium",
}

# Weights used in the quality score formula.  Each resolved segment
# contributes (weight * 1.0) to the numerator; perfect score is when
# every segment has weight 1.0.
METHOD_QUALITY_WEIGHT: dict[str, float] = {
    "llm_conflict": 1.0,
    "llm_near_agree": 1.0,
    "llm_rescue_resolved": 0.95,
    "llm_conflict_rescue_resolved": 0.95,
    "llm_near_agree_rescue_resolved": 0.95,
    "llm_rescue_intentional_drop": 0.95,
    "llm_conflict_rescue_intentional_drop": 0.95,
    "llm_near_agree_rescue_intentional_drop": 0.95,
    "llm_gap": 1.0,
    "median_source": 1.0,
    "deterministic_two_source": 0.85,
    "zone_fallback_best_source": 0.40,
    "llm_conflict_fallback_best_source": 0.40,
    "llm_near_agree_fallback_best_source": 0.40,
}

# Section keywords used for heuristic section detection.
SECTION_KEYWORDS: dict[str, list[str]] = {
    "abstract": ["abstract", "summary"],
    "introduction": ["introduction", "background"],
    "methods": ["methods", "materials", "experimental", "procedures",
                "methodology", "subjects"],
    "results": ["results", "findings", "observations"],
    "discussion": ["discussion", "interpretation"],
    "conclusion": ["conclusion", "concluding"],
    "references": ["references", "bibliography", "literature cited",
                    "works cited"],
    "front_matter": ["author", "affiliation", "correspondence",
                     "keyword", "abbreviation", "funding",
                     "acknowledgment", "acknowledgement"],
}

# Sections where degradation matters most.
CRITICAL_SECTIONS = {"abstract", "results", "methods"}


# ---------------------------------------------------------------------------
# Quality score computation
# ---------------------------------------------------------------------------

def compute_quality_score(
    total_blocks: int,
    resolution_metadata: dict[str, dict],
) -> float:
    """Compute an overall quality score (0.0 - 1.0) for the extraction.

    Formula:
        quality_score = (agreed_weight + sum(method_weight_i)) / total_blocks

    Where:
    - agreed_weight accounts for AGREE_EXACT and AGREE_NEAR blocks that
      needed no conflict resolution (weight 1.0 each).
    - method_weight_i is the quality weight for each resolved segment's
      resolution method (see METHOD_QUALITY_WEIGHT).
    - total_blocks is the total number of aligned segments (triples).
    """
    if total_blocks == 0:
        return 1.0

    resolved_count = len(resolution_metadata) if resolution_metadata else 0
    agreed_count = total_blocks - resolved_count

    # Agreed segments contribute weight 1.0 each
    weighted_sum = float(agreed_count) * 1.0

    # Resolved segments contribute their method-specific weight
    for meta in (resolution_metadata or {}).values():
        method = meta.get("method", "")
        weight = METHOD_QUALITY_WEIGHT.get(method, 0.5)
        weighted_sum += weight

    return min(weighted_sum / total_blocks, 1.0)


def quality_grade(score: float) -> str:
    """Map a quality score to a letter grade.

    A  (>= 0.95): Excellent — nearly all segments resolved at high quality
    B  (>= 0.85): Good — minor degradation, likely acceptable
    C  (>= 0.70): Fair — noticeable degradation, review recommended
    D  (>= 0.50): Poor — significant degradation, manual review needed
    F  (<  0.50): Fail — extraction substantially degraded
    """
    if score >= 0.95:
        return "A"
    if score >= 0.85:
        return "B"
    if score >= 0.70:
        return "C"
    if score >= 0.50:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Section detection heuristic
# ---------------------------------------------------------------------------

def _detect_section(
    segment_id: str,
    triples: list,
    seg_id_to_idx: dict[str, int],
) -> str:
    """Guess which paper section a segment belongs to by scanning
    preceding headings.

    Walks backwards from the segment looking for the nearest heading
    block, then matches heading text against SECTION_KEYWORDS.
    Returns the section name or "unknown".
    """
    idx = seg_id_to_idx.get(segment_id)
    if idx is None:
        return "unknown"

    for i in range(idx, -1, -1):
        triple = triples[i]
        for attr in ("grobid_block", "docling_block", "marker_block"):
            block = getattr(triple, attr, None)
            if block is not None and block.block_type == "heading":
                heading_text = (block.raw_text or "").lower().strip().lstrip("#").strip()
                for section, keywords in SECTION_KEYWORDS.items():
                    for kw in keywords:
                        if kw in heading_text:
                            return section
                break

    return "unknown"


# ---------------------------------------------------------------------------
# Confidence distribution computation
# ---------------------------------------------------------------------------

def _compute_confidence_distribution(
    confidences: list[float],
) -> dict:
    """Compute summary statistics for the confidence distribution."""
    if not confidences:
        return {
            "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0,
            "std_dev": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0,
            "below_50_pct": 0.0, "below_25_pct": 0.0,
        }

    n = len(confidences)
    sorted_c = sorted(confidences)

    def _percentile(data: list[float], p: float) -> float:
        k = (len(data) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        return data[f] * (c - k) + data[c] * (k - f)

    mean_val = sum(sorted_c) / n
    median_val = _percentile(sorted_c, 50)
    std_val = statistics.stdev(sorted_c) if n > 1 else 0.0

    below_50 = sum(1 for c in sorted_c if c < 0.50)
    below_25 = sum(1 for c in sorted_c if c < 0.25)

    return {
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "min": round(sorted_c[0], 4),
        "max": round(sorted_c[-1], 4),
        "std_dev": round(std_val, 4),
        "p10": round(_percentile(sorted_c, 10), 4),
        "p25": round(_percentile(sorted_c, 25), 4),
        "p75": round(_percentile(sorted_c, 75), 4),
        "p90": round(_percentile(sorted_c, 90), 4),
        "below_50_pct": round(100.0 * below_50 / n, 1),
        "below_25_pct": round(100.0 * below_25 / n, 1),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_degradation_metrics(
    triples: list,
    resolution_metadata: dict[str, dict],
    audit_entries: list[dict],
    total_blocks: int,
    zone_resolution_tokens: int = 0,
    rescue_call_tokens: int = 0,
) -> dict:
    """Build the full degradation_metrics dict from pipeline outputs.

    This is called at the end of merge_with_consensus() after conflict
    resolution and assembly are complete.

    Args:
        triples: The list of AlignedTriple objects from the consensus pipeline.
        resolution_metadata: The dict of {segment_id: metadata_dict} produced
            by _resolve_conflicts_layered().  Each metadata dict has at minimum:
            method, chosen_source, confidence, sources_agreeing,
            max_pair_similarity, and optionally zone_id, degraded,
            rescue_explanation.
        audit_entries: The list of per-segment audit dicts from
            _build_audit_entries().
        total_blocks: Total number of aligned triples (including AGREE_EXACT).
        zone_resolution_tokens: Total tokens used by zone LLM calls.
        rescue_call_tokens: Total tokens used by rescue LLM calls.

    Returns:
        A dict ready to be included in consensus_metrics as
        consensus_metrics["degradation_metrics"].
    """
    metadata = resolution_metadata or {}

    # --- Build index helpers ---
    seg_id_to_idx: dict[str, int] = {}
    for i, t in enumerate(triples):
        seg_id_to_idx[t.segment_id] = i

    # --- Quality score ---
    q_score = compute_quality_score(total_blocks, metadata)
    q_grade = quality_grade(q_score)

    # --- Resolution summary ---
    all_methods = sorted(set(
        list(METHOD_QUALITY_TIER.keys())
        + [meta.get("method", "") for meta in metadata.values() if meta.get("method")]
    ))
    method_counts: dict[str, int] = {m: 0 for m in all_methods}
    for meta in metadata.values():
        m = meta.get("method", "")
        if m in method_counts:
            method_counts[m] += 1
        else:
            method_counts[m] = method_counts.get(m, 0) + 1

    total_resolved = len(metadata)
    by_method = {}
    for m in all_methods:
        c = method_counts.get(m, 0)
        by_method[m] = {
            "count": c,
            "pct": round(100.0 * c / total_resolved, 1) if total_resolved > 0 else 0.0,
        }

    # Quality tier grouping
    tier_groups: dict[str, list[str]] = {
        "high": [],
        "medium_high": [],
        "medium": [],
    }
    for m in all_methods:
        tier = METHOD_QUALITY_TIER.get(m, "medium")
        if tier in tier_groups:
            tier_groups[tier].append(m)

    by_tier = {}
    for tier_name, tier_methods in tier_groups.items():
        tier_count = sum(method_counts.get(m, 0) for m in tier_methods)
        by_tier[tier_name] = {
            "count": tier_count,
            "pct": round(100.0 * tier_count / total_resolved, 1) if total_resolved > 0 else 0.0,
            "methods": tier_methods,
        }

    resolution_summary = {
        "total_resolved_segments": total_resolved,
        "by_method": by_method,
        "by_quality_tier": by_tier,
    }

    # --- Degraded segments ---
    degraded_details: list[dict] = []
    for seg_id, meta in metadata.items():
        if not meta.get("degraded", False):
            continue
        audit_entry = next(
            (e for e in audit_entries if e.get("segment_id") == seg_id),
            {},
        )
        block_type = audit_entry.get("block_type", "unknown")
        section_hint = _detect_section(seg_id, triples, seg_id_to_idx)

        chosen_text = audit_entry.get("chosen_text", "")
        text_len = len(chosen_text) if chosen_text else 0

        degraded_details.append({
            "segment_id": seg_id,
            "zone_id": meta.get("zone_id", ""),
            "block_type": block_type,
            "section_hint": section_hint,
            "method": meta.get("method", ""),
            "chosen_source": meta.get("chosen_source", ""),
            "confidence": round(meta.get("confidence", 0.0), 4),
            "sources_present": meta.get("sources_agreeing", []),
            "max_pair_similarity": round(meta.get("max_pair_similarity", 0.0), 4),
            "rescue_attempted": True,
            "rescue_explanation": meta.get("rescue_explanation"),
            "text_length_chars": text_len,
        })

    degraded_count = len(degraded_details)
    degraded_segments = {
        "count": degraded_count,
        "pct_of_total": round(100.0 * degraded_count / total_blocks, 1) if total_blocks > 0 else 0.0,
        "pct_of_resolved": round(100.0 * degraded_count / total_resolved, 1) if total_resolved > 0 else 0.0,
        "segment_ids": [d["segment_id"] for d in degraded_details],
        "details": degraded_details,
    }

    # --- Rescue segments (successfully rescued, not degraded) ---
    rescue_details: list[dict] = []
    rescue_method_patterns = ("rescue_resolved", "rescue_intentional_drop")
    for seg_id, meta in metadata.items():
        method = meta.get("method", "")
        if any(p in method for p in rescue_method_patterns):
            audit_entry = next(
                (e for e in audit_entries if e.get("segment_id") == seg_id),
                {},
            )
            rescue_details.append({
                "segment_id": seg_id,
                "zone_id": meta.get("zone_id", ""),
                "block_type": audit_entry.get("block_type", "unknown"),
                "method": method,
                "rescue_explanation": meta.get("rescue_explanation", ""),
                "confidence": round(meta.get("confidence", 0.0), 4),
            })

    rescue_segments = {
        "count": len(rescue_details),
        "details": rescue_details,
    }

    # --- Confidence distribution ---
    all_confidences = [
        float(meta.get("confidence", 0.0))
        for meta in metadata.values()
    ]
    conf_dist = _compute_confidence_distribution(all_confidences)

    # --- Section risk ---
    section_total: dict[str, int] = {}
    section_degraded: dict[str, int] = {}

    for t in triples:
        section = _detect_section(t.segment_id, triples, seg_id_to_idx)
        section_total[section] = section_total.get(section, 0) + 1

    for d in degraded_details:
        s = d["section_hint"]
        section_degraded[s] = section_degraded.get(s, 0) + 1

    canonical_sections = [
        "abstract", "front_matter", "introduction", "methods",
        "results", "discussion", "conclusion", "references",
    ]
    section_risk: dict[str, dict] = {}
    for sec in canonical_sections:
        total_sec = section_total.get(sec, 0)
        deg_sec = section_degraded.get(sec, 0)
        pct_deg = round(100.0 * deg_sec / total_sec, 1) if total_sec > 0 else 0.0

        if deg_sec == 0:
            risk = "none"
        elif pct_deg <= 10.0:
            risk = "low"
        elif pct_deg <= 25.0:
            risk = "medium"
        else:
            risk = "high"

        section_risk[sec] = {
            "total_segments": total_sec,
            "degraded": deg_sec,
            "pct_degraded": pct_deg,
            "risk": risk,
        }

    if "unknown" in section_total:
        total_sec = section_total["unknown"]
        deg_sec = section_degraded.get("unknown", 0)
        pct_deg = round(100.0 * deg_sec / total_sec, 1) if total_sec > 0 else 0.0
        section_risk["unknown"] = {
            "total_segments": total_sec,
            "degraded": deg_sec,
            "pct_degraded": pct_deg,
            "risk": "low" if deg_sec > 0 else "none",
        }

    # --- Risk flags ---
    abstract_degraded = section_degraded.get("abstract", 0) > 0
    references_degraded = section_degraded.get("references", 0) > 0
    critical_degraded = any(
        section_degraded.get(s, 0) > 0 for s in CRITICAL_SECTIONS
    )
    high_degradation = (
        (100.0 * degraded_count / total_blocks) > 10.0
        if total_blocks > 0 else False
    )

    # Concentration check: are >50% of degraded segments in a single zone?
    zone_counts: dict[str, int] = {}
    for d in degraded_details:
        z = d.get("zone_id", "")
        if z:
            zone_counts[z] = zone_counts.get(z, 0) + 1
    concentrated = False
    if degraded_count >= 2 and zone_counts:
        max_in_zone = max(zone_counts.values())
        concentrated = max_in_zone > (degraded_count * 0.5)

    # Low-confidence cluster: 3+ consecutive segments with confidence < 0.50
    low_conf_cluster = False
    consecutive_low = 0
    for t in triples:
        meta = metadata.get(t.segment_id)
        if meta and float(meta.get("confidence", 1.0)) < 0.50:
            consecutive_low += 1
            if consecutive_low >= 3:
                low_conf_cluster = True
                break
        else:
            consecutive_low = 0

    risk_flags = {
        "abstract_degraded": abstract_degraded,
        "references_degraded": references_degraded,
        "critical_section_degraded": critical_degraded,
        "high_degradation_rate": high_degradation,
        "degradation_concentrated": concentrated,
        "low_confidence_cluster": low_conf_cluster,
    }

    # --- Token efficiency ---
    # Only compute token efficiency when actual token data is available.
    # When both inputs are zero, token tracking is not yet wired in and
    # we should not surface misleading numbers.
    has_token_data = zone_resolution_tokens > 0 or rescue_call_tokens > 0
    if has_token_data:
        total_consensus_tokens = zone_resolution_tokens + rescue_call_tokens
        estimated_full_merge = total_blocks * 300 + 2000
        tokens_saved = max(0, estimated_full_merge - total_consensus_tokens)
        savings_pct = round(
            100.0 * tokens_saved / estimated_full_merge, 1,
        ) if estimated_full_merge > 0 else 0.0

        token_efficiency = {
            "zone_resolution_tokens": zone_resolution_tokens,
            "rescue_call_tokens": rescue_call_tokens,
            "total_consensus_tokens": total_consensus_tokens,
            "estimated_full_merge_tokens": estimated_full_merge,
            "tokens_saved": tokens_saved,
            "savings_pct": savings_pct,
        }
    else:
        token_efficiency = {"status": "not_tracked"}

    # --- Time-series fields ---
    rescue_attempt_count = degraded_count + len(rescue_details)
    rescue_success_count = len(rescue_details)
    rescue_success_rate = round(
        100.0 * rescue_success_count / rescue_attempt_count, 1,
    ) if rescue_attempt_count > 0 else 0.0

    time_series = {
        "quality_score": round(q_score, 3),
        "degraded_segment_count": degraded_count,
        "degraded_segment_pct": round(
            100.0 * degraded_count / total_blocks, 1,
        ) if total_blocks > 0 else 0.0,
        "rescue_attempt_count": rescue_attempt_count,
        "rescue_success_count": rescue_success_count,
        "rescue_success_rate": rescue_success_rate,
        "fallback_best_source_count": method_counts.get("zone_fallback_best_source", 0)
        + method_counts.get("llm_conflict_fallback_best_source", 0)
        + method_counts.get("llm_near_agree_fallback_best_source", 0),
        "mean_confidence": round(conf_dist.get("mean", 0.0), 4),
        "min_confidence": round(conf_dist.get("min", 0.0), 4),
        "abstract_degraded": abstract_degraded,
        "total_consensus_tokens": (zone_resolution_tokens + rescue_call_tokens) if has_token_data else None,
        "full_merge_avoided": True,
    }

    # --- Assemble final dict ---
    return {
        "quality_score": round(q_score, 3),
        "quality_grade": q_grade,
        "resolution_summary": resolution_summary,
        "degraded_segments": degraded_segments,
        "rescue_segments": rescue_segments,
        "confidence_distribution": conf_dist,
        "section_risk": section_risk,
        "risk_flags": risk_flags,
        "token_efficiency": token_efficiency,
        "time_series_fields": time_series,
    }
