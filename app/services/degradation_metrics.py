"""Degradation measurement models for the consensus pipeline.

Computes quality scores, risk indicators, and per-segment degradation
detail from the resolution_metadata dict and audit_entries list that
the consensus pipeline already produces.
"""

from __future__ import annotations

import math
import re
import statistics


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
    "llm_conflict_numeric_guard_rescue_resolved": "medium_high",
    "llm_near_agree_numeric_guard_rescue_resolved": "medium_high",
    "llm_gap_numeric_guard_rescue_resolved": "medium_high",
    "llm_conflict_numeric_guard_fallback_best_source": "medium",
    "llm_near_agree_numeric_guard_fallback_best_source": "medium",
    "llm_gap_numeric_guard_fallback_best_source": "medium",
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
    # Numeric integrity guard: rescue succeeded — the guard caught a problem
    # and the retry LLM fixed it. Output is correct. Mild penalty to signal
    # the initial LLM was unreliable on this segment.
    "llm_conflict_numeric_guard_rescue_resolved": 0.85,
    "llm_near_agree_numeric_guard_rescue_resolved": 0.85,
    "llm_gap_numeric_guard_rescue_resolved": 0.85,
    # Numeric integrity guard: rescue also failed, fell back to picking the
    # best single extractor's text. Output is correct but not merged.
    "llm_conflict_numeric_guard_fallback_best_source": 0.30,
    "llm_near_agree_numeric_guard_fallback_best_source": 0.30,
    "llm_gap_numeric_guard_fallback_best_source": 0.30,
}

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

    base = min(weighted_sum / total_blocks, 1.0)

    # Paper-level penalty only when numeric integrity rescue FAILED and we
    # had to fall back to best-source (the output has unmerged text).
    # Successful rescues are handled by per-segment weights alone — a working
    # safety system should not catastrophically penalize the score.
    metas = list((resolution_metadata or {}).values())
    numeric_guard_fallback = any(
        str(m.get("method", "")).endswith("numeric_guard_fallback_best_source")
        for m in metas
    )
    if numeric_guard_fallback:
        return round(base * 0.80, 6)
    return base


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

_WS_RE = re.compile(r"\s+")

def _norm_heading_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "")).strip().lower()


def _get_present_blocks(triple) -> list:
    blocks = []
    for attr in ("grobid_block", "docling_block", "marker_block"):
        b = getattr(triple, attr, None)
        if b is not None:
            blocks.append(b)
    return blocks


def _build_heading_labels(heading_hierarchy: list[dict]) -> list[dict]:
    """Build hierarchical labels from a resolved heading list.

    Each returned entry includes:
    - text: heading text
    - level: heading level (1-6)
    - label: hierarchical path label using resolved levels, e.g.
             "Materials and Methods > RNA Extraction"
    """
    stack: list[tuple[str, int]] = []
    labeled: list[dict] = []

    for h in (heading_hierarchy or []):
        text = (h.get("text") or "").strip()
        level = int(h.get("level") or 0)
        if not text or level < 1 or level > 6:
            continue

        while stack and stack[-1][1] >= level:
            stack.pop()
        stack.append((text, level))
        label = " > ".join(t for t, _lvl in stack)

        labeled.append({
            "text": text,
            "level": level,
            "label": label,
            "norm_text": _norm_heading_text(text),
        })

    return labeled


def build_degradation_metrics(
    triples: list,
    resolution_metadata: dict[str, dict],
    audit_entries: list[dict],
    total_blocks: int,
    heading_hierarchy: list[dict],
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
        heading_hierarchy: Finalized heading list (text + level) extracted from
            the post-hierarchy merged markdown. Section risk is computed from
            this resolved hierarchy, not keyword guessing.
        zone_resolution_tokens: Total tokens used by zone LLM calls.
        rescue_call_tokens: Total tokens used by rescue LLM calls.

    Returns:
        A dict ready to be included in consensus_metrics as
        consensus_metrics["degradation_metrics"].
    """
    metadata = resolution_metadata or {}

    # --- Section assignment from resolved heading hierarchy ---
    resolved_headings = _build_heading_labels(heading_hierarchy)
    resolved_cursor = 0
    current_section_label: str | None = None
    current_section_level: int | None = None

    seg_to_section: dict[str, str | None] = {}
    seg_to_section_level: dict[str, int | None] = {}

    for t in triples:
        blocks = _get_present_blocks(t)
        heading_block = next((b for b in blocks if getattr(b, "block_type", "") == "heading"), None)

        if heading_block is not None:
            candidate = _norm_heading_text(getattr(heading_block, "raw_text", "") or "")
            matched = False
            for i in range(resolved_cursor, len(resolved_headings)):
                if resolved_headings[i]["norm_text"] == candidate:
                    current_section_label = resolved_headings[i]["label"]
                    current_section_level = resolved_headings[i]["level"]
                    resolved_cursor = i + 1
                    matched = True
                    break
            # Heading segments themselves are not counted toward section totals.
            if not matched:
                continue
            continue

        seg_to_section[t.segment_id] = current_section_label
        seg_to_section_level[t.segment_id] = current_section_level

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
        section_label = seg_to_section.get(seg_id)
        section_level = seg_to_section_level.get(seg_id)

        chosen_text = audit_entry.get("chosen_text", "")
        text_len = len(chosen_text) if chosen_text else 0

        degraded_details.append({
            "segment_id": seg_id,
            "zone_id": meta.get("zone_id", ""),
            "block_type": block_type,
            "section_heading": section_label,
            "section_heading_level": section_level,
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
    section_level: dict[str, int | None] = {}

    for seg_id, label in seg_to_section.items():
        if not label:
            continue
        section_total[label] = section_total.get(label, 0) + 1
        if label not in section_level:
            section_level[label] = seg_to_section_level.get(seg_id)

    for d in degraded_details:
        s = d.get("section_heading")
        if not s:
            continue
        section_degraded[s] = section_degraded.get(s, 0) + 1

    section_risk: dict[str, dict] = {}
    for sec, total_sec in section_total.items():
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
            "heading_level": section_level.get(sec),
            "total_segments": total_sec,
            "degraded": deg_sec,
            "pct_degraded": pct_deg,
            "risk": risk,
        }

    # --- Risk flags ---
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

    high_risk_top_level_heading = any(
        (info.get("risk") == "high" and (info.get("heading_level") or 99) <= 2)
        for info in section_risk.values()
        if isinstance(info, dict)
    )

    risk_flags = {
        "high_degradation_rate": high_degradation,
        "degradation_concentrated": concentrated,
        "low_confidence_cluster": low_conf_cluster,
        "high_risk_top_level_heading": high_risk_top_level_heading,
    }

    # --- Numeric integrity (critical scientific accuracy) ---
    numeric_guard_ids = [
        seg_id for seg_id, meta in metadata.items()
        if isinstance(meta.get("numeric_integrity"), dict) and meta["numeric_integrity"].get("novel_numbers_initial")
    ]
    numeric_guard_fallback_ids = [
        seg_id for seg_id, meta in metadata.items()
        if str(meta.get("method", "")).endswith("numeric_guard_fallback_best_source")
    ]
    numeric_integrity = {
        "count": len(numeric_guard_ids),
        "fallback_count": len(numeric_guard_fallback_ids),
        "segment_ids": numeric_guard_ids,
    }
    risk_flags["numeric_integrity_violation"] = len(numeric_guard_ids) > 0
    risk_flags["numeric_integrity_fallback"] = len(numeric_guard_fallback_ids) > 0

    # --- Token efficiency ---
    # Only compute token efficiency when actual token data is available.
    # When both inputs are zero, token tracking is not yet wired in and
    # we should not surface misleading numbers.
    has_token_data = zone_resolution_tokens > 0 or rescue_call_tokens > 0
    if has_token_data:
        total_consensus_tokens = zone_resolution_tokens + rescue_call_tokens

        token_efficiency = {
            "zone_resolution_tokens": zone_resolution_tokens,
            "rescue_call_tokens": rescue_call_tokens,
            "total_consensus_tokens": total_consensus_tokens,
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
        "high_risk_top_level_heading": high_risk_top_level_heading,
        "numeric_integrity_violation": len(numeric_guard_ids) > 0,
        "numeric_integrity_violation_count": len(numeric_guard_ids),
        "total_consensus_tokens": (zone_resolution_tokens + rescue_call_tokens) if has_token_data else None,
    }

    # --- Assemble final dict ---
    return {
        "quality_score": round(q_score, 3),
        "quality_grade": q_grade,
        "resolution_summary": resolution_summary,
        "degraded_segments": degraded_segments,
        "rescue_segments": rescue_segments,
        "numeric_integrity": numeric_integrity,
        "confidence_distribution": conf_dist,
        "section_risk": section_risk,
        "risk_flags": risk_flags,
        "token_efficiency": token_efficiency,
        "time_series_fields": time_series,
    }
