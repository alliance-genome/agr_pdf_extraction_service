"""Layered conflict resolution logic for consensus segments."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from app.services.consensus_models import AGREE_NEAR, CONFLICT, GAP, AlignedTriple
from app.services.consensus_pipeline_steps import (
    _LAYERED_MEDIUM_SIM_THRESHOLD,
    _build_flanking_context,
    _content_similarity_check,
    _get_present_blocks,
    _numeric_count_ratio,
    _numeric_integrity_dropped_numbers,
    _numeric_integrity_novel_numbers,
    _extract_numeric_tokens_integrity,
    _pick_preferred_text,
    _segment_allowed_numeric_tokens,
    _triple_source_count,
    clean_output_md,
    reconstruct_segment_from_micro_conflicts,
)

if TYPE_CHECKING:
    from app.services.llm_service import LLM

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Conflict resolution helpers
# ---------------------------------------------------------------------------

def _bundle_source_texts(bundle: dict) -> dict[str, str]:
    """Return non-empty extractor texts from a conflict bundle."""
    result: dict[str, str] = {}
    for source in ("grobid", "docling", "marker"):
        value = (bundle.get(source) or "").strip()
        if value:
            result[source] = value
    return result


def _best_source_fallback(seg: dict) -> str:
    """Pick the best available source text when both zone and rescue resolution fail.

    Strategy:
    1. If one source text contains another (containment), pick the longer one
    2. Otherwise, pick the longest source text (most complete)

    Returns empty string only if ALL sources are empty.
    """
    source_texts = _bundle_source_texts(seg)
    if not source_texts:
        return ""

    texts = list(source_texts.items())  # [(name, text), ...]
    texts.sort(key=lambda x: len(x[1]), reverse=True)

    longest_name, longest_text = texts[0]

    # Containment check — if shorter texts are substrings of longest, it's clearly best
    for name, text in texts[1:]:
        if text in longest_text:
            return longest_text

    # Default: return longest source text
    return longest_text


def _gather_rescue_context(
    seg_id: str,
    triples: list[AlignedTriple],
    resolved_so_far: dict[str, str],
    flanking_count: int = 5,
) -> tuple[list[dict], dict[str, str]]:
    """Gather enriched context for a rescue LLM call on a single failed segment.

    Returns:
        extra_flanking: list of dicts with segment_id + text for surrounding segments
        neighboring_resolved: dict of seg_id -> resolved text for nearby segments
    """
    # Find the segment's index in the triples list
    seg_idx = None
    for i, t in enumerate(triples):
        if t.segment_id == seg_id:
            seg_idx = i
            break
    if seg_idx is None:
        return [], {}

    # Gather flanking segments (more than the normal 2)
    extra_flanking = []
    for offset in range(-flanking_count, flanking_count + 1):
        idx = seg_idx + offset
        if idx < 0 or idx >= len(triples) or idx == seg_idx:
            continue
        t = triples[idx]
        text = t.agreed_text or ""
        if text:
            extra_flanking.append({
                "segment_id": t.segment_id,
                "text": text,
                "classification": t.classification,
            })

    # Gather already-resolved neighbors
    neighboring_resolved = {}
    for offset in range(-3, 4):
        idx = seg_idx + offset
        if idx < 0 or idx >= len(triples) or idx == seg_idx:
            continue
        neighbor_id = triples[idx].segment_id
        if neighbor_id in resolved_so_far:
            neighboring_resolved[neighbor_id] = resolved_so_far[neighbor_id]

    return extra_flanking, neighboring_resolved


def _rescue_segment(
    seg_id: str,
    seg: dict,
    resolution_ctx: dict,
    triples: list[AlignedTriple],
    resolved: dict[str, str],
    metadata: dict[str, dict],
    source_texts: dict[str, str],
    max_pair_sim: float,
    llm: "LLM",
    method_prefix: str = "llm_conflict",
) -> None:
    """Three-tier rescue strategy for a segment that resolution returned empty.

    Tier 1: Rescue LLM call with enriched context and explanation request.
    Tier 2: Best-source fallback (deterministic, no LLM).
    Tier 3: Skip (all sources empty).

    Mutates ``resolved`` and ``metadata`` dicts in place.
    """
    # TIER 1: Rescue LLM call — focused single-segment with enriched context
    extra_flanking, neighboring_resolved = _gather_rescue_context(
        seg_id, triples, resolved, flanking_count=5,
    )
    from config import Config
    rescue_result = llm.rescue_single_segment(
        seg=seg,
        zone=resolution_ctx,
        neighboring_resolved=neighboring_resolved,
        extra_flanking=extra_flanking,
        model=Config.LLM_MODEL_GENERAL_RESCUE,
    )

    if rescue_result is not None:
        if rescue_result.is_intentionally_empty:
            # LLM says this segment SHOULD be empty — respect that with audit trail
            resolved[seg_id] = ""
            metadata[seg_id] = {
                "method": f"{method_prefix}_rescue_intentional_drop",
                "chosen_source": "llm",
                "confidence": 0.0,
                "sources_agreeing": sorted(source_texts.keys()),
                "max_pair_similarity": round(max_pair_sim, 4),
                "context_id": resolution_ctx["context_id"],
                "rescue_explanation": rescue_result.explanation,
            }
            logger.info(
                "Rescue for %s: LLM says intentionally empty — %s",
                seg_id, rescue_result.explanation,
            )
            return
        elif rescue_result.resolved_text.strip():
            # LLM provided text on the rescue attempt — use it
            resolved[seg_id] = rescue_result.resolved_text.strip()
            metadata[seg_id] = {
                "method": f"{method_prefix}_rescue_resolved",
                "chosen_source": "llm",
                "confidence": round(
                    _mean_similarity_to_sources(
                        rescue_result.resolved_text, source_texts
                    ), 4,
                ),
                "sources_agreeing": sorted(source_texts.keys()),
                "max_pair_similarity": round(max_pair_sim, 4),
                "context_id": resolution_ctx["context_id"],
                "rescue_explanation": rescue_result.explanation,
            }
            logger.info(
                "Rescue for %s: LLM provided text — %s",
                seg_id, rescue_result.explanation,
            )
            return
        # else: rescue returned empty WITHOUT is_intentionally_empty — fall through to Tier 2

    # TIER 2: Best-source fallback — deterministic, no LLM
    fallback_text = _best_source_fallback(seg)
    if fallback_text:
        resolved[seg_id] = fallback_text
        metadata[seg_id] = {
            "method": f"{method_prefix}_fallback_best_source",
            "chosen_source": "best_available",
            "confidence": 0.0,
            "sources_agreeing": sorted(source_texts.keys()),
            "max_pair_similarity": round(max_pair_sim, 4),
            "context_id": resolution_ctx["context_id"],
            "degraded": True,
        }
        logger.warning(
            "Rescue also failed for %s; using best-source fallback", seg_id,
        )
        return

    # TIER 3: All sources empty — skip entirely
    logger.warning(
        "All resolution paths exhausted for %s with no text; skipping", seg_id,
    )


def _apply_numeric_integrity_guard(
    *,
    seg_id: str,
    seg: dict,
    resolution_ctx: dict,
    triples: list[AlignedTriple],
    resolved: dict[str, str],
    metadata: dict[str, dict],
    allowed_numbers: set[str],
    source_texts: dict[str, str],
    max_pair_sim: float,
    llm: "LLM",
    method_prefix: str,
) -> None:
    """Ensure resolved text does not introduce numeric tokens not present in sources.

    If novel numbers are detected, retries via single-segment rescue with an
    explicit numeric-integrity constraint and required justification. The rescue
    LLM's output is trusted (novel numbers after rescue are logged but accepted).
    Falls back to deterministic best-source only if rescue returns no text.
    """
    current = (resolved.get(seg_id, "") or "").strip()
    if not current:
        return

    novel = _numeric_integrity_novel_numbers(current, allowed_numbers)
    if not novel:
        return

    logger.error(
        "Numeric integrity guard: %s introduced novel number(s): %s",
        seg_id, ", ".join(novel),
    )

    # Retry once with a focused rescue call that explicitly enforces numeric integrity.
    from config import Config
    extra_flanking, neighboring_resolved = _gather_rescue_context(
        seg_id, triples, resolved, flanking_count=5,
    )
    rescue_result = llm.rescue_single_segment(
        seg=seg,
        zone=resolution_ctx,
        neighboring_resolved=neighboring_resolved,
        extra_flanking=extra_flanking,
        model=Config.LLM_MODEL_NUMERIC_RESCUE,
        reason="numeric_integrity",
        previous_text=current,
        novel_numbers=novel,
    )

    seg_status = seg.get("status", "conflict")
    candidate: str | None = None
    explanation: str = ""
    if rescue_result is not None:
        explanation = rescue_result.explanation or ""
        if rescue_result.is_intentionally_empty:
            # Only GAP segments may be dropped intentionally.
            if seg_status == "gap":
                candidate = ""
            else:
                candidate = None
        else:
            candidate = (rescue_result.resolved_text or "").strip()

    # Conflict/near_agree segments must not become empty.
    if seg_status in ("conflict", "near_agree") and not candidate:
        candidate = None

    # Trust the rescue LLM's output — it was explicitly told about the
    # numeric issue and given context to fix it.  Still log any remaining
    # novel numbers for telemetry, but accept the text.
    if candidate is not None:
        novel2 = _numeric_integrity_novel_numbers(candidate, allowed_numbers)
        if novel2:
            logger.info(
                "Numeric integrity guard: %s rescue still has novel number(s) %s — trusting rescue LLM",
                seg_id, ", ".join(novel2),
            )
        resolved[seg_id] = candidate
        metadata[seg_id] = {
            "method": f"{method_prefix}_numeric_guard_rescue_resolved",
            "chosen_source": "llm",
            "confidence": round(_mean_similarity_to_sources(candidate, source_texts), 4),
            "sources_agreeing": sorted(source_texts.keys()),
            "max_pair_similarity": round(max_pair_sim, 4),
            "context_id": resolution_ctx.get("context_id", ""),
            "degraded": bool(novel2),
            "numeric_integrity": {
                "severity": "critical" if novel2 else "resolved",
                "action": "rescue_resolved",
                "novel_numbers_initial": novel,
                "novel_numbers_after_rescue": novel2,
                "initial_text": current,
                "rescued_text": candidate,
                "explanation": explanation,
            },
        }
        return

    # Rescue returned nothing — deterministic fallback to best source.
    if seg_status == "gap":
        fallback_text = (seg.get("text") or "").strip()
    else:
        fallback_text = _best_source_fallback(seg).strip()

    final_novel = _numeric_integrity_novel_numbers(fallback_text, allowed_numbers)
    resolved[seg_id] = fallback_text
    metadata[seg_id] = {
        "method": f"{method_prefix}_numeric_guard_fallback_best_source",
        "chosen_source": "best_available",
        "confidence": 0.0,
        "sources_agreeing": sorted(source_texts.keys()),
        "max_pair_similarity": round(max_pair_sim, 4),
        "context_id": resolution_ctx.get("context_id", ""),
        "degraded": True,
        "numeric_integrity": {
            "severity": "critical",
            "action": "fallback_no_rescue_text",
            "novel_numbers_initial": novel,
            "novel_numbers_final": final_novel,
            "initial_text": current,
            "fallback_text": fallback_text,
        },
    }


_CONTENT_SIMILARITY_MIN = 0.50
_NUMERIC_COUNT_RATIO_MIN = 0.50
_NUMERIC_COUNT_SOURCE_MIN = 4


def _apply_post_resolution_validation(
    *,
    seg_id: str,
    seg: dict,
    resolution_ctx: dict,
    triples: list["AlignedTriple"],
    resolved: dict[str, str],
    metadata: dict[str, dict],
    source_texts: dict[str, str],
    max_pair_sim: float,
    llm: "LLM",
    method_prefix: str,
) -> None:
    """Run bidirectional numeric, content-similarity, and truncation checks.

    If any check fails, the segment is routed to rescue (same path as the
    existing numeric integrity guard).  The checks are:

    1. **Dropped numbers** — numbers in >=2 sources but missing from output.
    2. **Content similarity** — max similarity to any source is too low,
       suggesting the LLM pulled content from the wrong section.
    3. **Numeric truncation** — output has far fewer numeric tokens than the
       most complete source (catches p-value/measurement truncation).

    Mutates ``resolved`` and ``metadata`` in place.
    """
    current = (resolved.get(seg_id, "") or "").strip()
    if not current:
        return

    # --- Check 1: Dropped numbers ---
    dropped = _numeric_integrity_dropped_numbers(current, seg)
    if dropped:
        logger.warning(
            "Post-validation: %s dropped consensus number(s): %s",
            seg_id, ", ".join(dropped),
        )
        _route_to_rescue(
            seg_id=seg_id, seg=seg, resolution_ctx=resolution_ctx, triples=triples,
            resolved=resolved, metadata=metadata, source_texts=source_texts,
            max_pair_sim=max_pair_sim, llm=llm, method_prefix=method_prefix,
            reason="dropped_numbers", detail=dropped,
        )
        return

    # --- Check 2: Content similarity ---
    if source_texts:
        max_sim = _content_similarity_check(current, source_texts)
        if max_sim < _CONTENT_SIMILARITY_MIN:
            logger.warning(
                "Post-validation: %s content similarity %.3f < %.2f threshold",
                seg_id, max_sim, _CONTENT_SIMILARITY_MIN,
            )
            _route_to_rescue(
                seg_id=seg_id, seg=seg, resolution_ctx=resolution_ctx, triples=triples,
                resolved=resolved, metadata=metadata,
                source_texts=source_texts, max_pair_sim=max_pair_sim,
                llm=llm, method_prefix=method_prefix,
                reason="low_content_similarity",
                detail=f"max_sim={max_sim:.3f}",
            )
            return

    # --- Check 3: Numeric truncation ---
    if source_texts:
        out_count, max_src_count = _numeric_count_ratio(current, source_texts)
        if (
            max_src_count >= _NUMERIC_COUNT_SOURCE_MIN
            and out_count < max_src_count * _NUMERIC_COUNT_RATIO_MIN
        ):
            logger.warning(
                "Post-validation: %s numeric truncation — output has %d nums vs source max %d",
                seg_id, out_count, max_src_count,
            )
            _route_to_rescue(
                seg_id=seg_id, seg=seg, resolution_ctx=resolution_ctx, triples=triples,
                resolved=resolved, metadata=metadata,
                source_texts=source_texts, max_pair_sim=max_pair_sim,
                llm=llm, method_prefix=method_prefix,
                reason="numeric_truncation",
                detail=f"output={out_count}, source_max={max_src_count}",
            )
            return


def _route_to_rescue(
    *,
    seg_id: str,
    seg: dict,
    resolution_ctx: dict,
    triples: list["AlignedTriple"],
    resolved: dict[str, str],
    metadata: dict[str, dict],
    source_texts: dict[str, str],
    max_pair_sim: float,
    llm: "LLM",
    method_prefix: str,
    reason: str,
    detail: object,
) -> None:
    """Route a segment that failed post-resolution validation to rescue.

    Uses the numeric-integrity rescue path (gpt-5.2 with explanation) since
    that prompt already enforces strict fidelity to source content.
    """
    current = (resolved.get(seg_id, "") or "").strip()
    from config import Config
    extra_flanking, neighboring_resolved = _gather_rescue_context(
        seg_id, triples, resolved, flanking_count=5,
    )
    rescue_result = llm.rescue_single_segment(
        seg=seg,
        zone=resolution_ctx,
        neighboring_resolved=neighboring_resolved,
        extra_flanking=extra_flanking,
        model=Config.LLM_MODEL_NUMERIC_RESCUE,
        reason="numeric_integrity",
        previous_text=current,
        novel_numbers=[],
    )

    candidate: str | None = None
    explanation: str = ""
    if rescue_result is not None:
        explanation = rescue_result.explanation or ""
        seg_status = seg.get("status", "conflict")
        if rescue_result.is_intentionally_empty:
            if seg_status == "gap":
                candidate = ""
            else:
                candidate = None
        else:
            candidate = (rescue_result.resolved_text or "").strip()

        if seg_status in ("conflict", "near_agree") and not candidate:
            candidate = None

    if candidate is not None:
        resolved[seg_id] = candidate
        metadata[seg_id] = {
            "method": f"{method_prefix}_post_validation_rescue",
            "chosen_source": "llm",
            "confidence": round(_mean_similarity_to_sources(candidate, source_texts), 4),
            "sources_agreeing": sorted(source_texts.keys()),
            "max_pair_similarity": round(max_pair_sim, 4),
            "context_id": resolution_ctx.get("context_id", ""),
            "post_validation": {
                "reason": reason,
                "detail": str(detail),
                "initial_text": current,
                "rescued_text": candidate,
                "explanation": explanation,
            },
        }
    else:
        # Rescue failed — keep current text but flag as degraded
        logger.warning(
            "Post-validation rescue failed for %s (%s); keeping original",
            seg_id, reason,
        )
        existing_meta = metadata.get(seg_id, {})
        existing_meta["degraded"] = True
        existing_meta["post_validation_failed"] = {
            "reason": reason,
            "detail": str(detail),
        }
        metadata[seg_id] = existing_meta


def _pairwise_similarities(texts: list[str]) -> list[float]:
    """Pairwise token-set similarity scores in 0..1 range."""
    sims: list[float] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sims.append(fuzz.token_set_ratio(texts[i], texts[j]) / 100.0)
    return sims


def _mean_similarity_to_sources(candidate: str, source_texts: dict[str, str]) -> float:
    """Average similarity between a candidate and source extractor texts."""
    if not candidate or not source_texts:
        return 0.0
    sims = [
        fuzz.token_set_ratio(candidate, source_text) / 100.0
        for source_text in source_texts.values()
        if source_text
    ]
    if not sims:
        return 0.0
    return float(sum(sims) / len(sims))


def _resolve_conflict_median_source(bundle: dict) -> tuple[str, float, str]:
    """Layer 2: median-source selection — pick the source most similar to all others.

    Unlike token-level voting (which can produce chimeric output by mixing tokens
    from different sources), this always returns one source's text verbatim.
    """
    source_texts = _bundle_source_texts(bundle)
    if len(source_texts) < 2:
        raise ValueError("Median-source requires at least two source texts")

    sources = list(source_texts.keys())
    texts = list(source_texts.values())

    # Median-source is meaningless with exactly 2 sources: sim(A,B)==sim(B,A)
    # so both get the same score and the "winner" is arbitrary dict order.
    # Require 3+ sources for a meaningful median vote.
    if len(source_texts) < 3:
        raise ValueError("Median-source requires 3+ sources to break ties; escalate to LLM")

    best_text = texts[0]
    best_source = sources[0]
    best_score = -1.0

    for i, candidate in enumerate(texts):
        sims = [
            fuzz.token_set_ratio(candidate, other) / 100.0
            for other in texts if other != candidate
        ]
        avg_sim = (sum(sims) / len(sims)) if sims else 0.0
        if avg_sim > best_score:
            best_score = avg_sim
            best_text = candidate
            best_source = sources[i]

    chosen = best_text.strip()
    if not chosen:
        raise ValueError("Median-source produced empty consensus")

    confidence = _mean_similarity_to_sources(chosen, source_texts)

    logger.info(
        "Median-source for %s: chose %s (avg_sim=%.3f, confidence=%.3f)",
        bundle["segment_id"], best_source, best_score, confidence,
    )

    return chosen, confidence, best_source


def _tokens_to_text(tokens: list[str]) -> str:
    """Join token list into readable text while preserving punctuation spacing."""
    text = " ".join(tokens).strip()
    if not text:
        return ""
    text = re.sub(r" ([.,;:!?)])", r"\1", text)
    text = re.sub(r"([(]) ", r"\1", text)
    return text.strip()


def _triple_to_segment_dict(triple: AlignedTriple, *, status: str) -> dict:
    """Build a segment dict compatible with existing guard/rescue utilities."""
    block_type = "paragraph"
    blocks = _get_present_blocks(triple)
    if blocks:
        block_type = blocks[0].block_type

    seg = {
        "segment_id": triple.segment_id,
        "status": status,
        "block_type": block_type,
        "grobid": (triple.grobid_block.source_md or triple.grobid_block.raw_text) if triple.grobid_block else "",
        "docling": (triple.docling_block.source_md or triple.docling_block.raw_text) if triple.docling_block else "",
        "marker": (triple.marker_block.source_md or triple.marker_block.raw_text) if triple.marker_block else "",
    }
    if status == "gap":
        blocks = _get_present_blocks(triple)
        seg["gap_source"] = blocks[0].source if blocks else "unknown"
        seg["text"] = triple.agreed_text or ""
    return seg


def _build_micro_conflict_payload(
    triple: AlignedTriple,
    result: MicroConflictResult,
) -> dict:
    """Build compact payload containing only disagreement spans + local context."""
    payload_conflicts = []
    for conflict in result.conflicts:
        payload_conflicts.append({
            "conflict_id": conflict.conflict_id,
            "context_before": _tokens_to_text(conflict.context_before),
            "disagreement": {
                "grobid": _tokens_to_text(conflict.grobid_tokens),
                "docling": _tokens_to_text(conflict.docling_tokens),
                "marker": _tokens_to_text(conflict.marker_tokens),
            },
            "context_after": _tokens_to_text(conflict.context_after),
        })

    return {
        "segment_id": triple.segment_id,
        "block_type": result.block_type,
        "micro_conflicts": payload_conflicts,
    }


def _resolve_conflicts_micro(
    triples: list[AlignedTriple],
    micro_results: dict[str, MicroConflictResult],
    llm: "LLM",
    medium_similarity_threshold: float = _LAYERED_MEDIUM_SIM_THRESHOLD,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Resolve CONFLICT and GAP segments with per-segment micro-conflict extraction."""
    resolved: dict[str, str] = {}
    metadata: dict[str, dict] = {}

    gap_triples: list[AlignedTriple] = []
    triple_index = {tr.segment_id: idx for idx, tr in enumerate(triples)}

    for triple in triples:
        if triple.classification == GAP:
            gap_triples.append(triple)
            continue
        if triple.classification != CONFLICT:
            continue

        result = micro_results.get(triple.segment_id)
        if result is None:
            logger.warning("Missing micro-conflict result for %s; routing to rescue", triple.segment_id)
            seg = _triple_to_segment_dict(triple, status="conflict")
            resolution_ctx = {"context_id": f"micro_{triple.segment_id}", "segments": [seg], "context_before": [], "context_after": []}
            source_texts = _bundle_source_texts(seg)
            source_values = list(source_texts.values())
            pair_sims = _pairwise_similarities(source_values)
            max_pair_sim = max(pair_sims) if pair_sims else 0.0
            _rescue_segment(
                triple.segment_id, seg, resolution_ctx, triples, resolved, metadata,
                source_texts, max_pair_sim, llm, method_prefix="rescue_missing_micro",
            )
            continue

        seg = _triple_to_segment_dict(triple, status="conflict")
        resolution_ctx = {
            "context_id": f"micro_{triple.segment_id}",
            "segments": [seg],
            "context_before": [],
            "context_after": [],
        }
        source_texts = _bundle_source_texts(seg)
        source_values = list(source_texts.values())
        pair_sims = _pairwise_similarities(source_values)
        max_pair_sim = max(pair_sims) if pair_sims else 0.0
        allowed_numbers = _segment_allowed_numeric_tokens(seg)

        # Optional pre-LLM deterministic median-source resolution.
        if len(source_texts) >= 3 and max_pair_sim >= medium_similarity_threshold:
            try:
                median_text, confidence, chosen_extractor = _resolve_conflict_median_source(seg)
                resolved[triple.segment_id] = median_text.strip()
                metadata[triple.segment_id] = {
                    "method": "median_source",
                    "chosen_source": chosen_extractor,
                    "confidence": round(float(confidence), 4),
                    "sources_agreeing": sorted(source_texts.keys()),
                    "max_pair_similarity": round(max_pair_sim, 4),
                    "context_id": resolution_ctx["context_id"],
                    "micro_conflicts": len(result.conflicts),
                    "majority_agree_ratio": round(result.majority_agree_ratio, 4),
                }
                _apply_numeric_integrity_guard(
                    seg_id=triple.segment_id,
                    seg=seg,
                    resolution_ctx=resolution_ctx,
                    triples=triples,
                    resolved=resolved,
                    metadata=metadata,
                    allowed_numbers=allowed_numbers,
                    source_texts=source_texts,
                    max_pair_sim=max_pair_sim,
                    llm=llm,
                    method_prefix="llm_conflict",
                )
                _apply_post_resolution_validation(
                    seg_id=triple.segment_id,
                    seg=seg,
                    resolution_ctx=resolution_ctx,
                    triples=triples,
                    resolved=resolved,
                    metadata=metadata,
                    source_texts=source_texts,
                    max_pair_sim=max_pair_sim,
                    llm=llm,
                    method_prefix="llm_conflict",
                )
                continue
            except Exception as exc:
                logger.info("Median-source skipped for %s: %s", triple.segment_id, exc)

        if not result.conflicts:
            blocks = _get_present_blocks(triple)
            resolved_text = _pick_preferred_text(
                blocks,
                triple.grobid_block.block_type if triple.grobid_block else "paragraph",
            )
            resolved[triple.segment_id] = resolved_text
            metadata[triple.segment_id] = {
                "method": "majority_vote",
                "chosen_source": "majority_vote",
                "confidence": round(_mean_similarity_to_sources(resolved_text, source_texts), 4),
                "sources_agreeing": sorted(source_texts.keys()),
                "max_pair_similarity": round(max_pair_sim, 4),
                "context_id": resolution_ctx["context_id"],
                "micro_conflicts": 0,
                "majority_agree_ratio": round(result.majority_agree_ratio, 4),
            }
            _apply_numeric_integrity_guard(
                seg_id=triple.segment_id,
                seg=seg,
                resolution_ctx=resolution_ctx,
                triples=triples,
                resolved=resolved,
                metadata=metadata,
                allowed_numbers=allowed_numbers,
                source_texts=source_texts,
                max_pair_sim=max_pair_sim,
                llm=llm,
                method_prefix="llm_conflict",
            )
            _apply_post_resolution_validation(
                seg_id=triple.segment_id,
                seg=seg,
                resolution_ctx=resolution_ctx,
                triples=triples,
                resolved=resolved,
                metadata=metadata,
                source_texts=source_texts,
                max_pair_sim=max_pair_sim,
                llm=llm,
                method_prefix="llm_conflict",
            )
            continue

        payload = _build_micro_conflict_payload(triple, result)
        try:
            llm_response = llm.resolve_micro_conflicts(payload)
            resolved_map = {}
            for item in llm_response.resolved:
                if item.action == "drop":
                    resolved_map[item.conflict_id] = ""
                elif (item.text or "").strip():
                    resolved_map[item.conflict_id] = item.text

            # Fall back to longest source tokens for any unresolved spans
            for conflict in result.conflicts:
                if conflict.conflict_id not in resolved_map:
                    candidates = [
                        conflict.grobid_tokens,
                        conflict.docling_tokens,
                        conflict.marker_tokens,
                    ]
                    best = max(candidates, key=len) if candidates else []
                    fallback = _tokens_to_text(best)
                    if fallback:
                        resolved_map[conflict.conflict_id] = fallback
                        logger.warning(
                            "Micro-conflict %s unresolved; using longest source tokens as fallback",
                            conflict.conflict_id,
                        )

            resolved_text = reconstruct_segment_from_micro_conflicts(
                result.agreed_tokens,
                result.conflicts,
                resolved_map,
            )
        except Exception as exc:
            logger.warning("Micro-conflict LLM resolution failed for %s: %s", triple.segment_id, exc)
            resolved_text = ""

        if not resolved_text:
            _rescue_segment(
                triple.segment_id,
                seg,
                resolution_ctx,
                triples,
                resolved,
                metadata,
                source_texts,
                max_pair_sim,
                llm,
                method_prefix="llm_conflict",
            )
        else:
            resolved[triple.segment_id] = resolved_text
            metadata[triple.segment_id] = {
                "method": "llm_conflict",
                "chosen_source": "llm",
                "confidence": round(_mean_similarity_to_sources(resolved_text, source_texts), 4),
                "sources_agreeing": sorted(source_texts.keys()),
                "max_pair_similarity": round(max_pair_sim, 4),
                "context_id": resolution_ctx["context_id"],
                "micro_conflicts": len(result.conflicts),
                "majority_agree_ratio": round(result.majority_agree_ratio, 4),
                "resolution_strategy": "micro_conflict",
            }

        _apply_numeric_integrity_guard(
            seg_id=triple.segment_id,
            seg=seg,
            resolution_ctx=resolution_ctx,
            triples=triples,
            resolved=resolved,
            metadata=metadata,
            allowed_numbers=allowed_numbers,
            source_texts=source_texts,
            max_pair_sim=max_pair_sim,
            llm=llm,
            method_prefix="llm_conflict",
        )
        _apply_post_resolution_validation(
            seg_id=triple.segment_id,
            seg=seg,
            resolution_ctx=resolution_ctx,
            triples=triples,
            resolved=resolved,
            metadata=metadata,
            source_texts=source_texts,
            max_pair_sim=max_pair_sim,
            llm=llm,
            method_prefix="llm_conflict",
        )

    # GAP segments are handled individually.
    for triple in gap_triples:
        seg = _triple_to_segment_dict(triple, status="gap")
        resolution_ctx = {
            "context_id": f"micro_{triple.segment_id}",
            "segments": [seg],
            "context_before": [],
            "context_after": [],
        }
        gap_text = (seg.get("text") or "").strip()
        gap_source = seg.get("gap_source", "unknown")
        source_texts = {gap_source: gap_text} if gap_text else {}
        allowed_numbers = _segment_allowed_numeric_tokens(seg)

        idx = triple_index.get(triple.segment_id, 0)
        context_before, context_after = _build_flanking_context(triples, idx)
        payload = {
            "segment_id": triple.segment_id,
            "block_type": seg.get("block_type", "paragraph"),
            "context_before": context_before,
            "context_after": context_after,
            "grobid": seg.get("grobid", ""),
            "docling": seg.get("docling", ""),
            "marker": seg.get("marker", ""),
        }

        try:
            gap_resolved = llm.resolve_conflicts([payload])
            resolved_text = gap_resolved.get(triple.segment_id, "")
        except Exception as exc:
            logger.warning("GAP resolution failed for %s: %s", triple.segment_id, exc)
            resolved_text = gap_text

        resolved[triple.segment_id] = resolved_text
        metadata[triple.segment_id] = {
            "method": "llm_gap",
            "chosen_source": "llm" if resolved_text != gap_text else gap_source,
            "confidence": 0.0,
            "sources_agreeing": sorted(source_texts.keys()),
            "max_pair_similarity": 0.0,
            "context_id": resolution_ctx["context_id"],
        }
        _apply_numeric_integrity_guard(
            seg_id=triple.segment_id,
            seg=seg,
            resolution_ctx=resolution_ctx,
            triples=triples,
            resolved=resolved,
            metadata=metadata,
            allowed_numbers=allowed_numbers,
            source_texts=source_texts,
            max_pair_sim=0.0,
            llm=llm,
            method_prefix="llm_gap",
        )
        _apply_post_resolution_validation(
            seg_id=triple.segment_id,
            seg=seg,
            resolution_ctx=resolution_ctx,
            triples=triples,
            resolved=resolved,
            metadata=metadata,
            source_texts=source_texts,
            max_pair_sim=0.0,
            llm=llm,
            method_prefix="llm_gap",
        )

    return resolved, metadata
