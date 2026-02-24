"""Consensus pipeline orchestrator and backwards-compatible API surface.

Implementation details now live in smaller focused modules:
- `consensus_models`: data classes and enum-like status constants
- `consensus_pipeline_steps`: compatibility re-export layer for split step modules
- `consensus_resolution`: layered conflict and rescue resolution
- `consensus_reporting`: metrics and audit entry builders
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import TYPE_CHECKING

from app.services import consensus_models as _consensus_models
from app.services import consensus_pipeline_steps as _consensus_pipeline_steps
from app.services import consensus_reporting as _consensus_reporting
from app.services import consensus_resolution as _consensus_resolution
from app.services.degradation_metrics import build_degradation_metrics

if TYPE_CHECKING:
    from app.services.llm_service import LLM


def _reexport_module_symbols(module) -> None:
    """Expose internal module symbols to preserve legacy import paths."""
    for name in dir(module):
        if name.startswith("__"):
            continue
        globals().setdefault(name, getattr(module, name))


_reexport_module_symbols(_consensus_models)
_reexport_module_symbols(_consensus_pipeline_steps)
_reexport_module_symbols(_consensus_resolution)
_reexport_module_symbols(_consensus_reporting)

logger = logging.getLogger(__name__)


def _resolve_conflicts_micro(*args, **kwargs):
    """Compatibility wrapper so patched rescue hooks in this module still apply."""
    original_rescue = _consensus_resolution._rescue_segment
    _consensus_resolution._rescue_segment = _rescue_segment
    try:
        return _consensus_resolution._resolve_conflicts_micro(*args, **kwargs)
    finally:
        _consensus_resolution._rescue_segment = original_rescue


def merge_with_consensus(
    grobid_md: str,
    docling_md: str,
    marker_md: str,
    llm: "LLM",
) -> tuple[str | None, dict, list]:
    """
    Attempt selective LLM merge. Returns (merged_markdown, metrics, audit_entries).
    Returns (None, metrics, audit_entries) if the pipeline fails (missing extractors,
    alignment too low, or LLM error). Audit entries may be non-empty on failure
    if some pipeline stages ran before the failure point.
    """
    from config import Config

    def _is_mock_value(value) -> bool:
        return value.__class__.__module__.startswith("unittest.mock")

    def _cfg_float(value, default: float) -> float:
        if value is None or _is_mock_value(value):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _cfg_int(value, default: int) -> int:
        if value is None or _is_mock_value(value):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _cfg_bool(value, default: bool) -> bool:
        if value is None or _is_mock_value(value):
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() == "true"

    # Validate all 3 inputs
    if not grobid_md or not docling_md or not marker_md:
        metrics = compute_metrics([], 0.0, True, "missing_extractor")
        return None, metrics, []

    pipeline_start = time.monotonic()

    # Step 1: Source-level normalization + parse
    step_start = time.monotonic()
    logger.info("Consensus pipeline: parsing markdown outputs...")
    grobid_normalized = normalize_extractor_output(grobid_md)
    docling_normalized = normalize_extractor_output(docling_md)
    marker_normalized = normalize_extractor_output(marker_md)

    grobid_blocks = parse_markdown(grobid_normalized, "grobid")
    docling_blocks = parse_markdown(docling_normalized, "docling")
    marker_blocks = parse_markdown(marker_normalized, "marker")

    blocks_by_source = {
        "grobid": grobid_blocks,
        "docling": docling_blocks,
        "marker": marker_blocks,
    }

    logger.info(
        "Parsed blocks: grobid=%d, docling=%d, marker=%d (%.1fs)",
        len(grobid_blocks), len(docling_blocks), len(marker_blocks),
        time.monotonic() - step_start,
    )

    # Exclude extractors with dramatically fewer blocks to avoid poisoning
    # consensus alignment in sparse-output failure modes.
    block_counts = {src: len(blocks) for src, blocks in blocks_by_source.items()}
    max_count = max(block_counts.values()) if block_counts else 0
    disparity_threshold = 0.30  # extractor must have >=30% of max block count

    if max_count > 0:
        sparse_sources = [
            src for src, count in block_counts.items()
            if 0 < count < (max_count * disparity_threshold)
        ]
        for src in sparse_sources:
            logger.warning(
                "Consensus pipeline: excluding %s from alignment (only %d blocks vs max %d, "
                "below %.0f%% threshold)",
                src,
                block_counts[src],
                max_count,
                disparity_threshold * 100,
            )
            del blocks_by_source[src]

    # Step 2: Align
    step_start = time.monotonic()
    logger.info("Consensus pipeline: aligning blocks...")
    triples, alignment_confidence = align_blocks(
        blocks_by_source,
        llm=llm,
        anchor_partitioning_enabled=_cfg_bool(
            getattr(Config, "CONSENSUS_ALIGNMENT_ANCHOR_PARTITIONING_ENABLED", True), True,
        ),
        anchor_min_score=_cfg_float(
            getattr(Config, "CONSENSUS_ALIGNMENT_ANCHOR_MIN_SCORE", 0.72), 0.72,
        ),
        anchor_include_structural_secondary=_cfg_bool(
            getattr(Config, "CONSENSUS_ALIGNMENT_ANCHOR_INCLUDE_STRUCTURAL", True), True,
        ),
        anchor_max_heading_level=int(
            getattr(Config, "CONSENSUS_ALIGNMENT_ANCHOR_MAX_HEADING_LEVEL", 2),
        ),
        ambiguity_delta=_cfg_float(
            getattr(Config, "CONSENSUS_ALIGNMENT_AMBIGUITY_DELTA", 0.03), 0.03,
        ),
        semantic_rerank_enabled=_cfg_bool(
            getattr(Config, "CONSENSUS_ALIGNMENT_SEMANTIC_RERANK_ENABLED", True), True,
        ),
        semantic_margin=_cfg_float(
            getattr(Config, "CONSENSUS_ALIGNMENT_SEMANTIC_MARGIN", 0.02), 0.02,
        ),
        llm_tiebreak_enabled=_cfg_bool(
            getattr(Config, "CONSENSUS_ALIGNMENT_LLM_TIEBREAK_ENABLED", False), False,
        ),
    )
    logger.info(
        "Aligned %d triples, confidence=%.3f (%.1fs)",
        len(triples), alignment_confidence, time.monotonic() - step_start,
    )

    # Step 3: Classify
    logger.info("Consensus pipeline: classifying triples...")
    classify_triples(
        triples,
        near_threshold=Config.CONSENSUS_NEAR_THRESHOLD,
        levenshtein_threshold=Config.CONSENSUS_LEVENSHTEIN_THRESHOLD,
        always_escalate_tables=Config.CONSENSUS_ALWAYS_ESCALATE_TABLES,
        strict_numeric_near=getattr(Config, "CONSENSUS_STRICT_NUMERIC_NEAR", True),
    )

    # Step 3b: remove near-duplicate GAP blocks (local window)
    gap_dups_removed = dedup_gap_triples(triples)
    if gap_dups_removed > 0:
        logger.info("Consensus pipeline: removed %d local GAP duplicates", gap_dups_removed)

    classifications = {}
    classifications_by_type = {}
    for t in triples:
        classifications[t.classification] = classifications.get(t.classification, 0) + 1
        present = _get_present_blocks(t)
        btype = present[0].block_type if present else "unknown"
        key = (t.classification, btype)
        classifications_by_type[key] = classifications_by_type.get(key, 0) + 1
    logger.info("Classifications: %s", classifications)
    for (cls, btype), count in sorted(classifications_by_type.items()):
        if cls != AGREE_EXACT:
            logger.info("  %s / %s: %d", cls, btype, count)

    base_conflict_threshold = _cfg_float(
        getattr(Config, "CONSENSUS_CONFLICT_RATIO_FALLBACK", 0.4), 0.4,
    )
    guard_telemetry = _compute_conflict_telemetry(
        triples,
        conflict_ratio_threshold=base_conflict_threshold,
        localized_conflict_span_max=_cfg_float(
            getattr(Config, "CONSENSUS_LOCALIZED_CONFLICT_SPAN_MAX", 0.35), 0.35,
        ),
        localized_conflict_relief=_cfg_float(
            getattr(Config, "CONSENSUS_LOCALIZED_CONFLICT_RELIEF", 0.15), 0.15,
        ),
        localized_conflict_max_blocks=_cfg_int(
            getattr(Config, "CONSENSUS_LOCALIZED_CONFLICT_MAX_BLOCKS", 25), 25,
        ),
    )

    # Step 4: Guard checks — alignment confidence hard-fail + telemetry logging
    alignment_confidence_min = _cfg_float(
        getattr(Config, "CONSENSUS_ALIGNMENT_CONFIDENCE_MIN", 0.5), 0.5,
    )

    if not triples:
        logger.info("Consensus pipeline: no blocks after alignment — failing")
        metrics = compute_metrics([], 0.0, True, "no_blocks", guard_telemetry=guard_telemetry)
        return None, metrics, []

    if alignment_confidence < alignment_confidence_min:
        logger.info(
            "Consensus pipeline: alignment confidence %.3f < %.3f — failing",
            alignment_confidence, alignment_confidence_min,
        )
        audit_entries = _build_audit_entries(
            triples, {},
            near_threshold=Config.CONSENSUS_NEAR_THRESHOLD,
            levenshtein_threshold=Config.CONSENSUS_LEVENSHTEIN_THRESHOLD,
        )
        metrics = compute_metrics(
            triples, alignment_confidence, True, "alignment_too_low",
            guard_telemetry=guard_telemetry,
        )
        return None, metrics, audit_entries

    # Log guard telemetry for monitoring (no longer used for branching decisions)
    logger.info(
        "Consensus pipeline guard telemetry: conflict_ratio=%.3f, "
        "conflict_ratio_textual=%.3f, conflict_ratio_structured=%.3f, "
        "conflicts_localized=%s",
        guard_telemetry.get("conflict_ratio", 0.0),
        guard_telemetry.get("conflict_ratio_textual", 0.0),
        guard_telemetry.get("conflict_ratio_structured", 0.0),
        guard_telemetry.get("conflicts_localized", False),
    )

    # Step 5: Extract micro-conflicts and resolve via token DP + LLM
    # The micro-conflict pipeline now handles CONFLICT, AGREE_NEAR, and
    # multi-source GAP triples. Single-source GAPs still go directly to LLM.
    conflict_triples = [t for t in triples if t.classification == CONFLICT]
    near_triples = [t for t in triples if t.classification == AGREE_NEAR]
    gap_triples = [t for t in triples if t.classification == GAP]
    resolved_conflicts: dict[str, str] = {}
    resolution_metadata: dict[str, dict] = {}
    micro_results: dict[str, MicroConflictResult] = {}

    if conflict_triples or gap_triples or near_triples:
        step_start = time.monotonic()
        logger.info(
            "Consensus pipeline: resolving %d conflict(s), %d near-agree, %d GAP segment(s) "
            "via micro-conflict...",
            len(conflict_triples), len(near_triples), len(gap_triples),
        )

        try:
            # Extract per-segment micro-conflicts for eligible triples.
            micro_results = extract_micro_conflicts(triples)
            logger.info(
                "Consensus pipeline: extracted micro-conflicts for %d segment(s), total spans=%d",
                len(micro_results),
                sum(len(res.conflicts) for res in micro_results.values()),
            )

            resolved_conflicts, resolution_metadata = _resolve_conflicts_micro(
                triples,
                micro_results,
                llm,
                medium_similarity_threshold=_cfg_float(
                    getattr(
                        Config,
                        "CONSENSUS_LAYERED_MEDIUM_SIM_THRESHOLD",
                        _LAYERED_MEDIUM_SIM_THRESHOLD,
                    ),
                    _LAYERED_MEDIUM_SIM_THRESHOLD,
                ),
                median_source_max_micro_conflicts=int(
                    getattr(Config, "CONSENSUS_MEDIAN_SOURCE_MAX_MICRO_CONFLICTS", 20),
                ),
            )
        except Exception as e:
            logger.warning("Consensus pipeline: micro-conflict resolution failed: %s", e, exc_info=True)
            audit_entries = _build_audit_entries(
                triples, {}, {},
                near_threshold=Config.CONSENSUS_NEAR_THRESHOLD,
                levenshtein_threshold=Config.CONSENSUS_LEVENSHTEIN_THRESHOLD,
            )
            metrics = compute_metrics(
                triples, alignment_confidence, True, "llm_error",
                guard_telemetry=guard_telemetry,
                micro_results=micro_results,
            )
            return None, metrics, audit_entries

    # Log conflict resolution summary
    if resolution_metadata:
        resolve_duration = time.monotonic() - step_start
        method_counts = Counter(
            m.get("method", "unknown") for m in resolution_metadata.values()
        )
        confidences = [
            float(m.get("confidence", 0.0)) for m in resolution_metadata.values()
        ]
        logger.info(
            "Conflict resolution complete in %.1fs: %s, confidence min=%.3f max=%.3f mean=%.3f",
            resolve_duration,
            dict(method_counts),
            min(confidences) if confidences else 0,
            max(confidences) if confidences else 0,
            (sum(confidences) / len(confidences)) if confidences else 0,
        )
        # Per-conflict detail at DEBUG level
        for seg_id, meta in resolution_metadata.items():
            logger.debug(
                "  %s: method=%s, confidence=%.3f, pair_sim=%.3f, sources=%s",
                seg_id, meta["method"], meta["confidence"],
                meta.get("max_pair_similarity", 0),
                meta.get("sources_agreeing", []),
            )

    # Build audit entries (after classification, dedup, and conflict resolution)
    audit_entries = _build_audit_entries(
        triples, resolved_conflicts, resolution_metadata,
        near_threshold=Config.CONSENSUS_NEAR_THRESHOLD,
        levenshtein_threshold=Config.CONSENSUS_LEVENSHTEIN_THRESHOLD,
    )

    # Step 6: Assemble
    logger.info("Consensus pipeline: assembling final document...")
    merged_md = assemble(triples, resolved_conflicts, resolution_metadata)

    # Step 6a2: Post-assembly structural heading safety net
    merged_md, abstract_injected = ensure_abstract_heading(merged_md, triples)
    if abstract_injected:
        logger.info("Consensus pipeline: injected missing ## Abstract heading (safety net)")

    # Step 6b: Post-assembly global paragraph dedup (safety net)
    merged_md, assembled_dups_removed = dedup_assembled_paragraphs(merged_md)
    if assembled_dups_removed > 0:
        logger.info(
            "Consensus pipeline: removed %d post-assembly duplicate paragraphs",
            assembled_dups_removed,
        )

    # Step 6c: Header hierarchy resolution (optional)
    if Config.CONSENSUS_HIERARCHY_ENABLED:
        try:
            merged_md = resolve_header_hierarchy(merged_md, llm)
        except Exception as e:
            logger.warning("Header hierarchy resolution failed: %s — using original headers", e)

    # Step 6d: Post-hierarchy dedup — heading demotion can expose duplicates
    #          that were invisible to the first dedup pass (they were headings).
    merged_md, post_hierarchy_dups = dedup_assembled_paragraphs(merged_md)
    assembled_dups_removed += post_hierarchy_dups
    if post_hierarchy_dups > 0:
        logger.info(
            "Consensus pipeline: removed %d post-hierarchy duplicate paragraphs",
            post_hierarchy_dups,
        )

    # Finalized heading hierarchy (post optional hierarchy resolution).
    heading_hierarchy = extract_heading_hierarchy(merged_md)

    # Step 7: QA gates (global duplicate detection)
    qa_results = run_qa_gates(merged_md)

    metrics = compute_metrics(
        triples,
        alignment_confidence,
        False,
        None,
        guard_telemetry=guard_telemetry,
        micro_results=micro_results,
    )
    if resolution_metadata:
        method_counts = Counter(
            m.get("method", "unknown") for m in resolution_metadata.values()
        )
        metrics["resolution_methods"] = dict(method_counts)
        confidences = [
            float(m.get("confidence", 0.0))
            for m in resolution_metadata.values()
            if isinstance(m, dict)
        ]
        if confidences:
            metrics["resolution_confidence_mean"] = round(
                float(sum(confidences) / len(confidences)), 4,
            )
    metrics["gap_dedup_removed"] = gap_dups_removed
    metrics["assembled_dedup_removed"] = assembled_dups_removed
    metrics["qa"] = qa_results

    # Compute degradation metrics from resolution metadata
    degradation = build_degradation_metrics(
        triples=triples,
        resolution_metadata=resolution_metadata,
        audit_entries=audit_entries,
        total_blocks=metrics["total_blocks"],
        heading_hierarchy=heading_hierarchy,
        zone_resolution_tokens=llm.usage.tokens_for_types("micro_conflict"),
        rescue_call_tokens=llm.usage.tokens_for_types("rescue")
    )
    metrics["degradation_metrics"] = degradation

    degraded_count = degradation["degraded_segments"]["count"]
    if degraded_count > 0:
        logger.warning(
            "Micro-conflict resolution: %d segment(s) required numeric integrity intervention "
            "(quality_score=%.3f, grade=%s)",
            degraded_count,
            degradation["quality_score"],
            degradation["quality_grade"],
        )

    # Step 8: Log surviving global duplicates (monitoring only)
    surviving_dupes = qa_results.get("global_duplicate_count", 0)
    if surviving_dupes > 0:
        logger.warning(
            "Consensus pipeline: %d global duplicates survived both dedup layers "
            "— continuing with micro-conflict-resolved output",
            surviving_dupes,
        )

    pipeline_duration = time.monotonic() - pipeline_start
    logger.info(
        "Consensus pipeline complete in %.1fs: %d blocks, %d conflicts resolved, "
        "%d GAP dedup, %d post-assembly dedup",
        pipeline_duration,
        metrics["total_blocks"],
        metrics["conflict"],
        gap_dups_removed,
        assembled_dups_removed,
    )

    return merged_md, metrics, audit_entries
