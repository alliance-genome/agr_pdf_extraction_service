"""Build `AlignedTriple` rows from traceback columns."""

from __future__ import annotations

from app.services.alignment.scoring import ScoreConfig, transition_score
from app.services.alignment.traceback import AlignmentColumn
from app.services.consensus_models import AlignedTriple


def _column_confidence(column: AlignmentColumn, *, config: ScoreConfig) -> tuple[float, list[float]]:
    """Return column confidence and pair score list in [0, 1]."""
    detail = transition_score(
        column.grobid_block,
        column.docling_block,
        column.marker_block,
        config=config,
    )
    pair_values = [max(0.0, min(1.0, p.total)) for p in detail.pair_scores.values()]
    if not pair_values:
        return 0.0, []
    return sum(pair_values) / len(pair_values), pair_values


def _assert_monotonic_triples(triples: list[AlignedTriple]) -> None:
    """Monotonic order is guaranteed by DP; this checks wiring mistakes."""
    last = {"grobid": -1, "docling": -1, "marker": -1}
    for triple in triples:
        for source_name in ("grobid", "docling", "marker"):
            block = getattr(triple, f"{source_name}_block")
            if block is None:
                continue
            if block.order_index <= last[source_name]:
                raise ValueError(f"Non-monotonic triple order for {source_name}")
            last[source_name] = block.order_index


def build_aligned_triples(
    columns: list[AlignmentColumn],
    *,
    config: ScoreConfig | None = None,
) -> tuple[list[AlignedTriple], float]:
    """Convert columns to `AlignedTriple` objects and global confidence."""
    cfg = config or ScoreConfig()
    triples: list[AlignedTriple] = []
    all_pair_scores: list[float] = []

    for seg_idx, column in enumerate(columns):
        confidence, pair_scores = _column_confidence(column, config=cfg)
        all_pair_scores.extend(pair_scores)
        triple = AlignedTriple(
            segment_id=f"seg_{seg_idx:03d}",
            grobid_block=column.grobid_block,
            docling_block=column.docling_block,
            marker_block=column.marker_block,
            confidence=confidence,
        )
        triples.append(triple)

    _assert_monotonic_triples(triples)
    alignment_confidence = sum(all_pair_scores) / len(all_pair_scores) if all_pair_scores else 0.0
    return triples, alignment_confidence
