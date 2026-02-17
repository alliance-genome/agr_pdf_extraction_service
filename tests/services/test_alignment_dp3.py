"""Unit tests for the 3-way global DP alignment path."""

import numpy as np

from app.services.alignment.arbitration import (
    ArbitrationConfig,
    ArbitrationContext,
    choose_end_mode,
    choose_repair_candidate,
)
from app.services.alignment.dp3 import MODE_COUNT, TRANSITIONS, START_MODE, align_three_way_global
from app.services.alignment.partitioning import AnchorPartitionConfig, build_alignment_windows
from app.services.alignment.repair import repair_split_merge_columns
from app.services.alignment.scoring import ScoreConfig, pair_score
from app.services.alignment.traceback import AlignmentColumn, traceback_columns
from app.services.alignment.triples import build_aligned_triples
from app.services.consensus_models import Block
from app.services.consensus_parsing_alignment import align_blocks


def _block(source: str, idx: int, text: str, block_type: str = "paragraph") -> Block:
    return Block(
        block_id=f"{source}_{idx}",
        block_type=block_type,
        raw_text=text,
        normalized_text=text.lower(),
        heading_level=1 if block_type == "heading" else None,
        order_index=idx,
        source=source,
        source_md=text,
    )


def test_transition_contract_has_all_7_moves():
    assert MODE_COUNT == 7
    assert len(TRANSITIONS) == 7
    assert START_MODE == 7
    assert {t.name for t in TRANSITIONS} == {"111", "110", "101", "011", "100", "010", "001"}


def test_pair_score_penalizes_type_and_numeric_mismatch():
    heading = _block("grobid", 0, "Results", block_type="heading")
    paragraph = _block("docling", 0, "Results", block_type="paragraph")
    equal_heading = _block("marker", 0, "Results", block_type="heading")
    numeric_a = _block("grobid", 1, "accuracy 95.2")
    numeric_b = _block("docling", 1, "accuracy 96.2")

    mismatch = pair_score(heading, paragraph).total
    matched = pair_score(heading, equal_heading).total
    numeric_mismatch = pair_score(numeric_a, numeric_b).total
    numeric_match = pair_score(numeric_a, _block("marker", 1, "accuracy 95.2")).total

    assert matched > mismatch
    assert numeric_match > numeric_mismatch


def test_traceback_is_monotonic_and_consumes_all_blocks():
    grobid = [_block("grobid", i, f"g{i}") for i in range(3)]
    docling = [_block("docling", i, f"d{i}") for i in range(2)]
    marker = [_block("marker", i, f"m{i}") for i in range(4)]

    result = align_three_way_global(grobid, docling, marker)
    columns = traceback_columns(result)
    triples, _ = build_aligned_triples(columns)

    seen = {"grobid": set(), "docling": set(), "marker": set()}
    for triple in triples:
        for source in seen:
            block = getattr(triple, f"{source}_block")
            if block is not None:
                seen[source].add(block.block_id)

    assert len(seen["grobid"]) == len(grobid)
    assert len(seen["docling"]) == len(docling)
    assert len(seen["marker"]) == len(marker)


def test_align_blocks_global_dp_high_confidence_when_identical():
    grobid = [_block("grobid", i, txt) for i, txt in enumerate(["# Intro", "One", "Two"])]
    docling = [_block("docling", i, txt) for i, txt in enumerate(["# Intro", "One", "Two"])]
    marker = [_block("marker", i, txt) for i, txt in enumerate(["# Intro", "One", "Two"])]

    triples, confidence = align_blocks({
        "grobid": grobid,
        "docling": docling,
        "marker": marker,
    })

    assert len(triples) == 3
    assert confidence >= 0.80


def test_repair_split_merge_match_gap_motif():
    left = AlignmentColumn(
        grobid_block=_block("grobid", 0, "alpha beta gamma"),
        docling_block=_block("docling", 0, "alpha beta"),
        marker_block=_block("marker", 0, "alpha beta gamma"),
    )
    right = AlignmentColumn(
        grobid_block=None,
        docling_block=_block("docling", 1, "gamma"),
        marker_block=None,
    )

    repaired = repair_split_merge_columns([left, right], acceptance_threshold=0.55, min_gain=0.01)

    assert len(repaired) == 1
    merged_docling = repaired[0].docling_block
    assert merged_docling is not None
    assert "alpha beta" in merged_docling.raw_text
    assert "gamma" in merged_docling.raw_text
    assert repaired[0].metadata.get("repairs")


def test_repair_split_merge_gap_match_motif():
    left = AlignmentColumn(
        grobid_block=None,
        docling_block=_block("docling", 0, "intro"),
        marker_block=_block("marker", 0, "intro section"),
    )
    right = AlignmentColumn(
        grobid_block=_block("grobid", 1, "intro section"),
        docling_block=_block("docling", 1, "section"),
        marker_block=None,
    )

    repaired = repair_split_merge_columns([left, right], acceptance_threshold=0.55, min_gain=0.01)

    repaired_column = next(col for col in repaired if col.grobid_block is not None)
    merged_docling = repaired_column.docling_block
    assert merged_docling is not None
    assert merged_docling.raw_text == "intro\nsection"
    assert repaired_column.metadata.get("repairs")


def test_anchor_partitioning_builds_multiple_windows_on_headings():
    grobid = [
        _block("grobid", 0, "Introduction", block_type="heading"),
        _block("grobid", 1, "g intro text"),
        _block("grobid", 2, "Methods", block_type="heading"),
        _block("grobid", 3, "g methods text"),
    ]
    docling = [
        _block("docling", 0, "Introduction", block_type="heading"),
        _block("docling", 1, "d intro text"),
        _block("docling", 2, "Methods", block_type="heading"),
        _block("docling", 3, "d methods text"),
    ]
    marker = [
        _block("marker", 0, "Introduction", block_type="heading"),
        _block("marker", 1, "m intro text"),
        _block("marker", 2, "Methods", block_type="heading"),
        _block("marker", 3, "m methods text"),
    ]

    windows = build_alignment_windows(
        grobid,
        docling,
        marker,
        score_config=ScoreConfig(),
        partition_config=AnchorPartitionConfig(enabled=True, min_anchor_score=0.60),
    )

    assert len(windows) >= 3
    assert any(
        any(block.block_type == "heading" for block in source_blocks)
        for window in windows
        for source_blocks in window.values()
    )


def test_choose_end_mode_uses_llm_tiebreak_when_close():
    grobid = [_block("grobid", 0, "a"), _block("grobid", 1, "b")]
    docling = [_block("docling", 0, "a"), _block("docling", 1, "b")]
    marker = [_block("marker", 0, "a"), _block("marker", 1, "b")]
    result = align_three_way_global(grobid, docling, marker)

    n_g, n_d, n_m = len(grobid), len(docling), len(marker)
    finite_modes = [
        idx for idx in range(MODE_COUNT)
        if np.isfinite(result.scores[n_g, n_d, n_m, idx])
    ]
    assert len(finite_modes) >= 2
    best_mode, second_mode = finite_modes[0], finite_modes[1]
    for mode in finite_modes:
        result.scores[n_g, n_d, n_m, mode] = -10.0
    result.scores[n_g, n_d, n_m, best_mode] = 1.0
    result.scores[n_g, n_d, n_m, second_mode] = 0.99
    result.best_mode = best_mode

    class _ChoiceLLM:
        @staticmethod
        def resolve_alignment_tiebreak(_payload):
            return {"choice": "candidate_b"}

    chosen = choose_end_mode(
        result,
        config=ArbitrationConfig(
            ambiguity_delta=0.05,
            semantic_rerank_enabled=False,
            llm_tiebreak_enabled=True,
        ),
        context=ArbitrationContext(),
        llm=_ChoiceLLM(),
    )
    assert chosen == second_mode


def test_partial_anchor_split_point_includes_missing_source_blocks():
    """When an anchor exists in 2/3 sources, the missing source's pre-anchor blocks
    should still be included in the window via the best-split-point heuristic."""
    grobid = [
        _block("grobid", 0, "Introduction", block_type="heading"),
        _block("grobid", 1, "g intro text"),
        _block("grobid", 2, "Methods", block_type="heading"),
        _block("grobid", 3, "g methods text"),
    ]
    docling = [
        _block("docling", 0, "Introduction", block_type="heading"),
        _block("docling", 1, "d intro text"),
        _block("docling", 2, "Methods", block_type="heading"),
        _block("docling", 3, "d methods text"),
    ]
    # Marker is missing the "Methods" heading — only has intro blocks + methods text
    marker = [
        _block("marker", 0, "Introduction", block_type="heading"),
        _block("marker", 1, "m intro text"),
        _block("marker", 2, "m methods text"),
    ]

    windows = build_alignment_windows(
        grobid,
        docling,
        marker,
        score_config=ScoreConfig(),
        partition_config=AnchorPartitionConfig(enabled=True, min_anchor_score=0.60),
    )

    # Collect all marker blocks across all windows
    all_marker_blocks = []
    for window in windows:
        all_marker_blocks.extend(window.get("marker", []))

    # All marker blocks should appear in some window (not deferred/lost)
    marker_ids = {b.block_id for b in all_marker_blocks}
    assert marker_ids == {b.block_id for b in marker}, (
        f"Expected all marker block IDs, got {marker_ids}"
    )


def test_choose_repair_candidate_semantic_rerank():
    decision = choose_repair_candidate(
        after_score=0.70,
        before_score=0.69,
        merged_text="alpha beta gamma",
        best_fragment_text="alpha",
        anchor_text="alpha beta gamma",
        config=ArbitrationConfig(
            repair_ambiguity_delta=0.05,
            repair_semantic_margin=0.01,
            semantic_rerank_enabled=True,
        ),
        context=ArbitrationContext(),
        llm=None,
    )
    assert decision is True
