"""Ambiguity arbitration for close-score alignment decisions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from app.services.alignment.dp3 import MODE_COUNT, DPResult
from app.services.alignment.scoring import transition_score
from app.services.alignment.traceback import AlignmentColumn, traceback_columns

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArbitrationConfig:
    """Controls for close-score arbitration."""

    ambiguity_delta: float = 0.03
    semantic_rerank_enabled: bool = True
    semantic_margin: float = 0.02
    llm_tiebreak_enabled: bool = True
    summary_columns: int = 3
    repair_ambiguity_delta: float = 0.03
    repair_semantic_margin: float = 0.02


@dataclass
class ArbitrationContext:
    """Mutable budget/telemetry state for one alignment run."""

    llm_tiebreak_calls: int = 0

    def record_llm_call(self) -> None:
        self.llm_tiebreak_calls += 1


def _column_semantic_score(column: AlignmentColumn) -> float:
    detail = transition_score(column.grobid_block, column.docling_block, column.marker_block)
    lexical = [pair.lexical for pair in detail.pair_scores.values()]
    if not lexical:
        return 0.0
    return sum(lexical) / len(lexical)


def _candidate_semantic_score(columns: list[AlignmentColumn]) -> float:
    values = [_column_semantic_score(col) for col in columns]
    values = [v for v in values if v > 0]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _pairwise_semantic(text_a: str, text_b: str) -> float:
    """Secondary semantic-lite similarity for repair arbitration."""
    token_sort = fuzz.token_sort_ratio(text_a, text_b) / 100.0
    token_set = fuzz.token_set_ratio(text_a, text_b) / 100.0
    raw = fuzz.ratio(text_a, text_b) / 100.0
    return (0.40 * token_set) + (0.35 * token_sort) + (0.25 * raw)


def _llm_callable(llm: Any) -> bool:
    return callable(getattr(llm, "resolve_alignment_tiebreak", None))


def _coerce_llm_choice(raw_choice: Any) -> str | None:
    if isinstance(raw_choice, str):
        choice = raw_choice.strip().lower()
        if choice in {"candidate_a", "a", "keep"}:
            return "candidate_a"
        if choice in {"candidate_b", "b", "drop"}:
            return "candidate_b"
        return None

    if isinstance(raw_choice, dict):
        for key in ("choice", "chosen_candidate", "candidate"):
            value = raw_choice.get(key)
            parsed = _coerce_llm_choice(value)
            if parsed:
                return parsed
    return None


def choose_end_mode(
    result: DPResult,
    *,
    config: ArbitrationConfig,
    context: ArbitrationContext,
    llm: Any = None,
) -> int:
    """Choose terminal traceback mode, with arbitration for close scores."""
    n_g = len(result.grobid_blocks)
    n_d = len(result.docling_blocks)
    n_m = len(result.marker_blocks)
    terminal_scores = result.scores[n_g, n_d, n_m, :MODE_COUNT]
    finite = [(idx, float(score)) for idx, score in enumerate(terminal_scores) if np.isfinite(score)]
    if not finite:
        return result.best_mode

    ranked = sorted(finite, key=lambda item: (-item[1], item[0]))
    best_mode, best_score = ranked[0]
    if len(ranked) < 2:
        return best_mode

    second_mode, second_score = ranked[1]
    if (best_score - second_score) > config.ambiguity_delta:
        return best_mode

    try:
        cols_best = traceback_columns(result, end_mode=best_mode)
        cols_second = traceback_columns(result, end_mode=second_mode)
    except Exception:
        logger.exception("Alignment arbitration traceback failed; using deterministic best mode")
        return best_mode

    if config.semantic_rerank_enabled:
        sem_best = _candidate_semantic_score(cols_best)
        sem_second = _candidate_semantic_score(cols_second)
        if abs(sem_best - sem_second) >= config.semantic_margin:
            return best_mode if sem_best > sem_second else second_mode

    if config.llm_tiebreak_enabled and _llm_callable(llm):
        payload = {
            "task": "choose_best_alignment_candidate",
            "candidates": [
                {
                    "candidate_id": "candidate_a",
                    "mode": best_mode,
                    "dp_score": best_score,
                    "sample_columns": _summarize_columns(cols_best, max_cols=config.summary_columns),
                },
                {
                    "candidate_id": "candidate_b",
                    "mode": second_mode,
                    "dp_score": second_score,
                    "sample_columns": _summarize_columns(cols_second, max_cols=config.summary_columns),
                },
            ],
        }
        try:
            decision = llm.resolve_alignment_tiebreak(payload)
            context.record_llm_call()
            choice = _coerce_llm_choice(decision)
            if choice == "candidate_b":
                return second_mode
            if choice == "candidate_a":
                return best_mode
        except Exception:
            logger.exception("LLM alignment tiebreak failed; using deterministic best mode")

    return best_mode


def choose_repair_candidate(
    *,
    after_score: float,
    before_score: float,
    merged_text: str,
    best_fragment_text: str,
    anchor_text: str,
    config: ArbitrationConfig,
    context: ArbitrationContext,
    llm: Any = None,
) -> bool:
    """Return True to accept repair candidate, False to keep deterministic gap split."""
    if after_score > before_score + config.repair_ambiguity_delta:
        return True
    if before_score > after_score + config.repair_ambiguity_delta:
        return False

    if config.semantic_rerank_enabled:
        sem_after = _pairwise_semantic(merged_text, anchor_text)
        sem_before = _pairwise_semantic(best_fragment_text, anchor_text)
        if sem_after > sem_before + config.repair_semantic_margin:
            return True
        if sem_before > sem_after + config.repair_semantic_margin:
            return False

    if config.llm_tiebreak_enabled and _llm_callable(llm):
        payload = {
            "task": "choose_best_alignment_candidate",
            "candidates": [
                {
                    "candidate_id": "candidate_a",
                    "description": "merge split fragments",
                    "score": after_score,
                    "text": merged_text[:1200],
                    "anchor_text": anchor_text[:1200],
                },
                {
                    "candidate_id": "candidate_b",
                    "description": "keep best single fragment",
                    "score": before_score,
                    "text": best_fragment_text[:1200],
                    "anchor_text": anchor_text[:1200],
                },
            ],
        }
        try:
            decision = llm.resolve_alignment_tiebreak(payload)
            context.record_llm_call()
            choice = _coerce_llm_choice(decision)
            if choice == "candidate_a":
                return True
            if choice == "candidate_b":
                return False
        except Exception:
            logger.exception("LLM repair tiebreak failed; falling back to deterministic choice")

    return after_score >= before_score


def _summarize_columns(columns: list[AlignmentColumn], *, max_cols: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for col in columns[:max_cols]:
        out.append({
            "grobid": (col.grobid_block.raw_text[:200] if col.grobid_block else ""),
            "docling": (col.docling_block.raw_text[:200] if col.docling_block else ""),
            "marker": (col.marker_block.raw_text[:200] if col.marker_block else ""),
        })
    return out
