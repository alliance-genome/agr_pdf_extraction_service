"""Full 3-way global DP alignment core."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numba
import numpy as np

from app.services.alignment.scoring import ScoreConfig, pair_score, transition_score
from app.services.consensus_models import Block

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DPTransition:
    """One legal state transition for 3-way alignment."""

    name: str
    consume: tuple[int, int, int]


TRANSITIONS: tuple[DPTransition, ...] = (
    DPTransition("111", (1, 1, 1)),
    DPTransition("110", (1, 1, 0)),
    DPTransition("101", (1, 0, 1)),
    DPTransition("011", (0, 1, 1)),
    DPTransition("100", (1, 0, 0)),
    DPTransition("010", (0, 1, 0)),
    DPTransition("001", (0, 0, 1)),
)
MODE_COUNT = len(TRANSITIONS)
START_MODE = MODE_COUNT
_NEG_INF = float("-inf")


@dataclass
class DPResult:
    """Raw DP tensors plus block references for traceback."""

    grobid_blocks: list[Block]
    docling_blocks: list[Block]
    marker_blocks: list[Block]
    scores: np.ndarray
    prev_mode: np.ndarray
    move_score: np.ndarray
    best_mode: int
    best_score: float
    config: ScoreConfig


def _consume_bits(mode: int) -> tuple[int, int, int]:
    if mode == START_MODE:
        # Treat start as "no active gaps" so first gap incurs GAP_OPEN.
        return (1, 1, 1)
    return TRANSITIONS[mode].consume


def affine_gap_penalty(
    prev_mode: int,
    curr_mode: int,
    *,
    config: ScoreConfig,
) -> tuple[float, dict[str, float]]:
    """Affine penalties per source for one transition edge."""
    prev_bits = _consume_bits(prev_mode)
    curr_bits = _consume_bits(curr_mode)
    details = {"grobid": 0.0, "docling": 0.0, "marker": 0.0}
    total = 0.0

    for idx, source_name in enumerate(("grobid", "docling", "marker")):
        if curr_bits[idx] == 1:
            continue
        prev_had_gap = prev_bits[idx] == 0
        penalty = config.gap_extend if prev_had_gap else config.gap_open
        details[source_name] = penalty
        total += penalty

    return total, details


def _prefer_prev_mode(candidate: int, incumbent: int) -> bool:
    """Stable deterministic tie-break for backpointers."""
    if incumbent < 0:
        return True
    return candidate < incumbent


def _precompute_pairwise_scores(
    grobid_blocks: list[Block],
    docling_blocks: list[Block],
    marker_blocks: list[Block],
    config: ScoreConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute all pairwise block scores into 2D numpy arrays.

    Returns (gd_scores, gm_scores, dm_scores) where:
      gd_scores[i, j] = pair_score(grobid[i], docling[j]).total
      gm_scores[i, k] = pair_score(grobid[i], marker[k]).total
      dm_scores[j, k] = pair_score(docling[j], marker[k]).total
    """
    n_g, n_d, n_m = len(grobid_blocks), len(docling_blocks), len(marker_blocks)

    gd_scores = np.empty((n_g, n_d), dtype=np.float64)
    gm_scores = np.empty((n_g, n_m), dtype=np.float64)
    dm_scores = np.empty((n_d, n_m), dtype=np.float64)

    for i in range(n_g):
        for j in range(n_d):
            gd_scores[i, j] = pair_score(grobid_blocks[i], docling_blocks[j], config=config).total
    for i in range(n_g):
        for k in range(n_m):
            gm_scores[i, k] = pair_score(grobid_blocks[i], marker_blocks[k], config=config).total
    for j in range(n_d):
        for k in range(n_m):
            dm_scores[j, k] = pair_score(docling_blocks[j], marker_blocks[k], config=config).total

    return gd_scores, gm_scores, dm_scores


def _precompute_gap_penalties(config: ScoreConfig) -> np.ndarray:
    """Precompute affine gap penalties for all (prev_mode, curr_mode) pairs.

    Returns gap_penalties[prev_mode, curr_mode] as a (MODE_COUNT+1, MODE_COUNT) array.
    """
    gap_penalties = np.zeros((MODE_COUNT + 1, MODE_COUNT), dtype=np.float64)
    for prev_m in range(MODE_COUNT + 1):
        for curr_m in range(MODE_COUNT):
            gap_penalties[prev_m, curr_m], _ = affine_gap_penalty(prev_m, curr_m, config=config)
    return gap_penalties


def align_three_way_global(
    grobid_blocks: list[Block],
    docling_blocks: list[Block],
    marker_blocks: list[Block],
    *,
    config: ScoreConfig | None = None,
) -> DPResult:
    """Compute exact 3-way global alignment with affine gaps."""
    cfg = config or ScoreConfig()
    n_g, n_d, n_m = len(grobid_blocks), len(docling_blocks), len(marker_blocks)

    cells = (n_g + 1) * (n_d + 1) * (n_m + 1)
    mem_est_mb = cells * (MODE_COUNT + 1) * 8 * 3 / (1024 * 1024)  # 3 tensors, 8 bytes each
    total_pairs = n_g * n_d + n_g * n_m + n_d * n_m
    logger.info(
        "3-way DP: dimensions=(%d, %d, %d), cells=%d, est_memory=%.1fMB, precompute_pairs=%d",
        n_g, n_d, n_m, cells, mem_est_mb, total_pairs,
    )

    # --- Precompute pairwise scores and gap penalties ---
    t0 = time.monotonic()
    gd_scores, gm_scores, dm_scores = _precompute_pairwise_scores(
        grobid_blocks, docling_blocks, marker_blocks, cfg,
    )
    gap_penalties = _precompute_gap_penalties(cfg)
    precompute_s = time.monotonic() - t0
    logger.info("3-way DP: precomputed %d pair scores in %.2fs", total_pairs, precompute_s)

    # Precompute which pairs each transition needs.
    # For consume bits (di, dj, dk): pairs are (di&dj → gd), (di&dk → gm), (dj&dk → dm).
    trans_needs_gd = np.array([t.consume[0] & t.consume[1] for t in TRANSITIONS], dtype=bool)
    trans_needs_gm = np.array([t.consume[0] & t.consume[2] for t in TRANSITIONS], dtype=bool)
    trans_needs_dm = np.array([t.consume[1] & t.consume[2] for t in TRANSITIONS], dtype=bool)
    trans_consume = np.array([t.consume for t in TRANSITIONS], dtype=np.intp)

    # --- DP recurrence ---
    t1 = time.monotonic()
    scores = np.full((n_g + 1, n_d + 1, n_m + 1, MODE_COUNT + 1), _NEG_INF, dtype=np.float64)
    prev_mode_arr = np.full((n_g + 1, n_d + 1, n_m + 1, MODE_COUNT), -1, dtype=np.int16)
    move_score_arr = np.full((n_g + 1, n_d + 1, n_m + 1, MODE_COUNT), np.nan, dtype=np.float64)
    scores[0, 0, 0, START_MODE] = 0.0

    for i in range(n_g + 1):
        for j in range(n_d + 1):
            for k in range(n_m + 1):
                if i == 0 and j == 0 and k == 0:
                    continue

                for mode_idx in range(MODE_COUNT):
                    di = trans_consume[mode_idx, 0]
                    dj = trans_consume[mode_idx, 1]
                    dk = trans_consume[mode_idx, 2]
                    if i < di or j < dj or k < dk:
                        continue

                    # Lookup precomputed transition score (sum of relevant pair scores).
                    trans_total = 0.0
                    if trans_needs_gd[mode_idx]:
                        trans_total += gd_scores[i - 1, j - 1]
                    if trans_needs_gm[mode_idx]:
                        trans_total += gm_scores[i - 1, k - 1]
                    if trans_needs_dm[mode_idx]:
                        trans_total += dm_scores[j - 1, k - 1]

                    prev_i, prev_j, prev_k = i - di, j - dj, k - dk

                    best_prev_m = -1
                    best_local = _NEG_INF
                    best_total_score = _NEG_INF

                    for prev_m_idx in range(MODE_COUNT + 1):
                        prev_total = scores[prev_i, prev_j, prev_k, prev_m_idx]
                        if prev_total == _NEG_INF:
                            continue

                        local = trans_total + gap_penalties[prev_m_idx, mode_idx]
                        candidate = prev_total + local

                        if candidate > best_total_score + 1e-12:
                            best_total_score = candidate
                            best_prev_m = prev_m_idx
                            best_local = local
                        elif (
                            abs(candidate - best_total_score) <= 1e-12
                            and (best_prev_m < 0 or prev_m_idx < best_prev_m)
                        ):
                            best_prev_m = prev_m_idx
                            best_local = local

                    if best_prev_m >= 0:
                        scores[i, j, k, mode_idx] = best_total_score
                        prev_mode_arr[i, j, k, mode_idx] = best_prev_m
                        move_score_arr[i, j, k, mode_idx] = best_local

    dp_s = time.monotonic() - t1
    logger.info("3-way DP: recurrence completed in %.2fs", dp_s)

    best_mode = 0
    best_score = _NEG_INF
    for mode_idx in range(MODE_COUNT):
        score = scores[n_g, n_d, n_m, mode_idx]
        if score > best_score + 1e-12:
            best_score = score
            best_mode = mode_idx
        elif (
            np.isfinite(score)
            and np.isfinite(best_score)
            and abs(score - best_score) <= 1e-12
            and mode_idx < best_mode
        ):
            best_mode = mode_idx

    if not np.isfinite(best_score):
        # Edge case: all three streams empty.
        best_mode = START_MODE
        best_score = 0.0

    return DPResult(
        grobid_blocks=grobid_blocks,
        docling_blocks=docling_blocks,
        marker_blocks=marker_blocks,
        scores=scores,
        prev_mode=prev_mode_arr,
        move_score=move_score_arr,
        best_mode=best_mode,
        best_score=best_score,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Token-level 3-way alignment (for micro-conflict extraction)
# ---------------------------------------------------------------------------

# Pre-build transition tables as module-level numpy arrays for Numba.
_T_CONSUME = np.array([t.consume for t in TRANSITIONS], dtype=np.int32)
_T_AB = np.array([t.consume[0] & t.consume[1] for t in TRANSITIONS], dtype=np.bool_)
_T_AC = np.array([t.consume[0] & t.consume[2] for t in TRANSITIONS], dtype=np.bool_)
_T_BC = np.array([t.consume[1] & t.consume[2] for t in TRANSITIONS], dtype=np.bool_)


def _build_token_gap_penalties(
    gap_open: float, gap_extend: float,
) -> np.ndarray:
    """Build gap-penalty matrix for token DP (pure Python, called once)."""
    gap_pen = np.zeros((MODE_COUNT + 1, MODE_COUNT), dtype=np.float64)
    for pm in range(MODE_COUNT + 1):
        prev_bits = (1, 1, 1) if pm == START_MODE else TRANSITIONS[pm].consume
        for cm in range(MODE_COUNT):
            curr_bits = TRANSITIONS[cm].consume
            total = 0.0
            for dim in range(3):
                if curr_bits[dim] == 1:
                    continue
                total += gap_extend if prev_bits[dim] == 0 else gap_open
            gap_pen[pm, cm] = total
    return gap_pen


@numba.njit(nogil=True, cache=True)
def _token_dp_kernel(
    n_a: int, n_b: int, n_c: int,
    ab: np.ndarray, ac: np.ndarray, bc: np.ndarray,
    gap_pen: np.ndarray,
    t_consume: np.ndarray,
    t_ab: np.ndarray, t_ac: np.ndarray, t_bc: np.ndarray,
    mode_count: int, start_mode: int,
    scores: np.ndarray, backptr: np.ndarray,
) -> None:
    """Numba-JIT compiled 3-way token DP recurrence.

    Fills scores and backptr arrays in-place.  Runs ~50-100x faster than
    pure Python and releases the GIL so multiple segments can align in
    parallel threads.
    """
    neg_inf = -1e308
    eps = 1e-12

    for i in range(n_a + 1):
        for j in range(n_b + 1):
            for k in range(n_c + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                for mi in range(mode_count):
                    di = t_consume[mi, 0]
                    dj = t_consume[mi, 1]
                    dk = t_consume[mi, 2]
                    if i < di or j < dj or k < dk:
                        continue

                    ts = 0.0
                    if t_ab[mi]:
                        ts += ab[i - 1, j - 1]
                    if t_ac[mi]:
                        ts += ac[i - 1, k - 1]
                    if t_bc[mi]:
                        ts += bc[j - 1, k - 1]

                    pi = i - di
                    pj = j - dj
                    pk = k - dk
                    best_pm = -1
                    best_sc = neg_inf
                    for pm in range(start_mode + 1):
                        prev_sc = scores[pi, pj, pk, pm]
                        if prev_sc <= neg_inf:
                            continue
                        cand = prev_sc + ts + gap_pen[pm, mi]
                        if cand > best_sc + eps:
                            best_sc = cand
                            best_pm = pm
                        elif abs(cand - best_sc) <= eps and (best_pm < 0 or pm < best_pm):
                            best_pm = pm

                    if best_pm >= 0:
                        scores[i, j, k, mi] = best_sc
                        backptr[i, j, k, mi] = best_pm


def align_three_way_tokens(
    tokens_a: list[str],
    tokens_b: list[str],
    tokens_c: list[str],
    *,
    match_score: float = 1.0,
    mismatch_score: float = -0.30,
    gap_open: float = -0.40,
    gap_extend: float = -0.15,
    normalize_fn: Callable[[str], str] | None = None,
) -> list[tuple[str | None, str | None, str | None]]:
    """3-way global DP alignment for word tokens.

    Uses a Numba-JIT compiled inner loop (~50-100x faster than pure Python)
    that releases the GIL so multiple segments can align in parallel threads.

    When *normalize_fn* is provided, token comparisons use normalized forms
    (e.g. stripping markdown emphasis markers) so that formatting-only
    differences do not produce mismatches.  The traceback still returns the
    **original** tokens so downstream code preserves formatting.

    Returns a list of alignment columns, each a tuple of
    (token_a_or_None, token_b_or_None, token_c_or_None).
    """
    n_a, n_b, n_c = len(tokens_a), len(tokens_b), len(tokens_c)
    if n_a == 0 and n_b == 0 and n_c == 0:
        return []

    # Optionally pre-normalize tokens for comparison only.
    # Original tokens are still used in the traceback output.
    if normalize_fn is not None:
        norm_a = [normalize_fn(t) for t in tokens_a]
        norm_b = [normalize_fn(t) for t in tokens_b]
        norm_c = [normalize_fn(t) for t in tokens_c]
    else:
        norm_a, norm_b, norm_c = tokens_a, tokens_b, tokens_c

    # Precompute pairwise exact-match score matrices on (optionally normalized) tokens.
    ab = np.empty((n_a, n_b), dtype=np.float64)
    ac = np.empty((n_a, n_c), dtype=np.float64)
    bc = np.empty((n_b, n_c), dtype=np.float64)
    for i in range(n_a):
        for j in range(n_b):
            ab[i, j] = match_score if norm_a[i] == norm_b[j] else mismatch_score
    for i in range(n_a):
        for k in range(n_c):
            ac[i, k] = match_score if norm_a[i] == norm_c[k] else mismatch_score
    for j in range(n_b):
        for k in range(n_c):
            bc[j, k] = match_score if norm_b[j] == norm_c[k] else mismatch_score

    gap_pen = _build_token_gap_penalties(gap_open, gap_extend)

    # Allocate DP tables.
    scores = np.full((n_a + 1, n_b + 1, n_c + 1, MODE_COUNT + 1), -1e308, dtype=np.float64)
    backptr = np.full((n_a + 1, n_b + 1, n_c + 1, MODE_COUNT), -1, dtype=np.int32)
    scores[0, 0, 0, START_MODE] = 0.0

    # Run the Numba-JIT compiled kernel.
    _token_dp_kernel(
        n_a, n_b, n_c,
        ab, ac, bc, gap_pen,
        _T_CONSUME, _T_AB, _T_AC, _T_BC,
        MODE_COUNT, START_MODE,
        scores, backptr,
    )

    # Find best terminal mode.
    best_mode = 0
    best_score = -1e308
    for mi in range(MODE_COUNT):
        sc = scores[n_a, n_b, n_c, mi]
        if sc > best_score + 1e-12:
            best_score = sc
            best_mode = mi
        elif np.isfinite(sc) and np.isfinite(best_score) and abs(sc - best_score) <= 1e-12 and mi < best_mode:
            best_mode = mi

    if not np.isfinite(best_score):
        return []

    # Traceback (O(n) — fast even in pure Python).
    columns: list[tuple[str | None, str | None, str | None]] = []
    i, j, k, mode = n_a, n_b, n_c, best_mode
    while not (i == 0 and j == 0 and k == 0):
        di = int(_T_CONSUME[mode, 0])
        dj = int(_T_CONSUME[mode, 1])
        dk = int(_T_CONSUME[mode, 2])
        col = (
            tokens_a[i - 1] if di else None,
            tokens_b[j - 1] if dj else None,
            tokens_c[k - 1] if dk else None,
        )
        columns.append(col)
        prev_m = int(backptr[i, j, k, mode])
        i -= di
        j -= dj
        k -= dk
        mode = prev_m
        if mode == START_MODE:
            break

    columns.reverse()
    return columns
