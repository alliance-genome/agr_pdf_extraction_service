"""Composable scoring for 3-way block alignment."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from app.services.consensus_models import Block

_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_CITATION_KEY_RE = re.compile(
    r"\["
    r"(?:"
    r"\d+(?:\s*[-–,]\s*\d+)*"
    r"|"
    r"[A-Za-z]+(?:\s+et\s+al\.?)?\s*\d{4}"
    r")"
    r"\]"
)


@dataclass(frozen=True)
class ScoreConfig:
    """Scoring hyperparameters in a normalized (approximately -1 to +1) space."""

    match_min: float = 0.60
    gap_open: float = -0.40
    gap_extend: float = -0.15
    strong_ratio_threshold: float = 0.15
    mild_ratio_threshold: float = 0.35
    min_length_multiplier: float = 0.20
    cross_family_penalty: float = 0.55
    heading_mismatch_penalty: float = 0.30
    heading_pair_bonus: float = 0.10
    heading_level_bonus: float = 0.06
    family_match_bonus: float = 0.06
    numeric_mismatch_penalty: float = 0.18
    citation_mismatch_penalty: float = 0.12
    weak_match_penalty: float = 0.35


@dataclass(frozen=True)
class PairScoreComponents:
    """Breakdown for one pair score."""

    lexical: float
    family_bonus_or_penalty: float
    heading_bonus_or_penalty: float
    length_multiplier: float
    weak_match_penalty: float
    numeric_penalty: float
    citation_penalty: float
    total: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TransitionScore:
    """Score contribution for one DP transition."""

    total: float
    pair_scores: dict[str, PairScoreComponents]


_FAMILY_ALIASES = {
    "heading": "heading",
    "paragraph": "paragraph",
    "citation_list": "citation",
    "table": "table",
    "equation": "equation",
    "figure_ref": "figure",
}

# Whitelist small parser disagreements that should not get full cross-family penalties.
_FAMILY_WHITELIST = {
    frozenset(("heading", "paragraph")),
    frozenset(("paragraph", "citation")),
}


def _family(block_type: str) -> str:
    return _FAMILY_ALIASES.get(block_type, "paragraph")


def _families_compatible(fam_a: str, fam_b: str) -> bool:
    if fam_a == fam_b:
        return True
    return frozenset((fam_a, fam_b)) in _FAMILY_WHITELIST


def _length_ratio(text_a: str, text_b: str) -> float:
    len_a = len(text_a or "")
    len_b = len(text_b or "")
    if len_a == 0 or len_b == 0:
        return 0.0
    return min(len_a, len_b) / max(len_a, len_b)


def _extract_numeric_tokens(text: str) -> set[str]:
    return set(_NUMERIC_RE.findall(text or ""))


def _extract_citation_tokens(text: str) -> set[str]:
    return set(_CITATION_KEY_RE.findall(text or ""))


def pair_score(
    block_a: Block,
    block_b: Block,
    *,
    config: ScoreConfig | None = None,
) -> PairScoreComponents:
    """Return a composable pair score breakdown."""
    cfg = config or ScoreConfig()

    token_set = fuzz.token_set_ratio(block_a.normalized_text, block_b.normalized_text) / 100.0
    token_sort = fuzz.token_sort_ratio(block_a.normalized_text, block_b.normalized_text) / 100.0
    raw_ratio = fuzz.ratio(block_a.normalized_text, block_b.normalized_text) / 100.0
    # Blend overlap and strict sequence-level similarity so tiny fragments
    # do not get near-perfect scores against long blocks.
    lexical = (0.55 * token_set) + (0.30 * token_sort) + (0.15 * raw_ratio)

    fam_a = _family(block_a.block_type)
    fam_b = _family(block_b.block_type)
    family_component = 0.0
    reasons: list[str] = []
    if fam_a == fam_b and fam_a not in {"paragraph", "citation"}:
        family_component += cfg.family_match_bonus
        reasons.append("family_match")
    elif not _families_compatible(fam_a, fam_b):
        family_component -= cfg.cross_family_penalty
        reasons.append("family_mismatch")

    heading_component = 0.0
    if fam_a == "heading" and fam_b == "heading":
        heading_component += cfg.heading_pair_bonus
        if block_a.heading_level is not None and block_b.heading_level is not None:
            level_delta = abs(block_a.heading_level - block_b.heading_level)
            heading_component += max(0.0, 1.0 - 0.25 * level_delta) * cfg.heading_level_bonus
        reasons.append("heading_pair")
    elif "heading" in (fam_a, fam_b):
        heading_component -= cfg.heading_mismatch_penalty
        reasons.append("heading_mismatch")

    ratio = _length_ratio(block_a.normalized_text, block_b.normalized_text)
    length_multiplier = 1.0
    if fam_a in {"paragraph", "citation", "heading"} and fam_b in {"paragraph", "citation", "heading"}:
        if ratio < cfg.strong_ratio_threshold:
            length_multiplier = max(cfg.min_length_multiplier, ratio / cfg.strong_ratio_threshold)
            reasons.append("strong_length_dampen")
        elif ratio < cfg.mild_ratio_threshold:
            length_multiplier = 0.80
            reasons.append("mild_length_dampen")

    weak_match_component = 0.0
    if lexical < cfg.match_min:
        weak_match_component = (cfg.match_min - lexical) * cfg.weak_match_penalty
        reasons.append("weak_match")

    numeric_penalty = 0.0
    nums_a = _extract_numeric_tokens(block_a.normalized_text)
    nums_b = _extract_numeric_tokens(block_b.normalized_text)
    if nums_a != nums_b:
        numeric_penalty = cfg.numeric_mismatch_penalty
        reasons.append("numeric_mismatch")

    citation_penalty = 0.0
    cites_a = _extract_citation_tokens(block_a.normalized_text)
    cites_b = _extract_citation_tokens(block_b.normalized_text)
    if cites_a != cites_b:
        citation_penalty = cfg.citation_mismatch_penalty
        reasons.append("citation_mismatch")

    base = lexical + family_component + heading_component
    total = (base * length_multiplier) - weak_match_component - numeric_penalty - citation_penalty
    total = max(-1.0, min(1.0, total))

    return PairScoreComponents(
        lexical=lexical,
        family_bonus_or_penalty=family_component,
        heading_bonus_or_penalty=heading_component,
        length_multiplier=length_multiplier,
        weak_match_penalty=weak_match_component,
        numeric_penalty=numeric_penalty,
        citation_penalty=citation_penalty,
        total=total,
        reasons=tuple(reasons),
    )


def transition_score(
    grobid_block: Block | None,
    docling_block: Block | None,
    marker_block: Block | None,
    *,
    config: ScoreConfig | None = None,
) -> TransitionScore:
    """Return sum-of-pairs similarity for one DP transition."""
    cfg = config or ScoreConfig()
    pair_scores: dict[str, PairScoreComponents] = {}
    total = 0.0

    if grobid_block is not None and docling_block is not None:
        score = pair_score(grobid_block, docling_block, config=cfg)
        pair_scores["grobid_docling"] = score
        total += score.total
    if grobid_block is not None and marker_block is not None:
        score = pair_score(grobid_block, marker_block, config=cfg)
        pair_scores["grobid_marker"] = score
        total += score.total
    if docling_block is not None and marker_block is not None:
        score = pair_score(docling_block, marker_block, config=cfg)
        pair_scores["docling_marker"] = score
        total += score.total

    return TransitionScore(total=total, pair_scores=pair_scores)
