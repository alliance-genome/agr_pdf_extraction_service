"""
Selective LLM Merge Consensus Pipeline.

Parses markdown outputs from three extractors (Grobid, Docling, Marker),
identifies agreement programmatically, and only sends disagreement sections
to the LLM for resolution. Expected to save 50-70% of LLM token usage.

Consensus classes:
    AGREE_EXACT  - Normalized text identical in >= 2 extractors
    AGREE_NEAR   - High similarity with no numeric/citation differences
    GAP          - Block present in only one extractor (structural gap)
    CONFLICT     - Everything else; sent to LLM for resolution
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import mistune
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from app.services.llm_service import LLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

AGREE_EXACT = "AGREE_EXACT"
AGREE_NEAR = "AGREE_NEAR"
GAP = "GAP"
CONFLICT = "CONFLICT"


@dataclass
class Block:
    block_id: str
    block_type: str  # "heading", "paragraph", "table", "figure_ref", "equation", "citation_list"
    raw_text: str
    normalized_text: str
    heading_level: int | None  # 1-6 for headings, None otherwise
    order_index: int  # 0-based position in source document
    source: str  # "grobid", "docling", "marker"
    source_md: str = ""  # original markdown syntax; falls back to raw_text when empty


@dataclass
class AlignedTriple:
    segment_id: str  # immutable, e.g., "seg_001"
    grobid_block: Block | None = None
    docling_block: Block | None = None
    marker_block: Block | None = None
    classification: str = ""  # set during classify step
    agreed_text: str | None = None  # resolved text for non-CONFLICT
    confidence: float = 0.0  # alignment confidence for this triple


# ---------------------------------------------------------------------------
# Step 1: PARSE — Markdown to Block list
# ---------------------------------------------------------------------------

_TABLE_RE = re.compile(r"^\|.*\|", re.MULTILINE)
_EQUATION_RE = re.compile(r"\$\$.+?\$\$", re.DOTALL)
_FIGURE_RE = re.compile(r"!\[.*?\]\(.*?\)")
_CITATION_LIST_RE = re.compile(
    r"^(?:\d+\.\s+|[-*]\s+|\[\d+\]\s+).+",
    re.MULTILINE,
)


def _has_child_type(token: dict, child_type: str) -> bool:
    """Check if a token has a child of the given type (recursive)."""
    children = token.get("children", [])
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                if child.get("type") == child_type:
                    return True
                if _has_child_type(child, child_type):
                    return True
    return False


def _classify_block_type(token: dict) -> tuple[str, int | None]:
    """Return (block_type, heading_level) from a mistune AST token."""
    ttype = token.get("type", "")
    if ttype == "heading":
        level = token.get("level", 1)
        return "heading", level
    if ttype in ("table", "table_head", "table_body"):
        return "table", None
    if ttype == "block_math" or ttype == "math":
        return "equation", None
    # Check for image/figure children inside paragraphs
    if _has_child_type(token, "image"):
        return "figure_ref", None
    return "paragraph", None


def _extract_text(token: dict) -> str:
    """Recursively extract raw text from a mistune AST token."""
    if isinstance(token, str):
        return token
    raw = token.get("raw", "")
    if raw:
        return raw
    text = token.get("text", "")
    if text:
        return text
    children = token.get("children")
    if children:
        if isinstance(children, list):
            return "".join(_extract_text(c) for c in children)
        if isinstance(children, str):
            return children
    return ""


def _reconstruct_children(token: dict) -> str:
    """Reconstruct markdown from a token's children list."""
    children = token.get("children", [])
    if isinstance(children, str):
        return children
    if isinstance(children, list):
        return "".join(_reconstruct_markdown(c) for c in children)
    return ""


def _reconstruct_markdown(token: dict) -> str:
    """Reconstruct markdown source from a mistune AST token, preserving syntax.

    Unlike ``_extract_text`` (which returns plain text for comparison),
    this function preserves heading markers, bold/italic, links, and image
    references so the final assembled document retains its formatting.
    """
    if isinstance(token, str):
        return token

    ttype = token.get("type", "")

    # --- Block-level tokens ---
    if ttype == "heading":
        level = token.get("level", token.get("attrs", {}).get("level", 1))
        inner = _reconstruct_children(token)
        return "#" * level + " " + inner

    if ttype == "paragraph":
        return _reconstruct_children(token)

    if ttype in ("table", "table_head", "table_body"):
        raw = token.get("raw", "")
        if raw:
            return raw
        return _extract_text(token)

    if ttype in ("block_code", "code_block"):
        raw = token.get("raw", "")
        info = token.get("attrs", {}).get("info", token.get("info", ""))
        if raw:
            return f"```{info}\n{raw}```"
        return _extract_text(token)

    if ttype in ("block_math", "math"):
        raw = token.get("raw", "")
        if raw:
            return f"$${raw}$$"
        return _extract_text(token)

    # --- Inline tokens ---
    if ttype == "text":
        raw = token.get("raw", "")
        if raw:
            return raw
        text = token.get("text", "")
        if text:
            return text
        children = token.get("children", "")
        return children if isinstance(children, str) else ""

    if ttype == "strong":
        return f"**{_reconstruct_children(token)}**"

    if ttype == "emphasis":
        return f"*{_reconstruct_children(token)}*"

    if ttype == "link":
        inner = _reconstruct_children(token)
        url = token.get("attrs", {}).get("url", token.get("link", ""))
        return f"[{inner}]({url})"

    if ttype == "image":
        alt = token.get("attrs", {}).get("alt", "")
        url = token.get("attrs", {}).get("url", token.get("src", ""))
        if not alt:
            alt = _extract_text(token)
        return f"![{alt}]({url})"

    if ttype == "codespan":
        raw = token.get("raw", token.get("text", ""))
        if not raw:
            children = token.get("children", "")
            raw = children if isinstance(children, str) else _extract_text(token)
        return f"`{raw}`"

    if ttype in ("linebreak", "softbreak"):
        return "\n"

    # --- Fallback: try children, then raw/text ---
    children = token.get("children")
    if children and isinstance(children, list):
        return _reconstruct_children(token)

    raw = token.get("raw", "")
    if raw:
        return raw
    return _extract_text(token)


def _detect_special_blocks(text: str) -> str | None:
    """Detect table/equation/figure/citation blocks from raw text."""
    if _TABLE_RE.search(text):
        return "table"
    if _EQUATION_RE.search(text):
        return "equation"
    if _FIGURE_RE.search(text):
        return "figure_ref"
    return None


def normalize_text(text: str) -> str:
    """Normalize text for comparison: unicode, whitespace, markdown syntax."""
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # Normalize markdown bold/italic syntax variations
    text = re.sub(r"__(.+?)__", r"**\1**", text)
    text = re.sub(r"_(.+?)_", r"*\1*", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Lowercase for comparison
    text = text.lower()
    return text


def parse_markdown(markdown_text: str, source: str) -> list[Block]:
    """Parse markdown into a list of typed Blocks."""
    md = mistune.create_markdown(renderer=None)
    tokens = md(markdown_text)

    blocks: list[Block] = []
    idx = 0

    if not isinstance(tokens, list):
        # Fallback: split by double newlines if AST returns unexpected type
        paragraphs = [p.strip() for p in markdown_text.split("\n\n") if p.strip()]
        tokens = [{"type": "paragraph", "raw": p} for p in paragraphs]

    for token in tokens:
        if isinstance(token, str):
            token = {"type": "paragraph", "raw": token}

        ttype = token.get("type", "")
        if ttype in ("newline", "space", "thematic_break"):
            continue

        raw_text = _extract_text(token)
        if not raw_text.strip():
            continue

        block_type, heading_level = _classify_block_type(token)

        # Override with content-based detection for paragraphs
        if block_type == "paragraph":
            detected = _detect_special_blocks(raw_text)
            if detected:
                block_type = detected

        source_md = _reconstruct_markdown(token).strip()

        blocks.append(Block(
            block_id=f"{source}_block_{idx:03d}",
            block_type=block_type,
            raw_text=raw_text.strip(),
            normalized_text=normalize_text(raw_text),
            heading_level=heading_level,
            order_index=idx,
            source=source,
            source_md=source_md,
        ))
        idx += 1

    return blocks


# ---------------------------------------------------------------------------
# Step 2: ALIGN — Hungarian assignment with gap states
# ---------------------------------------------------------------------------

_HEADING_WEIGHT = 0.4
_CONTENT_WEIGHT = 0.4
_POSITION_WEIGHT = 0.2
_GAP_PENALTY = 40.0  # cost of assigning a block to the gap (dummy) column


def _pair_score(block_a: Block, block_b: Block, max_index_a: int, max_index_b: int) -> float:
    """Score similarity between two blocks (0-100, higher is better)."""
    # Heading similarity
    if block_a.block_type == "heading" and block_b.block_type == "heading":
        heading_sim = fuzz.token_set_ratio(block_a.normalized_text, block_b.normalized_text)
    elif block_a.block_type == "heading" or block_b.block_type == "heading":
        heading_sim = 0.0
    else:
        heading_sim = 50.0  # neutral for non-heading pairs

    # Content overlap
    content_sim = fuzz.token_set_ratio(block_a.normalized_text, block_b.normalized_text)

    # Positional prior
    pos_a = block_a.order_index / max(max_index_a, 1)
    pos_b = block_b.order_index / max(max_index_b, 1)
    position_sim = max(0, 100 * (1.0 - abs(pos_a - pos_b)))

    return (
        _HEADING_WEIGHT * heading_sim
        + _CONTENT_WEIGHT * content_sim
        + _POSITION_WEIGHT * position_sim
    )


def align_blocks(
    blocks_by_source: dict[str, list[Block]],
) -> tuple[list[AlignedTriple], float]:
    """
    Align blocks across extractors using reference-based Hungarian assignment.

    Uses the extractor with the most blocks as the reference, then aligns
    the other two to it. Returns aligned triples and mean alignment confidence.
    """
    sources = sorted(blocks_by_source.keys())
    if not sources:
        return [], 0.0

    # Pick reference: extractor with the most blocks
    ref_source = max(sources, key=lambda s: len(blocks_by_source[s]))
    other_sources = [s for s in sources if s != ref_source]
    ref_blocks = blocks_by_source[ref_source]

    if not ref_blocks:
        return [], 0.0

    # Build alignment mapping: ref_index -> {source: (block, score)}
    alignments: dict[int, dict[str, tuple[Block, float]]] = {
        i: {} for i in range(len(ref_blocks))
    }
    leftover_blocks: list[tuple[Block, float]] = []  # (block, approx_position_ratio)

    for other_src in other_sources:
        other_blocks = blocks_by_source.get(other_src, [])
        if not other_blocks:
            continue

        n_ref = len(ref_blocks)
        n_other = len(other_blocks)
        max_idx_ref = max(b.order_index for b in ref_blocks) if ref_blocks else 1
        max_idx_other = max(b.order_index for b in other_blocks) if other_blocks else 1

        # Build cost matrix (minimize cost = maximize similarity)
        # Add dummy columns/rows for gap assignment
        size = max(n_ref, n_other)
        cost_matrix = np.full((size, size), _GAP_PENALTY, dtype=float)

        for i in range(n_ref):
            for j in range(n_other):
                score = _pair_score(ref_blocks[i], other_blocks[j], max_idx_ref, max_idx_other)
                cost_matrix[i, j] = 100.0 - score  # convert similarity to cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_other_indices = set()
        for r, c in zip(row_ind, col_ind):
            if r < n_ref and c < n_other:
                score = 100.0 - cost_matrix[r, c]
                if score > (100.0 - _GAP_PENALTY):  # only accept if similarity beats gap cost
                    alignments[r][other_src] = (other_blocks[c], score)
                    matched_other_indices.add(c)

        # Collect leftover unmatched blocks from this source
        for j in range(n_other):
            if j not in matched_other_indices:
                pos_ratio = other_blocks[j].order_index / max(max_idx_other, 1)
                leftover_blocks.append((other_blocks[j], pos_ratio))

    # Build aligned triples with position ratios for interleaving
    # Each entry: (position_ratio, triple, scores_list)
    positioned_triples: list[tuple[float, AlignedTriple, list[float]]] = []
    max_ref_idx = max(b.order_index for b in ref_blocks) if ref_blocks else 1

    for ref_idx, ref_block in enumerate(ref_blocks):
        triple = AlignedTriple(segment_id="")  # segment_id assigned after sorting
        setattr(triple, f"{ref_source}_block", ref_block)

        scores_for_triple: list[float] = []
        for other_src in other_sources:
            if other_src in alignments[ref_idx]:
                block, score = alignments[ref_idx][other_src]
                setattr(triple, f"{other_src}_block", block)
                scores_for_triple.append(score)

        triple.confidence = sum(scores_for_triple) / len(scores_for_triple) if scores_for_triple else 0.0
        pos_ratio = ref_block.order_index / max(max_ref_idx, 1)
        positioned_triples.append((pos_ratio, triple, scores_for_triple))

    # Interleave leftover blocks by position ratio (not appended at end)
    for block, pos_ratio in leftover_blocks:
        triple = AlignedTriple(segment_id="")
        setattr(triple, f"{block.source}_block", block)
        triple.confidence = 0.0
        positioned_triples.append((pos_ratio, triple, []))

    # Sort all triples by document position, then assign segment_ids
    positioned_triples.sort(key=lambda x: x[0])

    triples: list[AlignedTriple] = []
    all_scores: list[float] = []
    for seg_idx, (_, triple, scores) in enumerate(positioned_triples):
        triple.segment_id = f"seg_{seg_idx:03d}"
        all_scores.extend(scores)
        triples.append(triple)

    # Mean alignment confidence
    alignment_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0
    alignment_confidence /= 100.0  # normalize to 0-1 range

    return triples, alignment_confidence


# ---------------------------------------------------------------------------
# Step 3: CLASSIFY — Consensus state for each triple
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_CITATION_KEY_RE = re.compile(r"\[(?:\d+|[A-Za-z]+(?:\s+et\s+al\.?)?\s*\d{4})\]")

# Per-block-type source preference for AGREE_NEAR
_SOURCE_PREFERENCE = {
    "heading": "grobid",
    "citation_list": "grobid",
    "table": "docling",
    "figure_ref": "marker",
    "paragraph": "marker",
    "equation": "docling",
}


def _extract_numeric_tokens(text: str) -> set[str]:
    return set(_NUMERIC_RE.findall(text))


def _extract_citation_keys(text: str) -> set[str]:
    return set(_CITATION_KEY_RE.findall(text))


def _get_present_blocks(triple: AlignedTriple) -> list[Block]:
    """Return the non-None blocks from a triple."""
    blocks = []
    for attr in ("grobid_block", "docling_block", "marker_block"):
        b = getattr(triple, attr)
        if b is not None:
            blocks.append(b)
    return blocks


def _pick_preferred_text(
    blocks: list[Block], block_type: str, agreeing_indices: tuple[int, int] | None = None,
) -> str:
    """Pick the best source text, preferring an agreeing block over an outlier.

    When *agreeing_indices* is provided (the pair that matched), the preferred
    source is chosen only among the agreeing blocks so the outlier is never
    selected.  Falls back to source-preference among all blocks if no agreeing
    pair is supplied.
    """
    candidates = (
        [blocks[i] for i in agreeing_indices] if agreeing_indices is not None else blocks
    )
    preferred_source = _SOURCE_PREFERENCE.get(block_type, "marker")
    for b in candidates:
        if b.source == preferred_source:
            return b.source_md or b.raw_text
    return candidates[0].source_md or candidates[0].raw_text


def classify_triples(
    triples: list[AlignedTriple],
    near_threshold: float = 0.92,
    levenshtein_threshold: float = 0.90,
    always_escalate_tables: bool = True,
) -> None:
    """Classify each triple in-place. Mutates triple.classification and triple.agreed_text."""
    for triple in triples:
        blocks = _get_present_blocks(triple)

        if len(blocks) == 0:
            triple.classification = GAP
            triple.agreed_text = ""
            continue

        if len(blocks) == 1:
            triple.classification = GAP
            triple.agreed_text = blocks[0].source_md or blocks[0].raw_text
            continue

        # Tables and equations escalate to CONFLICT when configured (Phase 1 default).
        # Escalate if ANY block in the triple is table/equation, not just the first,
        # since extractors may disagree on block type.
        block_types = {b.block_type for b in blocks}
        if always_escalate_tables and block_types & {"table", "equation"}:
            triple.classification = CONFLICT
            continue

        # Use first block's type for source-preference routing
        block_type = blocks[0].block_type

        # Check all pairs for agreement
        normalized_texts = [b.normalized_text for b in blocks]
        raw_texts = [b.raw_text for b in blocks]

        # Check for AGREE_EXACT: any pair has identical normalized text
        exact_pair = None
        for i in range(len(normalized_texts)):
            for j in range(i + 1, len(normalized_texts)):
                if normalized_texts[i] == normalized_texts[j]:
                    exact_pair = (i, j)
                    break
            if exact_pair is not None:
                break

        if exact_pair is not None:
            triple.classification = AGREE_EXACT
            triple.agreed_text = _pick_preferred_text(blocks, block_type, exact_pair)
            continue

        # Check for AGREE_NEAR: pairwise similarity >= thresholds
        # AND no numeric/citation differences
        near_pair = None
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                token_ratio = fuzz.token_set_ratio(
                    normalized_texts[i], normalized_texts[j]
                ) / 100.0

                # Normalized Levenshtein distance
                max_len = max(len(normalized_texts[i]), len(normalized_texts[j]), 1)
                lev_dist = Levenshtein.distance(normalized_texts[i], normalized_texts[j])
                lev_sim = 1.0 - (lev_dist / max_len)

                if token_ratio >= near_threshold and lev_sim >= levenshtein_threshold:
                    # Check numeric/citation guardrail
                    nums_i = _extract_numeric_tokens(raw_texts[i])
                    nums_j = _extract_numeric_tokens(raw_texts[j])
                    cites_i = _extract_citation_keys(raw_texts[i])
                    cites_j = _extract_citation_keys(raw_texts[j])

                    if nums_i == nums_j and cites_i == cites_j:
                        near_pair = (i, j)
                        break
            if near_pair is not None:
                break

        if near_pair is not None:
            triple.classification = AGREE_NEAR
            triple.agreed_text = _pick_preferred_text(blocks, block_type, near_pair)
            continue

        # Everything else is CONFLICT
        triple.classification = CONFLICT


# ---------------------------------------------------------------------------
# Step 4: GUARD — Check fallback conditions
# ---------------------------------------------------------------------------

def check_guards(
    triples: list[AlignedTriple],
    alignment_confidence: float,
    conflict_ratio_threshold: float = 0.4,
    alignment_confidence_threshold: float = 0.5,
) -> tuple[bool, str | None]:
    """
    Check whether the consensus pipeline should fall back to full-LLM merge.

    Returns (should_fallback, reason).
    """
    total = len(triples)
    if total == 0:
        return True, "no_blocks"

    num_gap = sum(1 for t in triples if t.classification == GAP)
    num_conflict = sum(1 for t in triples if t.classification == CONFLICT)

    # conflict_ratio excludes GAP blocks from denominator
    denominator = total - num_gap
    if denominator == 0:
        # All blocks are GAP — no real conflicts
        conflict_ratio = 0.0
    else:
        conflict_ratio = num_conflict / denominator

    if conflict_ratio > conflict_ratio_threshold:
        return True, "conflict_ratio"

    if alignment_confidence < alignment_confidence_threshold:
        return True, "alignment_confidence"

    return False, None


# ---------------------------------------------------------------------------
# Step 5: ASSEMBLE — Build final merged markdown
# ---------------------------------------------------------------------------

def assemble(
    triples: list[AlignedTriple],
    resolved_conflicts: dict[str, str],
) -> str:
    """Assemble the final merged markdown from classified triples and LLM resolutions."""
    parts: list[str] = []

    for triple in triples:
        if triple.classification in (AGREE_EXACT, AGREE_NEAR, GAP):
            if triple.agreed_text:
                parts.append(triple.agreed_text)
        elif triple.classification == CONFLICT:
            resolved = resolved_conflicts.get(triple.segment_id)
            if resolved:
                parts.append(resolved)
            else:
                # Fallback: use first available block text
                blocks = _get_present_blocks(triple)
                if blocks:
                    parts.append(blocks[0].source_md or blocks[0].raw_text)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 6: METRICS — Compute pipeline statistics
# ---------------------------------------------------------------------------

def compute_metrics(
    triples: list[AlignedTriple],
    alignment_confidence: float,
    fallback_triggered: bool,
    fallback_reason: str | None,
) -> dict:
    """Compute metrics for the consensus pipeline run."""
    total = len(triples)
    num_exact = sum(1 for t in triples if t.classification == AGREE_EXACT)
    num_near = sum(1 for t in triples if t.classification == AGREE_NEAR)
    num_gap = sum(1 for t in triples if t.classification == GAP)
    num_conflict = sum(1 for t in triples if t.classification == CONFLICT)

    denominator = total - num_gap
    conflict_ratio = num_conflict / denominator if denominator > 0 else 0.0

    # Estimate tokens saved: agreed + gap blocks not sent to LLM
    agreed_blocks = num_exact + num_near + num_gap
    # Rough estimate: 100 tokens per block on average
    tokens_saved_estimate = agreed_blocks * 100

    return {
        "total_blocks": total,
        "agree_exact": num_exact,
        "agree_near": num_near,
        "gap": num_gap,
        "conflict": num_conflict,
        "conflict_ratio": round(conflict_ratio, 4),
        "alignment_confidence": round(alignment_confidence, 4),
        "tokens_saved_estimate": tokens_saved_estimate,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
    }


# ---------------------------------------------------------------------------
# Orchestrator — Top-level entry point
# ---------------------------------------------------------------------------

def merge_with_consensus(
    grobid_md: str,
    docling_md: str,
    marker_md: str,
    llm: "LLM",
) -> tuple[str | None, dict]:
    """
    Attempt selective LLM merge. Returns (merged_markdown, metrics).
    Returns (None, metrics) if fallback to full-LLM merge is needed.

    Requires all 3 extractor outputs. If any are empty/None, returns (None, metrics)
    immediately with fallback_reason="missing_extractor".
    """
    from config import Config

    # Validate all 3 inputs
    if not grobid_md or not docling_md or not marker_md:
        metrics = compute_metrics([], 0.0, True, "missing_extractor")
        return None, metrics

    # Step 1: Parse and normalize
    logger.info("Consensus pipeline: parsing markdown outputs...")
    grobid_blocks = parse_markdown(grobid_md, "grobid")
    docling_blocks = parse_markdown(docling_md, "docling")
    marker_blocks = parse_markdown(marker_md, "marker")

    blocks_by_source = {
        "grobid": grobid_blocks,
        "docling": docling_blocks,
        "marker": marker_blocks,
    }

    logger.info(
        "Parsed blocks: grobid=%d, docling=%d, marker=%d",
        len(grobid_blocks), len(docling_blocks), len(marker_blocks),
    )

    # Step 2: Align
    logger.info("Consensus pipeline: aligning blocks...")
    triples, alignment_confidence = align_blocks(blocks_by_source)
    logger.info(
        "Aligned %d triples, confidence=%.3f", len(triples), alignment_confidence
    )

    # Step 3: Classify
    logger.info("Consensus pipeline: classifying triples...")
    classify_triples(
        triples,
        near_threshold=Config.CONSENSUS_NEAR_THRESHOLD,
        levenshtein_threshold=Config.CONSENSUS_LEVENSHTEIN_THRESHOLD,
        always_escalate_tables=Config.CONSENSUS_ALWAYS_ESCALATE_TABLES,
    )

    classifications = {}
    for t in triples:
        classifications[t.classification] = classifications.get(t.classification, 0) + 1
    logger.info("Classifications: %s", classifications)

    # Step 4: Guard checks
    should_fallback, fallback_reason = check_guards(
        triples,
        alignment_confidence,
        conflict_ratio_threshold=Config.CONSENSUS_CONFLICT_RATIO_FALLBACK,
        alignment_confidence_threshold=Config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK,
    )

    if should_fallback:
        logger.info("Consensus pipeline: fallback triggered (%s)", fallback_reason)
        metrics = compute_metrics(triples, alignment_confidence, True, fallback_reason)
        return None, metrics

    # Step 5: Resolve conflicts via LLM
    conflict_triples = [t for t in triples if t.classification == CONFLICT]
    resolved_conflicts: dict[str, str] = {}

    if conflict_triples:
        logger.info("Consensus pipeline: resolving %d conflicts via LLM...", len(conflict_triples))
        conflict_bundles = []
        for t in conflict_triples:
            bundle = {
                "segment_id": t.segment_id,
                "block_type": _get_present_blocks(t)[0].block_type if _get_present_blocks(t) else "paragraph",
                "grobid": (t.grobid_block.source_md or t.grobid_block.raw_text) if t.grobid_block else "",
                "docling": (t.docling_block.source_md or t.docling_block.raw_text) if t.docling_block else "",
                "marker": (t.marker_block.source_md or t.marker_block.raw_text) if t.marker_block else "",
            }
            conflict_bundles.append(bundle)

        try:
            resolved_conflicts = llm.resolve_conflicts(conflict_bundles)
        except Exception as e:
            logger.warning("Consensus pipeline: LLM conflict resolution failed: %s", e)
            metrics = compute_metrics(triples, alignment_confidence, True, "llm_error")
            return None, metrics

    # Step 6: Assemble
    logger.info("Consensus pipeline: assembling final document...")
    merged_md = assemble(triples, resolved_conflicts)

    metrics = compute_metrics(triples, alignment_confidence, False, None)
    logger.info(
        "Consensus pipeline complete: %d blocks, %d conflicts resolved, "
        "~%d tokens saved",
        metrics["total_blocks"],
        metrics["conflict"],
        metrics["tokens_saved_estimate"],
    )

    return merged_md, metrics
