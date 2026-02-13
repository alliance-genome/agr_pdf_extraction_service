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

# Output cleaning and QA: strip extractor artifacts, keep meaningful markdown.
_SPAN_OPEN_RE = re.compile(r"<span[^>]*>")
_SPAN_CLOSE_RE = re.compile(r"</span>")
_HTML_COMMENT_OUTPUT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"  +")

# Flanking context budget (~150 tokens, conservative 4 chars/token estimate).
_CHARS_PER_TOKEN = 4
_FLANK_TOKEN_BUDGET = 150
_FLANK_CHAR_BUDGET = _FLANK_TOKEN_BUDGET * _CHARS_PER_TOKEN


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
    """Normalize text for comparison: unicode, whitespace, markdown syntax.

    Aggressively strips extractor-specific artifacts so that identical
    content from different extractors produces identical normalized text.
    The original source_md on each Block is preserved for final assembly.
    """
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Marker: <span id="page-X-Y"> ... </span> anchors
    text = re.sub(r"<span[^>]*>", "", text)
    text = re.sub(r"</span>", "", text)

    # Docling: HTML comments like <!-- image -->
    text = re.sub(r"<!--.*?-->", "", text)

    # Marker: image references ![alt](path)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)

    # Markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # Strip all bold/italic markers for comparison
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)

    # Strip heading markers (treat heading levels as equivalent for comparison)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Lowercase for comparison
    text = text.lower()
    return text


def clean_output_md(text: str) -> str:
    """Strip extractor-specific artifacts while preserving markdown formatting."""
    text = _SPAN_OPEN_RE.sub("", text)
    text = _SPAN_CLOSE_RE.sub("", text)
    text = _HTML_COMMENT_OUTPUT_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


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
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_URL_RE = re.compile(r"\]\([^)]*\)")
_CITATION_KEY_RE = re.compile(
    r"\["
    r"(?:"
    r"\d+(?:\s*[-–,]\s*\d+)*"  # [8], [8-10], [8–10], [6,7]
    r"|"
    r"[A-Za-z]+(?:\s+et\s+al\.?)?\s*\d{4}"  # [Smith et al. 2024]
    r")"
    r"\]"
)

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
    """Extract numeric tokens, ignoring numbers inside tags and URLs."""
    cleaned = _HTML_TAG_RE.sub("", text)
    cleaned = _IMAGE_REF_RE.sub("", cleaned)
    cleaned = _LINK_URL_RE.sub("]", cleaned)
    return set(_NUMERIC_RE.findall(cleaned))


def _extract_citation_keys(text: str) -> set[str]:
    """Extract citation keys after removing HTML tags."""
    cleaned = _HTML_TAG_RE.sub("", text)
    return set(_CITATION_KEY_RE.findall(cleaned))


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

        # Debug diagnostics: why this triple fell through to CONFLICT
        if logger.isEnabledFor(logging.DEBUG):
            reasons: list[str] = []
            best_token = 0.0
            best_lev = 0.0
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    token_ratio = (
                        fuzz.token_set_ratio(normalized_texts[i], normalized_texts[j]) / 100.0
                    )
                    max_len = max(len(normalized_texts[i]), len(normalized_texts[j]), 1)
                    lev_dist = Levenshtein.distance(normalized_texts[i], normalized_texts[j])
                    lev_sim = 1.0 - (lev_dist / max_len)
                    best_token = max(best_token, token_ratio)
                    best_lev = max(best_lev, lev_sim)

                    if token_ratio < near_threshold:
                        reasons.append(f"token_ratio={token_ratio:.3f}<{near_threshold}")
                    if lev_sim < levenshtein_threshold:
                        reasons.append(f"lev_sim={lev_sim:.3f}<{levenshtein_threshold}")

                    nums_i = _extract_numeric_tokens(raw_texts[i])
                    nums_j = _extract_numeric_tokens(raw_texts[j])
                    if nums_i != nums_j:
                        reasons.append(f"numeric_diff={nums_i.symmetric_difference(nums_j)}")

                    cites_i = _extract_citation_keys(raw_texts[i])
                    cites_j = _extract_citation_keys(raw_texts[j])
                    if cites_i != cites_j:
                        reasons.append(f"citation_diff={cites_i.symmetric_difference(cites_j)}")

            logger.debug(
                "CONFLICT %s: best_token=%.3f best_lev=%.3f reasons=%s",
                triple.segment_id,
                best_token,
                best_lev,
                "; ".join(reasons[:5]),
            )


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


def dedup_gap_triples(
    triples: list[AlignedTriple],
    window: int = 3,
    similarity_threshold: float = 0.85,
    length_ratio_threshold: float = 0.7,
) -> int:
    """Remove near-duplicate GAP blocks within a local window."""
    removed = 0

    for i, t_i in enumerate(triples):
        if t_i.classification != GAP or not t_i.agreed_text or not t_i.agreed_text.strip():
            continue

        norm_i = normalize_text(t_i.agreed_text)
        if not norm_i:
            continue

        for j in range(i + 1, min(i + window + 1, len(triples))):
            t_j = triples[j]
            if t_j.classification != GAP or not t_j.agreed_text or not t_j.agreed_text.strip():
                continue

            norm_j = normalize_text(t_j.agreed_text)
            if not norm_j:
                continue

            len_ratio = min(len(norm_i), len(norm_j)) / max(len(norm_i), len(norm_j))
            if len_ratio < length_ratio_threshold:
                continue

            similarity = fuzz.token_set_ratio(norm_i, norm_j) / 100.0
            if similarity >= similarity_threshold:
                if len(norm_i) <= len(norm_j):
                    logger.debug(
                        "GAP dedup: dropping %s (dup of %s, sim=%.2f)",
                        t_i.segment_id, t_j.segment_id, similarity,
                    )
                    t_i.agreed_text = ""
                    removed += 1
                    break

                logger.debug(
                    "GAP dedup: dropping %s (dup of %s, sim=%.2f)",
                    t_j.segment_id, t_i.segment_id, similarity,
                )
                t_j.agreed_text = ""
                removed += 1

    return removed


def dedup_gap_against_all(
    triples: list[AlignedTriple],
    similarity_threshold: float = 0.85,
    partial_threshold: float = 0.90,
    length_ratio_cap: float = 0.7,
    min_text_len: int = 50,
) -> int:
    """Remove GAP blocks that duplicate content from ANY other block.

    Unlike ``dedup_gap_triples`` (which only checks GAP-vs-GAP within a
    local window), this function compares every GAP block against ALL other
    blocks regardless of classification or distance.  Two checks:

    1. **Near-equal**: ``token_set_ratio > similarity_threshold``
    2. **Containment**: ``partial_ratio > partial_threshold`` AND the shorter
       normalised text is less than ``length_ratio_cap`` × the longer one.

    Headings (``block_type == "heading"``) and very short texts are skipped
    to avoid false positives.
    """
    removed = 0

    # Pre-build candidate texts from all non-empty triples.
    # Each entry: (index, normalised_text, is_gap, raw_text)
    # NOTE: CONFLICT triples are excluded — their text is provisional
    # (not yet LLM-resolved) and comparing against it could wrongly
    # remove valid GAP content.
    candidates: list[tuple[int, str, bool, str]] = []
    for idx, t in enumerate(triples):
        if t.classification == CONFLICT:
            continue  # skip unresolved conflicts
        text = t.agreed_text
        if not text or not text.strip():
            continue
        norm = normalize_text(text)
        if not norm or len(norm) < min_text_len:
            continue
        # Skip headings
        blocks = _get_present_blocks(t)
        if blocks and all(b.block_type == "heading" for b in blocks):
            continue
        if not blocks and text.lstrip().startswith("#"):
            continue
        candidates.append((idx, norm, t.classification == GAP, text))

    # For each GAP candidate, check against every other candidate.
    gap_indices_to_blank: set[int] = set()
    for c_idx, (triple_idx, gap_norm, is_gap, gap_raw) in enumerate(candidates):
        if not is_gap:
            continue
        if triple_idx in gap_indices_to_blank:
            continue

        for o_idx, (other_triple_idx, other_norm, other_is_gap, other_raw) in enumerate(candidates):
            if c_idx == o_idx:
                continue
            if other_triple_idx in gap_indices_to_blank:
                continue

            # Near-equal check
            tsr = fuzz.token_set_ratio(gap_norm, other_norm) / 100.0
            near_equal_match = False
            if tsr >= similarity_threshold:
                # Numeric/citation guardrail: for containment (subset),
                # allow if shorter's values are a subset of longer's.
                gap_nums = _extract_numeric_tokens(gap_raw)
                other_nums = _extract_numeric_tokens(other_raw)
                gap_cites = _extract_citation_keys(gap_raw)
                other_cites = _extract_citation_keys(other_raw)
                shorter_nums = gap_nums if len(gap_norm) <= len(other_norm) else other_nums
                longer_nums = other_nums if len(gap_norm) <= len(other_norm) else gap_nums
                shorter_cites = gap_cites if len(gap_norm) <= len(other_norm) else other_cites
                longer_cites = other_cites if len(gap_norm) <= len(other_norm) else gap_cites
                if shorter_nums.issubset(longer_nums) and shorter_cites.issubset(longer_cites):
                    near_equal_match = True
                elif gap_nums == other_nums and gap_cites == other_cites:
                    near_equal_match = True

            if near_equal_match:
                # GAP vs non-GAP: always blank the GAP regardless of length
                if not other_is_gap:
                    gap_indices_to_blank.add(triple_idx)
                    break
                # GAP vs GAP: drop the shorter one
                if len(gap_norm) <= len(other_norm):
                    gap_indices_to_blank.add(triple_idx)
                    break
                gap_indices_to_blank.add(other_triple_idx)
                continue

            # Containment check: is the GAP a fragment of the other?
            shorter, longer = (gap_norm, other_norm) if len(gap_norm) <= len(other_norm) else (other_norm, gap_norm)
            if len(shorter) < len(longer) * length_ratio_cap:
                pr = fuzz.partial_ratio(shorter, longer) / 100.0
                if pr >= partial_threshold:
                    # GAP vs non-GAP: always blank the GAP
                    if not other_is_gap:
                        gap_indices_to_blank.add(triple_idx)
                        break
                    # GAP vs GAP: drop the shorter side
                    if len(gap_norm) <= len(other_norm):
                        gap_indices_to_blank.add(triple_idx)
                        break
                    gap_indices_to_blank.add(other_triple_idx)

    # Blank the identified duplicates
    for idx in gap_indices_to_blank:
        t = triples[idx]
        logger.info(
            "Global GAP dedup: dropping %s (text=%.60s...)",
            t.segment_id,
            (t.agreed_text or "")[:60],
        )
        t.agreed_text = ""
        removed += 1

    return removed


# ---------------------------------------------------------------------------
# Step 5: ASSEMBLE — Build final merged markdown
# ---------------------------------------------------------------------------

def assemble(
    triples: list[AlignedTriple],
    resolved_conflicts: dict[str, str],
) -> str:
    """Assemble final markdown and clean non-LLM blocks for output quality."""
    parts: list[str] = []

    for triple in triples:
        if triple.classification in (AGREE_EXACT, AGREE_NEAR, GAP):
            if triple.agreed_text:
                parts.append(clean_output_md(triple.agreed_text))
        elif triple.classification == CONFLICT:
            resolved = resolved_conflicts.get(triple.segment_id)
            if resolved:
                parts.append(resolved)
            else:
                # Fallback: use first available block text
                blocks = _get_present_blocks(triple)
                if blocks:
                    parts.append(clean_output_md(blocks[0].source_md or blocks[0].raw_text))

    return "\n\n".join(parts)


def dedup_assembled_paragraphs(
    text: str,
    similarity_threshold: float = 0.85,
    partial_threshold: float = 0.90,
    length_ratio_cap: float = 0.7,
    min_text_len: int = 50,
) -> tuple[str, int]:
    """Post-assembly global paragraph dedup — safety net for Layer 1.

    Splits assembled text on blank lines, does O(n²) pairwise comparison
    on qualifying paragraphs (>min_text_len chars, not headings), and removes
    the shorter duplicate.  Returns (cleaned_text, removed_count).
    """
    paragraphs = text.split("\n\n")
    removed_indices: set[int] = set()

    # Build list of (index, normalised, raw) for qualifying paragraphs
    qualifying: list[tuple[int, str, str]] = []
    for i, para in enumerate(paragraphs):
        stripped = para.strip()
        if not stripped or len(stripped) < min_text_len:
            continue
        # Skip headings
        if stripped.startswith("#"):
            continue
        # Skip table rows
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        qualifying.append((i, normalize_text(stripped), stripped))

    # O(n²) pairwise
    for a in range(len(qualifying)):
        idx_a, norm_a, raw_a = qualifying[a]
        if idx_a in removed_indices:
            continue
        for b in range(a + 1, len(qualifying)):
            idx_b, norm_b, raw_b = qualifying[b]
            if idx_b in removed_indices:
                continue

            # Near-equal check
            tsr = fuzz.token_set_ratio(norm_a, norm_b) / 100.0
            near_equal_match = False
            if tsr >= similarity_threshold:
                # Numeric/citation guardrail: for containment (subset),
                # allow if shorter's values are a subset of longer's.
                nums_a = _extract_numeric_tokens(raw_a)
                nums_b = _extract_numeric_tokens(raw_b)
                cites_a = _extract_citation_keys(raw_a)
                cites_b = _extract_citation_keys(raw_b)
                shorter_nums = nums_a if len(norm_a) <= len(norm_b) else nums_b
                longer_nums = nums_b if len(norm_a) <= len(norm_b) else nums_a
                shorter_cites = cites_a if len(norm_a) <= len(norm_b) else cites_b
                longer_cites = cites_b if len(norm_a) <= len(norm_b) else cites_a
                if shorter_nums.issubset(longer_nums) and shorter_cites.issubset(longer_cites):
                    near_equal_match = True
                elif nums_a == nums_b and cites_a == cites_b:
                    near_equal_match = True

            if near_equal_match:
                # Drop the shorter paragraph
                victim = idx_a if len(norm_a) <= len(norm_b) else idx_b
                logger.info(
                    "Post-assembly dedup: dropping paragraph %d (near-equal to %d, tsr=%.2f)",
                    victim, idx_a if victim == idx_b else idx_b, tsr,
                )
                removed_indices.add(victim)
                if victim == idx_a:
                    break
                continue

            # Containment check
            shorter_norm, longer_norm = (norm_a, norm_b) if len(norm_a) <= len(norm_b) else (norm_b, norm_a)
            shorter_idx = idx_a if len(norm_a) <= len(norm_b) else idx_b
            if len(shorter_norm) < len(longer_norm) * length_ratio_cap:
                pr = fuzz.partial_ratio(shorter_norm, longer_norm) / 100.0
                if pr >= partial_threshold:
                    logger.info(
                        "Post-assembly dedup: dropping paragraph %d (contained in %d, pr=%.2f)",
                        shorter_idx,
                        idx_b if shorter_idx == idx_a else idx_a,
                        pr,
                    )
                    removed_indices.add(shorter_idx)
                    if shorter_idx == idx_a:
                        break

    cleaned = "\n\n".join(
        para for i, para in enumerate(paragraphs) if i not in removed_indices
    )
    return cleaned, len(removed_indices)


_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def resolve_header_hierarchy(merged_md: str, llm: "LLM") -> str:
    """Resolve flat heading levels into a proper hierarchy via LLM.

    Extracts all heading lines, asks the LLM to classify each one, validates
    the response, and rewrites heading levels in-place.  On any validation
    failure the original markdown is returned unchanged.
    """
    from config import Config

    # 1. Extract headings with their line positions
    matches = list(_HEADING_LINE_RE.finditer(merged_md))
    if not matches:
        logger.info("Header hierarchy: no headings found, skipping")
        return merged_md

    # 2. Build payload for LLM
    lines = merged_md.split("\n")
    headers: list[dict] = []
    for idx, m in enumerate(matches):
        text = m.group(2).strip()
        # Content preview: grab ~100 chars of text following this heading
        line_end = m.end()
        preview = merged_md[line_end:line_end + 200].strip()
        # Trim to first ~100 meaningful chars (skip blank lines)
        preview_clean = " ".join(preview.split())[:100]
        headers.append({
            "index": idx,
            "text": text,
            "content_preview": preview_clean,
        })

    # 2b. Extract opening text (~1000 chars) for title detection
    opening_text = merged_md[:1000]

    logger.info(
        "Header hierarchy: extracted %d headings, calling LLM (model=%s, reasoning=%s)",
        len(headers), Config.HIERARCHY_LLM_MODEL, Config.HIERARCHY_LLM_REASONING,
    )

    # 3. Call LLM
    response = llm.resolve_header_hierarchy(
        headers,
        model=Config.HIERARCHY_LLM_MODEL,
        reasoning_effort=Config.HIERARCHY_LLM_REASONING,
        opening_text=opening_text,
    )

    # 4. Validate (includes text-fidelity check against what we sent)
    decisions = response.decisions
    detected_title = (response.detected_title or "").strip() or None
    original_levels = [len(m.group(1)) for m in matches]
    validation_error = _validate_hierarchy_decisions(
        decisions, len(headers), headers, original_levels,
        detected_title=detected_title,
    )
    if validation_error:
        logger.warning("Header hierarchy validation failed: %s — returning original markdown", validation_error)
        return merged_md

    # 5. Apply — build replacement map (match start → new line text)
    # Process in reverse order so character offsets stay valid
    result = merged_md
    for decision in sorted(decisions, key=lambda d: d.heading_index, reverse=True):
        match = matches[decision.heading_index]
        original_level = len(match.group(1))
        heading_text = match.group(2)

        if decision.action == "keep_level":
            logger.info(
                "  [%d] KEEP H%d: %s",
                decision.heading_index, original_level, heading_text[:80],
            )
            continue
        elif decision.action == "set_level":
            new_prefix = "#" * decision.new_level
            new_line = f"{new_prefix} {heading_text}"
            logger.info(
                "  [%d] SET H%d → H%d: %s",
                decision.heading_index, original_level, decision.new_level, heading_text[:80],
            )
            result = result[:match.start()] + new_line + result[match.end():]
        elif decision.action == "demote_to_text":
            logger.info(
                "  [%d] DEMOTE H%d → text: %s",
                decision.heading_index, original_level, heading_text[:80],
            )
            result = result[:match.start()] + heading_text + result[match.end():]

    # 6. Insert detected title as H1 if the LLM found one in the opening text
    if detected_title:
        title_stripped = detected_title.strip()
        # Try to find the title as a standalone line first
        lines = result.split("\n")
        inserted = False
        for i, line in enumerate(lines):
            if line.strip() == title_stripped:
                lines[i] = f"# {title_stripped}"
                logger.info("  TITLE MARKED as H1 (existing line): %s", title_stripped[:80])
                inserted = True
                break
        if inserted:
            result = "\n".join(lines)
        else:
            # Title exists only embedded in other text (e.g. citation line) —
            # prepend it as a new H1 at the top of the document.
            result = f"# {title_stripped}\n\n{result.lstrip()}"
            logger.info("  TITLE PREPENDED as H1: %s", title_stripped[:80])

    # Log summary
    actions = {"keep_level": 0, "set_level": 0, "demote_to_text": 0}
    for d in decisions:
        actions[d.action] = actions.get(d.action, 0) + 1
    logger.info(
        "Header hierarchy complete: %d headings (set_level=%d, demote=%d, keep=%d, title_detected=%s)",
        len(decisions), actions["set_level"], actions["demote_to_text"], actions["keep_level"],
        bool(detected_title),
    )

    return result


def _validate_hierarchy_decisions(
    decisions: list,
    expected_count: int,
    original_headers: list[dict] | None = None,
    original_levels: list[int] | None = None,
    detected_title: str | None = None,
) -> str | None:
    """Validate LLM hierarchy decisions. Returns error string or None if valid.

    Args:
        decisions: List of HeaderDecision from the LLM.
        expected_count: Number of headings we sent.
        original_headers: The header dicts we sent to the LLM (with 'text' keys).
            Used to verify the LLM didn't fabricate or modify heading text.
        original_levels: Original heading levels (1-6) from the markdown, indexed
            by heading position.  Used to resolve keep_level into an effective
            level for the H1 and jump checks.
        detected_title: If set, the LLM found the title in the opening text
            (not among headings), so 0 H1 headings is acceptable.
    """
    # Count check
    if len(decisions) != expected_count:
        return f"expected {expected_count} decisions, got {len(decisions)}"

    # Index uniqueness and completeness: must be exactly {0..n-1}
    indices = [d.heading_index for d in decisions]
    if sorted(indices) != list(range(expected_count)):
        seen = set()
        dupes = [i for i in indices if i in seen or seen.add(i)]
        if dupes:
            return f"duplicate heading_index values: {dupes}"
        return f"heading_index values {sorted(indices)} do not cover 0..{expected_count - 1}"

    # CRITICAL: Text fidelity check — the LLM must not modify heading text
    if original_headers:
        for d in decisions:
            if d.heading_index < 0 or d.heading_index >= len(original_headers):
                return f"heading_index {d.heading_index} out of range (0-{len(original_headers) - 1})"
            expected_text = original_headers[d.heading_index]["text"]
            if d.original_text.strip() != expected_text.strip():
                return (
                    f"LLM modified heading text at index {d.heading_index}: "
                    f"expected {expected_text!r}, got {d.original_text!r}"
                )

    # Validate new_level values for set_level actions
    for d in decisions:
        if d.action == "set_level":
            if d.new_level is None:
                return f"heading_index {d.heading_index}: set_level requires new_level but got None"
            if not (1 <= d.new_level <= 6):
                return f"heading_index {d.heading_index}: new_level {d.new_level} out of range (1-6)"

    # Build effective levels (resolving keep_level using original_levels)
    def _effective_level(d) -> int | None:
        if d.action == "set_level":
            return d.new_level
        if d.action == "keep_level" and original_levels:
            return original_levels[d.heading_index]
        return None  # unknown

    # H1 count: exactly 1 among headings, OR 0 if detected_title is set
    h1_count = sum(
        1 for d in decisions
        if _effective_level(d) == 1 and d.action != "demote_to_text"
    )
    if detected_title:
        # Title is in plain text — no heading should be H1
        if h1_count != 0:
            return f"detected_title is set but {h1_count} headings are also H1 (expected 0)"
    else:
        if h1_count != 1:
            return f"expected exactly 1 H1 (title), got {h1_count}"

    # No level jumps > 1 (among headings with known levels)
    # keep_level headings without original_levels reset tracking to avoid
    # false positives.
    prev_level = None
    for d in sorted(decisions, key=lambda x: x.heading_index):
        if d.action == "demote_to_text":
            continue
        level = _effective_level(d)
        if level is None:
            # Unknown level — reset so we don't compare across the gap
            prev_level = None
            continue
        if prev_level is not None and level > prev_level + 1:
            return f"level jump from {prev_level} to {level} (max allowed jump is 1)"
        prev_level = level

    # Demotion cap: < 50%
    demote_count = sum(1 for d in decisions if d.action == "demote_to_text")
    if expected_count > 0 and demote_count >= expected_count * 0.5:
        return f"too many demotions: {demote_count}/{expected_count} (>= 50%)"

    return None


def run_qa_gates(
    merged_md: str,
    similarity_threshold: float = 0.85,
    partial_threshold: float = 0.90,
    length_ratio_cap: float = 0.7,
    min_text_len: int = 50,
) -> dict:
    """Run post-assembly quality checks including **global** duplicate detection.

    The duplicate scan is now O(n²) over all paragraph pairs (not just
    adjacent) and includes containment checks, matching the thresholds used
    by the dedup layers.
    """
    results: dict[str, int | bool] = {}

    span_count = len(_SPAN_OPEN_RE.findall(merged_md))
    results["span_tag_count"] = span_count
    if span_count > 0:
        logger.warning("QA gate: found %d <span> tags in assembled output", span_count)

    comment_count = len(_HTML_COMMENT_OUTPUT_RE.findall(merged_md))
    results["html_comment_count"] = comment_count
    if comment_count > 0:
        logger.warning("QA gate: found %d HTML comments in assembled output", comment_count)

    # --- Global duplicate detection (replaces adjacent-only check) ----------
    paragraphs = [p.strip() for p in merged_md.split("\n\n") if p.strip()]

    # Build qualifying list (skip headings, tables, short paragraphs)
    qualifying: list[tuple[int, str, str]] = []  # (index, normalised, raw)
    for i, para in enumerate(paragraphs):
        if len(para) < min_text_len:
            continue
        if para.startswith("#"):
            continue
        if para.startswith("|") and para.endswith("|"):
            continue
        qualifying.append((i, normalize_text(para), para))

    global_dup_count = 0
    for a in range(len(qualifying)):
        idx_a, norm_a, raw_a = qualifying[a]
        for b in range(a + 1, len(qualifying)):
            idx_b, norm_b, raw_b = qualifying[b]

            # Near-equal
            tsr = fuzz.token_set_ratio(norm_a, norm_b) / 100.0
            near_equal_match = False
            if tsr >= similarity_threshold:
                # Numeric/citation guardrail: for containment (subset),
                # allow if shorter's values are a subset of longer's.
                nums_a = _extract_numeric_tokens(raw_a)
                nums_b = _extract_numeric_tokens(raw_b)
                cites_a = _extract_citation_keys(raw_a)
                cites_b = _extract_citation_keys(raw_b)
                shorter_nums = nums_a if len(norm_a) <= len(norm_b) else nums_b
                longer_nums = nums_b if len(norm_a) <= len(norm_b) else nums_a
                shorter_cites = cites_a if len(norm_a) <= len(norm_b) else cites_b
                longer_cites = cites_b if len(norm_a) <= len(norm_b) else cites_a
                if shorter_nums.issubset(longer_nums) and shorter_cites.issubset(longer_cites):
                    near_equal_match = True
                elif nums_a == nums_b and cites_a == cites_b:
                    near_equal_match = True

            if near_equal_match:
                logger.warning(
                    "QA gate: paragraphs %d and %d are %.0f%% similar (global duplicate)",
                    idx_a, idx_b, tsr * 100,
                )
                global_dup_count += 1
                continue

            # Containment
            shorter, longer = (norm_a, norm_b) if len(norm_a) <= len(norm_b) else (norm_b, norm_a)
            if len(shorter) < len(longer) * length_ratio_cap:
                pr = fuzz.partial_ratio(shorter, longer) / 100.0
                if pr >= partial_threshold:
                    shorter_idx = idx_a if len(norm_a) <= len(norm_b) else idx_b
                    longer_idx = idx_b if shorter_idx == idx_a else idx_a
                    logger.warning(
                        "QA gate: paragraph %d contained in %d (partial=%.0f%%, global duplicate)",
                        shorter_idx, longer_idx, pr * 100,
                    )
                    global_dup_count += 1

    results["global_duplicate_count"] = global_dup_count

    results["qa_passed"] = (span_count == 0 and comment_count == 0 and global_dup_count == 0)
    if results["qa_passed"]:
        logger.info("QA gates: all checks passed")
    else:
        logger.warning(
            "QA gates: %d issues detected (spans=%d, comments=%d, global_dupes=%d)",
            span_count + comment_count + global_dup_count,
            span_count, comment_count, global_dup_count,
        )

    return results


def _find_flanking_text(
    triples: list[AlignedTriple],
    conflict_index: int,
    direction: str,
) -> str:
    """Return nearest non-empty, non-CONFLICT agreed_text in the given direction."""
    if direction == "before":
        indices = range(conflict_index - 1, -1, -1)
    else:
        indices = range(conflict_index + 1, len(triples))

    for i in indices:
        triple = triples[i]
        if triple.classification != CONFLICT and triple.agreed_text and triple.agreed_text.strip():
            return triple.agreed_text
    return ""


def _truncate_to_token_budget(text: str, budget_chars: int = _FLANK_CHAR_BUDGET) -> str:
    """Truncate to a conservative character budget approximating token limits."""
    if len(text) <= budget_chars:
        return text
    return text[:budget_chars].rstrip() + "..."


def _build_flanking_context(
    triples: list[AlignedTriple],
    conflict_index: int,
) -> tuple[str, str]:
    """Build cleaned context_before/context_after around a conflict triple."""
    raw_before = _find_flanking_text(triples, conflict_index, "before")
    if raw_before:
        cleaned_before = clean_output_md(raw_before)
        if len(cleaned_before) > _FLANK_CHAR_BUDGET:
            cleaned_before = "..." + cleaned_before[-_FLANK_CHAR_BUDGET:].lstrip()
    else:
        cleaned_before = ""

    raw_after = _find_flanking_text(triples, conflict_index, "after")
    if raw_after:
        cleaned_after = _truncate_to_token_budget(clean_output_md(raw_after))
    else:
        cleaned_after = ""

    return cleaned_before, cleaned_after


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
# Audit entries — Per-decision log for curator review
# ---------------------------------------------------------------------------


def _build_audit_entries(
    triples: list[AlignedTriple],
    resolved_conflicts: dict[str, str],
    near_threshold: float = 0.92,
    levenshtein_threshold: float = 0.90,
) -> list[dict]:
    """Build audit entries for every non-AGREE_EXACT triple.

    Each entry records the classification, what each extractor produced,
    what text was chosen, and classification-specific details so curators
    can review pipeline decisions.
    """
    entries: list[dict] = []

    for idx, triple in enumerate(triples):
        if triple.classification == AGREE_EXACT:
            continue

        blocks = _get_present_blocks(triple)
        if not blocks:
            continue

        block_type = blocks[0].block_type
        sources_present = [b.source for b in blocks]
        extractor_texts = {b.source: b.source_md or b.raw_text for b in blocks}

        entry: dict = {
            "segment_id": triple.segment_id,
            "classification": triple.classification,
            "block_type": block_type,
            "sources_present": sources_present,
            "extractor_texts": extractor_texts,
            "chosen_text": "",
            "chosen_source": "",
            "details": {},
        }

        if triple.classification == AGREE_NEAR:
            normalized_texts = [b.normalized_text for b in blocks]
            raw_texts = [b.raw_text for b in blocks]
            near_pair = None
            best_score = 0.0

            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    token_ratio = fuzz.token_set_ratio(
                        normalized_texts[i], normalized_texts[j],
                    ) / 100.0
                    max_len = max(len(normalized_texts[i]), len(normalized_texts[j]), 1)
                    lev_dist = Levenshtein.distance(
                        normalized_texts[i], normalized_texts[j],
                    )
                    lev_sim = 1.0 - (lev_dist / max_len)

                    if token_ratio >= near_threshold and lev_sim >= levenshtein_threshold:
                        nums_i = _extract_numeric_tokens(raw_texts[i])
                        nums_j = _extract_numeric_tokens(raw_texts[j])
                        cites_i = _extract_citation_keys(raw_texts[i])
                        cites_j = _extract_citation_keys(raw_texts[j])
                        if nums_i == nums_j and cites_i == cites_j:
                            near_pair = (i, j)
                            best_score = token_ratio
                            break
                if near_pair is not None:
                    break

            preferred_source = _SOURCE_PREFERENCE.get(block_type, "marker")
            if near_pair is not None:
                agreeing = [blocks[near_pair[0]], blocks[near_pair[1]]]
                chosen_source = preferred_source
                if not any(b.source == preferred_source for b in agreeing):
                    chosen_source = agreeing[0].source
                entry["details"] = {
                    "agreeing_pair": [agreeing[0].source, agreeing[1].source],
                    "similarity_score": round(best_score, 4),
                    "source_preference_rule": f"{block_type} \u2192 {preferred_source}",
                }
            else:
                chosen_source = blocks[0].source

            entry["chosen_text"] = triple.agreed_text or ""
            entry["chosen_source"] = chosen_source

        elif triple.classification == GAP:
            sole_source = blocks[0].source
            deduped = not (triple.agreed_text and triple.agreed_text.strip())
            entry["chosen_text"] = triple.agreed_text or ""
            entry["chosen_source"] = sole_source
            entry["details"] = {
                "sole_source": sole_source,
                "deduped": deduped,
            }

        elif triple.classification == CONFLICT:
            resolved_text = resolved_conflicts.get(triple.segment_id)
            context_before, context_after = _build_flanking_context(triples, idx)

            if resolved_text is not None:
                entry["chosen_text"] = resolved_text
                entry["chosen_source"] = "llm"
            elif blocks:
                entry["chosen_text"] = clean_output_md(
                    blocks[0].source_md or blocks[0].raw_text,
                )
                entry["chosen_source"] = blocks[0].source

            entry["details"] = {
                "context_before": context_before,
                "context_after": context_after,
                "llm_resolved": resolved_text is not None,
            }

        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Orchestrator — Top-level entry point
# ---------------------------------------------------------------------------

def merge_with_consensus(
    grobid_md: str,
    docling_md: str,
    marker_md: str,
    llm: "LLM",
) -> tuple[str | None, dict, list]:
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
        return None, metrics, []

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

    # Step 3b: remove near-duplicate GAP blocks (local window)
    gap_dups_removed = dedup_gap_triples(triples)
    if gap_dups_removed > 0:
        logger.info("Consensus pipeline: removed %d local GAP duplicates", gap_dups_removed)

    # Step 3c: remove GAP blocks that duplicate ANY other block (global)
    gap_cross_removed = dedup_gap_against_all(triples)
    if gap_cross_removed > 0:
        logger.info("Consensus pipeline: removed %d cross-class GAP duplicates", gap_cross_removed)

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
        return None, metrics, []

    # Step 5: Resolve conflicts via LLM
    conflict_triples = [t for t in triples if t.classification == CONFLICT]
    resolved_conflicts: dict[str, str] = {}

    if conflict_triples:
        logger.info("Consensus pipeline: resolving %d conflicts via LLM...", len(conflict_triples))
        segment_id_to_index = {t.segment_id: i for i, t in enumerate(triples)}

        conflict_bundles = []
        for t in conflict_triples:
            triple_index = segment_id_to_index[t.segment_id]
            context_before, context_after = _build_flanking_context(triples, triple_index)
            present_blocks = _get_present_blocks(t)
            bundle = {
                "segment_id": t.segment_id,
                "block_type": present_blocks[0].block_type if present_blocks else "paragraph",
                "context_before": context_before,
                "context_after": context_after,
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
            return None, metrics, []

    # Build audit entries (after classification, dedup, and conflict resolution)
    audit_entries = _build_audit_entries(
        triples, resolved_conflicts,
        near_threshold=Config.CONSENSUS_NEAR_THRESHOLD,
        levenshtein_threshold=Config.CONSENSUS_LEVENSHTEIN_THRESHOLD,
    )

    # Step 6: Assemble
    logger.info("Consensus pipeline: assembling final document...")
    merged_md = assemble(triples, resolved_conflicts)

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

    # Step 7: QA gates (global duplicate detection)
    qa_results = run_qa_gates(merged_md)

    metrics = compute_metrics(triples, alignment_confidence, False, None)
    metrics["gap_cross_dedup_removed"] = gap_cross_removed
    metrics["assembled_dedup_removed"] = assembled_dups_removed
    metrics["qa"] = qa_results

    # Step 8: Fail-hard on surviving global duplicates
    surviving_dupes = qa_results.get("global_duplicate_count", 0)
    if surviving_dupes > 0 and Config.CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES:
        logger.warning(
            "Consensus pipeline: %d global duplicates survived both dedup layers "
            "— triggering fallback (CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES=True)",
            surviving_dupes,
        )
        metrics["fallback_triggered"] = True
        metrics["fallback_reason"] = "global_duplicates"
        return None, metrics, audit_entries

    logger.info(
        "Consensus pipeline complete: %d blocks, %d conflicts resolved, "
        "~%d tokens saved, %d GAP cross-dedup, %d post-assembly dedup",
        metrics["total_blocks"],
        metrics["conflict"],
        metrics["tokens_saved_estimate"],
        gap_cross_removed,
        assembled_dups_removed,
    )

    return merged_md, metrics, audit_entries
