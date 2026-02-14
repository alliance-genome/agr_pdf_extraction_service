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
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import mistune
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from app.services.degradation_metrics import build_degradation_metrics

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
_SPAN_REF_RE = re.compile(r"<span id=['\"][^'\"]*['\"]>(.*?)</span>", re.DOTALL)
_HTML_COMMENT_OUTPUT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"  +")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)]*\)")
_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# Flanking context budget (~150 tokens, conservative 4 chars/token estimate).
_CHARS_PER_TOKEN = 4
_FLANK_TOKEN_BUDGET = 150
_FLANK_CHAR_BUDGET = _FLANK_TOKEN_BUDGET * _CHARS_PER_TOKEN

# Layered conflict resolver defaults
_LAYERED_MEDIUM_SIM_THRESHOLD = 0.60

# Regexes for comparison-only normalization (false conflict reduction)
_EN_DASH_RE = re.compile(r'[\u2013\u2014]')            # en-dash, em-dash → hyphen
_CITE_SPACE_OPEN_RE = re.compile(r'\[\s+')              # "[ 1" → "[1"
_CITE_SPACE_CLOSE_RE = re.compile(r'\s+\]')             # "1 ]" → "1]"
_CITE_COMMA_SPACE_RE = re.compile(r'(\d)\s*,\s*(\d)')   # "7, 8" → "7,8"
_MULTI_SPACE_NORM_RE = re.compile(r' {2,}')              # collapse multiple spaces


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for similarity comparison only (not for output).

    Handles the most common false-conflict triggers found in the 16-paper
    test batch:
    - Citation dash variants (en-dash/em-dash → hyphen)
    - Citation spacing variants
    - Multiple spaces → single space
    - Unicode NFC normalization
    """
    text = unicodedata.normalize("NFC", text)
    text = _EN_DASH_RE.sub('-', text)
    text = _CITE_SPACE_OPEN_RE.sub('[', text)
    text = _CITE_SPACE_CLOSE_RE.sub(']', text)
    text = _CITE_COMMA_SPACE_RE.sub(r'\1,\2', text)
    text = _MULTI_SPACE_NORM_RE.sub(' ', text)
    text = text.strip()
    return text


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


def normalize_extractor_output(markdown_text: str) -> str:
    """Normalize full extractor markdown before block parsing.

    This removes extractor-specific formatting artifacts at the source layer
    so alignment/classification operates on cleaner inputs.
    """
    text = unicodedata.normalize("NFKC", markdown_text or "")
    text = _SPAN_REF_RE.sub(r"\1", text)
    text = _HTML_COMMENT_OUTPUT_RE.sub("", text)
    text = _IMAGE_REF_RE.sub("", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    # Common Docling ligature artifacts.
    text = text.replace("/uniFB01", "fi").replace("/uniFB02", "fl")
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


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
_REFERENCE_NUM_RE = re.compile(
    r"(?:fig(?:ure)?|table|eq(?:uation)?|sec(?:tion)?|ref(?:erence)?)\.?\s*(\d+)",
    re.IGNORECASE,
)
_NUMERIC_CITATION_ONLY_RE = re.compile(r"^\[\d+(?:\s*[-–,]\s*\d+)*\]$")

# Per-block-type source preference for AGREE_NEAR
_SOURCE_PREFERENCE = {
    "heading": "grobid",
    "citation_list": "grobid",
    "table": "docling",
    "figure_ref": "marker",
    "paragraph": "marker",
    "equation": "docling",
}
_TEXTUAL_BLOCK_TYPES = {"heading", "paragraph", "figure_ref", "citation_list"}
_STRUCTURED_BLOCK_TYPES = {"table", "equation"}


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


def _extract_reference_numbers(text: str) -> set[str]:
    """Extract numbers used in local references (Figure/Table/Section/etc.)."""
    cleaned = _HTML_TAG_RE.sub("", text)
    return set(_REFERENCE_NUM_RE.findall(cleaned))


def _is_minor_numeric_citation_delta(delta: set[str], max_delta: int = 1) -> bool:
    """Allow only tiny citation deltas with numeric bracket keys."""
    if not delta:
        return True
    if len(delta) > max_delta:
        return False
    return all(_NUMERIC_CITATION_ONLY_RE.match(key) for key in delta)


def _allow_reference_variance_exception(
    block_type: str,
    token_ratio: float,
    lev_sim: float,
    alignment_confidence: float,
    raw_i: str,
    raw_j: str,
    nums_i: set[str],
    nums_j: set[str],
    cites_i: set[str],
    cites_j: set[str],
) -> bool:
    """Allow AGREE_NEAR for likely alignment noise in references.

    This is intentionally conservative:
    - Applies only to textual blocks (not tables/equations/citation lists)
    - Requires high textual confidence
    - Allows only tiny numeric/citation deltas tied to Figure/Table/Section refs
    """
    if block_type not in {"heading", "paragraph", "figure_ref"}:
        return False
    if token_ratio < 0.97 or lev_sim < 0.95:
        return False
    if alignment_confidence < 0.55:
        return False

    numeric_delta = nums_i.symmetric_difference(nums_j)
    citation_delta = cites_i.symmetric_difference(cites_j)

    if len(numeric_delta) > 2 or len(citation_delta) > 1:
        return False
    if any("." in n for n in numeric_delta):
        return False

    if numeric_delta:
        ref_numbers = _extract_reference_numbers(raw_i) | _extract_reference_numbers(raw_j)
        if not numeric_delta.issubset(ref_numbers):
            return False
    if citation_delta and not _is_minor_numeric_citation_delta(citation_delta):
        return False

    return True


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
        normalized_texts = [_normalize_for_comparison(b.normalized_text) for b in blocks]
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
                    # Check numeric/citation guardrail (normalized for comparison)
                    norm_raw_i = _normalize_for_comparison(raw_texts[i])
                    norm_raw_j = _normalize_for_comparison(raw_texts[j])
                    nums_i = _extract_numeric_tokens(norm_raw_i)
                    nums_j = _extract_numeric_tokens(norm_raw_j)
                    cites_i = _extract_citation_keys(norm_raw_i)
                    cites_j = _extract_citation_keys(norm_raw_j)

                    if nums_i == nums_j and cites_i == cites_j:
                        near_pair = (i, j)
                        break

                    if _allow_reference_variance_exception(
                        block_type=block_type,
                        token_ratio=token_ratio,
                        lev_sim=lev_sim,
                        alignment_confidence=triple.confidence,
                        raw_i=raw_texts[i],
                        raw_j=raw_texts[j],
                        nums_i=nums_i,
                        nums_j=nums_j,
                        cites_i=cites_i,
                        cites_j=cites_j,
                    ):
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

                    diag_norm_i = _normalize_for_comparison(raw_texts[i])
                    diag_norm_j = _normalize_for_comparison(raw_texts[j])
                    nums_i = _extract_numeric_tokens(diag_norm_i)
                    nums_j = _extract_numeric_tokens(diag_norm_j)
                    if nums_i != nums_j:
                        reasons.append(f"numeric_diff={nums_i.symmetric_difference(nums_j)}")

                    cites_i = _extract_citation_keys(diag_norm_i)
                    cites_j = _extract_citation_keys(diag_norm_j)
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

def _triple_block_family(triple: AlignedTriple) -> str:
    """Map a triple to a coarse block family for guard calibration."""
    blocks = _get_present_blocks(triple)
    if not blocks:
        return "unknown"

    block_types = {b.block_type for b in blocks}
    if block_types & _STRUCTURED_BLOCK_TYPES:
        return "structured"
    if block_types & _TEXTUAL_BLOCK_TYPES:
        return "textual"
    return "textual"


def _triple_source_count(triple: AlignedTriple) -> int:
    """Count how many extractors produced a block for this triple."""
    return sum(1 for b in (triple.grobid_block, triple.docling_block, triple.marker_block) if b is not None)


def _compute_conflict_telemetry(
    triples: list[AlignedTriple],
    conflict_ratio_threshold: float = 0.4,
    localized_conflict_span_max: float = 0.35,
    localized_conflict_relief: float = 0.15,
    localized_conflict_max_blocks: int = 25,
) -> dict:
    """Compute guard telemetry for calibration and adaptive thresholding.

    Two-source conflicts (where median-source is degenerate and we escalate
    to a single LLM call) are excluded from the conflict ratios that drive
    the nuclear full-LLM fallback.  They're still tracked for transparency.
    """
    total = len(triples)
    num_gap = sum(1 for t in triples if t.classification == GAP)
    denominator = total - num_gap
    num_conflict_all = sum(1 for t in triples if t.classification == CONFLICT)
    # Two-source conflicts are "handleable" — they just need one LLM call
    # to pick the better text, so they shouldn't trip the nuclear fallback.
    num_two_source_conflict = sum(
        1 for t in triples
        if t.classification == CONFLICT and _triple_source_count(t) < 3
    )
    num_conflict = num_conflict_all - num_two_source_conflict
    conflict_ratio = (num_conflict / denominator) if denominator > 0 else 0.0

    textual_denominator = 0
    textual_conflict = 0
    structured_denominator = 0
    structured_conflict = 0
    non_gap_triples = [t for t in triples if t.classification != GAP]
    conflict_positions: list[int] = []

    for idx, triple in enumerate(non_gap_triples):
        is_two_source = _triple_source_count(triple) < 3
        family = _triple_block_family(triple)
        if family == "structured":
            structured_denominator += 1
            if triple.classification == CONFLICT and not is_two_source:
                structured_conflict += 1
                conflict_positions.append(idx)
        elif family == "textual":
            textual_denominator += 1
            if triple.classification == CONFLICT and not is_two_source:
                textual_conflict += 1
                conflict_positions.append(idx)
        elif triple.classification == CONFLICT and not is_two_source:
            conflict_positions.append(idx)

    textual_conflict_ratio = (
        textual_conflict / textual_denominator if textual_denominator > 0 else 0.0
    )
    structured_conflict_ratio = (
        structured_conflict / structured_denominator if structured_denominator > 0 else 0.0
    )

    conflict_span_ratio = 0.0
    conflicts_localized = False
    adaptive_conflict_ratio_threshold = conflict_ratio_threshold
    if conflict_positions and denominator > 0:
        conflict_span = (max(conflict_positions) - min(conflict_positions) + 1)
        # Span over all triples (including GAPs) captures whether conflicts are
        # concentrated in one region of the full document timeline.
        conflict_span_ratio = conflict_span / max(total, 1)
        conflicts_localized = (
            len(conflict_positions) >= 2
            and len(conflict_positions) <= max(1, int(localized_conflict_max_blocks))
            and conflict_span_ratio <= max(0.0, float(localized_conflict_span_max))
        )
        if conflicts_localized:
            adaptive_conflict_ratio_threshold = min(
                0.95, conflict_ratio_threshold + max(0.0, float(localized_conflict_relief)),
            )

    return {
        "conflict_ratio": round(conflict_ratio, 4),
        "conflict_ratio_textual": round(textual_conflict_ratio, 4),
        "conflict_ratio_structured": round(structured_conflict_ratio, 4),
        "non_gap_blocks": denominator,
        "conflict_blocks": num_conflict,
        "conflict_blocks_two_source": num_two_source_conflict,
        "conflict_blocks_total": num_conflict_all,
        "textual_blocks": textual_denominator,
        "structured_blocks": structured_denominator,
        "conflicts_localized": conflicts_localized,
        "conflict_span_ratio": round(conflict_span_ratio, 4),
        "adaptive_conflict_ratio_threshold": round(adaptive_conflict_ratio_threshold, 4),
    }


def check_guards(
    triples: list[AlignedTriple],
    alignment_confidence: float,
    conflict_ratio_threshold: float = 0.4,
    alignment_confidence_threshold: float = 0.5,
    textual_conflict_ratio_threshold: float = 0.4,
    structured_conflict_ratio_threshold: float = 0.85,
    localized_conflict_span_max: float = 0.35,
    localized_conflict_relief: float = 0.15,
    localized_conflict_max_blocks: int = 25,
) -> tuple[bool, str | None]:
    """
    Check whether the consensus pipeline should fall back to full-LLM merge.

    Returns (should_fallback, reason).
    """
    if not triples:
        return True, "no_blocks"

    telemetry = _compute_conflict_telemetry(
        triples,
        conflict_ratio_threshold=conflict_ratio_threshold,
        localized_conflict_span_max=localized_conflict_span_max,
        localized_conflict_relief=localized_conflict_relief,
        localized_conflict_max_blocks=localized_conflict_max_blocks,
    )

    if telemetry["conflict_ratio_textual"] > textual_conflict_ratio_threshold:
        return True, "conflict_ratio_textual"

    if (
        telemetry["structured_blocks"] > 0
        and telemetry["conflict_ratio_structured"] > structured_conflict_ratio_threshold
    ):
        return True, "conflict_ratio_structured"

    if telemetry["conflict_ratio"] > telemetry["adaptive_conflict_ratio_threshold"]:
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
            # Check if this segment was resolved by zone-based LLM resolution
            if triple.segment_id in resolved_conflicts:
                resolved_text = resolved_conflicts[triple.segment_id]
                if resolved_text:
                    parts.append(resolved_text)
                elif triple.classification == GAP:
                    pass  # LLM explicitly dropped this gap — do not include
                elif triple.agreed_text:
                    # Non-GAP (e.g. AGREE_NEAR) with empty LLM result — keep agreed_text
                    parts.append(clean_output_md(triple.agreed_text))
            elif triple.agreed_text:
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
# Conflict zone grouping — group adjacent conflicts for LLM resolution
# ---------------------------------------------------------------------------


def _build_conflict_zones(
    triples: list[AlignedTriple],
    flanking_count: int = 2,
) -> list[dict]:
    """Group non-AGREE_EXACT segments into zones bounded by AGREE_EXACT flanking context.

    Starting from each CONFLICT, the zone expands outward through any segment
    that is NOT AGREE_EXACT (i.e. CONFLICT, GAP, AGREE_NEAR all become part of
    the zone core).  Expansion stops when it hits an AGREE_EXACT segment or the
    document boundary.  Up to *flanking_count* AGREE_EXACT segments on each side
    are included as read-only context.

    AGREE_NEAR segments in the core are sent with all extractor versions so the
    LLM can pick or merge the cleanest text.

    Returns a list of zone dicts, each with:
        zone_id, context_before, segments, context_after, triple_indices
    """
    n = len(triples)
    if n == 0:
        return []

    is_exact = [t.classification == AGREE_EXACT for t in triples]

    # 1. Find all CONFLICT indices — these seed the zones.
    conflict_indices = [i for i in range(n) if triples[i].classification == CONFLICT]
    if not conflict_indices:
        return []

    # 2. For each conflict index, expand outward until hitting AGREE_EXACT or
    #    document boundary.  Record the (core_start, core_end) range.
    #    Then merge overlapping / adjacent ranges.
    raw_ranges: list[tuple[int, int]] = []
    for ci in conflict_indices:
        # Expand left: stop at first AGREE_EXACT (exclusive)
        left = ci
        while left > 0 and not is_exact[left - 1]:
            left -= 1
        # Expand right: stop at first AGREE_EXACT (exclusive)
        right = ci
        while right < n - 1 and not is_exact[right + 1]:
            right += 1
        raw_ranges.append((left, right))

    # Merge overlapping/adjacent ranges
    raw_ranges.sort()
    merged_ranges: list[tuple[int, int]] = [raw_ranges[0]]
    for start, end in raw_ranges[1:]:
        prev_start, prev_end = merged_ranges[-1]
        if start <= prev_end + 1:
            merged_ranges[-1] = (prev_start, max(prev_end, end))
        else:
            merged_ranges.append((start, end))

    # 3. Build zone dicts with flanking AGREE_EXACT context.
    result: list[dict] = []
    for zone_num, (core_start, core_end) in enumerate(merged_ranges):
        core_indices = list(range(core_start, core_end + 1))

        # Flanking context: up to flanking_count AGREE_EXACT segments on each side
        ctx_before_indices = []
        probe = core_start - 1
        while probe >= 0 and len(ctx_before_indices) < flanking_count:
            if is_exact[probe]:
                ctx_before_indices.insert(0, probe)
            probe -= 1

        ctx_after_indices = []
        probe = core_end + 1
        while probe < n and len(ctx_after_indices) < flanking_count:
            if is_exact[probe]:
                ctx_after_indices.append(probe)
            probe += 1

        # Build context_before entries
        context_before = []
        for idx in ctx_before_indices:
            t = triples[idx]
            context_before.append({
                "segment_id": t.segment_id,
                "text": t.agreed_text or "",
                "status": "agreed",
            })

        # Build core segments — everything non-AGREE_EXACT gets its full treatment
        segments = []
        for idx in core_indices:
            t = triples[idx]
            if t.classification == CONFLICT:
                blocks = _get_present_blocks(t)
                seg_entry: dict = {
                    "segment_id": t.segment_id,
                    "status": "conflict",
                    "block_type": blocks[0].block_type if blocks else "paragraph",
                    "grobid": (t.grobid_block.source_md or t.grobid_block.raw_text) if t.grobid_block else "",
                    "docling": (t.docling_block.source_md or t.docling_block.raw_text) if t.docling_block else "",
                    "marker": (t.marker_block.source_md or t.marker_block.raw_text) if t.marker_block else "",
                }
                segments.append(seg_entry)
            elif t.classification == GAP:
                blocks = _get_present_blocks(t)
                gap_source = blocks[0].source if blocks else "unknown"
                segments.append({
                    "segment_id": t.segment_id,
                    "status": "gap",
                    "gap_source": gap_source,
                    "text": t.agreed_text or "",
                })
            elif t.classification == AGREE_NEAR:
                # Near-agree: include all extractor versions so LLM can pick cleanest
                blocks = _get_present_blocks(t)
                segments.append({
                    "segment_id": t.segment_id,
                    "status": "near_agree",
                    "block_type": blocks[0].block_type if blocks else "paragraph",
                    "grobid": (t.grobid_block.source_md or t.grobid_block.raw_text) if t.grobid_block else "",
                    "docling": (t.docling_block.source_md or t.docling_block.raw_text) if t.docling_block else "",
                    "marker": (t.marker_block.source_md or t.marker_block.raw_text) if t.marker_block else "",
                    "current_choice": t.agreed_text or "",
                })
            else:
                # AGREE_EXACT inside the core — shouldn't happen given algorithm,
                # but handle defensively
                segments.append({
                    "segment_id": t.segment_id,
                    "status": "agreed",
                    "text": t.agreed_text or "",
                })

        # Build context_after entries
        context_after = []
        for idx in ctx_after_indices:
            t = triples[idx]
            context_after.append({
                "segment_id": t.segment_id,
                "text": t.agreed_text or "",
                "status": "agreed",
            })

        result.append({
            "zone_id": f"zone_{zone_num}",
            "context_before": context_before,
            "segments": segments,
            "context_after": context_after,
            "triple_indices": core_indices,
        })

    return result


# ---------------------------------------------------------------------------
# Layered conflict resolution (median-source + LLM batch)
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
    zone: dict,
    triples: list[AlignedTriple],
    resolved: dict[str, str],
    metadata: dict[str, dict],
    source_texts: dict[str, str],
    max_pair_sim: float,
    llm: "LLM",
    method_prefix: str = "llm_conflict",
) -> None:
    """Three-tier rescue strategy for a segment that zone resolution returned empty.

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
        zone=zone,
        neighboring_resolved=neighboring_resolved,
        extra_flanking=extra_flanking,
        model=Config.LLM_MODEL_RESCUE,
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
                "zone_id": zone["zone_id"],
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
                "zone_id": zone["zone_id"],
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
            "zone_id": zone["zone_id"],
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


def _resolve_conflicts_layered(
    conflict_zones: list[dict],
    triples: list[AlignedTriple],
    llm: "LLM",
    medium_similarity_threshold: float = _LAYERED_MEDIUM_SIM_THRESHOLD,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Resolve conflicts with layered meta-consensus using conflict zones.

    Layer 1: Median-source selection for 3+-source conflict segments with
             medium+ similarity (picks best whole source text verbatim).
    Layer 2: Zone-based LLM resolution for remaining conflicts. Pre-resolved
             segments are included as read-only context so the LLM sees the
             full picture within each zone.
    """
    resolved: dict[str, str] = {}
    metadata: dict[str, dict] = {}
    llm_zones: list[dict] = []  # zones that still need LLM resolution

    # Build a quick lookup from segment_id to triple
    seg_id_to_triple = {t.segment_id: t for t in triples}

    for zone in conflict_zones:
        zone_needs_llm = False

        for seg in zone["segments"]:
            if seg["status"] != "conflict":
                continue

            seg_id = seg["segment_id"]
            source_texts = _bundle_source_texts(seg)
            source_values = list(source_texts.values())
            pair_sims = _pairwise_similarities(source_values)
            max_pair_sim = max(pair_sims) if pair_sims else 0.0
            n_sources = len(source_texts)

            # Layer 1: Median-source for 3+-source conflicts with medium+ similarity
            if n_sources >= 3 and max_pair_sim >= medium_similarity_threshold:
                try:
                    resolved_text, confidence, chosen_extractor = _resolve_conflict_median_source(seg)
                    resolved[seg_id] = resolved_text.strip()
                    metadata[seg_id] = {
                        "method": "median_source",
                        "chosen_source": chosen_extractor,
                        "confidence": round(float(confidence), 4),
                        "sources_agreeing": sorted(source_texts.keys()),
                        "max_pair_similarity": round(max_pair_sim, 4),
                    }
                    # Mark as pre_resolved in the segment so LLM sees it as context
                    seg["status"] = "pre_resolved"
                    seg["text"] = resolved_text.strip()
                    continue
                except Exception as exc:
                    logger.warning("Layer 1 median-source failed for %s: %s", seg_id, exc)

            # This segment needs LLM
            zone_needs_llm = True
            if n_sources < 3:
                logger.info("Queuing %s for LLM zone resolution (2-source conflict)", seg_id)
            else:
                logger.info(
                    "Queuing %s for LLM zone resolution (max_pair_sim=%.3f < %.3f threshold)",
                    seg_id, max_pair_sim, medium_similarity_threshold,
                )

        # Also check for gap and near_agree segments that need resolution
        for seg in zone["segments"]:
            if seg["status"] in ("gap", "near_agree"):
                zone_needs_llm = True

        if zone_needs_llm:
            llm_zones.append(zone)

    # Layer 2: Resolve zones via LLM (parallel, one API call per zone)
    if llm_zones:
        logger.info("Resolving %d zone(s) via LLM", len(llm_zones))
        zone_results, unresolved_ids = llm.resolve_conflict_zones(llm_zones)
        if unresolved_ids:
            logger.warning(
                "Zone resolution left %d unresolved segment(s) — rescue will handle: %s",
                len(unresolved_ids), sorted(unresolved_ids),
            )

        for zone in llm_zones:
            for seg in zone["segments"]:
                seg_id = seg["segment_id"]
                if seg["status"] == "pre_resolved" or seg["status"] == "agreed":
                    continue

                resolved_text = (zone_results.get(seg_id, "") or "").strip()

                if seg["status"] == "conflict":
                    source_texts = _bundle_source_texts(seg)
                    source_values = list(source_texts.values())
                    pair_sims = _pairwise_similarities(source_values)
                    max_pair_sim = max(pair_sims) if pair_sims else 0.0

                    if not resolved_text:
                        # Three-tier rescue strategy instead of raising ValueError
                        _rescue_segment(
                            seg_id, seg, zone, triples, resolved, metadata,
                            source_texts, max_pair_sim, llm,
                            method_prefix="llm_conflict",
                        )
                        continue

                    resolved[seg_id] = resolved_text
                    metadata[seg_id] = {
                        "method": "llm_conflict",
                        "chosen_source": "llm",
                        "confidence": round(_mean_similarity_to_sources(resolved_text, source_texts), 4),
                        "sources_agreeing": sorted(source_texts.keys()),
                        "max_pair_similarity": round(max_pair_sim, 4),
                        "zone_id": zone["zone_id"],
                    }

                elif seg["status"] == "near_agree":
                    source_texts = _bundle_source_texts(seg)
                    source_values = list(source_texts.values())
                    pair_sims = _pairwise_similarities(source_values)
                    max_pair_sim = max(pair_sims) if pair_sims else 0.0

                    if not resolved_text:
                        # Three-tier rescue strategy instead of raising ValueError
                        _rescue_segment(
                            seg_id, seg, zone, triples, resolved, metadata,
                            source_texts, max_pair_sim, llm,
                            method_prefix="llm_near_agree",
                        )
                        continue

                    resolved[seg_id] = resolved_text
                    metadata[seg_id] = {
                        "method": "llm_near_agree",
                        "chosen_source": "llm",
                        "confidence": round(_mean_similarity_to_sources(resolved_text, source_texts), 4),
                        "sources_agreeing": sorted(source_texts.keys()),
                        "max_pair_similarity": round(max_pair_sim, 4),
                        "zone_id": zone["zone_id"],
                    }

                elif seg["status"] == "gap":
                    # GAP segments: LLM may return empty string to drop, or cleaned text
                    if seg_id in zone_results:
                        resolved[seg_id] = zone_results[seg_id]
                        metadata[seg_id] = {
                            "method": "llm_gap",
                            "chosen_source": "llm",
                            "confidence": 0.0,
                            "sources_agreeing": [],
                            "max_pair_similarity": 0.0,
                            "zone_id": zone["zone_id"],
                        }

    return resolved, metadata


# ---------------------------------------------------------------------------
# Step 6: METRICS — Compute pipeline statistics
# ---------------------------------------------------------------------------

def compute_metrics(
    triples: list[AlignedTriple],
    alignment_confidence: float,
    fallback_triggered: bool,
    fallback_reason: str | None,
    guard_telemetry: dict | None = None,
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

    metrics = {
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
    if guard_telemetry:
        metrics.update({
            "conflict_ratio_textual": guard_telemetry.get("conflict_ratio_textual", 0.0),
            "conflict_ratio_structured": guard_telemetry.get("conflict_ratio_structured", 0.0),
            "conflicts_localized": guard_telemetry.get("conflicts_localized", False),
            "conflict_span_ratio": guard_telemetry.get("conflict_span_ratio", 0.0),
            "adaptive_conflict_ratio_threshold": guard_telemetry.get(
                "adaptive_conflict_ratio_threshold", metrics["conflict_ratio"],
            ),
        })
    return metrics


# ---------------------------------------------------------------------------
# Audit entries — Per-decision log for curator review
# ---------------------------------------------------------------------------


def _build_audit_entries(
    triples: list[AlignedTriple],
    resolved_conflicts: dict[str, str],
    resolution_metadata: dict[str, dict] | None = None,
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
            resolution_details = (resolution_metadata or {}).get(triple.segment_id, {})
            resolved_text = resolved_conflicts.get(triple.segment_id)

            resolution_method = resolution_details.get("method", "")
            if resolved_text is not None and resolution_method.startswith("llm_near_agree"):
                # LLM-resolved near_agree (zone-based, rescue, or fallback)
                entry["chosen_text"] = resolved_text
                entry["chosen_source"] = resolution_details.get("chosen_source", "llm")
                entry["details"] = {
                    "zone_id": resolution_details.get("zone_id", ""),
                    "llm_resolved": True,
                    "resolution_method": resolution_method,
                    "resolution_confidence": resolution_details.get("confidence", 0.0),
                    "sources_agreeing": resolution_details.get("sources_agreeing", []),
                }
                if resolution_details.get("rescue_explanation"):
                    entry["details"]["rescue_explanation"] = resolution_details["rescue_explanation"]
                if resolution_details.get("degraded"):
                    entry["details"]["degraded"] = True
            else:
                # Programmatic near-agree resolution (not in a zone)
                normalized_texts = [_normalize_for_comparison(b.normalized_text) for b in blocks]
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
                            assem_norm_i = _normalize_for_comparison(raw_texts[i])
                            assem_norm_j = _normalize_for_comparison(raw_texts[j])
                            nums_i = _extract_numeric_tokens(assem_norm_i)
                            nums_j = _extract_numeric_tokens(assem_norm_j)
                            cites_i = _extract_citation_keys(assem_norm_i)
                            cites_j = _extract_citation_keys(assem_norm_j)
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
            resolution_details = (resolution_metadata or {}).get(triple.segment_id, {})

            if resolved_text is not None:
                entry["chosen_text"] = resolved_text
                entry["chosen_source"] = resolution_details.get("chosen_source", resolution_details.get("method", "llm"))
            elif blocks:
                entry["chosen_text"] = clean_output_md(
                    blocks[0].source_md or blocks[0].raw_text,
                )
                entry["chosen_source"] = blocks[0].source

            entry["details"] = {
                "zone_id": resolution_details.get("zone_id", ""),
                "llm_resolved": resolved_text is not None,
                "resolution_method": resolution_details.get("method", ""),
                "resolution_confidence": resolution_details.get("confidence", 0.0),
                "sources_agreeing": resolution_details.get("sources_agreeing", []),
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
    triples, alignment_confidence = align_blocks(blocks_by_source)
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

    # Step 4: Guard checks
    should_fallback, fallback_reason = check_guards(
        triples,
        alignment_confidence,
        conflict_ratio_threshold=base_conflict_threshold,
        alignment_confidence_threshold=_cfg_float(
            getattr(Config, "CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK", 0.5), 0.5,
        ),
        textual_conflict_ratio_threshold=_cfg_float(
            getattr(Config, "CONSENSUS_CONFLICT_RATIO_TEXTUAL_FALLBACK", base_conflict_threshold),
            base_conflict_threshold,
        ),
        structured_conflict_ratio_threshold=_cfg_float(
            getattr(Config, "CONSENSUS_CONFLICT_RATIO_STRUCTURED_FALLBACK", 0.85), 0.85,
        ),
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

    if should_fallback:
        logger.info("Consensus pipeline: fallback triggered (%s)", fallback_reason)
        audit_entries = _build_audit_entries(
            triples, {},
            near_threshold=Config.CONSENSUS_NEAR_THRESHOLD,
            levenshtein_threshold=Config.CONSENSUS_LEVENSHTEIN_THRESHOLD,
        )
        metrics = compute_metrics(
            triples, alignment_confidence, True, fallback_reason,
            guard_telemetry=guard_telemetry,
        )
        return None, metrics, audit_entries

    # Step 5: Resolve conflicts via zone-based layered resolver
    conflict_triples = [t for t in triples if t.classification == CONFLICT]
    resolved_conflicts: dict[str, str] = {}
    resolution_metadata: dict[str, dict] = {}

    if conflict_triples:
        step_start = time.monotonic()
        flanking_count = _cfg_int(
            getattr(Config, "CONSENSUS_ZONE_FLANKING_COUNT", 2), 2,
        )
        layered_enabled = _cfg_bool(
            getattr(Config, "CONSENSUS_LAYERED_ENABLED", True), True,
        )
        logger.info(
            "Consensus pipeline: resolving %d conflicts (layered=%s, flanking=%d)...",
            len(conflict_triples), layered_enabled, flanking_count,
        )

        # Build conflict zones
        conflict_zones = _build_conflict_zones(triples, flanking_count=flanking_count)
        logger.info(
            "Consensus pipeline: grouped %d conflicts into %d zone(s)",
            len(conflict_triples), len(conflict_zones),
        )
        for zone in conflict_zones:
            conflict_count = sum(1 for s in zone["segments"] if s["status"] == "conflict")
            gap_count = sum(1 for s in zone["segments"] if s["status"] == "gap")
            near_agree_count = sum(1 for s in zone["segments"] if s["status"] == "near_agree")
            logger.info(
                "  %s: %d conflict(s), %d gap(s), %d near_agree(s), %d before, %d after",
                zone["zone_id"], conflict_count, gap_count, near_agree_count,
                len(zone["context_before"]), len(zone["context_after"]),
            )

        try:
            if layered_enabled:
                resolved_conflicts, resolution_metadata = _resolve_conflicts_layered(
                    conflict_zones,
                    triples,
                    llm,
                    medium_similarity_threshold=_cfg_float(
                        getattr(Config, "CONSENSUS_LAYERED_MEDIUM_SIM_THRESHOLD",
                                _LAYERED_MEDIUM_SIM_THRESHOLD),
                        _LAYERED_MEDIUM_SIM_THRESHOLD,
                    ),
                )
            else:
                # Non-layered: send all zones directly to LLM (skip median-source)
                zone_results, _unresolved = llm.resolve_conflict_zones(conflict_zones)
                for zone in conflict_zones:
                    for seg in zone["segments"]:
                        seg_id = seg["segment_id"]
                        if seg["status"] == "conflict":
                            source_texts = _bundle_source_texts(seg)
                            resolved_text = (zone_results.get(seg_id, "") or "").strip()
                            if not resolved_text:
                                # Three-tier rescue strategy for empty conflict
                                max_pair_sim = max(_pairwise_similarities(list(source_texts.values()))) if source_texts else 0.0
                                _rescue_segment(
                                    seg_id, seg, zone, triples, resolved_conflicts,
                                    resolution_metadata, source_texts, max_pair_sim, llm,
                                    method_prefix="llm_conflict",
                                )
                                continue
                            resolved_conflicts[seg_id] = resolved_text
                            resolution_metadata[seg_id] = {
                                "method": "llm_conflict",
                                "chosen_source": "llm",
                                "confidence": round(
                                    _mean_similarity_to_sources(resolved_text, source_texts), 4,
                                ),
                                "sources_agreeing": sorted(source_texts.keys()),
                                "max_pair_similarity": round(
                                    max(_pairwise_similarities(list(source_texts.values()))) if source_texts else 0.0,
                                    4,
                                ),
                                "zone_id": zone["zone_id"],
                            }
                        elif seg["status"] == "near_agree":
                            source_texts = _bundle_source_texts(seg)
                            resolved_text = (zone_results.get(seg_id, "") or "").strip()
                            if not resolved_text:
                                # Three-tier rescue strategy instead of raising ValueError
                                max_pair_sim = max(_pairwise_similarities(list(source_texts.values()))) if source_texts else 0.0
                                _rescue_segment(
                                    seg_id, seg, zone, triples, resolved_conflicts,
                                    resolution_metadata, source_texts, max_pair_sim, llm,
                                    method_prefix="llm_near_agree",
                                )
                                continue
                            resolved_conflicts[seg_id] = resolved_text
                            resolution_metadata[seg_id] = {
                                "method": "llm_near_agree",
                                "chosen_source": "llm",
                                "confidence": round(
                                    _mean_similarity_to_sources(resolved_text, source_texts), 4,
                                ) if resolved_text else 0.0,
                                "sources_agreeing": sorted(source_texts.keys()),
                                "max_pair_similarity": round(
                                    max(_pairwise_similarities(list(source_texts.values()))) if source_texts else 0.0,
                                    4,
                                ),
                                "zone_id": zone["zone_id"],
                            }
                        elif seg["status"] == "gap" and seg_id in zone_results:
                            resolved_conflicts[seg_id] = zone_results[seg_id]
                            resolution_metadata[seg_id] = {
                                "method": "llm_gap",
                                "chosen_source": "llm",
                                "confidence": 0.0,
                                "sources_agreeing": [],
                                "max_pair_similarity": 0.0,
                                "zone_id": zone["zone_id"],
                            }
        except Exception as e:
            logger.warning("Consensus pipeline: conflict zone resolution failed: %s", e)
            audit_entries = _build_audit_entries(
                triples, {}, {},
                near_threshold=Config.CONSENSUS_NEAR_THRESHOLD,
                levenshtein_threshold=Config.CONSENSUS_LEVENSHTEIN_THRESHOLD,
            )
            metrics = compute_metrics(
                triples, alignment_confidence, True, "llm_error",
                guard_telemetry=guard_telemetry,
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

    metrics = compute_metrics(
        triples, alignment_confidence, False, None, guard_telemetry=guard_telemetry,
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
    metrics["gap_cross_dedup_removed"] = gap_cross_removed
    metrics["assembled_dedup_removed"] = assembled_dups_removed
    metrics["qa"] = qa_results

    # Compute degradation metrics from resolution metadata
    degradation = build_degradation_metrics(
        triples=triples,
        resolution_metadata=resolution_metadata,
        audit_entries=audit_entries,
        total_blocks=metrics["total_blocks"],
        zone_resolution_tokens=llm.usage.tokens_for_types("zone_resolution"),
        rescue_call_tokens=llm.usage.tokens_for_types("rescue")
    )
    metrics["degradation_metrics"] = degradation

    degraded_count = degradation["degraded_segments"]["count"]
    if degraded_count > 0:
        logger.warning(
            "Zone resolution used best-source fallback for %d segment(s) "
            "(quality_score=%.3f, grade=%s) — quality may be reduced",
            degraded_count,
            degradation["quality_score"],
            degradation["quality_grade"],
        )

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

    pipeline_duration = time.monotonic() - pipeline_start
    logger.info(
        "Consensus pipeline complete in %.1fs: %d blocks, %d conflicts resolved, "
        "~%d tokens saved, %d GAP cross-dedup, %d post-assembly dedup",
        pipeline_duration,
        metrics["total_blocks"],
        metrics["conflict"],
        metrics["tokens_saved_estimate"],
        gap_cross_removed,
        assembled_dups_removed,
    )

    return merged_md, metrics, audit_entries
