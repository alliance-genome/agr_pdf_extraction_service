"""Markdown normalization, block parsing, and cross-extractor alignment."""

from __future__ import annotations

import logging
import re
import unicodedata

import mistune
import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from app.services.consensus_models import AlignedTriple, Block

logger = logging.getLogger(__name__)
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


