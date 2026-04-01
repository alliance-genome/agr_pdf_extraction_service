"""Markdown normalization, block parsing, and cross-extractor alignment."""

from __future__ import annotations

import logging
import re
import unicodedata

import mistune

from app.services.alignment.arbitration import ArbitrationConfig, ArbitrationContext, choose_end_mode
from app.services.alignment.dp3 import align_three_way_global
from app.services.alignment.partitioning import AnchorPartitionConfig, build_alignment_windows
from app.services.alignment.repair import repair_split_merge_columns
from app.services.alignment.scoring import ScoreConfig
from app.services.alignment.traceback import traceback_columns
from app.services.alignment.triples import build_aligned_triples

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
_SUP_TAG_RE = re.compile(r"</?sup>", re.IGNORECASE)
_HTML_COMMENT_OUTPUT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_PAGE_COMMENT_RE = re.compile(r"<!--\s*page\s*[:=]?\s*(\d+)\s*-->", re.IGNORECASE)
_PAGE_BRACKET_RE = re.compile(r"^\[\s*page\s+(\d+)\s*\]$", re.IGNORECASE)
_PAGE_SPAN_ID_RE = re.compile(r"<span[^>]*id=['\"]page-(\d+)-[^'\"]*['\"][^>]*>", re.IGNORECASE)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"  +")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)]*\)")
_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# Inline formatting stripping (for embedding-clean output).
# Order matters: strip bold-italic (***) before bold (**) before italic (*).
_BOLD_ITALIC_RE = re.compile(r"\*{3}(.+?)\*{3}")
_BOLD_RE = re.compile(r"\*{2}(.+?)\*{2}")
_ITALIC_RE = re.compile(r"\*(.+?)\*")
_BOLD_ALT_RE = re.compile(r"__(.+?)__")
_ITALIC_ALT_RE = re.compile(r"(?<!\w)_(.+?)_(?!\w)")  # avoid snake_case
_STRIKETHROUGH_RE = re.compile(r"~~(.+?)~~")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
_FENCED_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
_HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:---+|\*\*\*+|___+)\s*$", re.MULTILINE)

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


def _extract_page_marker(text: str) -> int | None:
    """Extract a page marker from supported syntaxes."""
    if not text:
        return None
    comment_match = _PAGE_COMMENT_RE.search(text)
    if comment_match:
        return max(1, int(comment_match.group(1)))
    bracket_match = _PAGE_BRACKET_RE.match(text.strip())
    if bracket_match:
        return max(1, int(bracket_match.group(1)))
    span_match = _PAGE_SPAN_ID_RE.search(text)
    if span_match:
        return max(1, int(span_match.group(1)))
    return None


def _strip_non_page_comments(text: str) -> str:
    """Remove HTML comments except page markers."""
    def _replacer(match: re.Match[str]) -> str:
        raw = match.group(0)
        return raw if _PAGE_COMMENT_RE.fullmatch(raw.strip()) else ""

    return _HTML_COMMENT_OUTPUT_RE.sub(_replacer, text)


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
    text = _SUP_TAG_RE.sub("", text)

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

    This removes extractor-specific formatting artifacts AND inline cosmetic
    markdown at the source layer so alignment/classification operates on
    cleaner inputs.  Structural markdown (headings, lists, tables, blockquotes)
    is preserved; inline decoration (bold, italic, strikethrough, code) is
    stripped because the output is used for embedding, not rendering.
    """
    text = unicodedata.normalize("NFKC", markdown_text or "")
    # Convert marker page anchors into explicit page comments for downstream parsing.
    text = re.sub(
        r"<span\s+id=['\"]page-(\d+)-[^'\"]*['\"]>(.*?)</span>",
        lambda m: f"<!-- page: {m.group(1)} -->\n{m.group(2)}",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = _SPAN_REF_RE.sub(r"\1", text)
    text = _SUP_TAG_RE.sub("", text)
    text = _strip_non_page_comments(text)
    text = _IMAGE_REF_RE.sub("", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    # Common Docling ligature artifacts.
    text = text.replace("/uniFB01", "fi").replace("/uniFB02", "fl")

    # --- Strip cosmetic markdown (keep structural) ---
    # Block-level: fenced code blocks and horizontal rules
    text = _FENCED_CODE_BLOCK_RE.sub("", text)
    text = _HORIZONTAL_RULE_RE.sub("", text)

    # Inline: bold-italic (***) before bold (**) before italic (*)
    text = _BOLD_ITALIC_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    # Underscore variants
    text = _BOLD_ALT_RE.sub(r"\1", text)
    text = _ITALIC_ALT_RE.sub(r"\1", text)
    # Strikethrough and inline code
    text = _STRIKETHROUGH_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)

    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_output_md(text: str) -> str:
    """Strip extractor-specific artifacts while preserving markdown formatting."""
    text = _SPAN_OPEN_RE.sub("", text)
    text = _SPAN_CLOSE_RE.sub("", text)
    text = _SUP_TAG_RE.sub("", text)
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
    current_page = 1

    if not isinstance(tokens, list):
        # Fallback: split by double newlines if AST returns unexpected type
        paragraphs = [p.strip() for p in markdown_text.split("\n\n") if p.strip()]
        tokens = [{"type": "paragraph", "raw": p} for p in paragraphs]

    for token in tokens:
        if isinstance(token, str):
            token = {"type": "paragraph", "raw": token}

        ttype = token.get("type", "")
        if ttype in ("newline", "space", "thematic_break", "blank_line"):
            continue

        if ttype == "block_html":
            page_marker = _extract_page_marker(token.get("raw", ""))
            if page_marker is not None:
                current_page = page_marker
            continue

        inline_page_marker = _extract_page_marker(_extract_text(token))
        if inline_page_marker is not None:
            current_page = inline_page_marker

        raw_text = _extract_text(token)
        if not raw_text.strip():
            continue
        if _PAGE_BRACKET_RE.match(raw_text.strip()):
            current_page = max(1, int(_PAGE_BRACKET_RE.match(raw_text.strip()).group(1)))
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
            page_no=current_page,
        ))
        idx += 1

    return blocks


# ---------------------------------------------------------------------------
# Step 2: ALIGN — Full 3-way global DP (Needleman family)
# ---------------------------------------------------------------------------

def align_blocks(
    blocks_by_source: dict[str, list[Block]],
    *,
    llm=None,
    anchor_partitioning_enabled: bool = True,
    anchor_min_score: float = 0.72,
    anchor_include_structural_secondary: bool = True,
    anchor_max_heading_level: int = 2,
    ambiguity_delta: float = 0.03,
    semantic_rerank_enabled: bool = True,
    semantic_margin: float = 0.02,
    llm_tiebreak_enabled: bool = True,
) -> tuple[list[AlignedTriple], float]:
    """Align blocks across extractors using only full 3-way global DP."""
    grobid_blocks = blocks_by_source.get("grobid", [])
    docling_blocks = blocks_by_source.get("docling", [])
    marker_blocks = blocks_by_source.get("marker", [])
    if not grobid_blocks and not docling_blocks and not marker_blocks:
        return [], 0.0

    score_config = ScoreConfig()
    arbitration_config = ArbitrationConfig(
        ambiguity_delta=ambiguity_delta,
        semantic_rerank_enabled=semantic_rerank_enabled,
        semantic_margin=semantic_margin,
        llm_tiebreak_enabled=llm_tiebreak_enabled,
        repair_ambiguity_delta=ambiguity_delta,
        repair_semantic_margin=semantic_margin,
    )
    arbitration_context = ArbitrationContext()
    partition_config = AnchorPartitionConfig(
        enabled=anchor_partitioning_enabled,
        min_anchor_score=anchor_min_score,
        ambiguity_delta=ambiguity_delta,
        include_structural_secondary=anchor_include_structural_secondary,
        llm_tiebreak_enabled=llm_tiebreak_enabled,
        max_heading_level=anchor_max_heading_level,
    )
    windows = build_alignment_windows(
        grobid_blocks,
        docling_blocks,
        marker_blocks,
        score_config=score_config,
        partition_config=partition_config,
        llm=llm,
    )

    columns = []
    for win_idx, window in enumerate(windows):
        logger.info(
            "Alignment window %d/%d: grobid=%d, docling=%d, marker=%d blocks",
            win_idx + 1, len(windows),
            len(window.get("grobid", [])),
            len(window.get("docling", [])),
            len(window.get("marker", [])),
        )
        dp_result = align_three_way_global(
            window.get("grobid", []),
            window.get("docling", []),
            window.get("marker", []),
            config=score_config,
        )
        end_mode = choose_end_mode(
            dp_result,
            config=arbitration_config,
            context=arbitration_context,
            llm=llm,
        )
        window_columns = traceback_columns(dp_result, end_mode=end_mode)
        window_columns = repair_split_merge_columns(
            window_columns,
            config=score_config,
            arbitration_config=arbitration_config,
            arbitration_context=arbitration_context,
            llm=llm,
        )
        columns.extend(window_columns)

    if arbitration_context.llm_tiebreak_calls > 0:
        logger.info("Alignment LLM tiebreak calls: %d", arbitration_context.llm_tiebreak_calls)

    return build_aligned_triples(columns, config=score_config)
