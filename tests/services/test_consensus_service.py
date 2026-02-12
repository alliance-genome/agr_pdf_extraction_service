"""Tests for the selective LLM merge consensus pipeline."""

import json
import pytest
from unittest.mock import MagicMock, patch

from app.services.consensus_service import (
    Block,
    AlignedTriple,
    AGREE_EXACT,
    AGREE_NEAR,
    GAP,
    CONFLICT,
    _SOURCE_PREFERENCE,
    _pick_preferred_text,
    normalize_text,
    parse_markdown,
    align_blocks,
    classify_triples,
    check_guards,
    assemble,
    compute_metrics,
    merge_with_consensus,
)


# ---------------------------------------------------------------------------
# Fixtures: sample markdown for each extractor
# ---------------------------------------------------------------------------

GROBID_MD = """# Introduction

This paper presents a novel approach to gene annotation in model organisms.
The approach leverages machine learning techniques [1] to improve accuracy.

## Methods

We used a dataset of 500 annotated genes from FlyBase (Release 2024).
The precision was 0.95 and recall was 0.87.

## Results

Our method achieved 95% accuracy on the test set.

## References

[1] Smith et al. 2023. Gene annotation methods.
"""

DOCLING_MD = """# Introduction

This paper presents a novel approach to gene annotation in model organisms.
The approach leverages machine learning techniques [1] to improve accuracy.

## Methods

We used a dataset of 500 annotated genes from FlyBase (Release 2024).
The precision was 0.95 and recall was 0.87.

| Metric | Value |
|--------|-------|
| Precision | 0.95 |
| Recall | 0.87 |

## Results

Our method achieved 95% accuracy on the test set.

## References

[1] Smith et al. 2023. Gene annotation methods.
"""

MARKER_MD = """# Introduction

This paper presents a novel approach to gene annotation in model organisms.
The approach leverages machine learning techniques [1] to improve accuracy.

## Methods

We used a dataset of 500 annotated genes from FlyBase (Release 2024).
The precision was 0.95 and recall was 0.87.

## Results

Our method achieved 95% accuracy on the test set.

## References

[1] Smith et al. 2023. Gene annotation methods.
"""


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParseMarkdown:
    def test_basic_parse(self):
        md = "# Title\n\nSome paragraph text.\n\n## Section\n\nMore text."
        blocks = parse_markdown(md, "grobid")
        assert len(blocks) >= 3
        assert blocks[0].block_type == "heading"
        assert blocks[0].source == "grobid"

    def test_heading_detection(self):
        md = "# H1\n\n## H2\n\n### H3\n\nParagraph."
        blocks = parse_markdown(md, "docling")
        headings = [b for b in blocks if b.block_type == "heading"]
        assert len(headings) >= 1

    def test_heading_level(self):
        md = "# Top Level\n\nBody text."
        blocks = parse_markdown(md, "grobid")
        heading = [b for b in blocks if b.block_type == "heading"]
        assert len(heading) >= 1
        assert heading[0].heading_level is not None

    def test_paragraph_detection(self):
        md = "This is a paragraph.\n\nThis is another paragraph."
        blocks = parse_markdown(md, "marker")
        paragraphs = [b for b in blocks if b.block_type == "paragraph"]
        assert len(paragraphs) >= 1

    def test_table_detection(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        blocks = parse_markdown(md, "docling")
        tables = [b for b in blocks if b.block_type == "table"]
        assert len(tables) >= 1

    def test_order_index_sequential(self):
        md = "# H1\n\nPara 1.\n\n## H2\n\nPara 2."
        blocks = parse_markdown(md, "grobid")
        for i, b in enumerate(blocks):
            assert b.order_index == i

    def test_block_id_format(self):
        md = "# Title\n\nText."
        blocks = parse_markdown(md, "grobid")
        for b in blocks:
            assert b.block_id.startswith("grobid_block_")

    def test_empty_markdown(self):
        blocks = parse_markdown("", "grobid")
        assert blocks == []

    def test_figure_ref_detection(self):
        md = "![Figure 1](image.png)"
        blocks = parse_markdown(md, "marker")
        # The image may be parsed as a paragraph containing a figure ref
        has_figure = any(b.block_type == "figure_ref" for b in blocks)
        has_text_with_figure = any("![" in b.raw_text for b in blocks)
        assert has_figure or has_text_with_figure


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_whitespace_collapsing(self):
        assert normalize_text("hello   world") == "hello world"

    def test_unicode_normalization(self):
        # NFKC normalizes compatibility characters
        result = normalize_text("\uff41\uff42\uff43")  # fullwidth abc
        assert result == "abc"

    def test_markdown_bold_equivalence(self):
        assert normalize_text("__bold__") == normalize_text("**bold**")

    def test_markdown_italic_equivalence(self):
        assert normalize_text("_italic_") == normalize_text("*italic*")

    def test_lowercasing(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strip_whitespace(self):
        assert normalize_text("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# Alignment tests
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_identical_documents_align(self):
        """Three identical documents should produce well-aligned triples."""
        md = "# Title\n\nParagraph one.\n\n## Section\n\nParagraph two."
        blocks_a = parse_markdown(md, "grobid")
        blocks_b = parse_markdown(md, "docling")
        blocks_c = parse_markdown(md, "marker")

        triples, confidence = align_blocks({
            "grobid": blocks_a,
            "docling": blocks_b,
            "marker": blocks_c,
        })

        assert len(triples) >= 3
        assert confidence > 0.5

        # All triples should have all 3 blocks
        for t in triples[:len(blocks_a)]:
            present = sum(1 for attr in ("grobid_block", "docling_block", "marker_block")
                         if getattr(t, attr) is not None)
            assert present >= 2

    def test_gap_handling(self):
        """Missing section in one extractor should produce GAP triples."""
        md_full = "# Title\n\nPara 1.\n\n## Methods\n\nPara 2.\n\n## Results\n\nPara 3."
        md_short = "# Title\n\nPara 1.\n\n## Results\n\nPara 3."

        blocks_full = parse_markdown(md_full, "grobid")
        blocks_full2 = parse_markdown(md_full, "docling")
        blocks_short = parse_markdown(md_short, "marker")

        triples, confidence = align_blocks({
            "grobid": blocks_full,
            "docling": blocks_full2,
            "marker": blocks_short,
        })

        # Should have triples for all blocks (no drops)
        assert len(triples) >= len(blocks_full)

    def test_leftover_blocks_collected(self):
        """Unmatched blocks from non-reference extractors should not be dropped."""
        md_a = "# A\n\nPara A."
        md_b = "# B\n\nPara B.\n\n## Extra Section\n\nExtra content."

        blocks_a = parse_markdown(md_a, "grobid")
        blocks_b = parse_markdown(md_b, "docling")
        blocks_c = parse_markdown(md_a, "marker")

        triples, _ = align_blocks({
            "grobid": blocks_a,
            "docling": blocks_b,
            "marker": blocks_c,
        })

        # All blocks from all sources should appear
        all_block_ids = set()
        for t in triples:
            for attr in ("grobid_block", "docling_block", "marker_block"):
                b = getattr(t, attr)
                if b is not None:
                    all_block_ids.add(b.block_id)

        total_input_blocks = len(blocks_a) + len(blocks_b) + len(blocks_c)
        assert len(all_block_ids) == total_input_blocks

    def test_empty_source(self):
        """Empty extractor output should be handled gracefully."""
        md = "# Title\n\nText."
        blocks = parse_markdown(md, "grobid")

        triples, confidence = align_blocks({
            "grobid": blocks,
            "docling": [],
            "marker": [],
        })

        assert len(triples) == len(blocks)
        assert confidence == 0.0

    def test_no_blocks_at_all(self):
        """All empty should return empty."""
        triples, confidence = align_blocks({
            "grobid": [],
            "docling": [],
            "marker": [],
        })
        assert triples == []
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassification:
    def _make_triple(self, seg_id, grobid_text, docling_text, marker_text,
                     block_type="paragraph"):
        triple = AlignedTriple(segment_id=seg_id)
        if grobid_text is not None:
            triple.grobid_block = Block(
                block_id="g", block_type=block_type, raw_text=grobid_text,
                normalized_text=normalize_text(grobid_text),
                heading_level=None, order_index=0, source="grobid",
            )
        if docling_text is not None:
            triple.docling_block = Block(
                block_id="d", block_type=block_type, raw_text=docling_text,
                normalized_text=normalize_text(docling_text),
                heading_level=None, order_index=0, source="docling",
            )
        if marker_text is not None:
            triple.marker_block = Block(
                block_id="m", block_type=block_type, raw_text=marker_text,
                normalized_text=normalize_text(marker_text),
                heading_level=None, order_index=0, source="marker",
            )
        return triple

    def test_agree_exact(self):
        t = self._make_triple("seg_001", "Hello world", "Hello world", "Different")
        classify_triples([t])
        assert t.classification == AGREE_EXACT
        assert t.agreed_text is not None

    def test_agree_exact_picks_agreeing_source_not_outlier(self):
        """When grobid==docling but marker differs, agreed_text must NOT be marker's text.

        For paragraphs, source preference is marker, but the outlier must be excluded.
        """
        t = self._make_triple("seg_001", "Hello world", "Hello world", "OUTLIER TEXT")
        classify_triples([t])
        assert t.classification == AGREE_EXACT
        assert t.agreed_text == "Hello world"
        assert t.agreed_text != "OUTLIER TEXT"

    def test_agree_exact_all_three(self):
        t = self._make_triple("seg_001", "Same text here", "Same text here", "Same text here")
        classify_triples([t])
        assert t.classification == AGREE_EXACT

    def test_agree_near_picks_agreeing_source_not_outlier(self):
        """When two sources are near-matches, agreed_text should come from an agreeing source."""
        t = self._make_triple(
            "seg_001",
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumped over the lazy dog",
            "Completely different outlier text here",
        )
        classify_triples([t], near_threshold=0.85, levenshtein_threshold=0.85)
        assert t.classification == AGREE_NEAR
        # Must NOT pick the outlier marker text
        assert "outlier" not in t.agreed_text.lower()

    def test_agree_near(self):
        t = self._make_triple(
            "seg_001",
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumped over the lazy dog",
            "Completely different text about something else",
        )
        classify_triples([t], near_threshold=0.85, levenshtein_threshold=0.85)
        assert t.classification == AGREE_NEAR

    def test_conflict_on_numeric_difference(self):
        """Even high similarity should be CONFLICT if numbers differ."""
        t = self._make_triple(
            "seg_001",
            "The accuracy was 95% on the test set",
            "The accuracy was 96% on the test set",
            "The accuracy was 97% on the test set",
        )
        classify_triples([t])
        assert t.classification == CONFLICT

    def test_gap_single_source(self):
        t = self._make_triple("seg_001", "Only in grobid", None, None)
        classify_triples([t])
        assert t.classification == GAP
        assert t.agreed_text == "Only in grobid"

    def test_gap_no_sources(self):
        t = self._make_triple("seg_001", None, None, None)
        classify_triples([t])
        assert t.classification == GAP

    def test_table_always_conflict(self):
        t = self._make_triple(
            "seg_001",
            "| A | B |\n|---|---|\n| 1 | 2 |",
            "| A | B |\n|---|---|\n| 1 | 2 |",
            "| A | B |\n|---|---|\n| 1 | 2 |",
            block_type="table",
        )
        classify_triples([t])
        assert t.classification == CONFLICT

    def test_mixed_block_type_table_escalation(self):
        """If ANY extractor sees a table, escalate to CONFLICT even if others say paragraph."""
        t = self._make_triple(
            "seg_001",
            "| A | B |\n|---|---|\n| 1 | 2 |",
            "A B 1 2",
            "A B 1 2",
        )
        # Override block types: grobid sees table, others see paragraph
        t.grobid_block.block_type = "table"
        t.docling_block.block_type = "paragraph"
        t.marker_block.block_type = "paragraph"
        classify_triples([t])
        assert t.classification == CONFLICT

    def test_mixed_block_type_equation_escalation(self):
        """If ANY extractor sees an equation, escalate to CONFLICT."""
        t = self._make_triple(
            "seg_001",
            "E = mc^2",
            "E = mc^2",
            "E = mc^2",
        )
        t.grobid_block.block_type = "paragraph"
        t.docling_block.block_type = "equation"
        t.marker_block.block_type = "paragraph"
        classify_triples([t])
        assert t.classification == CONFLICT

    def test_no_escalation_when_disabled(self):
        """Mixed block types with escalation disabled should NOT force CONFLICT."""
        t = self._make_triple(
            "seg_001",
            "Same text here",
            "Same text here",
            "Same text here",
        )
        t.grobid_block.block_type = "table"
        t.docling_block.block_type = "paragraph"
        t.marker_block.block_type = "paragraph"
        classify_triples([t], always_escalate_tables=False)
        # With escalation off, exact match should win
        assert t.classification == AGREE_EXACT

    def test_conflict_on_low_similarity(self):
        t = self._make_triple(
            "seg_001",
            "Alpha beta gamma delta epsilon",
            "One two three four five",
            "Completely unrelated text here",
        )
        classify_triples([t])
        assert t.classification == CONFLICT


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------

class TestGuards:
    def _make_classified_triples(self, classifications):
        """Helper to create triples with given classifications."""
        triples = []
        for i, cls in enumerate(classifications):
            t = AlignedTriple(segment_id=f"seg_{i:03d}")
            t.classification = cls
            t.confidence = 0.8
            triples.append(t)
        return triples

    def test_no_fallback_when_healthy(self):
        triples = self._make_classified_triples([
            AGREE_EXACT, AGREE_EXACT, AGREE_NEAR, CONFLICT, GAP,
        ])
        should_fallback, reason = check_guards(triples, alignment_confidence=0.7)
        assert not should_fallback
        assert reason is None

    def test_fallback_on_high_conflict_ratio(self):
        triples = self._make_classified_triples([
            CONFLICT, CONFLICT, CONFLICT, AGREE_EXACT, AGREE_EXACT,
        ])
        # 3/5 = 0.6 > 0.4
        should_fallback, reason = check_guards(triples, alignment_confidence=0.7)
        assert should_fallback
        assert reason == "conflict_ratio"

    def test_fallback_on_low_alignment_confidence(self):
        triples = self._make_classified_triples([AGREE_EXACT, AGREE_EXACT])
        should_fallback, reason = check_guards(triples, alignment_confidence=0.3)
        assert should_fallback
        assert reason == "alignment_confidence"

    def test_gap_excluded_from_conflict_ratio(self):
        """GAP blocks should NOT inflate the conflict ratio."""
        triples = self._make_classified_triples([
            AGREE_EXACT, CONFLICT, GAP, GAP, GAP,
        ])
        # denominator = 5 - 3 = 2, conflict = 1/2 = 0.5 > 0.4
        should_fallback, reason = check_guards(triples, alignment_confidence=0.7)
        assert should_fallback
        assert reason == "conflict_ratio"

    def test_all_gap_no_fallback(self):
        """All-GAP should have conflict_ratio = 0.0 (no fallback from ratio)."""
        triples = self._make_classified_triples([GAP, GAP, GAP])
        should_fallback, reason = check_guards(triples, alignment_confidence=0.7)
        assert not should_fallback

    def test_empty_triples_triggers_fallback(self):
        should_fallback, reason = check_guards([], alignment_confidence=0.7)
        assert should_fallback
        assert reason == "no_blocks"


# ---------------------------------------------------------------------------
# Assembly tests
# ---------------------------------------------------------------------------

class TestAssembly:
    def test_basic_assembly(self):
        triples = [
            AlignedTriple(segment_id="seg_000", classification=AGREE_EXACT,
                          agreed_text="# Introduction"),
            AlignedTriple(segment_id="seg_001", classification=AGREE_NEAR,
                          agreed_text="Paragraph text here."),
            AlignedTriple(segment_id="seg_002", classification=GAP,
                          agreed_text="Gap text from one source."),
            AlignedTriple(segment_id="seg_003", classification=CONFLICT),
        ]

        resolved = {"seg_003": "LLM resolved this conflict."}
        result = assemble(triples, resolved)

        assert "# Introduction" in result
        assert "Paragraph text here." in result
        assert "Gap text from one source." in result
        assert "LLM resolved this conflict." in result

    def test_correct_ordering(self):
        triples = [
            AlignedTriple(segment_id="seg_000", classification=AGREE_EXACT,
                          agreed_text="First"),
            AlignedTriple(segment_id="seg_001", classification=AGREE_EXACT,
                          agreed_text="Second"),
            AlignedTriple(segment_id="seg_002", classification=AGREE_EXACT,
                          agreed_text="Third"),
        ]
        result = assemble(triples, {})
        assert result.index("First") < result.index("Second") < result.index("Third")

    def test_no_missing_segments(self):
        triples = [
            AlignedTriple(segment_id="seg_000", classification=CONFLICT),
            AlignedTriple(segment_id="seg_001", classification=CONFLICT),
        ]
        resolved = {"seg_000": "Resolved A", "seg_001": "Resolved B"}
        result = assemble(triples, resolved)
        assert "Resolved A" in result
        assert "Resolved B" in result


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_basic_metrics(self):
        triples = []
        for cls in [AGREE_EXACT, AGREE_EXACT, AGREE_NEAR, GAP, CONFLICT]:
            t = AlignedTriple(segment_id=f"seg_{len(triples):03d}")
            t.classification = cls
            triples.append(t)

        metrics = compute_metrics(triples, 0.75, False, None)
        assert metrics["total_blocks"] == 5
        assert metrics["agree_exact"] == 2
        assert metrics["agree_near"] == 1
        assert metrics["gap"] == 1
        assert metrics["conflict"] == 1
        # conflict_ratio = 1/(5-1) = 0.25
        assert metrics["conflict_ratio"] == 0.25
        assert metrics["alignment_confidence"] == 0.75
        assert metrics["fallback_triggered"] is False
        assert metrics["tokens_saved_estimate"] == 400  # (2+1+1)*100


# ---------------------------------------------------------------------------
# End-to-end / orchestrator tests
# ---------------------------------------------------------------------------

class TestMergeWithConsensus:
    @patch("config.Config")
    def test_missing_extractor_returns_fallback(self, mock_config):
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.4
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.5

        llm = MagicMock()
        result, metrics = merge_with_consensus("grobid", "", "marker", llm)
        assert result is None
        assert metrics["fallback_triggered"] is True
        assert metrics["fallback_reason"] == "missing_extractor"

    @patch("config.Config")
    def test_none_input_returns_fallback(self, mock_config):
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.4
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.5

        llm = MagicMock()
        result, metrics = merge_with_consensus("grobid", None, "marker", llm)
        assert result is None
        assert metrics["fallback_reason"] == "missing_extractor"

    @patch("config.Config")
    def test_full_pipeline_with_agreement(self, mock_config):
        """When all 3 extractors agree, should succeed without calling LLM."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.4
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.5

        llm = MagicMock()

        # Use identical text so everything agrees
        md = "# Title\n\nThis is the body text of the paper."
        result, metrics = merge_with_consensus(md, md, md, llm)

        assert result is not None
        assert "Title" in result
        assert "body text" in result
        assert metrics["fallback_triggered"] is False
        # LLM should NOT have been called for conflict resolution
        llm.resolve_conflicts.assert_not_called()

    @patch("config.Config")
    def test_full_pipeline_with_conflicts(self, mock_config):
        """When extractors disagree, LLM should be called for conflicts."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8  # high threshold to avoid fallback
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.1  # low threshold

        llm = MagicMock()
        llm.resolve_conflicts.return_value = {}  # will be filled dynamically

        md_a = "# Title\n\nThe accuracy was 95%."
        md_b = "# Title\n\nThe accuracy was 96%."
        md_c = "# Title\n\nThe accuracy was 97%."

        # Mock resolve_conflicts to return resolved text for any segment
        def mock_resolve(conflicts):
            return {c["segment_id"]: "The accuracy was 95.5%." for c in conflicts}
        llm.resolve_conflicts.side_effect = mock_resolve

        result, metrics = merge_with_consensus(md_a, md_b, md_c, llm)

        if result is not None:
            # Pipeline succeeded with conflict resolution
            assert metrics["conflict"] >= 1
            llm.resolve_conflicts.assert_called()
        else:
            # Fallback triggered (acceptable if confidence too low)
            assert metrics["fallback_triggered"] is True

    @patch("config.Config")
    def test_llm_error_triggers_fallback(self, mock_config):
        """LLM failure during conflict resolution should trigger fallback."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.1

        llm = MagicMock()
        llm.resolve_conflicts.side_effect = Exception("LLM is down")

        md_a = "# Title\n\nThe accuracy was 95%."
        md_b = "# Title\n\nThe accuracy was 96%."
        md_c = "# Title\n\nThe accuracy was 97%."

        result, metrics = merge_with_consensus(md_a, md_b, md_c, llm)

        # Should return fallback due to LLM error
        # (only if there were conflicts that needed resolution)
        if metrics.get("conflict", 0) > 0:
            assert result is None
            assert metrics["fallback_reason"] == "llm_error"

    @patch("config.Config")
    def test_token_savings_estimate(self, mock_config):
        """Verify token savings are reported when blocks agree."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.4
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.5

        llm = MagicMock()
        md = "# Title\n\nIdentical text in all three extractors."
        result, metrics = merge_with_consensus(md, md, md, llm)

        if result is not None:
            assert metrics["tokens_saved_estimate"] > 0


# ---------------------------------------------------------------------------
# Source preference tests
# ---------------------------------------------------------------------------

class TestSourcePreference:
    def test_figure_ref_prefers_marker(self):
        """figure_ref blocks should prefer Marker source text."""
        assert _SOURCE_PREFERENCE["figure_ref"] == "marker"

        blocks = [
            Block(block_id="d", block_type="figure_ref",
                  raw_text="![Fig 1](docling_img.png)",
                  normalized_text="fig 1", heading_level=None,
                  order_index=0, source="docling"),
            Block(block_id="m", block_type="figure_ref",
                  raw_text="![Fig 1](marker_img.png)",
                  normalized_text="fig 1", heading_level=None,
                  order_index=0, source="marker"),
        ]
        result = _pick_preferred_text(blocks, "figure_ref")
        assert result == "![Fig 1](marker_img.png)"

    def test_source_md_preferred_over_raw_text(self):
        """When source_md is set, _pick_preferred_text should return it."""
        blocks = [
            Block(block_id="m", block_type="heading",
                  raw_text="Methods", normalized_text="methods",
                  heading_level=2, order_index=0, source="grobid",
                  source_md="## Methods"),
        ]
        result = _pick_preferred_text(blocks, "heading")
        assert result == "## Methods"

    def test_source_md_fallback_to_raw_text(self):
        """When source_md is empty, _pick_preferred_text should fall back to raw_text."""
        blocks = [
            Block(block_id="m", block_type="paragraph",
                  raw_text="plain text", normalized_text="plain text",
                  heading_level=None, order_index=0, source="marker"),
        ]
        result = _pick_preferred_text(blocks, "paragraph")
        assert result == "plain text"


# ---------------------------------------------------------------------------
# Markdown preservation tests
# ---------------------------------------------------------------------------

class TestMarkdownPreservation:
    def test_headings_preserved_in_parse(self):
        """parse_markdown should preserve heading markers in source_md."""
        blocks = parse_markdown("## Methods\n\nBody text.", "grobid")
        headings = [b for b in blocks if b.block_type == "heading"]
        assert len(headings) >= 1
        assert headings[0].source_md.startswith("## ")
        assert "Methods" in headings[0].source_md

    def test_image_refs_preserved_in_parse(self):
        """parse_markdown should preserve image markdown in source_md."""
        blocks = parse_markdown("![Figure 1](image.png)", "marker")
        assert len(blocks) >= 1
        has_image_md = any("![" in b.source_md and "](image.png)" in b.source_md
                          for b in blocks)
        assert has_image_md

    def test_bold_preserved_in_parse(self):
        """parse_markdown should preserve bold markers in source_md."""
        blocks = parse_markdown("This is **important** text.", "grobid")
        paragraphs = [b for b in blocks if b.block_type == "paragraph"]
        assert len(paragraphs) >= 1
        assert "**important**" in paragraphs[0].source_md

    @patch("config.Config")
    def test_heading_markers_survive_full_pipeline(self, mock_config):
        """Heading markers (##) should be present in final assembled output."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.4
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.5
        mock_config.CONSENSUS_ALWAYS_ESCALATE_TABLES = True

        llm = MagicMock()
        md = "# Title\n\n## Methods\n\nBody text of the methods section."
        result, metrics = merge_with_consensus(md, md, md, llm)

        assert result is not None
        assert "# Title" in result
        assert "## Methods" in result
        assert metrics["fallback_triggered"] is False
