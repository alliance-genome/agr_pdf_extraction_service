"""Tests for the selective LLM merge consensus pipeline."""

from unittest.mock import MagicMock, patch

from app.services.llm_service import TokenAccumulator
from app.services.consensus_service import (
    Block,
    AlignedTriple,
    AGREE_EXACT,
    AGREE_NEAR,
    GAP,
    CONFLICT,
    _SOURCE_PREFERENCE,
    _pick_preferred_text,
    _extract_numeric_tokens,
    _extract_citation_keys,
    _build_conflict_zones,
    clean_output_md,
    dedup_gap_triples,
    dedup_gap_against_all,
    dedup_assembled_paragraphs,
    normalize_text,
    normalize_extractor_output,
    parse_markdown,
    align_blocks,
    classify_triples,
    check_guards,
    assemble,
    compute_metrics,
    _build_audit_entries,
    merge_with_consensus,
    run_qa_gates,
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

    def test_strip_marker_span_artifacts(self):
        text = '<span id="page-0-3">Drosophila</span> sleep genes'
        assert normalize_text(text) == "drosophila sleep genes"

    def test_strip_image_and_comment_artifacts(self):
        text = "Alpha <!-- image --> ![fig](_page_4_Picture_7.jpeg) Beta"
        assert normalize_text(text) == "alpha beta"

    def test_strip_link_url_keep_text(self):
        text = "See [Sleep genes](https://doi.org/10.1234/example)."
        assert normalize_text(text) == "see sleep genes."

    def test_strip_heading_markers(self):
        assert normalize_text("# Methods") == normalize_text("#### Methods")

    def test_normalize_extractor_output_strips_artifacts(self):
        text = (
            "<span id='page-1-0'>Alpha</span>\n\n"
            "<!-- image -->\n\n"
            "![Figure 1](image.png)\n\n"
            "[DOI](https://doi.org/10.1234/x)\n\n"
            "/uniFB01 /uniFB02"
        )
        result = normalize_extractor_output(text)
        assert "Alpha" in result
        assert "<span" not in result
        assert "<!--" not in result
        assert "![" not in result
        assert "https://doi.org" not in result
        assert "fi fl" in result


class TestOutputCleaning:
    def test_clean_output_strips_span_and_comments(self):
        text = '<span id="page-1-0">Alpha</span> <!-- image --> Beta'
        assert clean_output_md(text) == "Alpha Beta"

    def test_clean_output_preserves_markdown(self):
        text = "# Heading\n\n**Bold** [Link](https://example.com) ![img](fig.png)"
        assert clean_output_md(text) == text


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

    def test_agree_near_allows_reference_number_variance_when_confident(self):
        """Figure/table reference-number drift should not force CONFLICT."""
        # Use realistic-length text so the fig/citation differences are a small
        # fraction of the total, producing high similarity scores (>0.97).
        # Only vary the figure number (not citations) to stay within the
        # exception's max-2-numeric-delta / max-1-citation-delta limits.
        base_a = (
            "The calcium signaling pathway was analyzed using fluorescence "
            "microscopy as shown in Fig. 1 and confirmed by western blot "
            "analysis [8]."
        )
        base_b = (
            "The calcium signaling pathway was analyzed using fluorescence "
            "microscopy as shown in Fig. 2 and confirmed by western blot "
            "analysis [8]."
        )
        t = self._make_triple(
            "seg_001",
            base_a,
            base_b,
            "A completely different paragraph that does not match.",
        )
        t.confidence = 0.8
        classify_triples([t], near_threshold=0.85, levenshtein_threshold=0.85)
        assert t.classification == AGREE_NEAR

    def test_reference_variance_exception_not_applied_to_citation_list_blocks(self):
        """Citation lists remain strict even when text similarity is high."""
        t = self._make_triple(
            "seg_001",
            "[8] Author A et al. 2020",
            "[9] Author A et al. 2020",
            "Different citation record text",
            block_type="citation_list",
        )
        t.confidence = 0.9
        classify_triples([t], near_threshold=0.85, levenshtein_threshold=0.85)
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

    def test_numeric_tokens_ignore_html_tag_attributes(self):
        assert _extract_numeric_tokens('<span id="page-0-3">25 6 7 5</span>') == {
            "25", "6", "7", "5",
        }

    def test_citation_keys_include_ranges_and_lists(self):
        keys = _extract_citation_keys("Prior work [8-10], [6,7], [Smith et al. 2024].")
        assert "[8-10]" in keys
        assert "[6,7]" in keys
        assert "[Smith et al. 2024]" in keys


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------

class TestGuards:
    def _make_classified_triples(self, classifications):
        """Helper to create triples with given classifications.

        All triples get 3 extractor blocks so block-family ratios are
        realistic and CONFLICT triples count as 3-source (not excluded
        from guard ratios).
        """
        triples = []
        for i, cls in enumerate(classifications):
            t = AlignedTriple(segment_id=f"seg_{i:03d}")
            t.classification = cls
            t.confidence = 0.8
            t.grobid_block = Block(block_id=f"g_{i}", raw_text="x", normalized_text="x", block_type="paragraph", heading_level=None, order_index=i, source="grobid")
            t.docling_block = Block(block_id=f"d_{i}", raw_text="y", normalized_text="y", block_type="paragraph", heading_level=None, order_index=i, source="docling")
            t.marker_block = Block(block_id=f"m_{i}", raw_text="z", normalized_text="z", block_type="paragraph", heading_level=None, order_index=i, source="marker")
            triples.append(t)
        return triples

    def _make_block_triple(self, seg_id, classification, block_type):
        t = AlignedTriple(segment_id=seg_id)
        t.classification = classification
        if classification != GAP:
            t.grobid_block = Block(
                block_id=f"g_{seg_id}",
                block_type=block_type,
                raw_text="Sample text",
                normalized_text=normalize_text("Sample text"),
                heading_level=None,
                order_index=0,
                source="grobid",
            )
            t.docling_block = Block(
                block_id=f"d_{seg_id}",
                block_type=block_type,
                raw_text="Sample text",
                normalized_text=normalize_text("Sample text"),
                heading_level=None,
                order_index=0,
                source="docling",
            )
            t.marker_block = Block(
                block_id=f"m_{seg_id}",
                block_type=block_type,
                raw_text="Sample text",
                normalized_text=normalize_text("Sample text"),
                heading_level=None,
                order_index=0,
                source="marker",
            )
        return t

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
        # 3/5 = 0.6 > 0.4 — textual check fires first since all blocks are paragraphs
        should_fallback, reason = check_guards(triples, alignment_confidence=0.7)
        assert should_fallback
        assert reason in ("conflict_ratio", "conflict_ratio_textual")

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
        assert reason in ("conflict_ratio", "conflict_ratio_textual")

    def test_all_gap_no_fallback(self):
        """All-GAP should have conflict_ratio = 0.0 (no fallback from ratio)."""
        triples = self._make_classified_triples([GAP, GAP, GAP])
        should_fallback, reason = check_guards(triples, alignment_confidence=0.7)
        assert not should_fallback

    def test_empty_triples_triggers_fallback(self):
        should_fallback, reason = check_guards([], alignment_confidence=0.7)
        assert should_fallback
        assert reason == "no_blocks"

    def test_structured_conflicts_use_structured_threshold(self):
        triples = [
            self._make_block_triple("seg_000", CONFLICT, "table"),
            self._make_block_triple("seg_001", CONFLICT, "table"),
            self._make_block_triple("seg_002", AGREE_EXACT, "table"),
            self._make_block_triple("seg_003", AGREE_EXACT, "table"),
        ]

        should_fallback, reason = check_guards(
            triples,
            alignment_confidence=0.8,
            conflict_ratio_threshold=0.9,  # disable global ratio fallback
            structured_conflict_ratio_threshold=0.75,
        )
        assert not should_fallback
        assert reason is None

        should_fallback, reason = check_guards(
            triples,
            alignment_confidence=0.8,
            conflict_ratio_threshold=0.9,
            structured_conflict_ratio_threshold=0.45,
        )
        assert should_fallback
        assert reason == "conflict_ratio_structured"

    def test_localized_conflict_relief_for_gap_heavy_runs(self):
        triples = []
        # Conflicts localized near the start.
        for i in range(20):
            triples.append(self._make_block_triple(f"seg_c_{i:03d}", CONFLICT, "paragraph"))
        # Large GAP region inflates non-gap ratio in current denominator logic.
        for i in range(20):
            t = AlignedTriple(segment_id=f"seg_g_{i:03d}")
            t.classification = GAP
            triples.append(t)
        # Healthy region later in the document.
        for i in range(20):
            triples.append(self._make_block_triple(f"seg_a_{i:03d}", AGREE_EXACT, "paragraph"))

        should_fallback, reason = check_guards(
            triples,
            alignment_confidence=0.8,
            conflict_ratio_threshold=0.4,
            textual_conflict_ratio_threshold=0.6,
            localized_conflict_span_max=0.35,
            localized_conflict_relief=0.15,
            localized_conflict_max_blocks=25,
        )
        assert not should_fallback
        assert reason is None


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

    def test_assembly_cleans_non_llm_segments(self):
        triples = [
            AlignedTriple(
                segment_id="seg_000",
                classification=AGREE_EXACT,
                agreed_text='<span id="page-1-0">Alpha</span>',
            ),
            AlignedTriple(
                segment_id="seg_001",
                classification=CONFLICT,
            ),
        ]
        result = assemble(triples, {"seg_001": "Resolved text"})
        assert "<span" not in result
        assert "Alpha" in result
        assert "Resolved text" in result


class TestGapDedup:
    def test_dedup_gap_triples_drops_shorter_duplicate(self):
        triples = [
            AlignedTriple(
                segment_id="seg_000",
                classification=GAP,
                agreed_text="Gill Institute for Neuroscience and Department of Biology, IU.",
            ),
            AlignedTriple(
                segment_id="seg_001",
                classification=AGREE_EXACT,
                agreed_text="## Methods",
            ),
            AlignedTriple(
                segment_id="seg_002",
                classification=GAP,
                agreed_text="Gill Institute for Neuroscience and Department of Biology.",
            ),
        ]

        removed = dedup_gap_triples(triples, window=3, similarity_threshold=0.85, length_ratio_threshold=0.7)
        assert removed == 1
        assert sum(1 for t in triples if t.classification == GAP and t.agreed_text) == 1


class TestQAGates:
    def test_run_qa_gates_detects_artifacts_and_duplicates(self):
        merged = (
            '<span id="page-0-1"></span>Repeated paragraph content that is definitely longer than fifty chars.\n\n'
            "Repeated paragraph content that is definitely longer than fifty chars.\n\n"
            "<!-- image -->"
        )
        qa = run_qa_gates(merged)
        assert qa["span_tag_count"] >= 1
        assert qa["html_comment_count"] >= 1
        assert qa["global_duplicate_count"] >= 1
        assert qa["qa_passed"] is False


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

    def test_metrics_include_guard_telemetry(self):
        triples = []
        for cls in [AGREE_EXACT, CONFLICT]:
            t = AlignedTriple(segment_id=f"seg_{len(triples):03d}")
            t.classification = cls
            triples.append(t)

        metrics = compute_metrics(
            triples,
            0.8,
            True,
            "conflict_ratio",
            guard_telemetry={
                "conflict_ratio_textual": 0.5,
                "conflict_ratio_structured": 0.0,
                "conflicts_localized": True,
                "conflict_span_ratio": 0.2,
                "adaptive_conflict_ratio_threshold": 0.55,
            },
        )
        assert metrics["conflict_ratio_textual"] == 0.5
        assert metrics["conflict_ratio_structured"] == 0.0
        assert metrics["conflicts_localized"] is True
        assert metrics["adaptive_conflict_ratio_threshold"] == 0.55


# ---------------------------------------------------------------------------
# Phase 2 metadata tests
# ---------------------------------------------------------------------------

class TestPhase2Metadata:
    def test_audit_includes_resolution_method_metadata(self):
        triple = AlignedTriple(segment_id="seg_001", classification=CONFLICT)
        triple.grobid_block = Block(
            block_id="g_001",
            block_type="paragraph",
            raw_text="Alpha text.",
            normalized_text=normalize_text("Alpha text."),
            heading_level=None,
            order_index=0,
            source="grobid",
        )
        triple.docling_block = Block(
            block_id="d_001",
            block_type="paragraph",
            raw_text="Alpha text updated.",
            normalized_text=normalize_text("Alpha text updated."),
            heading_level=None,
            order_index=0,
            source="docling",
        )

        audit = _build_audit_entries(
            [triple],
            {"seg_001": "Resolved text"},
            {"seg_001": {
                "method": "median_source",
                "confidence": 0.91,
                "sources_agreeing": ["docling", "grobid"],
            }},
        )

        assert len(audit) == 1
        assert audit[0]["chosen_source"] == "median_source"
        assert audit[0]["details"]["resolution_method"] == "median_source"
        assert audit[0]["details"]["resolution_confidence"] == 0.91


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
        llm.usage = TokenAccumulator()
        result, metrics, _audit = merge_with_consensus("grobid", "", "marker", llm)
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
        llm.usage = TokenAccumulator()
        result, metrics, _audit = merge_with_consensus("grobid", None, "marker", llm)
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
        llm.usage = TokenAccumulator()

        # Use identical text so everything agrees
        md = "# Title\n\nThis is the body text of the paper."
        result, metrics, _audit = merge_with_consensus(md, md, md, llm)

        assert result is not None
        assert "Title" in result
        assert "body text" in result
        assert metrics["fallback_triggered"] is False
        assert metrics["qa"]["qa_passed"] is True
        # LLM should NOT have been called for conflict resolution
        llm.resolve_conflicts.assert_not_called()

    @patch("config.Config")
    def test_full_pipeline_with_conflicts(self, mock_config):
        """When extractors disagree, LLM should be called for zone resolution."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8  # high threshold to avoid fallback
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.1  # low threshold
        mock_config.CONSENSUS_LAYERED_ENABLED = False

        llm = MagicMock()
        llm.usage = TokenAccumulator()

        md_a = "# Title\n\nThe accuracy was 95%."
        md_b = "# Title\n\nThe accuracy was 96%."
        md_c = "# Title\n\nThe accuracy was 97%."

        # Mock resolve_conflict_zones to return resolved text for any conflict/gap/near_agree segments
        def mock_resolve_zones(zones):
            result = {}
            for zone in zones:
                for seg in zone["segments"]:
                    if seg["status"] in ("conflict", "gap", "near_agree"):
                        result[seg["segment_id"]] = "The accuracy was 95.5%."
            return result
        llm.resolve_conflict_zones.side_effect = mock_resolve_zones

        result, metrics, _audit = merge_with_consensus(md_a, md_b, md_c, llm)

        if result is not None:
            # Pipeline succeeded with zone-based conflict resolution
            assert metrics["conflict"] >= 1
            llm.resolve_conflict_zones.assert_called()
        else:
            # Fallback triggered (acceptable if confidence too low)
            assert metrics["fallback_triggered"] is True

    @patch("config.Config")
    @patch("app.services.consensus_service._resolve_conflicts_layered")
    def test_layered_resolution_methods_are_reported(self, mock_layered, mock_config):
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.1
        mock_config.CONSENSUS_LAYERED_ENABLED = True
        mock_config.CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES = False
        mock_config.CONSENSUS_HIERARCHY_ENABLED = False

        llm = MagicMock()
        llm.usage = TokenAccumulator()
        mock_layered.return_value = (
            {"seg_001": "Resolved by layered resolver."},
            {
                "seg_001": {
                    "method": "median_source",
                    "confidence": 0.87,
                    "sources_agreeing": ["docling", "grobid", "marker"],
                },
            },
        )

        md_a = "# Title\n\nThe accuracy was 95%."
        md_b = "# Title\n\nThe accuracy was 96%."
        md_c = "# Title\n\nThe accuracy was 97%."

        result, metrics, audit = merge_with_consensus(md_a, md_b, md_c, llm)
        assert result is not None
        assert metrics["fallback_triggered"] is False
        assert metrics["resolution_methods"]["median_source"] == 1
        assert metrics["resolution_confidence_mean"] == 0.87
        assert any(
            entry.get("details", {}).get("resolution_method") == "median_source"
            for entry in audit
        )

    @patch("config.Config")
    def test_llm_error_triggers_fallback(self, mock_config):
        """LLM failure during conflict resolution should trigger fallback."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.1
        mock_config.CONSENSUS_LAYERED_ENABLED = False

        llm = MagicMock()
        llm.usage = TokenAccumulator()
        llm.resolve_conflict_zones.side_effect = Exception("LLM is down")

        md_a = "# Title\n\nThe accuracy was 95%."
        md_b = "# Title\n\nThe accuracy was 96%."
        md_c = "# Title\n\nThe accuracy was 97%."

        result, metrics, _audit = merge_with_consensus(md_a, md_b, md_c, llm)

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
        llm.usage = TokenAccumulator()
        md = "# Title\n\nIdentical text in all three extractors."
        result, metrics, _audit = merge_with_consensus(md, md, md, llm)

        if result is not None:
            assert metrics["tokens_saved_estimate"] > 0

    @patch("config.Config")
    def test_sparse_extractor_is_excluded_from_alignment(self, mock_config, caplog):
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.4
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.5
        mock_config.CONSENSUS_ALWAYS_ESCALATE_TABLES = True

        llm = MagicMock()
        llm.usage = TokenAccumulator()
        grobid_md = "# Tiny\n\nOnly one paragraph."

        shared_blocks = "\n\n".join([
            f"Paragraph {i}: matching content for docling and marker." for i in range(12)
        ])
        docling_md = f"# Title\n\n{shared_blocks}"
        marker_md = f"# Title\n\n{shared_blocks}"

        with caplog.at_level("WARNING"):
            result, metrics, _audit = merge_with_consensus(grobid_md, docling_md, marker_md, llm)

        assert result is not None
        assert metrics["fallback_triggered"] is False
        assert "excluding grobid from alignment" in caplog.text


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
        llm.usage = TokenAccumulator()
        md = "# Title\n\n## Methods\n\nBody text of the methods section."
        result, metrics, _audit = merge_with_consensus(md, md, md, llm)

        assert result is not None
        assert "# Title" in result
        assert "## Methods" in result
        assert metrics["fallback_triggered"] is False


# ---------------------------------------------------------------------------
# Global dedup: dedup_gap_against_all tests
# ---------------------------------------------------------------------------

class TestGlobalGapDedup:
    def _make_triple(self, seg_id, classification, agreed_text,
                     block_type="paragraph", source="grobid"):
        t = AlignedTriple(segment_id=seg_id, classification=classification)
        if agreed_text is not None:
            t.agreed_text = agreed_text
            block = Block(
                block_id=f"{source}_{seg_id}",
                block_type=block_type,
                raw_text=agreed_text,
                normalized_text=normalize_text(agreed_text),
                heading_level=1 if block_type == "heading" else None,
                order_index=0,
                source=source,
            )
            setattr(t, f"{source}_block", block)
        return t

    def test_dedup_gap_against_all_drops_contained_fragment(self):
        """GAP block whose text is a substring of an AGREE block gets blanked."""
        long_text = (
            "Proteomics relates the abundances of proteins to other biomolecules "
            "such as lipids or DNA and facilitates systems biology modeling of "
            "chemical processes underlying development and metabolism."
        )
        fragment = (
            "as lipids or DNA and facilitates systems biology modeling of "
            "chemical processes underlying development and metabolism."
        )
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, long_text),
            self._make_triple("seg_001", AGREE_EXACT, "## Methods", block_type="heading"),
            self._make_triple("seg_002", AGREE_NEAR, "Some other unique paragraph content here that is different."),
            # ... many blocks apart ...
            self._make_triple("seg_010", GAP, fragment, source="docling"),
        ]
        removed = dedup_gap_against_all(triples)
        assert removed == 1
        assert triples[3].agreed_text == ""  # the GAP fragment is blanked

    def test_dedup_gap_against_all_drops_near_equal_far_apart(self):
        """Two blocks far apart with high similarity — GAP gets blanked."""
        text_a = (
            "Opsin is amongst the three most abundant proteins we have quantified "
            "with an amount of 266 plus-minus 51 fmoles per eye in wild-type flies."
        )
        text_b = (
            "Opsin is amongst the three most abundant proteins we have quantified "
            "with an amount of 266 plus-minus 51 fmoles per eye in wild-type flies. "
            "This represents a substantial fraction of the total retinal protein pool."
        )
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, text_b),
            # Many blocks in between
            self._make_triple("seg_001", AGREE_EXACT, "Filler paragraph with unique content number one."),
            self._make_triple("seg_002", AGREE_EXACT, "Filler paragraph with unique content number two."),
            self._make_triple("seg_003", AGREE_EXACT, "Filler paragraph with unique content number three."),
            self._make_triple("seg_020", GAP, text_a, source="marker"),
        ]
        removed = dedup_gap_against_all(triples)
        assert removed == 1
        assert triples[4].agreed_text == ""

    def test_dedup_gap_against_all_preserves_unique_gap(self):
        """GAP block with genuinely unique content is NOT removed."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, "The first paragraph about protein quantification methods used in this study."),
            self._make_triple("seg_001", GAP, "This is completely unique content from only one extractor about a different topic entirely.", source="docling"),
        ]
        removed = dedup_gap_against_all(triples)
        assert removed == 0
        assert triples[1].agreed_text != ""

    def test_dedup_gap_against_all_skips_headings(self):
        """Heading-type GAP blocks are not deduplicated even if text overlaps."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, "Methods section describing the experimental approach used in great detail."),
            self._make_triple("seg_001", GAP, "## Methods", block_type="heading", source="docling"),
        ]
        removed = dedup_gap_against_all(triples)
        assert removed == 0
        assert triples[1].agreed_text == "## Methods"


# ---------------------------------------------------------------------------
# Global dedup: dedup_assembled_paragraphs tests
# ---------------------------------------------------------------------------

class TestAssembledDedup:
    def test_dedup_assembled_drops_non_adjacent_duplicate(self):
        """Post-assembly dedup catches non-adjacent duplicates."""
        text = (
            "# Introduction\n\n"
            "Proteomics relates the abundances of proteins to other biomolecules "
            "such as lipids or DNA and facilitates systems biology modeling.\n\n"
            "## Methods\n\n"
            "We used mass spectrometry to analyze protein samples from Drosophila eyes.\n\n"
            "## Results\n\n"
            "Proteomics relates the abundances of proteins to other biomolecules "
            "such as lipids or DNA and facilitates systems biology modeling.\n\n"
            "## Discussion\n\n"
            "The findings demonstrate the value of quantitative proteomics approaches."
        )
        cleaned, removed = dedup_assembled_paragraphs(text)
        assert removed == 1
        # The duplicate paragraph should appear only once
        assert cleaned.count("Proteomics relates the abundances") == 1
        # Headings should be preserved
        assert "# Introduction" in cleaned
        assert "## Methods" in cleaned
        assert "## Results" in cleaned
        assert "## Discussion" in cleaned

    def test_dedup_assembled_preserves_headings(self):
        """Headings are not stripped by post-assembly dedup even if similar."""
        text = (
            "# Methods\n\n"
            "The methods section describes the experimental approach.\n\n"
            "## Methods\n\n"
            "A different paragraph about something entirely unrelated to the heading above."
        )
        cleaned, removed = dedup_assembled_paragraphs(text)
        assert removed == 0
        assert "# Methods" in cleaned
        assert "## Methods" in cleaned


# ---------------------------------------------------------------------------
# Global dedup: QA gates global detection tests
# ---------------------------------------------------------------------------

class TestQAGatesGlobal:
    def test_qa_gates_global_duplicate_detection(self):
        """QA catches global (not just adjacent) duplicates."""
        merged = (
            "First unique paragraph that is definitely longer than fifty characters for testing.\n\n"
            "## Section Header\n\n"
            "Second unique paragraph that is also long enough to qualify for comparison.\n\n"
            "Some middle content that separates the duplicate paragraphs significantly.\n\n"
            "First unique paragraph that is definitely longer than fifty characters for testing."
        )
        qa = run_qa_gates(merged)
        assert qa["global_duplicate_count"] >= 1
        assert qa["qa_passed"] is False


# ---------------------------------------------------------------------------
# Global dedup: fail-hard pipeline tests
# ---------------------------------------------------------------------------

class TestFailHardDedup:
    @patch("config.Config")
    def test_fail_hard_on_surviving_duplicates(self, mock_config):
        """Pipeline returns None when dupes survive both layers and flag is True."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.1
        mock_config.CONSENSUS_ALWAYS_ESCALATE_TABLES = True
        mock_config.CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES = True

        llm = MagicMock()
        llm.usage = TokenAccumulator()

        # Craft inputs where a near-duplicate sneaks through from two different
        # extractors but with just enough variation in the surrounding context
        # that dedup layers don't catch it.  We mock dedup_assembled_paragraphs
        # to simulate duplicates surviving.
        md = "# Title\n\nBody text of the paper for testing."
        with patch(
            "app.services.consensus_service.dedup_assembled_paragraphs",
            return_value=("# Title\n\nBody text of the paper for testing.", 0),
        ), patch(
            "app.services.consensus_service.run_qa_gates",
            return_value={
                "span_tag_count": 0,
                "html_comment_count": 0,
                "global_duplicate_count": 2,
                "qa_passed": False,
            },
        ):
            result, metrics, _audit = merge_with_consensus(md, md, md, llm)

        assert result is None
        assert metrics["fallback_triggered"] is True
        assert metrics["fallback_reason"] == "global_duplicates"

    @patch("config.Config")
    def test_fail_hard_disabled_returns_result(self, mock_config):
        """Pipeline returns result when flag is False despite dupes in QA."""
        mock_config.CONSENSUS_NEAR_THRESHOLD = 0.92
        mock_config.CONSENSUS_LEVENSHTEIN_THRESHOLD = 0.90
        mock_config.CONSENSUS_CONFLICT_RATIO_FALLBACK = 0.8
        mock_config.CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK = 0.1
        mock_config.CONSENSUS_ALWAYS_ESCALATE_TABLES = True
        mock_config.CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES = False

        llm = MagicMock()
        llm.usage = TokenAccumulator()

        md = "# Title\n\nBody text of the paper for testing."
        with patch(
            "app.services.consensus_service.dedup_assembled_paragraphs",
            return_value=("# Title\n\nBody text of the paper for testing.", 0),
        ), patch(
            "app.services.consensus_service.run_qa_gates",
            return_value={
                "span_tag_count": 0,
                "html_comment_count": 0,
                "global_duplicate_count": 2,
                "qa_passed": False,
            },
        ):
            result, metrics, _audit = merge_with_consensus(md, md, md, llm)

        # With flag False, pipeline should return the result despite dupes
        assert result is not None
        assert metrics["fallback_triggered"] is False


# ---------------------------------------------------------------------------
# Conflict zone grouping tests
# ---------------------------------------------------------------------------

class TestBuildConflictZones:
    """Tests for _build_conflict_zones()."""

    def _make_triple(self, seg_id, classification, agreed_text=None,
                     grobid_text=None, docling_text=None, marker_text=None,
                     block_type="paragraph"):
        """Helper to create a classified triple."""
        t = AlignedTriple(segment_id=seg_id, classification=classification)
        t.agreed_text = agreed_text
        if grobid_text is not None:
            t.grobid_block = Block(
                block_id=f"g_{seg_id}", block_type=block_type, raw_text=grobid_text,
                normalized_text=normalize_text(grobid_text),
                heading_level=None, order_index=0, source="grobid",
                source_md=grobid_text,
            )
        if docling_text is not None:
            t.docling_block = Block(
                block_id=f"d_{seg_id}", block_type=block_type, raw_text=docling_text,
                normalized_text=normalize_text(docling_text),
                heading_level=None, order_index=0, source="docling",
                source_md=docling_text,
            )
        if marker_text is not None:
            t.marker_block = Block(
                block_id=f"m_{seg_id}", block_type=block_type, raw_text=marker_text,
                normalized_text=normalize_text(marker_text),
                heading_level=None, order_index=0, source="marker",
                source_md=marker_text,
            )
        return t

    def test_adjacent_conflicts_grouped(self):
        """Two adjacent CONFLICT triples should be in one zone."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, agreed_text="Before text A"),
            self._make_triple("seg_001", AGREE_EXACT, agreed_text="Before text B"),
            self._make_triple("seg_002", CONFLICT, grobid_text="gA", docling_text="dA", marker_text="mA"),
            self._make_triple("seg_003", CONFLICT, grobid_text="gB", docling_text="dB", marker_text="mB"),
            self._make_triple("seg_004", AGREE_EXACT, agreed_text="After text A"),
            self._make_triple("seg_005", AGREE_EXACT, agreed_text="After text B"),
        ]
        zones = _build_conflict_zones(triples, flanking_count=2)
        assert len(zones) == 1
        zone = zones[0]
        conflict_ids = [s["segment_id"] for s in zone["segments"] if s["status"] == "conflict"]
        assert conflict_ids == ["seg_002", "seg_003"]
        assert len(zone["context_before"]) == 2
        assert len(zone["context_after"]) == 2

    def test_gap_between_conflicts_absorbed(self):
        """A GAP between two CONFLICTs should be absorbed into the zone."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, agreed_text="Before"),
            self._make_triple("seg_001", AGREE_EXACT, agreed_text="Before 2"),
            self._make_triple("seg_002", CONFLICT, grobid_text="g1", docling_text="d1", marker_text="m1"),
            self._make_triple("seg_003", GAP, agreed_text="Gap text"),
            self._make_triple("seg_004", CONFLICT, grobid_text="g2", docling_text="d2", marker_text="m2"),
            self._make_triple("seg_005", AGREE_EXACT, agreed_text="After"),
            self._make_triple("seg_006", AGREE_EXACT, agreed_text="After 2"),
        ]
        # GAP between conflicts within flanking_count should be absorbed
        zones = _build_conflict_zones(triples, flanking_count=2)
        assert len(zones) == 1
        seg_ids = [s["segment_id"] for s in zones[0]["segments"]]
        assert "seg_003" in seg_ids  # GAP absorbed into zone

    def test_isolated_conflict_gets_flanking(self):
        """A single isolated CONFLICT gets a zone of size 1 with flanking."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, agreed_text="A"),
            self._make_triple("seg_001", AGREE_EXACT, agreed_text="B"),
            self._make_triple("seg_002", CONFLICT, grobid_text="g", docling_text="d", marker_text="m"),
            self._make_triple("seg_003", AGREE_EXACT, agreed_text="C"),
            self._make_triple("seg_004", AGREE_EXACT, agreed_text="D"),
        ]
        zones = _build_conflict_zones(triples, flanking_count=2)
        assert len(zones) == 1
        zone = zones[0]
        conflict_segs = [s for s in zone["segments"] if s["status"] == "conflict"]
        assert len(conflict_segs) == 1
        assert conflict_segs[0]["segment_id"] == "seg_002"
        assert len(zone["context_before"]) == 2
        assert len(zone["context_after"]) == 2

    def test_agree_exact_between_conflicts_creates_separate_zones(self):
        """An AGREE_EXACT between two conflicts keeps them in separate zones."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, agreed_text="A"),
            self._make_triple("seg_001", AGREE_EXACT, agreed_text="B"),
            self._make_triple("seg_002", CONFLICT, grobid_text="g1", docling_text="d1", marker_text="m1"),
            self._make_triple("seg_003", AGREE_EXACT, agreed_text="Middle"),  # AGREE_EXACT boundary
            self._make_triple("seg_004", CONFLICT, grobid_text="g2", docling_text="d2", marker_text="m2"),
            self._make_triple("seg_005", AGREE_EXACT, agreed_text="C"),
            self._make_triple("seg_006", AGREE_EXACT, agreed_text="D"),
        ]
        zones = _build_conflict_zones(triples, flanking_count=2)
        # AGREE_EXACT at seg_003 is a trusted boundary — zones stay separate
        assert len(zones) == 2
        assert zones[0]["segments"][0]["segment_id"] == "seg_002"
        assert zones[1]["segments"][0]["segment_id"] == "seg_004"

    def test_non_exact_between_conflicts_merges_zones(self):
        """Non-AGREE_EXACT segments between conflicts cause zones to merge."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, agreed_text="A"),
            self._make_triple("seg_001", AGREE_EXACT, agreed_text="B"),
            self._make_triple("seg_002", CONFLICT, grobid_text="g1", docling_text="d1", marker_text="m1"),
            self._make_triple("seg_003", AGREE_NEAR, agreed_text="Middle",
                             grobid_text="Middle", docling_text="Middle.", marker_text="Middle"),
            self._make_triple("seg_004", CONFLICT, grobid_text="g2", docling_text="d2", marker_text="m2"),
            self._make_triple("seg_005", AGREE_EXACT, agreed_text="C"),
            self._make_triple("seg_006", AGREE_EXACT, agreed_text="D"),
        ]
        zones = _build_conflict_zones(triples, flanking_count=2)
        # AGREE_NEAR is not a boundary — both conflicts merge into one zone
        assert len(zones) == 1
        seg_ids = [s["segment_id"] for s in zones[0]["segments"]]
        assert "seg_002" in seg_ids
        assert "seg_003" in seg_ids  # near_agree absorbed
        assert "seg_004" in seg_ids
        # Verify the absorbed AGREE_NEAR has near_agree status with extractor versions
        near_seg = [s for s in zones[0]["segments"] if s["segment_id"] == "seg_003"][0]
        assert near_seg["status"] == "near_agree"
        assert "grobid" in near_seg
        assert "docling" in near_seg
        assert "marker" in near_seg

    def test_document_boundary_partial_flanking(self):
        """Zone at document start should have fewer flanking segments."""
        triples = [
            self._make_triple("seg_000", CONFLICT, grobid_text="g", docling_text="d", marker_text="m"),
            self._make_triple("seg_001", AGREE_EXACT, agreed_text="A"),
            self._make_triple("seg_002", AGREE_EXACT, agreed_text="B"),
        ]
        zones = _build_conflict_zones(triples, flanking_count=2)
        assert len(zones) == 1
        assert len(zones[0]["context_before"]) == 0  # no segments before
        assert len(zones[0]["context_after"]) == 2

    def test_document_end_boundary(self):
        """Zone at document end should have fewer flanking segments."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, agreed_text="A"),
            self._make_triple("seg_001", AGREE_EXACT, agreed_text="B"),
            self._make_triple("seg_002", CONFLICT, grobid_text="g", docling_text="d", marker_text="m"),
        ]
        zones = _build_conflict_zones(triples, flanking_count=2)
        assert len(zones) == 1
        assert len(zones[0]["context_before"]) == 2
        assert len(zones[0]["context_after"]) == 0

    def test_multiple_disjoint_zones(self):
        """Well-separated conflicts produce separate zones.

        Need at least 2*flanking_count+1 agreed segments between conflicts
        so the flanking windows don't overlap or become adjacent.
        """
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, agreed_text="A"),
            self._make_triple("seg_001", AGREE_EXACT, agreed_text="B"),
            self._make_triple("seg_002", CONFLICT, grobid_text="g1", docling_text="d1", marker_text="m1"),
            self._make_triple("seg_003", AGREE_EXACT, agreed_text="C"),
            self._make_triple("seg_004", AGREE_EXACT, agreed_text="D"),
            self._make_triple("seg_005", AGREE_EXACT, agreed_text="E"),  # buffer
            self._make_triple("seg_006", AGREE_EXACT, agreed_text="F"),
            self._make_triple("seg_007", AGREE_EXACT, agreed_text="G"),
            self._make_triple("seg_008", CONFLICT, grobid_text="g2", docling_text="d2", marker_text="m2"),
            self._make_triple("seg_009", AGREE_EXACT, agreed_text="H"),
            self._make_triple("seg_010", AGREE_EXACT, agreed_text="I"),
        ]
        zones = _build_conflict_zones(triples, flanking_count=2)
        assert len(zones) == 2
        conflict_ids_0 = [s["segment_id"] for s in zones[0]["segments"] if s["status"] == "conflict"]
        conflict_ids_1 = [s["segment_id"] for s in zones[1]["segments"] if s["status"] == "conflict"]
        assert "seg_002" in conflict_ids_0
        assert "seg_008" in conflict_ids_1

    def test_no_conflicts_returns_empty(self):
        """When there are no CONFLICT triples, returns empty list."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, agreed_text="A"),
            self._make_triple("seg_001", GAP, agreed_text="B"),
            self._make_triple("seg_002", AGREE_NEAR, agreed_text="C"),
        ]
        zones = _build_conflict_zones(triples, flanking_count=2)
        assert len(zones) == 0

    def test_near_agree_in_zone_has_extractor_versions(self):
        """AGREE_NEAR segments in zones should have all extractor versions and near_agree status."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, agreed_text="Before"),
            self._make_triple("seg_001", AGREE_NEAR, agreed_text="Near text",
                             grobid_text="Near text", docling_text="Near text.", marker_text="Near text"),
            self._make_triple("seg_002", CONFLICT, grobid_text="g", docling_text="d", marker_text="m"),
            self._make_triple("seg_003", AGREE_EXACT, agreed_text="After"),
        ]
        zones = _build_conflict_zones(triples, flanking_count=2)
        assert len(zones) == 1
        # Near agree segment should be in the core with status "near_agree"
        near_segs = [s for s in zones[0]["segments"] if s["status"] == "near_agree"]
        assert len(near_segs) == 1
        assert near_segs[0]["segment_id"] == "seg_001"
        assert "grobid" in near_segs[0]
        assert "docling" in near_segs[0]
        assert "marker" in near_segs[0]
        assert "current_choice" in near_segs[0]

    def test_gap_between_conflicts_no_exact_boundary(self):
        """GAP between conflicts should NOT create a boundary (GAP is not AGREE_EXACT)."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, agreed_text="A"),
            self._make_triple("seg_001", CONFLICT, grobid_text="g1", docling_text="d1", marker_text="m1"),
            self._make_triple("seg_002", GAP, agreed_text="Gap"),
            self._make_triple("seg_003", CONFLICT, grobid_text="g2", docling_text="d2", marker_text="m2"),
            self._make_triple("seg_004", AGREE_EXACT, agreed_text="B"),
        ]
        zones = _build_conflict_zones(triples, flanking_count=2)
        assert len(zones) == 1
        seg_ids = [s["segment_id"] for s in zones[0]["segments"]]
        assert seg_ids == ["seg_001", "seg_002", "seg_003"]

    def test_zone_segment_structure(self):
        """Verify zone segment dicts have the expected keys."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, agreed_text="Before"),
            self._make_triple("seg_001", CONFLICT, grobid_text="g", docling_text="d", marker_text="m"),
            self._make_triple("seg_002", AGREE_EXACT, agreed_text="After"),
        ]
        zones = _build_conflict_zones(triples, flanking_count=1)
        assert len(zones) == 1
        seg = zones[0]["segments"][0]
        assert seg["status"] == "conflict"
        assert "grobid" in seg
        assert "docling" in seg
        assert "marker" in seg
        assert "segment_id" in seg

    def test_isolated_near_agree_not_in_zones(self):
        """An AGREE_NEAR separated from conflicts by AGREE_EXACT should NOT be in any zone."""
        triples = [
            self._make_triple("seg_000", AGREE_EXACT, agreed_text="A"),
            self._make_triple("seg_001", AGREE_NEAR, agreed_text="Near",
                             grobid_text="Near", docling_text="Near.", marker_text="Near"),
            self._make_triple("seg_002", AGREE_EXACT, agreed_text="B"),
            self._make_triple("seg_003", CONFLICT, grobid_text="g", docling_text="d", marker_text="m"),
            self._make_triple("seg_004", AGREE_EXACT, agreed_text="C"),
        ]
        zones = _build_conflict_zones(triples, flanking_count=2)
        assert len(zones) == 1
        # Only the conflict should be in the zone, not the isolated near_agree
        all_seg_ids = [s["segment_id"] for s in zones[0]["segments"]]
        assert "seg_001" not in all_seg_ids
        assert "seg_003" in all_seg_ids
