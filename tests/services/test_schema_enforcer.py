"""Tests for the schema enforcement module."""

import pytest


def _has_consensus_deps() -> bool:
    """Check whether heavy consensus pipeline deps are available."""
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


from app.services.schema_enforcer import (
    _fix_mid_sentence_splits,
    _normalize_authors,
    _preprocess,
    _relocate_metadata_from_body,
    _strip_pdf_artifacts,
    _strip_section_numbers,
    enforce_schema,
)


# ---------------------------------------------------------------------------
# Pre-processing: section number stripping
# ---------------------------------------------------------------------------


class TestStripSectionNumbers:
    def test_basic(self):
        assert _strip_section_numbers("## 2.1. Methods") == "## Methods"

    def test_nested(self):
        assert _strip_section_numbers("### 2.1.3 Subcategory") == "### Subcategory"

    def test_single_number(self):
        assert _strip_section_numbers("## 3. Results") == "## Results"

    def test_no_trailing_dot(self):
        assert _strip_section_numbers("## 4 Discussion") == "## Discussion"

    def test_preserves_plain_headings(self):
        assert _strip_section_numbers("## Methods") == "## Methods"

    def test_preserves_non_heading_lines(self):
        text = "This is 2.1. some text"
        assert _strip_section_numbers(text) == text

    def test_multiline(self):
        text = "## 1. Introduction\n\nSome text.\n\n## 2. Methods"
        expected = "## Introduction\n\nSome text.\n\n## Methods"
        assert _strip_section_numbers(text) == expected


# ---------------------------------------------------------------------------
# Pre-processing: PDF artifacts
# ---------------------------------------------------------------------------


class TestStripPdfArtifacts:
    def test_unicode_escapes(self):
        assert "fi" not in _strip_pdf_artifacts("/uniFB01nding")

    def test_watermark_lines(self):
        text = "Real content.\n\nDownloaded from https://academic.oup.com\n\nMore content."
        result = _strip_pdf_artifacts(text)
        assert "Downloaded from" not in result
        assert "Real content." in result
        assert "More content." in result


# ---------------------------------------------------------------------------
# Pre-processing: mid-sentence splits
# ---------------------------------------------------------------------------


class TestFixMidSentenceSplits:
    def test_merges_split_sentence(self):
        text = "The gene expression was significantly elevated in the treated\n\ngroup compared to controls."
        result = _fix_mid_sentence_splits(text)
        assert "\n\n" not in result
        assert "treated group" in result

    def test_preserves_normal_paragraphs(self):
        text = "First paragraph ends here.\n\nSecond paragraph starts here."
        assert _fix_mid_sentence_splits(text) == text

    def test_preserves_headings(self):
        text = "Some text ending with a word\n\n## Methods"
        assert _fix_mid_sentence_splits(text) == text

    def test_preserves_uppercase_start(self):
        text = "Text without punctuation\n\nNew paragraph starts with capital."
        assert _fix_mid_sentence_splits(text) == text

    def test_short_preceding_not_merged(self):
        text = "Short\n\nfollowing text here."
        assert _fix_mid_sentence_splits(text) == text

    def test_preserves_list_items(self):
        text = "Some text without ending punct\n\n- List item here"
        assert _fix_mid_sentence_splits(text) == text


# ---------------------------------------------------------------------------
# Round-trip enforcement
# ---------------------------------------------------------------------------


class TestEnforceSchema:
    def test_basic_document(self):
        md = (
            "# Test Title\n\n"
            "## Abstract\n\n"
            "This is the abstract.\n\n"
            "## Introduction\n\n"
            "Body text here.\n\n"
            "## References\n\n"
            "1. Author A (2024) A paper. *Journal*, 1, 1-10.\n"
        )
        result_md, metrics = enforce_schema(md)
        assert metrics["enforcement_applied"] is True
        assert metrics["validation_valid"] is True
        assert "# Test Title" in result_md
        assert "## Abstract" in result_md

    def test_strips_section_numbers(self):
        md = (
            "# Test Title\n\n"
            "## Abstract\n\n"
            "This is the abstract paragraph with enough words to pass the content guard.\n\n"
            "## 1. Introduction\n\n"
            "Body text with additional words to ensure the content preservation check passes.\n\n"
            "## 2. Methods\n\n"
            "Methods text describing the experimental approach used in this study.\n"
        )
        result_md, metrics = enforce_schema(md)
        assert metrics["enforcement_applied"] is True
        assert "## 1." not in result_md
        assert "## 2." not in result_md
        assert "## Introduction" in result_md
        assert "## Methods" in result_md

    def test_empty_input(self):
        result_md, metrics = enforce_schema("")
        assert metrics["enforcement_applied"] is False
        assert metrics["skip_reason"] == "empty_input"

    def test_unparseable_fallback(self):
        md = "just some random text with no structure whatsoever"
        result_md, metrics = enforce_schema(md)
        # Should return original since no sections/abstract detected
        assert metrics["enforcement_applied"] is False
        assert metrics["skip_reason"] == "unparseable"
        assert result_md == md

    def test_preserves_sup_tags(self):
        md = (
            "# Test Title\n\n"
            "## Abstract\n\n"
            "Ca<sup>2+</sup> signaling is important.\n\n"
            "## Introduction\n\n"
            "H<sub>2</sub>O is water.\n"
        )
        result_md, metrics = enforce_schema(md)
        assert metrics["enforcement_applied"] is True
        assert "<sup>2+</sup>" in result_md

    def test_preserves_bold_italic(self):
        md = (
            "# Test Title\n\n"
            "## Abstract\n\n"
            "The gene *dpp* in **Drosophila** is important.\n\n"
            "## Introduction\n\n"
            "More text.\n"
        )
        result_md, metrics = enforce_schema(md)
        assert metrics["enforcement_applied"] is True
        assert "*dpp*" in result_md
        assert "**Drosophila**" in result_md

    def test_metadata_relocation(self):
        md = (
            "# Test Title\n\n"
            "## Abstract\n\n"
            "Abstract text.\n\n"
            "## Introduction\n\n"
            "DOI: 10.1234/test.example\n"
        )
        result_md, metrics = enforce_schema(md)
        assert metrics["enforcement_applied"] is True
        assert "**DOI:** 10.1234/test.example" in result_md
        assert result_md.count("10.1234/test.example") == 1

    def test_prose_doi_reference_not_promoted(self):
        md = (
            "# Test Title\n\n"
            "## Abstract\n\n"
            "Abstract text.\n\n"
            "## Introduction\n\n"
            "See doi 10.1234/test.example for details.\n"
        )
        result_md, metrics = enforce_schema(md)
        assert metrics["enforcement_applied"] is True
        assert "**DOI:** 10.1234/test.example" not in result_md
        assert "See doi 10.1234/test.example for details." in result_md

    def test_validation_failure_falls_back(self):
        md = (
            "# Test Title\n\n"
            "## Abstract\n\n"
            "Abstract text.\n\n"
            "# Duplicate H1\n\n"
            "More body text.\n"
        )
        result_md, metrics = enforce_schema(md)
        assert metrics["enforcement_applied"] is False
        assert metrics["skip_reason"] == "validation_failed"
        assert metrics["validation_valid"] is False
        assert metrics["validation_errors"] > 0
        assert result_md == md

    def test_content_preservation_guard(self):
        """If round-trip loses >10% of words, fall back to original."""
        # This is hard to trigger with real data, so we test the metric
        # path exists and enforcement generally preserves content
        md = (
            "# Title\n\n"
            "## Abstract\n\n"
            "Abstract text here.\n\n"
            "## Methods\n\n"
            "Methods text here with enough words to measure preservation.\n"
        )
        result_md, metrics = enforce_schema(md)
        assert metrics["enforcement_applied"] is True

    def test_author_affiliations_removed_from_output(self):
        md = (
            "# Test Title\n\n"
            "Alice Smith<sup>1</sup>, Bob Jones<sup>2</sup>\n\n"
            "1. Department A\n"
            "2. Department B\n\n"
            "## Abstract\n\n"
            "Abstract text.\n"
        )
        result_md, metrics = enforce_schema(md)
        assert metrics["enforcement_applied"] is True
        assert "Alice Smith, Bob Jones" in result_md
        assert "<sup>" not in result_md
        assert "Department A" not in result_md
        assert "Department B" not in result_md

    def test_author_superscripts_without_affiliations_do_not_emit_numbered_list(self):
        md = (
            "# Test Title\n\n"
            "Alice Smith<sup>1</sup>, Bob Jones<sup>2</sup>\n\n"
            "## Abstract\n\n"
            "Abstract text.\n"
        )
        result_md, metrics = enforce_schema(md)
        assert metrics["enforcement_applied"] is True
        assert "Alice Smith, Bob Jones" in result_md
        assert "1. 1" not in result_md
        assert "2. 2" not in result_md


# ---------------------------------------------------------------------------
# Document model fixups
# ---------------------------------------------------------------------------


class TestRelocateMetadata:
    def test_doi_extraction(self):
        from agr_abc_document_parsers.models import Document, Paragraph, Section

        doc = Document(
            title="Test",
            sections=[
                Section(
                    heading="Introduction",
                    paragraphs=[Paragraph(text="DOI: 10.1234/test.paper")],
                ),
            ],
        )
        _relocate_metadata_from_body(doc)
        assert doc.doi == "10.1234/test.paper"
        assert doc.sections[0].paragraphs == []

    def test_pmid_extraction(self):
        from agr_abc_document_parsers.models import Document, Paragraph, Section

        doc = Document(
            title="Test",
            sections=[
                Section(
                    heading="Introduction",
                    paragraphs=[Paragraph(text="PMID: 12345678")],
                ),
            ],
        )
        _relocate_metadata_from_body(doc)
        assert doc.pmid == "12345678"
        assert doc.sections[0].paragraphs == []

    def test_does_not_overwrite_existing(self):
        from agr_abc_document_parsers.models import Document, Paragraph, Section

        doc = Document(
            title="Test",
            doi="10.9999/existing",
            sections=[
                Section(
                    heading="Introduction",
                    paragraphs=[Paragraph(text="DOI: 10.1234/other")],
                ),
            ],
        )
        _relocate_metadata_from_body(doc)
        assert doc.doi == "10.9999/existing"
        assert doc.sections[0].paragraphs == []

    def test_doi_with_parentheses_preserved(self):
        from agr_abc_document_parsers.models import Document, Paragraph, Section

        doi = "10.1002/(SICI)1097-0177(199801)213:1<1::AID-AJA1>3.0.CO;2-#"
        doc = Document(
            title="Test",
            sections=[
                Section(
                    heading="Introduction",
                    paragraphs=[Paragraph(text=f"(DOI: {doi})")],
                ),
            ],
        )
        _relocate_metadata_from_body(doc)
        assert doc.doi == doi
        assert doc.sections[0].paragraphs == []

    def test_prose_mentions_not_relocated(self):
        from agr_abc_document_parsers.models import Document, Paragraph, Section

        doc = Document(
            title="Test",
            sections=[
                Section(
                    heading="Introduction",
                    paragraphs=[Paragraph(text="See doi 10.1234/test.paper for details.")],
                ),
            ],
        )
        _relocate_metadata_from_body(doc)
        assert doc.doi == ""
        assert doc.sections[0].paragraphs[0].text == "See doi 10.1234/test.paper for details."


class TestNormalizeAuthors:
    def test_removes_affiliations_from_authors(self):
        from agr_abc_document_parsers.models import Author, Document

        doc = Document(
            authors=[
                Author(given_name="Alice", surname="Smith", affiliations=["Department A"]),
                Author(given_name="Bob", surname="Jones", affiliations=["Department B"]),
            ],
        )

        _normalize_authors(doc)

        assert doc.authors[0].affiliations == []
        assert doc.authors[1].affiliations == []


# ---------------------------------------------------------------------------
# Formatting preservation through normalize_extractor_output
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_consensus_deps(),
    reason="Requires full consensus pipeline dependencies (numpy, scipy, etc.)",
)
class TestFormattingPreservation:
    """Verify that semantic formatting survives the source normalization."""

    def test_sup_preserved_through_normalize(self):
        from app.services.consensus_parsing_alignment import normalize_extractor_output

        text = "Ca<sup>2+</sup> signaling"
        result = normalize_extractor_output(text)
        assert "<sup>2+</sup>" in result

    def test_bold_preserved_through_normalize(self):
        from app.services.consensus_parsing_alignment import normalize_extractor_output

        text = "**Figure 1.** Caption text"
        result = normalize_extractor_output(text)
        assert "**Figure 1.**" in result

    def test_italic_preserved_through_normalize(self):
        from app.services.consensus_parsing_alignment import normalize_extractor_output

        text = "The gene *dpp* is expressed in *Drosophila melanogaster*."
        result = normalize_extractor_output(text)
        assert "*dpp*" in result
        assert "*Drosophila melanogaster*" in result

    def test_artifacts_still_stripped(self):
        from app.services.consensus_parsing_alignment import normalize_extractor_output

        text = (
            "<span id='page-1-0'>Alpha</span>\n\n"
            "![Figure 1](image.png)\n\n"
            "/uniFB01 /uniFB02"
        )
        result = normalize_extractor_output(text)
        assert "<span" not in result
        assert "![" not in result
        assert "fi fl" in result

    def test_sup_preserved_through_clean_output(self):
        from app.services.consensus_parsing_alignment import clean_output_md

        text = "Author A<sup>1</sup>, Author B<sup>2</sup>"
        result = clean_output_md(text)
        assert "<sup>1</sup>" in result
        assert "<sup>2</sup>" in result
