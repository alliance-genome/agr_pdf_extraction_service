"""Tests for header hierarchy resolution in the consensus pipeline."""

import re
from unittest.mock import MagicMock, patch

import pytest

from app.services.consensus_service import (
    _HEADING_LINE_RE,
    _validate_hierarchy_decisions,
    resolve_header_hierarchy,
)
from app.services.llm_service import (
    HeaderDecision,
    HeaderHierarchyResponse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MD = """\
## Quantitative proteomics of Drosophila melanogaster

Some introductory text about the paper.

## Abstract

This study examines protein expression...

## 1. Introduction

Background information here.

## 2. Results and Discussion

Main findings paragraph.

## 2.1. Fly Strains

We used Oregon-R and Canton-S strains...

## 2.2. Protein Extraction

Proteins were extracted using...

## 3. Methods

Detailed methods follow.

## References

1. Smith et al. (2024)...

## https://doi.org/10.1234/example

## journal@publisher.com
"""


# ---------------------------------------------------------------------------
# Test: Heading extraction regex
# ---------------------------------------------------------------------------

class TestHeadingExtraction:
    def test_extracts_all_headings(self):
        matches = list(_HEADING_LINE_RE.finditer(SAMPLE_MD))
        texts = [m.group(2).strip() for m in matches]
        assert len(texts) == 10
        assert texts[0] == "Quantitative proteomics of Drosophila melanogaster"
        assert texts[1] == "Abstract"
        assert texts[4] == "2.1. Fly Strains"
        assert texts[8] == "https://doi.org/10.1234/example"
        assert texts[9] == "journal@publisher.com"

    def test_captures_heading_level(self):
        md = "# Title\n\n## Section\n\n### Subsection\n\n#### Deep"
        matches = list(_HEADING_LINE_RE.finditer(md))
        levels = [len(m.group(1)) for m in matches]
        assert levels == [1, 2, 3, 4]

    def test_no_headings(self):
        md = "Just plain text.\n\nAnother paragraph."
        matches = list(_HEADING_LINE_RE.finditer(md))
        assert len(matches) == 0


# ---------------------------------------------------------------------------
# Test: Validation rules
# ---------------------------------------------------------------------------

class TestValidateHierarchyDecisions:
    def _make_decision(self, index, text, action, new_level=None):
        return HeaderDecision(
            heading_index=index,
            original_text=text,
            action=action,
            new_level=new_level,
        )

    def test_valid_decisions(self):
        decisions = [
            self._make_decision(0, "Paper Title", "set_level", 1),
            self._make_decision(1, "Abstract", "set_level", 2),
            self._make_decision(2, "Introduction", "set_level", 2),
            self._make_decision(3, "2.1. Methods", "set_level", 3),
        ]
        headers = [
            {"index": 0, "text": "Paper Title"},
            {"index": 1, "text": "Abstract"},
            {"index": 2, "text": "Introduction"},
            {"index": 3, "text": "2.1. Methods"},
        ]
        assert _validate_hierarchy_decisions(decisions, 4, headers) is None

    def test_wrong_count(self):
        decisions = [
            self._make_decision(0, "Title", "set_level", 1),
        ]
        err = _validate_hierarchy_decisions(decisions, 3)
        assert "expected 3 decisions, got 1" in err

    def test_no_h1(self):
        decisions = [
            self._make_decision(0, "Intro", "set_level", 2),
            self._make_decision(1, "Methods", "set_level", 2),
        ]
        err = _validate_hierarchy_decisions(decisions, 2)
        assert "expected exactly 1 H1" in err

    def test_multiple_h1(self):
        decisions = [
            self._make_decision(0, "Title A", "set_level", 1),
            self._make_decision(1, "Title B", "set_level", 1),
        ]
        err = _validate_hierarchy_decisions(decisions, 2)
        assert "expected exactly 1 H1" in err

    def test_level_jump(self):
        decisions = [
            self._make_decision(0, "Title", "set_level", 1),
            self._make_decision(1, "Deep", "set_level", 3),  # jumps from 1 to 3
        ]
        err = _validate_hierarchy_decisions(decisions, 2)
        assert "level jump from 1 to 3" in err

    def test_level_jump_allowed_with_intermediate(self):
        decisions = [
            self._make_decision(0, "Title", "set_level", 1),
            self._make_decision(1, "Section", "set_level", 2),
            self._make_decision(2, "Sub", "set_level", 3),
        ]
        assert _validate_hierarchy_decisions(decisions, 3) is None

    def test_too_many_demotions(self):
        decisions = [
            self._make_decision(0, "Title", "set_level", 1),
            self._make_decision(1, "DOI", "demote_to_text"),
            self._make_decision(2, "URL", "demote_to_text"),
            self._make_decision(3, "Email", "demote_to_text"),
        ]
        err = _validate_hierarchy_decisions(decisions, 4)
        assert "too many demotions" in err

    def test_text_fidelity_check_passes(self):
        decisions = [
            self._make_decision(0, "My Title", "set_level", 1),
            self._make_decision(1, "Abstract", "set_level", 2),
        ]
        headers = [
            {"index": 0, "text": "My Title"},
            {"index": 1, "text": "Abstract"},
        ]
        assert _validate_hierarchy_decisions(decisions, 2, headers) is None

    def test_text_fidelity_check_fails_on_modification(self):
        """CRITICAL: LLM must not rename headings."""
        decisions = [
            self._make_decision(0, "Corrected Title", "set_level", 1),  # LLM changed text!
            self._make_decision(1, "Abstract", "set_level", 2),
        ]
        headers = [
            {"index": 0, "text": "My Original Title"},
            {"index": 1, "text": "Abstract"},
        ]
        err = _validate_hierarchy_decisions(decisions, 2, headers)
        assert "LLM modified heading text" in err

    def test_text_fidelity_check_fails_on_index_out_of_range(self):
        decisions = [
            self._make_decision(0, "Title", "set_level", 1),
            self._make_decision(5, "Ghost", "set_level", 2),  # no index 5
        ]
        headers = [
            {"index": 0, "text": "Title"},
            {"index": 1, "text": "Section"},
        ]
        err = _validate_hierarchy_decisions(decisions, 2, headers)
        assert "do not cover" in err  # fails index completeness before range check

    def test_keep_level_skips_jump_check_without_original_levels(self):
        """keep_level headings without original_levels reset tracking."""
        decisions = [
            self._make_decision(0, "Title", "set_level", 1),
            self._make_decision(1, "Unknown", "keep_level"),
            self._make_decision(2, "Sub", "set_level", 3),
        ]
        # No jump error because index 1 is keep_level (unknown level)
        assert _validate_hierarchy_decisions(decisions, 3) is None

    def test_keep_level_uses_original_levels_for_jump_check(self):
        """When original_levels provided, keep_level resolves to actual level."""
        decisions = [
            self._make_decision(0, "Title", "set_level", 1),
            self._make_decision(1, "Section", "keep_level"),
            self._make_decision(2, "Sub", "set_level", 3),
        ]
        # original_levels: index 1 was ## (level 2), so 2→3 is valid
        assert _validate_hierarchy_decisions(decisions, 3, original_levels=[1, 2, 3]) is None

    def test_keep_level_on_existing_h1_counts_as_title(self):
        """keep_level on an already-# heading should count as the H1 title."""
        decisions = [
            self._make_decision(0, "Title", "keep_level"),  # original is #
            self._make_decision(1, "Section", "set_level", 2),
        ]
        # With original_levels, keep_level on level-1 counts as H1
        assert _validate_hierarchy_decisions(decisions, 2, original_levels=[1, 2]) is None

    def test_duplicate_indices_rejected(self):
        """Duplicate heading_index values must be caught."""
        decisions = [
            self._make_decision(0, "Title", "set_level", 1),
            self._make_decision(0, "Title", "set_level", 2),  # duplicate!
        ]
        err = _validate_hierarchy_decisions(decisions, 2)
        assert "duplicate" in err

    def test_non_contiguous_indices_rejected(self):
        """Indices must be exactly 0..n-1."""
        decisions = [
            self._make_decision(0, "Title", "set_level", 1),
            self._make_decision(2, "Section", "set_level", 2),  # skips 1
        ]
        err = _validate_hierarchy_decisions(decisions, 2)
        assert "do not cover" in err

    def test_set_level_with_none_new_level_rejected(self):
        """set_level must have a new_level value."""
        decisions = [
            self._make_decision(0, "Title", "set_level", 1),
            self._make_decision(1, "Section", "set_level", None),  # missing!
        ]
        err = _validate_hierarchy_decisions(decisions, 2)
        assert "set_level requires new_level" in err

    def test_set_level_out_of_range_rejected(self):
        """new_level must be 1-6."""
        decisions = [
            self._make_decision(0, "Title", "set_level", 1),
            self._make_decision(1, "Section", "set_level", 7),  # invalid
        ]
        err = _validate_hierarchy_decisions(decisions, 2)
        assert "out of range (1-6)" in err

    def test_set_level_zero_rejected(self):
        """new_level=0 would produce no heading marker."""
        decisions = [
            self._make_decision(0, "Title", "set_level", 1),
            self._make_decision(1, "Section", "set_level", 0),
        ]
        err = _validate_hierarchy_decisions(decisions, 2)
        assert "out of range (1-6)" in err

    # --- detected_title tests ---

    def test_detected_title_with_zero_h1_passes(self):
        """When detected_title is set, 0 H1 headings among decisions is valid."""
        decisions = [
            self._make_decision(0, "Abstract", "set_level", 2),
            self._make_decision(1, "Introduction", "set_level", 2),
            self._make_decision(2, "Methods", "set_level", 2),
        ]
        err = _validate_hierarchy_decisions(
            decisions, 3, detected_title="Proteomics of Drosophila melanogaster",
        )
        assert err is None

    def test_detected_title_with_h1_heading_fails(self):
        """When detected_title is set, any H1 heading is rejected."""
        decisions = [
            self._make_decision(0, "Title Heading", "set_level", 1),
            self._make_decision(1, "Abstract", "set_level", 2),
        ]
        err = _validate_hierarchy_decisions(
            decisions, 2, detected_title="Proteomics of Drosophila melanogaster",
        )
        assert "detected_title is set but 1 headings are also H1" in err

    def test_no_detected_title_and_zero_h1_fails(self):
        """Without detected_title, 0 H1 headings is still invalid."""
        decisions = [
            self._make_decision(0, "Abstract", "set_level", 2),
            self._make_decision(1, "Introduction", "set_level", 2),
        ]
        err = _validate_hierarchy_decisions(decisions, 2)
        assert "expected exactly 1 H1" in err


# ---------------------------------------------------------------------------
# Test: Apply function (set_level, demote_to_text, keep_level)
# ---------------------------------------------------------------------------

class TestApplyDecisions:
    """Test resolve_header_hierarchy with a mock LLM to verify apply logic."""

    def _mock_llm_with_response(self, response):
        """Create a mock LLM that returns a fixed HeaderHierarchyResponse."""
        llm = MagicMock()
        llm.resolve_header_hierarchy.return_value = response
        return llm

    @patch("config.Config")
    def test_set_level_changes_heading(self, mock_config):
        mock_config.CONSENSUS_HIERARCHY_ENABLED = True
        mock_config.HIERARCHY_LLM_MODEL = "gpt-5.2"
        mock_config.HIERARCHY_LLM_REASONING = "medium"

        md = "## Paper Title\n\nSome text.\n\n## Introduction\n\nMore text."
        response = HeaderHierarchyResponse(decisions=[
            HeaderDecision(heading_index=0, original_text="Paper Title", action="set_level", new_level=1),
            HeaderDecision(heading_index=1, original_text="Introduction", action="set_level", new_level=2),
        ])
        llm = self._mock_llm_with_response(response)

        result = resolve_header_hierarchy(md, llm)
        assert result.startswith("# Paper Title")
        assert "## Introduction" in result

    @patch("config.Config")
    def test_demote_to_text_strips_heading(self, mock_config):
        mock_config.CONSENSUS_HIERARCHY_ENABLED = True
        mock_config.HIERARCHY_LLM_MODEL = "gpt-5.2"
        mock_config.HIERARCHY_LLM_REASONING = "medium"

        md = (
            "## Title\n\nText.\n\n"
            "## Introduction\n\nIntro text.\n\n"
            "## https://doi.org/10.1234/example\n\nMore."
        )
        response = HeaderHierarchyResponse(decisions=[
            HeaderDecision(heading_index=0, original_text="Title", action="set_level", new_level=1),
            HeaderDecision(heading_index=1, original_text="Introduction", action="set_level", new_level=2),
            HeaderDecision(heading_index=2, original_text="https://doi.org/10.1234/example", action="demote_to_text"),
        ])
        llm = self._mock_llm_with_response(response)

        result = resolve_header_hierarchy(md, llm)
        assert "# Title" in result
        assert "## Introduction" in result
        # DOI line should be plain text now, no heading markers
        assert "## https://doi.org" not in result
        assert "https://doi.org/10.1234/example" in result

    @patch("config.Config")
    def test_keep_level_no_change(self, mock_config):
        """keep_level on existing # title is valid (H1 detected via original_levels)."""
        mock_config.CONSENSUS_HIERARCHY_ENABLED = True
        mock_config.HIERARCHY_LLM_MODEL = "gpt-5.2"
        mock_config.HIERARCHY_LLM_REASONING = "medium"

        md = "# Title\n\nText.\n\n## Methods\n\nMore."
        response = HeaderHierarchyResponse(decisions=[
            HeaderDecision(heading_index=0, original_text="Title", action="keep_level"),
            HeaderDecision(heading_index=1, original_text="Methods", action="keep_level"),
        ])
        llm = self._mock_llm_with_response(response)

        result = resolve_header_hierarchy(md, llm)
        # Both keep their original levels
        assert "# Title" in result
        assert "## Methods" in result

    @patch("config.Config")
    def test_subsection_gets_deeper_level(self, mock_config):
        mock_config.CONSENSUS_HIERARCHY_ENABLED = True
        mock_config.HIERARCHY_LLM_MODEL = "gpt-5.2"
        mock_config.HIERARCHY_LLM_REASONING = "medium"

        md = "## Title\n\nText.\n\n## Results\n\nText.\n\n## 2.1. Fly Strains\n\nStrains info."
        response = HeaderHierarchyResponse(decisions=[
            HeaderDecision(heading_index=0, original_text="Title", action="set_level", new_level=1),
            HeaderDecision(heading_index=1, original_text="Results", action="set_level", new_level=2),
            HeaderDecision(heading_index=2, original_text="2.1. Fly Strains", action="set_level", new_level=3),
        ])
        llm = self._mock_llm_with_response(response)

        result = resolve_header_hierarchy(md, llm)
        assert "# Title" in result
        assert "## Results" in result
        assert "### 2.1. Fly Strains" in result

    # --- detected_title apply tests ---

    @patch("config.Config")
    def test_detected_title_inserts_h1(self, mock_config):
        """When LLM returns detected_title, the title line in the document gets # prefix."""
        mock_config.CONSENSUS_HIERARCHY_ENABLED = True
        mock_config.HIERARCHY_LLM_MODEL = "gpt-5.2"
        mock_config.HIERARCHY_LLM_REASONING = "medium"

        md = "Proteomics of Drosophila\n\nSome intro text.\n\n## Abstract\n\nAbstract text.\n\n## Introduction\n\nIntro text."
        response = HeaderHierarchyResponse(
            decisions=[
                HeaderDecision(heading_index=0, original_text="Abstract", action="set_level", new_level=2),
                HeaderDecision(heading_index=1, original_text="Introduction", action="set_level", new_level=2),
            ],
            detected_title="Proteomics of Drosophila",
        )
        llm = self._mock_llm_with_response(response)

        result = resolve_header_hierarchy(md, llm)
        assert "# Proteomics of Drosophila" in result
        assert "## Abstract" in result
        assert "## Introduction" in result

    @patch("config.Config")
    def test_detected_title_not_found_as_line_prepends_h1(self, mock_config):
        """If detected_title isn't a standalone line, prepend it as H1."""
        mock_config.CONSENSUS_HIERARCHY_ENABLED = True
        mock_config.HIERARCHY_LLM_MODEL = "gpt-5.2"
        mock_config.HIERARCHY_LLM_REASONING = "medium"

        # Title is embedded in a sentence, not a standalone line
        md = "This paper about Proteomics is important.\n\n## Abstract\n\nText."
        response = HeaderHierarchyResponse(
            decisions=[
                HeaderDecision(heading_index=0, original_text="Abstract", action="set_level", new_level=2),
            ],
            detected_title="Proteomics of Drosophila",
        )
        llm = self._mock_llm_with_response(response)

        result = resolve_header_hierarchy(md, llm)
        # Title prepended as H1 at the top
        assert result.startswith("# Proteomics of Drosophila\n")
        assert "## Abstract" in result


# ---------------------------------------------------------------------------
# Test: Fallback on invalid LLM response
# ---------------------------------------------------------------------------

class TestFallbackBehavior:
    @patch("config.Config")
    def test_wrong_count_returns_original(self, mock_config):
        mock_config.CONSENSUS_HIERARCHY_ENABLED = True
        mock_config.HIERARCHY_LLM_MODEL = "gpt-5.2"
        mock_config.HIERARCHY_LLM_REASONING = "medium"

        md = "## Title\n\nText.\n\n## Section\n\nMore."
        # Return only 1 decision for 2 headings → validation fails
        response = HeaderHierarchyResponse(decisions=[
            HeaderDecision(heading_index=0, original_text="Title", action="set_level", new_level=1),
        ])
        llm = MagicMock()
        llm.resolve_header_hierarchy.return_value = response

        result = resolve_header_hierarchy(md, llm)
        assert result == md  # original unchanged

    @patch("config.Config")
    def test_fabricated_text_returns_original(self, mock_config):
        """CRITICAL: If LLM modifies heading text, reject everything."""
        mock_config.CONSENSUS_HIERARCHY_ENABLED = True
        mock_config.HIERARCHY_LLM_MODEL = "gpt-5.2"
        mock_config.HIERARCHY_LLM_REASONING = "medium"

        md = "## My Actual Title\n\nText.\n\n## Methods\n\nMore."
        response = HeaderHierarchyResponse(decisions=[
            HeaderDecision(heading_index=0, original_text="A Better Title", action="set_level", new_level=1),
            HeaderDecision(heading_index=1, original_text="Methods", action="set_level", new_level=2),
        ])
        llm = MagicMock()
        llm.resolve_header_hierarchy.return_value = response

        result = resolve_header_hierarchy(md, llm)
        assert result == md  # original unchanged — LLM tried to rename

    @patch("config.Config")
    def test_llm_exception_propagates(self, mock_config):
        """If LLM call itself throws, resolve_header_hierarchy should raise."""
        mock_config.CONSENSUS_HIERARCHY_ENABLED = True
        mock_config.HIERARCHY_LLM_MODEL = "gpt-5.2"
        mock_config.HIERARCHY_LLM_REASONING = "medium"

        md = "## Title\n\nText."
        llm = MagicMock()
        llm.resolve_header_hierarchy.side_effect = Exception("API timeout")

        with pytest.raises(Exception, match="API timeout"):
            resolve_header_hierarchy(md, llm)

    @patch("config.Config")
    def test_no_headings_returns_original(self, mock_config):
        mock_config.CONSENSUS_HIERARCHY_ENABLED = True
        mock_config.HIERARCHY_LLM_MODEL = "gpt-5.2"
        mock_config.HIERARCHY_LLM_REASONING = "medium"

        md = "Just plain text with no headings at all."
        llm = MagicMock()

        result = resolve_header_hierarchy(md, llm)
        assert result == md
        llm.resolve_header_hierarchy.assert_not_called()
