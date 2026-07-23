"""Baseline selection and exact source-span assembly."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import unicodedata
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Callable, Literal, Mapping

import regex
import mistune
from rapidfuzz.fuzz import ratio

from app.services.source_contracts import (
    CandidateGraph,
    CandidateStore,
    ConsensusContractError,
    MergeQuality,
    RegionCandidateGraph,
    RegionSelectionDecision,
    SelectionDecisionResponse,
    SourceArtifact,
    SourceName,
    SourceSpanCandidate,
    require_publishable_text,
    unsafe_unicode_characters,
)


StructuralUnitType = Literal[
    "heading",
    "paragraph",
    "list",
    "table",
    "table_cell",
    "fenced_block",
    "equation",
    "figure_caption",
    "reference",
]
DecisionMethod = Literal[
    "untouched_baseline",
    "deterministic",
    "model_selected",
    "baseline_fallback",
]
SelectionQuality = Literal["terra_selected", "sol_selected"]

_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$)")
_LIST_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)]|\[\d{1,4}\])\s+")
_REFERENCE_RE = re.compile(r"^\s*(?:\[\d{1,4}\]|\d{1,4}[.)])\s+")
_REFERENCE_YEAR_RE = re.compile(r"\(?(?:19|20)\d{2}[a-z]?\)?\b")
_REFERENCE_IDENTIFIER_RE = re.compile(
    r"^\s*(?:[-*]\s+)?(?:"
    r"(?:doi\s*:?[ \t]*)?(?:https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/\S+"
    r"|pmid\s*:?[ \t]*\d+\b)",
    re.IGNORECASE,
)
_REFERENCE_SURNAME = r"[A-ZÀ-ÖØ-Þ][\w'’.-]+"
_REFERENCE_INITIALS = r"[A-Z](?:[.-]?[A-Z])?\.?"
_REFERENCE_AUTHOR_INITIAL_RE = re.compile(
    rf"^{_REFERENCE_SURNAME}(?:\s*,\s*|\s+){_REFERENCE_INITIALS}(?:\s|,|$)"
)
_REFERENCE_AUTHOR_JOIN_RE = re.compile(
    rf"^{_REFERENCE_SURNAME}[^\n]{{0,100}}?(?:\band\b|\s*&)\s*{_REFERENCE_SURNAME}\b",
    re.IGNORECASE,
)
_FIGURE_RE = re.compile(
    r"^\s*(?:\*{1,2})?"
    r"(?:(?:supplementary|extended\s+data)\s+)?"
    r"(?:figure|fig\.)\s*[A-Za-z0-9]+",
    re.IGNORECASE,
)
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_WHITESPACE_RE = re.compile(r"\s+")
_ANCHOR_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_REFERENCE_FOOTNOTE_RE = re.compile(r"^\s*\[\^\d+\]:\s+")
_SOURCE_TIE_BREAK = {"grobid": 3, "docling": 2, "marker": 1}
_MAX_COMPARISON_UNIT_CHARS = 20_000
_ALIGNMENT_POSITIONAL_WINDOW = 48
_ALIGNMENT_RARE_ANCHOR_MAX_OCCURRENCES = 8
_ALIGNMENT_MAX_ANCHOR_CANDIDATES = 64
_MAX_TABLE_ROW_BYTES = 65_536
_MAX_TABLE_CELLS = 256
_ALIGNMENT_AMBIGUITY_DELTA = 3.0
_COMPOSITE_ALIGNMENT_MAX_UNITS = 8
_COMPOSITE_ALIGNMENT_UNIT_TYPES = frozenset(
    {"paragraph", "list", "reference", "figure_caption"}
)
_COMPOSITE_ALIGNMENT_ANCHOR_TYPES = frozenset(
    {*_COMPOSITE_ALIGNMENT_UNIT_TYPES, "heading"}
)
_SCOPE_NEIGHBORHOOD_UNIT_RADIUS = 64
_SCOPE_NEIGHBORHOOD_TOKEN_LIMIT = 100_000
_SCOPE_NONLOCAL_MIN_TOKENS = 5
_SCOPE_NONLOCAL_MIN_CHARACTERS = 24
_SCOPE_REPLACEMENT_MIN_SIMILARITY = 55.0
_REPETITION_POLICY_VERSION = "excess-repetition-v3"
_REPETITION_PARAGRAPH_MIN_NORMALIZED_CHARACTERS = 80
_REPETITION_PARAGRAPH_MIN_ANCHOR_TOKENS = 8
_REPETITION_SHINGLE_WIDTH = 12
_REPETITION_SHINGLE_MIN_ANCHOR_CHARACTERS = 48
_SOURCE_REPETITION_MIN_LONG_BLOCKS = 20
_SOURCE_REPETITION_MIN_EXCESS_BLOCKS = 8
_SOURCE_REPETITION_MIN_EXCESS_RATIO_PPM = 300_000
_SOURCE_REPETITION_CLEAN_PEER_COVERAGE_PPM = 350_000
_ALIGNMENT_CORE_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_ATOMIC_CLIP_UNIT_TYPES = frozenset(
    {"heading", "table", "table_cell", "fenced_block", "equation"}
)
_ITALIC_PRESERVATION_POLICY_VERSION = "union-source-italics-v4"
_ITALIC_PREFERENCE_UNIT_TYPES = frozenset(
    {
        "heading",
        "paragraph",
        "list",
        "table_cell",
        "figure_caption",
        "reference",
    }
)
_POTENTIAL_UNPROFILED_EMPHASIS_RE = re.compile(
    r"(?:<\s*/?\s*(?:em|i)\b|\\(?:textit|emph)\s*\{|"
    r"(?<!\\)(?:\*|_)(?=\S)[\s\S]*?(?<=\S)(?<!\\)(?:\*|_))",
    re.IGNORECASE,
)
@dataclass(frozen=True)
class _InlineItalicProfile:
    """Parsed visible coordinates plus content-free inline-markup identities."""

    visible_digest: str
    non_emphasis_ast_digest: str
    emphasis_occurrence_ids: tuple[str, ...]
    emphasis_spans: tuple[tuple[int, int], ...] = ()
    visible_text: str = ""


def _normalized_visible_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", value)).strip()


def _emphasis_occurrence_id(visible: str, start: int, end: int) -> str:
    """Return a content-free identity for one exact emphasis boundary."""

    prefix = _normalized_visible_text(visible[:start])
    payload = _normalized_visible_text(visible[start:end])
    suffix = _normalized_visible_text(visible[end:])
    if not payload:
        raise ValueError("emphasis payload is empty")
    return hashlib.sha256(
        "\x00".join((prefix, payload, suffix)).encode("utf-8")
    ).hexdigest()


def _canonical_non_emphasis_ast(ast: list[object]) -> list[object]:
    """Return a deterministic AST with only emphasis wrappers removed.

    Removing an emphasis wrapper can split one plain text node into adjacent
    nodes, so adjacent text leaves are coalesced. Every other node type,
    attribute, raw value, and container remains part of the digest.
    """

    def canonical_value(value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): canonical_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [canonical_value(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise ValueError("unsupported Markdown AST value")

    def coalesce_text(nodes: list[object]) -> list[object]:
        coalesced: list[object] = []
        for node in nodes:
            if (
                coalesced
                and isinstance(coalesced[-1], dict)
                and coalesced[-1].get("type") == "text"
                and isinstance(node, dict)
                and node.get("type") == "text"
            ):
                previous_raw = coalesced[-1].get("raw")
                current_raw = node.get("raw")
                if isinstance(previous_raw, str) and isinstance(current_raw, str):
                    coalesced[-1] = {**coalesced[-1], "raw": previous_raw + current_raw}
                    continue
            coalesced.append(node)
        return coalesced

    def canonical_nodes(value: object) -> list[object]:
        if isinstance(value, list):
            return coalesce_text(
                [node for item in value for node in canonical_nodes(item)]
            )
        if not isinstance(value, Mapping):
            raise ValueError("invalid Markdown AST node")
        node_type = value.get("type")
        if not isinstance(node_type, str):
            raise ValueError("Markdown AST node has no type")
        if node_type == "emphasis":
            children = value.get("children")
            if not isinstance(children, list):
                raise ValueError("emphasis node has no children")
            return canonical_nodes(children)
        node: dict[str, object] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            key_text = str(key)
            if key_text == "children":
                if not isinstance(item, list):
                    raise ValueError("Markdown AST children are not a list")
                node[key_text] = canonical_nodes(item)
            else:
                node[key_text] = canonical_value(item)
        return [node]

    return canonical_nodes(ast)


def _inline_italic_analysis(
    text: str,
    unit_type: StructuralUnitType,
) -> tuple[_InlineItalicProfile, str] | None:
    """Parse a safe Markdown unit without treating marker characters as text edits.

    The returned hashes contain no publication text.  Code, equations, whole
    tables, raw HTML blocks, and malformed/unknown AST leaves are deliberately
    ineligible for the deterministic formatting preference.
    """

    if unit_type not in _ITALIC_PREFERENCE_UNIT_TYPES:
        return None
    raw_text = text.encode("utf-8")
    content_start = _container_content_start(raw_text, unit_type)
    profile_text = raw_text[content_start:].decode("utf-8", errors="strict")
    try:
        ast = mistune.create_markdown(renderer="ast")(profile_text)
    except Exception:
        return None
    if not isinstance(ast, list):
        return None

    visible_parts: list[str] = []
    emphasis_ranges: list[tuple[int, int]] = []
    invalid = False

    def visible_length() -> int:
        return sum(len(part) for part in visible_parts)

    def walk(value: object) -> None:
        nonlocal invalid
        if isinstance(value, list):
            for item in value:
                walk(item)
            return
        if not isinstance(value, Mapping):
            invalid = True
            return
        node_type = value.get("type")
        if not isinstance(node_type, str):
            invalid = True
            return
        if node_type in {
            "block_code",
            "block_html",
            "thematic_break",
            "inline_math",
            "block_math",
        }:
            invalid = True
            return
        if node_type == "emphasis":
            start = visible_length()
            walk(value.get("children", []))
            end = visible_length()
            if end <= start:
                invalid = True
            else:
                emphasis_ranges.append((start, end))
            return
        if node_type in {"text", "codespan"}:
            raw = value.get("raw")
            if not isinstance(raw, str):
                invalid = True
            else:
                visible_parts.append(raw)
            return
        if node_type in {"softbreak", "linebreak"}:
            visible_parts.append("\n")
            return
        if node_type in {"blank_line", "inline_html"}:
            return
        children = value.get("children")
        if isinstance(children, list):
            walk(children)
            return
        raw = value.get("raw")
        if raw not in {None, ""}:
            invalid = True

    walk(ast)
    if invalid:
        return None
    visible = "".join(visible_parts)
    normalized_visible = _normalized_visible_text(visible)
    if not normalized_visible:
        return None
    visible_digest = hashlib.sha256(normalized_visible.encode("utf-8")).hexdigest()
    try:
        canonical_ast = _canonical_non_emphasis_ast(ast)
    except ValueError:
        return None
    non_emphasis_ast_digest = hashlib.sha256(
        json.dumps(
            canonical_ast,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    occurrence_ids = []
    for start, end in emphasis_ranges:
        try:
            occurrence_id = _emphasis_occurrence_id(visible, start, end)
        except ValueError:
            return None
        occurrence_ids.append(occurrence_id)
    return (
        _InlineItalicProfile(
            visible_digest=visible_digest,
            non_emphasis_ast_digest=non_emphasis_ast_digest,
            emphasis_occurrence_ids=tuple(occurrence_ids),
            emphasis_spans=tuple(emphasis_ranges),
            visible_text=visible,
        ),
        normalized_visible,
    )


def _inline_italic_profile(
    text: str,
    unit_type: StructuralUnitType,
) -> _InlineItalicProfile | None:
    analysis = _inline_italic_analysis(text, unit_type)
    return None if analysis is None else analysis[0]


def _composite_non_emphasis_ast_digest(
    artifact: "SourceArtifact",
    units: tuple["StructuralUnitSpan", ...],
) -> str | None:
    """Hash inline structure while ignoring paragraph segmentation and italics.

    Visible-text equality is proved separately. This digest exists only to stop
    bold, links, code, HTML, or other collateral inline markup from riding along
    with a deterministic italic preference when extractors split paragraphs at
    different boundaries.
    """

    inline_nodes: list[object] = []
    for ordinal, unit in enumerate(units):
        raw = artifact.raw_utf8[unit.byte_start:unit.byte_end]
        content_start = _container_content_start(raw, unit.unit_type)
        try:
            ast = mistune.create_markdown(renderer="ast")(
                raw[content_start:].decode("utf-8", errors="strict")
            )
            canonical = _canonical_non_emphasis_ast(ast)
        except (UnicodeError, ValueError, TypeError):
            return None
        if not isinstance(canonical, list):
            return None
        if ordinal:
            inline_nodes.append({"type": "text", "raw": " "})
        for node in canonical:
            if not isinstance(node, Mapping):
                return None
            node_type = node.get("type")
            if node_type == "blank_line":
                continue
            if node_type != "paragraph" or not isinstance(
                node.get("children"), list
            ):
                return None
            inline_nodes.extend(node["children"])

    def normalized_nodes(nodes: list[object]) -> list[object]:
        result: list[object] = []
        for raw_node in nodes:
            if not isinstance(raw_node, Mapping):
                raise ValueError("invalid composite Markdown AST node")
            node = dict(raw_node)
            node_type = node.get("type")
            if node_type in {"softbreak", "linebreak"}:
                node = {"type": "text", "raw": " "}
            elif node_type == "text":
                raw_value = node.get("raw")
                if not isinstance(raw_value, str):
                    raise ValueError("composite text node has no raw value")
                node["raw"] = re.sub(r"\s+", " ", raw_value)
            children = node.get("children")
            if children is not None:
                if not isinstance(children, list):
                    raise ValueError("composite AST children are not a list")
                node["children"] = normalized_nodes(children)
            if (
                result
                and result[-1].get("type") == "text"
                and node.get("type") == "text"
            ):
                result[-1]["raw"] = result[-1]["raw"] + node["raw"]
            else:
                result.append(node)
        return result

    try:
        normalized = normalized_nodes(inline_nodes)
    except (KeyError, TypeError, ValueError):
        return None
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class StructuralUnitSpan:
    """A comparison unit whose byte range is exact in one source artifact."""

    unit_id: str
    source: SourceName
    artifact_digest: str
    unit_type: StructuralUnitType
    byte_start: int
    byte_end: int
    comparison_key: str

    def candidate(
        self,
        *,
        candidate_id: str,
        occurrence_id: str,
        structural_unit_id: str | None = None,
        visible_text_digest: str | None = None,
        non_emphasis_ast_digest: str | None = None,
        emphasis_occurrence_ids: tuple[str, ...] = (),
    ) -> SourceSpanCandidate:
        candidate_type = {
            "heading": "heading",
            "equation": "equation",
            "table": "table_row",
            "table_cell": "table_cell",
            "figure_caption": "figure_caption",
            "reference": "citation",
            "fenced_block": "markdown",
        }.get(self.unit_type, "prose")
        return SourceSpanCandidate(
            candidate_id=candidate_id,
            occurrence_id=occurrence_id,
            structural_unit_id=structural_unit_id or self.unit_id,
            source=self.source,
            artifact_digest=self.artifact_digest,
            byte_start=self.byte_start,
            byte_end=self.byte_end,
            candidate_type=candidate_type,
            comparison_key=self.comparison_key,
            visible_text_digest=visible_text_digest,
            non_emphasis_ast_digest=non_emphasis_ast_digest,
            emphasis_occurrence_ids=emphasis_occurrence_ids,
        )


@dataclass(frozen=True)
class BaselineScore:
    agreement: float
    length_balance: float
    structural_unit_count: int
    heading_count: int
    non_whitespace_bytes: int


@dataclass(frozen=True)
class BaselineDocument:
    artifact: SourceArtifact
    units: tuple[StructuralUnitSpan, ...]
    score: BaselineScore
    quarantined_repeated_sources: tuple[SourceName, ...] = ()
    selection_trace: tuple[dict, ...] = ()


@dataclass(frozen=True)
class BaselineRequirements:
    """Absolute article-level floor applied before relative source scoring."""

    minimum_words: int = 40
    minimum_structural_units: int = 3
    minimum_non_whitespace_bytes: int = 200
    require_heading_or_five_units: bool = True
    required_heading_groups: tuple[tuple[str, ...], ...] = (
        ("abstract", "introduction", "background"),
        ("method", "result", "discussion", "experimental", "conclusion"),
        (
            "references",
            "bibliography",
            "literature cited",
            "acknowledg",
            "conflict of interest",
            "author contribution",
            "supporting information",
            "conclusion",
        ),
    )
    require_abc_validation: bool = True


@dataclass(frozen=True)
class BaselineValidation:
    eligible: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaselineCompletionEvidence:
    """Digest-bound extractor lifecycle evidence, with optional page coverage."""

    artifact_digest: str
    extraction_succeeded: bool
    artifact_complete: bool
    expected_page_count: int | None = None
    covered_page_count: int | None = None
    pdf_digest: str | None = None
    coverage_method: str | None = None
    page_coverage_digest: str | None = None
    completion_basis: Literal[
        "caller_asserted",
        "synchronous_return",
        "saved_artifact_replay",
        "page_coverage",
    ] = "caller_asserted"

    def rejection_reasons(self, artifact: SourceArtifact) -> tuple[str, ...]:
        reasons = []
        if self.artifact_digest != artifact.digest:
            reasons.append("completion_evidence_digest_mismatch")
        if not self.extraction_succeeded:
            reasons.append("extractor_did_not_succeed")
        if not self.artifact_complete:
            reasons.append("extractor_did_not_confirm_complete_artifact")
        if (self.expected_page_count is None) != (self.covered_page_count is None):
            reasons.append("incomplete_page_coverage_evidence")
        elif self.expected_page_count is not None and (
            self.expected_page_count < 1
            or self.covered_page_count is None
            or self.covered_page_count < self.expected_page_count
        ):
            reasons.append("incomplete_page_coverage")
        coverage_fields = (
            self.pdf_digest,
            self.coverage_method,
            self.page_coverage_digest,
        )
        if self.completion_basis == "page_coverage":
            if (
                self.expected_page_count is None
                or self.covered_page_count != self.expected_page_count
                or any(value is None for value in coverage_fields)
                or re.fullmatch(r"[0-9a-f]{64}", self.pdf_digest or "") is None
                or re.fullmatch(
                    r"[0-9a-f]{64}", self.page_coverage_digest or ""
                )
                is None
            ):
                reasons.append("invalid_page_coverage_proof")
            else:
                from app.services.page_coverage import page_coverage_proof_digest

                if self.page_coverage_digest != page_coverage_proof_digest(
                    source=artifact.source,
                    artifact_digest=artifact.digest,
                    pdf_digest=self.pdf_digest,
                    expected_page_count=self.expected_page_count,
                    covered_page_count=self.covered_page_count,
                    coverage_method=self.coverage_method,
                ):
                    reasons.append("page_coverage_proof_digest_mismatch")
        elif any(value is not None for value in coverage_fields) or (
            self.expected_page_count is not None
            or self.covered_page_count is not None
        ):
            reasons.append("page_coverage_basis_mismatch")
        return tuple(reasons)

    @property
    def page_coverage_verified(self) -> bool:
        return (
            self.completion_basis == "page_coverage"
            and self.expected_page_count is not None
            and self.covered_page_count == self.expected_page_count
            and self.pdf_digest is not None
            and self.coverage_method is not None
            and self.page_coverage_digest is not None
        )


BaselineValidator = Callable[
    [SourceArtifact, tuple[StructuralUnitSpan, ...], BaselineRequirements],
    BaselineValidation,
]


@dataclass(frozen=True)
class AssembledDocument:
    raw_utf8: bytes
    digest: str
    baseline_source: SourceName
    baseline_digest: str
    replaced_region_ids: tuple[str, ...]
    fallback_region_ids: tuple[str, ...]
    provenance: tuple["OutputSpanProvenance", ...]
    rejection_reasons: tuple[str, ...] = ()
    repetition_diagnostics: tuple["RepetitionDiagnostic", ...] = ()
    quarantined_repeated_sources: tuple[SourceName, ...] = ()

    @property
    def text(self) -> str:
        return self.raw_utf8.decode("utf-8", errors="strict")


@dataclass(frozen=True)
class OutputSpanProvenance:
    output_byte_start: int
    output_byte_end: int
    source: SourceName
    artifact_digest: str
    source_byte_start: int
    source_byte_end: int
    candidate_id: str | None
    region_id: str | None
    decision_method: DecisionMethod


@dataclass(frozen=True)
class RepetitionDiagnostic:
    """Privacy-safe identity and multiplicity for one excess repetition."""

    diagnostic_id: str
    kind: Literal["paragraph_fingerprint", "long_token_shingle"]
    fingerprint_sha256: str
    output_count: int
    max_source_count: int
    excess_count: int
    structural_classes: tuple[StructuralUnitType, ...]
    output_structural_classes: tuple[tuple[StructuralUnitType, ...], ...]
    output_byte_ranges: tuple[tuple[int, int], ...]

    def as_metric(self) -> dict:
        return {
            "diagnostic_id": self.diagnostic_id,
            "kind": self.kind,
            "fingerprint_sha256": self.fingerprint_sha256,
            "output_count": self.output_count,
            "max_source_count": self.max_source_count,
            "excess_count": self.excess_count,
            "structural_classes": list(self.structural_classes),
            "output_structural_classes": [
                list(classes) for classes in self.output_structural_classes
            ],
            "output_byte_ranges": [
                list(byte_range) for byte_range in self.output_byte_ranges
            ],
        }


@dataclass(frozen=True)
class CandidateMergePlan:
    """Executable full graph plus the unresolved subset exposed to a model."""

    store: CandidateStore
    graph: CandidateGraph
    selection_graph: CandidateGraph
    deterministic_response: SelectionDecisionResponse
    deterministic_reasons: Mapping[str, str] = field(default_factory=dict)
    construction_trace: tuple[dict, ...] = ()
    construction_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def unresolved_region_ids(self) -> tuple[str, ...]:
        return tuple(region.region_id for region in self.selection_graph.regions)

    def combined_response(
        self,
        model_response: SelectionDecisionResponse | None = None,
    ) -> SelectionDecisionResponse:
        """Combine deterministic votes with model choices for unresolved regions."""

        model_decisions = () if model_response is None else model_response.decisions
        unresolved = set(self.unresolved_region_ids)
        regions_by_id = {
            region.region_id: region for region in self.selection_graph.regions
        }
        valid_model_decisions = []
        for decision in model_decisions:
            if decision.region_id not in unresolved:
                continue
            region = regions_by_id[decision.region_id]
            if (
                decision.action == "select_candidates"
                and tuple(decision.candidate_ids) not in region.valid_paths
            ):
                continue
            valid_model_decisions.append(decision)
        return SelectionDecisionResponse(
            decisions=self.deterministic_response.decisions
            + tuple(valid_model_decisions)
        )

    def decision_methods(
        self,
        model_response: SelectionDecisionResponse | None = None,
    ) -> dict[str, DecisionMethod]:
        """Bind every executable selection to its authoritative decision origin."""

        methods: dict[str, DecisionMethod] = {
            decision.region_id: "deterministic"
            for decision in self.deterministic_response.decisions
        }
        if model_response is not None:
            unresolved = set(self.unresolved_region_ids)
            valid_decisions = self.combined_response(model_response).decisions
            methods.update(
                {
                    decision.region_id: "model_selected"
                    for decision in valid_decisions
                    if decision.region_id in unresolved
                }
            )
        return methods


@dataclass(frozen=True)
class BaselineFirstMergeResult:
    """One complete merge outcome with stable out-of-band fallback metadata."""

    document: AssembledDocument
    merge_quality: MergeQuality
    unresolved_region_count: int
    warnings: tuple[str, ...] = ()
    decision_trace: tuple[dict, ...] = ()
    candidate_construction_trace: tuple[dict, ...] = ()
    candidate_construction_counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionResolution:
    """Validated selector response plus the strongest model tier actually used."""

    response: SelectionDecisionResponse
    quality: SelectionQuality = "terra_selected"


SelectionResolver = Callable[
    [CandidateStore, CandidateGraph],
    SelectionDecisionResponse | SelectionResolution,
]


@dataclass(frozen=True)
class _AlignmentMatch:
    alternative_index: int
    similarity: float
    ambiguous: bool
    competing_baseline_indices: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class _CompositeAlignmentMatch:
    """One bounded segmentation alternative represented by exact source spans."""

    baseline_unit_indices: tuple[int, ...]
    alternative_unit_indices: tuple[int, ...]
    baseline_unit: StructuralUnitSpan
    alternative_unit: StructuralUnitSpan
    baseline_profile: _InlineItalicProfile
    alternative_profile: _InlineItalicProfile


@dataclass(frozen=True)
class _RawAlignmentRun:
    """A raw-byte partition element with one normalized comparison token."""

    byte_start: int
    byte_end: int
    key: str


def _comparison_key(text: str) -> str:
    return (
        _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()
    )


def _composite_run_is_oversized(
    units: tuple[StructuralUnitSpan, ...],
) -> bool:
    return any(
        len(unit.comparison_key) > _MAX_COMPARISON_UNIT_CHARS for unit in units
    ) or sum(len(unit.comparison_key) for unit in units) + max(0, len(units) - 1) > (
        _MAX_COMPARISON_UNIT_CHARS
    )


def _composite_unit(
    artifact: SourceArtifact,
    units: tuple[StructuralUnitSpan, ...],
    indices: tuple[int, ...],
) -> tuple[StructuralUnitSpan, _InlineItalicProfile] | None:
    """Collapse one small contiguous structural run into an exact candidate span."""

    if (
        not indices
        or len(indices) > _COMPOSITE_ALIGNMENT_MAX_UNITS
        or indices != tuple(range(indices[0], indices[-1] + 1))
    ):
        return None
    selected = tuple(units[index] for index in indices)
    if any(unit.unit_type not in _COMPOSITE_ALIGNMENT_UNIT_TYPES for unit in selected):
        return None
    if _composite_run_is_oversized(selected):
        return None
    byte_start = selected[0].byte_start
    byte_end = selected[-1].byte_end
    non_emphasis_ast_digest = _composite_non_emphasis_ast_digest(
        artifact, selected
    )
    if non_emphasis_ast_digest is None:
        return None
    visible_parts = []
    emphasis_occurrences = []
    for ordinal, unit in enumerate(selected):
        text = artifact.raw_utf8[unit.byte_start:unit.byte_end].decode(
            "utf-8", errors="strict"
        )
        analysis = _inline_italic_analysis(text, unit.unit_type)
        if analysis is None:
            return None
        unit_profile, normalized_visible = analysis
        visible_parts.append(normalized_visible)
        emphasis_occurrences.extend(
            hashlib.sha256(f"{ordinal}\x00{occurrence_id}".encode("ascii")).hexdigest()
            for occurrence_id in unit_profile.emphasis_occurrence_ids
        )
    normalized_visible = _normalized_visible_text(" ".join(visible_parts))
    visible_digest = hashlib.sha256(normalized_visible.encode("utf-8")).hexdigest()
    profile = _InlineItalicProfile(
        visible_digest=visible_digest,
        non_emphasis_ast_digest=non_emphasis_ast_digest,
        emphasis_occurrence_ids=tuple(emphasis_occurrences),
    )
    return (
        StructuralUnitSpan(
            unit_id=(
                f"{artifact.source}-composite-"
                f"{indices[0]:04d}-{indices[-1]:04d}"
            ),
            source=artifact.source,
            artifact_digest=artifact.digest,
            unit_type="paragraph",
            byte_start=byte_start,
            byte_end=byte_end,
            comparison_key=_comparison_key(normalized_visible),
        ),
        profile,
    )


def _increasing_exact_anchors(
    baseline_units: tuple[StructuralUnitSpan, ...],
    alternative_units: tuple[StructuralUnitSpan, ...],
    baseline_artifact: SourceArtifact,
    alternative_artifact: SourceArtifact,
    *,
    skeleton_slot_keys: Mapping[tuple[SourceName, str], str],
    construction_counts: Counter[str] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return rare exact visible-text anchors as one stable increasing chain."""

    def indexed_profiles(
        units: tuple[StructuralUnitSpan, ...], artifact: SourceArtifact
    ) -> tuple[dict[int, str], dict[str, list[int]]]:
        by_index: dict[int, str] = {}
        by_digest: dict[str, list[int]] = {}
        for index, unit in enumerate(units):
            if unit.unit_type not in _COMPOSITE_ALIGNMENT_ANCHOR_TYPES:
                continue
            if len(unit.comparison_key) > _MAX_COMPARISON_UNIT_CHARS:
                if construction_counts is not None:
                    construction_counts["composite_oversized_run"] += 1
                continue
            raw = artifact.raw_utf8[unit.byte_start:unit.byte_end].decode(
                "utf-8", errors="strict"
            )
            profile = _inline_italic_profile(raw, unit.unit_type)
            if profile is None:
                continue
            by_index[index] = profile.visible_digest
            by_digest.setdefault(profile.visible_digest, []).append(index)
        return by_index, by_digest

    baseline_profiles, baseline_by_digest = indexed_profiles(
        baseline_units, baseline_artifact
    )
    _alternative_profiles, alternative_by_digest = indexed_profiles(
        alternative_units, alternative_artifact
    )
    pairs = []
    for baseline_index, digest in baseline_profiles.items():
        baseline_matches = baseline_by_digest[digest]
        alternative_matches = alternative_by_digest.get(digest, [])
        if len(baseline_matches) != 1 or len(alternative_matches) != 1:
            continue
        alternative_index = alternative_matches[0]
        baseline_slot = skeleton_slot_keys.get(
            (baseline_artifact.source, baseline_units[baseline_index].unit_id)
        )
        alternative_slot = skeleton_slot_keys.get(
            (alternative_artifact.source, alternative_units[alternative_index].unit_id)
        )
        if (
            baseline_slot is not None
            and alternative_slot is not None
            and baseline_slot != alternative_slot
        ):
            continue
        pairs.append((baseline_index, alternative_index))

    # Longest increasing subsequence over alternative positions. Exact anchors
    # that cross after extractor reordering are not allowed to define a run.
    tails: list[int] = []
    tail_pair_indices: list[int] = []
    predecessors: list[int | None] = []
    for pair_index, (_baseline_index, alternative_index) in enumerate(pairs):
        position = bisect_left(tails, alternative_index)
        predecessor = None if position == 0 else tail_pair_indices[position - 1]
        predecessors.append(predecessor)
        if position == len(tails):
            tails.append(alternative_index)
            tail_pair_indices.append(pair_index)
        else:
            tails[position] = alternative_index
            tail_pair_indices[position] = pair_index
    if not tail_pair_indices:
        return ()
    selected: list[tuple[int, int]] = []
    cursor: int | None = tail_pair_indices[-1]
    while cursor is not None:
        selected.append(pairs[cursor])
        cursor = predecessors[cursor]
    return tuple(reversed(selected))


def _composite_alignment_matches(
    baseline_units: tuple[StructuralUnitSpan, ...],
    alternative_units: tuple[StructuralUnitSpan, ...],
    baseline_artifact: SourceArtifact,
    alternative_artifact: SourceArtifact,
    *,
    skeleton_slot_keys: Mapping[tuple[SourceName, str], str] | None = None,
    construction_counts: Counter[str] | None = None,
) -> tuple[_CompositeAlignmentMatch, ...]:
    """Find bounded complete unmatched-run alternatives between stable anchors."""

    slot_keys = dict(skeleton_slot_keys or {})
    anchors = _increasing_exact_anchors(
        baseline_units,
        alternative_units,
        baseline_artifact,
        alternative_artifact,
        skeleton_slot_keys=slot_keys,
        construction_counts=construction_counts,
    )
    boundaries = ((-1, -1), *anchors, (len(baseline_units), len(alternative_units)))
    matches = []
    for (left_baseline, left_alternative), (right_baseline, right_alternative) in zip(
        boundaries, boundaries[1:]
    ):
        baseline_indices = tuple(range(left_baseline + 1, right_baseline))
        alternative_indices = tuple(range(left_alternative + 1, right_alternative))
        sizes = (len(baseline_indices), len(alternative_indices))
        if (
            sizes == (1, 1)
            or not 1 <= sizes[0] <= _COMPOSITE_ALIGNMENT_MAX_UNITS
            or not 1 <= sizes[1] <= _COMPOSITE_ALIGNMENT_MAX_UNITS
        ):
            continue
        baseline_run = tuple(baseline_units[index] for index in baseline_indices)
        alternative_run = tuple(
            alternative_units[index] for index in alternative_indices
        )
        if _composite_run_is_oversized(
            baseline_run
        ) or _composite_run_is_oversized(alternative_run):
            if construction_counts is not None:
                construction_counts["composite_oversized_run"] += 1
            continue
        baseline_slots = {
            slot_keys.get((baseline_artifact.source, baseline_units[index].unit_id))
            for index in baseline_indices
        }
        alternative_slots = {
            slot_keys.get(
                (alternative_artifact.source, alternative_units[index].unit_id)
            )
            for index in alternative_indices
        }
        baseline_slots.discard(None)
        alternative_slots.discard(None)
        if (
            len(baseline_slots) > 1
            or len(alternative_slots) > 1
            or (
                baseline_slots
                and alternative_slots
                and baseline_slots != alternative_slots
            )
        ):
            continue
        baseline_composite = _composite_unit(
            baseline_artifact, baseline_units, baseline_indices
        )
        alternative_composite = _composite_unit(
            alternative_artifact, alternative_units, alternative_indices
        )
        if baseline_composite is None or alternative_composite is None:
            continue
        baseline_unit, baseline_profile = baseline_composite
        alternative_unit, alternative_profile = alternative_composite
        if not _bundle_is_compatible([baseline_unit, alternative_unit]):
            continue
        if not _replacement_preserves_baseline_coverage(
            baseline_unit,
            alternative_unit,
        ):
            continue
        if (
            baseline_artifact.raw_utf8[
                baseline_unit.byte_start:baseline_unit.byte_end
            ]
            == alternative_artifact.raw_utf8[
                alternative_unit.byte_start:alternative_unit.byte_end
            ]
        ):
            continue
        matches.append(
            _CompositeAlignmentMatch(
                baseline_unit_indices=baseline_indices,
                alternative_unit_indices=alternative_indices,
                baseline_unit=baseline_unit,
                alternative_unit=alternative_unit,
                baseline_profile=baseline_profile,
                alternative_profile=alternative_profile,
            )
        )
    return tuple(matches)


def _raw_alignment_runs(
    artifact: SourceArtifact,
    byte_start: int,
    byte_end: int,
) -> tuple[_RawAlignmentRun, ...]:
    """Partition an exact source span while retaining one comparison token per run."""

    raw = artifact.raw_utf8[byte_start:byte_end]
    text = raw.decode("utf-8", errors="strict")
    char_byte_offsets = [0]
    for char in text:
        char_byte_offsets.append(char_byte_offsets[-1] + len(char.encode("utf-8")))
    matches = tuple(_ALIGNMENT_CORE_TOKEN_RE.finditer(text))
    if not matches:
        return (
            _RawAlignmentRun(
                byte_start=byte_start,
                byte_end=byte_end,
                key="<whitespace>",
            ),
        )

    runs = []
    previous_end = 0
    for match in matches:
        runs.append(
            _RawAlignmentRun(
                byte_start=byte_start + char_byte_offsets[previous_end],
                byte_end=byte_start + char_byte_offsets[match.end()],
                key=unicodedata.normalize("NFKC", match.group(0)).casefold(),
            )
        )
        previous_end = match.end()
    if previous_end < len(text):
        runs[-1] = replace(runs[-1], byte_end=byte_end)
    return tuple(runs)


def _content_runs_for_scope(
    unit: StructuralUnitSpan,
    artifact: SourceArtifact,
) -> tuple[_RawAlignmentRun, ...]:
    runs = list(_raw_alignment_runs(artifact, unit.byte_start, unit.byte_end))
    if unit.unit_type == "list" and runs:
        index = 0
        if runs[index].key.isdigit():
            index += 1
            if index < len(runs) and runs[index].key in {".", ")"}:
                index += 1
        elif runs[index].key in {"-", "+", "*"}:
            index += 1
        runs = runs[index:]
    return tuple(runs)


def _boundary_match_size(
    left: tuple[_RawAlignmentRun, ...],
    right: tuple[_RawAlignmentRun, ...],
) -> int:
    """Return an exact high-information suffix/prefix token overlap."""

    maximum = min(len(left), len(right))
    for size in range(maximum, 0, -1):
        if tuple(run.key for run in left[-size:]) != tuple(
            run.key for run in right[:size]
        ):
            continue
        information = sum(
            len("".join(char for char in run.key if char.isalnum()))
            for run in left[-size:]
        )
        if information >= 7:
            return size
    return 0


def _clip_boundary_is_safe(
    unit: StructuralUnitSpan,
    artifact: SourceArtifact,
    byte_offset: int,
) -> bool:
    """Conservatively require a grapheme- and Markdown-atomic clip boundary."""

    if unit.unit_type in _ATOMIC_CLIP_UNIT_TYPES:
        return False
    raw = artifact.raw_utf8[unit.byte_start : unit.byte_end]
    relative = byte_offset - unit.byte_start
    if not 0 < relative < len(raw):
        return False
    try:
        before = raw[:relative].decode("utf-8", errors="strict")
        after = raw[relative:].decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    if not before or not after:
        return False
    text = before + after
    character_offset = len(before)
    if not any(
        match.end() == character_offset for match in regex.finditer(r"\X", text)
    ):
        return False
    first_after = after[0]
    last_before = before[-1]
    if (
        unicodedata.combining(first_after)
        or unicodedata.category(first_after).startswith("M")
        or first_after in {"\u200d", "\ufe0e", "\ufe0f"}
        or last_before == "\u200d"
    ):
        return False

    def unescaped_count(text: str, marker: str) -> int:
        return sum(
            1
            for index, char in enumerate(text)
            if char == marker
            and (
                index == 0
                or (len(text[:index]) - len(text[:index].rstrip("\\"))) % 2 == 0
            )
        )

    # Reject boundaries inside common inline Markdown constructs. This is
    # intentionally conservative: an uncertain clip simply remains baseline.
    if unescaped_count(before, "`") % 2:
        return False
    if unescaped_count(before, "[") > unescaped_count(before, "]"):
        return False
    if unescaped_count(before, "*") % 2:
        return False
    if after.startswith(("]", ")", "`", "*", "_")):
        return False
    if before.endswith(("[", "(", "`", "*", "_", "\\")):
        return False
    return True


def _clip_adjacent_scope_overlap(
    *,
    unit: StructuralUnitSpan,
    artifact: SourceArtifact,
    baseline_units: tuple[StructuralUnitSpan, ...],
    baseline_artifact: SourceArtifact,
    baseline_index: int,
) -> StructuralUnitSpan:
    """Clip only exact candidate-boundary overlap with an adjacent baseline unit."""

    clipped = unit
    candidate_runs = _content_runs_for_scope(clipped, artifact)
    if baseline_index > 0 and candidate_runs:
        previous_runs = _content_runs_for_scope(
            baseline_units[baseline_index - 1], baseline_artifact
        )
        overlap = _boundary_match_size(previous_runs, candidate_runs)
        if overlap:
            new_start = candidate_runs[overlap - 1].byte_end
            if new_start >= clipped.byte_end:
                raise ConsensusContractError(
                    "adjacent prefix clipping removed the candidate"
                )
            if not _clip_boundary_is_safe(clipped, artifact, new_start):
                raise ConsensusContractError(
                    "adjacent prefix clip is not boundary-safe"
                )
            clipped = replace(clipped, byte_start=new_start)
            candidate_runs = _content_runs_for_scope(clipped, artifact)
    if baseline_index + 1 < len(baseline_units) and candidate_runs:
        next_runs = _content_runs_for_scope(
            baseline_units[baseline_index + 1], baseline_artifact
        )
        overlap = _boundary_match_size(candidate_runs, next_runs)
        if overlap:
            new_end = candidate_runs[-overlap].byte_start
            if new_end <= clipped.byte_start:
                raise ConsensusContractError(
                    "adjacent suffix clipping removed the candidate"
                )
            if not _clip_boundary_is_safe(clipped, artifact, new_end):
                raise ConsensusContractError(
                    "adjacent suffix clip is not boundary-safe"
                )
            clipped = replace(clipped, byte_end=new_end)
    if clipped != unit:
        text = artifact.raw_utf8[clipped.byte_start : clipped.byte_end].decode(
            "utf-8", errors="strict"
        )
        clipped = replace(clipped, comparison_key=_comparison_key(text))
    return clipped


def _has_nonlocal_scope_spill(
    *,
    unit: StructuralUnitSpan,
    artifact: SourceArtifact,
    baseline_units: tuple[StructuralUnitSpan, ...],
    baseline_artifact: SourceArtifact,
    baseline_index: int,
    skeleton_slot_keys: Mapping[tuple[SourceName, str], str] | None = None,
) -> bool:
    skeleton_slot_keys = skeleton_slot_keys or {}
    candidate_runs = _content_runs_for_scope(unit, artifact)
    candidate_keys = tuple(run.key for run in candidate_runs)
    current_runs = _content_runs_for_scope(
        baseline_units[baseline_index], baseline_artifact
    )
    current_keys = tuple(run.key for run in current_runs)
    matcher = SequenceMatcher(None, current_keys, candidate_keys, autojunk=False)
    opcodes = matcher.get_opcodes()
    suspicious_segments: list[tuple[str, ...]] = []
    for tag, baseline_start, baseline_end, candidate_start, candidate_end in opcodes:
        if tag == "insert":
            suspicious_segments.append(candidate_keys[candidate_start:candidate_end])
        elif tag == "replace" and not _replacement_opcode_is_local(
            current_keys[baseline_start:baseline_end],
            candidate_keys[candidate_start:candidate_end],
        ):
            return True
    token_work = len(candidate_keys)
    neighborhood_start = max(0, baseline_index - _SCOPE_NEIGHBORHOOD_UNIT_RADIUS)
    neighborhood_end = min(
        len(baseline_units), baseline_index + _SCOPE_NEIGHBORHOOD_UNIT_RADIUS + 1
    )
    for neighbor_index in range(neighborhood_start, neighborhood_end):
        neighbor_unit = baseline_units[neighbor_index]
        current_slot = skeleton_slot_keys.get(
            (
                baseline_units[baseline_index].source,
                baseline_units[baseline_index].unit_id,
            )
        )
        neighbor_slot = skeleton_slot_keys.get(
            (neighbor_unit.source, neighbor_unit.unit_id)
        )
        if (
            neighbor_index != baseline_index
            and current_slot is not None
            and neighbor_slot is not None
            and current_slot != neighbor_slot
        ):
            continue
        neighbor_runs = _content_runs_for_scope(neighbor_unit, baseline_artifact)
        neighbor_keys = tuple(run.key for run in neighbor_runs)
        token_work += len(neighbor_keys)
        if token_work > _SCOPE_NEIGHBORHOOD_TOKEN_LIMIT:
            raise ConsensusContractError(
                "bounded scope proof exceeded its neighborhood token cap"
            )
        if neighbor_index != baseline_index:
            full_match = SequenceMatcher(
                None,
                candidate_keys,
                neighbor_keys,
                autojunk=False,
            ).find_longest_match()
            full_information = sum(
                len("".join(char for char in candidate_keys[index] if char.isalnum()))
                for index in range(full_match.a, full_match.a + full_match.size)
            )
            if (
                full_match.size >= _SCOPE_NONLOCAL_MIN_TOKENS
                and full_information >= _SCOPE_NONLOCAL_MIN_CHARACTERS
            ):
                return True
        for segment in suspicious_segments:
            match = SequenceMatcher(
                None,
                segment,
                neighbor_keys,
                autojunk=False,
            ).find_longest_match()
            information = sum(
                len("".join(char for char in segment[index] if char.isalnum()))
                for index in range(match.a, match.a + match.size)
            )
            complete_high_information_match = (
                match.size == len(segment) and information >= 7
            )
            long_match = (
                match.size >= _SCOPE_NONLOCAL_MIN_TOKENS
                and information >= _SCOPE_NONLOCAL_MIN_CHARACTERS
            )
            if complete_high_information_match or long_match:
                return True
    return False


def _replacement_opcode_is_local(
    baseline_keys: tuple[str, ...],
    candidate_keys: tuple[str, ...],
) -> bool:
    """Allow only bounded spelling/tokenization substitutions as local anchors."""

    if not baseline_keys or not candidate_keys:
        return False
    baseline_text = "".join(baseline_keys)
    candidate_text = "".join(candidate_keys)
    if not any(char.isalnum() for char in baseline_text):
        return not any(char.isalnum() for char in candidate_text)
    return ratio(baseline_text, candidate_text) >= _SCOPE_REPLACEMENT_MIN_SIMILARITY


def _classify_line(text: str) -> StructuralUnitType:
    stripped = text.strip()
    if _HEADING_RE.match(text):
        return "heading"
    if stripped.startswith("$$") or stripped.startswith("\\["):
        return "equation"
    if stripped.startswith("|") or (
        stripped.count("|") >= 2 and not _LIST_RE.match(text)
    ):
        return "table"
    if _FIGURE_RE.match(text):
        return "figure_caption"
    if _REFERENCE_FOOTNOTE_RE.match(text):
        # Keep adjacent Markdown footnote definitions independently bound.
        # Inside a proved bibliography the role renderer converts only their
        # markers to the Alliance numbered-reference form.
        return "list"
    if _LIST_RE.match(text):
        return "list"
    return "paragraph"


def _reference_entry_start_kind(
    text: str,
) -> Literal["author_year", "identifier"] | None:
    """Classify evidence that an unnumbered line begins a reference.

    Extractors commonly wrap one reference over multiple lines.  A new entry
    therefore needs a recognizable author/year prefix. Identifier-only lines
    are classified separately so a DOI wrapped from an author/title line stays
    with that entry, while consecutive identifier-only entries can still split.
    """

    if _REFERENCE_IDENTIFIER_RE.match(text):
        return "identifier"
    candidate = re.sub(r"^\s*(?:[-*]\s+)?", "", text)
    year = _REFERENCE_YEAR_RE.search(candidate[:180])
    if year is None:
        return None
    prefix = candidate[: year.start()].rstrip()
    if re.search(r"\bet\s+al\.?(?:\s|$)", prefix, re.IGNORECASE):
        return "author_year"
    if _REFERENCE_AUTHOR_INITIAL_RE.match(prefix):
        return "author_year"
    if _REFERENCE_AUTHOR_JOIN_RE.match(prefix):
        return "author_year"
    if year.group(0).startswith("(") and re.fullmatch(
        rf"{_REFERENCE_SURNAME}\s*\(?", prefix
    ):
        return "author_year"
    return None


def _fence_open(text: str) -> tuple[str, int, str] | None:
    match = _FENCE_OPEN_RE.match(text)
    if match is None:
        return None
    fence = match.group("fence")
    return fence[0], len(fence), match.group("info").strip()


def _fence_closes(text: str, fence_char: str, minimum_length: int) -> bool:
    """Match a CommonMark fence close without letting a shorter run close it."""

    stripped = text.lstrip(" ")
    indent = len(text) - len(stripped)
    if indent > 3 or not stripped.startswith(fence_char * minimum_length):
        return False
    run_length = len(stripped) - len(stripped.lstrip(fence_char))
    return run_length >= minimum_length and not stripped[run_length:].strip()


def scan_structural_units(artifact: SourceArtifact) -> tuple[StructuralUnitSpan, ...]:
    """Find conservative block spans without changing a single source byte.

    Blank lines remain outside units so the assembler naturally preserves the
    baseline document's original CRLF/LF style and separator width. Headings are
    always standalone. Other adjacent lines are grouped only while their coarse
    structural type remains compatible.
    """

    units: list[StructuralUnitSpan] = []
    pending_start: int | None = None
    pending_end = 0
    pending_type: StructuralUnitType | None = None
    equation_close: str | None = None
    fence_close: tuple[str, int] | None = None
    in_references_section = False
    pending_reference_kind: (
        Literal[
            "explicit_marker",
            "numbered_marker",
            "author_year",
            "identifier",
            "unknown",
        ]
        | None
    ) = None
    byte_cursor = 0

    def flush() -> None:
        nonlocal pending_start, pending_end, pending_type, pending_reference_kind
        if pending_start is None or pending_type is None:
            return
        raw = artifact.raw_utf8[pending_start:pending_end]
        text = raw.decode("utf-8", errors="strict")
        unit_index = len(units)
        units.append(
            StructuralUnitSpan(
                unit_id=f"{artifact.source}-unit-{unit_index:04d}",
                source=artifact.source,
                artifact_digest=artifact.digest,
                unit_type=pending_type,
                byte_start=pending_start,
                byte_end=pending_end,
                comparison_key=_comparison_key(text),
            )
        )
        pending_start = None
        pending_end = 0
        pending_type = None
        pending_reference_kind = None

    for line in artifact.text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        content_bytes = content.encode("utf-8")
        content_end = byte_cursor + len(content_bytes)
        stripped = content.strip()
        if fence_close is not None:
            pending_end = content_end
            if _fence_closes(content, *fence_close):
                fence_close = None
                flush()
            byte_cursor += len(line.encode("utf-8"))
            continue
        if not content.strip():
            if equation_close is None:
                flush()
            byte_cursor += len(line.encode("utf-8"))
            continue

        line_type = _classify_line(content)
        if equation_close is not None:
            pending_end = content_end
            if stripped.endswith(equation_close):
                equation_close = None
                flush()
        elif (fence := _fence_open(content)) is not None:
            flush()
            pending_start = byte_cursor
            pending_end = content_end
            fence_char, fence_length, info = fence
            language = info.split(maxsplit=1)[0].casefold() if info else ""
            pending_type = (
                "equation" if language in {"math", "latex", "tex"} else "fenced_block"
            )
            fence_close = (fence_char, fence_length)
        elif stripped.startswith("$$") or stripped.startswith("\\["):
            flush()
            pending_start = byte_cursor
            pending_end = content_end
            pending_type = "equation"
            if stripped.startswith("$$"):
                if not (len(stripped) > 4 and stripped.endswith("$$")):
                    equation_close = "$$"
            elif not stripped.endswith("\\]"):
                equation_close = "\\]"
            else:
                flush()
            if equation_close is None:
                flush()
        elif in_references_section and (
            line_type == "list" or _REFERENCE_FOOTNOTE_RE.match(content)
        ):
            # Alliance reference records are typed independently of the
            # extractor's choice of bullet versus ordered-list marker. Keep
            # each logical entry as one reference candidate; wrapped lines stay
            # attached until a blank, heading, or next explicit list marker.
            flush()
            pending_start = byte_cursor
            pending_end = content_end
            pending_type = "reference"
            pending_reference_kind = (
                "numbered_marker"
                if _REFERENCE_RE.match(content)
                else "explicit_marker"
            )
        elif line_type in {"heading", "table", "list"}:
            flush()
            pending_start = byte_cursor
            pending_end = content_end
            pending_type = line_type
            flush()
            if line_type == "heading":
                heading = _HEADING_RE.sub("", content).strip().casefold()
                in_references_section = any(
                    label in heading
                    for label in ("references", "bibliography", "literature cited")
                )
        elif in_references_section:
            reference_kind = _reference_entry_start_kind(content)
            starts_new_entry = (
                pending_type != "reference"
                or (
                    reference_kind == "author_year"
                    and pending_reference_kind != "explicit_marker"
                )
                or (
                    reference_kind == "identifier"
                    and pending_reference_kind == "identifier"
                )
            )
            if starts_new_entry:
                flush()
                pending_start = byte_cursor
                pending_type = "reference"
                pending_reference_kind = reference_kind or "unknown"
            pending_end = content_end
        else:
            if pending_type is not None and pending_type not in {
                line_type,
                "reference",
            }:
                flush()
            if pending_start is None:
                pending_start = byte_cursor
                pending_type = line_type
            pending_end = content_end
        byte_cursor += len(line.encode("utf-8"))

    # splitlines() omits an iteration only for the empty string. For a final
    # unterminated line, byte_cursor still advances by its full byte length.
    flush()
    return tuple(units)


def scan_table_cells(
    artifact: SourceArtifact,
    row: StructuralUnitSpan,
) -> tuple[StructuralUnitSpan, ...]:
    """Split one Markdown table row into exact cell-content byte spans.

    Pipe delimiters remain owned by the row/baseline. Escaped pipes are treated
    as cell content. Empty leading/trailing fields created by boundary pipes are
    not emitted as cells.
    """

    if row.source != artifact.source or row.artifact_digest != artifact.digest:
        raise ConsensusContractError(
            "table row does not belong to the supplied artifact"
        )
    if row.unit_type != "table":
        raise ConsensusContractError("scan_table_cells requires a table row unit")
    raw = artifact.raw_utf8[row.byte_start : row.byte_end]
    if len(raw) > _MAX_TABLE_ROW_BYTES or raw.count(b"|") > _MAX_TABLE_CELLS + 1:
        return ()
    text = raw.decode("utf-8", errors="strict")
    delimiters: list[int] = []
    byte_offset = 0
    preceding_backslashes = 0
    for char in text:
        if char == "|" and preceding_backslashes % 2 == 0:
            delimiters.append(byte_offset)
            if len(delimiters) > _MAX_TABLE_CELLS + 1:
                return ()
        if char == "\\":
            preceding_backslashes += 1
        else:
            preceding_backslashes = 0
        byte_offset += len(char.encode("utf-8"))
    if not delimiters:
        return ()

    field_count = len(delimiters) + 1
    if not raw[: delimiters[0]].strip():
        field_count -= 1
    if not raw[delimiters[-1] + 1 :].strip():
        field_count -= 1
    if field_count > _MAX_TABLE_CELLS:
        return ()

    boundaries = [-1, *delimiters, len(raw)]
    cells = []
    for cell_index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        relative_byte_start = left + 1
        relative_byte_end = right
        cell_raw = raw[relative_byte_start:relative_byte_end]
        cell_text = cell_raw.decode("utf-8", errors="strict")
        if (left == -1 or right == len(raw)) and not cell_text.strip():
            continue
        byte_start = row.byte_start + relative_byte_start
        byte_end = row.byte_start + relative_byte_end
        if byte_end <= byte_start:
            continue
        cells.append(
            StructuralUnitSpan(
                unit_id=f"{row.unit_id}-cell-{cell_index:03d}",
                source=row.source,
                artifact_digest=row.artifact_digest,
                unit_type="table_cell",
                byte_start=byte_start,
                byte_end=byte_end,
                comparison_key=_comparison_key(cell_text),
            )
        )
    return tuple(cells)


def _is_table_delimiter_cell(unit: StructuralUnitSpan) -> bool:
    return bool(re.fullmatch(r":?-{3,}:?", unit.comparison_key.strip()))


def _alignment_units(
    artifact: SourceArtifact,
    units: tuple[StructuralUnitSpan, ...],
) -> tuple[StructuralUnitSpan, ...]:
    expanded = []
    for unit in units:
        if unit.unit_type != "table":
            expanded.append(unit)
            continue
        cells = tuple(
            cell
            for cell in scan_table_cells(artifact, unit)
            if not _is_table_delimiter_cell(cell)
        )
        expanded.extend(cells)
    return tuple(expanded)


def _structural_balance_reasons(text: str) -> tuple[str, ...]:
    """Validate block structure with the same delimiter semantics as scanning."""

    reasons: list[str] = []
    fence: tuple[str, int] | None = None
    equation_close: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if fence is not None:
            if _fence_closes(line, *fence):
                fence = None
            continue
        if equation_close is not None:
            if equation_close == "$$" and stripped.endswith("$$"):
                equation_close = None
            elif equation_close == "\\]" and stripped.endswith("\\]"):
                equation_close = None
            continue
        if (opening := _fence_open(line)) is not None:
            fence = (opening[0], opening[1])
            continue
        if stripped == "\\]":
            reasons.append("orphan_bracket_equation_close")
            continue
        if stripped.startswith("$$"):
            if not (len(stripped) > 4 and stripped.endswith("$$")):
                equation_close = "$$"
        elif stripped.startswith("\\[") and not stripped.endswith("\\]"):
            equation_close = "\\]"
    if fence is not None:
        reasons.append(
            "unbalanced_backtick_fence" if fence[0] == "`" else "unbalanced_tilde_fence"
        )
    if equation_close == "$$":
        reasons.append("unbalanced_display_equation")
    elif equation_close == "\\]":
        reasons.append("unbalanced_bracket_equation")
    return tuple(reasons)


def _non_whitespace_bytes(artifact: SourceArtifact) -> int:
    return len("".join(artifact.text.split()).encode("utf-8"))


def validate_baseline_structure(
    artifact: SourceArtifact,
    units: tuple[StructuralUnitSpan, ...],
    requirements: BaselineRequirements,
) -> BaselineValidation:
    """Apply an absolute, extractor-independent article completeness floor.

    Production integration must compose this with ABC/Markdown and scientific
    invariant validation. This floor prevents a nonempty fragment—or two
    mutually agreeing fragments—from defining completeness by themselves.
    """

    reasons = []
    word_count = len(artifact.text.split())
    byte_count = _non_whitespace_bytes(artifact)
    heading_count = sum(unit.unit_type == "heading" for unit in units)
    if word_count < requirements.minimum_words:
        reasons.append("too_few_words")
    if len(units) < requirements.minimum_structural_units:
        reasons.append("too_few_structural_units")
    if byte_count < requirements.minimum_non_whitespace_bytes:
        reasons.append("too_few_content_bytes")
    if (
        requirements.require_heading_or_five_units
        and heading_count == 0
        and len(units) < 5
    ):
        reasons.append("insufficient_document_structure")
    reasons.extend(_structural_balance_reasons(artifact.text))
    heading_text = []
    for unit in units:
        if unit.unit_type != "heading":
            continue
        raw_heading = artifact.raw_utf8[unit.byte_start : unit.byte_end].decode("utf-8")
        heading_text.append(_HEADING_RE.sub("", raw_heading).strip().casefold())
    for group in requirements.required_heading_groups:
        if not any(
            expected in heading for expected in group for heading in heading_text
        ):
            reasons.append(f"missing_required_heading:{'|'.join(group)}")
    if requirements.require_abc_validation:
        try:
            from app.services.abc_markdown_policy import hard_abc_validation_reasons
            from agr_abc_document_parsers import read_markdown

            document = read_markdown(artifact.text)
            hard_abc_reasons = hard_abc_validation_reasons(artifact.text)
            if hard_abc_reasons:
                reasons.extend(hard_abc_reasons)
            if not document.sections and not document.abstract:
                reasons.append("abc_degenerate_document")
        except Exception:
            reasons.append("abc_validation_unavailable")
    return BaselineValidation(eligible=not reasons, reasons=tuple(reasons))


def final_document_safety_reasons(text: str) -> tuple[str, ...]:
    """Return only byte/structure failures that make delivery unsafe."""

    reasons = list(_structural_balance_reasons(text))
    try:
        require_publishable_text(text, candidate_id="final-document")
    except ConsensusContractError as exc:
        reasons.append(f"publishable_text:{type(exc).__name__}")
    return tuple(dict.fromkeys(reasons))


def select_baseline(
    artifacts: Mapping[SourceName, SourceArtifact],
    *,
    completion_evidence: Mapping[SourceName, BaselineCompletionEvidence],
    minimum_relative_coverage: float = 0.60,
    requirements: BaselineRequirements | None = None,
    validator: BaselineValidator = validate_baseline_structure,
) -> BaselineDocument:
    """Select the clean, structurally complete source closest to its peers.

    Every source must first pass an absolute article-level validator. Relative
    coverage is then measured against the largest validated source, and source
    agreement breaks ties. A caller may compose the default structural floor
    with ABC/Markdown, required-section, numeric, and citation validators.
    """

    if not 0 < minimum_relative_coverage <= 1:
        raise ValueError("minimum_relative_coverage must be in (0, 1]")
    if not artifacts:
        raise ConsensusContractError(
            "cannot select a baseline without source artifacts"
        )
    requirements = requirements or BaselineRequirements()

    prepared: dict[
        SourceName, tuple[SourceArtifact, tuple[StructuralUnitSpan, ...], int]
    ] = {}
    rejected: dict[str, str] = {}
    for source, artifact in artifacts.items():
        if source != artifact.source:
            raise ConsensusContractError(
                f"artifact mapping key {source!r} does not match {artifact.source!r}"
            )
        evidence = completion_evidence.get(source)
        if evidence is None:
            rejected[source] = "missing_completion_evidence"
            continue
        evidence_reasons = evidence.rejection_reasons(artifact)
        if evidence_reasons:
            rejected[source] = "+".join(evidence_reasons)
            continue
        if not artifact.text.strip():
            rejected[source] = "empty"
            continue
        try:
            require_publishable_text(artifact.text, candidate_id=f"baseline-{source}")
        except ConsensusContractError:
            rejected[source] = "forbidden_controls"
            continue
        units = scan_structural_units(artifact)
        if not units:
            rejected[source] = "no_structural_units"
            continue
        validation = validator(artifact, units, requirements)
        if not validation.eligible:
            rejected[source] = "+".join(validation.reasons) or "validator_rejected"
            continue
        prepared[source] = (artifact, units, _non_whitespace_bytes(artifact))

    if not prepared:
        reasons = ", ".join(
            f"{source}={reason}" for source, reason in sorted(rejected.items())
        )
        raise ConsensusContractError(
            f"no clean baseline source is available ({reasons})"
        )

    repetition_profiles = {
        source: source_repetition_profile(artifact)
        for source, (artifact, _units, _bytes) in prepared.items()
    }
    repeated_sources = {
        source
        for source, profile in repetition_profiles.items()
        if profile["repeated_document"] is True
    }
    clean_sources = set(prepared) - repeated_sources
    quarantined_repeated_sources = {
        repeated_source
        for repeated_source in repeated_sources
        if any(
            prepared[clean_source][2] * 1_000_000
            >= prepared[repeated_source][2]
            * _SOURCE_REPETITION_CLEAN_PEER_COVERAGE_PPM
            and len(prepared[clean_source][1]) * 1_000_000
            >= len(prepared[repeated_source][1])
            * _SOURCE_REPETITION_CLEAN_PEER_COVERAGE_PPM
            for clean_source in clean_sources
        )
    }
    scoring_pool = {
        source: entry
        for source, entry in prepared.items()
        if source not in quarantined_repeated_sources
    }
    if not scoring_pool:
        # Guaranteed delivery wins when every usable source is itself repeated.
        scoring_pool = prepared
        quarantined_repeated_sources = set()

    byte_counts = [entry[2] for entry in scoring_pool.values()]
    unit_counts = [len(entry[1]) for entry in scoring_pool.values()]
    median_bytes = statistics.median(byte_counts)
    max_bytes = max(byte_counts)
    max_units = max(unit_counts)
    eligible = {
        source: entry
        for source, entry in scoring_pool.items()
        if entry[2] >= max_bytes * minimum_relative_coverage
        and len(entry[1]) >= max_units * minimum_relative_coverage
    }
    if not eligible:
        raise ConsensusContractError(
            "no clean source meets the baseline completeness floor"
        )

    comparison_documents = {
        source: "\n".join(unit.comparison_key for unit in units)
        for source, (_, units, _) in scoring_pool.items()
    }
    scored: list[
        tuple[
            tuple[int, int, int, float, float, float, int, int, int, int],
            BaselineDocument,
        ]
    ] = []
    for source, (artifact, units, byte_count) in eligible.items():
        peers = [
            ratio(comparison_documents[source], peer_text)
            for peer_source, peer_text in comparison_documents.items()
            if peer_source != source
        ]
        agreement = sum(peers) / len(peers) if peers else 100.0
        length_balance = min(byte_count, median_bytes) / max(byte_count, median_bytes)
        heading_count = sum(unit.unit_type == "heading" for unit in units)
        score = BaselineScore(
            agreement=agreement,
            length_balance=length_balance,
            structural_unit_count=len(units),
            heading_count=heading_count,
            non_whitespace_bytes=byte_count,
        )
        document = BaselineDocument(
            artifact=artifact,
            units=units,
            score=score,
            quarantined_repeated_sources=tuple(
                sorted(quarantined_repeated_sources)
            ),
        )
        coverage = min(byte_count / max_bytes, len(units) / max_units)
        try:
            from app.services.abc_markdown_policy import abc_markdown_report

            alliance_clean = abc_markdown_report(artifact.text)[
                "validator_clean"
            ] is True
        except Exception:
            alliance_clean = False
        try:
            _profiles, italic_occurrence_count = _document_italic_profiles(artifact)
        except Exception:
            italic_occurrence_count = 0
        high_coverage_alliance_source = bool(
            coverage >= 0.95 and alliance_clean
        )
        sort_key = (
            int(completion_evidence[source].page_coverage_verified),
            int(high_coverage_alliance_source),
            int(high_coverage_alliance_source and italic_occurrence_count > 0),
            coverage,
            agreement,
            length_balance,
            heading_count,
            len(units),
            byte_count,
            _SOURCE_TIE_BREAK[source],
        )
        scored.append((sort_key, document))
    _selected_sort_key, selected = max(scored, key=lambda item: item[0])
    scored_by_source = {
        document.artifact.source: (sort_key, document)
        for sort_key, document in scored
    }
    selection_trace = []
    for source in sorted(artifacts):
        if source in rejected:
            selection_trace.append(
                {
                    "source": source,
                    "outcome": "rejected_absolute_floor",
                    "reason": rejected[source],
                }
            )
            continue
        if source in quarantined_repeated_sources:
            selection_trace.append(
                {
                    "source": source,
                    "outcome": "quarantined_repetition",
                    "reason": "clean_peer_has_sufficient_coverage",
                }
            )
            continue
        if source not in eligible:
            artifact, units, byte_count = scoring_pool[source]
            selection_trace.append(
                {
                    "source": source,
                    "outcome": "rejected_relative_floor",
                    "reason": "coverage_below_relative_floor",
                    "structural_unit_count": len(units),
                    "non_whitespace_bytes": byte_count,
                    "artifact_digest": artifact.digest,
                }
            )
            continue
        sort_key, document = scored_by_source[source]
        selection_trace.append(
            {
                "source": source,
                "outcome": (
                    "selected" if source == selected.artifact.source else "eligible_not_selected"
                ),
                "artifact_digest": document.artifact.digest,
                "page_coverage_verified": completion_evidence[
                    source
                ].page_coverage_verified,
                "agreement": round(document.score.agreement, 6),
                "length_balance": round(document.score.length_balance, 6),
                "structural_unit_count": document.score.structural_unit_count,
                "heading_count": document.score.heading_count,
                "non_whitespace_bytes": document.score.non_whitespace_bytes,
                "sort_key": list(sort_key),
            }
        )
    return replace(selected, selection_trace=tuple(selection_trace))


def _minimum_alignment_similarity(unit_type: StructuralUnitType) -> float:
    return {
        "heading": 62.0,
        "table": 38.0,
        "table_cell": 45.0,
        "fenced_block": 65.0,
        "equation": 38.0,
        "figure_caption": 48.0,
        "reference": 45.0,
        "list": 45.0,
        "paragraph": 42.0,
    }[unit_type]


def _anchor_tokens(unit: StructuralUnitSpan) -> set[str]:
    return {
        token
        for token in _ANCHOR_TOKEN_RE.findall(unit.comparison_key)
        if len(token) >= 4 or any(char.isdigit() for char in token)
    }


def _bundle_is_compatible(units: list[StructuralUnitSpan]) -> bool:
    """Require all-pair compatibility before exposing or voting a bundle."""

    if len(units) < 2 or len({unit.unit_type for unit in units}) != 1:
        return False
    threshold = _minimum_alignment_similarity(units[0].unit_type)
    for left_index, left in enumerate(units):
        for right in units[left_index + 1 :]:
            if ratio(left.comparison_key, right.comparison_key) < threshold:
                return False
            if max(len(left.comparison_key), len(right.comparison_key)) >= 80:
                if not (_anchor_tokens(left) & _anchor_tokens(right)):
                    return False
    return True


def _replacement_preserves_baseline_coverage(
    baseline_unit: StructuralUnitSpan,
    alternative_unit: StructuralUnitSpan,
) -> bool:
    """Reject alternatives that delete baseline scientific content.

    Deterministic consensus is an improvement layer over a complete baseline,
    not a license for two extractors sharing the same omission to shorten it.
    Formatting deletions are allowed; alphanumeric deletions are not. Replaced
    runs must retain at least as much alphanumeric evidence as the baseline run.
    """

    baseline_key = baseline_unit.comparison_key
    alternative_key = alternative_unit.comparison_key
    if sum(char.isalnum() for char in alternative_key) < sum(
        char.isalnum() for char in baseline_key
    ):
        return False
    matcher = SequenceMatcher(
        None,
        baseline_key,
        alternative_key,
        autojunk=False,
    )
    for (
        tag,
        baseline_start,
        baseline_end,
        alternative_start,
        alternative_end,
    ) in matcher.get_opcodes():
        baseline_part = baseline_key[baseline_start:baseline_end]
        alternative_part = alternative_key[alternative_start:alternative_end]
        if tag == "delete" and any(char.isalnum() for char in baseline_part):
            return False
        if tag == "replace" and sum(char.isalnum() for char in alternative_part) < sum(
            char.isalnum() for char in baseline_part
        ):
            return False
    return True


def _align_to_baseline(
    baseline_units: tuple[StructuralUnitSpan, ...],
    alternative_units: tuple[StructuralUnitSpan, ...],
    *,
    diagnostics: dict | None = None,
    skeleton_slot_keys: Mapping[tuple[SourceName, str], str] | None = None,
    require_matching_unit_types: bool = True,
) -> dict[int, _AlignmentMatch]:
    """Return a bounded, ordered, one-to-one structural-unit alignment.

    Candidate comparisons come from a small positional neighborhood plus rare
    exact/token anchors.  A maximum-weight increasing subsequence then chooses
    the monotonic mapping.  Work therefore scales with the bounded candidate
    edges rather than the full document cross-product; an oversized unit stays
    baseline locally instead of disabling merging for the entire paper.
    """

    skeleton_slot_keys = skeleton_slot_keys or {}
    baseline_count = len(baseline_units)
    alternative_count = len(alternative_units)
    if diagnostics is not None:
        diagnostics.update(
            {
                "algorithm": "bounded_ordered_structural_alignment",
                "baseline_unit_count": baseline_count,
                "alternative_unit_count": alternative_count,
            }
        )
    if not baseline_count or not alternative_count:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "candidate_edge_count": 0,
                    "matched_unit_count": 0,
                    "ambiguous_match_count": 0,
                }
            )
        return {}

    eligible_baseline = tuple(
        len(unit.comparison_key) <= _MAX_COMPARISON_UNIT_CHARS
        for unit in baseline_units
    )
    eligible_alternative = tuple(
        len(unit.comparison_key) <= _MAX_COMPARISON_UNIT_CHARS
        for unit in alternative_units
    )
    if diagnostics is not None:
        diagnostics.update(
            {
                "oversized_baseline_unit_count": eligible_baseline.count(False),
                "oversized_alternative_unit_count": eligible_alternative.count(False),
            }
        )

    exact_alternative_indices: dict[object, list[int]] = {}
    alternative_token_indices: dict[str, list[int]] = {}
    baseline_token_frequency: Counter[str] = Counter()
    for index, unit in enumerate(baseline_units):
        if eligible_baseline[index]:
            baseline_token_frequency.update(_anchor_tokens(unit))
    for index, unit in enumerate(alternative_units):
        if not eligible_alternative[index]:
            continue
        exact_key = (
            (unit.unit_type, unit.comparison_key)
            if require_matching_unit_types
            else unit.comparison_key
        )
        exact_alternative_indices.setdefault(exact_key, []).append(index)
        for token in _anchor_tokens(unit):
            alternative_token_indices.setdefault(token, []).append(index)

    # Each edge is (baseline index, alternative index, similarity).
    edges: list[tuple[int, int, float]] = []
    row_similarities: dict[int, list[float]] = {}
    column_similarities: dict[int, list[float]] = {}
    for baseline_index, baseline_unit in enumerate(baseline_units):
        if not eligible_baseline[baseline_index]:
            continue
        if baseline_count == 1:
            positional_center = 0
        else:
            positional_center = round(
                baseline_index * (alternative_count - 1) / (baseline_count - 1)
            )
        positional_start = max(0, positional_center - _ALIGNMENT_POSITIONAL_WINDOW)
        positional_end = min(
            alternative_count,
            positional_center + _ALIGNMENT_POSITIONAL_WINDOW + 1,
        )
        candidate_indices = set(range(positional_start, positional_end))

        exact_key = (
            (baseline_unit.unit_type, baseline_unit.comparison_key)
            if require_matching_unit_types
            else baseline_unit.comparison_key
        )
        exact_indices = exact_alternative_indices.get(exact_key, [])
        if len(exact_indices) <= _ALIGNMENT_RARE_ANCHOR_MAX_OCCURRENCES:
            candidate_indices.update(exact_indices)

        anchor_votes: Counter[int] = Counter()
        for token in _anchor_tokens(baseline_unit):
            alternative_indices = alternative_token_indices.get(token, [])
            if (
                baseline_token_frequency[token]
                <= _ALIGNMENT_RARE_ANCHOR_MAX_OCCURRENCES
                and len(alternative_indices)
                <= _ALIGNMENT_RARE_ANCHOR_MAX_OCCURRENCES
            ):
                anchor_votes.update(alternative_indices)
        candidate_indices.update(
            index
            for index, _vote_count in sorted(
                anchor_votes.items(),
                key=lambda item: (
                    -item[1],
                    abs(item[0] - positional_center),
                    item[0],
                ),
            )[:_ALIGNMENT_MAX_ANCHOR_CANDIDATES]
        )

        threshold = _minimum_alignment_similarity(baseline_unit.unit_type)
        for alternative_index in sorted(candidate_indices):
            alternative_unit = alternative_units[alternative_index]
            if (
                not eligible_alternative[alternative_index]
                or (
                    require_matching_unit_types
                    and baseline_unit.unit_type != alternative_unit.unit_type
                )
            ):
                continue
            baseline_slot = skeleton_slot_keys.get(
                (baseline_unit.source, baseline_unit.unit_id)
            )
            alternative_slot = skeleton_slot_keys.get(
                (alternative_unit.source, alternative_unit.unit_id)
            )
            if (
                baseline_slot is not None
                and alternative_slot is not None
                and baseline_slot != alternative_slot
            ):
                continue
            similarity = ratio(
                baseline_unit.comparison_key,
                alternative_unit.comparison_key,
            )
            if similarity < threshold:
                continue
            edges.append((baseline_index, alternative_index, similarity))
            row_similarities.setdefault(baseline_index, []).append(similarity)
            column_similarities.setdefault(alternative_index, []).append(similarity)

    if not edges:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "candidate_edge_count": 0,
                    "matched_unit_count": 0,
                    "ambiguous_match_count": 0,
                }
            )
        return {}

    # Fenwick-tree states contain (score, match count, negative positional
    # distance, negative edge id).  The secondary fields make exact ties stable.
    states: list[tuple[int, int, int, int]] = []
    previous_edge: list[int | None] = []
    fenwick: list[int | None] = [None] * (alternative_count + 1)

    def better(left: int | None, right: int | None) -> int | None:
        if left is None:
            return right
        if right is None:
            return left
        return left if states[left] >= states[right] else right

    def query(position: int) -> int | None:
        best: int | None = None
        while position > 0:
            best = better(best, fenwick[position])
            position -= position & -position
        return best

    def update(position: int, edge_id: int) -> None:
        while position <= alternative_count:
            fenwick[position] = better(fenwick[position], edge_id)
            position += position & -position

    edge_cursor = 0
    while edge_cursor < len(edges):
        baseline_index = edges[edge_cursor][0]
        group_end = edge_cursor
        pending_updates: list[tuple[int, int]] = []
        while group_end < len(edges) and edges[group_end][0] == baseline_index:
            _, alternative_index, similarity = edges[group_end]
            predecessor = query(alternative_index)
            predecessor_state = (
                (0, 0, 0, 0) if predecessor is None else states[predecessor]
            )
            similarity_milli = round(similarity * 1_000)
            match_weight = 2 * similarity_milli + 35_000
            edge_id = len(states)
            states.append(
                (
                    predecessor_state[0] + match_weight,
                    predecessor_state[1] + 1,
                    predecessor_state[2]
                    - abs(
                        alternative_index
                        - round(
                            baseline_index
                            * (alternative_count - 1)
                            / max(1, baseline_count - 1)
                        )
                    ),
                    -edge_id,
                )
            )
            previous_edge.append(predecessor)
            pending_updates.append((alternative_index + 1, edge_id))
            group_end += 1
        for position, edge_id in pending_updates:
            update(position, edge_id)
        edge_cursor = group_end

    selected_edge = query(alternative_count)
    competing_baselines: dict[int, list[tuple[int, float]]] = {}
    for baseline_index, alternative_index, similarity in edges:
        competing_baselines.setdefault(alternative_index, []).append(
            (baseline_index, similarity)
        )
    mapping: dict[int, _AlignmentMatch] = {}
    while selected_edge is not None:
        baseline_index, alternative_index, similarity = edges[selected_edge]
        row_competitors = sum(
            candidate >= similarity - _ALIGNMENT_AMBIGUITY_DELTA
            for candidate in row_similarities[baseline_index]
        )
        column_competitors = sum(
            candidate >= similarity - _ALIGNMENT_AMBIGUITY_DELTA
            for candidate in column_similarities[alternative_index]
        )
        candidate_baselines = tuple(
            sorted(
                (
                    (candidate_baseline, candidate_similarity)
                    for candidate_baseline, candidate_similarity in competing_baselines[
                        alternative_index
                    ]
                    if candidate_similarity >= similarity - _ALIGNMENT_AMBIGUITY_DELTA
                ),
                key=lambda item: (-item[1], item[0]),
            )
        )
        mapping[baseline_index] = _AlignmentMatch(
            alternative_index=alternative_index,
            similarity=similarity,
            ambiguous=row_competitors > 1 or column_competitors > 1,
            competing_baseline_indices=candidate_baselines,
        )
        selected_edge = previous_edge[selected_edge]
    if diagnostics is not None:
        diagnostics.update(
            {
                "candidate_edge_count": len(edges),
                "matched_unit_count": len(mapping),
                "ambiguous_match_count": sum(
                    match.ambiguous for match in mapping.values()
                ),
            }
        )
    return mapping


def _preferred_source_backed_italic_candidate(
    aligned_units: list[StructuralUnitSpan],
    candidate_ids: list[str],
    artifacts: Mapping[SourceName, SourceArtifact],
) -> str | None:
    """Prefer an exact source span when candidates differ only by italics.

    Plain candidates do not outvote a single italic-bearing candidate.  The
    preference is applied only when every candidate renders the same visible
    text and every italic-bearing source agrees on the same occurrence-aware
    emphasis profile.  Conflicting boundaries remain a model-selection case.
    """

    profiles: list[tuple[str, SourceName, _InlineItalicProfile]] = []
    for unit, candidate_id in zip(aligned_units, candidate_ids, strict=True):
        raw = artifacts[unit.source].raw_utf8[unit.byte_start : unit.byte_end]
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None
        profile = _inline_italic_profile(text, unit.unit_type)
        if profile is None:
            return None
        profiles.append((candidate_id, unit.source, profile))

    if len({profile.visible_digest for _, _, profile in profiles}) != 1:
        return None
    if len({profile.non_emphasis_ast_digest for _, _, profile in profiles}) != 1:
        return None
    italic_profiles = {
        profile.emphasis_occurrence_ids
        for _, _, profile in profiles
        if profile.emphasis_occurrence_ids
    }
    if len(italic_profiles) != 1:
        return None
    preferred_profile = next(iter(italic_profiles))
    eligible = [
        (candidate_id, source)
        for candidate_id, source, profile in profiles
        if profile.emphasis_occurrence_ids == preferred_profile
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda item: (-_SOURCE_TIE_BREAK[item[1]], item[0]),
    )[0][0]


def build_candidate_merge_plan(
    baseline: BaselineDocument,
    artifacts: Mapping[SourceName, SourceArtifact],
    *,
    skeleton_occurrence_ids: Mapping[tuple[SourceName, str], str] | None = None,
    skeleton_slot_keys: Mapping[tuple[SourceName, str], str] | None = None,
) -> CandidateMergePlan:
    """Build whole-structural-unit choices without generating publication text.

    This graph intentionally ignores unanchored insertions and does not splice
    inside a unit. Exact and normalized-near agreement are deterministic; other
    compatible choices remain unresolved for the selection-only model call.
    """

    if artifacts.get(baseline.artifact.source) != baseline.artifact:
        raise ConsensusContractError(
            "artifacts must include the selected baseline artifact"
        )

    skeleton_occurrence_ids = dict(skeleton_occurrence_ids or {})
    skeleton_slot_keys = dict(skeleton_slot_keys or {})
    units_by_source: dict[SourceName, tuple[StructuralUnitSpan, ...]] = {}
    alignments: dict[SourceName, dict[int, _AlignmentMatch]] = {}
    composite_by_baseline_range: dict[
        tuple[int, int],
        dict[str, object],
    ] = {}
    baseline_alignment_units = _alignment_units(baseline.artifact, baseline.units)
    construction_trace: list[dict] = []
    construction_counts: Counter[str] = Counter()
    for source, artifact in artifacts.items():
        if source != artifact.source:
            raise ConsensusContractError(
                f"artifact mapping key {source!r} does not match {artifact.source!r}"
            )
        source_units = (
            baseline.units
            if source == baseline.artifact.source
            else scan_structural_units(artifact)
        )
        if source != baseline.artifact.source:
            for match in _composite_alignment_matches(
                baseline.units,
                source_units,
                baseline.artifact,
                artifact,
                skeleton_slot_keys=skeleton_slot_keys,
                construction_counts=construction_counts,
            ):
                key = (
                    match.baseline_unit.byte_start,
                    match.baseline_unit.byte_end,
                )
                entry = composite_by_baseline_range.setdefault(
                    key,
                    {
                        "baseline_unit": match.baseline_unit,
                        "baseline_profile": match.baseline_profile,
                        "baseline_unit_indices": match.baseline_unit_indices,
                        "alternatives": [],
                    },
                )
                entry["alternatives"].append(
                    (match.alternative_unit, match.alternative_profile)
                )
                entry.setdefault("shapes", []).append(
                    (
                        len(match.baseline_unit_indices),
                        len(match.alternative_unit_indices),
                    )
                )
        units = _alignment_units(artifact, source_units)
        units_by_source[source] = units
        if source != baseline.artifact.source:
            alignment_diagnostics: dict = {
                "source": source,
                "baseline_source": baseline.artifact.source,
            }
            alignments[source] = _align_to_baseline(
                baseline_alignment_units,
                units,
                diagnostics=alignment_diagnostics,
                skeleton_slot_keys=skeleton_slot_keys,
            )
            construction_trace.append(alignment_diagnostics)

    artifact_registry = {
        (artifact.source, artifact.digest): artifact for artifact in artifacts.values()
    }
    candidate_registry: dict[str, SourceSpanCandidate] = {}
    graph_regions = []
    selection_regions = []
    deterministic_decisions = []
    deterministic_reasons: dict[str, str] = {}

    composite_baseline_ranges: list[tuple[int, int]] = []
    composite_owned_source_spans: dict[
        SourceName, list[tuple[int, int]]
    ] = {}
    # Pairwise anchor chains can yield overlapping baseline runs for different
    # peers. Give one region exclusive ownership before graph construction so a
    # broadened N:M path cannot force the whole job through invalid-graph
    # fallback. Prefer the run carrying the most available emphasis evidence,
    # then the run with more independent source alternatives and the smaller
    # baseline span. This is source evidence ordering, never authored content.
    ranked_composites = sorted(
        composite_by_baseline_range.items(),
        key=lambda item: (
            -max(
                (
                    len(profile.emphasis_occurrence_ids)
                    for unit, profile in item[1].get("alternatives", ())
                    if isinstance(unit, StructuralUnitSpan)
                    and isinstance(profile, _InlineItalicProfile)
                ),
                default=0,
            ),
            -len(item[1].get("alternatives", ())),
            item[0][1] - item[0][0],
            item[0],
        ),
    )
    owned_baseline_ranges: list[tuple[int, int]] = []
    selected_composites = []
    for baseline_range, entry in ranked_composites:
        byte_start, byte_end = baseline_range
        if any(
            owned_start < byte_end and byte_start < owned_end
            for owned_start, owned_end in owned_baseline_ranges
        ):
            construction_counts["composite_overlap_rejected"] += 1
            continue
        owned_baseline_ranges.append(baseline_range)
        selected_composites.append((baseline_range, entry))

    for composite_ordinal, ((byte_start, byte_end), entry) in enumerate(
        sorted(selected_composites)
    ):
        baseline_unit = entry["baseline_unit"]
        baseline_profile = entry["baseline_profile"]
        alternatives = entry["alternatives"]
        baseline_unit_indices = entry.get("baseline_unit_indices")
        if not isinstance(baseline_unit, StructuralUnitSpan) or not isinstance(
            baseline_profile, _InlineItalicProfile
        ):
            raise ConsensusContractError("invalid composite baseline evidence")
        aligned = [(baseline_unit, baseline_profile)]
        aligned.extend(
            (unit, profile)
            for unit, profile in alternatives
            if isinstance(unit, StructuralUnitSpan)
            and isinstance(profile, _InlineItalicProfile)
        )
        incomplete_aligned_peer_evidence = False
        represented_sources = {unit.source for unit, _profile in aligned}
        if not isinstance(baseline_unit_indices, tuple):
            raise ConsensusContractError("invalid composite baseline indices")
        for source in sorted(alignments):
            if source in represented_sources:
                continue
            peer_matches = [
                alignments[source].get(baseline_index)
                for baseline_index in baseline_unit_indices
            ]
            peer_is_relevant = any(match is not None for match in peer_matches)
            if not peer_is_relevant:
                continue
            peer_is_safe = all(
                match is not None and not match.ambiguous
                for match in peer_matches
            )
            indices = tuple(
                match.alternative_index
                for match in peer_matches
                if match is not None
            )
            if (
                not peer_is_safe
                or not indices
                or len(set(indices)) != len(indices)
                or indices != tuple(range(indices[0], indices[-1] + 1))
            ):
                incomplete_aligned_peer_evidence = True
                continue
            peer_composite = _composite_unit(
                artifacts[source], units_by_source[source], indices
            )
            if peer_composite is None:
                incomplete_aligned_peer_evidence = True
                continue
            aligned.append(peer_composite)
            represented_sources.add(source)
            construction_counts["composite_aligned_peer_included"] += 1
        # One source may contribute at most one exact span for this baseline
        # region. Stable source preference resolves duplicate discoveries.
        by_source: dict[
            SourceName, tuple[StructuralUnitSpan, _InlineItalicProfile]
        ] = {}
        for unit, profile in aligned:
            current = by_source.get(unit.source)
            if current is None or (
                unit.byte_start,
                unit.byte_end,
            ) < (
                current[0].byte_start,
                current[0].byte_end,
            ):
                by_source[unit.source] = (unit, profile)
        aligned = [by_source[source] for source in sorted(by_source)]
        if len(aligned) < 2:
            continue

        region_id = f"region-composite-{composite_ordinal:04d}"
        shared_unit_id = f"composite-baseline-{byte_start}-{byte_end}"
        candidate_ids = []
        for unit, profile in aligned:
            candidate_id = f"{region_id}-{unit.source}"
            candidate_registry[candidate_id] = SourceSpanCandidate(
                candidate_id=candidate_id,
                occurrence_id=f"occurrence-{candidate_id}",
                structural_unit_id=shared_unit_id,
                source=unit.source,
                artifact_digest=unit.artifact_digest,
                byte_start=unit.byte_start,
                byte_end=unit.byte_end,
                candidate_type="markdown",
                comparison_key=profile.visible_digest,
                visible_text_digest=profile.visible_digest,
                # Paragraph segmentation is abstracted, but every other inline
                # structure remains part of this content-free digest.
                non_emphasis_ast_digest=profile.non_emphasis_ast_digest,
                emphasis_occurrence_ids=profile.emphasis_occurrence_ids,
            )
            candidate_ids.append(candidate_id)
            composite_owned_source_spans.setdefault(unit.source, []).append(
                (unit.byte_start, unit.byte_end)
            )

        baseline_candidate_id = f"{region_id}-{baseline.artifact.source}"
        region = RegionCandidateGraph(
            region_id=region_id,
            baseline_candidate_id=baseline_candidate_id,
            valid_paths=tuple((candidate_id,) for candidate_id in candidate_ids),
        )
        graph_regions.append(region)
        composite_baseline_ranges.append((byte_start, byte_end))

        italic_profiles = {
            profile.emphasis_occurrence_ids
            for _unit, profile in aligned
            if profile.emphasis_occurrence_ids
        }
        non_emphasis_profiles = {
            profile.non_emphasis_ast_digest for _unit, profile in aligned
        }
        if (
            not incomplete_aligned_peer_evidence
            and len(italic_profiles) == 1
            and len(non_emphasis_profiles) == 1
        ):
            protected_profile = next(iter(italic_profiles))
            eligible = [
                (candidate_id, unit.source)
                for candidate_id, (unit, profile) in zip(
                    candidate_ids, aligned, strict=True
                )
                if profile.emphasis_occurrence_ids == protected_profile
            ]
            selected_id = sorted(
                eligible,
                key=lambda item: (-_SOURCE_TIE_BREAK[item[1]], item[0]),
            )[0][0]
            deterministic_reasons[region_id] = (
                "composite_source_backed_italic_preference"
            )
            deterministic_decisions.append(
                RegionSelectionDecision(
                    region_id=region_id,
                    action="select_candidates",
                    candidate_ids=(selected_id,),
                )
            )
            construction_counts[
                "region_composite_source_backed_italic_preference"
            ] += 1
        else:
            selection_regions.append(region)
            construction_counts["region_composite_model_selection_required"] += 1
        construction_counts["composite_alignment_region"] += 1
        shapes = entry.get("shapes", ())
        if any(
            isinstance(shape, tuple)
            and len(shape) == 2
            and shape[0] > 1
            and shape[1] > 1
            for shape in shapes
        ):
            construction_counts["composite_alignment_region_nm"] += 1

    for baseline_index, baseline_unit in enumerate(baseline_alignment_units):
        if any(
            start <= baseline_unit.byte_start
            and baseline_unit.byte_end <= end
            for start, end in composite_baseline_ranges
        ):
            construction_counts["baseline_owned_by_composite_region"] += 1
            continue
        aligned_units = [baseline_unit]
        for source in sorted(alignments):
            match = alignments[source].get(baseline_index)
            if match is None:
                construction_counts["peer_alignment_missing"] += 1
                continue
            if match.ambiguous:
                construction_counts["peer_alignment_ambiguous"] += 1
                continue
            alternative_unit = units_by_source[source][match.alternative_index]
            if any(
                start < alternative_unit.byte_end
                and alternative_unit.byte_start < end
                for start, end in composite_owned_source_spans.get(source, ())
            ):
                construction_counts["peer_owned_by_composite_region"] += 1
                continue
            baseline_slot = skeleton_slot_keys.get(
                (baseline_unit.source, baseline_unit.unit_id)
            )
            alternative_slot = skeleton_slot_keys.get(
                (alternative_unit.source, alternative_unit.unit_id)
            )
            if (
                baseline_slot is not None
                and alternative_slot is not None
                and baseline_slot != alternative_slot
            ):
                construction_counts["peer_skeleton_slot_mismatch"] += 1
                continue
            raw = artifacts[source].raw_utf8[
                alternative_unit.byte_start : alternative_unit.byte_end
            ]
            if unsafe_unicode_characters(raw.decode("utf-8", errors="strict")):
                construction_counts["peer_unsafe_unicode"] += 1
                continue
            try:
                alternative_unit = _clip_adjacent_scope_overlap(
                    unit=alternative_unit,
                    artifact=artifacts[source],
                    baseline_units=baseline_alignment_units,
                    baseline_artifact=baseline.artifact,
                    baseline_index=baseline_index,
                )
                if _has_nonlocal_scope_spill(
                    unit=alternative_unit,
                    artifact=artifacts[source],
                    baseline_units=baseline_alignment_units,
                    baseline_artifact=baseline.artifact,
                    baseline_index=baseline_index,
                    skeleton_slot_keys=skeleton_slot_keys,
                ):
                    construction_counts["peer_nonlocal_scope_spill"] += 1
                    continue
            except ConsensusContractError:
                construction_counts["peer_scope_proof_rejected"] += 1
                continue
            if not _replacement_preserves_baseline_coverage(
                baseline_unit,
                alternative_unit,
            ):
                construction_counts["peer_baseline_coverage_loss"] += 1
                continue
            aligned_units.append(alternative_unit)

        if not _bundle_is_compatible(aligned_units):
            construction_counts[
                "baseline_no_aligned_peer"
                if len(aligned_units) < 2
                else "baseline_incompatible_bundle"
            ] += 1
            continue

        raw_by_unit = [
            artifacts[unit.source].raw_utf8[unit.byte_start : unit.byte_end]
            for unit in aligned_units
        ]
        if len(aligned_units) < 2 or len(set(raw_by_unit)) == 1:
            construction_counts["baseline_exact_agreement"] += 1
            continue

        if len({unit.comparison_key for unit in aligned_units}) == 1:
            # A whitespace/normalization-only difference is already a clean,
            # source-backed near agreement. Retain the baseline unit instead of
            # asking a model to reproduce it.
            construction_counts["baseline_normalized_near_agreement"] += 1
            continue

        region_id = f"region-{baseline_index:04d}"
        shared_unit_id = skeleton_occurrence_ids.get(
            (baseline_unit.source, baseline_unit.unit_id),
            f"baseline-unit-{baseline_index:04d}",
        )
        candidate_ids = []
        for unit in aligned_units:
            candidate_id = f"{region_id}-{unit.source}"
            candidate_text = artifacts[unit.source].raw_utf8[
                unit.byte_start : unit.byte_end
            ].decode("utf-8", errors="strict")
            formatting_profile = _inline_italic_profile(
                candidate_text,
                unit.unit_type,
            )
            candidate = unit.candidate(
                candidate_id=candidate_id,
                occurrence_id=f"occurrence-{candidate_id}",
                structural_unit_id=shared_unit_id,
                visible_text_digest=(
                    None
                    if formatting_profile is None
                    else formatting_profile.visible_digest
                ),
                non_emphasis_ast_digest=(
                    None
                    if formatting_profile is None
                    else formatting_profile.non_emphasis_ast_digest
                ),
                emphasis_occurrence_ids=(
                    ()
                    if formatting_profile is None
                    else formatting_profile.emphasis_occurrence_ids
                ),
            )
            candidate_registry[candidate_id] = candidate
            candidate_ids.append(candidate_id)

        baseline_candidate_id = f"{region_id}-{baseline.artifact.source}"
        region = RegionCandidateGraph(
            region_id=region_id,
            baseline_candidate_id=baseline_candidate_id,
            valid_paths=tuple((candidate_id,) for candidate_id in candidate_ids),
        )
        graph_regions.append(region)

        raw_groups: dict[bytes, list[str]] = {}
        for candidate_id, raw in zip(candidate_ids, raw_by_unit):
            raw_groups.setdefault(raw, []).append(candidate_id)
        majority = [ids for ids in raw_groups.values() if len(ids) >= 2]
        italic_preference = _preferred_source_backed_italic_candidate(
            aligned_units,
            candidate_ids,
            artifacts,
        )
        if italic_preference is not None:
            selected_id = italic_preference
            deterministic_reasons[region_id] = "source_backed_italic_preference"
            deterministic_decisions.append(
                RegionSelectionDecision(
                    region_id=region_id,
                    action="select_candidates",
                    candidate_ids=(selected_id,),
                )
            )
            construction_counts["region_source_backed_italic_preference"] += 1
        elif majority:
            selected_id = sorted(
                majority[0],
                key=lambda candidate_id: (
                    -_SOURCE_TIE_BREAK[candidate_registry[candidate_id].source],
                    candidate_id,
                ),
            )[0]
            deterministic_reasons[region_id] = "exact_peer_majority"
            deterministic_decisions.append(
                RegionSelectionDecision(
                    region_id=region_id,
                    action="select_candidates",
                    candidate_ids=(selected_id,),
                )
            )
            construction_counts["region_exact_peer_majority"] += 1
        else:
            selection_regions.append(region)
            construction_counts["region_model_selection_required"] += 1

    store = CandidateStore(artifact_registry, candidate_registry)
    graph = CandidateGraph(regions=tuple(graph_regions))
    selection_graph = CandidateGraph(regions=tuple(selection_regions))
    store.validate_graph(graph)
    return CandidateMergePlan(
        store=store,
        graph=graph,
        selection_graph=selection_graph,
        deterministic_response=SelectionDecisionResponse(
            decisions=tuple(deterministic_decisions)
        ),
        deterministic_reasons=deterministic_reasons,
        construction_trace=tuple(construction_trace),
        construction_counts=dict(sorted(construction_counts.items())),
    )


def _document_italic_profiles(
    artifact: SourceArtifact,
) -> tuple[
    dict[tuple[StructuralUnitType, str, int], _InlineItalicProfile],
    int,
]:
    """Index eligible unit profiles with occurrence ordinals for repeated blocks."""

    indexed, italic_occurrence_count = _document_indexed_italic_profiles(artifact)
    return (
        {key: profile for key, (_unit, profile) in indexed.items()},
        italic_occurrence_count,
    )


def _italic_ledger_units(
    artifact: SourceArtifact,
) -> tuple[StructuralUnitSpan, ...]:
    """Expand safe Markdown table rows into exact cell spans for italics."""

    expanded: list[StructuralUnitSpan] = []
    for unit in scan_structural_units(artifact):
        if unit.unit_type != "table":
            expanded.append(unit)
            continue
        cells = tuple(
            cell
            for cell in scan_table_cells(artifact, unit)
            if not _is_table_delimiter_cell(cell)
        )
        if cells:
            expanded.extend(cells)
        else:
            # An oversized or malformed row is deliberately left unprofiled;
            # potential emphasis in it is reported by the conservative ledger.
            expanded.append(unit)
    return tuple(expanded)


def _unprofiled_emphasis_evidence_ids(
    artifact: SourceArtifact,
) -> tuple[str, ...]:
    """Identify emphasis-bearing units that cannot be safely profiled.

    These IDs contain only source/digest/span metadata.  Fenced code, math,
    malformed tables, raw HTML italics, and unknown Markdown constructs never
    silently disappear from qualification: ambiguous marker pairs are reported
    as unresolved evidence until a typed resolver can prove them safe.
    """

    identifiers = []
    for unit in _italic_ledger_units(artifact):
        raw = artifact.raw_utf8[unit.byte_start : unit.byte_end].decode(
            "utf-8", errors="strict"
        )
        if _inline_italic_profile(raw, unit.unit_type) is not None:
            continue
        if _POTENTIAL_UNPROFILED_EMPHASIS_RE.search(raw) is None:
            continue
        identifiers.append(
            hashlib.sha256(
                "\x00".join(
                    (
                        _ITALIC_PRESERVATION_POLICY_VERSION,
                        "unprofiled_emphasis",
                        artifact.source,
                        artifact.digest,
                        unit.unit_type,
                        str(unit.byte_start),
                        str(unit.byte_end),
                    )
                ).encode("ascii")
            ).hexdigest()
        )
    return tuple(sorted(set(identifiers)))


def _document_indexed_italic_profiles(
    artifact: SourceArtifact,
) -> tuple[
    dict[
        tuple[StructuralUnitType, str, int],
        tuple[StructuralUnitSpan, _InlineItalicProfile],
    ],
    int,
]:
    """Index exact unit spans and profiles for deterministic source selection."""

    profiles: dict[
        tuple[StructuralUnitType, str, int],
        tuple[StructuralUnitSpan, _InlineItalicProfile],
    ] = {}
    ordinals: Counter[tuple[StructuralUnitType, str]] = Counter()
    italic_occurrence_count = 0
    for unit in _italic_ledger_units(artifact):
        raw = artifact.raw_utf8[unit.byte_start : unit.byte_end].decode(
            "utf-8", errors="strict"
        )
        profile = _inline_italic_profile(raw, unit.unit_type)
        if profile is None:
            continue
        base_key = (unit.unit_type, profile.visible_digest)
        ordinal = ordinals[base_key]
        ordinals[base_key] += 1
        profiles[(unit.unit_type, profile.visible_digest, ordinal)] = (unit, profile)
        italic_occurrence_count += len(profile.emphasis_occurrence_ids)
    return profiles, italic_occurrence_count


def italic_preservation_metric(
    text: str,
    artifacts: Mapping[SourceName, SourceArtifact],
    *,
    baseline_source: SourceName | None = None,
) -> dict:
    """Return a union-source, occurrence-aware italics receipt.

    Every profiled occurrence seen by any successful extractor participates in
    the evidence universe.  A formatting profile is protected only when all
    source-backed instances of that visible occurrence agree on the surrounding
    non-emphasis AST and every italic-bearing instance agrees on the exact
    emphasis boundaries.  Missing output, conflicting boundaries, incompatible
    markup, and unprofiled emphasis remain distinct findings.
    """

    if not artifacts:
        raise ValueError("italic preservation requires at least one artifact")
    if baseline_source is None:
        baseline_source = next(iter(sorted(artifacts)))
    if baseline_source not in artifacts:
        raise ValueError("italic preservation baseline is not available")

    source_profiles: dict[
        SourceName,
        dict[tuple[StructuralUnitType, str, int], _InlineItalicProfile],
    ] = {}
    input_counts: dict[SourceName, int] = {}
    for source in sorted(artifacts):
        profiles, count = _document_italic_profiles(artifacts[source])
        source_profiles[source] = profiles
        input_counts[source] = count

    baseline_artifact = artifacts[baseline_source]
    baseline_profiles = source_profiles[baseline_source]
    output_source: SourceName = baseline_source
    output_profiles, output_count = _document_italic_profiles(
        SourceArtifact.from_text(output_source, text)
    )

    protected_blocks = 0
    protected_occurrences = 0
    retained_occurrences = 0
    whole_block_missing_occurrences = 0
    matched_block_formatting_lost_occurrences = 0
    ambiguous_blocks = 0
    unresolved_blocks = 0
    italic_evidence_blocks = 0
    fully_aligned_blocks = 0
    lost_block_ids: list[str] = []
    ambiguous_block_ids: list[str] = []
    unresolved_block_ids: list[str] = []
    unprofiled_emphasis_block_ids = sorted(
        {
            identifier
            for artifact in artifacts.values()
            for identifier in _unprofiled_emphasis_evidence_ids(artifact)
        }
    )
    unresolved_blocks = len(unprofiled_emphasis_block_ids)
    unresolved_block_ids.extend(unprofiled_emphasis_block_ids)

    source_digest_identity = tuple(
        f"{source}:{artifacts[source].digest}" for source in sorted(artifacts)
    )

    def block_id(
        classification: str,
        key: tuple[StructuralUnitType, str, int],
        profiles: tuple[str, ...],
    ) -> str:
        unit_type, visible_digest, ordinal = key
        return hashlib.sha256(
            "\x00".join(
                (
                    _ITALIC_PRESERVATION_POLICY_VERSION,
                    classification,
                    *source_digest_identity,
                    unit_type,
                    visible_digest,
                    str(ordinal),
                    *profiles,
                )
            ).encode("ascii")
        ).hexdigest()

    source_union_keys = {
        key for profiles in source_profiles.values() for key in profiles
    }
    output_aligned_keys = source_union_keys & set(output_profiles)
    for key in sorted(source_union_keys):
        present_profiles = {
            source: profiles[key]
            for source, profiles in source_profiles.items()
            if key in profiles
        }
        italic_profiles = {
            profile.emphasis_occurrence_ids
            for profile in present_profiles.values()
            if profile.emphasis_occurrence_ids
        }
        if not italic_profiles:
            continue
        italic_evidence_blocks += 1
        if len(present_profiles) == len(source_profiles):
            fully_aligned_blocks += 1

        unresolved = False
        ast_profiles = {
            profile.non_emphasis_ast_digest for profile in present_profiles.values()
        }
        conflicting_profiles = len(italic_profiles) > 1
        if conflicting_profiles:
            ambiguous_blocks += 1
            ambiguous_block_ids.append(
                block_id(
                    "ambiguous",
                    key,
                    tuple(sorted(digest for profile in italic_profiles for digest in profile)),
                )
            )
        if len(ast_profiles) > 1:
            unresolved = True
        if key not in output_profiles:
            unresolved = True

        protected_profile = (
            next(iter(italic_profiles))
            if len(italic_profiles) == 1 and len(ast_profiles) == 1
            else ()
        )

        if protected_profile:
            protected_blocks += 1
            protected_occurrences += len(protected_profile)
            output_profile = output_profiles.get(key)
            expected_ast = next(iter(ast_profiles))
            output_occurrences = (
                ()
                if output_profile is None
                or output_profile.non_emphasis_ast_digest
                != expected_ast
                else output_profile.emphasis_occurrence_ids
            )
            if (
                output_profile is not None
                and output_profile.non_emphasis_ast_digest
                != expected_ast
            ):
                unresolved = True
            retained = sum(
                (Counter(protected_profile) & Counter(output_occurrences)).values()
            )
            retained_occurrences += retained
            if retained != len(protected_profile):
                lost_block_ids.append(block_id("lost", key, protected_profile))
                lost = len(protected_profile) - retained
                if output_profile is None:
                    whole_block_missing_occurrences += lost
                else:
                    matched_block_formatting_lost_occurrences += lost

        if unresolved:
            unresolved_blocks += 1
            unresolved_block_ids.append(
                block_id(
                    "unresolved",
                    key,
                    tuple(sorted(digest for profile in italic_profiles for digest in profile)),
                )
            )

    lost_occurrences = protected_occurrences - retained_occurrences
    all_retained = (
        lost_occurrences == 0
        and ambiguous_blocks == 0
        and unresolved_blocks == 0
    )
    return {
        "policy_version": _ITALIC_PRESERVATION_POLICY_VERSION,
        "baseline_source": baseline_source,
        "baseline_artifact_digest": baseline_artifact.digest,
        "available_source_count": len(artifacts),
        "input_italic_occurrence_counts": input_counts,
        "output_italic_occurrence_count": output_count,
        "baseline_eligible_block_count": len(baseline_profiles),
        "union_eligible_block_count": len(output_aligned_keys),
        "source_union_block_count": len(source_union_keys),
        "italic_evidence_block_count": italic_evidence_blocks,
        "fully_aligned_italic_block_count": fully_aligned_blocks,
        "protected_block_count": protected_blocks,
        "protected_italic_occurrence_count": protected_occurrences,
        "retained_protected_italic_occurrence_count": retained_occurrences,
        "lost_protected_italic_occurrence_count": lost_occurrences,
        "whole_block_missing_protected_italic_occurrence_count": (
            whole_block_missing_occurrences
        ),
        "matched_block_formatting_lost_occurrence_count": (
            matched_block_formatting_lost_occurrences
        ),
        "ambiguous_italic_block_count": ambiguous_blocks,
        "unresolved_italic_block_count": unresolved_blocks,
        "unprofiled_emphasis_block_count": len(unprofiled_emphasis_block_ids),
        "lost_protected_block_ids": sorted(set(lost_block_ids)),
        "ambiguous_italic_block_ids": sorted(set(ambiguous_block_ids)),
        "unresolved_italic_block_ids": sorted(set(unresolved_block_ids)),
        "unprofiled_emphasis_block_ids": unprofiled_emphasis_block_ids,
        "all_protected_italics_retained": all_retained,
    }


def _container_content_start(raw: bytes, unit_type: StructuralUnitType) -> int:
    """Keep structural markers while selecting source-backed inline content."""

    patterns = {
        "heading": rb"^ {0,3}#{1,6}[ \t]+",
        "list": rb"^\s*(?:[-+*]|\d+[.)])[ \t]+",
        "reference": rb"^\s*(?:\[\d{1,4}\]|\d{1,4}[.)])[ \t]+",
    }
    pattern = patterns.get(unit_type)
    if pattern is None:
        return 0
    match = re.match(pattern, raw)
    return 0 if match is None else match.end()


def repetition_policy_metric() -> dict:
    """Return the frozen, publication-text-free detector policy."""

    return {
        "version": _REPETITION_POLICY_VERSION,
        "paragraph_min_normalized_characters": (
            _REPETITION_PARAGRAPH_MIN_NORMALIZED_CHARACTERS
        ),
        "paragraph_min_anchor_tokens": _REPETITION_PARAGRAPH_MIN_ANCHOR_TOKENS,
        "shingle_width": _REPETITION_SHINGLE_WIDTH,
        "shingle_min_anchor_characters": _REPETITION_SHINGLE_MIN_ANCHOR_CHARACTERS,
        "source_repetition_min_long_blocks": _SOURCE_REPETITION_MIN_LONG_BLOCKS,
        "source_repetition_min_excess_blocks": _SOURCE_REPETITION_MIN_EXCESS_BLOCKS,
        "source_repetition_min_excess_ratio_ppm": (
            _SOURCE_REPETITION_MIN_EXCESS_RATIO_PPM
        ),
        "source_repetition_clean_peer_coverage_ppm": (
            _SOURCE_REPETITION_CLEAN_PEER_COVERAGE_PPM
        ),
    }


def _paragraph_repetition_occurrences(
    text: str,
) -> dict[str, tuple[tuple[int, int], ...]]:
    char_ranges = []
    cursor = 0
    for separator in re.finditer(r"(?:\r?\n){2,}", text):
        char_ranges.append((cursor, separator.start()))
        cursor = separator.end()
    char_ranges.append((cursor, len(text)))

    positions = sorted(
        {position for byte_range in char_ranges for position in byte_range}
    )
    byte_offsets = {}
    char_cursor = 0
    byte_cursor = 0
    for position in positions:
        byte_cursor += len(text[char_cursor:position].encode("utf-8"))
        byte_offsets[position] = byte_cursor
        char_cursor = position

    occurrences: dict[str, list[tuple[int, int]]] = {}
    for start, end in char_ranges:
        block = text[start:end]
        normalized = _comparison_key(block)
        if (
            len(normalized) < _REPETITION_PARAGRAPH_MIN_NORMALIZED_CHARACTERS
            or len(_ANCHOR_TOKEN_RE.findall(normalized))
            < _REPETITION_PARAGRAPH_MIN_ANCHOR_TOKENS
        ):
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        occurrences.setdefault(digest, []).append(
            (byte_offsets[start], byte_offsets[end])
        )
    return {digest: tuple(ranges) for digest, ranges in occurrences.items()}


def source_repetition_profile(artifact: SourceArtifact) -> dict:
    """Classify source-level repeated-document evidence without publication text."""

    occurrences = _paragraph_repetition_occurrences(artifact.text)
    total_blocks = sum(len(ranges) for ranges in occurrences.values())
    unique_blocks = len(occurrences)
    excess_blocks = sum(max(0, len(ranges) - 1) for ranges in occurrences.values())
    ratio_ppm = (
        0 if total_blocks == 0 else (excess_blocks * 1_000_000) // total_blocks
    )
    repeated_document = bool(
        total_blocks >= _SOURCE_REPETITION_MIN_LONG_BLOCKS
        and excess_blocks >= _SOURCE_REPETITION_MIN_EXCESS_BLOCKS
        and ratio_ppm >= _SOURCE_REPETITION_MIN_EXCESS_RATIO_PPM
    )
    return {
        "policy_version": _REPETITION_POLICY_VERSION,
        "long_block_occurrence_count": total_blocks,
        "unique_long_block_count": unique_blocks,
        "excess_long_block_count": excess_blocks,
        "excess_ratio_ppm": ratio_ppm,
        "repeated_document": repeated_document,
    }


def source_repetition_profiles_metric(
    artifacts: Mapping[SourceName, SourceArtifact],
) -> dict[str, dict]:
    return {
        source: source_repetition_profile(artifacts[source])
        for source in sorted(artifacts)
    }


def _shingle_repetition_occurrences(
    text: str,
) -> dict[str, tuple[tuple[int, int], ...]]:
    tokens = []
    char_cursor = 0
    byte_cursor = 0
    for match in _ANCHOR_TOKEN_RE.finditer(text):
        byte_cursor += len(text[char_cursor : match.start()].encode("utf-8"))
        byte_start = byte_cursor
        raw_token = match.group(0)
        byte_cursor += len(raw_token.encode("utf-8"))
        tokens.append(
            (
                unicodedata.normalize("NFKC", raw_token).casefold(),
                byte_start,
                byte_cursor,
            )
        )
        char_cursor = match.end()

    occurrences: dict[str, list[tuple[int, int]]] = {}
    width = _REPETITION_SHINGLE_WIDTH
    for index in range(0, max(0, len(tokens) - width + 1)):
        shingle = tokens[index : index + width]
        if (
            sum(len(token) for token, _start, _end in shingle)
            < _REPETITION_SHINGLE_MIN_ANCHOR_CHARACTERS
        ):
            continue
        digest = hashlib.sha256(
            "\x1f".join(token for token, _start, _end in shingle).encode("utf-8")
        ).hexdigest()
        occurrences.setdefault(digest, []).append((shingle[0][1], shingle[-1][2]))
    return {digest: tuple(ranges) for digest, ranges in occurrences.items()}


def _structural_classes_for_range(
    byte_range: tuple[int, int],
    units: tuple[StructuralUnitSpan, ...],
) -> tuple[StructuralUnitType, ...]:
    start, end = byte_range
    classes = {
        unit.unit_type
        for unit in units
        if start < unit.byte_end and unit.byte_start < end
    }
    return tuple(sorted(classes)) or ("paragraph",)


def _output_structural_classes(
    byte_ranges: tuple[tuple[int, int], ...],
    units: tuple[StructuralUnitSpan, ...],
) -> tuple[tuple[StructuralUnitType, ...], ...]:
    return tuple(
        _structural_classes_for_range(byte_range, units) for byte_range in byte_ranges
    )


def _diagnostic_id(kind: str, fingerprint_sha256: str) -> str:
    return hashlib.sha256(
        f"{_REPETITION_POLICY_VERSION}:{kind}:{fingerprint_sha256}".encode("ascii")
    ).hexdigest()


def _excess_repetition_diagnostics(
    text: str,
    sources: tuple[SourceArtifact, ...],
) -> tuple[RepetitionDiagnostic, ...]:
    """Report replacement-attributable candidates for calibration; never mutate text."""

    output_by_kind = {
        "paragraph_fingerprint": _paragraph_repetition_occurrences(text),
        "long_token_shingle": _shingle_repetition_occurrences(text),
    }
    source_by_kind = {
        "paragraph_fingerprint": [
            _paragraph_repetition_occurrences(source.text) for source in sources
        ],
        "long_token_shingle": [
            _shingle_repetition_occurrences(source.text) for source in sources
        ],
    }
    diagnostics = []
    output_units = None
    for kind in ("long_token_shingle", "paragraph_fingerprint"):
        for fingerprint, ranges in sorted(output_by_kind[kind].items()):
            output_count = len(ranges)
            max_source_count = max(
                (
                    len(source_counts.get(fingerprint, ()))
                    for source_counts in source_by_kind[kind]
                ),
                default=0,
            )
            if output_count <= 1 or output_count <= max_source_count:
                continue
            if output_units is None:
                output_units = scan_structural_units(
                    SourceArtifact.from_text("grobid", text)
                )
            occurrence_classes = _output_structural_classes(ranges, output_units)
            diagnostics.append(
                RepetitionDiagnostic(
                    diagnostic_id=_diagnostic_id(kind, fingerprint),
                    kind=kind,
                    fingerprint_sha256=fingerprint,
                    output_count=output_count,
                    max_source_count=max_source_count,
                    excess_count=output_count - max_source_count,
                    structural_classes=tuple(
                        sorted(
                            {
                                value
                                for classes in occurrence_classes
                                for value in classes
                            }
                        )
                    ),
                    output_structural_classes=occurrence_classes,
                    output_byte_ranges=ranges,
                )
            )
    return tuple(diagnostics)


def repetition_diagnostics_metric(
    text: str,
    sources: tuple[SourceArtifact, ...],
) -> list[dict]:
    """Serialize canonical evidence recomputable from exact output/source bytes."""

    return [
        diagnostic.as_metric()
        for diagnostic in _excess_repetition_diagnostics(text, sources)
    ]


def assemble_selected_document(
    baseline: BaselineDocument,
    store: CandidateStore,
    graph: CandidateGraph,
    response: SelectionDecisionResponse | None = None,
    decision_methods: Mapping[str, DecisionMethod] | None = None,
) -> AssembledDocument:
    """Apply verified region selections while preserving all untouched bytes."""

    rejection_reasons: list[str] = []
    decision_methods = dict(decision_methods or {})
    try:
        store.validate_graph(graph)
        regions = []
        for region in graph.regions:
            candidate = store.candidate_metadata(region.baseline_candidate_id)
            if (
                candidate.source != baseline.artifact.source
                or candidate.artifact_digest != baseline.artifact.digest
            ):
                raise ConsensusContractError(
                    f"region {region.region_id!r} baseline candidate is not in "
                    "the baseline artifact"
                )
            store.candidate_bytes(candidate.candidate_id)
            regions.append((candidate.byte_start, candidate.byte_end, region))
    except ConsensusContractError as exc:
        raw_utf8 = baseline.artifact.raw_utf8
        require_publishable_text(
            baseline.artifact.text, candidate_id="baseline-document"
        )
        provenance = ()
        if raw_utf8:
            provenance = (
                OutputSpanProvenance(
                    output_byte_start=0,
                    output_byte_end=len(raw_utf8),
                    source=baseline.artifact.source,
                    artifact_digest=baseline.artifact.digest,
                    source_byte_start=0,
                    source_byte_end=len(raw_utf8),
                    candidate_id=None,
                    region_id=None,
                    decision_method="baseline_fallback",
                ),
            )
        return AssembledDocument(
            raw_utf8=raw_utf8,
            digest=hashlib.sha256(raw_utf8).hexdigest(),
            baseline_source=baseline.artifact.source,
            baseline_digest=baseline.artifact.digest,
            replaced_region_ids=(),
            fallback_region_ids=tuple(region.region_id for region in graph.regions),
            provenance=provenance,
            rejection_reasons=(f"invalid_graph:{type(exc).__name__}",),
            quarantined_repeated_sources=baseline.quarantined_repeated_sources,
        )
    decisions = {
        decision.region_id: decision
        for decision in (() if response is None else response.decisions)
    }
    graph_region_ids = {region.region_id for region in graph.regions}
    unexpected = set(decisions) - graph_region_ids
    if unexpected:
        rejection_reasons.extend(
            f"unknown_region:{region_id}" for region_id in sorted(unexpected)
        )
        decisions = {
            region_id: decision
            for region_id, decision in decisions.items()
            if region_id in graph_region_ids
        }

    regions.sort(key=lambda item: (item[0], item[1], item[2].region_id))

    effective_decisions: dict[str, RegionSelectionDecision] = {}
    selected_paths: dict[str, tuple[str, ...]] = {}
    for _byte_start, _byte_end, region in regions:
        decision = decisions.get(
            region.region_id,
            RegionSelectionDecision(region_id=region.region_id, action="keep_baseline"),
        )
        if (
            decision.action == "select_candidates"
            and region.region_id not in decision_methods
        ):
            rejection_reasons.append(f"missing_decision_method:{region.region_id}")
            decision = RegionSelectionDecision(
                region_id=region.region_id,
                action="keep_baseline",
            )
        selected_path = (
            (region.baseline_candidate_id,)
            if decision.action == "keep_baseline"
            else tuple(decision.candidate_ids)
        )
        if selected_path not in region.valid_paths:
            rejection_reasons.append(f"invalid_selection:{region.region_id}")
            selected_path = (region.baseline_candidate_id,)
            decision = RegionSelectionDecision(
                region_id=region.region_id,
                action="keep_baseline",
            )
        effective_decisions[region.region_id] = decision
        selected_paths[region.region_id] = selected_path

    cursor = 0
    output = bytearray()
    provenance: list[OutputSpanProvenance] = []
    replaced: list[str] = []
    fallback: list[str] = []
    for byte_start, byte_end, region in regions:
        if byte_start < cursor:
            rejection_reasons.append("overlapping_baseline_regions")
            return assemble_selected_document(
                baseline,
                store,
                CandidateGraph(regions=()),
                response=None,
            )
        untouched = baseline.artifact.raw_utf8[cursor:byte_start]
        output_start = len(output)
        output.extend(untouched)
        if untouched:
            provenance.append(
                OutputSpanProvenance(
                    output_byte_start=output_start,
                    output_byte_end=len(output),
                    source=baseline.artifact.source,
                    artifact_digest=baseline.artifact.digest,
                    source_byte_start=cursor,
                    source_byte_end=byte_start,
                    candidate_id=None,
                    region_id=None,
                    decision_method="untouched_baseline",
                )
            )
        decision = effective_decisions[region.region_id]
        selected_path = selected_paths[region.region_id]
        for candidate_id in selected_path:
            candidate = store.candidate_metadata(candidate_id)
            candidate_bytes = store.candidate_bytes(candidate_id)
            output_start = len(output)
            output.extend(candidate_bytes)
            provenance.append(
                OutputSpanProvenance(
                    output_byte_start=output_start,
                    output_byte_end=len(output),
                    source=candidate.source,
                    artifact_digest=candidate.artifact_digest,
                    source_byte_start=candidate.byte_start,
                    source_byte_end=candidate.byte_end,
                    candidate_id=candidate_id,
                    region_id=region.region_id,
                    decision_method=(
                        decision_methods.get(
                            region.region_id, "baseline_fallback"
                        )
                        if decision.action == "keep_baseline"
                        else decision_methods[region.region_id]
                    ),
                )
            )
        if decision.action == "keep_baseline":
            if region.region_id not in decision_methods:
                fallback.append(region.region_id)
        else:
            replaced.append(region.region_id)
        cursor = byte_end
    tail = baseline.artifact.raw_utf8[cursor:]
    output_start = len(output)
    output.extend(tail)
    if tail:
        provenance.append(
            OutputSpanProvenance(
                output_byte_start=output_start,
                output_byte_end=len(output),
                source=baseline.artifact.source,
                artifact_digest=baseline.artifact.digest,
                source_byte_start=cursor,
                source_byte_end=len(baseline.artifact.raw_utf8),
                candidate_id=None,
                region_id=None,
                decision_method="untouched_baseline",
            )
        )

    raw_utf8 = bytes(output)
    try:
        text = raw_utf8.decode("utf-8", errors="strict")
    except (
        UnicodeDecodeError
    ) as exc:  # pragma: no cover - source candidates already guard this
        raise ConsensusContractError("assembled document is not valid UTF-8") from exc
    require_publishable_text(text, candidate_id="assembled-document")
    structural_reasons = _structural_balance_reasons(text)
    if structural_reasons and replaced:
        fallback_result = assemble_selected_document(
            baseline,
            store,
            graph,
            response=None,
            decision_methods={},
        )
        return replace(
            fallback_result,
            rejection_reasons=fallback_result.rejection_reasons
            + tuple(f"structural_validation:{reason}" for reason in structural_reasons),
        )
    if structural_reasons:
        raise ConsensusContractError(
            f"baseline document failed structural validation: {structural_reasons!r}"
        )
    return AssembledDocument(
        raw_utf8=raw_utf8,
        digest=hashlib.sha256(raw_utf8).hexdigest(),
        baseline_source=baseline.artifact.source,
        baseline_digest=baseline.artifact.digest,
        replaced_region_ids=tuple(replaced),
        fallback_region_ids=tuple(fallback),
        provenance=tuple(provenance),
        rejection_reasons=tuple(rejection_reasons),
        repetition_diagnostics=_excess_repetition_diagnostics(
            text,
            store.source_artifacts() or (baseline.artifact,),
        ),
        quarantined_repeated_sources=baseline.quarantined_repeated_sources,
    )


def _candidate_decision_trace(
    plan: CandidateMergePlan,
    model_response: SelectionDecisionResponse | None,
    document: AssembledDocument,
    *,
    forced_fallback_reason: str | None = None,
) -> tuple[dict, ...]:
    """Describe every executable region choice without copying publication text."""

    combined = plan.combined_response(model_response)
    attempted = {decision.region_id: decision for decision in combined.decisions}
    model_decisions = {
        decision.region_id: decision
        for decision in combined.decisions
        if decision.region_id in set(plan.unresolved_region_ids)
    }
    replaced = set(document.replaced_region_ids)
    events = []
    for region in plan.graph.regions:
        decision = attempted.get(region.region_id)
        attempted_path = (
            (region.baseline_candidate_id,)
            if decision is None or decision.action == "keep_baseline"
            else tuple(decision.candidate_ids)
        )
        final_path = (
            attempted_path
            if region.region_id in replaced and forced_fallback_reason is None
            else (region.baseline_candidate_id,)
        )
        alternative_paths = [
            path
            for path in region.valid_paths
            if path != (region.baseline_candidate_id,)
        ]
        numbered_paths = [
            {"choice": 0, "candidate_ids": [region.baseline_candidate_id]}
        ] + [
            {"choice": index, "candidate_ids": list(path)}
            for index, path in enumerate(alternative_paths, start=1)
        ]
        selected_choice = next(
            item["choice"]
            for item in numbered_paths
            if tuple(item["candidate_ids"]) == final_path
        )
        if forced_fallback_reason is not None:
            decision_method = "baseline_fallback"
            decision_reason = forced_fallback_reason
        elif region.region_id in plan.deterministic_reasons:
            decision_method = "deterministic"
            decision_reason = plan.deterministic_reasons[region.region_id]
        elif region.region_id in model_decisions:
            model_decision = model_decisions[region.region_id]
            if model_decision.action == "keep_baseline":
                decision_method = "model_selected"
                decision_reason = "model_selected_keep_baseline"
            else:
                decision_method = "model_selected"
                decision_reason = "model_selected_numbered_path"
        else:
            decision_method = "baseline_fallback"
            decision_reason = "unresolved_selection"
        candidates = []
        candidate_ids = tuple(
            dict.fromkeys(
                candidate_id
                for path in region.valid_paths
                for candidate_id in path
            )
        )
        for candidate_id in candidate_ids:
            candidate = plan.store.candidate_metadata(candidate_id)
            raw = plan.store.candidate_bytes(candidate_id)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "source": candidate.source,
                    "artifact_digest": candidate.artifact_digest,
                    "occurrence_id": candidate.occurrence_id,
                    "structural_unit_id": candidate.structural_unit_id,
                    "candidate_type": candidate.candidate_type,
                    "source_byte_start": candidate.byte_start,
                    "source_byte_end": candidate.byte_end,
                    "byte_count": len(raw),
                    "content_sha256": hashlib.sha256(raw).hexdigest(),
                    "visible_text_digest": candidate.visible_text_digest,
                    "non_emphasis_ast_digest": candidate.non_emphasis_ast_digest,
                    "emphasis_occurrence_count": len(
                        candidate.emphasis_occurrence_ids
                    ),
                    "emphasis_occurrence_ids": list(
                        candidate.emphasis_occurrence_ids
                    ),
                }
            )
        events.append(
            {
                "region_id": region.region_id,
                "baseline_candidate_id": region.baseline_candidate_id,
                "decision_method": decision_method,
                "decision_reason": decision_reason,
                "candidate_count": len(candidates),
                "path_count": len(numbered_paths),
                "candidates": candidates,
                "numbered_paths": numbered_paths,
                "attempted_candidate_ids": list(attempted_path),
                "selected_choice": selected_choice,
                "selected_candidate_ids": list(final_path),
                "replaced_baseline": region.region_id in replaced,
            }
        )
    return tuple(events)


def merge_with_baseline_failsafe(
    baseline: BaselineDocument,
    artifacts: Mapping[SourceName, SourceArtifact],
    *,
    selection_resolver: SelectionResolver | None = None,
    selection_quality: SelectionQuality = "terra_selected",
    skeleton_occurrence_ids: Mapping[tuple[SourceName, str], str] | None = None,
    skeleton_slot_keys: Mapping[tuple[SourceName, str], str] | None = None,
) -> BaselineFirstMergeResult:
    """Complete from a valid baseline even when graph/model improvement fails.

    Candidate construction and selection are optional improvements. Any
    ordinary exception in those paths retains the complete verified baseline;
    it never asks a model to manufacture replacement text. A structurally or
    Unicode-invalid baseline still raises because extraction recovery, not
    silent publication, is required in that case.
    """

    baseline_store = CandidateStore(
        {(baseline.artifact.source, baseline.artifact.digest): baseline.artifact},
        {},
    )

    def baseline_only(
        warnings: tuple[str, ...],
        *,
        fallback_region_ids: tuple[str, ...] = (),
        plan: CandidateMergePlan | None = None,
        model_response: SelectionDecisionResponse | None = None,
        fallback_reason: str | None = None,
    ) -> BaselineFirstMergeResult:
        document = assemble_selected_document(
            baseline,
            baseline_store,
            CandidateGraph(regions=()),
        )
        document = replace(
            document,
            fallback_region_ids=fallback_region_ids,
            provenance=tuple(
                replace(span, decision_method="baseline_fallback")
                for span in document.provenance
            ),
        )
        baseline_reasons = final_document_safety_reasons(document.text)
        if baseline_reasons:
            raise ConsensusContractError(
                "baseline fallback failed final document validation: "
                f"{baseline_reasons!r}"
            )
        return BaselineFirstMergeResult(
            document=document,
            merge_quality="baseline_fallback",
            unresolved_region_count=len(fallback_region_ids),
            warnings=warnings,
            decision_trace=(
                ()
                if plan is None
                else _candidate_decision_trace(
                    plan,
                    model_response,
                    document,
                    forced_fallback_reason=fallback_reason or "baseline_fallback",
                )
            ),
            candidate_construction_trace=(
                () if plan is None else plan.construction_trace
            ),
            candidate_construction_counts=(
                {} if plan is None else plan.construction_counts
            ),
        )

    try:
        plan = build_candidate_merge_plan(
            baseline,
            artifacts,
            skeleton_occurrence_ids=skeleton_occurrence_ids,
            skeleton_slot_keys=skeleton_slot_keys,
        )
    except Exception as exc:
        return baseline_only((f"candidate_plan_failed:{type(exc).__name__}",))

    model_response = None
    resolved_selection_quality = selection_quality
    warnings: list[str] = []
    if plan.selection_graph.regions:
        if selection_resolver is None:
            warnings.append("selection_unavailable")
        else:
            try:
                resolution = selection_resolver(plan.store, plan.selection_graph)
                if isinstance(resolution, SelectionResolution):
                    model_response = resolution.response
                    resolved_selection_quality = resolution.quality
                else:
                    model_response = resolution
                if not isinstance(model_response, SelectionDecisionResponse):
                    raise TypeError(
                        "selection resolver returned an invalid response type"
                    )
                model_response = SelectionDecisionResponse(
                    decisions=tuple(
                        RegionSelectionDecision.model_validate(
                            decision.model_dump(mode="python")
                        )
                        for decision in model_response.decisions
                    )
                )
            except Exception as exc:
                warnings.append(f"selection_failed:{type(exc).__name__}")
                model_response = None

    try:
        combined_response = plan.combined_response(model_response)
        decision_methods = plan.decision_methods(model_response)
        document = assemble_selected_document(
            baseline,
            plan.store,
            plan.graph,
            combined_response,
            decision_methods=decision_methods,
        )
    except Exception as exc:
        return baseline_only(
            (*warnings, f"candidate_assembly_failed:{type(exc).__name__}"),
            fallback_region_ids=tuple(
                region.region_id for region in plan.graph.regions
            ),
            plan=plan,
            model_response=model_response,
            fallback_reason=f"candidate_assembly_failed:{type(exc).__name__}",
        )

    final_reasons = final_document_safety_reasons(document.text)
    if final_reasons:
        return baseline_only(
            (
                *warnings,
                *(f"selected_output_rejected:{reason}" for reason in final_reasons),
            ),
            fallback_region_ids=tuple(
                region.region_id for region in plan.graph.regions
            ),
            plan=plan,
            model_response=model_response,
            fallback_reason="selected_output_rejected",
        )

    if document.repetition_diagnostics:
        return baseline_only(
            (
                *warnings,
                "selected_output_rejected:excess_repetition",
            ),
            fallback_region_ids=tuple(
                region.region_id for region in plan.graph.regions
            ),
            plan=plan,
            model_response=model_response,
            fallback_reason="selected_output_rejected:excess_repetition",
        )

    unresolved_region_count = len(document.fallback_region_ids)
    model_decided_region_ids = {
        decision.region_id
        for decision in plan.combined_response(model_response).decisions
        if decision.region_id in set(plan.unresolved_region_ids)
    }
    if unresolved_region_count:
        merge_quality: MergeQuality = "baseline_fallback"
    elif model_decided_region_ids:
        merge_quality = resolved_selection_quality
    else:
        merge_quality = "consensus"
    return BaselineFirstMergeResult(
        document=document,
        merge_quality=merge_quality,
        unresolved_region_count=unresolved_region_count,
        warnings=tuple(warnings),
        decision_trace=_candidate_decision_trace(
            plan,
            model_response,
            document,
        ),
        candidate_construction_trace=plan.construction_trace,
        candidate_construction_counts=plan.construction_counts,
    )
