"""Minimal native-backed document outline for one-pass Alliance rendering.

The skeleton owns only heading roles and order.  Publication text continues to
come from exact extractor spans; the renderer may replace or remove a Markdown
heading marker, but it never authors body or heading text.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Literal, Mapping

from rapidfuzz.fuzz import ratio

from app.services.abc_markdown_policy import abc_markdown_report
from app.services.model_policy import MAX_BOUNDED_TARGET_CHOICES
from app.services.native_extractor_artifact import (
    load_native_extractor_artifact,
    load_native_style_artifact,
)
from app.services.native_style import validate_native_style_bytes
from app.services.source_contracts import ConsensusContractError, SourceArtifact, SourceName
from app.services.source_merge import (
    StructuralUnitSpan,
    _ALIGNMENT_AMBIGUITY_DELTA,
    _ALIGNMENT_POSITIONAL_WINDOW,
    _MAX_COMPARISON_UNIT_CHARS,
    _align_to_baseline,
    _comparison_key,
    _emphasis_occurrence_id,
    _inline_italic_profile,
    _inline_italic_analysis,
    scan_structural_units,
)


SkeletonRole = Literal["title", "section", "metadata"]
OccurrenceRegion = Literal["front", "body", "back"]
_MARKDOWN_HEADING = re.compile(rb"^ {0,3}(#{1,6})[ \t]+", re.MULTILINE)
_MARKDOWN_TABLE_ROW = re.compile(rb"^\|.*\|[ \t]*$")
_MARKDOWN_TABLE_SEPARATOR = re.compile(rb"^\|(?:[ \t]*:?-{3,}:?[ \t]*\|)+[ \t]*$")
_MARKDOWN_CODE_FENCE = re.compile(rb"^`{3,}[^`]*$")
_MARKDOWN_INLINE = re.compile(r"[`*_~\[\]()]|<[^>]+>")
_REFERENCE_MARKER = re.compile(
    rb"^\s*(?:[-+*]|\d+[.)]|\[\d+\]|\[\^\d+\]:)\s+"
)
_REFERENCE_ORDINAL = re.compile(rb"^\s*(?P<ordinal>\d+)[.)]\s+")
_BIBLIOGRAPHY_MARKER_AT_LINE_END = re.compile(
    rb"(?i)(?:#{1,6})[ \t]+(?:references?(?:[ \t]+and[ \t]+notes)?|bibliography|literature[ \t]+cited)[ \t]*$"
)
_FIGURE_CAPTION_LABEL = re.compile(
    rb"^(?P<indent>[ \t]*)(?P<opening>\*\*)?"
    rb"(?P<label>(?:(?:supplementary|extended[ \t]+data)[ \t]+)?"
    rb"(?:figure|fig\.)[ \t]*[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)"
    rb"(?P<punct>[.:])?",
    re.IGNORECASE,
)
_TABLE_CAPTION_LABEL = re.compile(
    rb"^(?P<indent>[ \t]*)(?P<opening>\*{1,3})"
    rb"(?P<label>(?:supplementary[ \t]+)?table[ \t]*[A-Za-z0-9]+"
    rb"(?:[._-][A-Za-z0-9]+)*)(?P<punct>\.)(?P<closing>\*{1,3})"
    rb"(?=[ \t]|$)",
    re.IGNORECASE,
)
_TABLE_ROW_ATTACHED_HEADING = re.compile(
    rb"^(?P<row>\|[^\r\n]*\|)(?P<heading>#{1,6}[ \t]+\S[^\r\n]*)$",
    re.MULTILINE,
)
_TABLE_CAPTION_VISIBLE_LABEL = re.compile(
    r"^(?:(?:supplementary)\s+)?table\s*[A-Za-z0-9]+"
    r"(?:[._-][A-Za-z0-9]+)*\.",
    re.IGNORECASE,
)
_NUMERIC_ORCID = re.compile(rb"(?P<id>\d{4}(?:-\d{4}){3})")
_BARE_ABSTRACT_LABEL = re.compile(
    rb"^(?P<label>(?i:abstract))(?P<separator>:[ \t]+)"
)
_BULLET_AFFILIATION = re.compile(
    rb"^(?P<bullet>[-+*][ \t]+)(?P<number>\d{1,2})(?P<space>[ \t]+)"
)
_BARE_AFFILIATION = re.compile(
    rb"^(?P<number>\d{1,2})(?P<space>[ \t]+)(?=[A-Z])"
)
_PROJECTION_LIST_MARKER = re.compile(rb"^\s*(?:[-+*]|\d+[.)]|\[\d+\])\s+")
_PROJECTION_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_FROZEN_NATIVE_MAPPING_UNIT_TYPES = frozenset(
    {"paragraph", "list", "figure_caption"}
)
_SOURCE_ORDER = {"grobid": 3, "docling": 2, "marker": 1}
_BACK_ROLE_KEYS = {
    "acknowledgments",
    "funding",
    "authornotes",
    "competinginterests",
    "dataavailability",
    "authorcontributions",
    "references",
}


def _identity(value: str) -> str:
    value = html.unescape(_MARKDOWN_INLINE.sub("", value))
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFC", value)
        if character.isalnum()
    )


def _digest_parts(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _node_text(node: ET.Element) -> str:
    return " ".join("".join(node.itertext()).split())


def _native_id(node: ET.Element, fallback: str) -> str:
    return (
        node.attrib.get("{http://www.w3.org/XML/1998/namespace}id")
        or node.attrib.get("id")
        or fallback
    )


@dataclass(frozen=True)
class NativeStructureArtifact:
    """Digest-bound native extractor bytes associated with exact Markdown."""

    source: SourceName
    markdown_digest: str
    native_digest: str
    receipt_digest: str
    native_bytes: bytes
    native_style_digest: str | None = None
    native_style_bytes: bytes | None = None
    expected_page_count: int | None = None
    covered_page_count: int | None = None

    def __post_init__(self) -> None:
        if hashlib.sha256(self.native_bytes).hexdigest() != self.native_digest:
            raise ConsensusContractError("native structure digest mismatch")
        if len(self.receipt_digest) != 64:
            raise ConsensusContractError("native structure receipt digest is invalid")
        if (self.native_style_digest is None) != (self.native_style_bytes is None):
            raise ConsensusContractError("native style digest and bytes must be paired")
        if self.native_style_bytes is not None and hashlib.sha256(
            self.native_style_bytes
        ).hexdigest() != self.native_style_digest:
            raise ConsensusContractError("native style digest mismatch")

    @classmethod
    def from_loaded(
        cls,
        source: SourceName,
        markdown: SourceArtifact,
        manifest: Mapping,
        native_bytes: bytes,
        native_style_bytes: bytes | None = None,
    ) -> "NativeStructureArtifact":
        if markdown.source != source:
            raise ConsensusContractError("native structure source mismatch")
        if manifest.get("markdown_sha256") != markdown.digest:
            raise ConsensusContractError("native structure Markdown digest mismatch")
        if manifest.get("native_sha256") != hashlib.sha256(native_bytes).hexdigest():
            raise ConsensusContractError("native structure manifest digest mismatch")
        return cls(
            source=source,
            markdown_digest=markdown.digest,
            native_digest=manifest["native_sha256"],
            receipt_digest=hashlib.sha256(
                json.dumps(
                    dict(manifest),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            native_bytes=native_bytes,
            native_style_digest=manifest.get("native_style_sha256"),
            native_style_bytes=native_style_bytes,
            expected_page_count=manifest.get("expected_page_count"),
            covered_page_count=(
                None
                if manifest.get("covered_pages") is None
                else len(manifest["covered_pages"])
            ),
        )

    @classmethod
    def for_test(
        cls,
        source: SourceName,
        markdown: SourceArtifact,
        native_bytes: bytes,
        native_style_bytes: bytes | None = None,
    ) -> "NativeStructureArtifact":
        return cls(
            source=source,
            markdown_digest=markdown.digest,
            native_digest=hashlib.sha256(native_bytes).hexdigest(),
            receipt_digest=hashlib.sha256(
                b"test-native-receipt\x00" + native_bytes
            ).hexdigest(),
            native_bytes=native_bytes,
            native_style_digest=(
                None
                if native_style_bytes is None
                else hashlib.sha256(native_style_bytes).hexdigest()
            ),
            native_style_bytes=native_style_bytes,
        )


@dataclass(frozen=True)
class NativeHeadingHint:
    native_id: str
    role: Literal["title", "section"]
    level: int
    identity: str
    native_order: int
    page_no: int | None = None


@dataclass(frozen=True)
class NativeEmphasisSpan:
    """One explicit native emphasis range over a block's visible text."""

    occurrence_id: str
    visible_start: int
    visible_end: int


@dataclass(frozen=True)
class NativeOccurrenceHint:
    native_id: str
    identity: str
    native_order: int
    page_no: int | None = None
    region: OccurrenceRegion = "body"
    native_emphasis_occurrence_ids: tuple[str, ...] = ()
    native_visible_text: str | None = None
    native_emphasis_spans: tuple[NativeEmphasisSpan, ...] = ()
    unit_type: str = "paragraph"

    @property
    def native_emphasis_count(self) -> int:
        return len(self.native_emphasis_occurrence_ids)


@dataclass(frozen=True)
class NativeStyleOccurrence:
    """One style-bearing PDF line mapped to a source Markdown region."""

    occurrence_id: str
    native_id: str
    native_order: int
    page_no: int
    region: OccurrenceRegion
    source_unit_id: str
    native_visible_text: str
    native_emphasis_spans: tuple[NativeEmphasisSpan, ...]
    unit_type: str = "paragraph"

    @property
    def native_emphasis_occurrence_ids(self) -> tuple[str, ...]:
        return tuple(span.occurrence_id for span in self.native_emphasis_spans)

    @property
    def native_emphasis_count(self) -> int:
        return len(self.native_emphasis_spans)


@dataclass(frozen=True)
class SkeletonHeading:
    unit_id: str
    occurrence_id: str
    native_id: str | None
    identity: str
    original_level: int
    final_level: int
    role: SkeletonRole
    source_byte_start: int
    source_byte_end: int


@dataclass(frozen=True)
class SkeletonOccurrence:
    unit_id: str
    occurrence_id: str
    unit_type: str
    native_id: str | None
    native_order: int | None
    page_no: int | None
    region: OccurrenceRegion
    section_slot: str | None
    slot_key: str
    source_byte_start: int
    source_byte_end: int
    native_emphasis_occurrence_ids: tuple[str, ...] = ()
    native_visible_text: str | None = None
    native_emphasis_spans: tuple[NativeEmphasisSpan, ...] = ()
    native_unit_type: str | None = None

    @property
    def native_emphasis_count(self) -> int:
        return len(self.native_emphasis_occurrence_ids)


@dataclass(frozen=True)
class DocumentSkeleton:
    skeleton_id: str
    projection_id: str
    source: SourceName
    artifact_digest: str
    headings: tuple[SkeletonHeading, ...]
    occurrences: tuple[SkeletonOccurrence, ...]
    native_artifact_digest: str | None
    native_receipt_digest: str | None
    payload_byte_count: int
    native_mapped_occurrence_count: int
    expected_page_count: int | None
    covered_page_count: int | None
    native_heading_count: int
    matched_native_heading_count: int
    title_proven: bool
    native_style_digest: str | None = None
    native_style_occurrences: tuple[NativeStyleOccurrence, ...] = ()
    unmapped_native_occurrences: tuple[NativeOccurrenceHint, ...] = ()
    unmapped_native_style_occurrences: tuple[NativeOccurrenceHint, ...] = ()
    native_style_emphasis_count: int = 0
    mapped_native_style_emphasis_count: int = 0
    unmapped_native_style_emphasis_count: int = 0
    auxiliary_markdown_body_emphasis_count: int = 0
    auxiliary_native_style_body_emphasis_count: int = 0
    native_body_emphasis_count: int = 0
    mapped_native_body_emphasis_count: int = 0
    native_reference_emphasis_count: int = 0
    mapped_native_reference_emphasis_count: int = 0
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkeletonSelection:
    skeleton: DocumentSkeleton
    trace: tuple[dict, ...]
    conflict: bool


class _MarkerHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.heading_level: int | None = None
        self._italic_starts: list[int] = []
        self._italic_ranges: list[tuple[int, int]] = []

    def _visible_length(self) -> int:
        return sum(len(part) for part in self.text)

    def handle_starttag(self, tag: str, _attrs) -> None:
        tag = tag.casefold()
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            self.heading_level = int(tag[1])
        if tag in {"em", "i"}:
            self._italic_starts.append(self._visible_length())

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() not in {"em", "i"} or not self._italic_starts:
            return
        start = self._italic_starts.pop()
        end = self._visible_length()
        if end > start:
            self._italic_ranges.append((start, end))

    def handle_data(self, data: str) -> None:
        self.text.append(data)

    def visible_text(self) -> str:
        return "".join(self.text)

    def emphasis_spans(self) -> tuple[NativeEmphasisSpan, ...]:
        visible = "".join(self.text)
        return tuple(
            NativeEmphasisSpan(
                occurrence_id=_emphasis_occurrence_id(visible, start, end),
                visible_start=start,
                visible_end=end,
            )
            for start, end in sorted(self._italic_ranges)
        )

    def emphasis_occurrence_ids(self) -> tuple[str, ...]:
        return tuple(span.occurrence_id for span in self.emphasis_spans())


def _grobid_hints(native_bytes: bytes) -> tuple[NativeHeadingHint, ...]:
    root = ET.fromstring(native_bytes)
    if _local_name(root.tag) != "TEI":
        raise ValueError("GROBID native root must be TEI")
    hints: list[NativeHeadingHint] = []
    for index, node in enumerate(root.iter()):
        if _local_name(node.tag) != "title":
            continue
        ancestors_are_header_title = any(
            _local_name(parent.tag) == "titleStmt"
            for parent in root.iter()
            if node in list(parent)
        )
        if not ancestors_are_header_title:
            continue
        identity = _identity(_node_text(node))
        if identity:
            hints.append(
                NativeHeadingHint(
                    native_id=_native_id(node, f"tei-title-{index}"),
                    role="title",
                    level=1,
                    identity=identity,
                    native_order=len(hints),
                )
            )
            break

    def visit(node: ET.Element, depth: int = 0) -> None:
        local = _local_name(node.tag)
        next_depth = depth + 1 if local == "div" else depth
        if local == "head":
            identity = _identity(_node_text(node))
            if identity:
                hints.append(
                    NativeHeadingHint(
                        native_id=_native_id(node, f"tei-head-{len(hints)}"),
                        role="section",
                        level=min(6, max(2, depth + 1)),
                        identity=identity,
                        native_order=len(hints),
                    )
                )
        for child in node:
            visit(child, next_depth)

    for child in root.iter():
        if _local_name(child.tag) in {"body", "back"}:
            visit(child)
    return tuple(hints)


def _docling_ordered_items(document: dict) -> list[dict]:
    indexed: dict[str, dict] = {}
    for collection_name in (
        "texts",
        "groups",
        "tables",
        "pictures",
        "key_value_items",
        "form_items",
    ):
        for item in document.get(collection_name, ()):
            if isinstance(item, dict) and isinstance(item.get("self_ref"), str):
                indexed[item["self_ref"]] = item

    def reference(value: object) -> str | None:
        if not isinstance(value, dict):
            return None
        for key in ("$ref", "cref"):
            if isinstance(value.get(key), str):
                return value[key]
        return None

    ordered_items: list[dict] = []
    visited: set[str] = set()

    def visit(value: object) -> None:
        ref = reference(value)
        if ref is not None:
            if ref in visited:
                return
            visited.add(ref)
            value = indexed.get(ref)
        if not isinstance(value, dict):
            return
        if value.get("self_ref") in indexed and "text" in value:
            ordered_items.append(value)
        for child in value.get("children") or ():
            visit(child)

    body = document.get("body")
    if isinstance(body, dict) and body.get("children"):
        for child in body["children"]:
            visit(child)
    else:
        ordered_items.extend(
            item for item in document.get("texts", ()) if isinstance(item, dict)
        )
    return ordered_items


def _docling_hints(native_bytes: bytes) -> tuple[NativeHeadingHint, ...]:
    document = json.loads(native_bytes)
    if not isinstance(document, dict):
        raise ValueError("Docling native document must be an object")
    if document.get("schema_name") != "DoclingDocument":
        raise ValueError("Docling native schema name is invalid")
    ordered_items = _docling_ordered_items(document)

    hints: list[NativeHeadingHint] = []
    for index, item in enumerate(ordered_items):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).casefold()
        if label not in {"title", "section_header"}:
            continue
        identity = _identity(str(item.get("text", "")))
        if not identity:
            continue
        role: Literal["title", "section"] = (
            "title" if label == "title" else "section"
        )
        native_level = item.get("level", 1)
        if type(native_level) is not int:
            native_level = 1
        page_no = None
        provenance = item.get("prov")
        if isinstance(provenance, list) and provenance and isinstance(provenance[0], dict):
            value = provenance[0].get("page_no")
            if type(value) is int and value >= 1:
                page_no = value
        hints.append(
            NativeHeadingHint(
                native_id=str(item.get("self_ref") or f"docling-text-{index}"),
                role=role,
                level=1 if role == "title" else min(6, max(2, native_level + 1)),
                identity=identity,
                native_order=len(hints),
                page_no=page_no,
            )
        )
    return tuple(hints)


def _marker_hints(native_bytes: bytes) -> tuple[NativeHeadingHint, ...]:
    document = json.loads(native_bytes)
    if not isinstance(document, dict):
        raise ValueError("Marker native document must be an object")
    if str(document.get("block_type", "")).rsplit(".", 1)[-1] != "Document":
        raise ValueError("Marker native root must be Document")
    hints: list[NativeHeadingHint] = []

    def visit(node: object) -> None:
        if not isinstance(node, dict):
            return
        block_type = str(node.get("block_type", "")).rsplit(".", 1)[-1]
        role: Literal["title", "section"] | None = None
        if block_type == "Title":
            role = "title"
        elif block_type == "SectionHeader":
            role = "section"
        if role is not None:
            parser = _MarkerHTML()
            parser.feed(str(node.get("html", "")))
            identity = _identity(" ".join(parser.text))
            if identity:
                hints.append(
                    NativeHeadingHint(
                        native_id=str(node.get("id") or f"marker-{len(hints)}"),
                        role=role,
                        level=(
                            1
                            if role == "title"
                            else min(6, max(2, parser.heading_level or 2))
                        ),
                        identity=identity,
                        native_order=len(hints),
                    )
                )
        for child in node.get("children") or ():
            visit(child)

    for child in document.get("children") or ():
        visit(child)
    return tuple(hints)


def native_heading_hints(native: NativeStructureArtifact) -> tuple[NativeHeadingHint, ...]:
    if native.source == "grobid":
        return _grobid_hints(native.native_bytes)
    if native.source == "docling":
        return _docling_hints(native.native_bytes)
    if native.source == "marker":
        return _marker_hints(native.native_bytes)
    raise ConsensusContractError(f"unsupported native structure source {native.source!r}")


def _grobid_occurrence_hints(native_bytes: bytes) -> tuple[NativeOccurrenceHint, ...]:
    root = ET.fromstring(native_bytes)
    if _local_name(root.tag) != "TEI":
        raise ValueError("GROBID native root must be TEI")
    hints: list[NativeOccurrenceHint] = []

    for index, node in enumerate(root.iter()):
        if _local_name(node.tag) != "title":
            continue
        if not any(
            _local_name(parent.tag) == "titleStmt" and node in list(parent)
            for parent in root.iter()
        ):
            continue
        identity = _identity(_node_text(node))
        if identity:
            hints.append(
                NativeOccurrenceHint(
                    native_id=_native_id(node, f"tei-title-{index}"),
                    identity=identity,
                    native_order=len(hints),
                    region="front",
                )
            )
        break

    payload_nodes = {"head", "p", "figDesc", "formula", "note", "biblStruct"}

    def is_explicit_italic(node: ET.Element) -> bool:
        return _local_name(node.tag) == "hi" and "italic" in " ".join(
            str(node.attrib.get(key, ""))
            for key in ("rend", "style", "type")
        ).casefold()

    def visible_text_and_italic_spans(
        node: ET.Element,
    ) -> tuple[str, tuple[NativeEmphasisSpan, ...]]:
        visible_parts: list[str] = []
        ranges: list[tuple[int, int]] = []

        def visible_length() -> int:
            return sum(len(part) for part in visible_parts)

        def visit_inline(current: ET.Element) -> None:
            italic = is_explicit_italic(current)
            start = visible_length()
            if current.text:
                visible_parts.append(current.text)
            for child in current:
                visit_inline(child)
                if child.tail:
                    visible_parts.append(child.tail)
            if italic and visible_length() > start:
                ranges.append((start, visible_length()))

        visit_inline(node)
        visible = "".join(visible_parts)
        return visible, tuple(
            NativeEmphasisSpan(
                occurrence_id=_emphasis_occurrence_id(visible, start, end),
                visible_start=start,
                visible_end=end,
            )
            for start, end in sorted(ranges)
        )

    def visit(node: ET.Element, region: OccurrenceRegion) -> None:
        local = _local_name(node.tag)
        if local in payload_nodes:
            identity = _identity(_node_text(node))
            if identity:
                native_visible_text, native_emphasis_spans = (
                    visible_text_and_italic_spans(node)
                )
                page_no = None
                coordinates = node.attrib.get("coords")
                if coordinates:
                    try:
                        page_no = int(coordinates.split(",", 1)[0])
                    except ValueError:
                        page_no = None
                hints.append(
                    NativeOccurrenceHint(
                        native_id=_native_id(node, f"tei-unit-{len(hints)}"),
                        identity=identity,
                        native_order=len(hints),
                        page_no=page_no,
                        region=region,
                        native_emphasis_occurrence_ids=tuple(
                            span.occurrence_id for span in native_emphasis_spans
                        ),
                        native_visible_text=native_visible_text,
                        native_emphasis_spans=native_emphasis_spans,
                        unit_type={
                            "head": "heading",
                            "figDesc": "figure_caption",
                            "formula": "equation",
                            "biblStruct": "reference",
                        }.get(local, "paragraph"),
                    )
                )
            return
        for child in node:
            visit(child, region)

    for node in root.iter():
        local = _local_name(node.tag)
        if local == "body":
            visit(node, "body")
        elif local == "back":
            visit(node, "back")
    return tuple(hints)


def _docling_occurrence_hints(native_bytes: bytes) -> tuple[NativeOccurrenceHint, ...]:
    document = json.loads(native_bytes)
    if not isinstance(document, dict) or document.get("schema_name") != "DoclingDocument":
        raise ValueError("Docling native schema name is invalid")
    hints = []

    def explicit_italic_spans(item: Mapping) -> tuple[NativeEmphasisSpan, ...]:
        formatting = item.get("formatting")
        text = str(item.get("text", ""))
        if isinstance(formatting, Mapping) and formatting.get("italic") is True and text:
            return (
                NativeEmphasisSpan(
                    occurrence_id=_emphasis_occurrence_id(text, 0, len(text)),
                    visible_start=0,
                    visible_end=len(text),
                ),
            )
        return ()

    current_region: OccurrenceRegion = "body"
    for index, item in enumerate(_docling_ordered_items(document)):
        native_visible_text = str(item.get("text", ""))
        identity = _identity(native_visible_text)
        if not identity:
            continue
        native_emphasis_spans = explicit_italic_spans(item)
        page_no = None
        provenance = item.get("prov")
        if isinstance(provenance, list) and provenance and isinstance(provenance[0], dict):
            value = provenance[0].get("page_no")
            if type(value) is int and value >= 1:
                page_no = value
        label = str(item.get("label", "")).casefold()
        if label in {"reference", "bibliography"} or (
            label == "section_header" and identity in _BACK_ROLE_KEYS
        ):
            current_region = "back"
        item_region: OccurrenceRegion = (
            "front" if label == "title" else current_region
        )
        hints.append(
            NativeOccurrenceHint(
                native_id=str(item.get("self_ref") or f"docling-unit-{index}"),
                identity=identity,
                native_order=len(hints),
                page_no=page_no,
                region=item_region,
                native_emphasis_occurrence_ids=tuple(
                    span.occurrence_id for span in native_emphasis_spans
                ),
                native_visible_text=native_visible_text,
                native_emphasis_spans=native_emphasis_spans,
                unit_type={
                    "section_header": "heading",
                    "list_item": "list",
                    "caption": "figure_caption",
                    "reference": "reference",
                    "bibliography": "reference",
                    "formula": "equation",
                }.get(label, "paragraph"),
            )
        )
    return tuple(hints)


def _marker_occurrence_hints(native_bytes: bytes) -> tuple[NativeOccurrenceHint, ...]:
    document = json.loads(native_bytes)
    if not isinstance(document, dict) or str(
        document.get("block_type", "")
    ).rsplit(".", 1)[-1] != "Document":
        raise ValueError("Marker native root must be Document")
    hints = []
    payload_types = {
        "Title",
        "SectionHeader",
        "Text",
        "ListItem",
        "Caption",
        "Footnote",
        "Reference",
        "Equation",
    }

    top_margin_italic_keys: dict[str, set[int]] = {}
    top_margin_nodes: list[tuple[str, tuple[str, ...]]] = []
    for page_index, page in enumerate(document.get("children") or ()):
        if not isinstance(page, Mapping) or str(
            page.get("block_type", "")
        ).rsplit(".", 1)[-1] != "Page":
            continue
        page_bbox = page.get("bbox")
        if not (
            isinstance(page_bbox, list)
            and len(page_bbox) == 4
            and all(isinstance(value, (int, float)) for value in page_bbox)
        ):
            continue
        page_top = min(page_bbox[1], page_bbox[3])
        page_height = abs(page_bbox[3] - page_bbox[1])
        if page_height <= 0:
            continue
        for child in page.get("children") or ():
            if not isinstance(child, Mapping) or str(
                child.get("block_type", "")
            ).rsplit(".", 1)[-1] != "Text":
                continue
            bbox = child.get("bbox")
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(value, (int, float)) for value in bbox)
                and max(bbox[1], bbox[3]) <= page_top + page_height * 0.08
            ):
                continue
            parser = _MarkerHTML()
            parser.feed(str(child.get("html", "")))
            visible = parser.visible_text()
            keys = tuple(
                key
                for span in parser.emphasis_spans()
                if (
                    key := _identity(
                        visible[span.visible_start : span.visible_end]
                    )
                )
            )
            if not keys:
                continue
            native_id = str(child.get("id") or f"marker-top-{page_index}")
            top_margin_nodes.append((native_id, keys))
            for key in keys:
                top_margin_italic_keys.setdefault(key, set()).add(page_index)
    page_furniture_ids = {
        native_id
        for native_id, keys in top_margin_nodes
        if any(len(top_margin_italic_keys[key]) >= 2 for key in keys)
    }

    def visit(
        node: object,
        region: OccurrenceRegion,
        page_no: int | None = None,
    ) -> OccurrenceRegion:
        if not isinstance(node, dict):
            return region
        block_type = str(node.get("block_type", "")).rsplit(".", 1)[-1]
        if block_type in payload_types:
            parser = _MarkerHTML()
            parser.feed(str(node.get("html", "")))
            visible_text = parser.visible_text()
            identity = _identity(visible_text)
            if block_type == "Reference" or (
                block_type == "SectionHeader" and identity in _BACK_ROLE_KEYS
            ):
                region = "back"
            if identity:
                node_page_no = node.get("page_id")
                if type(node_page_no) is int and node_page_no >= 0:
                    node_page_no += 1
                else:
                    node_page_no = page_no
                native_id = str(
                    node.get("id") or f"marker-unit-{len(hints)}"
                )
                hints.append(
                    NativeOccurrenceHint(
                        native_id=native_id,
                        identity=identity,
                        native_order=len(hints),
                        page_no=node_page_no,
                        region="front" if block_type == "Title" else region,
                        native_emphasis_occurrence_ids=(
                            parser.emphasis_occurrence_ids()
                        ),
                        native_visible_text=visible_text,
                        native_emphasis_spans=parser.emphasis_spans(),
                        unit_type={
                            "Title": "heading",
                            "SectionHeader": "heading",
                            "ListItem": "list",
                            "Caption": "figure_caption",
                            "Reference": "reference",
                            "Equation": "equation",
                        }.get(
                            block_type,
                            (
                                "page_header"
                                if native_id in page_furniture_ids
                                else "paragraph"
                            ),
                        ),
                    )
                )
            return region
        child_page_no = page_no
        if block_type == "Page" and page_no is None:
            page_id = str(node.get("id", ""))
            match = re.search(r"/page/(\d+)/", page_id)
            child_page_no = None if match is None else int(match.group(1)) + 1
        for child in node.get("children") or ():
            region = visit(child, region, child_page_no)
        return region

    current_region: OccurrenceRegion = "body"
    for page_index, child in enumerate(document.get("children") or ()):
        current_region = visit(child, current_region, page_index + 1)
    return tuple(hints)


def native_occurrence_hints(
    native: NativeStructureArtifact,
) -> tuple[NativeOccurrenceHint, ...]:
    if native.source == "grobid":
        return _grobid_occurrence_hints(native.native_bytes)
    if native.source == "docling":
        return _docling_occurrence_hints(native.native_bytes)
    if native.source == "marker":
        return _marker_occurrence_hints(native.native_bytes)
    raise ConsensusContractError(f"unsupported native structure source {native.source!r}")


def native_style_occurrence_hints(
    native: NativeStructureArtifact,
) -> tuple[NativeOccurrenceHint, ...]:
    """Return ordered positive-only PDF style lines from a bound sidecar."""

    if native.native_style_bytes is None:
        return ()
    document = validate_native_style_bytes(native.source, native.native_style_bytes)
    if document["status"] != "available":
        return ()
    hints = []
    for page in document["pages"]:
        for line in page["lines"]:
            text = line["text"]
            identity = _identity(text)
            spans = tuple(
                NativeEmphasisSpan(
                    occurrence_id=_emphasis_occurrence_id(
                        text, span["start"], span["end"]
                    ),
                    visible_start=span["start"],
                    visible_end=span["end"],
                )
                for span in line["italic_spans"]
            )
            hints.append(
                NativeOccurrenceHint(
                    native_id=line["native_id"],
                    identity=identity,
                    native_order=len(hints),
                    page_no=page["page_no"],
                    region="body",
                    native_emphasis_occurrence_ids=tuple(
                        span.occurrence_id for span in spans
                    ),
                    native_visible_text=text,
                    native_emphasis_spans=spans,
                )
            )
    return tuple(hints)


def effective_occurrence_region(
    region: OccurrenceRegion,
    unit_type: str,
) -> OccurrenceRegion:
    """Classify references once for skeleton totals and final receipts."""

    return "back" if unit_type == "reference" else region


def _markdown_heading_units(
    artifact: SourceArtifact,
) -> tuple[tuple[StructuralUnitSpan, int, str], ...]:
    headings = []
    for unit in scan_structural_units(artifact):
        if unit.unit_type != "heading":
            continue
        raw = artifact.raw_utf8[unit.byte_start : unit.byte_end]
        marker = _MARKDOWN_HEADING.match(raw)
        if marker is None:
            continue
        label = raw[marker.end() :].decode("utf-8", errors="strict")
        headings.append((unit, len(marker.group(1)), _identity(label)))
    return tuple(headings)


def _map_native_style_occurrences(
    artifact: SourceArtifact,
    native: NativeStructureArtifact,
    occurrences: list[SkeletonOccurrence],
    hints: tuple[NativeOccurrenceHint, ...],
) -> tuple[NativeStyleOccurrence, ...]:
    """Map PDF lines by ordered exact tokens; ambiguous lines abstain."""

    searchable = []
    for index, occurrence in enumerate(occurrences):
        if occurrence.unit_type not in _FROZEN_NATIVE_MAPPING_UNIT_TYPES:
            continue
        raw = artifact.raw_utf8[
            occurrence.source_byte_start : occurrence.source_byte_end
        ].decode("utf-8", errors="strict")
        analysis = _inline_italic_analysis(raw, occurrence.unit_type)
        if analysis is None:
            continue
        _profile, visible = analysis
        keys = tuple(token.key for token in _projection_tokens(visible))
        if keys:
            searchable.append((index, occurrence, keys))

    mapped = []
    cursor = 0
    for hint in hints:
        if not hint.identity:
            continue
        hint_keys = tuple(
            token.key for token in _projection_tokens(hint.native_visible_text or "")
        )
        if not hint_keys:
            continue
        candidates = []
        for index, occurrence, keys in searchable:
            if index < cursor:
                continue
            if (
                hint.page_no is not None
                and occurrence.page_no is not None
                and hint.page_no != occurrence.page_no
            ):
                continue
            count = sum(
                keys[start : start + len(hint_keys)] == hint_keys
                for start in range(0, len(keys) - len(hint_keys) + 1)
            )
            if count == 1:
                candidates.append((index, occurrence))
        if len(candidates) != 1:
            continue
        index, occurrence = candidates[0]
        cursor = index
        mapped.append(
            NativeStyleOccurrence(
                occurrence_id=_digest_parts(
                    "native-style-occurrence-v1",
                    artifact.source,
                    native.native_style_digest or "",
                    hint.native_id,
                    occurrence.occurrence_id,
                ),
                native_id=hint.native_id,
                native_order=hint.native_order,
                page_no=hint.page_no or 1,
                region=effective_occurrence_region(
                    occurrence.region, occurrence.unit_type
                ),
                source_unit_id=occurrence.unit_id,
                native_visible_text=hint.native_visible_text or "",
                native_emphasis_spans=hint.native_emphasis_spans,
            )
        )
    return tuple(mapped)


def build_document_skeleton(
    artifact: SourceArtifact,
    native: NativeStructureArtifact | None,
) -> DocumentSkeleton:
    """Project one native outline onto exact heading occurrences in Markdown."""

    if native is not None and (
        native.source != artifact.source or native.markdown_digest != artifact.digest
    ):
        raise ConsensusContractError("native structure is not bound to Markdown")
    units = scan_structural_units(artifact)
    markdown_headings = _markdown_heading_units(artifact)
    hints = () if native is None else native_heading_hints(native)
    occurrence_hints = () if native is None else native_occurrence_hints(native)
    style_hints = () if native is None else native_style_occurrence_hints(native)
    matches: dict[int, NativeHeadingHint] = {}
    used: set[int] = set()
    cursor = 0
    for hint in hints:
        match_index = next(
            (
                index
                for index in range(cursor, len(markdown_headings))
                if index not in used and markdown_headings[index][2] == hint.identity
            ),
            None,
        )
        if match_index is None:
            continue
        matches[match_index] = hint
        used.add(match_index)
        cursor = match_index + 1

    title_indexes = [
        index for index, hint in matches.items() if hint.role == "title"
    ]
    title_index = title_indexes[0] if len(title_indexes) == 1 else None
    title_proven = title_index is not None
    findings: list[str] = []
    if native is None:
        h1_indexes = [
            index for index, (_unit, level, _identity_value) in enumerate(markdown_headings)
            if level == 1
        ]
        title_index = 0 if h1_indexes == [0] else None
        title_proven = title_index is not None
        findings.append("native_structure_unavailable")
    elif not title_proven:
        findings.append("native_title_unproved")

    headings: list[SkeletonHeading] = []
    for index, (unit, original_level, identity) in enumerate(markdown_headings):
        hint = matches.get(index)
        if index == title_index:
            role: SkeletonRole = "title"
            final_level = 1
        elif title_index is not None and index < title_index:
            role = "metadata"
            final_level = 0
        elif hint is not None and hint.role == "section":
            role = "section"
            final_level = min(6, max(2, hint.level))
        else:
            role = "section"
            final_level = min(6, max(2, original_level))
        headings.append(
            SkeletonHeading(
                unit_id=unit.unit_id,
                occurrence_id=_digest_parts(
                    "skeleton-heading-occurrence-v1",
                    artifact.source,
                    artifact.digest,
                    unit.unit_id,
                    str(unit.byte_start),
                    str(unit.byte_end),
                    "" if hint is None else hint.native_id,
                    "" if native is None else native.receipt_digest,
                ),
                native_id=None if hint is None else hint.native_id,
                identity=identity,
                original_level=original_level,
                final_level=final_level,
                role=role,
                source_byte_start=unit.byte_start,
                source_byte_end=unit.byte_end,
            )
        )
    unit_identities = tuple(
        _identity(
            artifact.raw_utf8[unit.byte_start : unit.byte_end].decode(
                "utf-8", errors="strict"
            )
        )
        for unit in units
    )
    native_occurrence_matches: dict[int, NativeOccurrenceHint] = {}
    unit_cursor = 0
    for hint in occurrence_hints:
        match_index = next(
            (
                index
                for index in range(unit_cursor, len(units))
                if unit_identities[index] == hint.identity
            ),
            None,
        )
        if match_index is None:
            continue
        native_occurrence_matches[match_index] = hint
        unit_cursor = match_index + 1

    headings_by_unit = {heading.unit_id: heading for heading in headings}
    occurrences: list[SkeletonOccurrence] = []
    current_region: OccurrenceRegion = "front" if title_proven else "body"
    current_section_slot = None
    current_slot_key = f"{current_region}:root"
    section_ordinals: dict[tuple[str, str, str], int] = {}
    section_stack: list[tuple[int, str, int]] = []
    for index, unit in enumerate(units):
        heading = headings_by_unit.get(unit.unit_id)
        native_occurrence = native_occurrence_matches.get(index)
        if heading is not None:
            previous_region = current_region
            if heading.role in {"title", "metadata"}:
                current_region = "front"
            elif heading.identity in _BACK_ROLE_KEYS:
                current_region = "back"
            else:
                current_region = "body"
            current_section_slot = heading.occurrence_id
            if heading.role in {"title", "metadata"}:
                section_stack.clear()
                parent_path = ""
            else:
                if current_region != previous_region:
                    section_stack.clear()
                while (
                    section_stack
                    and section_stack[-1][0] >= heading.final_level
                ):
                    section_stack.pop()
                parent_path = "/".join(
                    f"{identity}:{ordinal}"
                    for _level, identity, ordinal in section_stack
                )
            ordinal_key = (current_region, parent_path, heading.identity)
            section_ordinals[ordinal_key] = section_ordinals.get(ordinal_key, 0) + 1
            current_heading = (
                heading.final_level,
                heading.identity,
                section_ordinals[ordinal_key],
            )
            if heading.role not in {"title", "metadata"}:
                section_stack.append(current_heading)
            path = "/".join(
                f"{identity}:{ordinal}"
                for _level, identity, ordinal in (
                    [current_heading]
                    if heading.role in {"title", "metadata"}
                    else section_stack
                )
            )
            current_slot_key = f"{current_region}:{path}"
        elif native_occurrence is not None and native_occurrence.region == "back":
            current_region = "back"
            current_slot_key = "back:root"
        native_id = (
            native_occurrence.native_id
            if native_occurrence is not None
            else None if heading is None else heading.native_id
        )
        occurrence_id = _digest_parts(
            "skeleton-body-occurrence-v2",
            artifact.source,
            artifact.digest,
            unit.unit_id,
            unit.unit_type,
            str(unit.byte_start),
            str(unit.byte_end),
            "" if native_id is None else native_id,
            "" if native is None else native.receipt_digest,
        )
        occurrence_region = effective_occurrence_region(
            current_region,
            unit.unit_type,
        )
        occurrence_slot_key = (
            "back:root"
            if occurrence_region == "back" and current_region != "back"
            else current_slot_key
        )
        occurrences.append(
            SkeletonOccurrence(
                unit_id=unit.unit_id,
                occurrence_id=occurrence_id,
                unit_type=unit.unit_type,
                native_id=native_id,
                native_order=(
                    None if native_occurrence is None else native_occurrence.native_order
                ),
                page_no=None if native_occurrence is None else native_occurrence.page_no,
                region=occurrence_region,
                section_slot=current_section_slot,
                slot_key=occurrence_slot_key,
                source_byte_start=unit.byte_start,
                source_byte_end=unit.byte_end,
                native_emphasis_occurrence_ids=(
                    ()
                    if native_occurrence is None
                    else native_occurrence.native_emphasis_occurrence_ids
                ),
                native_visible_text=(
                    None
                    if native_occurrence is None
                    else native_occurrence.native_visible_text
                ),
                native_emphasis_spans=(
                    ()
                    if native_occurrence is None
                    else native_occurrence.native_emphasis_spans
                ),
                native_unit_type=(
                    None
                    if native_occurrence is None
                    else native_occurrence.unit_type
                ),
            )
        )
    mapped_native_orders = {
        occurrence.native_order
        for occurrence in occurrences
        if occurrence.native_order is not None
    }
    for region in ("front", "body", "back"):
        candidate_indexes = [
            index
            for index, occurrence in enumerate(occurrences)
            if occurrence.native_id is None
            and occurrence.region == region
            and occurrence.unit_type in _FROZEN_NATIVE_MAPPING_UNIT_TYPES
        ]
        unmatched_hints = [
            hint
            for hint in occurrence_hints
            if hint.native_order not in mapped_native_orders
            and hint.region == region
            and hint.native_emphasis_count > 0
        ]
        if not candidate_indexes or not unmatched_hints:
            continue
        candidate_units = tuple(
            StructuralUnitSpan(
                unit_id=occurrences[index].unit_id,
                source=artifact.source,
                artifact_digest=artifact.digest,
                unit_type="paragraph",
                byte_start=occurrences[index].source_byte_start,
                byte_end=occurrences[index].source_byte_end,
                comparison_key=_comparison_key(
                    artifact.raw_utf8[
                        occurrences[index].source_byte_start :
                        occurrences[index].source_byte_end
                    ].decode("utf-8", errors="strict")
                ),
            )
            for index in candidate_indexes
        )
        hint_units = tuple(
            StructuralUnitSpan(
                unit_id=hint.native_id,
                source=artifact.source,
                artifact_digest=artifact.digest,
                unit_type="paragraph",
                byte_start=index,
                byte_end=index + 1,
                comparison_key=_comparison_key(
                    hint.native_visible_text or hint.identity
                ),
            )
            for index, hint in enumerate(unmatched_hints)
        )
        for candidate_position, match in _align_to_baseline(
            candidate_units,
            hint_units,
        ).items():
            occurrence_index = candidate_indexes[candidate_position]
            occurrence = occurrences[occurrence_index]
            hint = unmatched_hints[match.alternative_index]
            occurrences[occurrence_index] = replace(
                occurrence,
                occurrence_id=_digest_parts(
                    "skeleton-body-occurrence-v2",
                    artifact.source,
                    artifact.digest,
                    occurrence.unit_id,
                    occurrence.unit_type,
                    str(occurrence.source_byte_start),
                    str(occurrence.source_byte_end),
                    hint.native_id,
                    "" if native is None else native.receipt_digest,
                ),
                native_id=hint.native_id,
                native_order=hint.native_order,
                page_no=hint.page_no,
                native_emphasis_occurrence_ids=(
                    hint.native_emphasis_occurrence_ids
                ),
                native_visible_text=hint.native_visible_text,
                native_emphasis_spans=hint.native_emphasis_spans,
                native_unit_type=hint.unit_type,
            )
    native_style_occurrences = (
        ()
        if native is None
        else _map_native_style_occurrences(
            artifact, native, occurrences, style_hints
        )
    )
    projection_id = _digest_parts(
        "document-skeleton-projection-v1",
        *(
            f"{occurrence.unit_type}:{occurrence.region}"
            if occurrence.unit_type != "heading"
            else next(
                (
                    f"heading:{heading.role}:{heading.final_level}:{heading.identity}"
                    for heading in headings
                    if heading.unit_id == occurrence.unit_id
                ),
                "heading:unbound",
            )
            for occurrence in occurrences
        ),
    )
    skeleton_id = _digest_parts(
        "document-skeleton-v1",
        artifact.source,
        artifact.digest,
        "" if native is None else native.receipt_digest,
        projection_id,
        *(heading.occurrence_id for heading in headings),
        *(occurrence.occurrence_id for occurrence in occurrences),
    )
    mapped_native_orders = {
        occurrence.native_order
        for occurrence in occurrences
        if occurrence.native_order is not None
    }
    unmapped_native_occurrences = tuple(
        hint
        for hint in occurrence_hints
        if hint.native_order not in mapped_native_orders
    )
    mapped_native_style_orders = {
        occurrence.native_order for occurrence in native_style_occurrences
    }
    unmapped_native_style_occurrences = tuple(
        hint
        for hint in style_hints
        if hint.native_order not in mapped_native_style_orders
    )
    mapped_body_emphasis_count = sum(
        occurrence.native_emphasis_count
        for occurrence in occurrences
        if effective_occurrence_region(
            occurrence.region, occurrence.unit_type
        ) == "body"
    )
    mapped_reference_emphasis_count = sum(
        occurrence.native_emphasis_count
        for occurrence in occurrences
        if effective_occurrence_region(
            occurrence.region, occurrence.unit_type
        ) == "back"
    )
    unmapped_body_emphasis_count = sum(
        hint.native_emphasis_count
        for hint in occurrence_hints
        if hint.region == "body" and hint.native_order not in mapped_native_orders
    )
    unmapped_reference_emphasis_count = sum(
        hint.native_emphasis_count
        for hint in occurrence_hints
        if hint.region == "back" and hint.native_order not in mapped_native_orders
    )
    mapped_style_body_emphasis_count = sum(
        occurrence.native_emphasis_count
        for occurrence in native_style_occurrences
        if occurrence.region == "body"
    )
    mapped_style_reference_emphasis_count = sum(
        occurrence.native_emphasis_count
        for occurrence in native_style_occurrences
        if occurrence.region == "back"
    )
    native_style_emphasis_count = sum(
        hint.native_emphasis_count for hint in style_hints
    )
    mapped_native_style_emphasis_count = sum(
        occurrence.native_emphasis_count
        for occurrence in native_style_occurrences
    )
    return DocumentSkeleton(
        skeleton_id=skeleton_id,
        projection_id=projection_id,
        source=artifact.source,
        artifact_digest=artifact.digest,
        headings=tuple(headings),
        occurrences=tuple(occurrences),
        native_artifact_digest=None if native is None else native.native_digest,
        native_receipt_digest=None if native is None else native.receipt_digest,
        payload_byte_count=len(artifact.raw_utf8),
        native_mapped_occurrence_count=sum(
            occurrence.native_id is not None for occurrence in occurrences
        ),
        expected_page_count=(None if native is None else native.expected_page_count),
        covered_page_count=(None if native is None else native.covered_page_count),
        native_heading_count=len(hints),
        matched_native_heading_count=len(matches),
        title_proven=title_proven,
        native_style_digest=(None if native is None else native.native_style_digest),
        native_style_occurrences=native_style_occurrences,
        unmapped_native_occurrences=unmapped_native_occurrences,
        unmapped_native_style_occurrences=unmapped_native_style_occurrences,
        native_style_emphasis_count=native_style_emphasis_count,
        mapped_native_style_emphasis_count=mapped_native_style_emphasis_count,
        unmapped_native_style_emphasis_count=(
            sum(
                hint.native_emphasis_count
                for hint in unmapped_native_style_occurrences
            )
        ),
        auxiliary_native_style_body_emphasis_count=sum(
            hint.native_emphasis_count
            for hint in unmapped_native_style_occurrences
            if hint.region == "body"
        ),
        native_body_emphasis_count=(
            mapped_body_emphasis_count
            + unmapped_body_emphasis_count
            + mapped_style_body_emphasis_count
        ),
        mapped_native_body_emphasis_count=(
            mapped_body_emphasis_count + mapped_style_body_emphasis_count
        ),
        native_reference_emphasis_count=(
            mapped_reference_emphasis_count
            + unmapped_reference_emphasis_count
            + mapped_style_reference_emphasis_count
        ),
        mapped_native_reference_emphasis_count=(
            mapped_reference_emphasis_count + mapped_style_reference_emphasis_count
        ),
        findings=tuple(findings),
    )


def _whole_source_audit(artifact: SourceArtifact) -> list[dict]:
    if not artifact.raw_utf8:
        return []
    return [{
        "output_byte_start": 0,
        "output_byte_end": len(artifact.raw_utf8),
        "source": artifact.source,
        "artifact_digest": artifact.digest,
        "source_byte_start": 0,
        "source_byte_end": len(artifact.raw_utf8),
        "candidate_id": None,
        "region_id": None,
        "decision_method": "baseline_fallback",
    }]


def _copy_audit_interval(
    output: bytearray,
    rewritten_audit: list[dict],
    original: bytes,
    audit: list[dict],
    start: int,
    end: int,
) -> None:
    if start >= end:
        return
    copied = start
    for entry in audit:
        entry_start = entry.get("output_byte_start")
        entry_end = entry.get("output_byte_end")
        if type(entry_start) is not int or type(entry_end) is not int:
            raise ConsensusContractError("skeleton audit range is invalid")
        overlap_start = max(start, entry_start)
        overlap_end = min(end, entry_end)
        if overlap_start >= overlap_end:
            continue
        if overlap_start != copied:
            raise ConsensusContractError("skeleton audit has incomplete coverage")
        source_start = entry.get("source_byte_start")
        source_end = entry.get("source_byte_end")
        if type(source_start) is not int or type(source_end) is not int:
            raise ConsensusContractError("skeleton source range is invalid")
        clipped = dict(entry)
        clipped["source_byte_start"] = source_start + overlap_start - entry_start
        clipped["source_byte_end"] = source_end - (entry_end - overlap_end)
        output_start = len(output)
        output.extend(original[overlap_start:overlap_end])
        clipped["output_byte_start"] = output_start
        clipped["output_byte_end"] = len(output)
        rewritten_audit.append(clipped)
        copied = overlap_end
    if copied != end:
        raise ConsensusContractError("skeleton audit does not cover output")


def _transformation_id(
    operation: str,
    start: int,
    end: int,
    replacement: bytes,
) -> str:
    """Identify one concrete deterministic edit across later audit rewrites."""

    return hashlib.sha256(
        b"\x00".join(
            (
                operation.encode("ascii"),
                str(start).encode("ascii"),
                str(end).encode("ascii"),
                hashlib.sha256(replacement).hexdigest().encode("ascii"),
            )
        )
    ).hexdigest()


def _table_cell_count(row: bytes) -> int:
    """Count source-backed GFM cells without interpreting cell contents."""

    delimiters = 0
    backslashes = 0
    for value in row:
        if value == ord("\\"):
            backslashes += 1
            continue
        if value == ord("|") and backslashes % 2 == 0:
            delimiters += 1
        backslashes = 0
    return max(1, delimiters - 1)


def _render_missing_table_separators(
    text: str,
    audit: list[dict],
) -> tuple[str, list[dict], list[dict]]:
    """Add only the structural row required by Alliance S07."""

    original = text.encode("utf-8")
    lines = []
    cursor = 0
    for raw_line in original.splitlines(keepends=True):
        end = cursor + len(raw_line)
        lines.append((cursor, end, raw_line.rstrip(b"\r\n")))
        cursor = end
    if cursor < len(original):
        lines.append((cursor, len(original), original[cursor:]))

    insertions: list[tuple[int, bytes, int]] = []
    in_code_fence = False
    index = 0
    while index < len(lines):
        _start, _end, line = lines[index]
        if _MARKDOWN_CODE_FENCE.match(line):
            in_code_fence = not in_code_fence
            index += 1
            continue
        if in_code_fence or not _MARKDOWN_TABLE_ROW.match(line) or _MARKDOWN_TABLE_SEPARATOR.match(line):
            index += 1
            continue
        block_end = index + 1
        has_separator = False
        while block_end < len(lines):
            candidate = lines[block_end][2]
            if not (
                _MARKDOWN_TABLE_ROW.match(candidate)
                or _MARKDOWN_TABLE_SEPARATOR.match(candidate)
            ):
                break
            has_separator = has_separator or bool(
                _MARKDOWN_TABLE_SEPARATOR.match(candidate)
            )
            block_end += 1
        if not has_separator:
            _line_start, line_end, header = lines[index]
            cell_count = _table_cell_count(header)
            separator = b"|" + (b"---|" * cell_count) + b"\n"
            if line_end == len(original) and not original.endswith((b"\n", b"\r")):
                separator = b"\n" + separator
            insertions.append((line_end, separator, cell_count))
        index = block_end

    if not insertions:
        return text, audit, []

    output = bytearray()
    rewritten_audit: list[dict] = []
    events = []
    cursor = 0
    for ordinal, (position, separator, cell_count) in enumerate(insertions, start=1):
        _copy_audit_interval(
            output,
            rewritten_audit,
            original,
            audit,
            cursor,
            position,
        )
        output_start = len(output)
        output.extend(separator)
        transformation_id = _transformation_id(
            "alliance_table_separator", position, position, separator
        )
        rewritten_audit.append({
            "output_byte_start": output_start,
            "output_byte_end": len(output),
            "source": "deterministic_markup",
            "artifact_digest": hashlib.sha256(separator).hexdigest(),
            "source_byte_start": 0,
            "source_byte_end": len(separator),
            "candidate_id": None,
            "region_id": None,
            "decision_method": "deterministic",
            "transformation": "alliance_table_separator",
            "transformation_id": transformation_id,
        })
        events.append({
            "operation": "alliance_table_separator",
            "audit_span_emitted": True,
            "table_ordinal": ordinal,
            "column_count": cell_count,
            "reason": "alliance_s07_missing_separator",
            "transformation_id": transformation_id,
        })
        cursor = position
    _copy_audit_interval(
        output,
        rewritten_audit,
        original,
        audit,
        cursor,
        len(original),
    )
    return output.decode("utf-8", errors="strict"), rewritten_audit, events


def _audit_entry_at(audit: list[dict], position: int) -> dict | None:
    return next(
        (
            entry
            for entry in audit
            if type(entry.get("output_byte_start")) is int
            and type(entry.get("output_byte_end")) is int
            and entry["output_byte_start"] <= position < entry["output_byte_end"]
        ),
        None,
    )


def _heading_identity_at(raw: bytes, match: re.Match[bytes]) -> str:
    line_end = raw.find(b"\n", match.end())
    if line_end < 0:
        line_end = len(raw)
    label = raw[match.end() : line_end].rstrip(b"\r").decode(
        "utf-8", errors="strict"
    )
    return _identity(label)


def _bound_heading_unit_ids(
    raw: bytes,
    matches: list[re.Match[bytes]],
    audit: list[dict],
    skeleton: DocumentSkeleton,
    decision_trace: tuple[dict, ...],
) -> tuple[str | None, ...]:
    selected_candidates = {
        candidate_id
        for event in decision_trace
        for candidate_id in event.get("selected_candidate_ids", ())
    }
    candidates = {
        candidate.get("candidate_id"): candidate
        for event in decision_trace
        for candidate in event.get("candidates", ())
        if candidate.get("candidate_type") == "heading"
    }
    headings_by_unit = {heading.unit_id: heading for heading in skeleton.headings}
    aligned_unit_bindings: dict[str, str] = {}
    ambiguous_structural_units: set[str] = set()
    for candidate in candidates.values():
        if (
            candidate.get("source") != skeleton.source
            or candidate.get("artifact_digest") != skeleton.artifact_digest
        ):
            continue
        start = candidate.get("source_byte_start")
        end = candidate.get("source_byte_end")
        structural_unit_id = candidate.get("structural_unit_id")
        if type(start) is not int or type(end) is not int or not structural_unit_id:
            continue
        heading_unit_id = next(
            (
                heading.unit_id
                for heading in skeleton.headings
                if heading.source_byte_start <= start
                and end <= heading.source_byte_end
            ),
            None,
        )
        if heading_unit_id is not None:
            if structural_unit_id in ambiguous_structural_units:
                continue
            previous = aligned_unit_bindings.get(structural_unit_id)
            if previous is None:
                aligned_unit_bindings[structural_unit_id] = heading_unit_id
            elif previous != heading_unit_id:
                aligned_unit_bindings.pop(structural_unit_id, None)
                ambiguous_structural_units.add(structural_unit_id)
    bound: list[str | None] = []
    for match in matches:
        label_position = match.end()
        entry = _audit_entry_at(audit, label_position)
        if entry is None:
            bound.append(None)
            continue
        candidate_id = entry.get("candidate_id")
        candidate = candidates.get(candidate_id)
        candidate_unit = (
            None if candidate is None else candidate.get("structural_unit_id")
        )
        aligned_heading_unit = aligned_unit_bindings.get(candidate_unit)
        if (
            candidate_id in selected_candidates
            and aligned_heading_unit in headings_by_unit
        ):
            bound.append(aligned_heading_unit)
            continue
        if (
            entry.get("source") != skeleton.source
            or entry.get("artifact_digest") != skeleton.artifact_digest
        ):
            bound.append(None)
            continue
        source_start = entry.get("source_byte_start")
        output_start = entry.get("output_byte_start")
        if type(source_start) is not int or type(output_start) is not int:
            bound.append(None)
            continue
        source_position = source_start + label_position - output_start
        output_identity = _heading_identity_at(raw, match)
        direct = next(
            (
                heading.unit_id
                for heading in skeleton.headings
                if heading.source_byte_start <= source_position < heading.source_byte_end
                and heading.identity == output_identity
            ),
            None,
        )
        bound.append(direct)
    return tuple(bound)


def render_document_skeleton(
    text: str,
    audit: list[dict],
    skeleton: DocumentSkeleton,
    *,
    decision_trace: tuple[dict, ...] = (),
    force_titleless: bool = False,
) -> tuple[str, list[dict], list[dict]]:
    """Render exact occurrence-bound roles or conservatively remove every H1."""

    original = text.encode("utf-8")
    matches = list(_MARKDOWN_HEADING.finditer(original))
    bound_unit_ids = _bound_heading_unit_ids(
        original,
        matches,
        audit,
        skeleton,
        decision_trace,
    )
    expected_unit_ids = tuple(heading.unit_id for heading in skeleton.headings)
    binding_valid = (
        len(matches) == len(skeleton.headings)
        and bound_unit_ids == expected_unit_ids
    )
    safe_titleless = force_titleless or not binding_valid
    replacements = []
    for ordinal, match in enumerate(matches, start=1):
        heading = (
            None
            if safe_titleless
            else skeleton.headings[ordinal - 1]
        )
        original_level = len(match.group(1))
        final_level = (
            max(2, original_level)
            if heading is None
            else heading.final_level
        )
        if final_level == 0:
            start = match.start()
            end = match.end()
            replacement = b""
        else:
            start = match.start(1)
            end = match.end(1)
            replacement = b"#" * final_level
        old_prefix = original[start:end]
        if replacement == old_prefix:
            continue
        replacements.append(
            (start, end, replacement, ordinal, heading, original_level, final_level)
        )
    if not replacements:
        events = []
        if safe_titleless:
            events.append({
                "operation": "selected_document_skeleton",
                "audit_span_emitted": False,
                "heading_ordinal": 0,
                "render_mode": "safe_titleless",
                "reason": (
                    "forced_titleless" if force_titleless else "occurrence_binding_mismatch"
                ),
                "skeleton_id": skeleton.skeleton_id,
            })
        text, audit, table_events = _render_missing_table_separators(text, audit)
        return text, audit, [*events, *table_events]

    output = bytearray()
    rewritten_audit: list[dict] = []
    events: list[dict] = []
    cursor = 0
    for (
        start,
        end,
        replacement,
        ordinal,
        heading,
        original_level,
        final_level,
    ) in replacements:
        _copy_audit_interval(output, rewritten_audit, original, audit, cursor, start)
        if replacement:
            transformation_id = _transformation_id(
                "selected_document_skeleton", start, end, replacement
            )
            output_start = len(output)
            output.extend(replacement)
            rewritten_audit.append({
                "output_byte_start": output_start,
                "output_byte_end": len(output),
                "source": "deterministic_markup",
                "artifact_digest": hashlib.sha256(replacement).hexdigest(),
                "source_byte_start": 0,
                "source_byte_end": len(replacement),
                "candidate_id": None,
                "region_id": None,
                "decision_method": "deterministic",
                "transformation": "selected_document_skeleton",
                "transformation_id": transformation_id,
            })
        else:
            transformation_id = _transformation_id(
                "selected_document_skeleton", start, end, replacement
            )
        events.append({
            "operation": "selected_document_skeleton",
            "audit_span_emitted": bool(replacement),
            "heading_ordinal": ordinal,
            "render_mode": "safe_titleless" if safe_titleless else "selected_skeleton",
            "reason": (
                "forced_titleless"
                if force_titleless
                else "occurrence_binding_mismatch"
                if safe_titleless
                else "occurrence_bound_role"
            ),
            "skeleton_id": skeleton.skeleton_id,
            "occurrence_id": None if heading is None else heading.occurrence_id,
            "role": "section" if heading is None else heading.role,
            "original_level": original_level,
            "final_level": final_level,
            "native_id": None if heading is None else heading.native_id,
            "transformation_id": transformation_id,
        })
        cursor = end
    _copy_audit_interval(output, rewritten_audit, original, audit, cursor, len(original))
    rendered = bytes(output).decode("utf-8", errors="strict")
    rendered, rewritten_audit, table_events = _render_missing_table_separators(
        rendered,
        rewritten_audit,
    )
    return rendered, rewritten_audit, [*events, *table_events]


def _permute_complete_audit_intervals(
    text: str,
    audit: list[dict],
    intervals: list[tuple[int, int]],
) -> tuple[str, list[dict]]:
    """Reorder a complete byte partition while proving exactly-once ownership."""

    original = text.encode("utf-8")
    coverage = sorted(intervals)
    cursor = 0
    for start, end in coverage:
        if start != cursor or end < start:
            raise ConsensusContractError("role intervals do not cover content once")
        cursor = end
    if cursor != len(original):
        raise ConsensusContractError("role intervals do not cover complete content")

    output = bytearray()
    rewritten_audit: list[dict] = []
    for start, end in intervals:
        _copy_audit_interval(
            output,
            rewritten_audit,
            original,
            audit,
            start,
            end,
        )
    if len(output) != len(original):
        raise ConsensusContractError("role permutation changed content byte count")
    return output.decode("utf-8", errors="strict"), rewritten_audit


def _heading_block_end(raw: bytes, match: re.Match[bytes]) -> int:
    """Include only the heading line and its following blank separator."""

    line_end = raw.find(b"\n", match.end())
    if line_end < 0:
        return len(raw)
    cursor = line_end + 1
    while cursor < len(raw):
        next_end = raw.find(b"\n", cursor)
        if next_end < 0:
            next_end = len(raw)
        if raw[cursor:next_end].strip(b"\r\t "):
            break
        cursor = next_end + (next_end < len(raw))
    return cursor


def _replace_deterministic_markup(
    text: str,
    audit: list[dict],
    replacements: list[tuple[int, int, bytes, str]],
) -> tuple[str, list[dict], list[dict]]:
    """Apply non-overlapping structural-marker edits with complete audit coverage."""

    if not replacements:
        return text, audit, []
    original = text.encode("utf-8")
    output = bytearray()
    rewritten_audit: list[dict] = []
    events = []
    cursor = 0
    for ordinal, (start, end, replacement, operation) in enumerate(
        sorted(replacements), start=1
    ):
        if start < cursor or end < start:
            raise ConsensusContractError("structural marker replacements overlap")
        _copy_audit_interval(
            output, rewritten_audit, original, audit, cursor, start
        )
        output_start = len(output)
        output.extend(replacement)
        transformation_id = _transformation_id(
            operation, start, end, replacement
        )
        if replacement:
            rewritten_audit.append({
                "output_byte_start": output_start,
                "output_byte_end": len(output),
                "source": "deterministic_markup",
                "artifact_digest": hashlib.sha256(replacement).hexdigest(),
                "source_byte_start": 0,
                "source_byte_end": len(replacement),
                "candidate_id": None,
                "region_id": None,
                "decision_method": "deterministic",
                "transformation": operation,
                "transformation_id": transformation_id,
            })
        events.append({
            "operation": operation,
            "audit_span_emitted": bool(replacement),
            "ordinal": ordinal,
            "source_marker_byte_count": end - start,
            "output_marker_byte_count": len(replacement),
            "transformation_id": transformation_id,
        })
        cursor = end
    _copy_audit_interval(
        output, rewritten_audit, original, audit, cursor, len(original)
    )
    return output.decode("utf-8", errors="strict"), rewritten_audit, events


def _bound_occurrences_for_output_unit(
    unit: StructuralUnitSpan,
    audit: Sequence[Mapping],
    skeletons: Mapping[SourceName, DocumentSkeleton],
) -> tuple[SkeletonOccurrence, ...]:
    """Return source occurrences that own bytes in one final structural unit."""

    occurrences_by_source = {
        source: tuple(skeleton.occurrences)
        for source, skeleton in skeletons.items()
    }
    bound: list[SkeletonOccurrence] = []
    seen: set[tuple[str, str]] = set()
    for entry in audit:
        output_start = entry.get("output_byte_start")
        output_end = entry.get("output_byte_end")
        if type(output_start) is not int or type(output_end) is not int:
            continue
        overlap_start = max(unit.byte_start, output_start)
        overlap_end = min(unit.byte_end, output_end)
        source = entry.get("source")
        source_start = entry.get("source_byte_start")
        if (
            overlap_start >= overlap_end
            or source == "deterministic_markup"
            or source not in occurrences_by_source
            or type(source_start) is not int
        ):
            continue
        selected_start = source_start + overlap_start - output_start
        selected_end = selected_start + overlap_end - overlap_start
        occurrence = next(
            (
                candidate
                for candidate in occurrences_by_source[source]
                if candidate.source_byte_start <= selected_start
                and selected_end <= candidate.source_byte_end
            ),
            None,
        )
        if occurrence is None:
            continue
        key = (source, occurrence.occurrence_id)
        if key not in seen:
            seen.add(key)
            bound.append(occurrence)
    return tuple(bound)


def _effective_bound_unit_role(
    unit: StructuralUnitSpan,
    audit: Sequence[Mapping],
    skeletons: Mapping[SourceName, DocumentSkeleton],
) -> tuple[str | None, str]:
    """Resolve one output role, preferring explicit native extractor roles."""

    bound = _bound_occurrences_for_output_unit(unit, audit, skeletons)
    native_roles = {
        occurrence.native_unit_type
        for occurrence in bound
        if occurrence.native_unit_type not in {None, "page_header"}
    }
    specific_native_roles = native_roles - {"heading", "paragraph", "list"}
    if len(specific_native_roles) == 1:
        return next(iter(specific_native_roles)), "native"
    if len(specific_native_roles) > 1:
        return None, "conflicting_native"
    source_roles = {occurrence.unit_type for occurrence in bound}
    if len(source_roles) == 1:
        return next(iter(source_roles)), "source_markdown"
    if len(source_roles) > 1:
        return None, "conflicting_source_markdown"
    if len(native_roles) == 1:
        return next(iter(native_roles)), "native"
    if len(native_roles) > 1:
        return None, "conflicting_native"
    return None, "unbound"


def _bibliography_marker_ranges(raw: bytes) -> list[tuple[int, int, bytes, str]]:
    """Find schema-only bibliography markers, including a marker glued to text."""

    ranges: list[tuple[int, int, bytes, str]] = []
    lines = raw.splitlines(keepends=True)
    offsets = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    for index, line in enumerate(lines):
        content = line.rstrip(b"\r\n")
        match = _BIBLIOGRAPHY_MARKER_AT_LINE_END.search(content)
        if match is not None:
            start = offsets[index] + match.start()
            end = offsets[index] + match.end()
            if not content[: match.start()].strip():
                start = offsets[index]
                end = offsets[index] + len(line)
                following = index + 1
                while following < len(lines) and not lines[following].strip():
                    end = offsets[following] + len(lines[following])
                    following += 1
            ranges.append(
                (
                    start,
                    end,
                    b"",
                    "alliance_bibliography_heading_remove",
                )
            )
    return ranges


def _canonical_inserted_heading(prefix: bytes, heading: bytes) -> bytes:
    if not prefix:
        return heading + b"\n\n"
    if prefix.endswith(b"\n\n"):
        return heading + b"\n\n"
    if prefix.endswith(b"\n"):
        return b"\n" + heading + b"\n\n"
    return b"\n\n" + heading + b"\n\n"


def _canonical_bibliography_heading(prefix: bytes) -> bytes:
    return _canonical_inserted_heading(prefix, b"## References")


def _figure_caption_marker_ranges(
    raw: bytes,
) -> list[tuple[int, int, bytes, str]]:
    """Convert visible figure-caption labels into Alliance H3 records.

    Only Markdown markers and separating whitespace change. The existing
    label and caption bytes retain their source audit spans.
    """

    replacements: list[tuple[int, int, bytes, str]] = []
    cursor = 0
    for line in raw.splitlines(keepends=True):
        content = line.rstrip(b"\r\n")
        match = _FIGURE_CAPTION_LABEL.match(content)
        if match is None:
            cursor += len(line)
            continue
        label_start = match.start("label")
        replacements.append((
            cursor,
            cursor + label_start,
            b"### ",
            "alliance_figure_label_heading",
        ))

        after_label = match.end()
        label_close_end = after_label
        if content[after_label : after_label + 2] == b"**":
            label_close_end += 2
        separator_end = label_close_end
        while separator_end < len(content) and content[separator_end] in b" \t":
            separator_end += 1
        outer_close = (
            match.group("opening") is not None
            and label_close_end == after_label
            and content.endswith(b"**")
            and len(content) - 2 >= separator_end
        )
        caption_end = len(content) - (2 if outer_close else 0)
        if separator_end < caption_end:
            replacements.append((
                cursor + after_label,
                cursor + separator_end,
                b"\n\n",
                "alliance_figure_label_caption_boundary",
            ))
        elif label_close_end > after_label:
            replacements.append((
                cursor + after_label,
                cursor + label_close_end,
                b"",
                "alliance_figure_label_bold_remove",
            ))
        if outer_close:
            replacements.append((
                cursor + len(content) - 2,
                cursor + len(content),
                b"",
                "alliance_figure_caption_outer_bold_remove",
            ))
        cursor += len(line)
    return replacements


def _table_caption_marker_ranges(
    raw: bytes,
) -> list[tuple[int, int, bytes, str]]:
    """Normalize only emphasis delimiters around a visible table label."""

    replacements: list[tuple[int, int, bytes, str]] = []
    cursor = 0
    for line in raw.splitlines(keepends=True):
        content = line.rstrip(b"\r\n")
        match = _TABLE_CAPTION_LABEL.match(content)
        if match is not None:
            for group in ("opening", "closing"):
                start = cursor + match.start(group)
                end = cursor + match.end(group)
                if raw[start:end] != b"**":
                    replacements.append((
                        start,
                        end,
                        b"**",
                        "alliance_table_label_emphasis_marker",
                    ))
        cursor += len(line)
    return replacements


def _table_heading_boundary_ranges(
    raw: bytes,
) -> list[tuple[int, int, bytes, str]]:
    """Separate a heading attached directly to the closing pipe of a table row."""

    return [
        (
            match.start("heading"),
            match.start("heading"),
            b"\n\n",
            "alliance_table_heading_boundary",
        )
        for match in _TABLE_ROW_ATTACHED_HEADING.finditer(raw)
    ]


def _render_figure_legend_slot(
    text: str,
    audit: list[dict],
    source: SourceName,
) -> tuple[str, list[dict], list[dict]]:
    """Move source-backed figure captions into one reader-visible role slot."""

    events: list[dict] = []
    artifact = SourceArtifact.from_text(source, text)
    units = scan_structural_units(artifact)
    figure_indexes = {
        index for index, unit in enumerate(units) if unit.unit_type == "figure_caption"
    }
    figure_legend_headings = [
        index
        for index, unit in enumerate(units)
        if unit.unit_type == "heading"
        and _identity(
            _MARKDOWN_HEADING.sub(
                b"", artifact.raw_utf8[unit.byte_start : unit.byte_end], count=1
            ).decode("utf-8", errors="strict")
        )
        == "figurelegends"
    ]
    if not figure_indexes:
        return text, audit, events

    if len(figure_legend_headings) != 1:
        starts = [unit.byte_start for unit in units]
        blocks = [
            (
                unit.byte_start,
                starts[index + 1] if index + 1 < len(starts) else len(artifact.raw_utf8),
            )
            for index, unit in enumerate(units)
        ]
        reference_heading_index = next(
            (
                index
                for index, unit in enumerate(units)
                if unit.unit_type == "heading"
                and _identity(
                    _MARKDOWN_HEADING.sub(
                        b"",
                        artifact.raw_utf8[unit.byte_start : unit.byte_end],
                        count=1,
                    ).decode("utf-8", errors="strict")
                )
                == "references"
            ),
            None,
        )
        nonfigure_indexes = [
            index for index in range(len(units)) if index not in figure_indexes
        ]
        insertion_position = (
            len(nonfigure_indexes)
            if reference_heading_index is None
            else nonfigure_indexes.index(reference_heading_index)
        )
        ordered_indexes = [
            *nonfigure_indexes[:insertion_position],
            *sorted(figure_indexes),
            *nonfigure_indexes[insertion_position:],
        ]
        prefix_end = starts[0] if starts else len(artifact.raw_utf8)
        if ordered_indexes != list(range(len(units))):
            text, audit = _permute_complete_audit_intervals(
                text,
                audit,
                [(0, prefix_end), *(blocks[index] for index in ordered_indexes)],
            )
            events.append({
                "operation": "alliance_figure_legend_role_order",
                "audit_span_emitted": False,
                "reason": "visible_figure_captions_moved_to_document_role_slot",
                "figure_unit_count": len(figure_indexes),
                "content_bytes_permuted_once": len(artifact.raw_utf8),
            })
        moved_artifact = SourceArtifact.from_text(source, text)
        moved_figure_units = [
            unit
            for unit in scan_structural_units(moved_artifact)
            if unit.unit_type == "figure_caption"
        ]
        if not moved_figure_units:
            return text, audit, events
        legend_start = moved_figure_units[0].byte_start
        raw = moved_artifact.raw_utf8
        text, audit, insert_events = _replace_deterministic_markup(
            text,
            audit,
            [(
                legend_start,
                legend_start,
                _canonical_inserted_heading(raw[:legend_start], b"## Figure Legends"),
                "alliance_figure_legend_heading_insert",
            )],
        )
        events.extend(insert_events)

    raw = text.encode("utf-8")
    legend_heading = next(
        (
            match
            for match in _MARKDOWN_HEADING.finditer(raw)
            if _identity(_heading_label(raw, match).decode("utf-8"))
            == "figurelegends"
        ),
        None,
    )
    if legend_heading is None:
        return text, audit, events
    references_heading = next(
        (
            match
            for match in _MARKDOWN_HEADING.finditer(raw)
            if match.start() > legend_heading.start()
            and _identity(_heading_label(raw, match).decode("utf-8")) == "references"
        ),
        None,
    )
    slot_end = len(raw) if references_heading is None else references_heading.start()
    slot_start = _heading_block_end(raw, legend_heading)
    replacements = [
        (start + slot_start, end + slot_start, value, operation)
        for start, end, value, operation in _figure_caption_marker_ranges(
            raw[slot_start:slot_end]
        )
    ]
    text, audit, marker_events = _replace_deterministic_markup(
        text, audit, replacements
    )
    events.extend(marker_events)
    return text, audit, events


def _front_matter_marker_ranges(
    raw: bytes,
) -> list[tuple[int, int, bytes, str]]:
    """Return bounded Alliance front-matter marker edits before the body."""

    replacements: list[tuple[int, int, bytes, str]] = []
    lines = raw.splitlines(keepends=True)
    offsets = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    first_body_heading = len(raw)
    for match in _MARKDOWN_HEADING.finditer(raw):
        label = _identity(_heading_label(raw, match).decode("utf-8"))
        if len(match.group(1)) >= 2 and label != "abstract":
            first_body_heading = match.start()
            break

    nonblank_front = [
        index
        for index, line in enumerate(lines)
        if offsets[index] < first_body_heading and line.strip()
    ]
    for position, index in enumerate(nonblank_front):
        line = lines[index].rstrip(b"\r\n")
        line_start = offsets[index]

        if line.startswith(b"**ORCIDs:**"):
            for match in _NUMERIC_ORCID.finditer(line):
                prefix = line[max(0, match.start() - 18) : match.start()].lower()
                if b"orcid.org/" not in prefix:
                    replacements.append((
                        line_start + match.start("id"),
                        line_start + match.start("id"),
                        b"https://orcid.org/",
                        "alliance_orcid_url_prefix",
                    ))

        abstract = _BARE_ABSTRACT_LABEL.match(line)
        if abstract is not None:
            replacements.extend([
                (
                    line_start,
                    line_start,
                    b"## ",
                    "alliance_abstract_heading_marker",
                ),
                (
                    line_start + abstract.start("separator"),
                    line_start + abstract.end("separator"),
                    b"\n\n",
                    "alliance_abstract_heading_separator",
                ),
            ])
            continue

        affiliation = _BULLET_AFFILIATION.match(line)
        if affiliation is not None:
            replacements.extend([
                (
                    line_start + affiliation.start("bullet"),
                    line_start + affiliation.end("bullet"),
                    b"",
                    "alliance_affiliation_list_marker",
                ),
                (
                    line_start + affiliation.end("number"),
                    line_start + affiliation.end("number"),
                    b".",
                    "alliance_affiliation_ordinal_marker",
                ),
            ])
            continue
        bare_affiliation = _BARE_AFFILIATION.match(line)
        if bare_affiliation is not None:
            replacements.append((
                line_start + bare_affiliation.end("number"),
                line_start + bare_affiliation.end("number"),
                b".",
                "alliance_affiliation_ordinal_marker",
            ))
            continue

        if position + 1 >= len(nonblank_front):
            continue
        next_line = lines[nonblank_front[position + 1]].rstrip(b"\r\n")
        words = line.split()
        probable_article_category = (
            0 < len(words) <= 4
            and len(line) <= 60
            and b"," not in line
            and not line.startswith((b"#", b"*", b"-", b"|"))
            and next_line.count(b"<sup>") >= 2
            and b"," in next_line
        )
        if probable_article_category:
            replacements.append((
                line_start,
                line_start,
                b"**Categories:** ",
                "alliance_article_category_marker",
            ))
        if line.lstrip().lower().startswith(b"* correspondence:") and index > 0:
            previous_line = lines[index - 1]
            if previous_line.strip() and (
                _BULLET_AFFILIATION.match(previous_line.rstrip(b"\r\n"))
                or _BARE_AFFILIATION.match(previous_line.rstrip(b"\r\n"))
            ):
                replacements.append((
                    line_start,
                    line_start,
                    b"\n",
                    "alliance_front_list_block_separator",
                ))
    return replacements


def reconcile_document_transformations(
    audit: list[dict],
    events: list[dict],
) -> list[dict]:
    """Mark only concrete edit occurrences present in the final audit."""

    remaining = Counter(
        entry.get("transformation_id")
        for entry in audit
        if entry.get("transformation_id")
    )
    reconciled = []
    for event in events:
        transformation_id = event.get("transformation_id")
        if event.get("audit_span_emitted") is True:
            if remaining[transformation_id] > 0:
                remaining[transformation_id] -= 1
            else:
                event = {
                    **event,
                    "audit_span_emitted": False,
                    "audit_span_superseded": True,
                }
        reconciled.append(event)
    return reconciled


@dataclass(frozen=True)
class _ProjectionToken:
    key: str
    visible_start: int
    visible_end: int
    output_byte_start: int | None = None
    output_byte_end: int | None = None
    target_occurrence_id: str | None = None


@dataclass(frozen=True)
class _ProjectionTarget:
    source: str
    artifact_digest: str
    occurrence: SkeletonOccurrence
    visible: str
    tokens: tuple[_ProjectionToken, ...]
    output_byte_start: int
    output_byte_end: int
    ordinal: int = 0
    content_byte_start: int = 0
    content_byte_end: int = 0
    visible_digest: str = ""
    non_emphasis_ast_digest: str = ""
    existing_emphasis_occurrence_ids: tuple[str, ...] = ()
    existing_emphasis_spans: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class _ProjectionUnitPair:
    target: _ProjectionTarget
    similarity: float
    ambiguous: bool
    candidates: tuple[tuple[_ProjectionTarget, float], ...]
    model_only: bool = False


@dataclass(frozen=True)
class _CharacterIntervalMapping:
    """One source character interval mapped onto an existing target string."""

    target_visible_start: int
    target_visible_end: int
    alignment_digest: str


@dataclass(frozen=True)
class _SharedCharacterAlignment:
    """One reusable Unicode character map for a complete donor/target unit pair."""

    source: str
    target: str
    source_positions: tuple[int, ...]
    target_positions: tuple[int, ...]
    source_to_target: tuple[int | None, ...]
    alignment_digest: str

    def interval(
        self,
        source_start: int,
        source_end: int,
        *,
        expand_complete_tokens: bool = True,
    ) -> _CharacterIntervalMapping | None:
        if not 0 <= source_start < source_end <= len(self.source):
            return None
        source_indexes = [
            index
            for index, position in enumerate(self.source_positions)
            if source_start <= position < source_end
        ]
        if not source_indexes:
            return None
        mapped_indexes = [
            self.source_to_target[index]
            for index in source_indexes
            if self.source_to_target[index] is not None
        ]
        if not mapped_indexes or any(
            right <= left
            for left, right in zip(mapped_indexes, mapped_indexes[1:])
        ):
            return None
        target_start = self.target_positions[mapped_indexes[0]]
        target_end = self.target_positions[mapped_indexes[-1]] + 1
        if expand_complete_tokens:
            source_tokens = _projection_tokens(self.source)
            overlapping_source_tokens = [
                token
                for token in source_tokens
                if token.visible_start < source_end
                and source_start < token.visible_end
            ]
            if (
                overlapping_source_tokens
                and source_start <= overlapping_source_tokens[0].visible_start
                and overlapping_source_tokens[-1].visible_end <= source_end
            ):
                overlapping_target_tokens = [
                    token
                    for token in _projection_tokens(self.target)
                    if token.visible_start < target_end
                    and target_start < token.visible_end
                ]
                if overlapping_target_tokens:
                    target_start = overlapping_target_tokens[0].visible_start
                    target_end = overlapping_target_tokens[-1].visible_end
        if target_end <= target_start:
            return None
        return _CharacterIntervalMapping(
            target_visible_start=target_start,
            target_visible_end=target_end,
            alignment_digest=self.alignment_digest,
        )


@dataclass(frozen=True)
class _ProjectionClaim:
    """One positive style observation normalized onto an ordered donor unit."""

    claim_id: str
    original_occurrence_id: str
    span: NativeEmphasisSpan
    donor_visible_start: int | None
    donor_visible_end: int | None
    evidence_kind: Literal["native_document", "native_style", "source_markdown"]
    protected: bool
    placement_kind: str
    placement_digest: str
    source_unit_id: str
    style_line_digest: str | None = None
    source_unit_digest: str | None = None


@dataclass
class _ProjectionDonor:
    """One occurrence in the source-order anchor spine with overlaid claims."""

    occurrence: SkeletonOccurrence
    native_visible_text: str
    claims: list[_ProjectionClaim]
    order_key: tuple[int, int, str]
    native_text_only: bool = False

    @property
    def occurrence_id(self) -> str:
        return self.occurrence.occurrence_id

    @property
    def unit_type(self) -> str:
        return self.occurrence.unit_type

    @property
    def region(self) -> OccurrenceRegion:
        return self.occurrence.region


@dataclass(frozen=True)
class _ProjectionClaimInventory:
    """Protected and auxiliary claims kept in separate reconciled ledgers."""

    donors: Mapping[SourceName, tuple[_ProjectionDonor, ...]]
    unplaced: Mapping[SourceName, tuple[tuple[_ProjectionClaim, str], ...]]
    protected_claim_ids: tuple[str, ...]
    auxiliary_claim_ids: tuple[str, ...]


def _projection_tokens(
    visible: str,
    *,
    output_byte_start: int | None = None,
    target_occurrence_id: str | None = None,
) -> tuple[_ProjectionToken, ...]:
    tokens = []
    for match in _PROJECTION_TOKEN.finditer(visible):
        byte_start = (
            None
            if output_byte_start is None
            else output_byte_start + len(visible[: match.start()].encode("utf-8"))
        )
        byte_end = (
            None
            if output_byte_start is None
            else output_byte_start + len(visible[: match.end()].encode("utf-8"))
        )
        tokens.append(
            _ProjectionToken(
                key=unicodedata.normalize("NFC", match.group(0)),
                visible_start=match.start(),
                visible_end=match.end(),
                output_byte_start=byte_start,
                output_byte_end=byte_end,
                target_occurrence_id=target_occurrence_id,
            )
        )
    return tuple(tokens)


def _alignment_characters(value: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return a case-insensitive style-alignment view with source coordinates.

    Letters and numbers carry ordinary prose alignment. Unicode symbols also
    carry style evidence (for example, a styled genotype ``+/−``); punctuation
    and whitespace remain alignment-neutral and are recovered by the mapped
    interval envelope between aligned characters.
    """

    characters = []
    positions = []
    for position, character in enumerate(value):
        for normalized in unicodedata.normalize("NFKC", character).casefold():
            if normalized.isalnum() or unicodedata.category(normalized).startswith("S"):
                characters.append(normalized)
                positions.append(position)
    return tuple(characters), tuple(positions)


def _shared_character_alignment(
    source: str,
    target: str,
) -> _SharedCharacterAlignment | None:
    source_chars, source_positions = _alignment_characters(source)
    target_chars, target_positions = _alignment_characters(target)
    if not source_chars or not target_chars:
        return None
    opcodes = SequenceMatcher(
        None, source_chars, target_chars, autojunk=False
    ).get_opcodes()
    character_map: list[int | None] = [None] * len(source_chars)
    for tag, left_start, left_end, right_start, _right_end in opcodes:
        if tag != "equal":
            continue
        for offset in range(left_end - left_start):
            character_map[left_start + offset] = right_start + offset
    return _SharedCharacterAlignment(
        source=source,
        target=target,
        source_positions=source_positions,
        target_positions=target_positions,
        source_to_target=tuple(character_map),
        alignment_digest=hashlib.sha256(
            json.dumps(opcodes, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )


def _aggressive_character_interval(
    source: str,
    source_start: int,
    source_end: int,
    target: str,
) -> _CharacterIntervalMapping | None:
    """Map styled characters through one ordered text alignment.

    Punctuation, whitespace, entity spelling, and OCR insertions may differ. The
    mapped interval is still an existing target substring; this function never
    creates or rewrites publication characters.
    """

    alignment = _shared_character_alignment(source, target)
    return (
        None
        if alignment is None
        else alignment.interval(source_start, source_end)
    )


def _projection_source_visible(
    artifact: SourceArtifact,
    occurrence: SkeletonOccurrence,
) -> str | None:
    raw = artifact.raw_utf8[
        occurrence.source_byte_start : occurrence.source_byte_end
    ].decode("utf-8", errors="strict")
    analysis = _inline_italic_analysis(raw, occurrence.unit_type)
    return None if analysis is None else analysis[0].visible_text


def _projection_claim_id(
    source: SourceName,
    evidence_kind: str,
    original_occurrence_id: str,
    span: NativeEmphasisSpan,
) -> str:
    return _digest_parts(
        "positive-style-claim-v1",
        source,
        evidence_kind,
        original_occurrence_id,
        span.occurrence_id,
        str(span.visible_start),
        str(span.visible_end),
    )


def _standalone_projection_occurrence(
    source: SourceName,
    hint: NativeOccurrenceHint,
    *,
    evidence_kind: str,
) -> SkeletonOccurrence:
    """Represent an unmapped native block without pretending it came from Markdown."""

    occurrence_id = _digest_parts(
        "positive-style-donor-unit-v1",
        source,
        evidence_kind,
        hint.native_id,
        str(hint.native_order),
        hint.identity,
    )
    return SkeletonOccurrence(
        unit_id=f"native:{source}:{evidence_kind}:{hint.native_id}",
        occurrence_id=occurrence_id,
        unit_type=hint.unit_type,
        native_id=hint.native_id,
        native_order=hint.native_order,
        page_no=hint.page_no,
        region=hint.region,
        section_slot=None,
        slot_key=f"native:{hint.native_order}",
        source_byte_start=hint.native_order * 2,
        source_byte_end=hint.native_order * 2 + 1,
        native_emphasis_occurrence_ids=hint.native_emphasis_occurrence_ids,
        native_visible_text=hint.native_visible_text,
        native_emphasis_spans=hint.native_emphasis_spans,
        native_unit_type=hint.unit_type,
    )


def _projection_donor_spine(
    source: SourceName,
    artifact: SourceArtifact,
    skeleton: DocumentSkeleton,
) -> tuple[list[_ProjectionDonor], list[tuple[_ProjectionClaim, str]]]:
    """Build one positive-evidence donor sequence without source ranking.

    The protected ledger is exactly the historical native-document plus mapped
    native-style cohort.  Unmapped style-sidecar and Markdown-only observations
    use the same claim shape but remain auxiliary.
    """

    donors: list[_ProjectionDonor] = []
    by_unit: dict[str, _ProjectionDonor] = {}
    unplaced: list[tuple[_ProjectionClaim, str]] = []

    def add_claim(
        donor: _ProjectionDonor | None,
        *,
        original_occurrence_id: str,
        span: NativeEmphasisSpan,
        native_visible_text: str | None,
        evidence_kind: Literal[
            "native_document", "native_style", "source_markdown"
        ],
        protected: bool,
        source_unit_id: str,
        style_line_digest: str | None = None,
        source_unit_digest: str | None = None,
    ) -> None:
        mapping = None
        if donor is not None and native_visible_text:
            mapping = _aggressive_character_interval(
                native_visible_text,
                span.visible_start,
                span.visible_end,
                donor.native_visible_text,
            )
        donor_start = None if mapping is None else mapping.target_visible_start
        donor_end = None if mapping is None else mapping.target_visible_end
        placement_kind = (
            "native_text_donor_character_alignment"
            if mapping is not None and donor is not None and donor.native_text_only
            else "donor_unit_character_alignment"
            if mapping is not None
            else "unplaced"
        )
        placement_digest = _digest_parts(
            "positive-style-donor-placement-v1",
            source,
            evidence_kind,
            original_occurrence_id,
            source_unit_id,
            "" if donor is None else donor.occurrence.occurrence_id,
            "" if mapping is None else mapping.alignment_digest,
            "" if donor_start is None else str(donor_start),
            "" if donor_end is None else str(donor_end),
            span.occurrence_id,
        )
        claim = _ProjectionClaim(
            claim_id=_projection_claim_id(
                source, evidence_kind, original_occurrence_id, span
            ),
            original_occurrence_id=original_occurrence_id,
            span=span,
            donor_visible_start=donor_start,
            donor_visible_end=donor_end,
            evidence_kind=evidence_kind,
            protected=protected,
            placement_kind=placement_kind,
            placement_digest=placement_digest,
            source_unit_id=source_unit_id,
            style_line_digest=style_line_digest,
            source_unit_digest=source_unit_digest,
        )
        if mapping is None:
            unplaced.append(
                (
                    claim,
                    "donor_visible_text_unavailable"
                    if donor is None or not native_visible_text
                    else "character_interval_unmapped",
                )
            )
        else:
            donor.claims.append(claim)

    native_style_unit_ids = {
        style.source_unit_id
        for style in skeleton.native_style_occurrences
        if style.region == "body" and style.native_emphasis_spans
    }
    for source_ordinal, occurrence in enumerate(skeleton.occurrences):
        if effective_occurrence_region(
            occurrence.region, occurrence.unit_type
        ) != "body":
            continue
        visible = _projection_source_visible(artifact, occurrence)
        native_text_only = False
        if (
            not visible
            and occurrence.native_visible_text
            and (
                occurrence.native_emphasis_spans
                or occurrence.unit_id in native_style_unit_ids
            )
        ):
            visible = occurrence.native_visible_text
            native_text_only = True
        if not visible or not _projection_tokens(visible):
            continue
        donor = _ProjectionDonor(
            occurrence,
            visible,
            [],
            (
                occurrence.native_order
                if occurrence.native_order is not None
                else source_ordinal,
                1,
                occurrence.occurrence_id,
            ),
            native_text_only=native_text_only,
        )
        donors.append(donor)
        by_unit[occurrence.unit_id] = donor

    for occurrence in skeleton.occurrences:
        if (
            effective_occurrence_region(occurrence.region, occurrence.unit_type)
            != "body"
        ):
            continue
        donor = by_unit.get(occurrence.unit_id)
        for span in occurrence.native_emphasis_spans:
            add_claim(
                donor,
                original_occurrence_id=occurrence.occurrence_id,
                span=span,
                native_visible_text=occurrence.native_visible_text,
                evidence_kind="native_document",
                protected=True,
                source_unit_id=occurrence.unit_id,
            )

    for hint in skeleton.unmapped_native_occurrences:
        if hint.region != "body" or not hint.native_emphasis_spans:
            continue
        occurrence = _standalone_projection_occurrence(
            source, hint, evidence_kind="native_document"
        )
        visible = hint.native_visible_text or ""
        donor = (
            None
            if not visible or not _projection_tokens(visible)
            else _ProjectionDonor(
                occurrence,
                visible,
                [],
                (hint.native_order, 0, occurrence.occurrence_id),
            )
        )
        if donor is not None:
            donors.append(donor)
        for span in hint.native_emphasis_spans:
            add_claim(
                donor,
                original_occurrence_id=occurrence.occurrence_id,
                span=span,
                native_visible_text=visible,
                evidence_kind="native_document",
                protected=True,
                source_unit_id=occurrence.unit_id,
            )

    for style in skeleton.native_style_occurrences:
        if style.region != "body":
            continue
        donor = by_unit.get(style.source_unit_id)
        style_line_digest = hashlib.sha256(
            style.native_visible_text.encode("utf-8")
        ).hexdigest()
        source_unit_digest = (
            None
            if donor is None
            else hashlib.sha256(donor.native_visible_text.encode("utf-8")).hexdigest()
        )
        for span in style.native_emphasis_spans:
            add_claim(
                donor,
                original_occurrence_id=style.occurrence_id,
                span=span,
                native_visible_text=style.native_visible_text,
                evidence_kind="native_style",
                protected=True,
                source_unit_id=style.source_unit_id,
                style_line_digest=style_line_digest,
                source_unit_digest=source_unit_digest,
            )

    for hint in skeleton.unmapped_native_style_occurrences:
        if hint.region != "body" or not hint.native_emphasis_spans:
            continue
        occurrence = _standalone_projection_occurrence(
            source, hint, evidence_kind="native_style"
        )
        visible = hint.native_visible_text or ""
        donor = (
            None
            if not visible or not _projection_tokens(visible)
            else _ProjectionDonor(
                occurrence,
                visible,
                [],
                (hint.native_order, 2, occurrence.occurrence_id),
            )
        )
        if donor is not None:
            donors.append(donor)
        style_line_digest = (
            None
            if not visible
            else hashlib.sha256(visible.encode("utf-8")).hexdigest()
        )
        for span in hint.native_emphasis_spans:
            add_claim(
                donor,
                original_occurrence_id=occurrence.occurrence_id,
                span=span,
                native_visible_text=visible,
                evidence_kind="native_style",
                protected=False,
                source_unit_id=occurrence.unit_id,
                style_line_digest=style_line_digest,
            )

    for donor in tuple(donors):
        if donor.occurrence.unit_id not in by_unit:
            continue
        raw = artifact.raw_utf8[
            donor.occurrence.source_byte_start : donor.occurrence.source_byte_end
        ].decode("utf-8", errors="strict")
        profile = _inline_italic_profile(raw, donor.occurrence.unit_type)
        if profile is None:
            continue
        protected_intervals = {
            (claim.donor_visible_start, claim.donor_visible_end)
            for claim in donor.claims
            if claim.protected
        }
        for occurrence_id, (start, end) in zip(
            profile.emphasis_occurrence_ids, profile.emphasis_spans
        ):
            if (start, end) in protected_intervals:
                continue
            span = NativeEmphasisSpan(occurrence_id, start, end)
            add_claim(
                donor,
                original_occurrence_id=donor.occurrence.occurrence_id,
                span=span,
                native_visible_text=profile.visible_text,
                evidence_kind="source_markdown",
                protected=False,
                source_unit_id=donor.occurrence.unit_id,
            )

    donors.sort(key=lambda donor: donor.order_key)
    for donor in donors:
        donor.claims.sort(
            key=lambda claim: (
                -1 if claim.donor_visible_start is None else claim.donor_visible_start,
                -1 if claim.donor_visible_end is None else claim.donor_visible_end,
                not claim.protected,
                claim.original_occurrence_id,
                claim.span.occurrence_id,
            )
        )
    return donors, unplaced


def _projection_claim_inventory(
    skeletons: Mapping[SourceName, DocumentSkeleton],
    artifacts: Mapping[SourceName, SourceArtifact],
) -> _ProjectionClaimInventory:
    donors_by_source: dict[SourceName, tuple[_ProjectionDonor, ...]] = {}
    unplaced_by_source: dict[
        SourceName, tuple[tuple[_ProjectionClaim, str], ...]
    ] = {}
    protected_ids: list[str] = []
    auxiliary_ids: list[str] = []
    for source in sorted(skeletons):
        donors, unplaced = _projection_donor_spine(
            source, artifacts[source], skeletons[source]
        )
        donors_by_source[source] = tuple(donors)
        unplaced_by_source[source] = tuple(unplaced)
        claims = [
            *(claim for donor in donors for claim in donor.claims),
            *(claim for claim, _reason in unplaced),
        ]
        protected_ids.extend(claim.claim_id for claim in claims if claim.protected)
        auxiliary_ids.extend(claim.claim_id for claim in claims if not claim.protected)
    if len(protected_ids) != len(set(protected_ids)):
        raise ConsensusContractError("protected style claim identities are not unique")
    if len(auxiliary_ids) != len(set(auxiliary_ids)):
        raise ConsensusContractError("auxiliary style claim identities are not unique")
    expected_protected = sum(
        skeleton.native_body_emphasis_count for skeleton in skeletons.values()
    )
    if len(protected_ids) != expected_protected:
        raise ConsensusContractError(
            "protected style claim inventory does not match native body denominator"
        )
    return _ProjectionClaimInventory(
        donors=donors_by_source,
        unplaced=unplaced_by_source,
        protected_claim_ids=tuple(sorted(protected_ids)),
        auxiliary_claim_ids=tuple(sorted(auxiliary_ids)),
    )


def _final_projection_targets(text: str) -> list[_ProjectionTarget]:
    """Scan final Markdown directly and expose existing emphasis coordinates."""

    final_artifact = SourceArtifact.from_text("marker", text)
    final_skeleton = build_document_skeleton(final_artifact, None)
    targets: list[_ProjectionTarget] = []
    for ordinal, occurrence in enumerate(final_skeleton.occurrences):
        effective_region = effective_occurrence_region(
            occurrence.region, occurrence.unit_type
        )
        if effective_region != "body" and not (
            effective_region == "front" and occurrence.unit_type == "heading"
        ):
            continue
        raw = final_artifact.raw_utf8[
            occurrence.source_byte_start : occurrence.source_byte_end
        ]
        analysis = _inline_italic_analysis(
            raw.decode("utf-8", errors="strict"), occurrence.unit_type
        )
        if analysis is None:
            continue
        profile, _normalized_visible = analysis
        if not profile.visible_text or not _projection_tokens(profile.visible_text):
            continue
        content_offset = 0
        if occurrence.unit_type == "list":
            marker = _PROJECTION_LIST_MARKER.match(raw)
            if marker is None:
                continue
            content_offset = marker.end()
        content = raw[content_offset:].rstrip(b" \t\r\n")
        content_byte_start = occurrence.source_byte_start + content_offset
        content_byte_end = content_byte_start + len(content)
        plain_coordinates = (
            not profile.emphasis_occurrence_ids
            and content.decode("utf-8", errors="strict") == profile.visible_text
        )
        target_id = _digest_parts(
            "final-positive-style-target-v1",
            final_artifact.digest,
            str(ordinal),
            occurrence.unit_type,
            str(occurrence.source_byte_start),
            str(occurrence.source_byte_end),
        )
        final_occurrence = replace(
            occurrence,
            unit_id=target_id,
            occurrence_id=target_id,
        )
        targets.append(
            _ProjectionTarget(
                source="final",
                artifact_digest=final_artifact.digest,
                occurrence=final_occurrence,
                visible=profile.visible_text,
                tokens=_projection_tokens(
                    profile.visible_text,
                    output_byte_start=(
                        content_byte_start if plain_coordinates else None
                    ),
                    target_occurrence_id=target_id,
                ),
                output_byte_start=occurrence.source_byte_start,
                output_byte_end=occurrence.source_byte_end,
                ordinal=ordinal,
                content_byte_start=content_byte_start,
                content_byte_end=content_byte_end,
                visible_digest=profile.visible_digest,
                non_emphasis_ast_digest=profile.non_emphasis_ast_digest,
                existing_emphasis_occurrence_ids=(
                    profile.emphasis_occurrence_ids
                ),
                existing_emphasis_spans=profile.emphasis_spans,
            )
        )
    return targets


def _projection_unit_pairs(
    donors: list[_ProjectionDonor],
    targets: list[_ProjectionTarget],
    *,
    donor_source: SourceName,
    donor_artifact_digest: str,
    alignment_cache: dict[
        tuple[str, str], _SharedCharacterAlignment | None
    ] | None = None,
) -> dict[int, _ProjectionUnitPair]:
    """Pair units monotonically, then choose the best bounded positional match."""

    if not donors or not targets:
        return {}
    donor_units = tuple(
        StructuralUnitSpan(
            unit_id=donor.occurrence.unit_id,
            source=donor_source,
            artifact_digest=donor_artifact_digest,
            unit_type=donor.occurrence.unit_type,
            byte_start=donor.occurrence.source_byte_start,
            byte_end=donor.occurrence.source_byte_end,
            comparison_key=_comparison_key(donor.native_visible_text),
        )
        for donor in donors
    )
    target_units = tuple(
        StructuralUnitSpan(
            unit_id=target.occurrence.unit_id,
            source=target.source,
            artifact_digest=target.artifact_digest,
            unit_type=target.occurrence.unit_type,
            byte_start=target.output_byte_start,
            byte_end=target.output_byte_end,
            comparison_key=_comparison_key(target.visible),
        )
        for target in targets
    )
    pairs: dict[int, _ProjectionUnitPair] = {}
    donor_exact: dict[str, list[int]] = {}
    target_exact: dict[str, list[int]] = {}
    for index, unit in enumerate(donor_units):
        donor_exact.setdefault(unit.comparison_key, []).append(index)
    for index, unit in enumerate(target_units):
        target_exact.setdefault(unit.comparison_key, []).append(index)
    for key, donor_indexes in donor_exact.items():
        target_indexes = target_exact.get(key, ())
        if len(donor_indexes) == len(target_indexes) == 1:
            target = targets[target_indexes[0]]
            pairs[donor_indexes[0]] = _ProjectionUnitPair(
                target=target,
                similarity=100.0,
                ambiguous=False,
                candidates=((target, 100.0),),
            )

    aligned = _align_to_baseline(
        target_units,
        donor_units,
        require_matching_unit_types=False,
    )
    for target_index, match in aligned.items():
        if match.alternative_index in pairs:
            continue
        candidate_pairs = tuple(
            (targets[candidate_index], candidate_similarity)
            for candidate_index, candidate_similarity in (
                match.competing_baseline_indices
                or ((target_index, match.similarity),)
            )
        )
        pairs[match.alternative_index] = _ProjectionUnitPair(
            target=targets[target_index],
            similarity=match.similarity,
            ambiguous=match.ambiguous,
            candidates=candidate_pairs,
        )
    for donor_index, donor in enumerate(donors):
        if donor_index in pairs or not donor.claims:
            continue
        single = _align_to_baseline(
            target_units,
            (donor_units[donor_index],),
            require_matching_unit_types=False,
        )
        if single:
            target_index, match = next(iter(single.items()))
            candidate_pairs = tuple(
                (targets[candidate_index], candidate_similarity)
                for candidate_index, candidate_similarity in (
                    match.competing_baseline_indices
                    or ((target_index, match.similarity),)
                )
            )
            pairs[donor_index] = _ProjectionUnitPair(
                target=targets[target_index],
                similarity=match.similarity,
                ambiguous=match.ambiguous,
                candidates=candidate_pairs,
            )
    alignment_cache = {} if alignment_cache is None else alignment_cache

    def target_maps_claim(
        donor: _ProjectionDonor,
        target: _ProjectionTarget,
        *,
        protected_only: bool,
    ) -> bool:
        cache_key = (
            donor.occurrence.occurrence_id,
            target.occurrence.occurrence_id,
        )
        if cache_key not in alignment_cache:
            alignment_cache[cache_key] = _shared_character_alignment(
                donor.native_visible_text, target.visible
            )
        alignment = alignment_cache[cache_key]
        return alignment is not None and any(
            (claim.protected or not protected_only)
            and claim.donor_visible_start is not None
            and claim.donor_visible_end is not None
            and alignment.interval(
                claim.donor_visible_start,
                claim.donor_visible_end,
            )
            is not None
            for claim in donor.claims
        )

    # Structural alignment proposes publication-unit correspondences, but it is
    # not executable style evidence by itself.  A proposal that cannot map any
    # protected claim must continue through the same claim-mappable candidate
    # path as an unmatched donor instead of becoming a terminal character miss.
    for donor_index, pair in tuple(pairs.items()):
        donor = donors[donor_index]
        protected_only = any(claim.protected for claim in donor.claims)
        executable_targets = (
            (candidate for candidate, _score in pair.candidates)
            if pair.ambiguous
            else (pair.target,)
        )
        if not any(
            target_maps_claim(
                donor,
                target,
                protected_only=protected_only,
            )
            for target in executable_targets
        ):
            del pairs[donor_index]

    for donor_index, donor in enumerate(donors):
        if donor_index in pairs or not donor.claims:
            continue
        if not any(claim.protected for claim in donor.claims):
            # Below-floor/global candidate admission is intentionally protected-
            # only.  Scanning every final target for an auxiliary-only donor can
            # never admit a candidate and adds no behavior.
            continue
        donor_key = donor_units[donor_index].comparison_key
        if len(donor_key) > _MAX_COMPARISON_UNIT_CHARS:
            continue
        mappable = []
        for target, target_unit in zip(targets, target_units):
            if len(target_unit.comparison_key) > _MAX_COMPARISON_UNIT_CHARS:
                continue
            if not target_maps_claim(
                donor,
                target,
                protected_only=True,
            ):
                continue
            mappable.append((target, ratio(donor_key, target_unit.comparison_key)))
        if not mappable:
            continue
        best_target, best_similarity = min(
            mappable,
            key=lambda item: (-item[1], item[0].ordinal),
        )
        candidates = tuple(
            (target, similarity)
            for target, similarity in sorted(
                mappable, key=lambda item: item[0].ordinal
            )
            if similarity >= best_similarity - _ALIGNMENT_AMBIGUITY_DELTA
        )
        pairs[donor_index] = _ProjectionUnitPair(
            target=best_target,
            similarity=best_similarity,
            ambiguous=True,
            candidates=candidates,
            model_only=True,
        )
    return pairs


def _split_incompatible_projection_donors(
    donors: list[_ProjectionDonor],
    pairs: Mapping[int, _ProjectionUnitPair],
    alignment_cache: dict[
        tuple[str, str], _SharedCharacterAlignment | None
    ],
) -> tuple[
    list[_ProjectionDonor],
    dict[int, _ProjectionUnitPair],
    bool,
]:
    """Split only protected claim groups that require different final units.

    The complete donor text remains the alignment context.  Splitting changes
    only the unit-selection granularity when no offered final unit can map every
    protected claim in a multi-claim donor.
    """

    expanded: list[_ProjectionDonor] = []
    retained_pairs: dict[int, _ProjectionUnitPair] = {}
    changed = False

    def alignment_for(
        donor: _ProjectionDonor,
        target: _ProjectionTarget,
    ) -> _SharedCharacterAlignment | None:
        cache_key = (
            donor.occurrence.occurrence_id,
            target.occurrence.occurrence_id,
        )
        if cache_key not in alignment_cache:
            alignment_cache[cache_key] = _shared_character_alignment(
                donor.native_visible_text, target.visible
            )
        return alignment_cache[cache_key]

    def mapped_claims(
        donor: _ProjectionDonor,
        target: _ProjectionTarget,
        claims: list[_ProjectionClaim],
    ) -> list[_ProjectionClaim]:
        alignment = alignment_for(donor, target)
        if alignment is None:
            return []
        return [
            claim
            for claim in claims
            if claim.donor_visible_start is not None
            and claim.donor_visible_end is not None
            and alignment.interval(
                claim.donor_visible_start,
                claim.donor_visible_end,
            )
            is not None
        ]

    def append_donor(
        donor: _ProjectionDonor,
        claims: list[_ProjectionClaim],
    ) -> int:
        expanded.append(
            _ProjectionDonor(
                occurrence=donor.occurrence,
                native_visible_text=donor.native_visible_text,
                claims=claims,
                order_key=donor.order_key,
                native_text_only=donor.native_text_only,
            )
        )
        return len(expanded) - 1

    for donor_index, donor in enumerate(donors):
        pair = pairs.get(donor_index)
        protected = [claim for claim in donor.claims if claim.protected]
        if pair is None or len(protected) <= 1:
            expanded_index = append_donor(donor, list(donor.claims))
            if pair is not None:
                retained_pairs[expanded_index] = pair
            continue

        candidate_scores = {
            candidate.occurrence.occurrence_id: (candidate, score)
            for candidate, score in pair.candidates
        }
        candidate_scores.setdefault(
            pair.target.occurrence.occurrence_id,
            (pair.target, pair.similarity),
        )
        candidates = tuple(
            candidate_scores[candidate_id]
            for candidate_id in sorted(
                candidate_scores,
                key=lambda candidate_id: candidate_scores[candidate_id][0].ordinal,
            )
        )
        coverage = {
            candidate.occurrence.occurrence_id: mapped_claims(
                donor, candidate, protected
            )
            for candidate, _score in candidates
        }
        if any(len(claims) == len(protected) for claims in coverage.values()):
            expanded_index = append_donor(donor, list(donor.claims))
            retained_pairs[expanded_index] = pair
            continue

        changed = True
        auxiliary = [claim for claim in donor.claims if not claim.protected]
        primary_target = pair.target
        primary_claims = coverage.get(
            primary_target.occurrence.occurrence_id, []
        )
        if not primary_claims:
            primary_target, _primary_score = min(
                candidates,
                key=lambda item: (
                    -len(coverage[item[0].occurrence.occurrence_id]),
                    item[0].ordinal,
                ),
            )
            primary_claims = coverage[primary_target.occurrence.occurrence_id]

        primary_ids = {claim.claim_id for claim in primary_claims}
        primary_index = append_donor(
            donor,
            [
                claim
                for claim in donor.claims
                if claim.claim_id in primary_ids or not claim.protected
            ],
        )
        compatible_candidates = tuple(
            (candidate, score)
            for candidate, score in candidates
            if primary_ids
            <= {
                claim.claim_id
                for claim in coverage[candidate.occurrence.occurrence_id]
            }
        )
        primary_score = next(
            score
            for candidate, score in candidates
            if candidate.occurrence.occurrence_id
            == primary_target.occurrence.occurrence_id
        )
        retained_pairs[primary_index] = _ProjectionUnitPair(
            target=primary_target,
            similarity=primary_score,
            ambiguous=(pair.model_only or len(compatible_candidates) > 1),
            candidates=compatible_candidates,
            model_only=pair.model_only,
        )

        remaining = [
            claim for claim in protected if claim.claim_id not in primary_ids
        ]
        while remaining:
            candidate, _score = min(
                candidates,
                key=lambda item: (
                    -sum(
                        claim.claim_id
                        in {
                            mapped.claim_id
                            for mapped in coverage[
                                item[0].occurrence.occurrence_id
                            ]
                        }
                        for claim in remaining
                    ),
                    item[0].ordinal,
                ),
            )
            candidate_ids = {
                claim.claim_id
                for claim in coverage[candidate.occurrence.occurrence_id]
            }
            group = [
                claim for claim in remaining if claim.claim_id in candidate_ids
            ]
            if not group:
                group = [remaining[0]]
            append_donor(donor, group)
            grouped_ids = {claim.claim_id for claim in group}
            remaining = [
                claim for claim in remaining if claim.claim_id not in grouped_ids
            ]
    return expanded, retained_pairs, changed


@dataclass(frozen=True)
class _TargetSerializationMap:
    target: _ProjectionTarget
    raw_content: str
    token_map: tuple[int | None, ...]
    character_alignment: _SharedCharacterAlignment | None
    digest: str

    def output_interval(
        self, visible_start: int, visible_end: int
    ) -> tuple[int, int] | None:
        visible_tokens = _projection_tokens(self.target.visible)
        raw_tokens = _projection_tokens(self.raw_content)
        visible_indexes = [
            index
            for index, token in enumerate(visible_tokens)
            if token.visible_start < visible_end
            and visible_start < token.visible_end
        ]
        mapped = [
            self.token_map[index]
            for index in visible_indexes
            if self.token_map[index] is not None
        ]
        if (
            visible_indexes
            and len(mapped) == len(visible_indexes)
            and visible_start <= visible_tokens[visible_indexes[0]].visible_start
            and visible_tokens[visible_indexes[-1]].visible_end <= visible_end
            and all(right > left for left, right in zip(mapped, mapped[1:]))
        ):
            raw_start = raw_tokens[mapped[0]].visible_start
            raw_end = raw_tokens[mapped[-1]].visible_end
        else:
            character_interval = (
                None
                if self.character_alignment is None
                else self.character_alignment.interval(
                    visible_start,
                    visible_end,
                    expand_complete_tokens=False,
                )
            )
            if character_interval is None:
                return None
            raw_start = character_interval.target_visible_start
            raw_end = character_interval.target_visible_end
        return (
            self.target.content_byte_start
            + len(self.raw_content[:raw_start].encode("utf-8")),
            self.target.content_byte_start
            + len(self.raw_content[:raw_end].encode("utf-8")),
        )


def _target_serialization_map(
    text: str,
    target: _ProjectionTarget,
) -> _TargetSerializationMap | None:
    document = text.encode("utf-8")
    raw_content_bytes = document[
        target.content_byte_start : target.content_byte_end
    ]
    raw_content = raw_content_bytes.decode("utf-8", errors="strict")
    visible_tokens = _projection_tokens(target.visible)
    raw_tokens = _projection_tokens(raw_content)
    token_opcodes = SequenceMatcher(
        None,
        tuple(token.key for token in visible_tokens),
        tuple(token.key for token in raw_tokens),
        autojunk=False,
    ).get_opcodes()
    token_map: list[int | None] = [None] * len(visible_tokens)
    for tag, left_start, left_end, right_start, _right_end in token_opcodes:
        if tag != "equal":
            continue
        for offset in range(left_end - left_start):
            token_map[left_start + offset] = right_start + offset
    character_alignment = _shared_character_alignment(
        target.visible, raw_content
    )
    if not any(index is not None for index in token_map) and character_alignment is None:
        return None
    digest = hashlib.sha256(
        json.dumps(
            {
                "target_id": target.occurrence.occurrence_id,
                "token_opcodes": token_opcodes,
                "character_alignment_digest": (
                    None
                    if character_alignment is None
                    else character_alignment.alignment_digest
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return _TargetSerializationMap(
        target=target,
        raw_content=raw_content,
        token_map=tuple(token_map),
        character_alignment=character_alignment,
        digest=digest,
    )


def _positive_style_candidate_intervals(
    target: _ProjectionTarget,
    visible_start: int,
    visible_end: int,
) -> tuple[tuple[str, int, int], ...]:
    candidates = [("exact", visible_start, visible_end)]
    tokens = [
        token
        for token in target.tokens
        if token.visible_start < visible_end and visible_start < token.visible_end
    ]
    if tokens:
        candidates.append(
            ("token_envelope", tokens[0].visible_start, tokens[-1].visible_end)
        )
    unique = []
    seen = set()
    for candidate in candidates:
        boundary_kind, candidate_start, candidate_end = candidate
        table_label = _TABLE_CAPTION_VISIBLE_LABEL.match(target.visible)
        if table_label is not None and candidate_start < table_label.end():
            caption_body_start = table_label.end()
            while (
                caption_body_start < len(target.visible)
                and target.visible[caption_body_start].isspace()
            ):
                caption_body_start += 1
            if candidate_end <= caption_body_start:
                continue
            candidate = (
                f"{boundary_kind}_table_caption_body",
                caption_body_start,
                candidate_end,
            )
        coordinates = candidate[1:]
        if coordinates not in seen:
            seen.add(coordinates)
            unique.append(candidate)
    return tuple(unique)


def _rendered_unit_profile(
    original_unit: bytes,
    unit_output_start: int,
    intervals: list[tuple[int, int, str]],
    unit_type: str,
) -> _InlineItalicProfile | None:
    rendered = bytearray()
    cursor = unit_output_start
    for start, end, _occurrence_id in sorted(intervals):
        if start < cursor or end <= start:
            return None
        rendered.extend(
            original_unit[cursor - unit_output_start : start - unit_output_start]
        )
        rendered.extend(b"*")
        rendered.extend(
            original_unit[start - unit_output_start : end - unit_output_start]
        )
        rendered.extend(b"*")
        cursor = end
    rendered.extend(original_unit[cursor - unit_output_start :])
    return _inline_italic_profile(
        rendered.decode("utf-8", errors="strict"), unit_type
    )


def _bounded_style_selection_context(
    value: str,
    start: int | None,
    end: int | None,
    *,
    radius: int = 600,
) -> str:
    """Return bounded review context; matching continues to use the complete unit."""

    if start is None or end is None or not 0 <= start < end <= len(value):
        return value if len(value) <= 2 * radius else value[:radius] + " … " + value[-radius:]
    left = max(0, start - radius)
    right = min(len(value), end + radius)
    return value[left:start] + "<<<" + value[start:end] + ">>>" + value[end:right]


def _compile_positive_style_overlay(
    text: str,
    audit: list[dict],
    skeletons: Mapping[SourceName, DocumentSkeleton],
    artifacts: Mapping[SourceName, SourceArtifact],
    *,
    style_selection_resolver: Callable[..., object] | None = None,
    recorded_style_selections: Mapping[str, Mapping] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """Map all positive claims to frozen final units and emit one style overlay."""

    inventory = _projection_claim_inventory(skeletons, artifacts)
    targets = _final_projection_targets(text)
    events: list[dict] = []
    mapped_by_target: dict[str, list[dict]] = {}
    recorded_style_selections = recorded_style_selections or {}
    alignment_cache: dict[
        tuple[str, str], _SharedCharacterAlignment | None
    ] = {}

    def alignment_for(
        donor: _ProjectionDonor,
        target: _ProjectionTarget,
    ) -> _SharedCharacterAlignment | None:
        key = (
            donor.occurrence.occurrence_id,
            target.occurrence.occurrence_id,
        )
        if key not in alignment_cache:
            alignment_cache[key] = _shared_character_alignment(
                donor.native_visible_text, target.visible
            )
        return alignment_cache[key]

    def claim_event(
        source: SourceName,
        claim: _ProjectionClaim,
        *,
        outcome: str,
        reason: str | None = None,
        **details: object,
    ) -> dict:
        event = {
            "operation": "native_emphasis_projection",
            "audit_span_emitted": False,
            "phase": "result" if outcome != "eligible" else "eligibility",
            "outcome": outcome,
            "donor_source": source,
            "donor_occurrence_id": claim.original_occurrence_id,
            "native_emphasis_occurrence_id": claim.span.occurrence_id,
            "native_evidence_kind": claim.evidence_kind,
            "protected_claim": claim.protected,
            "positive_style_claim_id": claim.claim_id,
            "direct_native_donor": (
                claim.source_unit_id.startswith("native:")
                or claim.placement_kind
                == "native_text_donor_character_alignment"
            ),
        }
        if reason is not None:
            event["reason"] = reason
        event.update(details)
        return event

    for source in sorted(inventory.donors):
        donors = []
        for claim, reason in inventory.unplaced[source]:
            events.append(claim_event(source, claim, outcome="eligible"))
            events.append(claim_event(source, claim, outcome="declined", reason=reason))
        for donor in inventory.donors[source]:
            if donor.occurrence.native_unit_type == "page_header":
                for claim in donor.claims:
                    events.append(claim_event(source, claim, outcome="eligible"))
                    events.append(
                        claim_event(
                            source,
                            claim,
                            outcome="declined",
                            reason="native_page_furniture",
                        )
                    )
                continue
            donors.append(donor)
        pairs = _projection_unit_pairs(
            donors,
            targets,
            donor_source=source,
            donor_artifact_digest=artifacts[source].digest,
            alignment_cache=alignment_cache,
        )
        donors, retained_pairs, donors_split = _split_incompatible_projection_donors(
            donors, pairs, alignment_cache
        )
        if donors_split:
            rerouted_indexes = [
                index for index in range(len(donors)) if index not in retained_pairs
            ]
            rerouted_pairs = _projection_unit_pairs(
                [donors[index] for index in rerouted_indexes],
                targets,
                donor_source=source,
                donor_artifact_digest=artifacts[source].digest,
                alignment_cache=alignment_cache,
            )
            pairs = dict(retained_pairs)
            pairs.update(
                {
                    rerouted_indexes[rerouted_index]: pair
                    for rerouted_index, pair in rerouted_pairs.items()
                }
            )
        accepted_ordinals: list[tuple[int, int, bool]] = []
        for donor_index, donor in enumerate(donors):
            if not donor.claims:
                continue
            pair = pairs.get(donor_index)
            if pair is None:
                for claim in donor.claims:
                    events.append(claim_event(source, claim, outcome="eligible"))
                    events.append(
                        claim_event(
                            source,
                            claim,
                            outcome="declined",
                            reason="final_unit_pair_unavailable",
                        )
                    )
                continue

            target = pair.target
            similarity = pair.similarity
            selection_details: dict[str, object] = {
                "unit_pair_ambiguous": pair.ambiguous,
                "style_selection_below_deterministic_floor": pair.model_only,
            }
            if pair.ambiguous:
                candidates = tuple(
                    sorted(
                        {candidate.occurrence.occurrence_id: (candidate, score)
                         for candidate, score in pair.candidates
                         if (
                             (
                                 candidate_alignment := alignment_for(
                                     donor, candidate
                                 )
                             )
                             is not None
                             and any(
                                 claim.donor_visible_start is not None
                                 and claim.donor_visible_end is not None
                                 and candidate_alignment.interval(
                                     claim.donor_visible_start,
                                     claim.donor_visible_end,
                                 )
                                 is not None
                                 for claim in donor.claims
                             )
                         )}.values(),
                        key=lambda item: item[0].ordinal,
                    )
                )
                if not candidates:
                    for claim in donor.claims:
                        events.append(claim_event(source, claim, outcome="eligible"))
                        events.append(
                            claim_event(
                                source,
                                claim,
                                outcome="declined",
                                reason="character_interval_unmapped",
                                unit_pair_ambiguous=True,
                            )
                        )
                    continue
                none_id = _digest_parts(
                    "positive-style-no-unit-selection-v1",
                    source,
                    donor.occurrence.occurrence_id,
                    *(claim.claim_id for claim in donor.claims),
                )
                selection_id = _digest_parts(
                    "positive-style-unit-selection-v1",
                    source,
                    donor.occurrence.occurrence_id,
                    *(claim.claim_id for claim in donor.claims),
                    *(candidate.occurrence.occurrence_id for candidate, _score in candidates),
                )
                candidate_ids = [
                    candidate.occurrence.occurrence_id
                    for candidate, _score in candidates
                ]
                recorded = recorded_style_selections.get(selection_id)
                selected_id = None
                decline_reason = "final_unit_pair_ambiguous"
                if recorded is not None:
                    selected_id = recorded.get("style_selected_candidate_id")
                    decline_reason = str(
                        recorded.get("reason", "final_unit_pair_ambiguous")
                    )
                    selection_details = {
                        key: value
                        for key, value in recorded.items()
                        if key != "reason"
                    }
                elif (
                    style_selection_resolver is not None
                    and len(candidates) <= MAX_BOUNDED_TARGET_CHOICES
                ):
                    donor_start = min(
                        claim.donor_visible_start
                        for claim in donor.claims
                        if claim.donor_visible_start is not None
                    )
                    donor_end = max(
                        claim.donor_visible_end
                        for claim in donor.claims
                        if claim.donor_visible_end is not None
                    )
                    choices = [{
                        "candidate_id": none_id,
                        "candidate_type": "positive_style_no_matching_final_unit",
                        "display": (
                            "Choose 0 when none of the final units is the same "
                            "publication unit as this italic-bearing donor.\nDONOR:\n"
                            + _bounded_style_selection_context(
                                donor.native_visible_text, donor_start, donor_end
                            )
                        ),
                    }]
                    for candidate, score in candidates:
                        alignment = alignment_for(donor, candidate)
                        mapped = (
                            None
                            if alignment is None
                            else alignment.interval(donor_start, donor_end)
                        )
                        choices.append({
                            "candidate_id": candidate.occurrence.occurrence_id,
                            "candidate_type": "positive_style_final_unit",
                            "similarity_milli": round(score * 1_000),
                            "display": (
                                "Choose this numbered final unit only if it is the "
                                "same publication unit as the donor.\nDONOR:\n"
                                + _bounded_style_selection_context(
                                    donor.native_visible_text, donor_start, donor_end
                                )
                                + "\nFINAL UNIT:\n"
                                + _bounded_style_selection_context(
                                    candidate.visible,
                                    None if mapped is None else mapped.target_visible_start,
                                    None if mapped is None else mapped.target_visible_end,
                                )
                            ),
                        })
                    receipt_details: dict[str, object] = {}
                    try:
                        selection_result = style_selection_resolver(
                            reason=selection_id,
                            baseline_id=none_id,
                            choices=choices,
                        )
                        if isinstance(selection_result, Mapping):
                            selected_id = selection_result.get(
                                "selected_candidate_id"
                            )
                            required_receipt = {
                                "style_selection_request_sha256": selection_result.get(
                                    "request_sha256"
                                ),
                                "style_selection_response_choice": selection_result.get(
                                    "response_choice"
                                ),
                                "style_selection_model": selection_result.get("model"),
                                "style_selection_reasoning_effort": selection_result.get(
                                    "reasoning_effort"
                                ),
                            }
                            if (
                                not isinstance(selected_id, str)
                                or not isinstance(
                                    required_receipt["style_selection_request_sha256"],
                                    str,
                                )
                                or type(
                                    required_receipt["style_selection_response_choice"]
                                ) is not int
                                or not isinstance(
                                    required_receipt["style_selection_model"], str
                                )
                                or not isinstance(
                                    required_receipt[
                                        "style_selection_reasoning_effort"
                                    ],
                                    str,
                                )
                            ):
                                raise ValueError("style selection receipt is incomplete")
                            receipt_details = required_receipt
                        else:
                            selected_id = selection_result
                        decline_reason = (
                            "model_selection_none"
                            if selected_id == none_id
                            else "final_unit_pair_ambiguous"
                        )
                    except Exception:
                        selected_id = none_id
                        decline_reason = "model_selection_unavailable"
                    selection_details = {
                        "unit_pair_ambiguous": True,
                        "style_selection_below_deterministic_floor": pair.model_only,
                        "style_selection_id": selection_id,
                        "style_selection_candidate_ids": candidate_ids,
                        "style_selection_candidate_count": len(candidate_ids),
                        "style_selection_none_id": none_id,
                        "style_selected_candidate_id": selected_id,
                        "style_selection_method": "sol_numbered_choice",
                        "model_selected_target": selected_id in candidate_ids,
                        **receipt_details,
                    }
                else:
                    selected_id = none_id
                    selection_details = {
                        "unit_pair_ambiguous": True,
                        "style_selection_below_deterministic_floor": pair.model_only,
                        "style_selection_id": selection_id,
                        "style_selection_candidate_ids": candidate_ids,
                        "style_selection_candidate_count": len(candidate_ids),
                        "style_selection_none_id": none_id,
                        "style_selected_candidate_id": selected_id,
                        "style_selection_method": "not_run",
                        "model_selected_target": False,
                    }
                selected = next(
                    (
                        (candidate, score)
                        for candidate, score in candidates
                        if candidate.occurrence.occurrence_id == selected_id
                    ),
                    None,
                )
                if selected is None and selected_id != none_id:
                    decline_reason = "model_selection_unavailable"
                if selected is None:
                    for claim in donor.claims:
                        events.append(claim_event(source, claim, outcome="eligible"))
                        events.append(
                            claim_event(
                                source,
                                claim,
                                outcome="declined",
                                reason=decline_reason,
                                **selection_details,
                            )
                        )
                    continue
                target, similarity = selected

            accepted_ordinals.append((donor_index, target.ordinal, pair.ambiguous))
            alignment = alignment_for(donor, target)
            pair_digest = _digest_parts(
                "positive-style-unit-pair-v1",
                source,
                donor.occurrence.occurrence_id,
                target.occurrence.occurrence_id,
                str(round(similarity * 1_000)),
            )
            for claim in donor.claims:
                events.append(claim_event(source, claim, outcome="eligible"))
                if (
                    alignment is None
                    or claim.donor_visible_start is None
                    or claim.donor_visible_end is None
                ):
                    events.append(
                        claim_event(
                            source,
                            claim,
                            outcome="declined",
                            reason="character_interval_unmapped",
                            **selection_details,
                        )
                    )
                    continue
                mapped = alignment.interval(
                    claim.donor_visible_start,
                    claim.donor_visible_end,
                )
                if mapped is None:
                    events.append(
                        claim_event(
                            source,
                            claim,
                            outcome="declined",
                            reason="character_interval_unmapped",
                            **selection_details,
                        )
                    )
                    continue
                mapped_by_target.setdefault(
                    target.occurrence.occurrence_id, []
                ).append(
                    {
                        "source": source,
                        "donor_index": donor_index,
                        "donor": donor,
                        "claim": claim,
                        "target": target,
                        "donor_unit_type": donor.occurrence.unit_type,
                        "target_unit_type": target.occurrence.unit_type,
                        "visible_start": mapped.target_visible_start,
                        "visible_end": mapped.target_visible_end,
                        "unit_similarity_milli": round(similarity * 1_000),
                        "unit_pair_digest": pair_digest,
                        "character_alignment_digest": alignment.alignment_digest,
                        **selection_details,
                    }
                )

        non_monotone_ambiguous = {
            donor_index
            for position, (donor_index, target_ordinal, ambiguous) in enumerate(
                accepted_ordinals
            )
            if ambiguous
            and (
                (position > 0 and accepted_ordinals[position - 1][1] > target_ordinal)
                or (
                    position + 1 < len(accepted_ordinals)
                    and target_ordinal > accepted_ordinals[position + 1][1]
                )
            )
        }
        for records in mapped_by_target.values():
            for record in records:
                if (
                    record["source"] != source
                    or "style_selection_id" not in record
                ):
                    continue
                record.update(
                    {
                        "style_selection_donor_ordinal": record["donor_index"],
                        "style_selection_target_ordinal": record["target"].ordinal,
                        "style_selection_order_crossing": (
                            record["donor_index"] in non_monotone_ambiguous
                        ),
                    }
                )

    existing_supports: list[dict] = []
    canonical_groups: list[dict] = []

    def unresolved_interval_details(
        target: _ProjectionTarget,
        visible_start: int,
        visible_end: int,
    ) -> dict:
        return {
            "target_occurrence_id": target.occurrence.occurrence_id,
            "target_visible_start": visible_start,
            "target_visible_end": visible_end,
            "unresolved_output_interval_id": _digest_parts(
                "positive-style-unresolved-interval-v1",
                target.occurrence.occurrence_id,
                str(visible_start),
                str(visible_end),
            ),
        }

    def unit_pair_event_details(record: Mapping) -> dict:
        return {
            key: value
            for key, value in record.items()
            if key in {
                "unit_pair_ambiguous",
                "donor_unit_type",
                "target_unit_type",
            }
            or key.startswith("style_selection_")
            or key == "style_selected_candidate_id"
            or key == "model_selected_target"
        }

    for target_id, records in sorted(mapped_by_target.items()):
        target = records[0]["target"]
        pending = []
        for record in sorted(
            records,
            key=lambda item: (
                item["visible_start"],
                item["visible_end"],
                item["claim"].claim_id,
            ),
        ):
            containing = next(
                (
                    span
                    for span in target.existing_emphasis_spans
                    if span[0] <= record["visible_start"]
                    and record["visible_end"] <= span[1]
                ),
                None,
            )
            if containing is not None:
                existing_supports.append({**record, "canonical_visible": containing})
            else:
                pending.append(record)
        components: list[list[dict]] = []
        for record in pending:
            if not components:
                components.append([record])
                continue
            previous_end = max(item["visible_end"] for item in components[-1])
            gap = target.visible[previous_end : record["visible_start"]]
            if record["visible_start"] <= previous_end or not any(
                character.isalnum() for character in gap
            ):
                components[-1].append(record)
            else:
                components.append([record])
        for members in components:
            visible_start = min(record["visible_start"] for record in members)
            visible_end = max(record["visible_end"] for record in members)
            if any(
                visible_start < existing_end and existing_start < visible_end
                for existing_start, existing_end in target.existing_emphasis_spans
            ):
                for record in members:
                    events.append(
                        claim_event(
                            record["source"],
                            record["claim"],
                            outcome="declined",
                            reason="existing_markdown_overlap_unrepresentable",
                            **unresolved_interval_details(
                                target, visible_start, visible_end
                            ),
                            **unit_pair_event_details(record),
                        )
                    )
                continue
            canonical_groups.append(
                {
                    "target": target,
                    "members": members,
                    "visible_start": visible_start,
                    "visible_end": visible_end,
                }
            )

    for support_record in existing_supports:
        target = support_record["target"]
        visible_start, visible_end = support_record["canonical_visible"]
        events.append(
            claim_event(
                support_record["source"],
                support_record["claim"],
                outcome="supported",
                reason="existing_final_emphasis",
                support_reconciled=True,
                canonical_interval_id=_digest_parts(
                    "positive-style-existing-interval-v1",
                    target.occurrence.occurrence_id,
                    str(visible_start),
                    str(visible_end),
                ),
                target_occurrence_id=target.occurrence.occurrence_id,
                target_visible_start=visible_start,
                target_visible_end=visible_end,
                target_emphasis_occurrence_id=_emphasis_occurrence_id(
                    target.visible, visible_start, visible_end
                ),
                existing_final_emphasis=True,
                **unit_pair_event_details(support_record),
            )
        )

    original = text.encode("utf-8")
    selected_records: list[dict] = []
    groups_by_target: dict[str, list[dict]] = {}
    for group in canonical_groups:
        groups_by_target.setdefault(
            group["target"].occurrence.occurrence_id, []
        ).append(group)
    for target_id, groups in sorted(groups_by_target.items()):
        target = groups[0]["target"]
        serialization = _target_serialization_map(text, target)
        original_unit = original[target.output_byte_start : target.output_byte_end]
        original_profile = _inline_italic_profile(
            original_unit.decode("utf-8", errors="strict"),
            target.occurrence.unit_type,
        )
        if serialization is None or original_profile is None:
            for group in groups:
                for record in group["members"]:
                    events.append(
                        claim_event(
                            record["source"],
                            record["claim"],
                            outcome="declined",
                            reason="markdown_emphasis_not_realized",
                            **unresolved_interval_details(
                                target,
                                group["visible_start"],
                                group["visible_end"],
                            ),
                            **unit_pair_event_details(record),
                        )
                    )
            continue

        chosen = []
        for group in groups:
            selected_candidate = None
            table_label = _TABLE_CAPTION_VISIBLE_LABEL.match(target.visible)
            table_label_only = (
                table_label is not None
                and group["visible_start"] < table_label.end()
                and group["visible_end"] <= len(
                    target.visible[: table_label.end()].rstrip()
                )
            )
            for boundary_kind, visible_start, visible_end in (
                _positive_style_candidate_intervals(
                    target, group["visible_start"], group["visible_end"]
                )
            ):
                output_interval = serialization.output_interval(
                    visible_start, visible_end
                )
                if output_interval is None:
                    continue
                emphasis_id = _emphasis_occurrence_id(
                    target.visible, visible_start, visible_end
                )
                transformed = _rendered_unit_profile(
                    original_unit,
                    target.output_byte_start,
                    [(output_interval[0], output_interval[1], emphasis_id)],
                    target.occurrence.unit_type,
                )
                if (
                    transformed is not None
                    and transformed.visible_digest == original_profile.visible_digest
                    and transformed.non_emphasis_ast_digest
                    == original_profile.non_emphasis_ast_digest
                    and Counter(transformed.emphasis_occurrence_ids)
                    == Counter(
                        (*original_profile.emphasis_occurrence_ids, emphasis_id)
                    )
                ):
                    selected_candidate = {
                        **group,
                        "boundary_kind": boundary_kind,
                        "visible_start": visible_start,
                        "visible_end": visible_end,
                        "output_start": output_interval[0],
                        "output_end": output_interval[1],
                        "emphasis_id": emphasis_id,
                        "serialization_digest": serialization.digest,
                    }
                    break
            if selected_candidate is None:
                for record in group["members"]:
                    events.append(
                        claim_event(
                            record["source"],
                            record["claim"],
                            outcome="declined",
                            reason=(
                                "alliance_table_label_is_structural"
                                if table_label_only
                                else "markdown_emphasis_not_realized"
                            ),
                            **unresolved_interval_details(
                                target,
                                group["visible_start"],
                                group["visible_end"],
                            ),
                            **unit_pair_event_details(record),
                        )
                    )
            else:
                chosen.append(selected_candidate)

        def combined_is_valid(records: list[dict]) -> bool:
            transformed = _rendered_unit_profile(
                original_unit,
                target.output_byte_start,
                [
                    (
                        record["output_start"],
                        record["output_end"],
                        record["emphasis_id"],
                    )
                    for record in records
                ],
                target.occurrence.unit_type,
            )
            return (
                transformed is not None
                and transformed.visible_digest == original_profile.visible_digest
                and transformed.non_emphasis_ast_digest
                == original_profile.non_emphasis_ast_digest
                and Counter(transformed.emphasis_occurrence_ids)
                == Counter(
                    (
                        *original_profile.emphasis_occurrence_ids,
                        *(record["emphasis_id"] for record in records),
                    )
                )
            )

        if not combined_is_valid(chosen):
            compatible = []
            for candidate in sorted(
                chosen,
                key=lambda record: (
                    -(record["visible_end"] - record["visible_start"]),
                    record["visible_start"],
                    record["emphasis_id"],
                ),
            ):
                if combined_is_valid([*compatible, candidate]):
                    compatible.append(candidate)
                else:
                    for member in candidate["members"]:
                        events.append(
                            claim_event(
                                member["source"],
                                member["claim"],
                                outcome="declined",
                                reason="markdown_emphasis_not_realized",
                                **unresolved_interval_details(
                                    target,
                                    candidate["visible_start"],
                                    candidate["visible_end"],
                                ),
                                **unit_pair_event_details(member),
                            )
                        )
            chosen = compatible
        selected_records.extend(chosen)

    selected_records.sort(
        key=lambda record: (
            record["output_start"],
            record["output_end"],
            record["emphasis_id"],
        )
    )
    if any(
        left["output_end"] >= right["output_start"]
        for left, right in zip(selected_records, selected_records[1:])
    ):
        for record in selected_records:
            for member in record["members"]:
                events.append(
                    claim_event(
                        member["source"],
                        member["claim"],
                        outcome="declined",
                        reason="markdown_emphasis_not_realized",
                        **unresolved_interval_details(
                            record["target"],
                            record["visible_start"],
                            record["visible_end"],
                        ),
                        **unit_pair_event_details(member),
                    )
                )
        selected_records = []

    output = bytearray()
    cursor = 0
    for record in selected_records:
        output.extend(original[cursor : record["output_start"]])
        output.extend(b"*")
        output.extend(original[record["output_start"] : record["output_end"]])
        output.extend(b"*")
        cursor = record["output_end"]
    output.extend(original[cursor:])
    rendered = output.decode("utf-8", errors="strict")
    final_report = abc_markdown_report(rendered)
    if final_report["error_rule_ids"]:
        for record in selected_records:
            for member in record["members"]:
                events.append(
                    claim_event(
                        member["source"],
                        member["claim"],
                        outcome="declined",
                        reason="markdown_emphasis_not_realized",
                        **unresolved_interval_details(
                            record["target"],
                            record["visible_start"],
                            record["visible_end"],
                        ),
                        **unit_pair_event_details(member),
                    )
                )
        return text, audit, events
    if not selected_records:
        return text, audit, events

    rewritten_audit: list[dict] = []
    final_output = bytearray()
    cursor = 0
    projected_events = []
    pre_digest = hashlib.sha256(original).hexdigest()
    for record in selected_records:
        start = record["output_start"]
        end = record["output_end"]
        _copy_audit_interval(final_output, rewritten_audit, original, audit, cursor, start)
        representative = min(
            record["members"], key=lambda member: member["claim"].claim_id
        )
        supporting_claim_ids = sorted(
            member["claim"].claim_id for member in record["members"]
        )
        supporting_sources = sorted(
            {member["source"] for member in record["members"]}
        )
        projection_id = _digest_parts(
            "positive-style-overlay-projection-v1",
            pre_digest,
            record["target"].occurrence.occurrence_id,
            str(record["visible_start"]),
            str(record["visible_end"]),
            *supporting_claim_ids,
        )
        common = {
            "projection_id": projection_id,
            "overlay_method": "post-merge-positive-style-overlay-v1",
            "reconciliation_method": "positive-interval-union-v1",
            "donor_source": representative["source"],
            "donor_artifact_digest": artifacts[representative["source"]].digest,
            "donor_occurrence_id": representative["claim"].original_occurrence_id,
            "donor_spine_occurrence_id": representative["donor"].occurrence.occurrence_id,
            "donor_unit_type": representative["donor"].occurrence.unit_type,
            "donor_spine_source_unit_id": representative["claim"].source_unit_id,
            "donor_spine_visible_start": representative["claim"].donor_visible_start,
            "donor_spine_visible_end": representative["claim"].donor_visible_end,
            "donor_spine_placement_digest": representative["claim"].placement_digest,
            "native_evidence_kind": representative["claim"].evidence_kind,
            "protected_claim": representative["claim"].protected,
            "positive_style_claim_id": representative["claim"].claim_id,
            "direct_native_donor": (
                representative["claim"].source_unit_id.startswith("native:")
                or representative["claim"].placement_kind
                == "native_text_donor_character_alignment"
            ),
            "native_visible_start": representative["claim"].span.visible_start,
            "native_visible_end": representative["claim"].span.visible_end,
            "native_emphasis_occurrence_id": representative["claim"].span.occurrence_id,
            "target_source": "final",
            "target_artifact_digest": record["target"].artifact_digest,
            "target_occurrence_id": record["target"].occurrence.occurrence_id,
            "target_unit_ordinal": record["target"].ordinal,
            "target_unit_type": record["target"].occurrence.unit_type,
            "target_unit_output_byte_start": record["target"].output_byte_start,
            "target_unit_output_byte_end": record["target"].output_byte_end,
            "target_visible_digest": record["target"].visible_digest,
            "target_non_emphasis_ast_digest": record["target"].non_emphasis_ast_digest,
            "target_visible_start": record["visible_start"],
            "target_visible_end": record["visible_end"],
            "target_pre_projection_output_byte_start": start,
            "target_pre_projection_output_byte_end": end,
            "target_emphasis_occurrence_id": record["emphasis_id"],
            "unit_pair_digest": representative["unit_pair_digest"],
            "alignment_method": "final-unit-shared-character-alignment-v1",
            "mapping_kind": "final_unit_alignment",
            "unit_alignment_similarity_milli": representative["unit_similarity_milli"],
            "character_alignment_digest": representative["character_alignment_digest"],
            "target_serialization_digest": record["serialization_digest"],
            "boundary_candidate": record["boundary_kind"],
            "existing_final_emphasis": False,
            "supporting_sources": supporting_sources,
            "supporting_claim_ids": supporting_claim_ids,
            "support_claim_count": len(supporting_claim_ids),
            **unit_pair_event_details(representative),
        }
        skeleton = skeletons[representative["source"]]
        if skeleton.native_artifact_digest is not None:
            common["native_artifact_digest"] = skeleton.native_artifact_digest
        if skeleton.native_receipt_digest is not None:
            common["native_receipt_digest"] = skeleton.native_receipt_digest
        if skeleton.native_style_digest is not None:
            common["native_style_digest"] = skeleton.native_style_digest
        delimiter_records = []
        for position, boundary in ((start, "open"), (end, "close")):
            if boundary == "close":
                _copy_audit_interval(
                    final_output, rewritten_audit, original, audit, start, end
                )
            marker_start = len(final_output)
            final_output.extend(b"*")
            transformation_id = _digest_parts(
                "positive-style-overlay-delimiter-v1",
                projection_id,
                boundary,
                str(position),
            )
            rewritten_audit.append(
                {
                    "output_byte_start": marker_start,
                    "output_byte_end": marker_start + 1,
                    "source": "deterministic_markup",
                    "artifact_digest": hashlib.sha256(b"*").hexdigest(),
                    "source_byte_start": 0,
                    "source_byte_end": 1,
                    "candidate_id": None,
                    "region_id": None,
                    "decision_method": "deterministic",
                    "transformation": "native_emphasis_projection",
                    "transformation_id": transformation_id,
                }
            )
            delimiter_records.append((boundary, transformation_id, marker_start))
        target_output_start = delimiter_records[0][2] + 1
        target_output_end = delimiter_records[1][2]
        for boundary, transformation_id, marker_start in delimiter_records:
            projected_events.append(
                {
                    "operation": "native_emphasis_projection",
                    "audit_span_emitted": True,
                    "phase": "result",
                    "outcome": "projected",
                    "projection_reconciled": True,
                    "boundary": boundary,
                    "transformation_id": transformation_id,
                    "delimiter_output_byte_start": marker_start,
                    "delimiter_output_byte_end": marker_start + 1,
                    "target_output_byte_start": target_output_start,
                    "target_output_byte_end": target_output_end,
                    **common,
                }
            )
        for member in record["members"]:
            if member is representative:
                continue
            events.append(
                claim_event(
                    member["source"],
                    member["claim"],
                    outcome="supported",
                    reason="canonical_interval_supported",
                    support_reconciled=True,
                    supported_projection_id=projection_id,
                    canonical_interval_id=projection_id,
                    target_occurrence_id=record["target"].occurrence.occurrence_id,
                    target_visible_start=record["visible_start"],
                    target_visible_end=record["visible_end"],
                    target_emphasis_occurrence_id=record["emphasis_id"],
                    existing_final_emphasis=False,
                    supporting_sources=supporting_sources,
                    supporting_claim_ids=supporting_claim_ids,
                    support_claim_count=len(supporting_claim_ids),
                    **unit_pair_event_details(member),
                )
            )
        cursor = end
    _copy_audit_interval(final_output, rewritten_audit, original, audit, cursor, len(original))
    post_digest = hashlib.sha256(bytes(final_output)).hexdigest()
    events.extend(
        {
            **event,
            "pre_projection_output_sha256": pre_digest,
            "post_projection_output_sha256": post_digest,
        }
        for event in projected_events
    )
    return final_output.decode("utf-8", errors="strict"), rewritten_audit, events


def project_native_emphasis(
    text: str,
    audit: list[dict],
    skeletons: Mapping[SourceName, DocumentSkeleton],
    artifacts: Mapping[SourceName, SourceArtifact],
    *,
    style_selection_resolver: Callable[..., object] | None = None,
    recorded_style_selections: Mapping[str, Mapping] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """Apply one unified post-merge positive style overlay."""

    return _compile_positive_style_overlay(
        text,
        audit,
        skeletons,
        artifacts,
        style_selection_resolver=style_selection_resolver,
        recorded_style_selections=recorded_style_selections,
    )


_TITLE_FRONT_HEADING_LIMIT = 12
_TITLE_FRONT_TEXT_LIMIT = 3000


@dataclass(frozen=True)
class _TitleChoiceCandidate:
    candidate_id: str
    heading_indices: tuple[int, ...]
    heading_ordinals: tuple[int, ...]
    visible_identity: str
    display: str


def _heading_label(raw: bytes, match: re.Match[bytes]) -> bytes:
    line_end = raw.find(b"\n", match.end())
    if line_end < 0:
        line_end = len(raw)
    return raw[match.end() : line_end].rstrip(b"\r\t ")


def _title_choice_candidates(text: str) -> tuple[_TitleChoiceCandidate, ...]:
    """Enumerate bounded exact final-heading chunks near the document front."""

    raw = text.encode("utf-8")
    matches = list(_MARKDOWN_HEADING.finditer(raw))[:_TITLE_FRONT_HEADING_LIMIT]
    candidates = []
    for start_index in range(len(matches)):
        for run_length in (1, 2, 3):
            stop_index = start_index + run_length
            if stop_index > len(matches):
                continue
            run = matches[start_index:stop_index]
            if any(
                raw[
                    raw.find(b"\n", left.end()) + 1 : right.start()
                ].strip()
                for left, right in zip(run, run[1:])
                if raw.find(b"\n", left.end()) >= 0
            ):
                continue
            labels = [_heading_label(raw, match) for match in run]
            try:
                visible = " ".join(
                    label.decode("utf-8", errors="strict") for label in labels
                )
            except UnicodeDecodeError:
                continue
            identity = _identity(visible)
            if not identity:
                continue
            ordinals = tuple(range(start_index + 1, stop_index + 1))
            digest = hashlib.sha256(
                b"\x00".join(labels)
                + b"\x00"
                + ",".join(str(value) for value in ordinals).encode("ascii")
            ).hexdigest()[:16]
            candidates.append(
                _TitleChoiceCandidate(
                    candidate_id=(
                        f"title-{ordinals[0]:03d}-{ordinals[-1]:03d}-{digest}"
                    ),
                    heading_indices=tuple(range(start_index, stop_index)),
                    heading_ordinals=ordinals,
                    visible_identity=identity,
                    display=(
                        f"Existing final heading {ordinals[0]}"
                        if len(ordinals) == 1
                        else (
                            "Existing contiguous final headings "
                            f"{ordinals[0]}-{ordinals[-1]}"
                        )
                    )
                    + f": {visible}",
                )
            )
    if len(candidates) > MAX_BOUNDED_TARGET_CHOICES:
        raise ConsensusContractError("title candidate set exceeded bounded choice limit")
    return tuple(candidates)


def _bounded_title_evidence(
    text: str,
    artifacts: Mapping[SourceName, SourceArtifact],
) -> str:
    """Provide bounded front matter for selection without persisting its text."""

    sections = [
        "Choose the complete article title from the numbered existing final-heading "
        "choices. The title can span adjacent title/subtitle headings. Do not choose "
        "an article type, access notice, journal name, author line, Abstract, or body "
        "section. Choose NONE when the supplied fronts do not establish one choice.",
        f"[current merged front]\n{text[:_TITLE_FRONT_TEXT_LIMIT]}",
    ]
    for source in sorted(artifacts):
        sections.append(
            f"[{source} extractor front]\n"
            f"{artifacts[source].text[:_TITLE_FRONT_TEXT_LIMIT]}"
        )
    return "\n\n".join(sections)


def _validated_reader_title(text: str, expected_identity: str) -> bool:
    raw = text.encode("utf-8")
    matches = list(_MARKDOWN_HEADING.finditer(raw))
    h1_matches = [match for match in matches if len(match.group(1)) == 1]
    if len(h1_matches) != 1 or h1_matches[0].start() != 0:
        return False
    if _identity(_heading_label(raw, h1_matches[0]).decode("utf-8")) != expected_identity:
        return False
    if sum(
        _identity(_heading_label(raw, match).decode("utf-8")) == expected_identity
        for match in matches
    ) != 1:
        return False
    report = abc_markdown_report(text)
    if report.get("valid") is not True or report.get("error_rule_ids"):
        return False
    try:
        from agr_abc_document_parsers import read_markdown

        reader_title = read_markdown(text).title
    except Exception:
        return False
    return isinstance(reader_title, str) and _identity(reader_title) == expected_identity


def _has_valid_reader_title(text: str) -> bool:
    raw = text.encode("utf-8")
    matches = list(_MARKDOWN_HEADING.finditer(raw))
    if not matches or matches[0].start() != 0 or len(matches[0].group(1)) != 1:
        return False
    try:
        identity = _identity(_heading_label(raw, matches[0]).decode("utf-8"))
    except UnicodeDecodeError:
        return False
    return bool(identity) and _validated_reader_title(text, identity)


def _render_title_choice(
    text: str,
    audit: list[dict],
    candidate: _TitleChoiceCandidate,
) -> tuple[str, list[dict], list[dict]] | None:
    """Join, mark, and move one selected existing heading run to byte zero."""

    raw = text.encode("utf-8")
    matches = list(_MARKDOWN_HEADING.finditer(raw))
    if not candidate.heading_indices or candidate.heading_indices[-1] >= len(matches):
        return None
    selected = [matches[index] for index in candidate.heading_indices]
    if any(
        raw[raw.find(b"\n", left.end()) + 1 : right.start()].strip()
        for left, right in zip(selected, selected[1:])
        if raw.find(b"\n", left.end()) >= 0
    ):
        return None

    selected_tail_indices = set(candidate.heading_indices[1:])
    replacements: list[tuple[int, int, bytes, str]] = []
    for index, match in enumerate(matches):
        if index in selected_tail_indices:
            continue
        current = match.group(1)
        desired = (
            b"#"
            if index == candidate.heading_indices[0]
            else b"##"
            if len(current) == 1
            else current
        )
        if desired != current:
            replacements.append(
                (match.start(1), match.end(1), desired, "alliance_heading_role_marker")
            )
    for left, right in zip(selected, selected[1:]):
        line_end = raw.find(b"\n", left.end())
        if line_end < 0:
            return None
        if line_end > left.end() and raw[line_end - 1 : line_end] == b"\r":
            line_end -= 1
        replacements.append(
            (line_end, right.end(), b" ", "alliance_title_composite_join")
        )

    rendered, rewritten_audit, events = _replace_deterministic_markup(
        text, audit, replacements
    )
    rendered_raw = rendered.encode("utf-8")
    rendered_matches = list(_MARKDOWN_HEADING.finditer(rendered_raw))
    title_matches = [
        match
        for match in rendered_matches
        if len(match.group(1)) == 1
        and _identity(_heading_label(rendered_raw, match).decode("utf-8"))
        == candidate.visible_identity
    ]
    if len(title_matches) != 1:
        return None
    title_match = title_matches[0]
    if title_match.start() > 0:
        title_end = _heading_block_end(rendered_raw, title_match)
        rendered, rewritten_audit = _permute_complete_audit_intervals(
            rendered,
            rewritten_audit,
            [
                (title_match.start(), title_end),
                (0, title_match.start()),
                (title_end, len(rendered_raw)),
            ],
        )
        events.append({
            "operation": "alliance_title_role_order",
            "audit_span_emitted": False,
            "reason": "model_selected_title_moved_to_byte_zero",
            "content_bytes_permuted_once": len(rendered_raw),
        })
    if not _validated_reader_title(rendered, candidate.visible_identity):
        return None
    return rendered, rewritten_audit, events

def render_document_role_slots(
    text: str,
    audit: list[dict],
    selected_skeleton: DocumentSkeleton,
    skeletons: Mapping[SourceName, DocumentSkeleton],
    *,
    decision_trace: tuple[dict, ...] = (),
    artifacts: Mapping[SourceName, SourceArtifact] | None = None,
    title_selection_resolver: Callable[..., object] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """Apply proved title/role ordering over one exact once-only content floor.

    The operation is a permutation of the already-assembled bytes followed by
    structural marker edits. It cannot insert publication text or discard a
    baseline occurrence. Conflicting or absent role evidence remains unchanged.
    """

    events: list[dict] = []
    text, audit, boundary_events = _replace_deterministic_markup(
        text,
        audit,
        _table_heading_boundary_ranges(text.encode("utf-8")),
    )
    events.extend(boundary_events)

    def heading_evidence(
        raw_value: bytes,
        heading_matches: list[re.Match[bytes]],
        current_audit: list[dict],
    ) -> dict[int, list[tuple[DocumentSkeleton, SkeletonHeading, SkeletonOccurrence]]]:
        evidence: dict[
            int, list[tuple[DocumentSkeleton, SkeletonHeading, SkeletonOccurrence]]
        ] = {index: [] for index in range(len(heading_matches))}
        for skeleton in skeletons.values():
            bound_ids = _bound_heading_unit_ids(
                raw_value,
                heading_matches,
                current_audit,
                skeleton,
                decision_trace,
            )
            headings = {heading.unit_id: heading for heading in skeleton.headings}
            occurrences = {
                occurrence.unit_id: occurrence
                for occurrence in skeleton.occurrences
            }
            for index, unit_id in enumerate(bound_ids):
                heading = headings.get(unit_id)
                occurrence = occurrences.get(unit_id)
                if heading is not None and occurrence is not None:
                    evidence[index].append((skeleton, heading, occurrence))
        return evidence

    def unresolved(reason: str, **details: object) -> None:
        events.append({
            "operation": "alliance_role_binding_unresolved",
            "audit_span_emitted": False,
            "reason": reason,
            **details,
        })

    raw = text.encode("utf-8")
    matches = list(_MARKDOWN_HEADING.finditer(raw))
    evidence = heading_evidence(raw, matches, audit)
    title_indices = {
        index
        for index, bindings in evidence.items()
        if any(
            skeleton.title_proven and heading.role == "title"
            for skeleton, heading, _occurrence in bindings
        )
    }
    title_index = next(iter(title_indices)) if len(title_indices) == 1 else None
    if title_index is not None and matches:
        title_match = matches[title_index]
        title_end = _heading_block_end(raw, title_match)
        if title_match.start() > 0:
            text, audit = _permute_complete_audit_intervals(
                text,
                audit,
                [
                    (title_match.start(), title_end),
                    (0, title_match.start()),
                    (title_end, len(raw)),
                ],
            )
            events.append({
                "operation": "alliance_title_role_order",
                "audit_span_emitted": False,
                "reason": "proved_title_moved_to_byte_zero",
                "content_bytes_permuted_once": len(raw),
            })

    model_title_identity: str | None = None
    if (
        title_index is None
        and title_selection_resolver is not None
        and artifacts
        and not _has_valid_reader_title(text)
    ):
        candidates = _title_choice_candidates(text)
        if candidates:
            none_id = "title-none"
            selection_id = (
                "document_title_choice:"
                + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
            )
            choices = [{
                "candidate_id": none_id,
                "candidate_type": "none",
                "display": _bounded_title_evidence(text, artifacts),
            }] + [
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_type": "final_heading_run",
                    "heading_ordinals": list(candidate.heading_ordinals),
                    "display": candidate.display,
                }
                for candidate in candidates
            ]
            try:
                receipt = title_selection_resolver(
                    reason=selection_id,
                    baseline_id=none_id,
                    choices=choices,
                )
                if not isinstance(receipt, Mapping):
                    raise ValueError("title selection receipt is missing")
                selected_id = receipt.get("selected_candidate_id")
                required_receipt = {
                    "title_selection_request_sha256": receipt.get("request_sha256"),
                    "title_selection_response_choice": receipt.get("response_choice"),
                    "title_selection_model": receipt.get("model"),
                    "title_selection_reasoning_effort": receipt.get("reasoning_effort"),
                }
                if (
                    not isinstance(required_receipt["title_selection_request_sha256"], str)
                    or len(required_receipt["title_selection_request_sha256"]) != 64
                    or type(required_receipt["title_selection_response_choice"]) is not int
                    or required_receipt["title_selection_model"] != "gpt-5.6-sol"
                    or required_receipt["title_selection_reasoning_effort"] != "high"
                ):
                    raise ValueError("title selection receipt is incomplete")
                selected_candidate = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.candidate_id == selected_id
                    ),
                    None,
                )
                selection_event = {
                    "operation": "alliance_model_title_selection",
                    "audit_span_emitted": False,
                    "title_selection_id": selection_id,
                    "title_selection_candidate_ids": [
                        candidate.candidate_id for candidate in candidates
                    ],
                    "title_selection_candidate_count": len(candidates),
                    "title_selection_none_id": none_id,
                    "title_selected_candidate_id": selected_id,
                    "title_selection_method": "sol_numbered_choice",
                    **required_receipt,
                }
                if selected_id == none_id:
                    events.append({**selection_event, "outcome": "none"})
                elif selected_candidate is None:
                    events.append({**selection_event, "outcome": "invalid_candidate"})
                else:
                    rendered = _render_title_choice(text, audit, selected_candidate)
                    if rendered is None:
                        events.append({**selection_event, "outcome": "validation_rejected"})
                    else:
                        text, audit, title_events = rendered
                        model_title_identity = selected_candidate.visible_identity
                        events.extend([
                            {
                                **selection_event,
                                "outcome": "selected",
                                "title_heading_ordinals": list(
                                    selected_candidate.heading_ordinals
                                ),
                                "title_visible_identity_sha256": hashlib.sha256(
                                    model_title_identity.encode("utf-8")
                                ).hexdigest(),
                            },
                            *title_events,
                        ])
            except Exception as exc:
                events.append({
                    "operation": "alliance_model_title_selection",
                    "audit_span_emitted": False,
                    "title_selection_id": selection_id,
                    "title_selection_candidate_count": len(candidates),
                    "title_selection_method": "unavailable",
                    "outcome": type(exc).__name__,
                })

    if (
        any(skeleton.title_proven for skeleton in skeletons.values())
        and title_index is None
        and model_title_identity is None
    ):
        unresolved(
            "proved_title_not_bound_to_one_output_occurrence",
            bound_title_occurrence_count=len(title_indices),
        )

    raw = text.encode("utf-8")
    matches = list(_MARKDOWN_HEADING.finditer(raw))
    evidence = heading_evidence(raw, matches, audit)
    title_indices = {
        index
        for index, bindings in evidence.items()
        if any(
            skeleton.title_proven and heading.role == "title"
            for skeleton, heading, _occurrence in bindings
        )
    }
    preserve_existing_reader_title = (
        model_title_identity is None
        and title_index is None
        and title_selection_resolver is not None
        and _has_valid_reader_title(text)
    )
    if model_title_identity is not None:
        model_title_matches = [
            match
            for match in matches
            if match.start() == 0
            and len(match.group(1)) == 1
            and _identity(_heading_label(raw, match).decode("utf-8"))
            == model_title_identity
        ]
        title_match = model_title_matches[0] if len(model_title_matches) == 1 else None
    elif preserve_existing_reader_title:
        title_match = matches[0]
    else:
        title_match = matches[next(iter(title_indices))] if len(title_indices) == 1 else None
    marker_replacements = []
    for match in matches:
        current = match.group(1)
        desired = b"#" if match is title_match else b"##" if len(current) == 1 else current
        if desired != current:
            marker_replacements.append(
                (match.start(1), match.end(1), desired, "alliance_heading_role_marker")
            )
    text, audit, marker_events = _replace_deterministic_markup(
        text, audit, marker_replacements
    )
    events.extend(marker_events)

    text, audit, front_events = _replace_deterministic_markup(
        text,
        audit,
        _front_matter_marker_ranges(text.encode("utf-8")),
    )
    events.extend(front_events)

    # Stable-sort only top-level role containers. Nested section order and all
    # bytes inside each container remain untouched.
    raw = text.encode("utf-8")
    all_headings = list(_MARKDOWN_HEADING.finditer(raw))
    all_evidence = heading_evidence(raw, all_headings, audit)
    top_heading_entries = [
        (index, match)
        for index, match in enumerate(all_headings)
        if len(match.group(1)) == 2
    ]
    top_headings = [match for _index, match in top_heading_entries]
    if top_headings:
        containers = []
        for index, (all_index, match) in enumerate(top_heading_entries):
            start = match.start()
            end = (
                top_headings[index + 1].start()
                if index + 1 < len(top_headings)
                else len(raw)
            )
            identity = _heading_identity_at(raw, match)
            bindings = all_evidence.get(all_index, [])
            regions = {occurrence.region for _skeleton, _heading, occurrence in bindings}
            bound_identities = {heading.identity for _skeleton, heading, _occurrence in bindings}
            if len(regions) > 1:
                unresolved(
                    "conflicting_bound_role_regions",
                    heading_ordinal=all_index + 1,
                )
                rank = 2
            elif regions == {"front"}:
                rank = 0
            elif identity == "abstract" and "abstract" in bound_identities:
                rank = 1
            elif regions == {"back"}:
                rank = 4 if identity == "references" and "references" in bound_identities else 3
            else:
                rank = 2
                if not bindings and (identity == "abstract" or identity in _BACK_ROLE_KEYS):
                    unresolved(
                        "semantic_heading_label_has_no_occurrence_binding",
                        heading_ordinal=all_index + 1,
                    )
            containers.append((rank, index, start, end, identity))
        ordered = sorted(containers, key=lambda item: (item[0], item[1]))
        if [item[1] for item in ordered] != list(range(len(containers))):
            prefix_end = top_headings[0].start()
            intervals = [(0, prefix_end)] + [
                (start, end) for _rank, _index, start, end, _identity_value in ordered
            ]
            text, audit = _permute_complete_audit_intervals(text, audit, intervals)
            events.append({
                "operation": "alliance_role_container_order",
                "audit_span_emitted": False,
                "reason": "proved_front_body_back_order",
                "container_count": len(containers),
                "content_bytes_permuted_once": len(raw),
            })

    # Build one final bibliography slot from bound occurrence roles. A heading
    # label is not proof by itself: native reference evidence can prove a
    # missing/malformed container, while Markdown reference roles require a
    # source-bound bibliography heading.
    artifact = SourceArtifact.from_text(selected_skeleton.source, text)
    units = scan_structural_units(artifact)
    heading_matches = list(_MARKDOWN_HEADING.finditer(artifact.raw_utf8))
    bibliography_heading_bound = any(
        _heading_identity_at(artifact.raw_utf8, match)
        in {
            "reference",
            "references",
            "referencesandnotes",
            "bibliography",
            "literaturecited",
        }
        and bool(bindings)
        for match, bindings in zip(
            heading_matches,
            heading_evidence(artifact.raw_utf8, heading_matches, audit).values(),
        )
    )
    initial_roles = [
        _effective_bound_unit_role(unit, audit, skeletons)
        for unit in units
    ]
    source_reference_syntax = [
        role == "reference"
        and evidence == "source_markdown"
        and _REFERENCE_MARKER.match(
            artifact.raw_utf8[unit.byte_start : unit.byte_end]
        )
        is not None
        for unit, (role, evidence) in zip(units, initial_roles)
    ]
    bibliography_proven = any(
        role == "reference" and evidence == "native"
        for role, evidence in initial_roles
    ) or any(source_reference_syntax) or (
        bibliography_heading_bound
        and any(
            role == "reference" and evidence == "source_markdown"
            for role, evidence in initial_roles
        )
    )
    if bibliography_proven:
        text, audit, heading_events = _replace_deterministic_markup(
            text,
            audit,
            _bibliography_marker_ranges(artifact.raw_utf8),
        )
        events.extend(heading_events)

        artifact = SourceArtifact.from_text(selected_skeleton.source, text)
        units = scan_structural_units(artifact)
        roles = [
            _effective_bound_unit_role(unit, audit, skeletons)
            for unit in units
        ]
        reference_indexes = {
            index
            for index, (unit, (role, evidence)) in enumerate(zip(units, roles))
            if role == "reference"
            and (
                evidence == "native"
                or (
                    evidence == "source_markdown"
                    and (
                        bibliography_heading_bound
                        or _REFERENCE_MARKER.match(
                            artifact.raw_utf8[unit.byte_start : unit.byte_end]
                        )
                        is not None
                    )
                )
            )
        }
        numbered_units = []
        for index, unit in enumerate(units):
            marker = _REFERENCE_ORDINAL.match(
                artifact.raw_utf8[unit.byte_start : unit.byte_end]
            )
            if marker is not None and roles[index][0] in {
                "list",
                "paragraph",
                "reference",
            }:
                numbered_units.append((index, int(marker.group("ordinal"))))
        active_ordinal = None
        for index, ordinal in numbered_units:
            if index in reference_indexes:
                active_ordinal = ordinal
            elif active_ordinal is not None and ordinal == active_ordinal + 1:
                reference_indexes.add(index)
                active_ordinal = ordinal
            else:
                active_ordinal = None
        active_ordinal = None
        for index, ordinal in reversed(numbered_units):
            if index in reference_indexes:
                active_ordinal = ordinal
            elif active_ordinal is not None and ordinal + 1 == active_ordinal:
                reference_indexes.add(index)
                active_ordinal = ordinal
            else:
                active_ordinal = None
        for index, (role, evidence) in enumerate(roles):
            if evidence.startswith("conflicting"):
                unresolved(
                    "conflicting_bound_occurrence_roles",
                    structural_unit_ordinal=index + 1,
                    evidence=evidence,
                )
        if reference_indexes:
            starts = [unit.byte_start for unit in units]
            blocks = [
                (
                    unit.byte_start,
                    starts[index + 1] if index + 1 < len(starts) else len(artifact.raw_utf8),
                    index in reference_indexes,
                )
                for index, unit in enumerate(units)
            ]
            prefix = (0, starts[0], False) if starts else (0, len(artifact.raw_utf8), False)
            non_reference_blocks = [
                (start, end)
                for start, end, is_reference in blocks
                if not is_reference
            ]
            reference_blocks = [
                (start, end)
                for start, end, is_reference in blocks
                if is_reference
            ]
            ordered_intervals = [
                (prefix[0], prefix[1]),
                *non_reference_blocks,
                *reference_blocks,
            ]
            current_intervals = [
                (prefix[0], prefix[1]),
                *((start, end) for start, end, _is_reference in blocks),
            ]
            if ordered_intervals != current_intervals:
                text, audit = _permute_complete_audit_intervals(
                    text,
                    audit,
                    ordered_intervals,
                )
                events.append({
                    "operation": "alliance_bibliography_role_order",
                    "audit_span_emitted": False,
                    "reason": "proved_references_moved_to_final_role_slot",
                    "reference_unit_count": len(reference_blocks),
                    "content_bytes_permuted_once": len(artifact.raw_utf8),
                })
            reference_start = sum(
                end - start
                for start, end in [
                    (prefix[0], prefix[1]),
                    *non_reference_blocks,
                ]
            )
            raw = text.encode("utf-8")
            heading = _canonical_bibliography_heading(raw[:reference_start])
            text, audit, insert_events = _replace_deterministic_markup(
                text,
                audit,
                [(
                    reference_start,
                    reference_start,
                    heading,
                    "alliance_bibliography_heading_insert",
                )],
            )
            events.extend(insert_events)

    text, audit, figure_events = _render_figure_legend_slot(
        text,
        audit,
        selected_skeleton.source,
    )
    events.extend(figure_events)

    text, audit, table_label_events = _replace_deterministic_markup(
        text,
        audit,
        _table_caption_marker_ranges(text.encode("utf-8")),
    )
    events.extend(table_label_events)

    # The pinned Alliance reader consumes numbered references. Normalize only
    # the structural entry marker inside the canonical proved container.
    artifact = SourceArtifact.from_text(selected_skeleton.source, text)
    units = scan_structural_units(artifact)
    reference_headings = [
        unit
        for unit in units
        if unit.unit_type == "heading"
        and _identity(
            _MARKDOWN_HEADING.sub(
                b"", artifact.raw_utf8[unit.byte_start:unit.byte_end], count=1
            ).decode("utf-8", errors="strict")
        ) == "references"
    ]
    if len(reference_headings) == 1:
        heading = reference_headings[0]
        reference_units = [
            unit
            for unit in units
            if unit.unit_type == "reference" and unit.byte_start > heading.byte_end
        ]
        reference_replacements = []
        for ordinal, unit in enumerate(reference_units, start=1):
            raw_unit = artifact.raw_utf8[unit.byte_start:unit.byte_end]
            marker = _REFERENCE_MARKER.match(raw_unit)
            start = unit.byte_start
            end = unit.byte_start if marker is None else unit.byte_start + marker.end()
            replacement = f"{ordinal}. ".encode("ascii")
            if artifact.raw_utf8[start:end] != replacement:
                reference_replacements.append(
                    (start, end, replacement, "alliance_reference_marker")
                )
            if ordinal > 1:
                previous = reference_units[ordinal - 2]
                separator = artifact.raw_utf8[previous.byte_end : unit.byte_start]
                if separator.count(b"\n") < 2:
                    reference_replacements.append((
                        unit.byte_start,
                        unit.byte_start,
                        b"\n",
                        "alliance_reference_blank_separator",
                    ))
        text, audit, reference_events = _replace_deterministic_markup(
            text, audit, reference_replacements
        )
        events.extend(reference_events)

    # S03 is a pure hierarchy constraint. A skipped level is reduced only to
    # the next legal depth; visible heading text and relative order are fixed.
    raw = text.encode("utf-8")
    heading_replacements = []
    previous_level = 0
    for match in _MARKDOWN_HEADING.finditer(raw):
        current_level = len(match.group(1))
        desired_level = current_level
        if previous_level and current_level > previous_level + 1:
            desired_level = previous_level + 1
        if desired_level != current_level:
            heading_replacements.append((
                match.start(1),
                match.end(1),
                b"#" * desired_level,
                "alliance_heading_depth",
            ))
        previous_level = desired_level
    text, audit, heading_events = _replace_deterministic_markup(
        text, audit, heading_replacements
    )
    events.extend(heading_events)

    # Role-slot permutations can place two formerly separated source blocks
    # next to each other. Reassert this schema boundary after every permutation
    # has completed; the operation is idempotent when the boundary is present.
    text, audit, boundary_events = _replace_deterministic_markup(
        text,
        audit,
        _table_heading_boundary_ranges(text.encode("utf-8")),
    )
    events.extend(boundary_events)

    return text, audit, events


def choose_document_skeleton(
    skeletons: Mapping[SourceName, DocumentSkeleton],
    artifacts: Mapping[SourceName, SourceArtifact],
    *,
    preferred_source: SourceName,
) -> SkeletonSelection:
    """Choose one complete outline using validator and native evidence."""

    if not skeletons:
        raise ConsensusContractError("no document skeleton candidate is available")
    ranked = []
    trace = []
    for source in sorted(skeletons):
        skeleton = skeletons[source]
        artifact = artifacts[source]
        rendered, _audit, _events = render_document_skeleton(
            artifact.text, _whole_source_audit(artifact), skeleton
        )
        report = abc_markdown_report(rendered)
        error_count = len(report.get("error_rule_ids", ()))
        warning_count = len(report.get("warning_rule_ids", ()))
        score = (
            int(error_count == 0),
            int(source == preferred_source),
            int(skeleton.title_proven),
            -warning_count,
            int(skeleton.native_artifact_digest is not None),
            int(
                skeleton.expected_page_count is not None
                and skeleton.covered_page_count == skeleton.expected_page_count
            ),
            skeleton.native_mapped_occurrence_count,
            len(skeleton.occurrences),
            skeleton.payload_byte_count,
            skeleton.matched_native_heading_count,
            _SOURCE_ORDER[source],
        )
        trace.append({
            "source": source,
            "skeleton_id": skeleton.skeleton_id,
            "projection_id": skeleton.projection_id,
            "title_proven": skeleton.title_proven,
            "native_heading_count": skeleton.native_heading_count,
            "matched_native_heading_count": skeleton.matched_native_heading_count,
            "payload_byte_count": skeleton.payload_byte_count,
            "occurrence_count": len(skeleton.occurrences),
            "native_mapped_occurrence_count": (
                skeleton.native_mapped_occurrence_count
            ),
            "expected_page_count": skeleton.expected_page_count,
            "covered_page_count": skeleton.covered_page_count,
            "validator_error_rule_ids": report.get("error_rule_ids", []),
            "validator_warning_rule_ids": report.get("warning_rule_ids", []),
            "selected": False,
        })
        ranked.append((score, source, skeleton))
    _selected_score, selected_source, selected = max(
        ranked, key=lambda item: item[0]
    )
    for entry in trace:
        entry["selected"] = entry["source"] == selected_source
    viable_projections = {
        item[2].projection_id
        for item in ranked
        if item[0][0] == 1
    }
    conflict = len(viable_projections) > 1
    return SkeletonSelection(
        skeleton=selected,
        trace=tuple(trace),
        conflict=conflict,
    )


def load_runtime_native_structures(
    artifacts: Mapping[SourceName, SourceArtifact],
    output_paths: Mapping[SourceName, str | Path],
    *,
    expected_pdf_sha256: str | None = None,
) -> tuple[dict[SourceName, NativeStructureArtifact], dict[SourceName, str]]:
    """Load and revalidate native structures for successful extractors."""

    loaded: dict[SourceName, NativeStructureArtifact] = {}
    failures: dict[SourceName, str] = {}
    for source, artifact in sorted(artifacts.items()):
        output_path = output_paths.get(source)
        if output_path is None:
            failures[source] = "output_path_missing"
            continue
        try:
            manifest, native_bytes = load_native_extractor_artifact(
                source=source,
                output_filename=output_path,
                expected_pdf_sha256=expected_pdf_sha256,
            )
            native_style_bytes = load_native_style_artifact(
                source=source,
                output_filename=output_path,
                manifest=manifest,
            )
            candidate = NativeStructureArtifact.from_loaded(
                source,
                artifact,
                manifest,
                native_bytes,
                native_style_bytes,
            )
            native_heading_hints(candidate)
            loaded[source] = candidate
        except (OSError, UnicodeError, ValueError, ET.ParseError, json.JSONDecodeError) as exc:
            failures[source] = type(exc).__name__
    return loaded, failures
