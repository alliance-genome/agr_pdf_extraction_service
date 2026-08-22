"""Static contracts for the small deterministic spans authored by PDFX."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence


AtomKind = Literal["content", "boundary_whitespace"]
BoundaryOwnership = Literal["left", "right"]


@dataclass(frozen=True)
class DeterministicMarkupShape:
    """Code-owned shape and composition policy for one generated atom."""

    validator: Callable[[bytes], bool]
    atom_kind: AtomKind = "content"
    boundary_ownership: BoundaryOwnership | None = None


def _exact(expected: bytes) -> Callable[[bytes], bool]:
    return lambda span: span == expected


def _one_of(*expected: bytes) -> Callable[[bytes], bool]:
    return lambda span: span in expected


def _heading_marker(span: bytes) -> bool:
    return 1 <= len(span) <= 6 and set(span) == {ord("#")}


def _table_separator(span: bytes) -> bool:
    return re.fullmatch(rb"\n?\|(?:---\|)+\n", span) is not None


def _reference_marker(span: bytes) -> bool:
    return re.fullmatch(rb"[1-9]\d*\. ", span) is not None


def _inserted_heading(label: bytes) -> Callable[[bytes], bool]:
    # Optional prefixes accept same-contract cached receipts. New emitters
    # record each prefix as a separate left-owned boundary atom.
    return lambda span: span in {
        label + b"\n\n",
        b"\n" + label + b"\n\n",
        b"\n\n" + label + b"\n\n",
    }


_LEFT_BOUNDARY = dict(
    atom_kind="boundary_whitespace",
    boundary_ownership="left",
)


DETERMINISTIC_MARKUP_SHAPES: Mapping[str, DeterministicMarkupShape] = {
    "trailing_newline_normalization": DeterministicMarkupShape(
        _exact(b"\n"), **_LEFT_BOUNDARY
    ),
    "selected_document_skeleton": DeterministicMarkupShape(_heading_marker),
    "alliance_table_separator": DeterministicMarkupShape(_table_separator),
    "alliance_table_separator_boundary": DeterministicMarkupShape(
        _exact(b"\n"), **_LEFT_BOUNDARY
    ),
    "alliance_heading_role_marker": DeterministicMarkupShape(_heading_marker),
    "alliance_reference_marker": DeterministicMarkupShape(_reference_marker),
    "alliance_bibliography_heading_insert": DeterministicMarkupShape(
        _inserted_heading(b"## References")
    ),
    "alliance_bibliography_heading_boundary": DeterministicMarkupShape(
        _one_of(b"\n", b"\n\n"), **_LEFT_BOUNDARY
    ),
    "alliance_figure_legend_heading_insert": DeterministicMarkupShape(
        _inserted_heading(b"## Figure Legends")
    ),
    "alliance_figure_legend_heading_boundary": DeterministicMarkupShape(
        _one_of(b"\n", b"\n\n"), **_LEFT_BOUNDARY
    ),
    "alliance_heading_depth": DeterministicMarkupShape(_heading_marker),
    "alliance_figure_label_heading": DeterministicMarkupShape(_exact(b"### ")),
    "alliance_table_label_emphasis_marker": DeterministicMarkupShape(
        _exact(b"**")
    ),
    "alliance_figure_label_caption_boundary": DeterministicMarkupShape(
        _exact(b"\n\n"), **_LEFT_BOUNDARY
    ),
    "alliance_abstract_heading_separator": DeterministicMarkupShape(
        _exact(b"\n\n"), **_LEFT_BOUNDARY
    ),
    "alliance_table_heading_boundary": DeterministicMarkupShape(
        _exact(b"\n\n"), **_LEFT_BOUNDARY
    ),
    "alliance_reference_blank_separator": DeterministicMarkupShape(
        _exact(b"\n"), **_LEFT_BOUNDARY
    ),
    "alliance_front_list_block_separator": DeterministicMarkupShape(
        _exact(b"\n"), **_LEFT_BOUNDARY
    ),
    "alliance_orcid_url_prefix": DeterministicMarkupShape(
        _exact(b"https://orcid.org/")
    ),
    "alliance_abstract_heading_marker": DeterministicMarkupShape(_exact(b"## ")),
    "alliance_affiliation_ordinal_marker": DeterministicMarkupShape(_exact(b".")),
    "alliance_article_category_marker": DeterministicMarkupShape(
        _exact(b"**Categories:** ")
    ),
    "alliance_title_composite_join": DeterministicMarkupShape(_exact(b" ")),
    "native_emphasis_projection": DeterministicMarkupShape(_exact(b"*")),
}


def deterministic_markup_shape(operation: object) -> DeterministicMarkupShape | None:
    return DETERMINISTIC_MARKUP_SHAPES.get(operation) if isinstance(operation, str) else None


def deterministic_markup_span_is_valid(operation: object, span: bytes) -> bool:
    shape = deterministic_markup_shape(operation)
    return shape is not None and shape.validator(span)


def left_owned_boundary_span_is_valid(operation: object, span: bytes) -> bool:
    """Validate both bytes and declared ownership for a boundary atom."""

    shape = deterministic_markup_shape(operation)
    return bool(
        shape is not None
        and shape.atom_kind == "boundary_whitespace"
        and shape.boundary_ownership == "left"
        and shape.validator(span)
    )


def deterministic_audit_entry(
    *,
    output_byte_start: int,
    span: bytes,
    operation: str,
    transformation_id: str | None = None,
) -> dict:
    """Build one complete generated atom from its static operation contract."""

    if not span or not deterministic_markup_span_is_valid(operation, span):
        raise ValueError(f"invalid deterministic markup span for {operation}")
    entry = {
        "output_byte_start": output_byte_start,
        "output_byte_end": output_byte_start + len(span),
        "source": "deterministic_markup",
        "artifact_digest": hashlib.sha256(span).hexdigest(),
        "source_byte_start": 0,
        "source_byte_end": len(span),
        "candidate_id": None,
        "region_id": None,
        "decision_method": "deterministic",
        "transformation": operation,
    }
    if transformation_id is not None:
        entry["transformation_id"] = transformation_id
    return entry


def interval_splits_deterministic_atom(
    audit: Sequence[Mapping],
    start: int,
    end: int,
) -> bool:
    """Return whether retaining this interval would clip a generated atom."""

    for entry in audit:
        if entry.get("source") != "deterministic_markup":
            continue
        entry_start = entry.get("output_byte_start")
        entry_end = entry.get("output_byte_end")
        if type(entry_start) is not int or type(entry_end) is not int:
            continue
        overlap_start = max(start, entry_start)
        overlap_end = min(end, entry_end)
        if overlap_start < overlap_end and (
            overlap_start != entry_start or overlap_end != entry_end
        ):
            return True
    return False
