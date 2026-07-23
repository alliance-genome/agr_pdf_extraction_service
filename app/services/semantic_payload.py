"""Ordered, content-free receipt for what the real Alliance reader retains."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from app.services.abc_markdown_policy import abc_markdown_report
from app.services.document_skeleton import DocumentSkeleton, build_document_skeleton
from app.services.source_contracts import SourceArtifact, SourceName
from app.services.source_merge import _inline_italic_profile, scan_structural_units


SEMANTIC_PAYLOAD_CONTRACT_VERSION = "ordered-semantic-payload-v2"
_SEMANTIC_ROLES = ("body", "tables", "figures", "back_matter", "references")
_BACK_MATTER_HEADINGS = {
    "acknowledgments",
    "funding",
    "authornotes",
    "competinginterests",
    "dataavailability",
    "authorcontributions",
}
_BIBLIOGRAPHY_HEADINGS = {
    "reference",
    "references",
    "referencesandnotes",
    "bibliography",
    "literaturecited",
}
_SECONDARY_ABSTRACT_HEADINGS = {
    "authorsummary",
    "elifedigest",
    "plainlanguagesummary",
    "tableofcontentssummary",
}
_ROLE_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_digest(value: object) -> str:
    return _digest(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _semantic_identity(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"[`*_~\[\]()]", "", value)
    return "".join(character.casefold() for character in value if character.isalnum())


def _semantic_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _ROLE_TOKEN.findall(value))


def _semantic_unit_tokens(value: str, unit_type: str) -> tuple[str, ...]:
    if unit_type in {"list", "reference"}:
        value = re.sub(
            r"^\s*(?:[-+*]|\d+[.)]|\[\d+\])\s+",
            "",
            value,
            count=1,
        )
    return _semantic_tokens(value)


def _semantic_front_unit_tokens(value: str, unit_type: str) -> tuple[str, ...]:
    """Ignore reader-authored affiliation links, not front-matter payload."""

    value = re.sub(
        r"<sup>[\d,\s*\\†‡]+</sup>",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return _semantic_unit_tokens(value, unit_type)


def _markdown_role_streams(text: str) -> dict[str, tuple[str, ...]]:
    """Project schema Markdown into fixed semantic-role token streams."""

    artifact = SourceArtifact.from_text("grobid", text)
    streams: dict[str, list[str]] = {role: [] for role in _SEMANTIC_ROLES}
    table_captions: list[str] = []
    table_rows: list[str] = []
    keywords: list[str] = []
    front_units: list[tuple[str, tuple[str, ...]]] = []
    body_slots: dict[str, list[str]] = {}
    back_slots: dict[str, dict[str, list[str]]] = {
        identity: {} for identity in sorted(_BACK_MATTER_HEADINGS)
    }
    back_slots["other"] = {}
    current_role = "body"
    current_back_slot = "other"
    in_front = True
    for unit in scan_structural_units(artifact):
        raw = artifact.raw_utf8[unit.byte_start : unit.byte_end].decode(
            "utf-8", errors="strict"
        )
        role = current_role
        if unit.unit_type == "heading":
            marker = re.match(r"^\s*#{1,6}\s+", raw)
            label = raw[marker.end() :] if marker is not None else raw
            identity = _semantic_identity(label)
            if identity in _BIBLIOGRAPHY_HEADINGS:
                current_role = "references"
                in_front = False
                continue
            if identity == "figurelegends":
                current_role = "figures"
                in_front = False
                continue
            if identity in _BACK_MATTER_HEADINGS:
                current_role = "back_matter"
                current_back_slot = identity
                in_front = False
                # Back-matter headings are schema structure, not publication
                # payload. The Alliance reader may omit an empty one while
                # retaining every byte of the following populated slots.
                continue
            if identity in _SECONDARY_ABSTRACT_HEADINGS:
                current_role = "body"
                in_front = True
            elif marker is not None and len(marker.group(0).lstrip().split()[0]) <= 2:
                current_role = "body"
                if identity != "abstract" and len(
                    marker.group(0).lstrip().split()[0]
                ) == 2:
                    in_front = False
            role = current_role
        elif unit.unit_type == "reference":
            role = "references"
        elif unit.unit_type == "table":
            role = "tables"
        elif unit.unit_type == "figure_caption":
            role = "figures"
        else:
            visible_start = re.sub(r"^[\s*_`#]+", "", raw).casefold()
            if re.match(r"(?:figure|fig\.)\s*\d+", visible_start):
                role = "figures"
            elif re.match(r"(?:supplementary\s+)?table\s*\d+", visible_start):
                role = "tables"
        tokens = (
            _semantic_front_unit_tokens(raw, unit.unit_type)
            if role == "body" and in_front
            else _semantic_unit_tokens(raw, unit.unit_type)
        )
        if role == "tables":
            if re.match(
                r"(?:supplementary\s+)?table\s*\d+",
                re.sub(r"^[\s*_`#]+", "", raw).casefold(),
            ):
                table_captions.extend(tokens)
            else:
                table_rows.extend(tokens)
        elif role == "back_matter":
            back_slots[current_back_slot].setdefault(unit.unit_type, []).extend(tokens)
        elif role == "body" and in_front:
            front_units.append((unit.unit_type, tokens))
        elif role == "body" and re.match(
            r"^\s*\*\*keywords:\*\*", raw, re.IGNORECASE
        ):
            keywords.extend(tokens)
        elif role == "body":
            body_slots.setdefault(unit.unit_type, []).extend(tokens)
        else:
            streams[role].extend(tokens)
    streams["tables"] = [
        *table_captions,
        "__pdfx_table_rows__",
        *table_rows,
    ]
    streams["body"] = [
        "__pdfx_front_matter__",
        *(
            token
            for unit_type, unit_tokens in sorted(front_units)
            for token in (
                f"__pdfx_front_unit_type_{unit_type}__",
                *unit_tokens,
                "__pdfx_front_unit_end__",
            )
        ),
        *keywords,
        "__pdfx_body_content__",
        *(
            token
            for unit_type in sorted(body_slots)
            for token in (
                f"__pdfx_body_unit_type_{unit_type}__",
                *body_slots[unit_type],
            )
        ),
    ]
    streams["back_matter"] = [
        token
        for slot in sorted(back_slots)
        for unit_type in sorted(back_slots[slot])
        for token in (
            f"__pdfx_back_slot_{slot}_{unit_type}__",
            *back_slots[slot][unit_type],
        )
    ]
    return {role: tuple(streams[role]) for role in _SEMANTIC_ROLES}


def _role_stream_receipt(streams: Mapping[str, Sequence[str]]) -> dict:
    return {
        role: {
            "token_count": len(streams[role]),
            "ordered_token_sha256": _canonical_digest(list(streams[role])),
        }
        for role in _SEMANTIC_ROLES
    }


def _schema_only_occurrence(text: str, occurrence: SemanticPayloadOccurrence) -> bool:
    if occurrence.unit_type != "heading":
        return False
    raw = text.encode("utf-8")[
        occurrence.output_byte_start : occurrence.output_byte_end
    ].decode("utf-8", errors="strict")
    marker = re.match(r"^\s*#{1,6}\s+", raw)
    label = raw[marker.end() :] if marker is not None else raw
    return _semantic_identity(label) in {"references", "figurelegends"}


@dataclass(frozen=True)
class SemanticPayloadOccurrence:
    occurrence_id: str
    order: int
    unit_type: str
    content_digest: str
    role_digest: str
    selected_source: str | None
    native_id_digest: str | None
    source_span_digest: str
    output_byte_start: int
    output_byte_end: int
    emphasis_occurrence_ids: tuple[str, ...]
    source_bound: bool


@dataclass(frozen=True)
class SemanticPayloadReceipt:
    contract_version: str
    output_sha256: str
    occurrences: tuple[SemanticPayloadOccurrence, ...]
    receipt_sha256: str

    def as_metric(self) -> dict:
        occurrence_metrics = []
        for occurrence in self.occurrences:
            payload = asdict(occurrence)
            payload["emphasis_occurrence_ids"] = list(
                occurrence.emphasis_occurrence_ids
            )
            occurrence_metrics.append(payload)
        return {
            "contract_version": self.contract_version,
            "output_sha256": self.output_sha256,
            "receipt_sha256": self.receipt_sha256,
            "occurrence_count": len(self.occurrences),
            "source_bound_occurrence_count": sum(
                occurrence.source_bound for occurrence in self.occurrences
            ),
            "occurrences": occurrence_metrics,
        }


def _unit_evidence(artifact: SourceArtifact) -> tuple[dict, ...]:
    skeleton = build_document_skeleton(artifact, None)
    occurrences_by_unit = {
        occurrence.unit_id: occurrence for occurrence in skeleton.occurrences
    }
    headings_by_unit = {heading.unit_id: heading for heading in skeleton.headings}
    evidence = []
    for unit in scan_structural_units(artifact):
        raw = artifact.raw_utf8[unit.byte_start : unit.byte_end]
        text = raw.decode("utf-8", errors="strict")
        profile = _inline_italic_profile(text, unit.unit_type)
        content_digest = (
            _digest(unit.comparison_key.encode("utf-8"))
            if profile is None
            else profile.visible_digest
        )
        occurrence = occurrences_by_unit[unit.unit_id]
        heading = headings_by_unit.get(unit.unit_id)
        evidence.append(
            {
                "unit": unit,
                "content_digest": content_digest,
                "role_digest": _canonical_digest(
                    {
                        "unit_type": unit.unit_type,
                        "slot_key": occurrence.slot_key,
                        "heading_role": None if heading is None else heading.role,
                        "heading_level": (
                            None if heading is None else heading.final_level
                        ),
                    }
                ),
                "emphasis_occurrence_ids": (
                    () if profile is None else profile.emphasis_occurrence_ids
                ),
            }
        )
    return tuple(evidence)


def _origin_for_unit(
    unit,
    audit: Sequence[Mapping],
    skeletons: Mapping[SourceName, DocumentSkeleton],
    candidate_occurrences: Mapping[str, str],
) -> tuple[str, str | None, str | None, str, bool]:
    source_spans = []
    origin_ids = []
    native_ids = []
    sources = []
    occurrences_by_source = {
        source: tuple(skeleton.occurrences)
        for source, skeleton in skeletons.items()
    }
    for entry in audit:
        output_start = entry.get("output_byte_start")
        output_end = entry.get("output_byte_end")
        if type(output_start) is not int or type(output_end) is not int:
            continue
        overlap_start = max(unit.byte_start, output_start)
        overlap_end = min(unit.byte_end, output_end)
        if overlap_start >= overlap_end or entry.get("source") == "deterministic_markup":
            continue
        source = entry.get("source")
        source_start = entry.get("source_byte_start")
        if source not in occurrences_by_source or type(source_start) is not int:
            continue
        selected_start = source_start + overlap_start - output_start
        selected_end = selected_start + overlap_end - overlap_start
        sources.append(source)
        source_spans.append(
            {
                "source": source,
                "artifact_digest": entry.get("artifact_digest"),
                "source_byte_start": selected_start,
                "source_byte_end": selected_end,
            }
        )
        candidate_occurrence = candidate_occurrences.get(entry.get("candidate_id"))
        matched = next(
            (
                occurrence
                for occurrence in occurrences_by_source[source]
                if occurrence.source_byte_start <= selected_start
                and selected_end <= occurrence.source_byte_end
            ),
            None,
        )
        # Prefer the exact source skeleton occurrence when the selected span
        # contains multiple output units (a bounded composite candidate). The
        # shared candidate structural ID remains a fallback for clipped or
        # otherwise unbound single-unit alternatives.
        occurrence_id = (
            matched.occurrence_id
            if matched is not None
            else candidate_occurrence
        )
        if occurrence_id is not None:
            origin_ids.append(occurrence_id)
        if matched is not None and matched.native_id is not None:
            native_ids.append(matched.native_id)

    unique_origins = tuple(dict.fromkeys(origin_ids))
    occurrence_id = (
        unique_origins[0]
        if len(unique_origins) == 1
        else _canonical_digest(
            {
                "kind": "composite_output_occurrence",
                "origin_ids": unique_origins,
                "output_byte_start": unit.byte_start,
                "output_byte_end": unit.byte_end,
            }
        )
    )
    unique_sources = tuple(dict.fromkeys(sources))
    unique_native_ids = tuple(dict.fromkeys(native_ids))
    return (
        occurrence_id,
        unique_sources[0] if len(unique_sources) == 1 else None,
        (
            _digest(unique_native_ids[0].encode("utf-8"))
            if len(unique_native_ids) == 1
            else None
        ),
        _canonical_digest(source_spans),
        bool(source_spans and unique_origins),
    )


def build_semantic_payload_receipt(
    text: str,
    audit: Sequence[Mapping],
    *,
    baseline_source: SourceName,
    skeletons: Mapping[SourceName, DocumentSkeleton],
    decision_trace: Sequence[Mapping] = (),
) -> SemanticPayloadReceipt:
    """Bind ordered final occurrences to exact source provenance and roles."""

    artifact = SourceArtifact.from_text(baseline_source, text)
    candidate_occurrences = {
        candidate.get("candidate_id"): candidate.get("structural_unit_id")
        for event in decision_trace
        for candidate in event.get("candidates", ())
        if candidate.get("candidate_id") and candidate.get("structural_unit_id")
    }
    occurrences = []
    used_occurrence_ids: set[str] = set()
    for order, evidence in enumerate(_unit_evidence(artifact)):
        unit = evidence["unit"]
        (
            occurrence_id,
            selected_source,
            native_id_digest,
            source_span_digest,
            source_bound,
        ) = _origin_for_unit(
            unit,
            audit,
            skeletons,
            candidate_occurrences,
        )
        if occurrence_id in used_occurrence_ids:
            occurrence_id = _canonical_digest({
                "kind": "split_output_occurrence",
                "origin_occurrence_id": occurrence_id,
                "output_byte_start": evidence["unit"].byte_start,
                "output_byte_end": evidence["unit"].byte_end,
            })
        used_occurrence_ids.add(occurrence_id)
        occurrences.append(
            SemanticPayloadOccurrence(
                occurrence_id=occurrence_id,
                order=order,
                unit_type=unit.unit_type,
                content_digest=evidence["content_digest"],
                role_digest=evidence["role_digest"],
                selected_source=selected_source,
                native_id_digest=native_id_digest,
                source_span_digest=source_span_digest,
                output_byte_start=unit.byte_start,
                output_byte_end=unit.byte_end,
                emphasis_occurrence_ids=tuple(
                    evidence["emphasis_occurrence_ids"]
                ),
                source_bound=source_bound,
            )
        )
    core = {
        "contract_version": SEMANTIC_PAYLOAD_CONTRACT_VERSION,
        "output_sha256": _digest(text.encode("utf-8")),
        "occurrences": [asdict(occurrence) for occurrence in occurrences],
    }
    return SemanticPayloadReceipt(
        contract_version=SEMANTIC_PAYLOAD_CONTRACT_VERSION,
        output_sha256=core["output_sha256"],
        occurrences=tuple(occurrences),
        receipt_sha256=_canonical_digest(core),
    )


def semantic_payload_reader_report(
    text: str,
    receipt: SemanticPayloadReceipt,
    *,
    validator_report: Mapping | None = None,
) -> dict:
    """Compare ordered receipt occurrences with the real pinned reader output."""

    output_digest = _digest(text.encode("utf-8"))
    validator = dict(validator_report or abc_markdown_report(text))
    report = {
        "contract_version": SEMANTIC_PAYLOAD_CONTRACT_VERSION,
        "validated_output_sha256": output_digest,
        "receipt_sha256": receipt.receipt_sha256,
        "required_occurrence_count": len(receipt.occurrences),
        "reader_occurrence_count": 0,
        "input_normalized_token_count": 0,
        "reader_normalized_token_count": 0,
        "retained_normalized_token_count": 0,
        "normalized_token_recall_ppm": 0,
        "normalized_token_precision_ppm": 0,
        "missing_occurrence_ids": [],
        "reordered_occurrence_ids": [],
        "role_changed_occurrence_ids": [],
        "formatting_lost_occurrence_ids": [],
        "unbound_occurrence_ids": [
            item.occurrence_id
            for item in receipt.occurrences
            if not item.source_bound and not _schema_only_occurrence(text, item)
        ],
        "expected_role_streams": {},
        "reader_role_streams": {},
        "role_stream_mismatch_roles": [],
        "reader_ast_reference_count": 0,
        "reader_ast_table_count": 0,
        "reader_ast_figure_count": 0,
        "reader_payload_retained": False,
        "protected_italics_retained": False,
        "reader_contract_pass": False,
        "mismatch_codes": [],
        "diagnostic_codes": [],
        "failure_code": None,
    }
    try:
        if receipt.output_sha256 != output_digest:
            raise ValueError("semantic receipt output digest mismatch")
        if (
            validator.get("parser_version_exact") is not True
            or validator.get("parser_implementation_exact") is not True
        ):
            raise ValueError("semantic reader parser identity mismatch")
        from agr_abc_document_parsers import emit_markdown, read_markdown

        document = read_markdown(text)
        reader_text = emit_markdown(document)
        expected_role_streams = _markdown_role_streams(text)
        reader_role_streams = _markdown_role_streams(reader_text)
        report["expected_role_streams"] = _role_stream_receipt(
            expected_role_streams
        )
        report["reader_role_streams"] = _role_stream_receipt(
            reader_role_streams
        )
        report["role_stream_mismatch_roles"] = [
            role
            for role in _SEMANTIC_ROLES
            if expected_role_streams[role] != reader_role_streams[role]
        ]

        def section_counts(sections) -> tuple[int, int]:
            table_count = 0
            figure_count = 0
            for section in sections:
                table_count += len(section.tables)
                figure_count += len(section.figures)
                child_tables, child_figures = section_counts(section.subsections)
                table_count += child_tables
                figure_count += child_figures
            return table_count, figure_count

        body_tables, body_figures = section_counts(document.sections)
        back_tables, back_figures = section_counts(document.back_matter)
        report.update({
            "reader_ast_reference_count": len(document.references),
            "reader_ast_table_count": len(document.tables) + body_tables + back_tables,
            "reader_ast_figure_count": len(document.figures) + body_figures + back_figures,
        })
        token_pattern = re.compile(r"[^\W_]+", re.UNICODE)
        input_tokens = Counter(
            token.casefold()
            for token in token_pattern.findall(text)
        )
        reader_tokens = Counter(
            token.casefold()
            for token in token_pattern.findall(reader_text)
        )
        retained_tokens = sum((input_tokens & reader_tokens).values())
        input_token_count = sum(input_tokens.values())
        reader_token_count = sum(reader_tokens.values())
        report.update({
            "input_normalized_token_count": input_token_count,
            "reader_normalized_token_count": reader_token_count,
            "retained_normalized_token_count": retained_tokens,
            "normalized_token_recall_ppm": (
                1_000_000
                if input_token_count == 0
                else retained_tokens * 1_000_000 // input_token_count
            ),
            "normalized_token_precision_ppm": (
                1_000_000
                if reader_token_count == 0
                else retained_tokens * 1_000_000 // reader_token_count
            ),
        })
        reader_artifact = SourceArtifact.from_text(
            "grobid",
            reader_text,
        )
        reader_units = _unit_evidence(reader_artifact)
        report["reader_occurrence_count"] = len(reader_units)
        positions: dict[str, list[int]] = {}
        for index, evidence in enumerate(reader_units):
            positions.setdefault(evidence["content_digest"], []).append(index)
        consumed: dict[str, int] = {}
        matched_positions = []
        for occurrence in receipt.occurrences:
            ordinal = consumed.get(occurrence.content_digest, 0)
            available = positions.get(occurrence.content_digest, [])
            if ordinal >= len(available):
                report["missing_occurrence_ids"].append(occurrence.occurrence_id)
                continue
            reader_index = available[ordinal]
            consumed[occurrence.content_digest] = ordinal + 1
            matched_positions.append((occurrence, reader_index))
            reader_evidence = reader_units[reader_index]
            if (
                occurrence.unit_type != reader_evidence["unit"].unit_type
                or occurrence.role_digest != reader_evidence["role_digest"]
            ):
                report["role_changed_occurrence_ids"].append(
                    occurrence.occurrence_id
                )
            if not set(occurrence.emphasis_occurrence_ids).issubset(
                reader_evidence["emphasis_occurrence_ids"]
            ):
                report["formatting_lost_occurrence_ids"].append(
                    occurrence.occurrence_id
                )
        maximum_position = -1
        for occurrence, reader_index in matched_positions:
            if reader_index <= maximum_position:
                report["reordered_occurrence_ids"].append(occurrence.occurrence_id)
            maximum_position = max(maximum_position, reader_index)

        payload_mismatches = (
            report["role_stream_mismatch_roles"]
            or report["unbound_occurrence_ids"]
        )
        report["reader_payload_retained"] = not bool(payload_mismatches)
        report["protected_italics_retained"] = not bool(
            report["formatting_lost_occurrence_ids"]
        )
        diagnostics = []
        for field, code in (
            ("missing_occurrence_ids", "reader_occurrence_missing"),
            ("reordered_occurrence_ids", "reader_occurrence_reordered"),
            ("role_changed_occurrence_ids", "reader_occurrence_role_changed"),
        ):
            if report[field]:
                diagnostics.append(code)
        report["diagnostic_codes"] = diagnostics
        mismatches = []
        if report["role_stream_mismatch_roles"]:
            mismatches.append("reader_semantic_role_stream_changed")
        if report["formatting_lost_occurrence_ids"]:
            mismatches.append("reader_occurrence_formatting_lost")
        if report["unbound_occurrence_ids"]:
            mismatches.append("semantic_occurrence_unbound")
        report["mismatch_codes"] = mismatches
        report["reader_contract_pass"] = bool(
            report["reader_payload_retained"]
            and report["protected_italics_retained"]
        )
    except Exception as exc:
        report["failure_code"] = type(exc).__name__
        report["mismatch_codes"] = ["reader_execution_failed"]
    return report
