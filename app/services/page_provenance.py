"""Digest-bound page provenance for extractor and merged Markdown artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path


SOURCE_PAGE_PROVENANCE_SCHEMA = "pdfx-source-page-provenance"
SOURCE_PAGE_PROVENANCE_CONTRACT_VERSION = "source-page-provenance-v1"
SOURCE_PAGE_PROVENANCE_MEDIA_TYPE = "application/json"
MERGED_PAGE_PROVENANCE_SCHEMA = "pdfx-merged-page-provenance"
MERGED_PAGE_PROVENANCE_CONTRACT_VERSION = "merged-page-provenance-v1"
MERGED_PAGE_PROVENANCE_MEDIA_TYPE = "application/json"
DOCLING_SENTINEL_PREFIX = "PDFX_DOCLING_PAGE_BOUNDARY_"
MARKER_PAGE_SEPARATOR = "PDFXMARKERPAGEBOUNDARY7E53C1"

_FOLLOWING_OWNER_OPERATIONS = frozenset(
    {
        "alliance_heading_role_marker",
        "selected_document_skeleton",
        "alliance_heading_depth",
        "alliance_reference_marker",
        "alliance_bibliography_heading_insert",
        "alliance_bibliography_heading_boundary",
        "alliance_figure_legend_heading_insert",
        "alliance_figure_legend_heading_boundary",
        "alliance_figure_label_heading",
        "alliance_figure_label_caption_boundary",
        "alliance_table_heading_boundary",
        "alliance_reference_blank_separator",
        "alliance_abstract_heading_marker",
        "alliance_abstract_heading_separator",
        "alliance_affiliation_ordinal_marker",
        "alliance_orcid_url_prefix",
        "alliance_article_category_marker",
        "alliance_front_list_block_separator",
    }
)
_PRECEDING_OWNER_OPERATIONS = frozenset({"trailing_newline_normalization"})
_SURROUNDING_OWNER_OPERATIONS = frozenset(
    {
        "alliance_table_label_emphasis_marker",
        "alliance_table_separator",
        "alliance_table_separator_boundary",
        "alliance_title_composite_join",
    }
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_line_sha256(value: object) -> str:
    """Match the repository's canonical newline-terminated JSON artifacts."""

    return _sha256(_canonical_json_bytes(value) + b"\n")


def source_page_provenance_path(
    output_filename: str | os.PathLike[str],
) -> Path:
    return Path(f"{output_filename}.page-provenance.json")


def merged_page_provenance_path(
    merged_filename: str | os.PathLike[str],
) -> Path:
    return Path(f"{merged_filename}.page-provenance.json")


def docling_page_sentinel(pdf_sha256: str) -> str:
    """Return a deterministic token that cannot be ordinary Markdown syntax."""

    if not isinstance(pdf_sha256, str) or len(pdf_sha256) != 64:
        raise ValueError("PDF digest is invalid")
    return f"{DOCLING_SENTINEL_PREFIX}{pdf_sha256.upper()}"


def _trimmed_partition(
    raw_parts: Sequence[tuple[str, int]],
) -> tuple[str, list[dict]]:
    """Strip a concatenated render and retain clipped UTF-8 page intervals."""

    raw = "".join(text for text, _page in raw_parts)
    markdown = raw.strip()
    if not markdown:
        return markdown, []
    leading_chars = len(raw) - len(raw.lstrip())
    trailing_char_end = len(raw.rstrip())
    trim_start = len(raw[:leading_chars].encode("utf-8"))
    trim_end = len(raw[:trailing_char_end].encode("utf-8"))
    ranges: list[dict] = []
    cursor = 0
    for text, page_number in raw_parts:
        end = cursor + len(text.encode("utf-8"))
        start_clipped = max(cursor, trim_start)
        end_clipped = min(end, trim_end)
        if start_clipped < end_clipped:
            ranges.append(
                {
                    "byte_start": start_clipped - trim_start,
                    "byte_end": end_clipped - trim_start,
                    "page_number": page_number,
                    "candidate_pages": [page_number],
                    "method": "direct",
                    "native_id": None,
                    "kind": None,
                    "residual_reason": None,
                }
            )
        cursor = end
    return markdown, ranges


def docling_markdown_with_page_ranges(
    paginated_markdown: str,
    *,
    sentinel: str,
    expected_page_count: int,
) -> tuple[str, list[dict]]:
    """Remove Docling's transient page token without another render pass."""

    if type(expected_page_count) is not int or expected_page_count < 1:
        raise ValueError("expected page count must be a positive integer")
    occurrences = paginated_markdown.count(sentinel)
    expected_transitions = expected_page_count - 1
    if occurrences > expected_transitions:
        raise ValueError("Docling page sentinel collides with rendered content")
    parts = paginated_markdown.split(sentinel)
    markdown = "".join(parts).strip()
    if not markdown:
        return markdown, []
    if occurrences != expected_transitions:
        return markdown, []
    return _trimmed_partition(
        [(part, page_number) for page_number, part in enumerate(parts, start=1)]
    )


def marker_markdown_with_page_ranges(
    cleaned_paginated_markdown: str,
    *,
    expected_page_count: int,
    separator: str = MARKER_PAGE_SEPARATOR,
) -> tuple[str, list[dict]]:
    """Remove ordered Marker pagination tokens with an exact state machine."""

    if type(expected_page_count) is not int or expected_page_count < 1:
        raise ValueError("expected page count must be a positive integer")
    cursor = 0
    raw_parts: list[tuple[str, int]] = []
    for page_id in range(expected_page_count):
        marker = f"{{{page_id}}}"
        separator_start = cleaned_paginated_markdown.find(separator, cursor)
        marker_start = separator_start - len(marker)
        if separator_start < 0:
            raise ValueError("Marker page token inventory is incomplete")
        if (
            marker_start < cursor
            or cleaned_paginated_markdown[marker_start:separator_start] != marker
        ):
            raise ValueError("Marker page token is missing or out of order")
        if page_id == 0:
            if cleaned_paginated_markdown[:marker_start]:
                raise ValueError("Marker first page token is not leading")
        else:
            previous = cleaned_paginated_markdown[cursor:marker_start]
            raw_parts.append((previous, page_id))
        cursor = separator_start + len(separator)
        if cleaned_paginated_markdown[cursor : cursor + 2] == "\n\n":
            cursor += 2
    if separator in cleaned_paginated_markdown[cursor:]:
        raise ValueError("Marker page separator appears outside an expected token")
    final_part = cleaned_paginated_markdown[cursor:]
    if not final_part and raw_parts and raw_parts[-1][0].endswith("\n\n"):
        previous, page_number = raw_parts[-1]
        raw_parts[-1] = (previous[:-2], page_number)
    raw_parts.append((final_part, expected_page_count))
    return _trimmed_partition(raw_parts)


def _canonical_source_range(value: Mapping) -> dict:
    return {
        "byte_start": value.get("byte_start"),
        "byte_end": value.get("byte_end"),
        "page_number": value.get("page_number"),
        "candidate_pages": list(value.get("candidate_pages") or []),
        "method": value.get("method"),
        "native_id": value.get("native_id"),
        "kind": value.get("kind"),
        "residual_reason": value.get("residual_reason"),
    }


def partition_source_ranges(
    markdown_bytes: bytes,
    evidence_ranges: Sequence[Mapping],
    *,
    residual_reason: str,
) -> list[dict]:
    """Return a gap-free partition while preserving exact native evidence."""

    size = len(markdown_bytes)
    if size < 1:
        raise ValueError("source Markdown is empty")
    ordered = [_canonical_source_range(item) for item in evidence_ranges]
    cursor = 0
    partition: list[dict] = []
    for item in ordered:
        start = item["byte_start"]
        end = item["byte_end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= size:
            raise ValueError("source page range is out of bounds")
        if start < cursor:
            raise ValueError("source page ranges overlap or are reversed")
        if start > cursor:
            partition.append(
                {
                    "byte_start": cursor,
                    "byte_end": start,
                    "page_number": None,
                    "candidate_pages": [],
                    "method": None,
                    "native_id": None,
                    "kind": None,
                    "residual_reason": residual_reason,
                }
            )
        partition.append(item)
        cursor = end
    if cursor < size:
        partition.append(
            {
                "byte_start": cursor,
                "byte_end": size,
                "page_number": None,
                "candidate_pages": [],
                "method": None,
                "native_id": None,
                "kind": None,
                "residual_reason": residual_reason,
            }
        )
    return partition


def build_source_page_provenance_bytes(
    *,
    source: str,
    pdf_sha256: str,
    native_bytes: bytes,
    markdown_bytes: bytes,
    expected_page_count: int,
    extractor_versions: Mapping[str, str],
    options: Mapping[str, object],
    evidence_ranges: Sequence[Mapping],
    residual_reason: str,
) -> bytes:
    """Build and validate a canonical source provenance record."""

    partition = partition_source_ranges(
        markdown_bytes,
        evidence_ranges,
        residual_reason=residual_reason,
    )
    core = {
        "schema": SOURCE_PAGE_PROVENANCE_SCHEMA,
        "contract_version": SOURCE_PAGE_PROVENANCE_CONTRACT_VERSION,
        "source": source,
        "pdf_sha256": pdf_sha256,
        "native_artifact_sha256": _sha256(native_bytes),
        "markdown_sha256": _sha256(markdown_bytes),
        "markdown_size_bytes": len(markdown_bytes),
        "expected_page_count": expected_page_count,
        "extractor_versions": dict(sorted(extractor_versions.items())),
        "options": dict(sorted(options.items())),
        "ranges": partition,
    }
    payload = {**core, "record_sha256": _sha256(_canonical_json_bytes(core))}
    serialized = _canonical_json_bytes(payload) + b"\n"
    validate_source_page_provenance_bytes(
        serialized,
        source=source,
        pdf_sha256=core["pdf_sha256"],
        native_sha256=core["native_artifact_sha256"],
        markdown_sha256=core["markdown_sha256"],
    )
    return serialized


def validate_source_page_provenance_bytes(
    raw: bytes,
    *,
    source: str | None = None,
    pdf_sha256: str | None = None,
    native_sha256: str | None = None,
    markdown_sha256: str | None = None,
) -> dict:
    """Reject stale, malformed, or non-partitioning source page evidence."""

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("source page provenance must be a JSON object")
    expected_keys = {
        "schema",
        "contract_version",
        "source",
        "pdf_sha256",
        "native_artifact_sha256",
        "markdown_sha256",
        "markdown_size_bytes",
        "expected_page_count",
        "extractor_versions",
        "options",
        "ranges",
        "record_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("source page provenance fields are invalid")
    if payload["schema"] != SOURCE_PAGE_PROVENANCE_SCHEMA:
        raise ValueError("source page provenance schema is invalid")
    if payload["contract_version"] != SOURCE_PAGE_PROVENANCE_CONTRACT_VERSION:
        raise ValueError("source page provenance contract is invalid")
    if source is not None and payload["source"] != source:
        raise ValueError("source page provenance source is invalid")
    expected_count = payload["expected_page_count"]
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("source page provenance page count is invalid")
    for field, expected in (
        ("pdf_sha256", pdf_sha256),
        ("native_artifact_sha256", native_sha256),
        ("markdown_sha256", markdown_sha256),
    ):
        actual = payload[field]
        if not _is_sha256(actual):
            raise ValueError(f"source page provenance {field} is invalid")
        if expected is not None and actual != expected:
            raise ValueError(f"source page provenance {field} mismatch")
    ranges = payload["ranges"]
    markdown_size = payload["markdown_size_bytes"]
    if type(markdown_size) is not int or markdown_size < 1:
        raise ValueError("source page provenance Markdown size is invalid")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("source page provenance ranges are invalid")
    cursor = 0
    for item in ranges:
        if not isinstance(item, Mapping) or set(item) != {
            "byte_start",
            "byte_end",
            "page_number",
            "candidate_pages",
            "method",
            "native_id",
            "kind",
            "residual_reason",
        }:
            raise ValueError("source page provenance range fields are invalid")
        if item["byte_start"] != cursor:
            raise ValueError("source page provenance ranges do not partition Markdown")
        end = item["byte_end"]
        if type(end) is not int or end <= cursor:
            raise ValueError("source page provenance range bounds are invalid")
        page = item["page_number"]
        candidates = item["candidate_pages"]
        if not isinstance(candidates, list) or candidates != list(dict.fromkeys(candidates)):
            raise ValueError("source page provenance candidates are invalid")
        if any(type(candidate) is not int or not 1 <= candidate <= expected_count for candidate in candidates):
            raise ValueError("source page provenance candidate page is invalid")
        if page is None:
            if item["method"] is not None or not item["residual_reason"]:
                raise ValueError("source page provenance residual is invalid")
        elif (
            type(page) is not int
            or not 1 <= page <= expected_count
            or page not in candidates
            or item["method"] not in {"direct", "native_start_page"}
            or item["residual_reason"] is not None
        ):
            raise ValueError("source page provenance direct evidence is invalid")
        cursor = end
    if cursor != markdown_size:
        raise ValueError("source page provenance ranges do not cover Markdown")
    core = {key: payload[key] for key in expected_keys - {"record_sha256"}}
    if payload["record_sha256"] != _sha256(_canonical_json_bytes(core)):
        raise ValueError("source page provenance record digest mismatch")
    return payload


def load_source_page_provenance(
    output_filename: str | os.PathLike[str],
    *,
    manifest: Mapping,
) -> dict:
    """Load a source map already bound by its native manifest."""

    path = source_page_provenance_path(output_filename)
    if manifest.get("page_provenance_filename") != path.name:
        raise ValueError("source page provenance filename is invalid")
    raw = path.read_bytes()
    if manifest.get("page_provenance_size_bytes") != len(raw):
        raise ValueError("source page provenance size mismatch")
    if manifest.get("page_provenance_sha256") != _sha256(raw):
        raise ValueError("source page provenance digest mismatch")
    payload = validate_source_page_provenance_bytes(
        raw,
        source=manifest.get("source"),
        pdf_sha256=manifest.get("pdf_sha256"),
        native_sha256=manifest.get("native_sha256"),
        markdown_sha256=manifest.get("markdown_sha256"),
    )
    if payload.get("markdown_size_bytes") != Path(output_filename).stat().st_size:
        raise ValueError("source page provenance Markdown size mismatch")
    return payload


def _evidence_digest(value: object) -> str:
    return _sha256(_canonical_json_bytes(value))


def _page_evidence_for_span(
    source_ranges: Sequence[Mapping],
    source_range_starts: Sequence[int],
    start: int,
    end: int,
) -> list[tuple[dict, int, int]]:
    """Intersect one byte span with a pre-indexed contiguous source map."""

    evidence = []
    index = max(0, bisect_right(source_range_starts, start) - 1)
    for range_index in range(index, len(source_ranges)):
        item = source_ranges[range_index]
        overlap_start = max(start, item["byte_start"])
        overlap_end = min(end, item["byte_end"])
        if overlap_start < overlap_end:
            evidence.append((item, overlap_start, overlap_end))
        if item["byte_start"] >= end:
            break
    return evidence


def _nearest_page_anchors(
    ranges: Sequence[Mapping],
) -> tuple[list[tuple[int, int] | None], list[tuple[int, int] | None]]:
    """Compute nearest page anchors for every range in two linear passes."""

    previous: list[tuple[int, int] | None] = [None] * len(ranges)
    anchor: tuple[int, int] | None = None
    for index, item in enumerate(ranges):
        if anchor is not None:
            previous[index] = (anchor[0], max(0, item["byte_start"] - anchor[1]))
        if type(item.get("page_number")) is int:
            anchor = (item["page_number"], item["byte_end"])

    following: list[tuple[int, int] | None] = [None] * len(ranges)
    anchor = None
    for index in range(len(ranges) - 1, -1, -1):
        item = ranges[index]
        if anchor is not None:
            following[index] = (anchor[0], max(0, anchor[1] - item["byte_end"]))
        if type(item.get("page_number")) is int:
            anchor = (item["page_number"], item["byte_start"])
    return previous, following


def _set_resolved_range(
    item: dict,
    *,
    page_number: int,
    candidate_pages: Sequence[int],
    method: str,
    evidence_identity: object,
) -> None:
    item["page_number"] = page_number
    item["candidate_pages"] = list(dict.fromkeys(candidate_pages))
    item["method"] = method
    item["evidence_digest"] = _evidence_digest(evidence_identity)


def project_merged_page_ranges(
    *,
    merged_bytes: bytes,
    audit: Sequence[Mapping],
    source_maps: Mapping[str, Mapping],
    page_candidate_regions: Sequence[Mapping] = (),
    transformation_events: Sequence[Mapping] = (),
) -> list[dict]:
    """Project the exact merge audit through source maps with interval joins."""

    source_indices = {
        source: (
            source_map["ranges"],
            [item["byte_start"] for item in source_map["ranges"]],
        )
        for source, source_map in source_maps.items()
    }
    emphasis_boundaries: dict[str, str] = {}
    for event in transformation_events:
        if (
            isinstance(event, Mapping)
            and event.get("operation") == "native_emphasis_projection"
            and event.get("audit_span_emitted") is True
            and event.get("boundary") in {"open", "close"}
            and isinstance(event.get("transformation_id"), str)
        ):
            transformation_id = event["transformation_id"]
            boundary = event["boundary"]
            if (
                transformation_id in emphasis_boundaries
                and emphasis_boundaries[transformation_id] != boundary
            ):
                raise ValueError("native emphasis ownership is inconsistent")
            emphasis_boundaries[transformation_id] = boundary

    ranges: list[dict] = []
    cursor = 0
    for audit_index, entry in enumerate(audit):
        output_start = entry.get("output_byte_start")
        output_end = entry.get("output_byte_end")
        if (
            type(output_start) is not int
            or type(output_end) is not int
            or output_start != cursor
            or not output_start < output_end <= len(merged_bytes)
        ):
            raise ValueError("merge audit does not partition output for page projection")
        source = entry.get("source")
        if source == "deterministic_markup":
            ranges.append(
                {
                    "byte_start": output_start,
                    "byte_end": output_end,
                    "page_number": None,
                    "candidate_pages": [],
                    "method": None,
                    "source": source,
                    "operation": entry.get("transformation"),
                    "region_id": entry.get("region_id"),
                    "_transformation_id": entry.get("transformation_id"),
                    "evidence_digest": None,
                    "_publication_text": False,
                    "_candidate_votes": {},
                    "_candidate_evidence": [],
                }
            )
        else:
            source_map = source_maps.get(source)
            if not isinstance(source_map, Mapping):
                raise ValueError(f"source page map is unavailable for {source!r}")
            source_start = entry.get("source_byte_start")
            source_end = entry.get("source_byte_end")
            if type(source_start) is not int or type(source_end) is not int:
                raise ValueError("merge audit source range is invalid for page projection")
            source_ranges, source_range_starts = source_indices[source]
            evidence = _page_evidence_for_span(
                source_ranges,
                source_range_starts,
                source_start,
                source_end,
            )
            if not evidence:
                raise ValueError("merge audit source range has no page-map partition")
            for source_range, overlap_start, overlap_end in evidence:
                relative_start = overlap_start - source_start
                relative_end = overlap_end - source_start
                item = {
                    "byte_start": output_start + relative_start,
                    "byte_end": output_start + relative_end,
                    "page_number": source_range["page_number"],
                    "candidate_pages": list(source_range["candidate_pages"]),
                    "method": source_range["method"],
                    "source": source,
                    "operation": None,
                    "region_id": entry.get("region_id"),
                    "_transformation_id": None,
                    "evidence_digest": None,
                    "_publication_text": True,
                    "_candidate_votes": {},
                    "_candidate_evidence": [],
                }
                if item["page_number"] is not None:
                    item["evidence_digest"] = _evidence_digest(
                        {
                            "source_record_sha256": source_map["record_sha256"],
                            "source_byte_start": source_range["byte_start"],
                            "source_byte_end": source_range["byte_end"],
                            "native_id": source_range.get("native_id"),
                        }
                    )
                ranges.append(item)
        cursor = output_end
    if cursor != len(merged_bytes):
        raise ValueError("merge audit does not cover output for page projection")

    candidate_regions = {
        item.get("region_id"): item
        for item in page_candidate_regions
        if isinstance(item, Mapping) and isinstance(item.get("region_id"), str)
    }
    for item in ranges:
        if item["page_number"] is not None or not item["_publication_text"]:
            continue
        region = candidate_regions.get(item.get("region_id"))
        candidate_evidence = []
        votes: Counter[int] = Counter()
        if isinstance(region, Mapping):
            for candidate in region.get("candidates", ()):
                if not isinstance(candidate, Mapping):
                    continue
                source_map = source_maps.get(candidate.get("source"))
                start = candidate.get("source_byte_start")
                end = candidate.get("source_byte_end")
                if (
                    not isinstance(source_map, Mapping)
                    or type(start) is not int
                    or type(end) is not int
                ):
                    continue
                pages = []
                page_intervals = []
                source_ranges, source_range_starts = source_indices[
                    candidate.get("source")
                ]
                for source_range, overlap_start, overlap_end in _page_evidence_for_span(
                    source_ranges,
                    source_range_starts,
                    start,
                    end,
                ):
                    if type(source_range.get("page_number")) is int:
                        pages.append(source_range["page_number"])
                        page_intervals.append(
                            {
                                "page_number": source_range["page_number"],
                                "source_byte_start": overlap_start,
                                "source_byte_end": overlap_end,
                            }
                        )
                pages = list(dict.fromkeys(pages))
                votes.update(pages)
                candidate_evidence.append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "source": candidate.get("source"),
                        "source_byte_start": start,
                        "source_byte_end": end,
                        "pages": pages,
                        "page_intervals": page_intervals,
                    }
                )
        item["_candidate_votes"] = dict(votes)
        item["_candidate_evidence"] = candidate_evidence
        unanimous = set(votes)
        if len(unanimous) == 1 and candidate_evidence:
            page = next(iter(unanimous))
            _set_resolved_range(
                item,
                page_number=page,
                candidate_pages=[page],
                method="aligned_agreement",
                evidence_identity={
                    "region_id": item.get("region_id"),
                    "candidate_evidence": candidate_evidence,
                },
            )

    previous_anchors, following_anchors = _nearest_page_anchors(ranges)
    for index, item in enumerate(ranges):
        if item["page_number"] is not None or item["source"] != "deterministic_markup":
            continue
        previous = previous_anchors[index]
        following = following_anchors[index]
        operation = item.get("operation")
        owner = None
        if operation == "native_emphasis_projection":
            boundary = emphasis_boundaries.get(item.get("_transformation_id"))
            if boundary == "open" and following is not None:
                owner = following[0]
            elif boundary == "close" and previous is not None:
                owner = previous[0]
        elif operation in _FOLLOWING_OWNER_OPERATIONS and following is not None:
            owner = following[0]
        elif operation in _PRECEDING_OWNER_OPERATIONS and previous is not None:
            owner = previous[0]
        elif (
            operation in _SURROUNDING_OWNER_OPERATIONS
            and previous is not None
            and following is not None
            and previous[0] == following[0]
        ):
            owner = previous[0]
        if owner is not None:
            _set_resolved_range(
                item,
                page_number=owner,
                candidate_pages=[owner],
                method="deterministic_owner",
                evidence_identity={
                    "operation": operation,
                    "owner_page": owner,
                },
            )

    previous_anchors, following_anchors = _nearest_page_anchors(ranges)
    for index, item in enumerate(ranges):
        if item["page_number"] is not None:
            continue
        previous = previous_anchors[index]
        following = following_anchors[index]
        item["_preceding_anchor"] = (
            None
            if previous is None
            else {"page_number": previous[0], "distance_bytes": previous[1]}
        )
        item["_following_anchor"] = (
            None
            if following is None
            else {"page_number": following[0], "distance_bytes": following[1]}
        )
        if previous is not None and following is not None and previous[0] == following[0]:
            _set_resolved_range(
                item,
                page_number=previous[0],
                candidate_pages=[previous[0]],
                method="aligned_agreement",
                evidence_identity={
                    "neighbor_page": previous[0],
                    "previous_distance": previous[1],
                    "following_distance": following[1],
                },
            )
        else:
            votes = Counter(item.get("_candidate_votes") or {})
            distances = {}
            for anchor in (previous, following):
                if anchor is not None:
                    votes[anchor[0]] += 1
                    distances[anchor[0]] = min(
                        distances.get(anchor[0], anchor[1]), anchor[1]
                    )
            item["candidate_pages"] = [
                page
                for page, _count in sorted(
                    votes.items(),
                    key=lambda pair: (-pair[1], distances.get(pair[0], 10**18), pair[0]),
                )
            ]
    coalesced = []
    residual_identity = (
        "candidate_pages",
        "source",
        "operation",
        "region_id",
        "_candidate_votes",
        "_candidate_evidence",
        "_preceding_anchor",
        "_following_anchor",
    )
    for item in ranges:
        previous = coalesced[-1] if coalesced else None
        if (
            previous is not None
            and previous["page_number"] is None
            and item["page_number"] is None
            and previous["_publication_text"] is True
            and item["_publication_text"] is True
            and previous["byte_end"] == item["byte_start"]
            and all(previous[field] == item[field] for field in residual_identity)
        ):
            previous["byte_end"] = item["byte_end"]
        else:
            coalesced.append(item)
    return coalesced


def _utf8_context_slice_with_bounds(
    raw: bytes, start: int, end: int
) -> tuple[str, int, int]:
    start = max(0, start)
    end = min(len(raw), end)
    while start < end:
        try:
            return raw[start:end].decode("utf-8"), start, end
        except UnicodeDecodeError as exc:
            if exc.start == 0:
                start += 1
            else:
                end -= 1
    return "", start, start


def _utf8_context_slice(raw: bytes, start: int, end: int) -> str:
    return _utf8_context_slice_with_bounds(raw, start, end)[0]


def bound_page_resolution_ranges(
    *,
    merged_bytes: bytes,
    projected_ranges: Sequence[Mapping],
    max_text_bytes: int,
) -> list[dict]:
    """Split only model-eligible residuals into exact bounded UTF-8 slices."""

    if type(max_text_bytes) is not int or max_text_bytes < 1:
        raise ValueError("page-resolution text limit is invalid")
    bounded = []
    for source_item in projected_ranges:
        item = dict(source_item)
        start = item["byte_start"]
        end = item["byte_end"]
        if (
            item.get("page_number") is not None
            or item.get("_publication_text") is not True
            or not item.get("candidate_pages")
            or end - start <= max_text_bytes
        ):
            bounded.append(item)
            continue
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + max_text_bytes, end)
            while (
                chunk_end > cursor
                and chunk_end < end
                and merged_bytes[chunk_end] & 0b11000000 == 0b10000000
            ):
                chunk_end -= 1
            if chunk_end == cursor:
                raise ValueError("page-resolution text limit splits a UTF-8 scalar")
            chunk = dict(item)
            chunk["byte_start"] = cursor
            chunk["byte_end"] = chunk_end
            merged_bytes[cursor:chunk_end].decode("utf-8")
            bounded.append(chunk)
            cursor = chunk_end
    return bounded


def build_page_resolution_batches(
    *,
    merged_bytes: bytes,
    projected_ranges: Sequence[Mapping],
    source_bytes: Mapping[str, bytes],
    max_ranges_per_batch: int,
    context_bytes: int,
    evidence_bytes_per_range: int,
) -> list[dict]:
    """Build bounded Luna choices from existing intervals and exact byte slices."""

    if (
        type(max_ranges_per_batch) is not int
        or max_ranges_per_batch < 1
        or type(context_bytes) is not int
        or context_bytes < 0
        or type(evidence_bytes_per_range) is not int
        or evidence_bytes_per_range < 1
    ):
        raise ValueError("page-resolution evidence limits are invalid")
    unresolved = []
    for item in projected_ranges:
        if item.get("page_number") is not None or item.get("_publication_text") is not True:
            continue
        choices = list(item.get("candidate_pages") or [])
        if not choices:
            continue
        start = item["byte_start"]
        end = item["byte_end"]
        if end - start > evidence_bytes_per_range:
            raise ValueError("page-resolution residual text exceeds its byte bound")
        range_id = _evidence_digest(
            {
                "byte_start": start,
                "byte_end": end,
                "source": item.get("source"),
                "operation": item.get("operation"),
            }
        )
        alternatives = []
        remaining = evidence_bytes_per_range
        for evidence in item.get("_candidate_evidence", ()):
            source = evidence.get("source")
            raw = source_bytes.get(source)
            if remaining <= 0 or raw is None:
                continue
            for interval in evidence.get("page_intervals", ()):
                source_start = interval.get("source_byte_start")
                source_end = interval.get("source_byte_end")
                page_number = interval.get("page_number")
                if (
                    remaining <= 0
                    or type(source_start) is not int
                    or type(source_end) is not int
                    or type(page_number) is not int
                ):
                    continue
                clipped_end = min(source_end, source_start + remaining)
                excerpt, excerpt_start, excerpt_end = _utf8_context_slice_with_bounds(
                    raw, source_start, clipped_end
                )
                excerpt_size = len(excerpt.encode("utf-8"))
                remaining -= excerpt_size
                alternatives.append(
                    {
                        "source": source,
                        "page_number": page_number,
                        "source_byte_start": excerpt_start,
                        "source_byte_end": excerpt_end,
                        "excerpt": excerpt,
                    }
                )
        preceding_context, preceding_start, preceding_end = (
            _utf8_context_slice_with_bounds(
                merged_bytes, start - context_bytes, start
            )
        )
        following_context, following_start, following_end = (
            _utf8_context_slice_with_bounds(
                merged_bytes, end, end + context_bytes
            )
        )
        unresolved.append(
            {
                "range_id": range_id,
                "byte_start": start,
                "byte_end": end,
                "text": merged_bytes[start:end].decode("utf-8"),
                "preceding_context": preceding_context,
                "preceding_context_byte_start": preceding_start,
                "preceding_context_byte_end": preceding_end,
                "following_context": following_context,
                "following_context_byte_start": following_start,
                "following_context_byte_end": following_end,
                "page_choices": choices,
                "alternative_evidence": alternatives,
                "preceding_anchor": item.get("_preceding_anchor"),
                "following_anchor": item.get("_following_anchor"),
            }
        )
    batches = []
    for offset in range(0, len(unresolved), max_ranges_per_batch):
        core = {"ranges": unresolved[offset : offset + max_ranges_per_batch]}
        batches.append(
            {"request_sha256": _evidence_digest(core), **core}
        )
    return batches


def finalize_merged_page_provenance_bytes(
    *,
    pdf_sha256: str,
    expected_page_count: int,
    merged_bytes: bytes,
    audit_sha256: str,
    merge_contract_id: str,
    source_map_sha256: Mapping[str, str],
    projected_ranges: Sequence[Mapping],
    llm_choices: Mapping[str, int] | None = None,
    llm_receipts: Sequence[Mapping] = (),
) -> tuple[bytes, dict]:
    """Require one valid primary page for every final Markdown byte range."""

    if type(expected_page_count) is not int or expected_page_count < 1:
        raise ValueError("merged page provenance page count is invalid")
    llm_choices = dict(llm_choices or {})
    global_pages = sorted(
        {
            item.get("page_number")
            for item in projected_ranges
            if type(item.get("page_number")) is int
        }
    )
    finalized = []
    for ordinal, source_item in enumerate(projected_ranges):
        item = dict(source_item)
        range_id = _evidence_digest(
            {
                "byte_start": item["byte_start"],
                "byte_end": item["byte_end"],
                "source": item.get("source"),
                "operation": item.get("operation"),
            }
        )
        candidates = list(item.get("candidate_pages") or [])
        choice = llm_choices.get(range_id)
        if item.get("page_number") is None:
            if (
                item.get("_publication_text") is True
                and type(choice) is int
                and choice in candidates
            ):
                _set_resolved_range(
                    item,
                    page_number=choice,
                    candidate_pages=candidates,
                    method="llm_selected",
                    evidence_identity={"range_id": range_id, "choice": choice},
                )
            else:
                fallback = candidates[0] if candidates else global_pages[0] if global_pages else 1
                _set_resolved_range(
                    item,
                    page_number=fallback,
                    candidate_pages=candidates or [fallback],
                    method="deterministic_fallback",
                    evidence_identity={
                        "range_id": range_id,
                        "ranked_candidates": candidates,
                        "global_evidence_pages": global_pages,
                    },
                )
        item["range_id"] = range_id
        item.pop("region_id", None)
        item.pop("_publication_text", None)
        item.pop("_candidate_votes", None)
        item.pop("_candidate_evidence", None)
        item.pop("_transformation_id", None)
        item.pop("_preceding_anchor", None)
        item.pop("_following_anchor", None)
        if not 1 <= item["page_number"] <= expected_page_count:
            raise ValueError("merged page provenance selected page is out of bounds")
        finalized.append(item)

    coalesced = []
    identity_fields = (
        "page_number",
        "candidate_pages",
        "method",
        "source",
        "operation",
        "evidence_digest",
    )
    for item in finalized:
        previous = coalesced[-1] if coalesced else None
        if previous is not None and previous["byte_end"] == item["byte_start"] and all(
            previous[field] == item[field] for field in identity_fields
        ):
            previous["byte_end"] = item["byte_end"]
        else:
            coalesced.append(item)
    cursor = 0
    for item in coalesced:
        if item["byte_start"] != cursor or item["byte_end"] <= cursor:
            raise ValueError("merged page provenance ranges do not partition output")
        cursor = item["byte_end"]
    if cursor != len(merged_bytes):
        raise ValueError("merged page provenance does not cover output")

    method_counts = Counter(item["method"] for item in coalesced)
    method_bytes = Counter()
    source_counts = Counter()
    source_bytes = Counter()
    for item in coalesced:
        size = item["byte_end"] - item["byte_start"]
        method_bytes[item["method"]] += size
        source_counts[item["source"]] += 1
        source_bytes[item["source"]] += size
    summary = {
        "range_counts_by_method": dict(sorted(method_counts.items())),
        "byte_counts_by_method": dict(sorted(method_bytes.items())),
        "range_counts_by_source": dict(sorted(source_counts.items())),
        "byte_counts_by_source": dict(sorted(source_bytes.items())),
    }
    core = {
        "schema": MERGED_PAGE_PROVENANCE_SCHEMA,
        "contract_version": MERGED_PAGE_PROVENANCE_CONTRACT_VERSION,
        "pdf_sha256": pdf_sha256,
        "merged_markdown_sha256": _sha256(merged_bytes),
        "merged_markdown_size_bytes": len(merged_bytes),
        "audit_sha256": audit_sha256,
        "merge_contract_id": merge_contract_id,
        "source_map_sha256": dict(sorted(source_map_sha256.items())),
        "expected_page_count": expected_page_count,
        "ranges": coalesced,
        "summary": summary,
        "llm_receipts": list(llm_receipts),
    }
    payload = {**core, "record_sha256": _sha256(_canonical_json_bytes(core))}
    return _canonical_json_bytes(payload) + b"\n", summary


def validate_merged_page_provenance_bytes(
    raw: bytes,
    *,
    pdf_sha256: str,
    merged_sha256: str,
    merged_size_bytes: int,
    audit_sha256: str,
    merge_contract_id: str,
    source_map_sha256: Mapping[str, str],
) -> dict:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("merged page provenance must be a JSON object")
    expected_keys = {
        "schema",
        "contract_version",
        "pdf_sha256",
        "merged_markdown_sha256",
        "merged_markdown_size_bytes",
        "audit_sha256",
        "merge_contract_id",
        "source_map_sha256",
        "expected_page_count",
        "ranges",
        "summary",
        "llm_receipts",
        "record_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("merged page provenance fields are invalid")
    core = {key: value for key, value in payload.items() if key != "record_sha256"}
    if (
        payload.get("schema") != MERGED_PAGE_PROVENANCE_SCHEMA
        or payload.get("contract_version") != MERGED_PAGE_PROVENANCE_CONTRACT_VERSION
        or payload.get("pdf_sha256") != pdf_sha256
        or payload.get("merged_markdown_sha256") != merged_sha256
        or payload.get("merged_markdown_size_bytes") != merged_size_bytes
        or payload.get("audit_sha256") != audit_sha256
        or payload.get("merge_contract_id") != merge_contract_id
        or payload.get("source_map_sha256") != dict(sorted(source_map_sha256.items()))
        or payload.get("record_sha256") != _sha256(_canonical_json_bytes(core))
    ):
        raise ValueError("merged page provenance binding is invalid")
    if not all(
        _is_sha256(value)
        for value in (
            payload["pdf_sha256"],
            payload["merged_markdown_sha256"],
            payload["audit_sha256"],
            payload["record_sha256"],
            *payload["source_map_sha256"].values(),
        )
    ):
        raise ValueError("merged page provenance digest is invalid")
    count = payload.get("expected_page_count")
    size = payload.get("merged_markdown_size_bytes")
    ranges = payload.get("ranges")
    if (
        type(count) is not int
        or count < 1
        or type(size) is not int
        or size < 1
        or not isinstance(ranges, list)
        or not ranges
    ):
        raise ValueError("merged page provenance shape is invalid")
    cursor = 0
    method_counts: Counter[str] = Counter()
    method_bytes: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_bytes: Counter[str] = Counter()
    for item in ranges:
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "byte_start",
                "byte_end",
                "page_number",
                "candidate_pages",
                "method",
                "source",
                "operation",
                "evidence_digest",
                "range_id",
            }
            or item.get("byte_start") != cursor
            or type(item.get("byte_end")) is not int
            or item["byte_end"] <= cursor
            or type(item.get("page_number")) is not int
            or not 1 <= item["page_number"] <= count
            or item.get("method")
            not in {
                "direct",
                "native_start_page",
                "deterministic_owner",
                "aligned_agreement",
                "llm_selected",
                "deterministic_fallback",
            }
        ):
            raise ValueError("merged page provenance range is invalid")
        candidates = item.get("candidate_pages")
        if (
            not isinstance(candidates, list)
            or candidates != list(dict.fromkeys(candidates))
            or item["page_number"] not in candidates
            or any(type(page) is not int or not 1 <= page <= count for page in candidates)
        ):
            raise ValueError("merged page provenance range candidates are invalid")
        if (
            not isinstance(item.get("source"), str)
            or not item["source"]
            or (
                item.get("operation") is not None
                and not isinstance(item["operation"], str)
            )
            or not _is_sha256(item.get("evidence_digest"))
            or not _is_sha256(item.get("range_id"))
        ):
            raise ValueError("merged page provenance range evidence is invalid")
        range_size = item["byte_end"] - cursor
        method_counts[item["method"]] += 1
        method_bytes[item["method"]] += range_size
        source_counts[item["source"]] += 1
        source_bytes[item["source"]] += range_size
        cursor = item["byte_end"]
    if cursor != size:
        raise ValueError("merged page provenance ranges do not cover output")
    expected_summary = {
        "range_counts_by_method": dict(sorted(method_counts.items())),
        "byte_counts_by_method": dict(sorted(method_bytes.items())),
        "range_counts_by_source": dict(sorted(source_counts.items())),
        "byte_counts_by_source": dict(sorted(source_bytes.items())),
    }
    if payload.get("summary") != expected_summary:
        raise ValueError("merged page provenance summary is invalid")
    if not isinstance(payload.get("llm_receipts"), list):
        raise ValueError("merged page provenance LLM receipts are invalid")
    return payload
