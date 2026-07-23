"""Compact, positive-only native PDF italic evidence for GROBID and Docling."""

from __future__ import annotations

import json
import io
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


NATIVE_STYLE_SCHEMA = "pdfx-native-style"
NATIVE_STYLE_CONTRACT_VERSION = "native-style-v1"
NATIVE_STYLE_MEDIA_TYPE = "application/vnd.alliance.pdfx-native-style+json"
_FONT_STYLE_TOKEN = re.compile(r"[a-z]+")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def unavailable_native_style_bytes(source: str, reason: str) -> bytes:
    """Return a valid sidecar that makes no typography claim."""

    return _json_bytes(
        {
            "schema": NATIVE_STYLE_SCHEMA,
            "contract_version": NATIVE_STYLE_CONTRACT_VERSION,
            "source": source,
            "status": "unavailable",
            "reason": reason,
            "pages": [],
        }
    )


def font_name_is_explicit_italic(font_name: str) -> bool:
    """Recognize explicit PostScript-style italic/oblique name components.

    This intentionally does not infer style from font family, slant geometry, or
    publication-specific names. Missing/unknown metadata makes no claim.
    """

    name = font_name.strip().lstrip("/")
    if "+" in name:
        prefix, remainder = name.split("+", 1)
        if len(prefix) == 6 and prefix.isalpha():
            name = remainder
    tokens = _FONT_STYLE_TOKEN.findall(name.casefold())
    return any(
        token in {"it", "ita", "italic", "italics", "oblique", "kursiv"}
        or token.startswith("italic")
        or token.startswith("oblique")
        for token in tokens
    )


@dataclass(frozen=True)
class _Cell:
    text: str
    italic: bool
    style: str
    left: float
    top: float
    right: float
    bottom: float

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)


def _same_line(previous: _Cell, current: _Cell) -> bool:
    previous_center = (previous.top + previous.bottom) / 2
    current_center = (current.top + current.bottom) / 2
    tolerance = max(previous.height, current.height) * 0.55
    return abs(previous_center - current_center) <= tolerance


def _cell_separator(previous: _Cell, current: _Cell) -> str:
    if not previous.text or not current.text:
        return ""
    if previous.text[-1].isspace() or current.text[0].isspace():
        return ""
    average_height = max((previous.height + current.height) / 2, 1.0)
    return "" if current.left - previous.right <= average_height * 0.12 else " "


def _line_record(
    *,
    native_id: str,
    cells: Iterable[_Cell],
) -> dict | None:
    cells = tuple(cell for cell in cells if cell.text)
    if not cells:
        return None
    parts: list[str] = []
    spans: list[dict] = []
    active_start: int | None = None
    active_styles: list[str] = []

    def length() -> int:
        return sum(len(part) for part in parts)

    for index, cell in enumerate(cells):
        cell_is_italic = cell.italic or (
            cell.text.isspace()
            and index > 0
            and index + 1 < len(cells)
            and cells[index - 1].italic
            and cells[index + 1].italic
        )
        if not cell_is_italic and active_start is not None:
            spans.append(
                {
                    "start": active_start,
                    "end": length(),
                    "styles": sorted(set(active_styles)),
                }
            )
            active_start = None
            active_styles = []
        if index:
            separator = _cell_separator(cells[index - 1], cell)
            parts.append(separator)
        if cell_is_italic and active_start is None:
            active_start = length()
            active_styles = []
        parts.append(cell.text)
        if cell_is_italic and cell.style:
            active_styles.append(cell.style)
    if active_start is not None:
        spans.append(
            {
                "start": active_start,
                "end": length(),
                "styles": sorted(set(active_styles)),
            }
        )
    text = "".join(parts)
    spans = [span for span in spans if span["end"] > span["start"]]
    return {
        "native_id": native_id,
        "text": text,
        "italic_spans": spans,
    }


def docling_native_style_bytes(conversion_result) -> bytes:
    """Serialize materialized, font-preserving Docling PDF word cells."""

    pages = []
    for page_index, page in enumerate(conversion_result.pages, 1):
        parsed_page = getattr(page, "parsed_page", None)
        # Docling 2.113 computes character cells for layout work but does not
        # materialize them in the retained SegmentedPdfPage. Word cells are the
        # smallest retained PDF cells that carry font names; OCR TextCells have
        # no font_name and therefore make no typography claim below.
        raw_cells = () if parsed_page is None else parsed_page.word_cells
        cells: list[_Cell] = []
        for raw_cell in raw_cells:
            font_name = getattr(raw_cell, "font_name", None)
            if not isinstance(font_name, str) or not font_name.strip():
                continue
            text = getattr(raw_cell, "text", None)
            rect = getattr(raw_cell, "rect", None)
            if not isinstance(text, str) or not text or rect is None:
                continue
            bbox = rect.to_bounding_box()
            vertical_start = min(float(bbox.t), float(bbox.b))
            vertical_end = max(float(bbox.t), float(bbox.b))
            cells.append(
                _Cell(
                    text=text,
                    italic=font_name_is_explicit_italic(font_name),
                    style=font_name,
                    left=float(bbox.l),
                    top=vertical_start,
                    right=float(bbox.r),
                    bottom=vertical_end,
                )
            )
        grouped: list[list[_Cell]] = []
        for cell in cells:
            if grouped and _same_line(grouped[-1][-1], cell):
                grouped[-1].append(cell)
            else:
                grouped.append([cell])
        lines = [
            record
            for line_index, group in enumerate(grouped)
            if (
                record := _line_record(
                    native_id=f"docling-page-{page_index}-line-{line_index}",
                    cells=group,
                )
            )
            is not None
        ]
        pages.append({"page_no": page_index, "lines": lines})
    return _json_bytes(
        {
            "schema": NATIVE_STYLE_SCHEMA,
            "contract_version": NATIVE_STYLE_CONTRACT_VERSION,
            "source": "docling",
            "status": "available",
            "pages": pages,
        }
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _alto_native_style_source(source) -> bytes:
    styles: dict[str, tuple[bool, str]] = {}
    pages = []
    current_page_no = None
    current_lines = None
    for event, node in ET.iterparse(source, events=("start", "end")):
        local = _local_name(node.tag)
        if event == "start" and local == "Page":
            page_index = len(pages) + 1
            try:
                current_page_no = int(
                    node.attrib.get("PHYSICAL_IMG_NR", page_index)
                )
            except ValueError:
                current_page_no = page_index
            current_lines = []
            continue
        if event != "end":
            continue
        if local == "TextStyle":
            style_id = node.attrib.get("ID")
            if style_id:
                style_value = node.attrib.get("FONTSTYLE", "")
                style_tokens = set(style_value.casefold().split())
                styles[style_id] = ("italics" in style_tokens, style_value)
            node.clear()
            continue
        if local == "TextLine" and current_lines is not None:
            cells = []
            for child in node:
                child_local = _local_name(child.tag)
                if child_local == "SP":
                    text = " "
                    italic = False
                    style = ""
                elif child_local == "String":
                    text = child.attrib.get("CONTENT", "")
                    referenced = child.attrib.get("STYLEREFS", "").split()
                    resolved = [styles[ref] for ref in referenced if ref in styles]
                    italic = any(item[0] for item in resolved)
                    style = " ".join(item[1] for item in resolved if item[1])
                else:
                    continue
                try:
                    left = float(child.attrib.get("HPOS", 0))
                    top = float(child.attrib.get("VPOS", 0))
                    width = float(child.attrib.get("WIDTH", 0))
                    height = float(child.attrib.get("HEIGHT", 0))
                except ValueError:
                    left = top = width = height = 0.0
                cells.append(
                    _Cell(
                        text=text,
                        italic=italic,
                        style=style,
                        left=left,
                        top=top,
                        right=left + width,
                        bottom=top + height,
                    )
                )
            record = _line_record(
                native_id=(
                    node.attrib.get("ID")
                    or f"grobid-page-{current_page_no}-line-{len(current_lines)}"
                ),
                cells=cells,
            )
            if record is not None:
                current_lines.append(record)
            node.clear()
            continue
        if local == "Page" and current_lines is not None:
            pages.append({"page_no": current_page_no, "lines": current_lines})
            current_page_no = None
            current_lines = None
            node.clear()
            continue
        if local not in {"String", "SP"}:
            node.clear()
    return _json_bytes(
        {
            "schema": NATIVE_STYLE_SCHEMA,
            "contract_version": NATIVE_STYLE_CONTRACT_VERSION,
            "source": "grobid",
            "status": "available",
            "pages": pages,
        }
    )


def _alto_native_style_bytes(alto_bytes: bytes) -> bytes:
    return _alto_native_style_source(io.BytesIO(alto_bytes))


def grobid_native_style_bytes(
    pdf_path: str | Path,
    *,
    pdfalto_path: str | Path,
    timeout_seconds: float,
) -> bytes:
    """Run GROBID's pinned PDFALTO and return compact explicit style evidence."""

    with tempfile.TemporaryDirectory(prefix="pdfx-pdfalto-") as directory:
        output_path = Path(directory) / "document.alto.xml"
        subprocess.run(
            [
                str(pdfalto_path),
                "-readingOrder",
                "-noImage",
                "-fullFontName",
                str(pdf_path),
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
        return _alto_native_style_source(output_path)


def validate_native_style_bytes(source: str, payload: bytes) -> dict:
    """Validate the compact style contract and return its decoded object."""

    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("native style sidecar must be an object")
    if value.get("schema") != NATIVE_STYLE_SCHEMA:
        raise ValueError("native style schema is invalid")
    if value.get("contract_version") != NATIVE_STYLE_CONTRACT_VERSION:
        raise ValueError("native style contract version is invalid")
    if value.get("source") != source:
        raise ValueError("native style source is invalid")
    if value.get("status") not in {"available", "unavailable"}:
        raise ValueError("native style status is invalid")
    pages = value.get("pages")
    if not isinstance(pages, list):
        raise ValueError("native style pages are invalid")
    previous_page = 0
    for page in pages:
        if not isinstance(page, dict) or type(page.get("page_no")) is not int:
            raise ValueError("native style page is invalid")
        if page["page_no"] <= previous_page or not isinstance(page.get("lines"), list):
            raise ValueError("native style page order is invalid")
        previous_page = page["page_no"]
        for line in page["lines"]:
            if not isinstance(line, dict) or not isinstance(line.get("text"), str):
                raise ValueError("native style line is invalid")
            if not isinstance(line.get("native_id"), str) or not line["native_id"]:
                raise ValueError("native style line identity is invalid")
            previous_end = 0
            for span in line.get("italic_spans", ()):
                if (
                    not isinstance(span, dict)
                    or type(span.get("start")) is not int
                    or type(span.get("end")) is not int
                    or not previous_end <= span["start"] < span["end"] <= len(line["text"])
                    or not isinstance(span.get("styles"), list)
                ):
                    raise ValueError("native style span is invalid")
                previous_end = span["end"]
    return value
