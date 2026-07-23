import json

import pytest

from app.services.native_style import (
    _alto_native_style_bytes,
    docling_native_style_bytes,
    font_name_is_explicit_italic,
    validate_native_style_bytes,
)


class _Box:
    def __init__(self, left, top, right, bottom):
        self.l = left
        self.t = top
        self.r = right
        self.b = bottom


class _Rect:
    def __init__(self, left, top, right, bottom):
        self.box = _Box(left, top, right, bottom)

    def to_bounding_box(self):
        return self.box


class _Cell:
    def __init__(self, text, font_name, left, top, right, bottom):
        self.text = text
        self.font_name = font_name
        self.rect = _Rect(left, top, right, bottom)


class _ParsedPage:
    def __init__(self, cells):
        self.char_cells = []
        self.word_cells = cells


class _Page:
    def __init__(self, cells):
        self.parsed_page = _ParsedPage(cells)


class _Result:
    def __init__(self, pages):
        self.pages = pages


@pytest.mark.parametrize(
    ("font_name", "expected"),
    [
        ("/ABCDEF+Times-Italic", True),
        ("TimesNewRomanPS-ItalicMT", True),
        ("Arial-Oblique", True),
        ("MinionPro-It", True),
        ("/ABCDEF+Times-Roman", False),
        ("Helvetica-Bold", False),
        ("", False),
    ],
)
def test_font_name_classifier_uses_general_explicit_style_components(
    font_name, expected
):
    assert font_name_is_explicit_italic(font_name) is expected


def test_docling_sidecar_uses_materialized_word_cells_and_ignores_cells_without_fonts():
    result = _Result(
        [
            _Page(
                [
                    _Cell("Gene", "Times-Roman", 0, 0, 30, 10),
                    _Cell("dpp", "Times-Italic", 36, 0, 55, 10),
                    _Cell("works.", "Times-Roman", 61, 0, 100, 10),
                    _Cell("Ignored OCR", "", 0, 20, 70, 30),
                ]
            )
        ]
    )

    payload = validate_native_style_bytes(
        "docling", docling_native_style_bytes(result)
    )

    line = payload["pages"][0]["lines"][0]
    assert line["text"] == "Gene dpp works."
    assert line["text"][line["italic_spans"][0]["start"] : line["italic_spans"][0]["end"]] == "dpp"
    assert line["italic_spans"][0]["styles"] == ["Times-Italic"]


def test_grobid_alto_sidecar_uses_explicit_style_refs_and_merges_italic_words():
    alto = b"""<alto><Styles>
      <TextStyle ID="normal" FONTFAMILY="Times"/>
      <TextStyle ID="italic" FONTFAMILY="Times-Italic" FONTSTYLE="italics"/>
      <TextStyle ID="bolditalic" FONTSTYLE="bold italics"/>
    </Styles><Layout><Page PHYSICAL_IMG_NR="1"><PrintSpace><TextBlock>
      <TextLine ID="line-1">
        <String CONTENT="Gene" STYLEREFS="normal" HPOS="0" WIDTH="25" HEIGHT="10"/>
        <SP HPOS="25" WIDTH="5" HEIGHT="10"/>
        <String CONTENT="dpp" STYLEREFS="italic" HPOS="30" WIDTH="20" HEIGHT="10"/>
        <SP HPOS="50" WIDTH="5" HEIGHT="10"/>
        <String CONTENT="family" STYLEREFS="bolditalic" HPOS="55" WIDTH="35" HEIGHT="10"/>
      </TextLine>
    </TextBlock></PrintSpace></Page></Layout></alto>"""

    payload = validate_native_style_bytes("grobid", _alto_native_style_bytes(alto))
    line = payload["pages"][0]["lines"][0]

    assert line["text"] == "Gene dpp family"
    span = line["italic_spans"][0]
    assert line["text"][span["start"] : span["end"]] == "dpp family"


def test_native_style_validation_rejects_overlapping_or_out_of_range_spans():
    payload = {
        "schema": "pdfx-native-style",
        "contract_version": "native-style-v1",
        "source": "docling",
        "status": "available",
        "pages": [
            {
                "page_no": 1,
                "lines": [
                    {
                        "native_id": "line",
                        "text": "abc",
                        "italic_spans": [
                            {"start": 0, "end": 2, "styles": ["Italic"]},
                            {"start": 1, "end": 3, "styles": ["Italic"]},
                        ],
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="span"):
        validate_native_style_bytes("docling", json.dumps(payload).encode())
