import pytest
from app.services import marker_service
from app.services.marker_service import Marker


class DummyMarker(Marker):
    def extract(self, pdf_path, output_path):
        with open(output_path, "w") as f:
            f.write("Marker output")

def test_marker_extract(tmp_path):
    marker = DummyMarker()
    pdf_path = tmp_path / "test.pdf"
    output_path = tmp_path / "output.md"
    pdf_path.write_bytes(b"dummy pdf content")
    marker.extract(str(pdf_path), str(output_path))
    assert output_path.exists()
    assert output_path.read_text() == "Marker output"


def test_marker_model_cache_is_shared_across_converter_options(monkeypatch):
    marker_service._cached_models.clear()
    marker_service._cached_converters.clear()
    model_calls = []

    def fake_create_model_dict(device, dtype):
        model_calls.append((str(device), str(dtype)))
        return {"model": "fake"}

    class FakeConverter:
        def __init__(self, artifact_dict, config):
            self.artifact_dict = artifact_dict
            self.config = config

    monkeypatch.setattr(marker_service, "create_model_dict", fake_create_model_dict)
    monkeypatch.setattr(marker_service, "PdfConverter", FakeConverter)

    first = marker_service._get_converter("cuda", "float16", extract_images=True)
    second = marker_service._get_converter("cuda", "float16", extract_images=False)

    assert first is not second
    assert first.artifact_dict is second.artifact_dict
    assert first.config["paginate_output"] is True
    assert first.config["page_separator"] == marker_service.MARKER_PAGE_SEPARATOR
    assert model_calls == [("cuda", "float16")]


def test_marker_clean_markdown_removes_internal_spans_without_page_comments():
    source = (
        '<span id="page-1-0">Body with *italics*.</span>\n\n'
        '<span id="page-2-0">Second page.</span>'
    )

    result = marker_service._clean_publication_markdown(source)

    assert result == "Body with *italics*.\n\nSecond page."
    assert "<!-- page:" not in result


def test_real_marker_renderer_pagination_preserves_cleaned_markdown():
    pytest.importorskip("torch")
    marker_document = pytest.importorskip("marker.schema.document")
    marker_page = pytest.importorskip("marker.schema.groups.page")
    marker_blocks = pytest.importorskip("marker.schema.blocks")
    marker_polygon = pytest.importorskip("marker.schema.polygon")
    marker_renderer = pytest.importorskip("marker.renderers.markdown")

    polygon = marker_polygon.PolygonBox.from_bbox([0, 0, 100, 100])
    page_html = (
        "<table><tr><th>Gene</th><th>Value</th></tr>"
        "<tr><td>abc</td><td>1</td></tr></table>",
        "<ul><li>First</li><li><a href='https://example.org'>Second</a> "
        "<img src='figure.png' alt='figure'></li></ul>",
        "",
        "<p>Terminal page.</p>",
    )
    pages = []
    for page_id, html in enumerate(page_html):
        children = []
        structure = []
        if html:
            block = marker_blocks.Text(
                polygon=polygon,
                page_id=page_id,
                block_id=0,
                html=html,
            )
            children.append(block)
            structure.append(block.id)
        pages.append(
            marker_page.PageGroup(
                polygon=polygon,
                page_id=page_id,
                children=children,
                structure=structure,
            )
        )
    document = marker_document.Document(filepath="fixture.pdf", pages=pages)

    plain = marker_renderer.MarkdownRenderer(
        {"paginate_output": False, "extract_images": False}
    )(document).markdown
    paginated = marker_renderer.MarkdownRenderer(
        {
            "paginate_output": True,
            "page_separator": marker_service.MARKER_PAGE_SEPARATOR,
            "extract_images": False,
        }
    )(document).markdown

    expected = marker_service._clean_publication_markdown(plain)
    cleaned_paginated = marker_service._clean_publication_markdown(paginated)
    actual, ranges = marker_service.marker_markdown_with_page_ranges(
        cleaned_paginated,
        expected_page_count=4,
    )

    assert actual == expected
    assert [item["page_number"] for item in ranges] == [1, 2, 4]
