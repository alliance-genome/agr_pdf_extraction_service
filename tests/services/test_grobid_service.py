from app.services.grobid_service import Grobid


class DummyGrobid(Grobid):
    def __init__(self):
        pass

    def extract(self, pdf_path, output_path):
        with open(output_path, "w") as f:
            f.write("GROBID output")


def test_grobid_extract(tmp_path):
    grobid = DummyGrobid()
    pdf_path = tmp_path / "test.pdf"
    output_path = tmp_path / "output.md"
    pdf_path.write_bytes(b"dummy pdf content")
    grobid.extract(str(pdf_path), str(output_path))
    assert output_path.exists()
    assert output_path.read_text() == "GROBID output"


def test_extract_plain_text_splits_paragraphs_and_avoids_heading_duplication():
    grobid = Grobid("http://example.org")
    tei_xml = """
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div>
        <head>Methods</head>
        <p>First paragraph.</p>
        <p>Second paragraph.</p>
      </div>
    </body>
  </text>
</TEI>
"""
    result = grobid.extract_plain_text(tei_xml)
    assert result is not None
    assert "## Methods" in result
    assert "First paragraph." in result
    assert "Second paragraph." in result
    assert "MethodsFirst paragraph." not in result


def test_extract_plain_text_fallback_without_paragraph_tags():
    grobid = Grobid("http://example.org")
    tei_xml = """
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div>
        <head>Results</head>
        <note>Fallback content.</note>
      </div>
    </body>
  </text>
</TEI>
"""
    result = grobid.extract_plain_text(tei_xml)
    assert result is not None
    assert "## Results" in result
    assert "Fallback content." in result
    assert "ResultsFallback content." not in result
