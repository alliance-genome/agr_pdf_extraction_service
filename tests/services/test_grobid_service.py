import pytest
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