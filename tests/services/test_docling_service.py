import pytest
from app.services.docling_service import Docling

class DummyDocling(Docling):
    def extract(self, pdf_path, output_path):
        with open(output_path, "w") as f:
            f.write("Docling output")

def test_docling_extract(tmp_path):
    docling = DummyDocling()
    pdf_path = tmp_path / "test.pdf"
    output_path = tmp_path / "output.md"
    pdf_path.write_bytes(b"dummy pdf content")
    docling.extract(str(pdf_path), str(output_path))
    assert output_path.exists()
    assert output_path.read_text() == "Docling output"