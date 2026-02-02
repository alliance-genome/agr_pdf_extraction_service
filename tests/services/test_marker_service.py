import pytest
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