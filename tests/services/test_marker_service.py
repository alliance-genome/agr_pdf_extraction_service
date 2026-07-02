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
    assert model_calls == [("cuda", "float16")]
