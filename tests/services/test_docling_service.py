import sys
import types

import pytest


def _install_docling_stubs():
    """Provide lightweight docling stubs so unit tests run without heavy deps."""
    if "docling.document_converter" in sys.modules:
        return

    docling_pkg = types.ModuleType("docling")
    document_converter = types.ModuleType("docling.document_converter")
    datamodel_pkg = types.ModuleType("docling.datamodel")
    base_models = types.ModuleType("docling.datamodel.base_models")
    pipeline_options = types.ModuleType("docling.datamodel.pipeline_options")
    accelerator_options = types.ModuleType("docling.datamodel.accelerator_options")

    class _DocumentConverter:
        def __init__(self, *args, **kwargs):
            pass

    class _PdfFormatOption:
        def __init__(self, *args, **kwargs):
            pass

    class _PdfPipelineOptions:
        def __init__(self, *args, **kwargs):
            pass

    class _ThreadedPdfPipelineOptions(_PdfPipelineOptions):
        pass

    class _AcceleratorOptions:
        def __init__(self, *args, **kwargs):
            pass

    class _AcceleratorDevice:
        CPU = "cpu"
        CUDA = "cuda"
        AUTO = "auto"

    class _InputFormat:
        PDF = "pdf"

    document_converter.DocumentConverter = _DocumentConverter
    document_converter.PdfFormatOption = _PdfFormatOption
    base_models.InputFormat = _InputFormat
    pipeline_options.PdfPipelineOptions = _PdfPipelineOptions
    pipeline_options.ThreadedPdfPipelineOptions = _ThreadedPdfPipelineOptions
    accelerator_options.AcceleratorOptions = _AcceleratorOptions
    accelerator_options.AcceleratorDevice = _AcceleratorDevice

    sys.modules["docling"] = docling_pkg
    sys.modules["docling.document_converter"] = document_converter
    sys.modules["docling.datamodel"] = datamodel_pkg
    sys.modules["docling.datamodel.base_models"] = base_models
    sys.modules["docling.datamodel.pipeline_options"] = pipeline_options
    sys.modules["docling.datamodel.accelerator_options"] = accelerator_options


_install_docling_stubs()

from app.services.docling_service import Docling


class _FakeDocument:
    def __init__(self):
        self.calls = []

    def num_pages(self):
        return 3

    def export_to_markdown(
        self,
        image_placeholder="",
        page_break_placeholder="",
        text_width=-1,
        page_no=None,
    ):
        kwargs = {
            "image_placeholder": image_placeholder,
            "page_break_placeholder": page_break_placeholder,
            "text_width": text_width,
            "page_no": page_no,
        }
        self.calls.append(kwargs)
        return f"Page {page_no} body"


class _FakeConverter:
    def __init__(self, fake_doc):
        self._doc = fake_doc

    class _Result:
        def __init__(self, fake_doc):
            self.document = fake_doc

    def convert(self, _pdf_path):
        return self._Result(self._doc)


def test_docling_extract_writes_page_markers(monkeypatch, tmp_path):
    fake_doc = _FakeDocument()
    monkeypatch.setattr(
        "app.services.docling_service._get_converter",
        lambda *_args, **_kwargs: _FakeConverter(fake_doc),
    )

    docling = Docling()
    pdf_path = tmp_path / "test.pdf"
    output_path = tmp_path / "output.md"
    pdf_path.write_bytes(b"dummy pdf content")

    docling.extract(str(pdf_path), str(output_path))

    content = output_path.read_text(encoding="utf-8")
    assert "<!-- page: 1 -->" in content
    assert "<!-- page: 2 -->" in content
    assert "<!-- page: 3 -->" in content
    assert "Page 1 body" in content
    assert "Page 2 body" in content
    assert "Page 3 body" in content

    assert len(fake_doc.calls) == 3
    assert fake_doc.calls[0]["page_no"] == 1
    assert fake_doc.calls[1]["page_no"] == 2
    assert fake_doc.calls[2]["page_no"] == 3


def test_docling_extract_requires_page_no_support(monkeypatch, tmp_path):
    class _NoPageNoDocument:
        def num_pages(self):
            return 1

        def export_to_markdown(self, image_placeholder="", page_break_placeholder="", text_width=-1):
            return "content"

    monkeypatch.setattr(
        "app.services.docling_service._get_converter",
        lambda *_args, **_kwargs: _FakeConverter(_NoPageNoDocument()),
    )

    docling = Docling()
    pdf_path = tmp_path / "test.pdf"
    output_path = tmp_path / "output.md"
    pdf_path.write_bytes(b"dummy pdf content")

    with pytest.raises(RuntimeError, match="missing export_to_markdown\\(page_no=...\\)"):
        docling.extract(str(pdf_path), str(output_path))
