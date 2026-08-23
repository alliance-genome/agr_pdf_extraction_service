import sys
import types

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
    docling_core = types.ModuleType("docling_core")
    docling_core_types = types.ModuleType("docling_core.types")
    docling_core_doc = types.ModuleType("docling_core.types.doc")

    class _DocumentConverter:
        def __init__(self, *args, **kwargs):
            pass

    class _PdfFormatOption:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _PdfPipelineOptions:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.accelerator_options = kwargs.get("accelerator_options")
            self.ocr_batch_size = kwargs.get("ocr_batch_size")
            self.layout_batch_size = kwargs.get("layout_batch_size")
            self.generate_parsed_pages = False
            self.do_ocr = None
            self.ocr_options = None

    class _RapidOcrOptions:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            pass

    class _ThreadedPdfPipelineOptions(_PdfPipelineOptions):
        pass

    class _AcceleratorOptions:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.device = kwargs.get("device")
            self.num_threads = None

    class _AcceleratorDevice:
        CPU = "cpu"
        CUDA = "cuda"
        AUTO = "auto"

    class _InputFormat:
        PDF = "pdf"

    class _ContentLayer:
        BODY = "body"

    document_converter.DocumentConverter = _DocumentConverter
    document_converter.PdfFormatOption = _PdfFormatOption
    base_models.InputFormat = _InputFormat
    pipeline_options.PdfPipelineOptions = _PdfPipelineOptions
    pipeline_options.RapidOcrOptions = _RapidOcrOptions
    pipeline_options.ThreadedPdfPipelineOptions = _ThreadedPdfPipelineOptions
    accelerator_options.AcceleratorOptions = _AcceleratorOptions
    accelerator_options.AcceleratorDevice = _AcceleratorDevice
    docling_core_doc.ContentLayer = _ContentLayer

    sys.modules["docling"] = docling_pkg
    sys.modules["docling.document_converter"] = document_converter
    sys.modules["docling.datamodel"] = datamodel_pkg
    sys.modules["docling.datamodel.base_models"] = base_models
    sys.modules["docling.datamodel.pipeline_options"] = pipeline_options
    sys.modules["docling.datamodel.accelerator_options"] = accelerator_options
    sys.modules["docling_core"] = docling_core
    sys.modules["docling_core.types"] = docling_core_types
    sys.modules["docling_core.types.doc"] = docling_core_doc


_install_docling_stubs()

rapidocr_pkg = types.ModuleType("rapidocr")
rapidocr_utils = types.ModuleType("rapidocr.utils")
rapidocr_typings = types.ModuleType("rapidocr.utils.typings")


class _EnumValue:
    def __init__(self, value):
        self.value = value


class _EngineType:
    ONNXRUNTIME = _EnumValue("onnxruntime")


class _OCRVersion:
    PPOCRV6 = _EnumValue("PP-OCRv6")


class _ModelType:
    SMALL = _EnumValue("small")
    MEDIUM = _EnumValue("medium")


class _LangDet:
    CH = _EnumValue("ch")
    EN = _EnumValue("en")


class _LangRec:
    CH = _EnumValue("ch")
    EN = _EnumValue("en")


rapidocr_typings.EngineType = _EngineType
rapidocr_typings.OCRVersion = _OCRVersion
rapidocr_typings.ModelType = _ModelType
rapidocr_typings.LangDet = _LangDet
rapidocr_typings.LangRec = _LangRec
sys.modules["rapidocr"] = rapidocr_pkg
sys.modules["rapidocr.utils"] = rapidocr_utils
sys.modules["rapidocr.utils.typings"] = rapidocr_typings

from app.services.docling_service import Docling, _cached_converters, _get_converter


class _FakeDocument:
    def __init__(self, page_order=(1, 2, 3)):
        self.calls = []
        self.page_order = page_order

    def num_pages(self):
        return 3

    def export_to_markdown(
        self,
        image_placeholder="",
        page_break_placeholder="",
        text_width=-1,
        included_content_layers=None,
        traverse_pictures=False,
    ):
        kwargs = {
            "image_placeholder": image_placeholder,
            "page_break_placeholder": page_break_placeholder,
            "text_width": text_width,
            "included_content_layers": included_content_layers,
            "traverse_pictures": traverse_pictures,
        }
        self.calls.append(kwargs)
        return (
            f"# Title{page_break_placeholder}\n\n"
            f"Body with *italics*.{page_break_placeholder}"
        )

    def iterate_items(
        self,
        *,
        with_groups,
        included_content_layers,
        traverse_pictures,
    ):
        self.iterate_kwargs = {
            "with_groups": with_groups,
            "included_content_layers": included_content_layers,
            "traverse_pictures": traverse_pictures,
        }
        for page_number in self.page_order:
            provenance = types.SimpleNamespace(page_no=page_number)
            yield types.SimpleNamespace(prov=[provenance]), 0

    def export_to_dict(self, **kwargs):
        self.export_dict_kwargs = kwargs
        return {
            "schema_name": "DoclingDocument",
            "texts": [
                {
                    "self_ref": f"#/texts/{page - 1}",
                    "text": f"page {page}",
                    "prov": [{"page_no": page}],
                }
                for page in (1, 2, 3)
            ],
        }


class _FakeConverter:
    def __init__(self, fake_doc):
        self._doc = fake_doc

    class _Result:
        def __init__(self, fake_doc):
            self.document = fake_doc

    def convert(self, _pdf_path):
        return self._Result(self._doc)


def test_docling_extract_writes_clean_markdown_and_native_document(monkeypatch, tmp_path):
    fake_doc = _FakeDocument()
    monkeypatch.setattr(
        "app.services.docling_service._get_converter",
        lambda *_args, **_kwargs: _FakeConverter(fake_doc),
    )
    monkeypatch.setattr(
        "app.services.docling_service.version",
        lambda package: {"docling": "2.113.0", "docling-core": "2.87.1"}[package],
    )

    docling = Docling()
    pdf_path = tmp_path / "test.pdf"
    output_path = tmp_path / "output.md"
    pdf_path.write_bytes(b"dummy pdf content")

    docling.extract(str(pdf_path), str(output_path))

    content = output_path.read_text(encoding="utf-8")
    assert content == "# Title\n\nBody with *italics*."
    assert "<!-- page:" not in content
    assert len(fake_doc.calls) == 1
    assert fake_doc.calls[0]["image_placeholder"] == ""
    assert fake_doc.calls[0]["page_break_placeholder"].startswith(
        "PDFX_DOCLING_PAGE_BOUNDARY_"
    )
    assert fake_doc.calls[0]["text_width"] == -1
    assert fake_doc.calls[0]["included_content_layers"] == {"body"}
    assert fake_doc.calls[0]["traverse_pictures"] is False
    assert fake_doc.iterate_kwargs == {
        "with_groups": True,
        "included_content_layers": {"body"},
        "traverse_pictures": False,
    }
    assert fake_doc.export_dict_kwargs == {
        "mode": "json",
        "by_alias": True,
        "exclude_none": True,
    }

    from app.services.native_extractor_artifact import load_native_extractor_artifact

    manifest, native = load_native_extractor_artifact(
        source="docling", output_filename=output_path
    )
    assert b'"schema_name":"DoclingDocument"' in native
    assert manifest["expected_page_count"] == 3
    assert manifest["covered_pages"] == [1, 2, 3]
    assert manifest["options"]["native_style_cell_collection"] == "word_cells"
    assert manifest["page_provenance_filename"].endswith("page-provenance.json")


def test_docling_non_monotonic_primary_page_order_fails_closed(monkeypatch, tmp_path):
    fake_doc = _FakeDocument(page_order=(1, 2, 1, 3))
    monkeypatch.setattr(
        "app.services.docling_service._get_converter",
        lambda *_args, **_kwargs: _FakeConverter(fake_doc),
    )
    monkeypatch.setattr(
        "app.services.docling_service.version",
        lambda package: {"docling": "2.113.0", "docling-core": "2.87.1"}[package],
    )

    pdf_path = tmp_path / "test.pdf"
    output_path = tmp_path / "output.md"
    pdf_path.write_bytes(b"dummy pdf content")

    Docling().extract(str(pdf_path), str(output_path))

    from app.services.native_extractor_artifact import load_native_extractor_artifact
    from app.services.page_provenance import load_source_page_provenance

    manifest, _native = load_native_extractor_artifact(
        source="docling", output_filename=output_path
    )
    page_map = load_source_page_provenance(output_path, manifest=manifest)

    assert output_path.read_text(encoding="utf-8") == (
        "# Title\n\nBody with *italics*."
    )
    assert page_map["ranges"] == [
        {
            "byte_start": 0,
            "byte_end": len(output_path.read_bytes()),
            "page_number": None,
            "candidate_pages": [],
            "method": None,
            "native_id": None,
            "kind": None,
            "residual_reason": "unsafe_docling_page_transition",
        }
    ]


def test_docling_converter_pins_rapidocr_onnxruntime_cpu(monkeypatch):
    created = {}

    class _CapturingConverter:
        def __init__(self, format_options):
            created["format_options"] = format_options

    monkeypatch.setattr("app.services.docling_service.DocumentConverter", _CapturingConverter)
    monkeypatch.setenv("DOCLING_RAPIDOCR_BACKEND", "onnxruntime")
    monkeypatch.setenv("DOCLING_RAPIDOCR_MODEL_TYPE", "medium")
    monkeypatch.setenv("DOCLING_RAPIDOCR_DET_LANG", "en")
    monkeypatch.setenv("DOCLING_RAPIDOCR_REC_LANG", "en")
    monkeypatch.setenv("DOCLING_RAPIDOCR_USE_CUDA", "false")
    _cached_converters.clear()

    _get_converter("cuda", num_threads=8)

    pipeline_options = created["format_options"]["pdf"].kwargs["pipeline_options"]
    assert pipeline_options.accelerator_options.device == "cuda"
    assert pipeline_options.generate_parsed_pages is True
    assert pipeline_options.do_ocr is True
    assert pipeline_options.ocr_options.kwargs["backend"] == "onnxruntime"
    assert pipeline_options.ocr_options.kwargs["force_full_page_ocr"] is False
    rapidocr_params = pipeline_options.ocr_options.kwargs["rapidocr_params"]
    assert rapidocr_params["EngineConfig.onnxruntime.use_cuda"] is False
    assert rapidocr_params["Det.model_type"].value == "medium"
    assert rapidocr_params["Rec.model_type"].value == "medium"
    assert rapidocr_params["Det.lang_type"].value == "en"
    assert rapidocr_params["Rec.lang_type"].value == "en"


def test_emergency_docling_converter_forces_full_page_ocr(monkeypatch):
    created = []

    class _CapturingConverter:
        def __init__(self, format_options):
            created.append(format_options)

    monkeypatch.setattr("app.services.docling_service.DocumentConverter", _CapturingConverter)
    _cached_converters.clear()

    normal = _get_converter("cpu", num_threads=2)
    emergency = _get_converter(
        "cpu",
        num_threads=2,
        force_full_page_ocr=True,
    )

    assert normal is not emergency
    options = created[-1]["pdf"].kwargs["pipeline_options"].ocr_options
    assert options.kwargs["force_full_page_ocr"] is True
