"""Docling-based PDF extraction service with process-local converter caching."""

import inspect
import os
import logging
from app.services.pdf_extractor import PDFExtractor

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

try:
    from rapidocr.utils.typings import EngineType, LangDet, LangRec, ModelType, OCRVersion
    _HAS_RAPIDOCR_TYPINGS = True
except ImportError:
    _HAS_RAPIDOCR_TYPINGS = False

try:
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
    _HAS_THREADED = True
except ImportError:
    _HAS_THREADED = False

logger = logging.getLogger(__name__)

# Process-level converter cache — survives across Celery tasks within the same fork
_cached_converters = {}  # keyed by Docling accelerator and OCR configuration


def _get_rapidocr_model_type():
    """Return the configured RapidOCR PP-OCRv6 model size."""
    raw_value = os.environ.get("DOCLING_RAPIDOCR_MODEL_TYPE", "medium").strip().lower()
    if raw_value not in {"small", "medium"}:
        logger.warning(
            "Unsupported DOCLING_RAPIDOCR_MODEL_TYPE=%r; falling back to medium",
            raw_value,
        )
        return "medium"
    return raw_value


def _get_rapidocr_use_cuda():
    """RapidOCR OCR runs on CPU ONNXRuntime by default, independent of Docling CUDA."""
    raw_value = os.environ.get("DOCLING_RAPIDOCR_USE_CUDA", "false").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _build_ocr_options():
    """Build explicit Docling OCR options.

    Docling's automatic OCR selection can choose RapidOCR's torch backend on
    CUDA hosts. RapidOCR 3.9 does not support the default PP-OCRv6 detector on
    torch, which fails as "Unsupported configuration: torch.PP-OCRv6.det.small".
    Keep Docling layout/table acceleration on CUDA, but pin OCR to RapidOCR's
    ONNXRuntime backend where PP-OCRv6 is supported.
    """
    model_type = _get_rapidocr_model_type()
    backend = os.environ.get("DOCLING_RAPIDOCR_BACKEND", "onnxruntime").strip().lower()
    use_cuda = _get_rapidocr_use_cuda()
    rapidocr_params = {
        "EngineConfig.onnxruntime.use_cuda": use_cuda,
    }

    if _HAS_RAPIDOCR_TYPINGS and backend == "onnxruntime":
        rapidocr_model_type = {
            "small": ModelType.SMALL,
            "medium": ModelType.MEDIUM,
        }[model_type]
        rapidocr_params.update(
            {
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.ocr_version": OCRVersion.PPOCRV6,
                "Det.model_type": rapidocr_model_type,
                "Det.lang_type": LangDet.CH,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.ocr_version": OCRVersion.PPOCRV6,
                "Rec.model_type": rapidocr_model_type,
                "Rec.lang_type": LangRec.CH,
            }
        )
    elif model_type != "small":
        logger.warning(
            "RapidOCR typing enums are unavailable; DOCLING_RAPIDOCR_MODEL_TYPE=%s "
            "may not be applied by Docling.",
            model_type,
        )

    logger.info(
        "Docling: OCR configured with RapidOCR backend=%s, ppocrv6_model=%s, onnxruntime_use_cuda=%s",
        backend,
        model_type,
        use_cuda,
    )
    return RapidOcrOptions(backend=backend, rapidocr_params=rapidocr_params)


def _get_converter(device="cpu", num_threads=None):
    """Return a cached DocumentConverter, creating one on first call."""
    if num_threads is None:
        num_threads = int(os.environ.get("DOCLING_NUM_THREADS", 0)) or None
    ocr_key = (
        os.environ.get("DOCLING_RAPIDOCR_BACKEND", "onnxruntime").strip().lower(),
        _get_rapidocr_model_type(),
        _get_rapidocr_use_cuda(),
    )
    key = (device, num_threads, ocr_key)
    if key not in _cached_converters:
        logger.info("Docling: creating converter for device=%s, num_threads=%s (first call in this worker process)", device, num_threads)

        use_gpu = device in ("cuda", "gpu", "auto")
        if device == "cpu":
            accelerator = AcceleratorDevice.CPU
        elif device == "cuda":
            accelerator = AcceleratorDevice.CUDA
        else:
            accelerator = AcceleratorDevice.AUTO

        accel_opts = AcceleratorOptions(device=accelerator)
        if num_threads:
            accel_opts.num_threads = num_threads

        # Use ThreadedPdfPipelineOptions with GPU batch sizes when available
        if use_gpu and _HAS_THREADED:
            batch_size = int(os.environ.get("DOCLING_BATCH_SIZE", 32))
            pipeline_options = ThreadedPdfPipelineOptions(
                accelerator_options=accel_opts,
                ocr_batch_size=batch_size,
                layout_batch_size=batch_size,
            )
            logger.info("Docling: using ThreadedPdfPipelineOptions with batch_size=%d", batch_size)
        else:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = accel_opts

        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = _build_ocr_options()

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
        _cached_converters[key] = DocumentConverter(format_options=format_options)
        logger.info("Docling: converter created and cached")
    return _cached_converters[key]


class Docling(PDFExtractor):
    def __init__(self, device="cpu"):
        self.device = device
        self.num_threads = int(os.environ.get("DOCLING_NUM_THREADS", 0)) or None

    def extract(self, pdf_path, output_filename):
        converter = _get_converter(self.device, self.num_threads)
        result = converter.convert(pdf_path)
        doc = result.document

        export_signature = inspect.signature(doc.export_to_markdown)
        if "page_no" not in export_signature.parameters:
            raise RuntimeError(
                "Installed Docling version does not support page-aware markdown export "
                "(missing export_to_markdown(page_no=...))."
            )

        if not hasattr(doc, "num_pages"):
            raise RuntimeError(
                "Installed Docling document object does not expose num_pages(); "
                "cannot produce page-aware markdown."
            )

        total_pages = int(doc.num_pages())
        if total_pages <= 0:
            raise RuntimeError("Docling reported zero pages for document; cannot export markdown.")

        page_blocks = []
        for page_no in range(1, total_pages + 1):
            page_markdown = doc.export_to_markdown(
                image_placeholder="",
                page_break_placeholder="",
                text_width=-1,
                page_no=page_no,
            ).strip()
            page_blocks.append(f"<!-- page: {page_no} -->")
            if page_markdown:
                page_blocks.append(page_markdown)

        markdown = "\n\n".join(page_blocks).strip()
        if not markdown:
            raise RuntimeError("Docling returned empty markdown during page-aware export.")

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(markdown)
