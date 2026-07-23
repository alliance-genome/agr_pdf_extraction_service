"""Docling-based PDF extraction service with process-local converter caching."""

import json
import os
import logging
from importlib.metadata import version

from app.services.pdf_extractor import PDFExtractor
from app.services.native_extractor_artifact import (
    persist_native_extractor_artifact,
)
from app.services.native_style import (
    docling_native_style_bytes,
    unavailable_native_style_bytes,
)

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


def _get_rapidocr_langs():
    """Return configured RapidOCR detector and recognizer language families."""
    det_lang = os.environ.get("DOCLING_RAPIDOCR_DET_LANG", "en").strip().lower()
    rec_lang = os.environ.get("DOCLING_RAPIDOCR_REC_LANG", "en").strip().lower()
    supported = {"ch", "en"}
    if det_lang not in supported:
        logger.warning(
            "Unsupported DOCLING_RAPIDOCR_DET_LANG=%r; falling back to en",
            det_lang,
        )
        det_lang = "en"
    if rec_lang not in supported:
        logger.warning(
            "Unsupported DOCLING_RAPIDOCR_REC_LANG=%r; falling back to en",
            rec_lang,
        )
        rec_lang = "en"
    return det_lang, rec_lang


def _build_ocr_options(*, force_full_page_ocr=False):
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
    det_lang, rec_lang = _get_rapidocr_langs()
    rapidocr_params = {
        "EngineConfig.onnxruntime.use_cuda": use_cuda,
    }

    if _HAS_RAPIDOCR_TYPINGS and backend == "onnxruntime":
        rapidocr_model_type = {
            "small": ModelType.SMALL,
            "medium": ModelType.MEDIUM,
        }[model_type]
        det_lang_type = {
            "ch": LangDet.CH,
            "en": LangDet.EN,
        }[det_lang]
        rec_lang_type = {
            "ch": LangRec.CH,
            "en": LangRec.EN,
        }[rec_lang]
        rapidocr_params.update(
            {
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.ocr_version": OCRVersion.PPOCRV6,
                "Det.model_type": rapidocr_model_type,
                "Det.lang_type": det_lang_type,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.ocr_version": OCRVersion.PPOCRV6,
                "Rec.model_type": rapidocr_model_type,
                "Rec.lang_type": rec_lang_type,
            }
        )
    elif model_type != "small":
        logger.warning(
            "RapidOCR typing enums are unavailable; DOCLING_RAPIDOCR_MODEL_TYPE=%s "
            "and language settings may not be applied by Docling.",
            model_type,
        )

    logger.info(
        "Docling: OCR configured with RapidOCR backend=%s, ppocrv6_model=%s, det_lang=%s, rec_lang=%s, onnxruntime_use_cuda=%s",
        backend,
        model_type,
        det_lang,
        rec_lang,
        use_cuda,
    )
    return RapidOcrOptions(
        backend=backend,
        rapidocr_params=rapidocr_params,
        force_full_page_ocr=bool(force_full_page_ocr),
    )


def _get_converter(device="cpu", num_threads=None, *, force_full_page_ocr=False):
    """Return a cached DocumentConverter, creating one on first call."""
    if num_threads is None:
        num_threads = int(os.environ.get("DOCLING_NUM_THREADS", 0)) or None
    ocr_key = (
        os.environ.get("DOCLING_RAPIDOCR_BACKEND", "onnxruntime").strip().lower(),
        _get_rapidocr_model_type(),
        *_get_rapidocr_langs(),
        _get_rapidocr_use_cuda(),
    )
    key = (device, num_threads, ocr_key, bool(force_full_page_ocr))
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

        # Docling discards parsed page cells after assembly by default. Native
        # font names live only on those cells, so retain them until the compact
        # style sidecar is serialized below.
        pipeline_options.generate_parsed_pages = True
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = _build_ocr_options(
            force_full_page_ocr=force_full_page_ocr
        )

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
        _cached_converters[key] = DocumentConverter(format_options=format_options)
        logger.info("Docling: converter created and cached")
    return _cached_converters[key]


class Docling(PDFExtractor):
    def __init__(self, device="cpu", *, force_full_page_ocr=False):
        self.device = device
        self.force_full_page_ocr = bool(force_full_page_ocr)
        self.num_threads = int(os.environ.get("DOCLING_NUM_THREADS", 0)) or None

    def extract(self, pdf_path, output_filename):
        converter = _get_converter(
            self.device,
            self.num_threads,
            force_full_page_ocr=self.force_full_page_ocr,
        )
        result = converter.convert(pdf_path)
        doc = result.document

        if not hasattr(doc, "num_pages"):
            raise RuntimeError(
                "Installed Docling document object does not expose num_pages()."
            )

        total_pages = int(doc.num_pages())
        if total_pages <= 0:
            raise RuntimeError("Docling reported zero pages for document; cannot export markdown.")

        markdown = doc.export_to_markdown(
            image_placeholder="",
            page_break_placeholder="",
            text_width=-1,
        ).strip()
        if not markdown:
            raise RuntimeError("Docling returned empty Markdown.")

        structured = doc.export_to_dict(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
        native_bytes = json.dumps(
            structured,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            native_style_bytes = docling_native_style_bytes(result)
        except Exception as exc:
            logger.warning(
                "Docling native PDF style capture unavailable: %s",
                type(exc).__name__,
            )
            native_style_bytes = unavailable_native_style_bytes(
                "docling", type(exc).__name__
            )
        del result

        from app.services.page_coverage import native_payload_covered_pages

        covered_pages = native_payload_covered_pages("docling", native_bytes)

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(markdown)

        persist_native_extractor_artifact(
            source="docling",
            output_filename=output_filename,
            native_bytes=native_bytes,
            native_media_type="application/json",
            pdf_path=pdf_path,
            extractor_versions={
                "docling": version("docling"),
                "docling-core": version("docling-core"),
            },
            options={
                "device": self.device,
                "do_ocr": True,
                "force_full_page_ocr": self.force_full_page_ocr,
                "ocr_backend": os.environ.get(
                    "DOCLING_RAPIDOCR_BACKEND",
                    "onnxruntime",
                ),
                "ocr_model_type": _get_rapidocr_model_type(),
                "generate_parsed_pages": True,
                "native_style_cell_collection": "word_cells",
                "native_style_sidecar": True,
            },
            expected_page_count=total_pages,
            covered_pages=covered_pages,
            native_style_bytes=native_style_bytes,
        )

        from app.services.page_coverage import write_extractor_page_coverage

        write_extractor_page_coverage(
            source="docling",
            output_filename=output_filename,
            pdf_path=pdf_path,
            expected_page_count=total_pages,
            covered_pages=covered_pages,
        )
