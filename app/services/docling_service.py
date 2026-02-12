import os
import logging
from app.services.pdf_extractor import PDFExtractor

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

try:
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
    _HAS_THREADED = True
except ImportError:
    _HAS_THREADED = False

logger = logging.getLogger(__name__)

# Process-level converter cache — survives across Celery tasks within the same fork
_cached_converters = {}  # keyed by (device, num_threads)


def _get_converter(device="cpu", num_threads=None):
    """Return a cached DocumentConverter, creating one on first call."""
    if num_threads is None:
        num_threads = int(os.environ.get("DOCLING_NUM_THREADS", 0)) or None
    key = (device, num_threads)
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

        markdown = doc.export_to_markdown()
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(markdown)
