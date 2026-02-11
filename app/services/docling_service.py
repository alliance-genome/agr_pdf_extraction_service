import logging
from app.services.pdf_extractor import PDFExtractor

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

logger = logging.getLogger(__name__)


class Docling(PDFExtractor):
    def __init__(self, device="cpu"):
        accelerator = AcceleratorDevice.CPU if device == "cpu" else AcceleratorDevice.AUTO
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(device=accelerator)
        self.format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }

    def extract(self, pdf_path, output_filename):
        converter = DocumentConverter(format_options=self.format_options)
        result = converter.convert(pdf_path)
        doc = result.document

        markdown = doc.export_to_markdown()
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(markdown)
