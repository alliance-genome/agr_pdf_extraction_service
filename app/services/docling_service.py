import json
from pathlib import Path

from app.services.pdf_extractor import PDFExtractor

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel


class Docling(PDFExtractor):
    def __init__(self):
        pass

    def extract(self, pdf_path, output_filename):
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc = result.document

        markdown = doc.export_to_markdown()
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(markdown)