import os
import sys
import torch
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

from app.services.pdf_extractor import PDFExtractor

class Marker(PDFExtractor):
    def __init__(self):
        pass

    def extract(self, pdf_path, output_filename):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        artifact_dict = create_model_dict(device=device, dtype=dtype)

        converter = PdfConverter(artifact_dict=artifact_dict)

        rendered = converter(pdf_path)
        text, file_ext, images = text_from_rendered(rendered)

        # Get metadata
        metadata = rendered.metadata
        num_pages = len(metadata.get('page_stats', []))

        print(f"Converted {num_pages} pages")
        print(f"Extracted {len(images)} images")

        # Save markdown output
        pdf_name = Path(pdf_path).stem
        output_dir = Path(output_filename).parent

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text)

        if images:
            images_dir = os.path.join(output_dir, f"images/{pdf_name}_marker")
            os.makedirs(images_dir, exist_ok=True)

            print(f"Saving images to: {images_dir}")
            for img_name, img in images.items():
                img_path = os.path.join(images_dir, img_name)
                img.save(img_path, "PNG")
                print(f"  - Saved: {img_name}")

