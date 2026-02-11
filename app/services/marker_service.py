import os
import logging
import torch
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from app.services.pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)


class Marker(PDFExtractor):
    def __init__(self, device="cpu", extract_images=False):
        self.device = device
        self.extract_images = extract_images

    def extract(self, pdf_path, output_filename):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if self.device == "cpu":
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if dev.type == "cuda" else torch.float32

        artifact_dict = create_model_dict(device=dev, dtype=dtype)
        converter = PdfConverter(artifact_dict=artifact_dict)

        rendered = converter(pdf_path)
        text, file_ext, images = text_from_rendered(rendered)

        metadata = rendered.metadata
        num_pages = len(metadata.get("page_stats", []))
        logger.info("Marker converted %d pages, extracted %d images", num_pages, len(images))

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text)

        if self.extract_images and images:
            pdf_name = Path(pdf_path).stem
            output_dir = Path(output_filename).parent
            images_dir = os.path.join(output_dir, f"images/{pdf_name}_marker")
            os.makedirs(images_dir, exist_ok=True)

            for img_name, img in images.items():
                img_path = os.path.join(images_dir, img_name)
                img.save(img_path, "PNG")
                logger.info("  Saved image: %s", img_name)
