"""Marker-based PDF extraction service with cached model/converter instances."""

import os
import re
import logging
import torch
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from app.services.pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)

# Process-level model cache — survives across Celery tasks within the same fork
_cached_models = {}  # keyed by (device_type, dtype)
_cached_converters = {}  # keyed by (device_type, dtype, extract_images, disable_links)

_SPAN_REF_RE = re.compile(r"<span id=['\"][^'\"]*['\"]>(.*?)</span>", re.DOTALL)
_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_REF_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)]*\)")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _get_converter(device, dtype, extract_images=False, disable_links=True):
    """Return a cached PdfConverter, creating models on first call."""
    key = (str(device), str(dtype), bool(extract_images), bool(disable_links))
    if key not in _cached_converters:
        logger.info("Marker: loading models for %s/%s (first call in this worker process)", device, dtype)
        artifact_dict = create_model_dict(device=device, dtype=dtype)
        _cached_models[key] = artifact_dict
        converter_config = {
            "extract_images": bool(extract_images),
            "disable_links": bool(disable_links),
        }
        _cached_converters[key] = PdfConverter(
            artifact_dict=artifact_dict,
            config=converter_config,
        )
        logger.info("Marker: models loaded and cached")
    return _cached_converters[key]


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

        converter = _get_converter(
            dev,
            dtype,
            extract_images=self.extract_images,
            disable_links=True,
        )

        with torch.inference_mode():
            rendered = converter(pdf_path)
        text, file_ext, images = text_from_rendered(rendered)
        text = _SPAN_REF_RE.sub(r"\1", text)
        text = _IMAGE_REF_RE.sub("", text)
        text = _LINK_REF_RE.sub(r"\1", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text).strip()

        metadata = rendered.metadata
        num_pages = len(metadata.get("page_stats", []))
        logger.info("Marker converted %d pages, extracted %d images", num_pages, len(images))

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text)

        if self.extract_images and images:
            output_stem = Path(output_filename).stem
            output_dir = Path(output_filename).parent
            images_dir = os.path.join(output_dir, "images", output_stem)
            os.makedirs(images_dir, exist_ok=True)

            for img_name, img in images.items():
                img_path = os.path.join(images_dir, img_name)
                img.save(img_path, "PNG")
                logger.info("  Saved image: %s", img_name)
