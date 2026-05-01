"""Marker-based PDF extraction service with cached model/converter instances."""

import os
import re
import shutil
import logging
import torch
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.schema import BlockTypes

from app.image_metadata import (
    annotate_image_diagnostics,
    build_image_manifest_entry,
    extract_image_references,
    write_image_manifest,
)
from app.services.pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)

# Process-level model cache — survives across Celery tasks within the same fork
_cached_models = {}  # keyed by (device_type, dtype)
_cached_converters = {}  # keyed by (device_type, dtype, extract_images, disable_links)

_PAGE_SPAN_RE = re.compile(
    r"<span\s+id=['\"]page-(\d+)-[^'\"]*['\"]>(.*?)</span>",
    re.DOTALL | re.IGNORECASE,
)
_SPAN_REF_RE = re.compile(r"<span id=['\"][^'\"]*['\"]>(.*?)</span>", re.DOTALL)
_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_REF_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)]*\)")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_IMAGE_BLOCK_TYPES = (BlockTypes.Figure, BlockTypes.Picture)
_IMAGE_GROUP_TYPES = (BlockTypes.FigureGroup, BlockTypes.PictureGroup)
_CAPTION_TYPES = (BlockTypes.Caption, BlockTypes.Footnote)


def _rounded_polygon_points(polygon):
    return [[round(float(x), 2), round(float(y), 2)] for x, y in polygon]


def _rounded_bbox(bbox):
    return [round(float(value), 2) for value in bbox]


def _image_filename_for_block(block, image_names):
    expected_stem = block.id.to_path()
    for image_name in image_names:
        if Path(image_name).stem == expected_stem:
            return image_name
    return None


def _caption_text(block, document):
    text = block.raw_text(document)
    return text.strip() if text else ""


def _collect_structured_image_metadata(document, image_names):
    """Collect caption and coordinate metadata from Marker's structured document."""
    image_names = set(image_names)
    metadata_by_filename = {}

    for block in document.contained_blocks(_IMAGE_BLOCK_TYPES):
        filename = _image_filename_for_block(block, image_names)
        if not filename:
            continue

        metadata_by_filename[filename] = {
            "page_index": block.page_id,
            "marker_image_type": block.block_type.name,
            "marker_image_index": block.block_id,
            "block_id": str(block.id),
            "group_id": None,
            "bbox": _rounded_bbox(block.polygon.bbox),
            "polygon": _rounded_polygon_points(block.polygon.polygon),
            "caption_text": None,
        }

    for group in document.contained_blocks(_IMAGE_GROUP_TYPES):
        child_blocks = group.structure_blocks(document)
        captions = [
            _caption_text(child, document)
            for child in child_blocks
            if child.block_type in _CAPTION_TYPES
        ]
        captions = [caption for caption in captions if caption]
        caption = "\n".join(captions) or None
        for child in child_blocks:
            if child.block_type not in _IMAGE_BLOCK_TYPES:
                continue
            filename = _image_filename_for_block(child, image_names)
            if not filename:
                continue
            metadata = metadata_by_filename.setdefault(filename, {})
            metadata.update({
                "group_id": str(group.id),
                "caption_text": caption,
            })

    return metadata_by_filename


def _save_image(img, img_path):
    ext = Path(img_path).suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_path, "JPEG")
    elif ext == ".png":
        img.save(img_path, "PNG")
    else:
        img.save(img_path)


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
            document = converter.build_document(pdf_path)
            converter.page_count = len(document.pages)
            renderer = converter.resolve_dependencies(converter.renderer)
            rendered = renderer(document)
        text, file_ext, images = text_from_rendered(rendered)
        image_references = extract_image_references(text)
        if self.extract_images:
            structured_metadata = _collect_structured_image_metadata(document, images.keys())
            for image_name, metadata in structured_metadata.items():
                image_references.setdefault(image_name, {})["structured_metadata"] = metadata
        # Preserve page provenance as explicit markdown markers before span cleanup.
        text = _PAGE_SPAN_RE.sub(lambda m: f"<!-- page: {m.group(1)} -->\n{m.group(2)}", text)
        text = _SPAN_REF_RE.sub(r"\1", text)
        text = _IMAGE_REF_RE.sub("", text)
        text = _LINK_REF_RE.sub(r"\1", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text).strip()

        metadata = rendered.metadata
        num_pages = len(metadata.get("page_stats", []))
        logger.info("Marker converted %d pages, extracted %d images", num_pages, len(images))

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text)

        if self.extract_images:
            output_stem = Path(output_filename).stem
            output_dir = Path(output_filename).parent
            images_dir = os.path.join(output_dir, "images", output_stem)
            if os.path.isdir(images_dir):
                shutil.rmtree(images_dir)
            os.makedirs(images_dir, exist_ok=True)

            manifest_entries = []
            for img_name, img in images.items():
                img_path = os.path.join(images_dir, img_name)
                _save_image(img, img_path)
                entry = build_image_manifest_entry(
                    img_name,
                    size_bytes=os.path.getsize(img_path),
                    references=image_references,
                )
                entry["image_width"], entry["image_height"] = img.size
                annotate_image_diagnostics(entry)
                manifest_entries.append(entry)
                logger.info("  Saved image: %s", img_name)

            manifest_entries.sort(key=lambda entry: entry["filename"])
            write_image_manifest(images_dir, manifest_entries)
