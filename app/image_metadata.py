"""Helpers for extracted image metadata.

Marker image filenames encode useful provenance, but the numeric suffix is an
internal image index, not necessarily the paper's author-facing figure number.
"""

import json
import os
import re

IMAGE_MANIFEST_FILENAME = ".pdfx-images.json"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif")
IMAGE_METADATA_FIELDS = (
    "page_index",
    "marker_image_type",
    "marker_image_index",
    "block_id",
    "group_id",
    "bbox",
    "polygon",
    "image_width",
    "image_height",
    "is_likely_figure",
    "diagnostic_flags",
    "alt_text",
    "caption_text",
    "nearby_text",
    "figure_label",
    "figure_number",
    "heuristic_is_likely_figure",
    "figure_decision_source",
    "image_reviewed",
    "image_review_method",
    "image_review_model",
    "image_review_classification",
    "image_review_is_scientific_figure",
    "image_review_confidence",
    "image_review_reason",
    "image_review_needs_vision",
    "image_review_error",
)

_MARKER_FILENAME_RE = re.compile(
    r"^_page_(?P<page_index>\d+)_(?P<image_type>[A-Za-z]+)_(?P<image_index>\d+)\.[A-Za-z0-9]+$"
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _nearby_text(text, limit=1000):
    """Keep the text around an image reference as evidence for LLM review."""
    if not text:
        return None
    if not text.strip():
        return None
    return text[:limit]


def metadata_from_filename(filename):
    """Return Marker provenance encoded in a generated image filename."""
    basename = os.path.basename(filename or "")
    metadata = {
        "page_index": None,
        "marker_image_type": None,
        "marker_image_index": None,
    }
    match = _MARKER_FILENAME_RE.match(basename)
    if not match:
        return metadata

    metadata["page_index"] = int(match.group("page_index"))
    metadata["marker_image_type"] = match.group("image_type")
    metadata["marker_image_index"] = int(match.group("image_index"))
    return metadata


def extract_image_references(markdown):
    """Map image basenames to raw alt text and nearby Markdown evidence."""
    references = {}
    if not markdown:
        return references

    matches = list(_MARKDOWN_IMAGE_RE.finditer(markdown))
    for index, match in enumerate(matches):
        alt_text = match.group(1).strip() or None
        filename = os.path.basename(match.group(2))
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        nearby_context = markdown[match.end():min(next_start, match.end() + 1000)]
        references[filename] = {
            "alt_text": alt_text,
            "nearby_text": _nearby_text(nearby_context),
        }

    return references


def build_image_manifest_entry(filename, size_bytes=None, references=None):
    """Build the manifest/API payload for a locally extracted image."""
    basename = os.path.basename(filename)
    structured_metadata = (references or {}).get(basename, {}).get("structured_metadata") or {}
    entry = {
        "filename": basename,
        "size_bytes": size_bytes,
        **metadata_from_filename(basename),
        "block_id": None,
        "group_id": None,
        "bbox": None,
        "polygon": None,
        "image_width": None,
        "image_height": None,
        "is_likely_figure": None,
        "diagnostic_flags": [],
        "alt_text": None,
        "caption_text": None,
        "nearby_text": None,
        "figure_label": None,
        "figure_number": None,
        "heuristic_is_likely_figure": None,
        "figure_decision_source": None,
        "image_reviewed": False,
        "image_review_method": None,
        "image_review_model": None,
        "image_review_classification": None,
        "image_review_is_scientific_figure": None,
        "image_review_confidence": None,
        "image_review_reason": None,
        "image_review_needs_vision": None,
        "image_review_error": None,
    }

    entry.update({
        key: structured_metadata.get(key)
        for key in IMAGE_METADATA_FIELDS
        if key in structured_metadata
    })

    reference = (references or {}).get(basename)
    if reference:
        for key in ("alt_text", "nearby_text"):
            if reference.get(key):
                entry[key] = reference.get(key)
    return entry


def annotate_image_diagnostics(entry):
    """Add conservative figure/usefulness diagnostics to an image payload."""
    flags = []
    width = entry.get("image_width")
    height = entry.get("image_height")
    image_type = entry.get("marker_image_type")

    has_size = width is not None and height is not None
    is_small = bool(has_size and (width < 120 or height < 80))
    if is_small:
        flags.append("small_image")
    if not entry.get("caption_text"):
        flags.append("no_caption")
    if not entry.get("figure_label"):
        flags.append("no_figure_label")
    if image_type == "Picture":
        flags.append("marker_picture_type")

    heuristic_is_likely_figure = bool(
        entry.get("figure_label")
        or (entry.get("caption_text") and image_type in {"Figure", "Picture"})
        or (image_type == "Figure" and has_size and not is_small)
    )
    entry["diagnostic_flags"] = flags
    entry["heuristic_is_likely_figure"] = heuristic_is_likely_figure
    if entry.get("figure_decision_source") != "llm_text":
        entry["is_likely_figure"] = heuristic_is_likely_figure
        entry["figure_decision_source"] = "heuristic"
    return entry


def apply_text_image_review(entry, review, model):
    """Apply a text-only LLM review response to an image manifest entry."""
    reviewed = dict(entry)
    heuristic = reviewed.get("heuristic_is_likely_figure")
    if heuristic is None:
        heuristic = reviewed.get("is_likely_figure")
    review_label = review.get("figure_label")
    review_number = review.get("figure_number")

    is_scientific_figure = bool(review.get("is_scientific_figure"))
    reviewed.update({
        "heuristic_is_likely_figure": bool(heuristic),
        "figure_decision_source": "llm_text",
        "is_likely_figure": is_scientific_figure,
        "image_reviewed": True,
        "image_review_method": "llm_text",
        "image_review_model": model,
        "image_review_classification": review.get("classification"),
        "image_review_is_scientific_figure": is_scientific_figure,
        "image_review_confidence": review.get("confidence"),
        "image_review_reason": review.get("reason"),
        "image_review_needs_vision": bool(review.get("needs_vision_review")),
        "image_review_error": None,
    })

    if review_number:
        reviewed["figure_number"] = review_number
    if review_label:
        reviewed["figure_label"] = review_label

    return reviewed


def strip_text_image_review(entry):
    """Return an image entry with cached LLM review fields removed."""
    stripped = dict(entry)
    stripped["figure_label"] = None
    stripped["figure_number"] = None
    stripped["figure_decision_source"] = None
    stripped["is_likely_figure"] = None
    stripped["image_reviewed"] = False
    stripped["image_review_method"] = None
    stripped["image_review_model"] = None
    stripped["image_review_classification"] = None
    stripped["image_review_is_scientific_figure"] = None
    stripped["image_review_confidence"] = None
    stripped["image_review_reason"] = None
    stripped["image_review_needs_vision"] = None
    stripped["image_review_error"] = None
    return annotate_image_diagnostics(stripped)


def write_image_manifest(images_dir, images):
    """Write image metadata manifest in the canonical on-disk shape."""
    manifest_path = os.path.join(images_dir, IMAGE_MANIFEST_FILENAME)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"images": images}, f, separators=(",", ":"))


def copy_image_metadata(source, target):
    """Copy optional image metadata fields from one image payload to another."""
    for key in IMAGE_METADATA_FIELDS:
        if key in source:
            target[key] = source.get(key)
    return target


def normalize_image_manifest_entry(entry, images_dir=None):
    """Normalize old string manifests and new object manifests to one shape."""
    if isinstance(entry, str):
        normalized = build_image_manifest_entry(entry)
    elif isinstance(entry, dict) and entry.get("filename"):
        filename = os.path.basename(entry["filename"])
        normalized = build_image_manifest_entry(filename)
        normalized.update(entry)
        normalized["filename"] = filename
    else:
        return None

    if images_dir:
        image_path = os.path.join(images_dir, normalized["filename"])
        if not os.path.exists(image_path):
            return None
        normalized["size_bytes"] = os.path.getsize(image_path)

    return annotate_image_diagnostics(normalized)


def list_manifest_images(images_dir):
    """Return image entries from a manifest, or None when no usable manifest exists."""
    manifest_path = os.path.join(images_dir, IMAGE_MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    entries = payload.get("images") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return None

    images = []
    for entry in entries:
        normalized = normalize_image_manifest_entry(entry, images_dir=images_dir)
        if normalized:
            images.append(normalized)
    return images


def list_image_directory(images_dir):
    """List images from a cache directory when no manifest is available."""
    images = []
    for filename in sorted(os.listdir(images_dir)):
        path = os.path.join(images_dir, filename)
        if os.path.isfile(path) and filename.lower().endswith(IMAGE_EXTENSIONS):
            images.append(normalize_image_manifest_entry({"filename": filename}, images_dir=images_dir))
    return [image for image in images if image]
