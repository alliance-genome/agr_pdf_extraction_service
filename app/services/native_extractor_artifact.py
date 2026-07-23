"""Small manifest-last contract for extractor-native replay artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

from app.services.native_style import (
    NATIVE_STYLE_MEDIA_TYPE,
    validate_native_style_bytes,
)


NATIVE_ARTIFACT_SCHEMA = "pdfx-native-extractor-artifact"
NATIVE_ARTIFACT_CONTRACT_VERSION = "native-structure-v2"
_NATIVE_SUFFIX = {
    "grobid": "tei.xml",
    "docling": "document.json",
    "marker": "document.json",
}
_NATIVE_MEDIA_TYPE = {
    "grobid": "application/tei+xml",
    "docling": "application/json",
    "marker": "application/json",
}
_REQUIRED_OPTIONS = {
    "grobid": {"generate_ids": True, "native_style_sidecar": True},
    "docling": {
        "do_ocr": True,
        "generate_parsed_pages": True,
        "native_style_cell_collection": "word_cells",
        "native_style_sidecar": True,
    },
    "marker": {"disable_links": True},
}
_REQUIRED_EXTRACTOR_VERSIONS = {
    "grobid": {
        "grobid": "0.8.2",
        "agr-abc-document-parsers": "1.6.0",
    },
    "docling": {
        "docling": "2.113.0",
        "docling-core": "2.87.1",
    },
    "marker": {"marker-pdf": "1.10.2"},
}


def _matches_required_options(options: object, source: str) -> bool:
    if not isinstance(options, Mapping):
        return False
    return all(
        type(options.get(key)) is type(expected) and options.get(key) == expected
        for key, expected in _REQUIRED_OPTIONS[source].items()
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_artifact_path(
    output_filename: str | os.PathLike[str],
    source: str,
) -> Path:
    try:
        suffix = _NATIVE_SUFFIX[source]
    except KeyError as exc:
        raise ValueError(f"unsupported native extractor source: {source!r}") from exc
    return Path(f"{output_filename}.native.{suffix}")


def native_manifest_path(output_filename: str | os.PathLike[str]) -> Path:
    return Path(f"{output_filename}.native-manifest.json")


def native_style_artifact_path(output_filename: str | os.PathLike[str]) -> Path:
    return Path(f"{output_filename}.native.style.json")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def persist_native_extractor_artifact(
    *,
    source: str,
    output_filename: str | os.PathLike[str],
    native_bytes: bytes,
    native_media_type: str,
    pdf_path: str | os.PathLike[str],
    extractor_versions: Mapping[str, str],
    options: Mapping[str, object],
    expected_page_count: int | None = None,
    covered_pages: list[int] | tuple[int, ...] | None = None,
    native_style_bytes: bytes | None = None,
) -> dict:
    """Write the native object first and its compact binding manifest last."""

    if source not in _NATIVE_SUFFIX:
        raise ValueError(f"unsupported native extractor source: {source!r}")
    if not native_bytes:
        raise ValueError("native extractor artifact is empty")
    if native_media_type != _NATIVE_MEDIA_TYPE[source]:
        raise ValueError("native extractor media type is invalid")
    if not extractor_versions or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in extractor_versions.items()
    ):
        raise ValueError("native extractor versions are invalid")
    if any(
        extractor_versions.get(key) != expected
        for key, expected in _REQUIRED_EXTRACTOR_VERSIONS[source].items()
    ):
        raise ValueError("native extractor versions do not match runtime pins")
    if not _matches_required_options(options, source):
        raise ValueError("native extractor options are invalid")
    style_required = source in {"grobid", "docling"}
    if style_required and native_style_bytes is None:
        raise ValueError("native style artifact is required for this source")
    if native_style_bytes is not None:
        validate_native_style_bytes(source, native_style_bytes)
    output_path = Path(output_filename)
    markdown_bytes = output_path.read_bytes()
    if not markdown_bytes.strip():
        raise ValueError("extractor Markdown artifact is empty")

    if expected_page_count is None:
        canonical_pages = None
    else:
        if type(expected_page_count) is not int or expected_page_count < 1:
            raise ValueError("expected page count must be a positive integer")
        canonical_pages = list(covered_pages or [])
        if (
            canonical_pages != sorted(set(canonical_pages))
            or any(
                type(page) is not int or not 1 <= page <= expected_page_count
                for page in canonical_pages
            )
        ):
            raise ValueError("native covered pages must be a sorted PDF-page subset")

    native_path = native_artifact_path(output_path, source)
    _atomic_write(native_path, native_bytes)
    style_fields = {}
    if native_style_bytes is not None:
        style_path = native_style_artifact_path(output_path)
        _atomic_write(style_path, native_style_bytes)
        style_fields = {
            "native_style_filename": style_path.name,
            "native_style_media_type": NATIVE_STYLE_MEDIA_TYPE,
            "native_style_sha256": sha256_bytes(native_style_bytes),
            "native_style_size_bytes": len(native_style_bytes),
        }
    manifest = {
        "schema": NATIVE_ARTIFACT_SCHEMA,
        "contract_version": NATIVE_ARTIFACT_CONTRACT_VERSION,
        "source": source,
        "pdf_sha256": sha256_file(pdf_path),
        "markdown_filename": output_path.name,
        "markdown_sha256": sha256_bytes(markdown_bytes),
        "native_filename": native_path.name,
        "native_media_type": native_media_type,
        "native_sha256": sha256_bytes(native_bytes),
        "native_size_bytes": len(native_bytes),
        "extractor_versions": dict(sorted(extractor_versions.items())),
        "options": dict(sorted(options.items())),
        "expected_page_count": expected_page_count,
        "covered_pages": canonical_pages,
        "page_coverage_status": (
            "unavailable"
            if expected_page_count is None
            else "complete"
            if canonical_pages == list(range(1, expected_page_count + 1))
            else "partial"
        ),
        **style_fields,
    }
    _atomic_write(native_manifest_path(output_path), _json_bytes(manifest))
    return manifest


def load_native_style_artifact(
    *,
    source: str,
    output_filename: str | os.PathLike[str],
    manifest: Mapping,
) -> bytes | None:
    """Load and validate the digest-bound positive style evidence sidecar."""

    if source not in {"grobid", "docling"}:
        if any(str(key).startswith("native_style_") for key in manifest):
            raise ValueError("native style artifact is not supported for this source")
        return None
    style_path = native_style_artifact_path(output_filename)
    if manifest.get("native_style_filename") != style_path.name:
        raise ValueError("native style manifest filename is invalid")
    if manifest.get("native_style_media_type") != NATIVE_STYLE_MEDIA_TYPE:
        raise ValueError("native style media type is invalid")
    payload = style_path.read_bytes()
    if manifest.get("native_style_size_bytes") != len(payload):
        raise ValueError("native style artifact size mismatch")
    if manifest.get("native_style_sha256") != sha256_bytes(payload):
        raise ValueError("native style artifact digest mismatch")
    validate_native_style_bytes(source, payload)
    return payload


def load_native_extractor_artifact(
    *,
    source: str,
    output_filename: str | os.PathLike[str],
    expected_pdf_sha256: str | None = None,
) -> tuple[dict, bytes]:
    """Validate and return one native artifact bound to exact Markdown bytes."""

    output_path = Path(output_filename)
    manifest_path = native_manifest_path(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("native manifest must be a JSON object")
    if manifest.get("schema") != NATIVE_ARTIFACT_SCHEMA:
        raise ValueError("native manifest schema is invalid")
    if manifest.get("contract_version") != NATIVE_ARTIFACT_CONTRACT_VERSION:
        raise ValueError("native manifest contract version is invalid")
    if manifest.get("source") != source:
        raise ValueError("native manifest source is invalid")
    if manifest.get("markdown_filename") != output_path.name:
        raise ValueError("native manifest Markdown filename is invalid")
    if manifest.get("markdown_sha256") != sha256_file(output_path):
        raise ValueError("native manifest Markdown digest mismatch")
    if manifest.get("native_media_type") != _NATIVE_MEDIA_TYPE[source]:
        raise ValueError("native manifest media type is invalid")
    extractor_versions = manifest.get("extractor_versions")
    if (
        not isinstance(extractor_versions, dict)
        or not extractor_versions
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in extractor_versions.items()
        )
    ):
        raise ValueError("native manifest extractor versions are invalid")
    if any(
        extractor_versions.get(key) != expected
        for key, expected in _REQUIRED_EXTRACTOR_VERSIONS[source].items()
    ):
        raise ValueError("native manifest extractor versions do not match runtime pins")
    options = manifest.get("options")
    if not _matches_required_options(options, source):
        raise ValueError("native manifest extractor options are invalid")
    if expected_pdf_sha256 is not None and manifest.get("pdf_sha256") != expected_pdf_sha256:
        raise ValueError("native manifest PDF digest mismatch")

    native_path = native_artifact_path(output_path, source)
    if manifest.get("native_filename") != native_path.name:
        raise ValueError("native manifest filename is invalid")
    native_bytes = native_path.read_bytes()
    if manifest.get("native_size_bytes") != len(native_bytes):
        raise ValueError("native artifact size mismatch")
    if manifest.get("native_sha256") != sha256_bytes(native_bytes):
        raise ValueError("native artifact digest mismatch")
    expected_page_count = manifest.get("expected_page_count")
    covered_pages = manifest.get("covered_pages")
    expected_status = (
        "unavailable"
        if expected_page_count is None
        else "complete"
        if covered_pages == list(range(1, expected_page_count + 1))
        else "partial"
    )
    if manifest.get("page_coverage_status") != expected_status:
        raise ValueError("native page coverage status is invalid")
    if expected_page_count is None:
        if covered_pages is not None:
            raise ValueError("native page coverage receipt is invalid")
    elif (
        type(expected_page_count) is not int
        or expected_page_count < 1
        or not isinstance(covered_pages, list)
        or covered_pages != sorted(set(covered_pages))
        or any(
            type(page) is not int or not 1 <= page <= expected_page_count
            for page in covered_pages
        )
    ):
        raise ValueError("native page coverage receipt is invalid")
    load_native_style_artifact(
        source=source,
        output_filename=output_path,
        manifest=manifest,
    )
    return manifest, native_bytes


def has_valid_native_extractor_artifact(
    *,
    source: str,
    output_filename: str | os.PathLike[str],
) -> bool:
    try:
        load_native_extractor_artifact(
            source=source,
            output_filename=output_filename,
        )
        return True
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
