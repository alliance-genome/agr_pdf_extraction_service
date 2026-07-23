"""Digest-bound extractor page-coverage evidence for candidate qualification."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

from app.services.source_contracts import SourceArtifact, SourceName
from app.services.native_extractor_artifact import load_native_extractor_artifact


PAGE_COVERAGE_SIDECAR_SCHEMA = "pdfx-extractor-page-coverage"
PAGE_COVERAGE_PROOF_SCHEMA = "pdfx-page-coverage-proof"
PAGE_COVERAGE_METHOD = "pdfium_page_count_plus_native_provenance"

logger = logging.getLogger(__name__)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_page_count(path: str | os.PathLike[str]) -> int:
    """Read the PDF container page count independently of either extractor."""

    from pypdfium2 import PdfDocument

    document = PdfDocument(path)
    try:
        count = len(document)
    finally:
        document.close()
    if count < 1:
        raise ValueError("PDF has no pages")
    return count


def page_coverage_sidecar_path(output_filename: str | os.PathLike[str]) -> str:
    return f"{output_filename}.page-coverage.json"


def native_payload_covered_pages(source: SourceName, native_bytes: bytes) -> list[int]:
    """Return page numbers actually represented by native extractor blocks."""

    if source not in {"docling", "marker"}:
        raise ValueError("native page provenance is unavailable for this source")
    document = json.loads(native_bytes)
    if not isinstance(document, Mapping):
        raise ValueError("native page provenance root is invalid")
    pages: set[int] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            if source == "docling":
                provenance = value.get("prov")
                if isinstance(provenance, list):
                    for item in provenance:
                        if isinstance(item, Mapping):
                            page = item.get("page_no")
                            if type(page) is int and page >= 1:
                                pages.add(page)
            else:
                page_id = value.get("page_id")
                block_type = str(value.get("block_type", "")).rsplit(".", 1)[-1]
                if type(page_id) is int and page_id >= 0 and block_type != "Document":
                    pages.add(page_id + 1)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    return sorted(pages)


def write_extractor_page_coverage(
    *,
    source: SourceName,
    output_filename: str | os.PathLike[str],
    pdf_path: str | os.PathLike[str],
    expected_page_count: int,
    covered_pages: list[int] | tuple[int, ...],
) -> dict:
    """Atomically bind extractor-native page coverage to exact Markdown bytes."""

    if type(expected_page_count) is not int or expected_page_count < 1:
        raise ValueError("expected page count must be a positive integer")
    canonical_pages = sorted(set(covered_pages))
    if (
        any(
            type(page) is not int or not 1 <= page <= expected_page_count
            for page in canonical_pages
        )
        or canonical_pages != list(covered_pages)
    ):
        raise ValueError("extractor covered pages must be a sorted PDF-page subset")
    output_path = Path(output_filename)
    native_manifest, native_bytes = load_native_extractor_artifact(
        source=source,
        output_filename=output_path,
    )
    pdf_digest = _sha256_file(pdf_path)
    if native_manifest.get("pdf_sha256") != pdf_digest:
        raise ValueError("native artifact is bound to a different PDF")
    if native_manifest.get("expected_page_count") != expected_page_count:
        raise ValueError("native artifact page count does not match coverage")
    if native_manifest.get("covered_pages") != canonical_pages:
        raise ValueError("native artifact page provenance does not match coverage")
    if native_payload_covered_pages(source, native_bytes) != canonical_pages:
        raise ValueError("native block page provenance does not match coverage")
    core = {
        "schema": PAGE_COVERAGE_SIDECAR_SCHEMA,
        "source": source,
        "artifact_sha256": _sha256(output_path.read_bytes()),
        "native_artifact_sha256": native_manifest["native_sha256"],
        "pdf_sha256": pdf_digest,
        "expected_page_count": expected_page_count,
        "covered_pages": canonical_pages,
    }
    payload = {
        **core,
        "record_sha256": _sha256(_canonical_json_bytes(core)),
    }
    sidecar = Path(page_coverage_sidecar_path(output_path))
    temporary = sidecar.with_name(f"{sidecar.name}.tmp.{os.getpid()}")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, sidecar)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return payload


def load_extractor_page_coverage(
    *,
    source: SourceName,
    artifact: SourceArtifact,
    output_filename: str | os.PathLike[str],
    pdf_path: str | os.PathLike[str],
) -> dict | None:
    """Return a validated sidecar or None; never trust stale cached metadata."""

    sidecar = Path(page_coverage_sidecar_path(output_filename))
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    expected_keys = {
        "schema",
        "source",
        "artifact_sha256",
        "native_artifact_sha256",
        "pdf_sha256",
        "expected_page_count",
        "covered_pages",
        "record_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        return None
    core = {key: payload[key] for key in expected_keys - {"record_sha256"}}
    expected_count = payload.get("expected_page_count")
    covered_pages = payload.get("covered_pages")
    try:
        native_manifest, _native_bytes = load_native_extractor_artifact(
            source=source,
            output_filename=output_filename,
            expected_pdf_sha256=_sha256_file(pdf_path),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema") != PAGE_COVERAGE_SIDECAR_SCHEMA
        or payload.get("source") != source
        or payload.get("artifact_sha256") != artifact.digest
        or payload.get("native_artifact_sha256") != native_manifest.get("native_sha256")
        or payload.get("pdf_sha256") != native_manifest.get("pdf_sha256")
        or type(expected_count) is not int
        or expected_count < 1
        or not isinstance(covered_pages, list)
        or covered_pages != native_manifest.get("covered_pages")
        or covered_pages != sorted(set(covered_pages))
        or any(
            type(page) is not int or not 1 <= page <= expected_count
            for page in covered_pages
        )
        or native_payload_covered_pages(source, _native_bytes) != covered_pages
        or payload.get("record_sha256") != _sha256(_canonical_json_bytes(core))
    ):
        return None
    return dict(payload)


def page_coverage_proof_digest(
    *,
    source: SourceName,
    artifact_digest: str,
    pdf_digest: str,
    expected_page_count: int,
    covered_page_count: int,
    coverage_method: str = PAGE_COVERAGE_METHOD,
) -> str:
    return _sha256(
        _canonical_json_bytes(
            {
                "schema": PAGE_COVERAGE_PROOF_SCHEMA,
                "source": source,
                "artifact_sha256": artifact_digest,
                "pdf_sha256": pdf_digest,
                "expected_page_count": expected_page_count,
                "covered_page_count": covered_page_count,
                "coverage_method": coverage_method,
            }
        )
    )


def validate_page_coverage_evidence_metric(
    value: object,
    *,
    expected_sources: set[str] | None = None,
) -> dict[str, dict]:
    """Validate the canonical privacy-safe runtime evidence projection."""

    if not isinstance(value, Mapping):
        raise ValueError("page-coverage evidence is not a mapping")
    if expected_sources is not None and set(value) != expected_sources:
        raise ValueError("page-coverage evidence source set is invalid")
    canonical = {}
    expected_keys = {
        "artifact_digest",
        "completion_basis",
        "expected_page_count",
        "covered_page_count",
        "pdf_digest",
        "coverage_method",
        "page_coverage_digest",
        "page_coverage_verified",
    }
    for source in sorted(value):
        item = value[source]
        if source not in {"grobid", "docling", "marker"}:
            raise ValueError("page-coverage evidence source is invalid")
        if not isinstance(item, Mapping) or set(item) != expected_keys:
            raise ValueError("page-coverage evidence item shape is invalid")
        artifact_digest = item.get("artifact_digest")
        basis = item.get("completion_basis")
        verified = item.get("page_coverage_verified")
        if (
            not isinstance(artifact_digest, str)
            or len(artifact_digest) != 64
            or any(char not in "0123456789abcdef" for char in artifact_digest)
            or basis
            not in {
                "caller_asserted",
                "synchronous_return",
                "saved_artifact_replay",
                "page_coverage",
            }
            or type(verified) is not bool
        ):
            raise ValueError("page-coverage evidence item binding is invalid")
        if verified:
            expected_count = item.get("expected_page_count")
            covered_count = item.get("covered_page_count")
            pdf_digest = item.get("pdf_digest")
            proof_digest = item.get("page_coverage_digest")
            if (
                basis != "page_coverage"
                or type(expected_count) is not int
                or expected_count < 1
                or covered_count != expected_count
                or not isinstance(pdf_digest, str)
                or len(pdf_digest) != 64
                or any(char not in "0123456789abcdef" for char in pdf_digest)
                or item.get("coverage_method") != PAGE_COVERAGE_METHOD
                or proof_digest
                != page_coverage_proof_digest(
                    source=source,
                    artifact_digest=artifact_digest,
                    pdf_digest=pdf_digest,
                    expected_page_count=expected_count,
                    covered_page_count=covered_count,
                )
            ):
                raise ValueError("page-coverage proof is invalid")
        elif basis == "page_coverage" or any(
            item.get(key) is not None
            for key in (
                "expected_page_count",
                "covered_page_count",
                "pdf_digest",
                "coverage_method",
                "page_coverage_digest",
            )
        ):
            raise ValueError("unverified page-coverage evidence is not empty")
        canonical[source] = dict(item)
    return canonical


def verified_runtime_page_coverage(
    artifacts: Mapping[SourceName, SourceArtifact],
    *,
    pdf_path: str | os.PathLike[str],
    output_paths: Mapping[SourceName, str | os.PathLike[str]],
) -> dict[SourceName, dict]:
    """Compare each extractor's native page provenance with the PDF itself."""

    try:
        independent_page_count = _pdf_page_count(pdf_path)
        pdf_digest = _sha256_file(pdf_path)
    except Exception as exc:
        logger.warning(
            "Page-coverage evidence unavailable: %s",
            type(exc).__name__,
        )
        return {}

    records = {}
    for source, artifact in artifacts.items():
        output_path = output_paths.get(source)
        if output_path is None:
            continue
        record = load_extractor_page_coverage(
            source=source,
            artifact=artifact,
            output_filename=output_path,
            pdf_path=pdf_path,
        )
        if (
            record is not None
            and record["expected_page_count"] == independent_page_count
            and record["covered_pages"]
            == list(range(1, independent_page_count + 1))
            and record["pdf_sha256"] == pdf_digest
        ):
            records[source] = record
    return {
        source: {
            "pdf_digest": pdf_digest,
            "expected_page_count": independent_page_count,
            "covered_page_count": len(records[source]["covered_pages"]),
            "coverage_method": PAGE_COVERAGE_METHOD,
            "page_coverage_digest": page_coverage_proof_digest(
                source=source,
                artifact_digest=artifacts[source].digest,
                pdf_digest=pdf_digest,
                expected_page_count=independent_page_count,
                covered_page_count=len(records[source]["covered_pages"]),
            ),
        }
        for source in sorted(records)
    }
