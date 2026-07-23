import hashlib
import json

from app.services.merge_service import (
    completion_evidence_for_runtime_artifacts,
)
from app.services.source_contracts import SourceArtifact
from app.services.native_extractor_artifact import persist_native_extractor_artifact
from app.services.native_style import unavailable_native_style_bytes
from app.services.page_coverage import (
    PAGE_COVERAGE_METHOD,
    load_extractor_page_coverage,
    page_coverage_sidecar_path,
    verified_runtime_page_coverage,
    write_extractor_page_coverage,
)


def _write_artifact(tmp_path, source, pdf, text="# Title\n\nBody.\n", pages=2):
    path = tmp_path / f"{source}.md"
    path.write_text(text, encoding="utf-8")
    if source == "docling":
        native = {
            "schema_name": "DoclingDocument",
            "texts": [
                {
                    "self_ref": f"#/texts/{page - 1}",
                    "text": f"page {page}",
                    "prov": [{"page_no": page}],
                }
                for page in range(1, pages + 1)
            ],
        }
    else:
        native = {
            "block_type": "Document",
            "children": [
                {
                    "block_type": "Text",
                    "page_id": page - 1,
                    "html": f"<p>page {page}</p>",
                }
                for page in range(1, pages + 1)
            ],
        }
    persist_native_extractor_artifact(
        source=source,
        output_filename=path,
        native_bytes=json.dumps(native).encode(),
        native_media_type="application/json",
        pdf_path=pdf,
        extractor_versions=(
            {"docling": "2.113.0", "docling-core": "2.87.1"}
            if source == "docling"
            else {"marker-pdf": "1.10.2"}
        ),
        options=(
            {
                "do_ocr": True,
                "generate_parsed_pages": True,
                "native_style_cell_collection": "word_cells",
                "native_style_sidecar": True,
            }
            if source == "docling"
            else {"disable_links": True}
        ),
        expected_page_count=pages,
        covered_pages=list(range(1, pages + 1)),
        native_style_bytes=(
            unavailable_native_style_bytes("docling", "fixture")
            if source == "docling"
            else None
        ),
    )
    return path, SourceArtifact.from_text(source, text)


def _write_coverage(tmp_path, source, pdf, pages=2):
    path, artifact = _write_artifact(tmp_path, source, pdf, pages=pages)
    payload = write_extractor_page_coverage(
        source=source,
        output_filename=path,
        pdf_path=pdf,
        expected_page_count=pages,
        covered_pages=list(range(1, pages + 1)),
    )
    return path, artifact, payload


def test_page_coverage_sidecar_binds_pdf_markdown_and_native_artifact(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nexact fixture bytes")
    output, artifact, payload = _write_coverage(tmp_path, "docling", pdf, pages=3)

    loaded = load_extractor_page_coverage(
        source="docling",
        artifact=artifact,
        output_filename=output,
        pdf_path=pdf,
    )
    assert loaded == payload

    output.write_text("# Changed\n", encoding="utf-8")
    changed = SourceArtifact.from_text("docling", "# Changed\n")
    assert load_extractor_page_coverage(
        source="docling",
        artifact=changed,
        output_filename=output,
        pdf_path=pdf,
    ) is None


def test_runtime_page_coverage_accepts_each_independently_verified_source(
    tmp_path, monkeypatch
):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nexact private fixture bytes")
    paths = {}
    artifacts = {}
    for source in ("docling", "marker"):
        paths[source], artifacts[source], _ = _write_coverage(
            tmp_path, source, pdf, pages=2
        )
    monkeypatch.setattr("app.services.page_coverage._pdf_page_count", lambda _path: 2)

    verified = verified_runtime_page_coverage(
        {"docling": artifacts["docling"]},
        pdf_path=pdf,
        output_paths={"docling": paths["docling"]},
    )

    assert set(verified) == {"docling"}
    assert verified["docling"]["expected_page_count"] == 2
    assert verified["docling"]["coverage_method"] == PAGE_COVERAGE_METHOD
    assert verified["docling"]["pdf_digest"] == hashlib.sha256(
        pdf.read_bytes()
    ).hexdigest()

    monkeypatch.setattr("app.services.page_coverage._pdf_page_count", lambda _path: 3)
    assert verified_runtime_page_coverage(
        artifacts,
        pdf_path=pdf,
        output_paths=paths,
    ) == {}


def test_runtime_page_coverage_rejects_missing_native_payload_page(
    tmp_path, monkeypatch
):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nthree-page fixture")
    output, artifact = _write_artifact(
        tmp_path,
        "docling",
        pdf,
        pages=3,
    )
    manifest_path = f"{output}.native-manifest.json"
    native_path = f"{output}.native.document.json"
    native = json.loads(open(native_path, encoding="utf-8").read())
    native["texts"] = native["texts"][:2]
    native_bytes = json.dumps(native, separators=(",", ":")).encode()
    open(native_path, "wb").write(native_bytes)
    manifest = json.loads(open(manifest_path, encoding="utf-8").read())
    manifest["native_sha256"] = hashlib.sha256(native_bytes).hexdigest()
    manifest["native_size_bytes"] = len(native_bytes)
    manifest["covered_pages"] = [1, 2]
    manifest["page_coverage_status"] = "partial"
    open(manifest_path, "w", encoding="utf-8").write(json.dumps(manifest))
    write_extractor_page_coverage(
        source="docling",
        output_filename=output,
        pdf_path=pdf,
        expected_page_count=3,
        covered_pages=[1, 2],
    )
    monkeypatch.setattr("app.services.page_coverage._pdf_page_count", lambda _path: 3)

    assert verified_runtime_page_coverage(
        {"docling": artifact},
        pdf_path=pdf,
        output_paths={"docling": output},
    ) == {}


def test_runtime_completion_evidence_rejects_only_tampered_source(
    tmp_path, monkeypatch
):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncoverage fixture")
    paths = {}
    artifacts = {}
    for source in ("docling", "marker"):
        paths[source], artifacts[source], _ = _write_coverage(
            tmp_path, source, pdf, pages=2
        )
    monkeypatch.setattr("app.services.page_coverage._pdf_page_count", lambda _path: 2)

    marker_sidecar = page_coverage_sidecar_path(paths["marker"])
    payload = json.loads(open(marker_sidecar, encoding="utf-8").read())
    payload["expected_page_count"] = 3
    with open(marker_sidecar, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    evidence = completion_evidence_for_runtime_artifacts(
        artifacts,
        pdf_path=str(pdf),
        output_paths={source: str(path) for source, path in paths.items()},
    )

    assert evidence["docling"].page_coverage_verified is True
    assert evidence["docling"].completion_basis == "page_coverage"
    assert evidence["marker"].page_coverage_verified is False
    assert evidence["marker"].completion_basis == "synchronous_return"
