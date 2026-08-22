import subprocess
from types import SimpleNamespace

from app.services.grobid_service import Grobid
from app.services.document_skeleton import NativeStructureArtifact, build_document_skeleton
from app.services.native_extractor_artifact import (
    load_native_extractor_artifact,
    load_native_style_artifact,
)
from app.services.native_style import validate_native_style_bytes
from app.services.source_contracts import SourceArtifact


def test_grobid_extract_uses_alliance_converter_and_retains_exact_tei(
    monkeypatch, tmp_path
):
    tei = b"<TEI xmlns='http://www.tei-c.org/ns/1.0'><text><body/></text></TEI>"
    seen = {}
    grobid = Grobid("http://example.org", include_coordinates=True)
    monkeypatch.setattr(grobid, "is_alive", lambda: True)
    monkeypatch.setattr(grobid, "process_fulltext", lambda _path: tei)

    def convert(value):
        seen["value"] = value
        return SimpleNamespace(
            markdown="# Alliance title\n\nBody with *italics*.",
            spans=(
                SimpleNamespace(
                    byte_start=0,
                    byte_end=len("# Alliance title".encode()),
                    page_numbers=(1,),
                    native_id="title-1",
                    kind="title",
                ),
            ),
        )

    monkeypatch.setattr(
        "app.services.grobid_service.convert_tei_to_markdown_with_provenance",
        convert,
    )
    monkeypatch.setattr("app.services.grobid_service.pdf_page_count", lambda _path: 2)
    monkeypatch.setattr(
        "app.services.grobid_service.version",
        lambda package: "1.7.0" if package == "agr-abc-document-parsers" else "0",
    )
    pdf = tmp_path / "paper.pdf"
    output = tmp_path / "grobid.md"
    pdf.write_bytes(b"fixture pdf")

    grobid.extract(pdf, output)

    assert output.read_text(encoding="utf-8") == "# Alliance title\n\nBody with *italics*."
    assert seen == {"value": tei}
    manifest, native = load_native_extractor_artifact(
        source="grobid", output_filename=output
    )
    assert native == tei
    assert manifest["options"]["include_coordinates"] is True
    assert manifest["expected_page_count"] is None
    assert manifest["covered_pages"] is None
    assert manifest["page_coverage_status"] == "unavailable"
    assert manifest["page_provenance_filename"].endswith("page-provenance.json")
    artifact = SourceArtifact.from_text("grobid", output.read_text(encoding="utf-8"))
    native_structure = NativeStructureArtifact.from_loaded(
        source="grobid",
        markdown=artifact,
        manifest=manifest,
        native_bytes=native,
        native_style_bytes=load_native_style_artifact(
            source="grobid", output_filename=output, manifest=manifest
        ),
    )
    skeleton = build_document_skeleton(artifact, native_structure)
    assert skeleton.expected_page_count is None
    assert skeleton.covered_page_count is None


def test_grobid_request_enables_ids_and_coordinates(monkeypatch, tmp_path):
    request = {}

    class Response:
        content = b"<TEI/>"

        def raise_for_status(self):
            return None

    def post(url, *, files, data, timeout):
        request.update(url=url, files=files, data=data, timeout=timeout)
        return Response()

    monkeypatch.setattr("app.services.grobid_service.requests.post", post)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"fixture pdf")

    result = Grobid(
        "http://example.org/",
        timeout=17,
        include_coordinates=True,
        include_raw_citations=True,
    ).process_fulltext(pdf)

    assert result == b"<TEI/>"
    assert request["url"] == "http://example.org/api/processFulltextDocument"
    assert request["timeout"] == 17
    assert request["data"] == {
        "teiCoordinates": [
            "p", "head", "figure", "biblStruct", "formula", "ref", "persName",
            "title", "affiliation", "note"
        ],
        "includeRawCitations": "1",
        "generateIDs": "1",
    }


def test_pdfalto_timeout_keeps_grobid_markdown_and_records_unavailable_style(
    monkeypatch, tmp_path
):
    tei = b"<TEI xmlns='http://www.tei-c.org/ns/1.0'><text><body/></text></TEI>"
    grobid = Grobid("http://example.org", include_coordinates=True)
    monkeypatch.setattr(grobid, "is_alive", lambda: True)
    monkeypatch.setattr(grobid, "process_fulltext", lambda _path: tei)
    monkeypatch.setattr(
        "app.services.grobid_service.convert_tei_to_markdown_with_provenance",
        lambda *_args, **_kwargs: SimpleNamespace(
            markdown="# Title\n\nBody.", spans=()
        ),
    )
    monkeypatch.setattr("app.services.grobid_service.pdf_page_count", lambda _path: 1)
    monkeypatch.setattr(
        "app.services.grobid_service.version",
        lambda package: "1.7.0" if package == "agr-abc-document-parsers" else "0",
    )

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("pdfalto", 900)

    monkeypatch.setattr(
        "app.services.grobid_service.grobid_native_style_bytes", timeout
    )
    pdf = tmp_path / "paper.pdf"
    output = tmp_path / "grobid.md"
    pdf.write_bytes(b"fixture pdf")

    grobid.extract(pdf, output)

    manifest, native = load_native_extractor_artifact(
        source="grobid", output_filename=output
    )
    style = load_native_style_artifact(
        source="grobid", output_filename=output, manifest=manifest
    )
    assert output.read_text(encoding="utf-8") == "# Title\n\nBody."
    assert native == tei
    assert validate_native_style_bytes("grobid", style)["status"] == "unavailable"
