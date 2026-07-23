import subprocess

from app.services.grobid_service import Grobid
from app.services.native_extractor_artifact import (
    load_native_extractor_artifact,
    load_native_style_artifact,
)
from app.services.native_style import validate_native_style_bytes


def test_grobid_extract_uses_alliance_converter_and_retains_exact_tei(
    monkeypatch, tmp_path
):
    tei = b"<TEI xmlns='http://www.tei-c.org/ns/1.0'><text><body/></text></TEI>"
    seen = {}
    grobid = Grobid("http://example.org", include_coordinates=True)
    monkeypatch.setattr(grobid, "is_alive", lambda: True)
    monkeypatch.setattr(grobid, "process_fulltext", lambda _path: tei)

    def convert(value, *, source_format):
        seen["value"] = value
        seen["source_format"] = source_format
        return "# Alliance title\n\nBody with *italics*."

    monkeypatch.setattr("app.services.grobid_service.convert_xml_to_markdown", convert)
    pdf = tmp_path / "paper.pdf"
    output = tmp_path / "grobid.md"
    pdf.write_bytes(b"fixture pdf")

    grobid.extract(pdf, output)

    assert output.read_text(encoding="utf-8") == "# Alliance title\n\nBody with *italics*."
    assert seen == {"value": tei, "source_format": "tei"}
    manifest, native = load_native_extractor_artifact(
        source="grobid", output_filename=output
    )
    assert native == tei
    assert manifest["options"]["include_coordinates"] is True
    assert manifest["expected_page_count"] is None


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
            "p", "head", "s", "figure", "biblStruct", "formula", "ref", "persName"
        ],
        "includeRawCitations": "1",
        "generateIDs": "1",
    }


def test_pdfalto_timeout_keeps_grobid_markdown_and_records_unavailable_style(
    monkeypatch, tmp_path
):
    tei = b"<TEI xmlns='http://www.tei-c.org/ns/1.0'><text><body/></text></TEI>"
    grobid = Grobid("http://example.org")
    monkeypatch.setattr(grobid, "is_alive", lambda: True)
    monkeypatch.setattr(grobid, "process_fulltext", lambda _path: tei)
    monkeypatch.setattr(
        "app.services.grobid_service.convert_xml_to_markdown",
        lambda *_args, **_kwargs: "# Title\n\nBody.",
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
