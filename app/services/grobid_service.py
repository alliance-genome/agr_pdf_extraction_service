"""GROBID extraction with exact TEI retention and Alliance conversion."""

import logging
import os
from importlib.metadata import version
from pathlib import Path

import requests
from agr_abc_document_parsers import convert_tei_to_markdown_with_provenance

from app.services.native_extractor_artifact import (
    persist_native_extractor_artifact,
    sha256_file,
)
from app.services.page_coverage import pdf_page_count
from app.services.page_provenance import build_source_page_provenance_bytes
from app.services.pdf_extractor import PDFExtractor
from app.services.native_style import (
    grobid_native_style_bytes,
    unavailable_native_style_bytes,
)

logger = logging.getLogger(__name__)
GROBID_VERSION = "0.8.2"
GROBID_COORDINATE_ELEMENTS = (
    "p",
    "head",
    "figure",
    "biblStruct",
    "formula",
    "ref",
    "persName",
    "title",
    "affiliation",
    "note",
)


class Grobid(PDFExtractor):
    def __init__(self, base_url, timeout=120,
                 include_coordinates=False, include_raw_citations=False):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.include_coordinates = include_coordinates
        self.include_raw_citations = include_raw_citations

    def is_alive(self):
        try:
            response = requests.get(f"{self.base_url}/api/isalive", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.warning("GROBID health check failed: %s", e)
            return False

    def process_fulltext(self, pdf_path):
        url = f"{self.base_url}/api/processFulltextDocument"
        logger.info("GROBID: sending PDF to %s (timeout=%ds)", url, self.timeout)

        with open(pdf_path, "rb") as pdf_file:
            files = {"input": pdf_file}
            data = {}

            if self.include_coordinates:
                data["teiCoordinates"] = list(GROBID_COORDINATE_ELEMENTS)

            if self.include_raw_citations:
                data["includeRawCitations"] = "1"
            data["generateIDs"] = "1"

            try:
                response = requests.post(url, files=files, data=data, timeout=self.timeout)
                response.raise_for_status()
                logger.info("GROBID: received TEI XML (%d bytes)", len(response.content))
                return response.content
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Error processing PDF with GROBID: {e}")

    def extract(self, pdf_path, output_filename):
        if not self.is_alive():
            raise RuntimeError("GROBID service is not running!")
        if not self.include_coordinates:
            raise RuntimeError("GROBID page provenance requires TEI coordinates")

        tei_xml = self.process_fulltext(pdf_path)
        emission = convert_tei_to_markdown_with_provenance(tei_xml)
        markdown = emission.markdown
        if not markdown.strip():
            raise RuntimeError("GROBID TEI produced no Alliance Markdown")

        logger.info("GROBID: converted TEI to %d chars of Alliance Markdown", len(markdown))

        try:
            pdfalto_timeout_seconds = float(
                os.environ.get("PDFALTO_TIMEOUT_SECONDS", "900")
            )
            if pdfalto_timeout_seconds <= 0:
                raise ValueError("PDFALTO_TIMEOUT_SECONDS must be positive")
            native_style_bytes = grobid_native_style_bytes(
                pdf_path,
                pdfalto_path=os.environ.get("PDFALTO_PATH", "/usr/local/bin/pdfalto"),
                timeout_seconds=pdfalto_timeout_seconds,
            )
        except Exception as exc:
            logger.warning(
                "GROBID PDFALTO style capture unavailable: %s", type(exc).__name__
            )
            native_style_bytes = unavailable_native_style_bytes(
                "grobid", type(exc).__name__
            )

        expected_page_count = pdf_page_count(pdf_path)
        pdf_digest = sha256_file(pdf_path)
        extractor_versions = {
            "grobid": GROBID_VERSION,
            "pdfalto": "0.5",
            "agr-abc-document-parsers": version("agr-abc-document-parsers"),
        }
        options = {
            "generate_ids": True,
            "include_coordinates": True,
            "coordinate_elements": list(GROBID_COORDINATE_ELEMENTS),
            "include_raw_citations": self.include_raw_citations,
            "native_style_sidecar": True,
            "page_provenance": "tei_coords_v1",
            "pdfalto_timeout_seconds": os.environ.get(
                "PDFALTO_TIMEOUT_SECONDS", "900"
            ),
        }
        page_ranges = []
        for span in emission.spans:
            candidate_pages = [
                page
                for page in span.page_numbers
                if type(page) is int and 1 <= page <= expected_page_count
            ]
            candidate_pages = list(dict.fromkeys(candidate_pages))
            if not candidate_pages:
                continue
            page_ranges.append(
                {
                    "byte_start": span.byte_start,
                    "byte_end": span.byte_end,
                    "page_number": candidate_pages[0],
                    "candidate_pages": candidate_pages,
                    "method": (
                        "direct" if len(candidate_pages) == 1 else "native_start_page"
                    ),
                    "native_id": span.native_id,
                    "kind": span.kind,
                    "residual_reason": None,
                }
            )
        markdown_bytes = markdown.encode("utf-8")
        page_provenance_bytes = build_source_page_provenance_bytes(
            source="grobid",
            pdf_sha256=pdf_digest,
            native_bytes=tei_xml,
            markdown_bytes=markdown_bytes,
            expected_page_count=expected_page_count,
            extractor_versions=extractor_versions,
            options=options,
            evidence_ranges=page_ranges,
            residual_reason="tei_coordinates_unavailable",
        )
        Path(output_filename).write_bytes(markdown_bytes)

        persist_native_extractor_artifact(
            source="grobid",
            output_filename=output_filename,
            native_bytes=tei_xml,
            native_media_type="application/tei+xml",
            pdf_path=pdf_path,
            extractor_versions=extractor_versions,
            options=options,
            page_provenance_bytes=page_provenance_bytes,
            pdf_sha256=pdf_digest,
            native_style_bytes=native_style_bytes,
        )
