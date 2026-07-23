"""GROBID extraction with exact TEI retention and Alliance conversion."""

import logging
import os
from importlib.metadata import version

import requests
from agr_abc_document_parsers import convert_xml_to_markdown

from app.services.native_extractor_artifact import (
    persist_native_extractor_artifact,
)
from app.services.pdf_extractor import PDFExtractor
from app.services.native_style import (
    grobid_native_style_bytes,
    unavailable_native_style_bytes,
)

logger = logging.getLogger(__name__)
GROBID_VERSION = "0.8.2"


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
                data["teiCoordinates"] = [
                    "p", "head", "s", "figure", "biblStruct", "formula", "ref", "persName"
                ]

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

        tei_xml = self.process_fulltext(pdf_path)
        markdown = convert_xml_to_markdown(tei_xml, source_format="tei")
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

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(markdown)

        persist_native_extractor_artifact(
            source="grobid",
            output_filename=output_filename,
            native_bytes=tei_xml,
            native_media_type="application/tei+xml",
            pdf_path=pdf_path,
            extractor_versions={
                "grobid": GROBID_VERSION,
                "pdfalto": "0.5",
                "agr-abc-document-parsers": version("agr-abc-document-parsers"),
            },
            options={
                "generate_ids": True,
                "include_coordinates": self.include_coordinates,
                "include_raw_citations": self.include_raw_citations,
                "native_style_sidecar": True,
                "pdfalto_timeout_seconds": os.environ.get(
                    "PDFALTO_TIMEOUT_SECONDS", "900"
                ),
            },
            native_style_bytes=native_style_bytes,
        )
