import logging
import requests
from xml.etree import ElementTree as ET
from app.services.pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)


class Grobid(PDFExtractor):
    def __init__(self, base_url, timeout=120,
                 include_coordinates=False, include_raw_citations=False):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.include_coordinates = include_coordinates
        self.include_raw_citations = include_raw_citations
        self.tei_ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    def is_alive(self):
        try:
            response = requests.get(f"{self.base_url}/api/isalive", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def process_fulltext(self, pdf_path):
        url = f"{self.base_url}/api/processFulltextDocument"

        with open(pdf_path, "rb") as pdf_file:
            files = {"input": pdf_file}
            data = {}

            if self.include_coordinates:
                data["teiCoordinates"] = ["figure", "biblStruct", "formula", "ref", "persName"]

            if self.include_raw_citations:
                data["includeRawCitations"] = "1"

            try:
                response = requests.post(url, files=files, data=data, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Error processing PDF with GROBID: {e}")

    def extract_plain_text(self, tei_xml):
        try:
            tree = ET.fromstring(tei_xml)

            body_elem = tree.find(".//tei:text/tei:body", self.tei_ns)
            if body_elem is not None:
                text_parts = []
                for div in body_elem.findall(".//tei:div", self.tei_ns):
                    head = div.find("tei:head", self.tei_ns)
                    if head is not None and head.text:
                        text_parts.append(f"\n## {head.text}\n")

                    div_text = "".join(div.itertext()).strip()
                    if div_text:
                        text_parts.append(div_text)

                return "\n\n".join(text_parts)
            return None

        except ET.ParseError as e:
            raise RuntimeError(f"Error parsing TEI XML: {e}")

    def extract(self, pdf_path, output_filename):
        if not self.is_alive():
            raise RuntimeError("GROBID service is not running!")

        tei_xml = self.process_fulltext(pdf_path)

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(self.extract_plain_text(tei_xml))
