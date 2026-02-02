import sys
import requests
from pathlib import Path
from xml.etree import ElementTree as ET
from app.services.pdf_extractor import PDFExtractor

GROBID_URL = "http://localhost:8070"

class Grobid(PDFExtractor):
    def __init__(self, base_url=GROBID_URL):
        self.base_url = base_url.rstrip('/')
        self.tei_ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    def is_alive(self):
        try:
            response = requests.get(f"{self.base_url}/api/isalive", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def process_fulltext(self, pdf_path, include_coordinates=False, include_raw_citations=False):
        url = f"{self.base_url}/api/processFulltextDocument"

        with open(pdf_path, 'rb') as pdf_file:
            files = {'input': pdf_file}
            data = {}

            if include_coordinates:
                data['teiCoordinates'] = ['figure', 'biblStruct', 'formula', 'ref', 'persName']

            if include_raw_citations:
                data['includeRawCitations'] = '1'

            try:
                response = requests.post(url, files=files, data=data, timeout=60)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Error processing PDF: {e}", file=sys.stderr)

    def extract_plain_text(self, tei_xml):
        try:
            tree = ET.fromstring(tei_xml)

            # Extract all text from the body
            body_elem = tree.find('.//tei:text/tei:body', self.tei_ns)
            if body_elem is not None:
                # Get all text content, preserving some structure
                text_parts = []
                for div in body_elem.findall('.//tei:div', self.tei_ns):
                    # Get section heading if present
                    head = div.find('tei:head', self.tei_ns)
                    if head is not None and head.text:
                        text_parts.append(f"\n## {head.text}\n")

                    # Get all text in this section
                    div_text = ''.join(div.itertext()).strip()
                    if div_text:
                        text_parts.append(div_text)

                return '\n\n'.join(text_parts)
            return None

        except ET.ParseError as e:
            raise RuntimeError(f"Error parsing TEI XML: {e}", file=sys.stderr)

    def extract(self, pdf_path, output_filename):
        if not self.is_alive():
            raise RuntimeError("Error: GROBID service is not running!")

        tei_xml = self.process_fulltext(
            pdf_path,
            include_coordinates=False,
            include_raw_citations=False
        )

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(self.extract_plain_text(tei_xml))
