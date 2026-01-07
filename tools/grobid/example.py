#!/usr/bin/env python3
"""
GROBID PDF Extraction Example

This script demonstrates basic usage of GROBID for extracting text and metadata
from scientific PDFs.

Prerequisites:
- GROBID service running (default: http://localhost:8070)
- Python requests library: pip install requests

Usage:
    python example.py path/to/document.pdf
"""

import sys
import requests
from pathlib import Path
from xml.etree import ElementTree as ET


class GrobidClient:
    """Simple GROBID client for PDF extraction."""

    def __init__(self, base_url="http://localhost:8070"):
        """
        Initialize GROBID client.

        Args:
            base_url: Base URL of GROBID service (default: http://localhost:8070)
        """
        self.base_url = base_url.rstrip('/')
        self.tei_ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    def is_alive(self):
        """
        Check if GROBID service is running.

        Returns:
            bool: True if service is running, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/isalive", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def process_fulltext(self, pdf_path, include_coordinates=False, include_raw_citations=False):
        """
        Extract full text from PDF using GROBID.

        Args:
            pdf_path: Path to PDF file
            include_coordinates: If True, include TEI coordinates for elements
            include_raw_citations: If True, include original citation strings

        Returns:
            str: TEI XML output from GROBID, or None if error
        """
        url = f"{self.base_url}/api/processFulltextDocument"

        with open(pdf_path, 'rb') as pdf_file:
            files = {'input': pdf_file}
            data = {}

            if include_coordinates:
                # Request coordinates for various elements
                data['teiCoordinates'] = ['figure', 'biblStruct', 'formula', 'ref', 'persName']

            if include_raw_citations:
                data['includeRawCitations'] = '1'

            try:
                response = requests.post(url, files=files, data=data, timeout=60)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                print(f"Error processing PDF: {e}", file=sys.stderr)
                return None

    def process_header(self, pdf_path):
        """
        Extract only header metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            str: TEI XML output containing header metadata, or None if error
        """
        url = f"{self.base_url}/api/processHeaderDocument"

        with open(pdf_path, 'rb') as pdf_file:
            files = {'input': pdf_file}

            try:
                response = requests.post(url, files=files, timeout=30)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                print(f"Error processing header: {e}", file=sys.stderr)
                return None

    def extract_metadata(self, tei_xml):
        """
        Parse TEI XML to extract basic metadata.

        Args:
            tei_xml: TEI XML string from GROBID

        Returns:
            dict: Dictionary containing extracted metadata
        """
        try:
            tree = ET.fromstring(tei_xml)
            metadata = {}

            # Extract title
            title_elem = tree.find('.//tei:titleStmt/tei:title[@type="main"]', self.tei_ns)
            if title_elem is None:
                title_elem = tree.find('.//tei:titleStmt/tei:title', self.tei_ns)
            metadata['title'] = title_elem.text if title_elem is not None else "Unknown"

            # Extract authors
            authors = []
            author_elems = tree.findall('.//tei:sourceDesc//tei:author/tei:persName', self.tei_ns)
            for author in author_elems:
                forename = author.find('tei:forename', self.tei_ns)
                surname = author.find('tei:surname', self.tei_ns)
                if forename is not None and surname is not None:
                    authors.append(f"{forename.text} {surname.text}")
                elif surname is not None:
                    authors.append(surname.text)
            metadata['authors'] = authors

            # Extract abstract
            abstract_elem = tree.find('.//tei:abstract', self.tei_ns)
            if abstract_elem is not None:
                # Get all text content, handling nested elements
                abstract_text = ''.join(abstract_elem.itertext()).strip()
                metadata['abstract'] = abstract_text
            else:
                metadata['abstract'] = None

            # Extract publication date
            date_elem = tree.find('.//tei:publicationStmt//tei:date', self.tei_ns)
            metadata['date'] = date_elem.get('when') if date_elem is not None else None

            # Count references
            ref_elems = tree.findall('.//tei:listBibl/tei:biblStruct', self.tei_ns)
            metadata['reference_count'] = len(ref_elems)

            return metadata

        except ET.ParseError as e:
            print(f"Error parsing TEI XML: {e}", file=sys.stderr)
            return None

    def extract_plain_text(self, tei_xml):
        """
        Extract plain text from TEI XML body.

        Args:
            tei_xml: TEI XML string from GROBID

        Returns:
            str: Plain text content, or None if error
        """
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
            print(f"Error parsing TEI XML: {e}", file=sys.stderr)
            return None


def main():
    """Main function demonstrating GROBID usage."""

    if len(sys.argv) < 2:
        print("Usage: python example.py <path_to_pdf>")
        print("\nExample:")
        print("  python example.py document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Initialize GROBID client
    print("Initializing GROBID client...")
    client = GrobidClient()

    # Check if GROBID service is running
    if not client.is_alive():
        print("Error: GROBID service is not running!", file=sys.stderr)
        print("\nPlease start GROBID with:", file=sys.stderr)
        print("  docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2", file=sys.stderr)
        sys.exit(1)

    print("✓ GROBID service is running\n")

    # Process the PDF
    print(f"Processing PDF: {pdf_path}")
    tei_xml = client.process_fulltext(
        pdf_path,
        include_coordinates=False,
        include_raw_citations=False
    )

    if tei_xml is None:
        print("Failed to process PDF", file=sys.stderr)
        sys.exit(1)

    # Save the TEI XML output
    output_path = Path(pdf_path).stem + ".tei.xml"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(tei_xml)
    print(f"✓ Saved TEI XML to: {output_path}\n")

    # Extract and display metadata
    print("Extracting metadata...")
    metadata = client.extract_metadata(tei_xml)

    if metadata:
        print("\n" + "="*60)
        print("EXTRACTED METADATA")
        print("="*60)
        print(f"\nTitle: {metadata['title']}")

        if metadata['authors']:
            print(f"\nAuthors ({len(metadata['authors'])}):")
            for author in metadata['authors']:
                print(f"  - {author}")

        if metadata['date']:
            print(f"\nPublication Date: {metadata['date']}")

        print(f"\nReferences: {metadata['reference_count']}")

        if metadata['abstract']:
            print(f"\nAbstract:\n{metadata['abstract'][:300]}...")
            if len(metadata['abstract']) > 300:
                print(f"  (truncated, {len(metadata['abstract'])} chars total)")

    # Extract and display a sample of plain text
    print("\n" + "="*60)
    print("PLAIN TEXT SAMPLE")
    print("="*60)

    plain_text = client.extract_plain_text(tei_xml)
    if plain_text:
        # Show first 500 characters
        sample = plain_text[:500]
        print(f"\n{sample}...")
        print(f"\n(Total: {len(plain_text)} characters)")

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nFull TEI XML output saved to: {output_path}")
    print("You can parse this XML file to extract structured information.")


if __name__ == "__main__":
    main()
