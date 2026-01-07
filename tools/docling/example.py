#!/usr/bin/env python3
"""
Docling PDF Extraction Example

This script demonstrates basic PDF text extraction using Docling.
It shows how to:
1. Convert a PDF document to Docling format
2. Extract text content
3. Export to different formats (Markdown, JSON)
4. Access document structure (headers, tables, metadata)

Usage:
    python example.py <pdf_path>
    python example.py test_pdfs/core/AGRKB_101000000645569_mbc-31-1411.pdf
"""

import json
import sys

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel


def basic_extraction(pdf_path):
    """
    Basic PDF extraction example - convert and export to Markdown

    Args:
        pdf_path: Path to PDF file (local path or URL)
    """
    print(f"\n=== Basic Extraction from {pdf_path} ===\n")

    # Initialize converter
    converter = DocumentConverter()

    # Convert document
    result = converter.convert(pdf_path)
    doc = result.document

    # Export to Markdown
    markdown = doc.export_to_markdown()
    print("Markdown Output (first 500 chars):")
    print(markdown[:500])
    print("...\n")

    return result


def extract_document_structure(pdf_path):
    """
    Extract and display document structure elements

    Args:
        pdf_path: Path to PDF file (local path or URL)
    """
    print(f"\n=== Document Structure Analysis ===\n")

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    # Extract section headers
    print("Section Headers:")
    headers = [item for item in doc.iterate_items()
               if item.label == DocItemLabel.SECTION_HEADER]
    for i, header in enumerate(headers[:5], 1):  # Show first 5
        print(f"  {i}. {header.text}")
    if len(headers) > 5:
        print(f"  ... and {len(headers) - 5} more headers")
    print()

    # Extract text paragraphs
    print("Text Paragraphs:")
    paragraphs = [item for item in doc.iterate_items()
                  if item.label == DocItemLabel.PARAGRAPH]
    for i, para in enumerate(paragraphs[:3], 1):  # Show first 3
        text_preview = para.text[:100] + "..." if len(para.text) > 100 else para.text
        print(f"  {i}. {text_preview}")
    if len(paragraphs) > 3:
        print(f"  ... and {len(paragraphs) - 3} more paragraphs")
    print()

    # Extract tables
    print(f"Tables: {len(doc.tables)} found")
    for i, table in enumerate(doc.tables[:2], 1):  # Show first 2
        df = table.export_to_dataframe()
        print(f"  Table {i}: {df.shape[0]} rows x {df.shape[1]} columns")
    print()

    # Document metadata
    print("Document Metadata:")
    print(f"  Pages: {result.input.page_count}")
    print(f"  File size: {result.input.filesize} bytes")
    print(f"  Format: {result.input.format}")
    print(f"  Status: {result.status}")
    print()


def export_to_multiple_formats(pdf_path, output_prefix="output"):
    """
    Export document to multiple formats

    Args:
        pdf_path: Path to PDF file (local path or URL)
        output_prefix: Prefix for output files
    """
    print(f"\n=== Exporting to Multiple Formats ===\n")

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    # Export to Markdown
    markdown = doc.export_to_markdown()
    markdown_file = f"{output_prefix}.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown)
    print(f"Exported to Markdown: {markdown_file}")

    # Export to JSON
    json_dict = doc.export_to_dict()
    json_file = f"{output_prefix}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, indent=2)
    print(f"Exported to JSON: {json_file}")

    # Export to HTML
    html = doc.export_to_html()
    html_file = f"{output_prefix}.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Exported to HTML: {html_file}")

    print("\nAll exports completed successfully!")


def extract_all_text(pdf_path):
    """
    Extract all text content from the PDF

    Args:
        pdf_path: Path to PDF file (local path or URL)

    Returns:
        str: All text content from the document
    """
    print(f"\n=== Extracting All Text ===\n")

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    # Collect all text from all items
    all_text = []
    for item in doc.iterate_items():
        if hasattr(item, 'text') and item.text:
            all_text.append(item.text)

    full_text = "\n".join(all_text)

    print(f"Total characters extracted: {len(full_text)}")
    print(f"Total items processed: {len(all_text)}")
    print("\nFirst 500 characters of extracted text:")
    print(full_text[:500])
    print("...\n")

    return full_text


def main():
    """
    Main function demonstrating Docling PDF extraction
    """
    # Get PDF path from command line or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Default: Docling technical report from arXiv
        pdf_path = "https://arxiv.org/pdf/2408.09869"
        print("No PDF path provided, using default example.")

    print("=" * 60)
    print("Docling PDF Extraction Example")
    print(f"Input: {pdf_path}")
    print("=" * 60)

    # Run basic extraction
    result = basic_extraction(pdf_path)

    # Analyze document structure
    extract_document_structure(pdf_path)

    # Extract all text
    text = extract_all_text(pdf_path)

    # Export to multiple formats
    # Uncomment the line below to save files locally
    # export_to_multiple_formats(pdf_path, "docling_example_output")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
