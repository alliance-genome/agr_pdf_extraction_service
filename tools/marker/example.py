#!/usr/bin/env python3
"""
Marker PDF to Markdown Conversion Example

This script demonstrates basic PDF to Markdown conversion using Marker.
It includes examples of:
- Basic conversion
- GPU/CPU detection and configuration
- Saving output to file
- Extracting and saving images
- Metadata extraction
"""

import os
import sys
import torch
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def convert_pdf_to_markdown(pdf_path: str, output_dir: str = "output") -> dict:
    """
    Convert a PDF file to Markdown using Marker.

    Args:
        pdf_path: Path to the PDF file to convert
        output_dir: Directory to save output files (default: "output")

    Returns:
        dict: Conversion results including text, images, and metadata
    """
    # Validate input
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Detect GPU/CPU and configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Create model artifacts with device configuration
    print("Loading models...")
    artifact_dict = create_model_dict(
        device=device,
        dtype=dtype,
    )

    # Initialize converter
    print("Initializing converter...")
    converter = PdfConverter(
        artifact_dict=artifact_dict,
    )

    # Convert the PDF
    print(f"Converting PDF: {pdf_path}")
    rendered = converter(pdf_path)

    # Extract text and images
    print("Extracting text and images...")
    text, file_ext, images = text_from_rendered(rendered)

    # Get metadata
    metadata = rendered.metadata
    num_pages = len(metadata.get('page_stats', []))

    print(f"Converted {num_pages} pages")
    print(f"Extracted {len(images)} images")

    # Save markdown output
    pdf_name = Path(pdf_path).stem
    output_file = os.path.join(output_dir, f"{pdf_name}.md")

    print(f"Saving markdown to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    # Save images
    if images:
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        print(f"Saving images to: {images_dir}")
        for img_name, img in images.items():
            img_path = os.path.join(images_dir, img_name)
            img.save(img_path, "PNG")
            print(f"  - Saved: {img_name}")

    # Save metadata
    metadata_file = os.path.join(output_dir, f"{pdf_name}_metadata.txt")
    print(f"Saving metadata to: {metadata_file}")
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write(f"PDF: {pdf_path}\n")
        f.write(f"Pages: {num_pages}\n")
        f.write(f"Images: {len(images)}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Data type: {dtype}\n")
        f.write("\nPage Statistics:\n")
        for i, stats in enumerate(metadata.get('page_stats', [])):
            f.write(f"  Page {i + 1}: {stats}\n")

    return {
        "text": text,
        "images": images,
        "metadata": metadata,
        "output_file": output_file,
        "num_pages": num_pages,
        "num_images": len(images),
    }


def main():
    """Main function to demonstrate PDF conversion."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python example.py <pdf_file> [output_dir]")
        print("\nExample:")
        print("  python example.py document.pdf")
        print("  python example.py document.pdf my_output")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    try:
        # Convert PDF
        results = convert_pdf_to_markdown(pdf_path, output_dir)

        # Print summary
        print("\n" + "=" * 60)
        print("CONVERSION COMPLETE")
        print("=" * 60)
        print(f"Input PDF: {pdf_path}")
        print(f"Output directory: {output_dir}")
        print(f"Markdown file: {results['output_file']}")
        print(f"Pages converted: {results['num_pages']}")
        print(f"Images extracted: {results['num_images']}")
        print("=" * 60)

        # Show first 500 characters of output
        print("\nFirst 500 characters of converted text:")
        print("-" * 60)
        print(results['text'][:500])
        if len(results['text']) > 500:
            print("...")
        print("-" * 60)

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
