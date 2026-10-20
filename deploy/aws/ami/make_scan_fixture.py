"""Regenerate the image-only OCR fixture from the synthetic text sample."""

from pathlib import Path

import pypdfium2 as pdfium


def main():
    source = Path(__file__).with_name("test-sample.pdf")
    with pdfium.PdfDocument(source) as document:
        page = document[0]
        bitmap = page.render(scale=2)
        try:
            image = bitmap.to_pil().convert("RGB")
            image.save(source.with_name("test-scan.pdf"), "PDF", resolution=144)
        finally:
            bitmap.close()
            page.close()


if __name__ == "__main__":
    main()
