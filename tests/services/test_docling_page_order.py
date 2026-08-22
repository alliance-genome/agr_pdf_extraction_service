import importlib.util
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("docling") is None,
    reason="real Docling renderer dependencies are unavailable",
)
def test_real_docling_order_guard_and_explicit_traversal_preserve_markdown():
    script = textwrap.dedent(
        """
        from docling_core.types.doc import (
            BoundingBox,
            ContentLayer,
            CoordOrigin,
            DoclingDocument,
            ProvenanceItem,
        )

        from app.services.docling_service import (
            DOCLING_MARKDOWN_CONTENT_LAYERS,
            DOCLING_MARKDOWN_TRAVERSE_PICTURES,
            _docling_primary_page_order_is_monotonic,
        )
        from app.services.page_provenance import docling_markdown_with_page_ranges

        sentinel = "PDFX_DOCLING_PAGE_BOUNDARY_TEST"

        def provenance(page_number):
            return ProvenanceItem(
                page_no=page_number,
                bbox=BoundingBox(
                    l=0,
                    t=10,
                    r=10,
                    b=0,
                    coord_origin=CoordOrigin.BOTTOMLEFT,
                ),
                charspan=(0, 1),
            )

        def document(page_order):
            result = DoclingDocument(name="page-order-fixture")
            for index, page_number in enumerate(page_order):
                result.add_text(
                    label="text",
                    text=f"item {index} on page {page_number}",
                    prov=provenance(page_number),
                )
            return result

        def export(doc, placeholder, *, explicit=True):
            kwargs = {
                "image_placeholder": "",
                "page_break_placeholder": placeholder,
                "text_width": -1,
            }
            if explicit:
                kwargs.update(
                    included_content_layers=set(DOCLING_MARKDOWN_CONTENT_LAYERS),
                    traverse_pictures=DOCLING_MARKDOWN_TRAVERSE_PICTURES,
                )
            return doc.export_to_markdown(**kwargs)

        monotonic = document((1, 2, 3))
        assert DOCLING_MARKDOWN_CONTENT_LAYERS == frozenset({ContentLayer.BODY})
        assert export(monotonic, sentinel) == export(
            monotonic, sentinel, explicit=False
        )
        assert _docling_primary_page_order_is_monotonic(monotonic) is True

        revisited = document((1, 2, 1, 3))
        paginated = export(revisited, sentinel)
        legacy = export(revisited, "")
        order_is_safe = _docling_primary_page_order_is_monotonic(revisited)
        markdown, ranges = docling_markdown_with_page_ranges(
            paginated,
            sentinel=sentinel,
            expected_page_count=3,
            primary_page_order_is_monotonic=order_is_safe,
        )
        assert paginated.count(sentinel) == 2
        assert order_is_safe is False
        assert markdown == legacy.strip()
        assert ranges == []
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)
