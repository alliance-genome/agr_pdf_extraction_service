import json
from dataclasses import replace

import pytest

from app.services.document_skeleton import (
    NativeEmphasisSpan,
    NativeStructureArtifact,
    _final_projection_targets,
    _projection_claim_inventory,
    build_document_skeleton,
    choose_document_skeleton,
    load_runtime_native_structures,
    native_heading_hints,
    native_occurrence_hints,
    project_native_emphasis,
    reconcile_document_transformations,
    render_document_skeleton,
    render_document_role_slots,
)
from app.services.source_contracts import SourceArtifact
from app.services.source_merge import scan_structural_units
from app.services.native_extractor_artifact import (
    native_artifact_path,
    persist_native_extractor_artifact,
    sha256_file,
)
from app.services.native_style import unavailable_native_style_bytes


def _native(source, markdown, value, native_style=None):
    if isinstance(value, dict):
        value = json.dumps(value).encode("utf-8")
    if isinstance(native_style, dict):
        native_style = json.dumps(native_style).encode("utf-8")
    return NativeStructureArtifact.for_test(
        source, markdown, value, native_style
    )


def _audit(artifact):
    return [{
        "output_byte_start": 0,
        "output_byte_end": len(artifact.raw_utf8),
        "source": artifact.source,
        "artifact_digest": artifact.digest,
        "source_byte_start": 0,
        "source_byte_end": len(artifact.raw_utf8),
        "candidate_id": None,
        "region_id": None,
        "decision_method": "baseline_fallback",
    }]


def _style(source, text, start, end):
    return {
        "schema": "pdfx-native-style",
        "contract_version": "native-style-v1",
        "source": source,
        "status": "available",
        "pages": [{
            "page_no": 1,
            "lines": [{
                "native_id": f"{source}-line-1",
                "text": text,
                "italic_spans": [{
                    "start": start,
                    "end": end,
                    "styles": ["Times-Italic"],
                }],
            }],
        }],
    }


def _marker_native_with_italic_paragraph(
    artifact, *, before, paragraph_html, after
):
    return _native(
        "marker",
        artifact,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {
                        "id": "/page/0/Text/0",
                        "block_type": "Text",
                        "html": f"<p>{before}</p>",
                    },
                    {
                        "id": "/page/0/Text/1",
                        "block_type": "Text",
                        "html": f"<p>{paragraph_html}</p>",
                    },
                    {
                        "id": "/page/0/Text/2",
                        "block_type": "Text",
                        "html": f"<p>{after}</p>",
                    },
                ],
            }],
        },
    )


def _audit_with_heading_candidate(artifact, label, candidate_id):
    heading = next(
        unit
        for unit in scan_structural_units(artifact)
        if unit.unit_type == "heading"
        and label in artifact.raw_utf8[unit.byte_start:unit.byte_end].decode()
    )
    ranges = [
        (0, heading.byte_start, None),
        (heading.byte_start, heading.byte_end, candidate_id),
        (heading.byte_end, len(artifact.raw_utf8), None),
    ]
    return [
        {
            "output_byte_start": start,
            "output_byte_end": end,
            "source": artifact.source,
            "artifact_digest": artifact.digest,
            "source_byte_start": start,
            "source_byte_end": end,
            "candidate_id": bound_candidate,
            "region_id": None,
            "decision_method": "model_selected" if bound_candidate else "baseline_fallback",
        }
        for start, end, bound_candidate in ranges
        if end > start
    ]


@pytest.mark.parametrize(
    ("source", "native_value", "body_native_id", "body_page"),
    [
        (
            "grobid",
            b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>
            <titleStmt><title xml:id="title-1">A scientific article title</title></titleStmt>
            </fileDesc></teiHeader><text><body><div><head xml:id="head-1">Results</head>
            <p xml:id="body-1" coords="2,0,0,1,1">Body.</p></div></body></text></TEI>""",
            "body-1",
            2,
        ),
        (
            "docling",
            {
                "schema_name": "DoclingDocument",
                "texts": [
                    {"self_ref": "#/texts/0", "label": "title", "text": "A scientific article title"},
                    {"self_ref": "#/texts/1", "label": "section_header", "level": 1, "text": "Results"},
                    {"self_ref": "#/texts/2", "label": "text", "text": "Body.", "prov": [{"page_no": 2}]},
                ],
            },
            "#/texts/2",
            2,
        ),
        (
            "marker",
            {
                "block_type": "Document",
                "children": [{
                    "block_type": "Page",
                    "children": [
                        {"id": "/page/0/Title/0", "block_type": "Title", "html": "<h1>A scientific article title</h1>"},
                        {"id": "/page/0/SectionHeader/1", "block_type": "SectionHeader", "html": "<h2>Results</h2>"},
                        {"id": "/page/1/Text/0", "block_type": "Text", "html": "<p>Body.</p>", "page_id": 1},
                    ],
                }],
            },
            "/page/1/Text/0",
            2,
        ),
    ],
)
def test_native_adapters_drive_one_coherent_skeleton(
    source, native_value, body_native_id, body_page
):
    artifact = SourceArtifact.from_text(
        source,
        "## REVIEW\n\n## A scientific article title\n\n# Results\n\nBody.\n",
    )
    native = _native(source, artifact, native_value)

    hints = native_heading_hints(native)
    skeleton = build_document_skeleton(artifact, native)
    rendered, audit, events = render_document_skeleton(
        artifact.text, _audit(artifact), skeleton
    )

    assert [hint.role for hint in hints] == ["title", "section"]
    assert skeleton.title_proven is True
    assert [heading.final_level for heading in skeleton.headings] == [0, 1, 2]
    assert rendered == "REVIEW\n\n# A scientific article title\n\n## Results\n\nBody.\n"
    assert len(events) == 3
    assert audit[0]["source_byte_start"] > 0
    assert audit[-1]["output_byte_end"] == len(rendered.encode("utf-8"))
    body_occurrence = next(
        occurrence
        for occurrence in skeleton.occurrences
        if occurrence.native_id == body_native_id
    )
    assert body_occurrence.native_order is not None
    assert body_occurrence.page_no == body_page
    assert artifact.raw_utf8[
        body_occurrence.source_byte_start : body_occurrence.source_byte_end
    ].decode().strip() == "Body."


@pytest.mark.parametrize(
    ("source", "native_value", "expected_payload"),
    [
        (
            "grobid",
            b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div>
            <head>Results</head><p xml:id="body-1">Gene <hi rend="italic">dpp</hi>.</p>
            </div></body></text></TEI>""",
            "dpp",
        ),
        (
            "docling",
            {
                "schema_name": "DoclingDocument",
                "texts": [
                    {"self_ref": "#/texts/0", "label": "section_header", "text": "Results"},
                    {
                        "self_ref": "#/texts/1",
                        "label": "text",
                        "text": "Gene dpp.",
                        "formatting": {"italic": True},
                    },
                ],
            },
            "Gene dpp.",
        ),
        (
            "marker",
            {
                "block_type": "Document",
                "children": [{
                    "block_type": "Page",
                    "children": [
                        {"id": "/page/0/SectionHeader/0", "block_type": "SectionHeader", "html": "<h2>Results</h2>"},
                        {"id": "/page/0/Text/1", "block_type": "Text", "html": "<p>Gene <i>dpp</i>.</p>"},
                    ],
                }],
            },
            "dpp",
        ),
    ],
)
def test_explicit_native_body_italics_are_bound_to_the_mapped_occurrence(
    source, native_value, expected_payload
):
    artifact = SourceArtifact.from_text(
        source,
        "## Results\n\nGene *dpp*.\n",
    )
    native = _native(source, artifact, native_value)

    skeleton = build_document_skeleton(artifact, native)
    body = next(
        occurrence
        for occurrence in skeleton.occurrences
        if occurrence.unit_type == "paragraph"
    )

    native_hint = next(
        hint
        for hint in native_occurrence_hints(native)
        if hint.native_emphasis_count
    )
    assert native_hint.native_visible_text is not None
    assert len(native_hint.native_emphasis_spans) == 1
    native_span = native_hint.native_emphasis_spans[0]
    assert native_hint.native_visible_text[
        native_span.visible_start : native_span.visible_end
    ] == expected_payload
    assert body.native_emphasis_count == 1
    assert body.native_visible_text == native_hint.native_visible_text
    assert body.native_emphasis_spans == native_hint.native_emphasis_spans
    assert skeleton.native_body_emphasis_count == 1
    assert skeleton.mapped_native_body_emphasis_count == 1


def test_docling_absent_formatting_and_grobid_reference_style_are_not_body_evidence():
    docling = SourceArtifact.from_text("docling", "## Results\n\nGene *dpp*.\n")
    docling_native = _native(
        "docling",
        docling,
        {
            "schema_name": "DoclingDocument",
            "texts": [
                {"self_ref": "#/texts/0", "label": "section_header", "text": "Results"},
                {"self_ref": "#/texts/1", "label": "text", "text": "Gene dpp."},
            ],
        },
    )
    grobid = SourceArtifact.from_text(
        "grobid",
        "## References\n\nSmith. *Journal Name*.\n",
    )
    grobid_native = _native(
        "grobid",
        grobid,
        b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><back><div>
        <head>References</head><listBibl><biblStruct xml:id="ref-1"><analytic>
        <author><persName><surname>Smith</surname></persName></author></analytic>
        <monogr><title level="j">Journal Name</title></monogr></biblStruct></listBibl>
        </div></back></text></TEI>""",
    )

    assert all(
        hint.native_emphasis_count == 0
        for hint in native_occurrence_hints(docling_native)
    )
    assert all(
        hint.native_emphasis_count == 0
        for hint in native_occurrence_hints(grobid_native)
    )


def test_docling_native_style_line_maps_and_projects_without_changing_visible_text():
    artifact = SourceArtifact.from_text(
        "docling", "# Title\n\n## Results\n\nGene dpp works.\n\n...\n"
    )
    style = _style("docling", "Gene dpp works.", 5, 8)
    style["pages"][0]["lines"].append({
        "native_id": "docling-unmatched-line",
        "text": "...",
        "italic_spans": [{
            "start": 0,
            "end": 3,
            "styles": ["Times-Italic"],
        }],
    })
    native = _native(
        "docling",
        artifact,
        {
            "schema_name": "DoclingDocument",
            "body": {
                "children": [
                    {"$ref": "#/texts/0"},
                    {"$ref": "#/texts/1"},
                    {"$ref": "#/texts/2"},
                ]
            },
            "texts": [
                {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                {
                    "self_ref": "#/texts/1",
                    "label": "text",
                    "text": "Results",
                    "prov": [{"page_no": 1}],
                },
                {
                    "self_ref": "#/texts/2",
                    "label": "text",
                    "text": "Gene dpp works.",
                    "prov": [{"page_no": 1}],
                },
            ],
        },
        style,
    )
    skeleton = build_document_skeleton(artifact, native)

    assert len(skeleton.native_style_occurrences) == 1
    assert skeleton.native_body_emphasis_count == 1
    assert skeleton.mapped_native_body_emphasis_count == 1
    assert skeleton.native_style_emphasis_count == 2
    assert skeleton.mapped_native_style_emphasis_count == 1
    assert skeleton.unmapped_native_style_emphasis_count == 1
    assert skeleton.auxiliary_native_style_body_emphasis_count == 1
    assert tuple(
        hint.native_id for hint in skeleton.unmapped_native_style_occurrences
    ) == ("docling-unmatched-line",)

    rendered, audit, events = project_native_emphasis(
        artifact.text,
        _audit(artifact),
        {"docling": skeleton},
        {"docling": artifact},
    )

    assert rendered == "# Title\n\n## Results\n\nGene *dpp* works.\n\n...\n"
    assert rendered.replace("*", "") == artifact.text
    projected = [
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    ]
    assert len(projected) == 1
    assert projected[0]["donor_source"] == "docling"
    assert projected[0]["native_style_digest"] == native.native_style_digest
    assert any(entry.get("transformation") == "native_emphasis_projection" for entry in audit)


def test_identity_empty_style_claim_is_raw_unmapped_and_not_body_evidence():
    artifact = SourceArtifact.from_text(
        "docling", "# Title\n\n## Results\n\n...\n"
    )
    native = _native(
        "docling",
        artifact,
        {
            "schema_name": "DoclingDocument",
            "body": {"children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"},
            ]},
            "texts": [
                {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
                {"self_ref": "#/texts/2", "label": "text", "text": "..."},
            ],
        },
        _style("docling", "...", 0, 3),
    )

    skeleton = build_document_skeleton(artifact, native)

    assert skeleton.native_style_emphasis_count == 1
    assert skeleton.mapped_native_style_emphasis_count == 0
    assert skeleton.unmapped_native_style_emphasis_count == 1
    assert skeleton.auxiliary_native_style_body_emphasis_count == 1
    assert len(skeleton.unmapped_native_style_occurrences) == 1
    assert skeleton.native_body_emphasis_count == 0
    assert skeleton.mapped_native_body_emphasis_count == 0


def test_positive_style_inventory_keeps_protected_and_auxiliary_ledgers_separate():
    artifact = SourceArtifact.from_text(
        "docling", "# Title\n\n## Results\n\nGene dpp works.\n\nAux *extra*.\n"
    )
    native = _native(
        "docling",
        artifact,
        {
            "schema_name": "DoclingDocument",
            "body": {"children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"},
                {"$ref": "#/texts/3"},
            ]},
            "texts": [
                {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
                {
                    "self_ref": "#/texts/2",
                    "label": "text",
                    "text": "Gene dpp works.",
                    "formatting": {"italic": True},
                },
                {"self_ref": "#/texts/3", "label": "text", "text": "Aux extra."},
            ],
        },
        _style("docling", "No matching source line", 3, 11),
    )
    skeleton = build_document_skeleton(artifact, native)

    inventory = _projection_claim_inventory(
        {"docling": skeleton}, {"docling": artifact}
    )
    claims = [
        claim
        for donor in inventory.donors["docling"]
        for claim in donor.claims
    ] + [claim for claim, _reason in inventory.unplaced["docling"]]

    assert len(inventory.protected_claim_ids) == skeleton.native_body_emphasis_count == 1
    assert len(inventory.auxiliary_claim_ids) == 2
    assert {claim.evidence_kind for claim in claims if not claim.protected} == {
        "native_style",
        "source_markdown",
    }
    assert not set(inventory.protected_claim_ids) & set(
        inventory.auxiliary_claim_ids
    )


def test_final_style_targets_are_scanned_without_source_audit_admission():
    text = (
        "# Title\n\n## Results\n\n"
        "Composite Gene *dpp* and [wg](https://example.org/wg).\n\n"
        + ("A" * 20_001)
        + "\n\n## References\n\nReference payload.\n"
    )

    targets = _final_projection_targets(text)

    assert len(targets) == 4
    assert targets[0].occurrence.unit_type == "heading"
    assert targets[1].occurrence.unit_type == "heading"
    styled = targets[2]
    assert styled.source == "final"
    assert styled.visible == "Composite Gene dpp and wg."
    assert len(styled.existing_emphasis_occurrence_ids) == 1
    start, end = styled.existing_emphasis_spans[0]
    assert styled.visible[start:end] == "dpp"
    assert len(targets[3].visible) == 20_001
    assert all(target.visible != "References" for target in targets)


def test_markdown_only_source_participates_without_mapped_native_claims():
    docling = SourceArtifact.from_text(
        "docling", "# Title\n\n## Results\n\nGene *dpp* works.\n"
    )
    marker = SourceArtifact.from_text(
        "marker", "# Title\n\n## Results\n\nGene dpp works.\n"
    )
    skeletons = {
        "docling": build_document_skeleton(docling, None),
        "marker": build_document_skeleton(marker, None),
    }

    rendered, rendered_audit, events = project_native_emphasis(
        marker.text,
        _audit(marker),
        skeletons,
        {"docling": docling, "marker": marker},
    )

    assert rendered == "# Title\n\n## Results\n\nGene *dpp* works.\n"
    assert any(
        event.get("outcome") == "projected"
        and event.get("boundary") == "open"
        and event.get("native_evidence_kind") == "source_markdown"
        for event in events
    )


def test_existing_italics_and_link_markup_survive_a_non_overlapping_addition():
    docling = SourceArtifact.from_text(
        "docling", "# Title\n\n## Results\n\nGene *dpp* and *wg*.\n"
    )
    marker = SourceArtifact.from_text(
        "marker",
        "# Title\n\n## Results\n\nGene *dpp* and [wg](https://example.org/wg).\n",
    )
    skeletons = {
        "docling": build_document_skeleton(docling, None),
        "marker": build_document_skeleton(marker, None),
    }

    rendered, rendered_audit, events = project_native_emphasis(
        marker.text,
        _audit(marker),
        skeletons,
        {"docling": docling, "marker": marker},
    )

    assert rendered == (
        "# Title\n\n## Results\n\n"
        "Gene *dpp* and [*wg*](https://example.org/wg).\n"
    )
    assert any(
        event.get("reason") == "existing_final_emphasis"
        and event.get("native_evidence_kind") == "source_markdown"
        for event in events
    )
    assert sum(
        event.get("outcome") == "projected" and event.get("boundary") == "open"
        for event in events
    ) == 1


def test_disagreeing_grobid_and_docling_intervals_select_wider_evidence():
    text = "# Title\n\n## Results\n\nGene dpp works.\n"
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("grobid", "docling")
    }
    native_documents = {
        "grobid": b"<TEI xmlns='http://www.tei-c.org/ns/1.0'><teiHeader><fileDesc><titleStmt><title>Title</title></titleStmt></fileDesc></teiHeader><text><body><div><head>Results</head><p>Gene dpp works.</p></div></body></text></TEI>",
        "docling": {
            "schema_name": "DoclingDocument",
            "body": {"children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"},
            ]},
            "texts": [
                {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
                {"self_ref": "#/texts/2", "label": "text", "text": "Gene dpp works."},
            ],
        },
    }
    natives = {
        "grobid": _native(
            "grobid",
            artifacts["grobid"],
            native_documents["grobid"],
            _style("grobid", "Gene dpp works.", 5, 8),
        ),
        "docling": _native(
            "docling",
            artifacts["docling"],
            native_documents["docling"],
            _style("docling", "Gene dpp works.", 5, 14),
        ),
    }
    skeletons = {
        source: build_document_skeleton(artifacts[source], natives[source])
        for source in artifacts
    }

    rendered, audit, events = project_native_emphasis(
        text,
        _audit(artifacts["grobid"]),
        skeletons,
        artifacts,
    )

    assert rendered == text.replace("dpp works", "*dpp works*")
    assert audit != _audit(artifacts["grobid"])
    narrower_supports = [
        event
        for event in events
        if event.get("reason") == "canonical_interval_supported"
    ]
    assert {event["donor_source"] for event in narrower_supports} == {"grobid"}


def test_identical_grobid_and_docling_intervals_project_once_with_peer_support():
    text = "# Title\n\n## Results\n\nGene dpp works.\n"
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("grobid", "docling")
    }
    native_documents = {
        "grobid": b"<TEI xmlns='http://www.tei-c.org/ns/1.0'><teiHeader><fileDesc><titleStmt><title>Title</title></titleStmt></fileDesc></teiHeader><text><body><div><head>Results</head><p>Gene dpp works.</p></div></body></text></TEI>",
        "docling": {
            "schema_name": "DoclingDocument",
            "body": {"children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"},
            ]},
            "texts": [
                {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
                {"self_ref": "#/texts/2", "label": "text", "text": "Gene dpp works."},
            ],
        },
    }
    natives = {
        source: _native(
            source,
            artifacts[source],
            native_documents[source],
            _style(source, "Gene dpp works.", 5, 8),
        )
        for source in artifacts
    }
    skeletons = {
        source: build_document_skeleton(artifacts[source], natives[source])
        for source in artifacts
    }

    rendered, _audit_result, events = project_native_emphasis(
        text,
        _audit(artifacts["grobid"]),
        skeletons,
        artifacts,
    )

    assert rendered == "# Title\n\n## Results\n\nGene *dpp* works.\n"
    projected = [
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    ]
    assert len(projected) == 1
    representative = projected[0]
    assert representative["supporting_sources"] == ["docling", "grobid"]
    assert representative["support_claim_count"] == 2
    peer_source = ({"docling", "grobid"} - {representative["donor_source"]}).pop()
    assert any(
        event.get("donor_source") == peer_source
        and event.get("outcome") == "supported"
        and event.get("reason") == "canonical_interval_supported"
        and event.get("supported_projection_id")
        == representative["projection_id"]
        for event in events
    )


def test_disjoint_safe_native_style_claims_from_different_sources_both_survive():
    text = (
        "# Title\n\n## Results\n\nAlpha dpp works.\n\n"
        "Beta hh works.\n"
    )
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("grobid", "docling")
    }
    styles = {
        "grobid": _style("grobid", "Alpha dpp works.", 6, 9),
        "docling": _style("docling", "Beta hh works.", 5, 7),
    }
    natives = {
        "grobid": _native(
            "grobid",
            artifacts["grobid"],
            b"<TEI><teiHeader><titleStmt><title>Title</title></titleStmt></teiHeader>"
            b"<text><body><div><head>Results</head><p>Alpha dpp works.</p>"
            b"<p>Beta hh works.</p></div></body></text></TEI>",
            styles["grobid"],
        ),
        "docling": _native(
            "docling",
            artifacts["docling"],
            {
                "schema_name": "DoclingDocument",
                "texts": [
                    {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                    {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
                    {"self_ref": "#/texts/2", "label": "text", "text": "Alpha dpp works."},
                    {"self_ref": "#/texts/3", "label": "text", "text": "Beta hh works."},
                ],
            },
            styles["docling"],
        ),
    }
    skeletons = {
        source: build_document_skeleton(artifacts[source], natives[source])
        for source in artifacts
    }

    rendered, _audit_result, events = project_native_emphasis(
        text,
        _audit(artifacts["grobid"]),
        skeletons,
        artifacts,
    )

    assert rendered == (
        "# Title\n\n## Results\n\nAlpha *dpp* works.\n\n"
        "Beta *hh* works.\n"
    )
    assert {
        event["donor_source"]
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    } == {"grobid", "docling"}


def test_two_style_lines_share_one_ordered_source_unit_without_duplicate_anchors():
    text = "# Title\n\n## Results\n\nAlpha dpp works. Beta hh works.\n"
    artifact = SourceArtifact.from_text("docling", text)
    style = _style("docling", "Alpha dpp works.", 6, 9)
    style["pages"][0]["lines"].append(
        {
            "native_id": "docling-line-2",
            "text": "Beta hh works.",
            "italic_spans": [
                {"start": 5, "end": 7, "styles": ["Times-Italic"]}
            ],
        }
    )
    native = _native(
        "docling",
        artifact,
        {
            "schema_name": "DoclingDocument",
            "texts": [
                {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
                {
                    "self_ref": "#/texts/2",
                    "label": "text",
                    "text": "Alpha dpp works. Beta hh works.",
                },
            ],
        },
        style,
    )
    skeleton = build_document_skeleton(artifact, native)

    rendered, _audit_result, events = project_native_emphasis(
        text,
        _audit(artifact),
        {"docling": skeleton},
        {"docling": artifact},
    )

    assert len(skeleton.native_style_occurrences) == 2
    assert rendered == "# Title\n\n## Results\n\nAlpha *dpp* works. Beta *hh* works.\n"
    projected = [
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    ]
    assert len(projected) == 2
    assert len({event["donor_spine_occurrence_id"] for event in projected}) == 1


def test_same_source_conflict_keeps_wider_and_disjoint_claims():
    text = "# Title\n\n## Results\n\nGene dpp works. Gene hh works.\n"
    artifact = SourceArtifact.from_text("docling", text)
    style = _style("docling", "Gene dpp works. Gene hh works.", 5, 8)
    style["pages"][0]["lines"].extend([
        {
            "native_id": "docling-line-overlap",
            "text": "Gene dpp works. Gene hh works.",
            "italic_spans": [
                {"start": 5, "end": 14, "styles": ["Times-Italic"]}
            ],
        },
        {
            "native_id": "docling-line-safe",
            "text": "Gene dpp works. Gene hh works.",
            "italic_spans": [
                {"start": 21, "end": 23, "styles": ["Times-Italic"]}
            ],
        },
    ])
    native = _native(
        "docling",
        artifact,
        {
            "schema_name": "DoclingDocument",
            "texts": [
                {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
                {
                    "self_ref": "#/texts/2",
                    "label": "text",
                    "text": "Gene dpp works. Gene hh works.",
                },
            ],
        },
        style,
    )
    skeleton = build_document_skeleton(artifact, native)

    rendered, _audit_result, events = project_native_emphasis(
        text,
        _audit(artifact),
        {"docling": skeleton},
        {"docling": artifact},
    )

    assert skeleton.mapped_native_body_emphasis_count == 3
    assert rendered == (
        "# Title\n\n## Results\n\nGene *dpp works*. Gene *hh* works.\n"
    )
    assert sum(
        event.get("reason") == "canonical_interval_supported"
        for event in events
    ) == 1
    assert sum(
        event.get("outcome") == "projected" and event.get("boundary") == "open"
        for event in events
    ) == 2


def test_identical_same_source_claims_render_once_with_support():
    text = "# Title\n\n## Results\n\nGene dpp works.\n"
    artifact = SourceArtifact.from_text("docling", text)
    style = _style("docling", "Gene dpp works.", 5, 8)
    style["pages"][0]["lines"].append({
        "native_id": "docling-line-identical-peer",
        "text": "Gene dpp works.",
        "italic_spans": [
            {"start": 5, "end": 8, "styles": ["Times-Italic"]}
        ],
    })
    native = _native(
        "docling",
        artifact,
        {
            "schema_name": "DoclingDocument",
            "texts": [
                {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
                {"self_ref": "#/texts/2", "label": "text", "text": "Gene dpp works."},
            ],
        },
        style,
    )
    skeleton = build_document_skeleton(artifact, native)

    rendered, _audit_result, events = project_native_emphasis(
        text,
        _audit(artifact),
        {"docling": skeleton},
        {"docling": artifact},
    )

    assert skeleton.mapped_native_body_emphasis_count == 2
    assert rendered == "# Title\n\n## Results\n\nGene *dpp* works.\n"
    projected = [
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    ]
    supported = [event for event in events if event.get("outcome") == "supported"]
    assert len(projected) == 1
    assert len(supported) == 1
    assert projected[0]["support_claim_count"] == 2
    assert supported[0]["supported_projection_id"] == projected[0]["projection_id"]


def test_unequal_cross_source_cluster_keeps_wider_member_and_safe_sibling():
    text = "# Title\n\n## Results\n\nGene dpp works.\n\nGene hh works.\n"
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("grobid", "docling")
    }
    grobid_style = _style("grobid", "Gene dpp works.", 5, 8)
    grobid_style["pages"][0]["lines"].append(
        {
            "native_id": "grobid-line-2",
            "text": "Gene hh works.",
            "italic_spans": [
                {"start": 5, "end": 7, "styles": ["Times-Italic"]}
            ],
        }
    )
    docling_style = _style("docling", "Gene dpp works.", 5, 14)
    native_documents = {
        "grobid": b"<TEI><teiHeader><titleStmt><title>Title</title></titleStmt></teiHeader>"
        b"<text><body><div><head>Results</head><p>Gene dpp works.</p>"
        b"<p>Gene hh works.</p></div></body></text></TEI>",
        "docling": {
            "schema_name": "DoclingDocument",
            "texts": [
                {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
                {"self_ref": "#/texts/2", "label": "text", "text": "Gene dpp works."},
                {"self_ref": "#/texts/3", "label": "text", "text": "Gene hh works."},
            ],
        },
    }
    natives = {
        "grobid": _native(
            "grobid", artifacts["grobid"], native_documents["grobid"], grobid_style
        ),
        "docling": _native(
            "docling", artifacts["docling"], native_documents["docling"], docling_style
        ),
    }
    skeletons = {
        source: build_document_skeleton(artifacts[source], natives[source])
        for source in artifacts
    }

    rendered, _audit_result, events = project_native_emphasis(
        text,
        _audit(artifacts["grobid"]),
        skeletons,
        artifacts,
    )

    assert rendered == (
        "# Title\n\n## Results\n\nGene *dpp works*.\n\nGene *hh* works.\n"
    )
    narrower_supports = [
        event
        for event in events
        if event.get("reason") == "canonical_interval_supported"
    ]
    assert {event["donor_source"] for event in narrower_supports} == {"grobid"}
    assert sum(
        event.get("outcome") == "projected" and event.get("boundary") == "open"
        for event in events
    ) == 2


def test_plain_peer_is_neutral_to_a_docling_native_style_projection():
    text = "# Title\n\n## Results\n\nGene dpp works.\n"
    artifacts = {
        source: SourceArtifact.from_text(source, text)
        for source in ("grobid", "docling")
    }
    docling_native = _native(
        "docling",
        artifacts["docling"],
        {
            "schema_name": "DoclingDocument",
            "body": {"children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"},
            ]},
            "texts": [
                {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                {"self_ref": "#/texts/1", "label": "section_header", "text": "Results"},
                {"self_ref": "#/texts/2", "label": "text", "text": "Gene dpp works."},
            ],
        },
        _style("docling", "Gene dpp works.", 5, 8),
    )
    skeletons = {
        "grobid": build_document_skeleton(artifacts["grobid"], None),
        "docling": build_document_skeleton(artifacts["docling"], docling_native),
    }

    rendered, _audit_result, events = project_native_emphasis(
        text,
        _audit(artifacts["grobid"]),
        skeletons,
        artifacts,
    )

    assert rendered == "# Title\n\n## Results\n\nGene *dpp* works.\n"
    projected = [
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    ]
    assert [event["donor_source"] for event in projected] == ["docling"]


def test_reference_unit_type_overrides_a_native_body_region_for_emphasis_totals():
    artifact = SourceArtifact.from_text(
        "marker",
        "## Literature Cited\n\n1. Smith Journal Name.\n",
    )
    native = _native(
        "marker",
        artifact,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {
                        "id": "/page/0/SectionHeader/0",
                        "block_type": "SectionHeader",
                        "html": "<h2>Literature Cited</h2>",
                    },
                    {
                        "id": "/page/0/Text/1",
                        "block_type": "Text",
                        "html": "<p>1. Smith <i>Journal Name</i>.</p>",
                    },
                ],
            }],
        },
    )

    skeleton = build_document_skeleton(artifact, native)
    reference = next(
        occurrence
        for occurrence in skeleton.occurrences
        if occurrence.native_emphasis_count
    )

    assert reference.unit_type == "reference"
    assert reference.region == "back"
    assert skeleton.native_body_emphasis_count == 0
    assert skeleton.native_reference_emphasis_count == 1


def test_cross_source_anchor_region_projects_one_bilateral_exact_payload():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor before.\n\nLeft Gene dpp right.\n\nAnchor after.\n",
    )
    target = SourceArtifact.from_text(
        "grobid",
        "Anchor before.\n\nPrefix. Left Gene dpp right. Suffix.\n\nAnchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html="Left Gene <i>dpp</i> right.",
        after="Anchor after.",
    )
    skeletons = {
        "marker": build_document_skeleton(marker, native),
        "grobid": build_document_skeleton(target, None),
    }

    rendered, audit, events = project_native_emphasis(
        target.text,
        _audit(target),
        skeletons,
        {"marker": marker, "grobid": target},
    )

    assert rendered == (
        "Anchor before.\n\nPrefix. Left Gene *dpp* right. Suffix.\n\n"
        "Anchor after.\n"
    )
    projected = [event for event in events if event.get("outcome") == "projected"]
    assert len(projected) == 2
    assert {event["mapping_kind"] for event in projected} == {
        "final_unit_alignment"
    }
    assert all(
            event["alignment_method"] == "final-unit-shared-character-alignment-v1"
        and event["target_emphasis_occurrence_id"]
        != event["native_emphasis_occurrence_id"]
        for event in projected
    )
    assert sum(
        entry.get("transformation") == "native_emphasis_projection"
        for entry in audit
    ) == 2


def test_cross_source_region_routes_a_weak_mappable_unit_only_to_selection():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor before.\n\nPreviously described (30).\n\nAnchor after.\n",
    )
    target = SourceArtifact.from_text(
        "grobid",
        "Anchor before.\n\nIncubate for 30 min.\n\nAnchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html="Previously described (<i>30</i>).",
        after="Anchor after.",
    )

    rendered, audit, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "grobid": build_document_skeleton(target, None),
        },
        {"marker": marker, "grobid": target},
    )

    assert rendered == target.text
    assert audit == _audit(target)
    assert any(
        event.get("reason") == "final_unit_pair_ambiguous"
        and event.get("style_selection_below_deterministic_floor") is True
        and event.get("style_selection_method") == "not_run"
        for event in events
    )


def test_cross_source_unit_alignment_selects_tied_repeated_payload_by_order():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor before.\n\nLeft Gene dpp right.\n\nAnchor after.\n",
    )
    target = SourceArtifact.from_text(
        "grobid",
        "Anchor before.\n\nLeft Gene dpp right. Left Gene dpp right.\n\n"
        "Anchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html="Left Gene <i>dpp</i> right.",
        after="Anchor after.",
    )

    rendered, audit, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "grobid": build_document_skeleton(target, None),
        },
        {"marker": marker, "grobid": target},
    )

    assert rendered == target.text.replace("Gene dpp", "Gene *dpp*", 1)
    assert audit != _audit(target)
    assert any(event.get("outcome") == "projected" for event in events)


def test_cross_source_unit_alignment_leaves_ambiguous_block_unresolved_without_selector():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor before.\n\nGene dpp.\n\nAnchor after.\n",
    )
    target = SourceArtifact.from_text(
        "grobid",
        "Anchor before.\n\nGene dpp.\n\nGene dpp.\n\nAnchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html="Gene <i>dpp</i>.",
        after="Anchor after.",
    )

    rendered, audit, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "grobid": build_document_skeleton(target, None),
        },
        {"marker": marker, "grobid": target},
    )

    assert rendered == target.text
    assert audit == _audit(target)
    declines = [event for event in events if event.get("outcome") == "declined"]
    assert len(declines) == 1
    assert declines[0]["reason"] == "final_unit_pair_ambiguous"
    assert declines[0]["unit_pair_ambiguous"] is True
    assert len(declines[0]["style_selection_candidate_ids"]) == 2
    assert declines[0]["style_selection_method"] == "not_run"


def test_cross_source_unit_alignment_uses_only_a_numbered_existing_target_choice():
    marker = SourceArtifact.from_text(
        "marker", "Anchor before.\n\nGene dpp.\n\nAnchor after.\n"
    )
    target = SourceArtifact.from_text(
        "grobid",
        "Anchor before.\n\nGene dpp.\n\nGene dpp.\n\nAnchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html="Gene <i>dpp</i>.",
        after="Anchor after.",
    )
    calls = []

    def choose_first_target(*, reason, baseline_id, choices):
        calls.append((reason, baseline_id, choices))
        return {
            "selected_candidate_id": choices[1]["candidate_id"],
            "request_sha256": "a" * 64,
            "response_choice": 1,
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
        }

    rendered, rendered_audit, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "grobid": build_document_skeleton(target, None),
        },
        {"marker": marker, "grobid": target},
        style_selection_resolver=choose_first_target,
    )

    assert rendered == target.text.replace("Gene dpp", "Gene *dpp*", 1)
    assert len(calls) == 1
    assert calls[0][1] == calls[0][2][0]["candidate_id"]
    assert all("display" in choice for choice in calls[0][2])
    projected = next(
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    )
    assert projected["model_selected_target"] is True
    assert projected["style_selection_method"] == "sol_numbered_choice"
    assert projected["style_selection_request_sha256"] == "a" * 64
    assert projected["style_selection_response_choice"] == 1
    recorded = {
        projected["style_selection_id"]: {
            key: value
            for key, value in projected.items()
            if key == "unit_pair_ambiguous"
            or key.startswith("style_selection_")
            or key == "style_selected_candidate_id"
            or key == "model_selected_target"
        }
    }
    replayed, replayed_audit, replayed_events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "grobid": build_document_skeleton(target, None),
        },
        {"marker": marker, "grobid": target},
        recorded_style_selections=recorded,
    )
    assert (replayed, replayed_audit, replayed_events) == (
        rendered,
        rendered_audit,
        events,
    )


def test_published_unit_alignment_preserves_partial_token_boundary():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor before.\n\nNtn4-dependent signaling.\n\nAnchor after.\n",
    )
    target = SourceArtifact.from_text(
        "grobid",
        "Anchor before.\n\nNtn4-dependent signaling.\n\nAnchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html="<i>Ntn</i>4-dependent signaling.",
        after="Anchor after.",
    )

    rendered, _audit_result, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "grobid": build_document_skeleton(target, None),
        },
        {"marker": marker, "grobid": target},
    )

    assert rendered == (
        "Anchor before.\n\n*Ntn*4-dependent signaling.\n\nAnchor after.\n"
    )
    assert any(
        event.get("alignment_method") == "final-unit-shared-character-alignment-v1"
        and event.get("outcome") == "projected"
        for event in events
    )


def test_published_unit_alignment_crosses_entity_tokenization_without_text_edit():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor before.\n\nResults with p < 0.05 were significant.\n\nAnchor after.\n",
    )
    target = SourceArtifact.from_text(
        "docling",
        "Anchor before.\n\nResults with p &lt; 0.05 were significant.\n\nAnchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html="Results with <i>p</i> &lt; 0.05 were significant.",
        after="Anchor after.",
    )

    rendered, _audit_result, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "docling": build_document_skeleton(target, None),
        },
        {"marker": marker, "docling": target},
    )

    assert rendered == (
        "Anchor before.\n\nResults with *p* &lt; 0.05 were significant.\n\n"
        "Anchor after.\n"
    )
    assert any(
        event.get("alignment_method") == "final-unit-shared-character-alignment-v1"
        and event.get("outcome") == "projected"
        for event in events
    )


def test_published_caption_alignment_maps_ordered_statistical_p_claims():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor before.\n\nFig. 1 Results. *P<0.05, **P<0.01, ***P<0.001.\n\n"
        "Anchor after.\n",
    )
    target = SourceArtifact.from_text(
        "docling",
        "Anchor before.\n\nFig. 1 Results. * P &lt; 0.05, ** P &lt; 0.01, "
        "*** P &lt; 0.001.\n\nAnchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html=(
            "Fig. 1 Results. *<i>P</i>&lt;0.05, **<i>P</i>&lt;0.01, "
            "***<i>P</i>&lt;0.001."
        ),
        after="Anchor after.",
    )

    rendered, _audit_result, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "docling": build_document_skeleton(target, None),
        },
        {"marker": marker, "docling": target},
    )

    assert rendered == (
        "Anchor before.\n\nFig. 1 Results. * *P* &lt; 0.05, ** *P* &lt; "
        "0.01, *** *P* &lt; 0.001.\n\nAnchor after.\n"
    )
    projected = [
        event
        for event in events
        if event.get("alignment_method") == "final-unit-shared-character-alignment-v1"
        and event.get("outcome") == "projected"
        and event.get("boundary") == "open"
    ]
    assert len(projected) == 3


def test_style_projection_does_not_use_donor_unit_type_as_a_veto():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor before.\n\nResults. Gene dpp responds to treatment.\n\nAnchor after.\n",
    )
    target = SourceArtifact.from_text(
        "docling",
        "Anchor before.\n\nFig. 1. Results. Gene dpp responds to treatment.\n\n"
        "Anchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html="Results. Gene <i>dpp</i> responds to treatment.",
        after="Anchor after.",
    )

    rendered, _audit_result, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "docling": build_document_skeleton(target, None),
        },
        {"marker": marker, "docling": target},
    )

    assert "Fig. 1. Results. Gene *dpp* responds to treatment." in rendered
    projected = next(
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    )
    assert projected["donor_unit_type"] == "paragraph"
    assert projected["target_unit_type"] == "figure_caption"


def test_published_paragraph_alignment_maps_the_ordered_second_p():
    marker_text = (
        "Anchor before.\n\nBackground p = 0.2. Differences were significant "
        "(p < 0.05 to p < 0.001; Figure 3). Tail p = 0.4.\n\nAnchor after.\n"
    )
    target_text = marker_text.replace(" < ", " &lt; ")
    marker = SourceArtifact.from_text("marker", marker_text)
    target = SourceArtifact.from_text("docling", target_text)
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html=(
            "Background p = 0.2. Differences were significant "
            "(p &lt; 0.05 to <i>p</i> &lt; 0.001; Figure 3). Tail p = 0.4."
        ),
        after="Anchor after.",
    )

    rendered, _audit_result, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "docling": build_document_skeleton(target, None),
        },
        {"marker": marker, "docling": target},
    )

    assert "p &lt; 0.05 to *p* &lt; 0.001" in rendered
    assert rendered.replace("*p*", "p") == target.text
    assert any(
        event.get("alignment_method") == "final-unit-shared-character-alignment-v1"
        and event.get("outcome") == "projected"
        for event in events
    )


def test_published_character_alignment_expands_ocr_fragment_to_target_tokens():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor before.\n\nPseudomonas fuores cens grew.\n\nAnchor after.\n",
    )
    target = SourceArtifact.from_text(
        "docling",
        "Anchor before.\n\nPseudomonas fluorescens grew.\n\nAnchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html="<i>Pseudomonas fuores</i> cens grew.",
        after="Anchor after.",
    )

    rendered, _audit_result, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "docling": build_document_skeleton(target, None),
        },
        {"marker": marker, "docling": target},
    )

    assert "*Pseudomonas fluorescens* grew" in rendered
    assert any(
        event.get("alignment_method")
        == "final-unit-shared-character-alignment-v1"
        and event.get("outcome") == "projected"
        for event in events
    )


def test_rejected_all_optimal_n_counterexample_remains_declined_without_a2():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor before.\n\none-way ANOVA, n > 4.\n\nAnchor after.\n",
    )
    target = SourceArtifact.from_text(
        "grobid",
        "Anchor before.\n\n4.0 plus or minus 2.3-fold lower, n = 3 transduction.\n\n"
        "Anchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html="one-way ANOVA, <i>n</i> &gt; 4.",
        after="Anchor after.",
    )

    rendered, audit, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "grobid": build_document_skeleton(target, None),
        },
        {"marker": marker, "grobid": target},
    )

    assert rendered == target.text
    assert audit == _audit(target)
    assert any(
        event.get("reason") == "final_unit_pair_unavailable"
        for event in events
    )


def test_identical_same_source_support_does_not_discard_separate_projection():
    marker = SourceArtifact.from_text(
        "marker",
        (
            "Anchor before.\n\nLeft Akkermansia right.\n\n"
            "Left muciniphila right.\n\nLeft muciniphila right.\n\n"
            "Anchor after.\n"
        ),
    )
    target = SourceArtifact.from_text(
        "grobid",
        (
            "Anchor before.\n\nLeft Akkermansia right; "
            "Left muciniphila right.\n\nAnchor after.\n"
        ),
    )
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {
                        "id": f"/page/0/Text/{index}",
                        "block_type": "Text",
                        "html": f"<p>{html}</p>",
                    }
                    for index, html in enumerate(
                        (
                            "Anchor before.",
                            "Left <i>Akkermansia</i> right.",
                            "Left <i>muciniphila</i> right.",
                            "Left <i>muciniphila</i> right.",
                            "Anchor after.",
                        )
                    )
                ],
            }],
        },
    )

    rendered, _audit_result, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "grobid": build_document_skeleton(target, None),
        },
        {"marker": marker, "grobid": target},
    )

    assert rendered == (
        "Anchor before.\n\nLeft *Akkermansia* right; "
        "Left *muciniphila* right.\n\nAnchor after.\n"
    )
    assert sum(
        event.get("outcome") == "projected" and event.get("boundary") == "open"
        for event in events
    ) == 2
    assert sum(
        event.get("outcome") == "declined"
        and event.get("reason") == "final_unit_pair_ambiguous"
        for event in events
    ) == 1


def test_model_selected_adjacent_donors_may_share_one_final_unit():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor.\n\nLeft one right.\n\nLeft two right.\n\nEnd.\n",
    )
    target = SourceArtifact.from_text(
        "grobid",
        "Anchor.\n\nLeft one right; Left two right.\n\nEnd.\n",
    )
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {
                        "id": f"/page/0/Text/{index}",
                        "block_type": "Text",
                        "html": f"<p>{html}</p>",
                    }
                    for index, html in enumerate((
                        "Anchor.",
                        "Left <i>one</i> right.",
                        "Left <i>two</i> right.",
                        "End.",
                    ))
                ],
            }],
        },
    )

    def choose_target(*, baseline_id, choices, **_kwargs):
        selected = next(
            choice["candidate_id"]
            for choice in choices
            if choice["candidate_id"] != baseline_id
        )
        return selected

    rendered, _audit_result, events = project_native_emphasis(
        target.text,
        _audit(target),
        {
            "marker": build_document_skeleton(marker, native),
            "grobid": build_document_skeleton(target, None),
        },
        {"marker": marker, "grobid": target},
        style_selection_resolver=choose_target,
    )

    assert "Left *one* right; Left *two* right." in rendered
    assert not any(
        event.get("reason") == "model_selection_non_monotone"
        for event in events
    )


@pytest.mark.parametrize(
    ("source", "native_value"),
    [
        (
            "docling",
            {
                "schema_name": "DoclingDocument",
                "texts": [
                    {"self_ref": "#/texts/0", "label": "title", "text": "Title"},
                    {"self_ref": "#/texts/1", "label": "section_header", "text": "References"},
                    {
                        "self_ref": "#/texts/2",
                        "label": "reference",
                        "text": "Journal Name",
                        "formatting": {"italic": True},
                    },
                ],
            },
        ),
        (
            "marker",
            {
                "block_type": "Document",
                "children": [{
                    "block_type": "Page",
                    "children": [
                        {"id": "/page/0/Title/0", "block_type": "Title", "html": "<h1>Title</h1>"},
                        {"id": "/page/0/SectionHeader/1", "block_type": "SectionHeader", "html": "<h2>References</h2>"},
                        {"id": "/page/0/Reference/2", "block_type": "Reference", "html": "<p><i>Journal Name</i></p>"},
                    ],
                }],
            },
        ),
    ],
)
def test_native_reference_italics_do_not_enter_the_body_denominator(
    source, native_value
):
    artifact = SourceArtifact.from_text(
        source,
        "# Title\n\n## References\n\n*Journal Name*\n",
    )
    skeleton = build_document_skeleton(
        artifact, _native(source, artifact, native_value)
    )

    assert skeleton.native_body_emphasis_count == 0
    assert skeleton.mapped_native_body_emphasis_count == 0
    assert skeleton.native_reference_emphasis_count == 1
    assert skeleton.mapped_native_reference_emphasis_count == 1


def test_unproved_late_h1_becomes_titleless_warning_shape():
    artifact = SourceArtifact.from_text(
        "marker",
        "## Front matter\n\n# A body subsection\n\nBody.\n",
    )

    skeleton = build_document_skeleton(artifact, native=None)
    rendered, _audit_result, _events = render_document_skeleton(
        artifact.text, _audit(artifact), skeleton
    )

    assert skeleton.title_proven is False
    assert skeleton.findings == ("native_structure_unavailable",)
    assert rendered == "## Front matter\n\n## A body subsection\n\nBody.\n"


def test_role_slot_renderer_promotes_one_peer_proved_title_orders_back_matter_and_numbers_references():
    docling = SourceArtifact.from_text(
        "docling",
        (
            "## REVIEW\n\n## Proven title\n\n## Results\n\nBody.\n\n"
            "## References\n\n- Alpha et al. (2024). One.\n"
            "- Beta et al. (2025). Two.\n\n"
            "## Acknowledgments\n\nThanks.\n"
        ),
    )
    grobid = SourceArtifact.from_text(
        "grobid",
        "# Proven title\n\n## Results\n\nBody.\n",
    )
    grobid_native = _native(
        "grobid",
        grobid,
        b"""<TEI><teiHeader><titleStmt><title>Proven title</title></titleStmt>
        </teiHeader><text><body><div><head>Results</head><p>Body.</p></div>
        </body></text></TEI>""",
    )
    selected = build_document_skeleton(docling, None)
    peer = build_document_skeleton(grobid, grobid_native)
    docling_title = next(
        unit
        for unit in scan_structural_units(docling)
        if unit.unit_type == "heading" and "Proven title" in docling.raw_utf8[unit.byte_start:unit.byte_end].decode()
    )
    grobid_title = next(
        unit
        for unit in scan_structural_units(grobid)
        if unit.unit_type == "heading" and "Proven title" in grobid.raw_utf8[unit.byte_start:unit.byte_end].decode()
    )
    decision_trace = ({
        "selected_candidate_ids": ["docling-title"],
        "baseline_candidate_id": "docling-title",
        "candidates": [
            {
                "candidate_id": "docling-title",
                "candidate_type": "heading",
                "source": "docling",
                "artifact_digest": docling.digest,
                "source_byte_start": docling_title.byte_start,
                "source_byte_end": docling_title.byte_end,
                "structural_unit_id": "aligned-title",
            },
            {
                "candidate_id": "grobid-title",
                "candidate_type": "heading",
                "source": "grobid",
                "artifact_digest": grobid.digest,
                "source_byte_start": grobid_title.byte_start,
                "source_byte_end": grobid_title.byte_end,
                "structural_unit_id": "aligned-title",
            },
        ],
    },)
    rendered, audit, events = render_document_skeleton(
        docling.text,
        _audit_with_heading_candidate(docling, "Proven title", "docling-title"),
        selected,
        decision_trace=decision_trace,
    )

    rendered, audit, role_events = render_document_role_slots(
        rendered,
        audit,
        selected,
        {"docling": selected, "grobid": peer},
        decision_trace=decision_trace,
    )

    assert rendered.startswith("# Proven title\n\n## REVIEW")
    assert rendered.index("## Acknowledgments") < rendered.index("## References")
    assert "1. Alpha et al. (2024). One." in rendered
    assert "2. Beta et al. (2025). Two." in rendered
    assert rendered.count("Proven title") == 1
    assert sum(
        entry["output_byte_end"] - entry["output_byte_start"] for entry in audit
    ) == len(rendered.encode())
    assert events or role_events


def _title_choice_receipt(selected_id, *, choice=1):
    return {
        "selected_candidate_id": selected_id,
        "request_sha256": "a" * 64,
        "response_choice": choice,
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
    }


def test_model_title_choice_moves_one_existing_heading_to_reader_visible_byte_zero():
    docling = SourceArtifact.from_text(
        "docling",
        (
            "Published in final edited form.\n\n"
            "## Research Article\n\n"
            "## A source-backed scientific title\n\n"
            "## Abstract\n\nSummary.\n\n## Results\n\nBody.\n"
        ),
    )
    grobid = SourceArtifact.from_text(
        "grobid",
        "# A source-backed scientific title\n\n## Abstract\n\nSummary.\n",
    )
    selected = build_document_skeleton(docling, None)
    peer = build_document_skeleton(
        grobid,
        _native(
            "grobid",
            grobid,
            b"<TEI><teiHeader><titleStmt><title>A source-backed scientific title</title>"
            b"</titleStmt></teiHeader><text><body/></text></TEI>",
        ),
    )
    calls = []

    def choose_title(**kwargs):
        calls.append(kwargs)
        selected_choice = next(
            choice
            for choice in kwargs["choices"]
            if choice.get("heading_ordinals") == [2]
        )
        return _title_choice_receipt(selected_choice["candidate_id"], choice=2)

    rendered, audit, events = render_document_role_slots(
        docling.text,
        _audit(docling),
        selected,
        {"docling": selected, "grobid": peer},
        artifacts={"docling": docling, "grobid": grobid},
        title_selection_resolver=choose_title,
    )

    from agr_abc_document_parsers import read_markdown

    assert rendered.startswith("# A source-backed scientific title\n\n")
    assert "Published in final edited form." in rendered
    assert read_markdown(rendered).title == "A source-backed scientific title"
    assert len(calls) == 1
    assert any(
        event.get("operation") == "alliance_model_title_selection"
        and event.get("outcome") == "selected"
        for event in events
    )
    assert not any(
        event.get("operation") == "alliance_role_binding_unresolved"
        for event in events
    )
    assert sum(
        entry["output_byte_end"] - entry["output_byte_start"] for entry in audit
    ) == len(rendered.encode())

    rerendered, rerendered_audit, _events = render_document_role_slots(
        rendered,
        audit,
        selected,
        {"docling": selected, "grobid": peer},
        artifacts={"docling": docling, "grobid": grobid},
        title_selection_resolver=choose_title,
    )
    assert rerendered == rendered
    assert rerendered_audit == audit
    assert len(calls) == 1


def test_model_title_choice_joins_only_existing_adjacent_title_headings():
    docling = SourceArtifact.from_text(
        "docling",
        (
            "Copyright notice.\n\n"
            "## Meeting Report\n\n"
            "## Review of transcriptional mechanisms\n\n"
            "## Taos symposium, February 1998\n\n"
            "## Abstract\n\nSummary.\n\n## Results\n\nBody.\n"
        ),
    )
    selected = build_document_skeleton(docling, None)

    def choose_composite(**kwargs):
        selected_choice = next(
            choice
            for choice in kwargs["choices"]
            if choice.get("heading_ordinals") == [2, 3]
        )
        return _title_choice_receipt(selected_choice["candidate_id"], choice=3)

    rendered, audit, events = render_document_role_slots(
        docling.text,
        _audit(docling),
        selected,
        {"docling": selected},
        artifacts={"docling": docling},
        title_selection_resolver=choose_composite,
    )

    from agr_abc_document_parsers import read_markdown

    expected = "Review of transcriptional mechanisms Taos symposium, February 1998"
    assert rendered.startswith(f"# {expected}\n\n")
    assert read_markdown(rendered).title == expected
    assert rendered.count("Review of transcriptional mechanisms") == 1
    assert any(
        event.get("operation") == "alliance_title_composite_join"
        for event in events
    )
    assert sum(
        entry["output_byte_end"] - entry["output_byte_start"] for entry in audit
    ) == len(rendered.encode())


def test_model_title_none_retains_titleless_output_and_unresolved_receipt():
    docling = SourceArtifact.from_text(
        "docling",
        "## Research Article\n\n## Results\n\nBody.\n",
    )
    grobid = SourceArtifact.from_text("grobid", "# Unknown title\n\nBody.\n")
    selected = build_document_skeleton(docling, None)
    peer = build_document_skeleton(
        grobid,
        _native(
            "grobid",
            grobid,
            b"<TEI><teiHeader><titleStmt><title>Unknown title</title>"
            b"</titleStmt></teiHeader><text><body/></text></TEI>",
        ),
    )

    rendered, _audit_result, events = render_document_role_slots(
        docling.text,
        _audit(docling),
        selected,
        {"docling": selected, "grobid": peer},
        artifacts={"docling": docling, "grobid": grobid},
        title_selection_resolver=lambda **kwargs: _title_choice_receipt(
            kwargs["baseline_id"], choice=0
        ),
    )

    assert rendered.startswith("## Research Article")
    assert any(
        event.get("operation") == "alliance_model_title_selection"
        and event.get("outcome") == "none"
        for event in events
    )
    assert any(
        event.get("operation") == "alliance_role_binding_unresolved"
        for event in events
    )


def test_role_slot_renderer_does_not_reorder_unbound_semantic_label():
    docling = SourceArtifact.from_text(
        "docling",
        "## Results\n\nBody.\n\n## References\n\nNot a proved bibliography container.\n",
    )
    selected = build_document_skeleton(docling, None)
    audit = _audit_with_heading_candidate(docling, "References", "unknown-heading")
    references_entry = next(
        entry for entry in audit if entry.get("candidate_id") == "unknown-heading"
    )
    references_entry.update({
        "source": "marker",
        "artifact_digest": "0" * 64,
    })

    rendered, _audit_result, events = render_document_role_slots(
        docling.text,
        audit,
        selected,
        {"docling": selected},
    )

    assert rendered.index("## Results") < rendered.index("## References")
    assert any(
        event.get("operation") == "alliance_role_binding_unresolved"
        for event in events
    )


def _marker_bibliography_skeleton(text):
    artifact = SourceArtifact.from_text("marker", text)
    children = [
        {"id": "title", "block_type": "Title", "html": "<h1>Title</h1>"},
        {
            "id": "results-heading",
            "block_type": "SectionHeader",
            "html": "<h2>Results</h2>",
        },
        {"id": "body", "block_type": "Text", "html": "<p>Body.</p>"},
        {
            "id": "references-heading",
            "block_type": "SectionHeader",
            "html": "<h2>References</h2>",
        },
        {
            "id": "reference-1",
            "block_type": "Reference",
            "html": "<p>1. Alpha et al. (2024). One.</p>",
        },
        {
            "id": "figure-1",
            "block_type": "Caption",
            "html": "<p>Figure 1. Reader-visible caption.</p>",
        },
    ]
    native = _native(
        "marker",
        artifact,
        {
            "block_type": "Document",
            "children": [{"block_type": "Page", "children": children}],
        },
    )
    return artifact, build_document_skeleton(artifact, native)


def test_role_slot_renderer_moves_native_nonreferences_before_final_bibliography():
    text = (
        "# Title\n\n## Results\n\nBody.\n\n## References\n\n"
        "1. Alpha et al. (2024). One.\n\n"
        "**Figure 1.** Reader-visible caption.\n\n"
        "| Gene | Value |\n|---|---|\n| dpp | 1 |\n"
    )
    artifact, skeleton = _marker_bibliography_skeleton(text)

    rendered, audit, _events = render_document_role_slots(
        text,
        _audit(artifact),
        skeleton,
        {"marker": skeleton},
    )

    from agr_abc_document_parsers import read_markdown

    document = read_markdown(rendered)
    assert rendered.count("## References") == 1
    assert rendered.count("## Figure Legends") == 1
    assert rendered.rindex("## References") > rendered.index("Reader-visible caption")
    assert rendered.rindex("## References") > rendered.index("| Gene | Value |")
    assert len(document.references) == 1
    assert sum(len(section.tables) for section in document.sections) == 1
    assert any("Reader-visible caption" in figure.caption for figure in document.figures)
    assert sum(
        entry["output_byte_end"] - entry["output_byte_start"] for entry in audit
    ) == len(rendered.encode())


@pytest.mark.parametrize(
    "marker",
    [
        "",
        "## Reference\n\n",
        "## References\n\n## References\n\n",
        "## Results\n\nBody.## References\n\n",
    ],
)
def test_role_slot_renderer_canonicalizes_proved_bibliography_heading(marker):
    text = (
        "# Title\n\n## Results\n\nBody.\n\n"
        + marker
        + "1. Alpha et al. (2024). One.\n"
    )
    artifact, skeleton = _marker_bibliography_skeleton(text)

    rendered, _audit_result, _events = render_document_role_slots(
        text,
        _audit(artifact),
        skeleton,
        {"marker": skeleton},
    )

    from agr_abc_document_parsers import read_markdown

    assert rendered.count("## References") == 1
    assert rendered.rstrip().endswith("1. Alpha et al. (2024). One.")
    assert len(read_markdown(rendered).references) == 1


def test_role_slot_renderer_emits_body_figures_in_reader_round_trip_form():
    text = (
        "# Title\n\n## Results\n\nBody.\n\n"
        "**Figure 1. A whole-bold caption.**\n\n"
        "Figure 2. A plain caption.\n\n"
        "## References\n\n1. Alpha et al. (2024). One.\n"
    )
    artifact, skeleton = _marker_bibliography_skeleton(text)

    rendered, audit, _events = render_document_role_slots(
        text,
        _audit(artifact),
        skeleton,
        {"marker": skeleton},
    )

    from agr_abc_document_parsers import emit_markdown, read_markdown

    document = read_markdown(rendered)
    reread = emit_markdown(document)
    assert rendered.count("## Figure Legends") == 1
    assert rendered.index("## Figure Legends") < rendered.index("## References")
    assert [figure.label.rstrip(".") for figure in document.figures] == [
        "Figure 1",
        "Figure 2",
    ]
    assert [figure.caption.strip() for figure in document.figures] == [
        "A whole-bold caption.",
        "A plain caption.",
    ]
    assert "A whole-bold caption." in reread
    assert "A plain caption." in reread
    assert sum(
        entry["output_byte_end"] - entry["output_byte_start"] for entry in audit
    ) == len(rendered.encode())


def test_role_slot_renderer_splits_and_numbers_proved_author_year_references():
    text = (
        "# Title\n\n## Results\n\nBody.\n\n## References\n\n"
        "1. Alpha et al. (2024). One.\n"
        "Beta et al. (2025). Two.\n"
        "2. Gamma et al. (2026). Three.\n"
    )
    artifact, skeleton = _marker_bibliography_skeleton(text)

    rendered, _audit_result, _events = render_document_role_slots(
        text,
        _audit(artifact),
        skeleton,
        {"marker": skeleton},
    )

    from agr_abc_document_parsers import read_markdown

    assert "1. Alpha et al. (2024). One.\n\n" in rendered
    assert "2. Beta et al. (2025). Two.\n\n" in rendered
    assert "3. Gamma et al. (2026). Three." in rendered
    assert len(read_markdown(rendered).references) == 3


def test_role_slot_renderer_converts_reference_footnote_records_to_numbered_entries():
    text = (
        "# Title\n\n## Results\n\nBody.\n\n## References\n\n"
        "[^1]: First retained record.\n"
        "[^2]: Second retained record.\n"
    )
    artifact, skeleton = _marker_bibliography_skeleton(text)

    rendered, _audit_result, _events = render_document_role_slots(
        text,
        _audit(artifact),
        skeleton,
        {"marker": skeleton},
    )

    from agr_abc_document_parsers import emit_markdown, read_markdown

    document = read_markdown(rendered)
    assert "1. First retained record.\n\n2. Second retained record." in rendered
    assert len(document.references) == 2
    assert "First retained record." in emit_markdown(document)
    assert "Second retained record." in emit_markdown(document)


def test_role_slot_renderer_normalizes_malformed_table_label_emphasis():
    text = (
        "# Title\n\n## Results\n\n"
        "| Gene | Value |\n|---|---|\n| dpp | 1 |\n\n"
        "***TABLE 1.** Result* in *flies*.\n"
    )
    artifact = SourceArtifact.from_text("marker", text)
    skeleton = build_document_skeleton(artifact, None)

    rendered, _audit_result, _events = render_document_role_slots(
        text,
        _audit(artifact),
        skeleton,
        {"marker": skeleton},
    )

    from agr_abc_document_parsers import emit_markdown, read_markdown

    document = read_markdown(rendered)
    assert "**TABLE 1.** Result* in *flies*." in rendered
    assert len(document.sections[0].tables) == 1
    reread = emit_markdown(document)
    assert reread.index("| Gene | Value |") < reread.index("**TABLE 1.**")


def test_positive_style_projection_preserves_table_label_schema_boundary():
    donor = SourceArtifact.from_text(
        "marker",
        "# Article\n\n## Results\n\nTABLE 1. Result in flies.\n",
    )
    final = SourceArtifact.from_text(
        "grobid",
        "# Article\n\n## Results\n\n**TABLE 1.** Result in flies.\n",
    )
    skeleton = build_document_skeleton(donor, None)
    paragraph = next(
        occurrence
        for occurrence in skeleton.occurrences
        if occurrence.unit_type == "paragraph"
    )
    span = NativeEmphasisSpan(
        occurrence_id="b" * 64,
        visible_start=0,
        visible_end=len("TABLE 1. Result"),
    )
    styled_paragraph = replace(
        paragraph,
        native_emphasis_occurrence_ids=(span.occurrence_id,),
        native_visible_text="TABLE 1. Result in flies.",
        native_emphasis_spans=(span,),
    )
    skeleton = replace(
        skeleton,
        occurrences=tuple(
            styled_paragraph if occurrence is paragraph else occurrence
            for occurrence in skeleton.occurrences
        ),
        native_body_emphasis_count=1,
        mapped_native_body_emphasis_count=1,
    )

    rendered, _audit_result, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": skeleton},
        {"marker": donor},
    )

    assert rendered == (
        "# Article\n\n## Results\n\n**TABLE 1.** *Result* in flies.\n"
    )
    projected = next(
        event
        for event in events
        if event.get("outcome") == "projected"
        and event.get("boundary") == "open"
    )
    assert projected["boundary_candidate"] == "exact_table_caption_body"
    assert projected["target_visible_start"] == len("TABLE 1. ")
    assert projected["target_visible_end"] == len("TABLE 1. Result")


def test_role_slot_renderer_separates_heading_attached_to_table_row():
    text = (
        "# Article\n\n## Results\n\n"
        "| Gene | Value |\n|---|---|\n"
        "| dpp | 1 |## Acknowledgments\n\nThanks.\n"
    )
    artifact = SourceArtifact.from_text("marker", text)
    skeleton = build_document_skeleton(artifact, None)

    rendered, audit, events = render_document_role_slots(
        text,
        _audit(artifact),
        skeleton,
        {"marker": skeleton},
    )

    from agr_abc_document_parsers import validate_markdown

    assert "| dpp | 1 |\n\n## Acknowledgments\n\nThanks." in rendered
    assert "S08" not in {
        warning.rule_id for warning in validate_markdown(rendered).warnings
    }
    assert any(
        event.get("operation") == "alliance_table_heading_boundary"
        and event.get("audit_span_emitted") is True
        for event in events
    )
    assert sum(
        entry["output_byte_end"] - entry["output_byte_start"]
        for entry in audit
    ) == len(rendered.encode("utf-8"))


def test_role_slot_renderer_normalizes_bounded_alliance_front_matter_markers():
    text = (
        "# Title\n\nReview\n\n"
        "Author One <sup>1</sup>, Author Two <sup>2</sup>\n\n"
        "- 1 First affiliation\n- 2 Second affiliation\n"
        "**ORCIDs:** Author One (0000-0002-5530-5341)\n\n"
        "Abstract: Summary text.\n\n"
        "**Keywords:** one, two\n\n## Introduction\n\nBody.\n"
    )
    artifact = SourceArtifact.from_text("marker", text)
    skeleton = build_document_skeleton(artifact, None)

    rendered, audit, _events = render_document_role_slots(
        text,
        _audit(artifact),
        skeleton,
        {"marker": skeleton},
        artifacts={"marker": artifact},
        title_selection_resolver=lambda **_kwargs: None,
    )

    from agr_abc_document_parsers import read_markdown

    document = read_markdown(rendered)
    assert "**Categories:** Review" in rendered
    assert "1. First affiliation\n2. Second affiliation" in rendered
    assert "https://orcid.org/0000-0002-5530-5341" in rendered
    assert "## Abstract\n\nSummary text." in rendered
    assert document.categories == ["Review"]
    assert len(document.authors) == 2
    assert [author.affiliations for author in document.authors] == [
        ["First affiliation"],
        ["Second affiliation"],
    ]
    assert document.keywords == ["one", "two"]
    assert sum(
        entry["output_byte_end"] - entry["output_byte_start"] for entry in audit
    ) == len(rendered.encode())


def test_transformation_receipts_reconcile_exact_surviving_heading_occurrence():
    docling = SourceArtifact.from_text(
        "docling",
        "# Front matter\n\n# Proven title\n\n## Results\n\nBody.\n",
    )
    grobid = SourceArtifact.from_text(
        "grobid",
        "# Proven title\n\n## Results\n\nBody.\n",
    )
    selected = build_document_skeleton(docling, None)
    peer = build_document_skeleton(
        grobid,
        _native(
            "grobid",
            grobid,
            b"""<TEI><teiHeader><titleStmt><title>Proven title</title></titleStmt>
            </teiHeader><text><body><div><head>Results</head><p>Body.</p></div>
            </body></text></TEI>""",
        ),
    )
    docling_title = next(
        unit
        for unit in scan_structural_units(docling)
        if unit.unit_type == "heading"
        and "Proven title" in docling.raw_utf8[unit.byte_start:unit.byte_end].decode()
    )
    grobid_title = next(
        unit
        for unit in scan_structural_units(grobid)
        if unit.unit_type == "heading"
        and "Proven title" in grobid.raw_utf8[unit.byte_start:unit.byte_end].decode()
    )
    decision_trace = ({
        "selected_candidate_ids": ["docling-second-title"],
        "baseline_candidate_id": "docling-second-title",
        "candidates": [
            {
                "candidate_id": "docling-second-title",
                "candidate_type": "heading",
                "source": "docling",
                "artifact_digest": docling.digest,
                "source_byte_start": docling_title.byte_start,
                "source_byte_end": docling_title.byte_end,
                "structural_unit_id": "aligned-second-title",
            },
            {
                "candidate_id": "grobid-title",
                "candidate_type": "heading",
                "source": "grobid",
                "artifact_digest": grobid.digest,
                "source_byte_start": grobid_title.byte_start,
                "source_byte_end": grobid_title.byte_end,
                "structural_unit_id": "aligned-second-title",
            },
        ],
    },)

    rendered, audit, skeleton_events = render_document_skeleton(
        docling.text,
        _audit_with_heading_candidate(
            docling, "Proven title", "docling-second-title"
        ),
        selected,
        decision_trace=decision_trace,
        force_titleless=True,
    )
    rendered, audit, role_events = render_document_role_slots(
        rendered,
        audit,
        selected,
        {"grobid": peer},
        decision_trace=decision_trace,
    )
    events = reconcile_document_transformations(
        audit, [*skeleton_events, *role_events]
    )
    demotions = [
        event
        for event in events
        if event.get("operation") == "selected_document_skeleton"
    ]

    assert rendered.startswith("# Proven title\n\n## Front matter")
    assert demotions[0]["heading_ordinal"] == 1
    assert demotions[0]["audit_span_emitted"] is True
    assert demotions[1]["heading_ordinal"] == 2
    assert demotions[1]["audit_span_emitted"] is False
    assert demotions[1]["audit_span_superseded"] is True
    final_ids = {
        entry.get("transformation_id")
        for entry in audit
        if entry.get("transformation_id")
    }
    assert demotions[0]["transformation_id"] in final_ids
    assert demotions[1]["transformation_id"] not in final_ids


def test_role_slot_renderer_does_not_promote_duplicate_title_occurrences():
    docling = SourceArtifact.from_text(
        "docling",
        "## Proven title\n\nBody.\n\n## Proven title\n\nMore body.\n",
    )
    grobid = SourceArtifact.from_text("grobid", "# Proven title\n\nBody.\n")
    peer = build_document_skeleton(
        grobid,
        _native(
            "grobid",
            grobid,
            b"""<TEI><teiHeader><titleStmt><title>Proven title</title></titleStmt>
            </teiHeader><text><body><p>Body.</p></body></text></TEI>""",
        ),
    )
    selected = build_document_skeleton(docling, None)

    rendered, audit, _events = render_document_role_slots(
        docling.text,
        _audit(docling),
        selected,
        {"docling": selected, "grobid": peer},
    )

    assert rendered.count("## Proven title") == 2
    assert "# Proven title" not in rendered.replace("## Proven title", "")
    assert sum(
        entry["output_byte_end"] - entry["output_byte_start"] for entry in audit
    ) == len(rendered.encode())


def test_same_heading_count_with_changed_occurrence_cannot_receive_title_role():
    artifact = SourceArtifact.from_text(
        "grobid",
        "# Proven title\n\n## Results\n\nBody.\n",
    )
    native = _native(
        "grobid",
        artifact,
        b"""<TEI><teiHeader><titleStmt><title>Proven title</title></titleStmt>
        </teiHeader><text><body><div><head>Results</head></div></body></text></TEI>""",
    )
    skeleton = build_document_skeleton(artifact, native)
    changed = "# Bogus title!\n\n## Results\n\nBody.\n"

    rendered, _audit_result, events = render_document_skeleton(
        changed,
        _audit(artifact),
        skeleton,
    )

    assert rendered.startswith("## Bogus title!")
    assert all(event["render_mode"] == "safe_titleless" for event in events)
    assert all(event["reason"] == "occurrence_binding_mismatch" for event in events)


def test_selected_peer_heading_in_exact_aligned_slot_receives_skeleton_role():
    artifact = SourceArtifact.from_text(
        "grobid",
        "# Proven title\n\nBody.\n",
    )
    native = _native(
        "grobid",
        artifact,
        b"""<TEI><teiHeader><titleStmt><title>Proven title</title></titleStmt>
        </teiHeader><text><body/></text></TEI>""",
    )
    skeleton = build_document_skeleton(artifact, native)
    peer = SourceArtifact.from_text("marker", "# Peer heading\n\nBody.\n")
    shared_unit_id = "baseline-unit-0000"
    baseline_candidate_id = "region-0000-grobid"
    peer_candidate_id = "region-0000-marker"
    audit = [{
        "output_byte_start": 0,
        "output_byte_end": len(peer.raw_utf8),
        "source": "marker",
        "artifact_digest": peer.digest,
        "source_byte_start": 0,
        "source_byte_end": len(peer.raw_utf8),
        "candidate_id": peer_candidate_id,
        "region_id": "region-0000",
        "decision_method": "model_selected",
    }]
    trace = ({
        "baseline_candidate_id": baseline_candidate_id,
        "selected_candidate_ids": [peer_candidate_id],
        "candidates": [
            {
                "candidate_id": baseline_candidate_id,
                "candidate_type": "heading",
                "source": "grobid",
                "artifact_digest": artifact.digest,
                "source_byte_start": skeleton.headings[0].source_byte_start,
                "source_byte_end": skeleton.headings[0].source_byte_end,
                "structural_unit_id": shared_unit_id,
            },
            {
                "candidate_id": peer_candidate_id,
                "candidate_type": "heading",
                "source": "marker",
                "artifact_digest": peer.digest,
                "source_byte_start": 0,
                "source_byte_end": len("# Peer heading".encode()),
                "structural_unit_id": shared_unit_id,
            },
        ],
    },)

    rendered, _audit_result, events = render_document_skeleton(
        peer.text,
        audit,
        skeleton,
        decision_trace=trace,
    )

    assert rendered.startswith("# Peer heading")
    assert events == []


def test_docling_body_order_proves_front_matter_before_title():
    artifact = SourceArtifact.from_text(
        "docling",
        "## REVIEW\n\n## Proven title\n\n# Results\n\nBody.\n",
    )
    native = _native(
        "docling",
        artifact,
        {
            "schema_name": "DoclingDocument",
            "body": {
                "children": [
                    {"$ref": "#/texts/0"},
                    {"$ref": "#/texts/1"},
                    {"$ref": "#/texts/2"},
                ]
            },
            "texts": [
                {"self_ref": "#/texts/0", "label": "section_header", "text": "REVIEW"},
                {"self_ref": "#/texts/1", "label": "title", "text": "Proven title"},
                {"self_ref": "#/texts/2", "label": "section_header", "text": "Results"},
            ],
        },
    )

    hints = native_heading_hints(native)
    skeleton = build_document_skeleton(artifact, native)
    rendered, _audit_result, _events = render_document_skeleton(
        artifact.text, _audit(artifact), skeleton
    )

    assert [hint.role for hint in hints] == ["section", "title", "section"]
    assert [heading.role for heading in skeleton.headings] == [
        "metadata",
        "title",
        "section",
    ]
    assert rendered.startswith("REVIEW\n\n# Proven title\n\n## Results")


def test_native_body_occurrences_keep_exact_order_pages_regions_and_slots():
    artifact = SourceArtifact.from_text(
        "docling",
        (
            "# Proven title\n\n## Results\n\nRepeated payload.\n\n"
            "## Results\n\nRepeated payload.\n\n"
            "## References\n\nReference payload.\n"
        ),
    )
    labels = [
        ("title", "Proven title", 1),
        ("section_header", "Results", 2),
        ("text", "Repeated payload.", 2),
        ("section_header", "Results", 3),
        ("text", "Repeated payload.", 3),
        ("section_header", "References", 4),
        ("text", "Reference payload.", 4),
    ]
    native = _native(
        "docling",
        artifact,
        {
            "schema_name": "DoclingDocument",
            "body": {
                "children": [
                    {"$ref": f"#/texts/{index}"}
                    for index in range(len(labels))
                ]
            },
            "texts": [
                {
                    "self_ref": f"#/texts/{index}",
                    "label": label,
                    "text": text,
                    "prov": [{"page_no": page}],
                }
                for index, (label, text, page) in enumerate(labels)
            ],
        },
    )

    skeleton = build_document_skeleton(artifact, native)
    repeated = [
        occurrence
        for occurrence in skeleton.occurrences
        if occurrence.native_id in {"#/texts/2", "#/texts/4"}
    ]
    references = [
        occurrence
        for occurrence in skeleton.occurrences
        if occurrence.native_id in {"#/texts/5", "#/texts/6"}
    ]

    assert len(skeleton.occurrences) == 7
    assert skeleton.native_mapped_occurrence_count == 7
    assert [item.page_no for item in repeated] == [2, 3]
    assert repeated[0].occurrence_id != repeated[1].occurrence_id
    assert repeated[0].source_byte_start < repeated[1].source_byte_start
    assert repeated[0].slot_key != repeated[1].slot_key
    assert all(item.region == "back" for item in references)
    assert references[0].section_slot == references[1].section_slot


def test_chooser_prefers_complete_zero_error_native_title_candidate(monkeypatch):
    title = SourceArtifact.from_text(
        "grobid", "# A scientific article title\n\n## Results\n\nBody.\n"
    )
    late = SourceArtifact.from_text(
        "marker", "## Front matter\n\n# A body subsection\n\nBody.\n"
    )
    title_native = _native(
        "grobid",
        title,
        b"""<TEI><teiHeader><titleStmt><title> A scientific article title </title>
        </titleStmt></teiHeader><text><body><div><head>Results</head></div></body></text></TEI>""",
    )
    skeletons = {
        "grobid": build_document_skeleton(title, title_native),
        "marker": build_document_skeleton(late, None),
    }
    monkeypatch.setattr(
        "app.services.document_skeleton.abc_markdown_report",
        lambda text: {
            "error_rule_ids": [] if text.startswith("# A scientific") else ["S02"],
            "warning_rule_ids": [],
        },
    )

    selection = choose_document_skeleton(
        skeletons,
        {"grobid": title, "marker": late},
        preferred_source="marker",
    )

    assert selection.skeleton.source == "grobid"
    assert [entry["selected"] for entry in selection.trace] == [True, False]


def test_runtime_native_file_disappearing_degrades_only_that_receipt(tmp_path):
    pdf = tmp_path / "paper.pdf"
    markdown = tmp_path / "docling.md"
    pdf.write_bytes(b"exact pdf bytes")
    markdown.write_text("# Title\n\nBody.\n", encoding="utf-8")
    artifact = SourceArtifact.from_text("docling", markdown.read_text())
    persist_native_extractor_artifact(
        source="docling",
        output_filename=markdown,
        native_bytes=json.dumps(
            {
                "schema_name": "DoclingDocument",
                "body": {"children": [{"$ref": "#/texts/0"}]},
                "texts": [
                    {"self_ref": "#/texts/0", "label": "title", "text": "Title"}
                ],
            }
        ).encode(),
        native_media_type="application/json",
        pdf_path=pdf,
        extractor_versions={"docling": "2.113.0", "docling-core": "2.87.1"},
        options={
            "do_ocr": True,
            "generate_parsed_pages": True,
            "native_style_cell_collection": "word_cells",
            "native_style_sidecar": True,
        },
        native_style_bytes=unavailable_native_style_bytes("docling", "fixture"),
    )
    native_artifact_path(markdown, "docling").unlink()

    loaded, failures = load_runtime_native_structures(
        {"docling": artifact},
        {"docling": markdown},
        expected_pdf_sha256=sha256_file(pdf),
    )

    assert loaded == {}
    assert failures == {"docling": "FileNotFoundError"}


def test_native_heading_emphasis_projects_to_a_final_heading():
    marker = SourceArtifact.from_text(
        "marker",
        "# Article\n\n## Gene dpp response\n\nBody.\n",
    )
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {
                        "id": "/page/0/Title/0",
                        "block_type": "Title",
                        "html": "<h1>Article</h1>",
                    },
                    {
                        "id": "/page/0/SectionHeader/1",
                        "block_type": "SectionHeader",
                        "html": "<h2>Gene <i>dpp</i> response</h2>",
                    },
                    {
                        "id": "/page/0/Text/2",
                        "block_type": "Text",
                        "html": "<p>Body.</p>",
                    },
                ],
            }],
        },
    )
    skeleton = build_document_skeleton(marker, native)

    rendered, _audit_result, events = project_native_emphasis(
        marker.text,
        _audit(marker),
        {"marker": skeleton},
        {"marker": marker},
    )

    assert rendered == "# Article\n\n## Gene *dpp* response\n\nBody.\n"
    projected = next(
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    )
    assert projected["donor_unit_type"] == "heading"
    assert projected["target_unit_type"] == "heading"


def test_native_heading_emphasis_projects_to_a_final_caption():
    marker = SourceArtifact.from_text(
        "marker",
        "# Article\n\n## Figure 1. Gene dpp response.\n",
    )
    final = SourceArtifact.from_text(
        "grobid",
        "# Article\n\n## Results\n\nFigure 1. Gene dpp response.\n",
    )
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {
                        "id": "/page/0/Title/0",
                        "block_type": "Title",
                        "html": "<h1>Article</h1>",
                    },
                    {
                        "id": "/page/0/SectionHeader/1",
                        "block_type": "SectionHeader",
                        "html": "<h2>Figure 1. Gene <i>dpp</i> response.</h2>",
                    },
                ],
            }],
        },
    )

    rendered, _audit_result, events = project_native_emphasis(
        final.text,
        _audit(final),
        {
            "marker": build_document_skeleton(marker, native),
            "grobid": build_document_skeleton(final, None),
        },
        {"marker": marker, "grobid": final},
    )

    assert rendered == (
        "# Article\n\n## Results\n\nFigure 1. Gene *dpp* response.\n"
    )
    projected = next(
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    )
    assert projected["donor_unit_type"] == "heading"
    assert projected["target_unit_type"] == "figure_caption"


def test_body_title_evidence_can_style_an_existing_front_h1():
    marker = SourceArtifact.from_text(
        "marker", "Study of species alpha in tissue.\n"
    )
    final = SourceArtifact.from_text(
        "grobid",
        "# Study of species alpha in tissue\n\nAuthors and affiliations.\n",
    )
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [{
                    "id": "/page/0/Text/0",
                    "block_type": "Text",
                    "html": "<p>Study of species <i>alpha</i> in tissue.</p>",
                }],
            }],
        },
    )

    rendered, _audit_result, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": build_document_skeleton(marker, native)},
        {"marker": marker},
    )

    assert rendered == (
        "# Study of species *alpha* in tissue\n\nAuthors and affiliations.\n"
    )
    projected = next(
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    )
    assert projected["target_unit_type"] == "heading"


def test_repeated_top_margin_native_italics_are_page_furniture_not_body_style():
    marker = SourceArtifact.from_text(
        "marker",
        "Journal Name Page 2\n\nBody one.\n\nJournal Name Page 3\n\nBody two.\n",
    )
    final = SourceArtifact.from_text("grobid", "Body one.\n\nBody two.\n")
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [
                {
                    "id": f"/page/{page}/Page/0",
                    "block_type": "Page",
                    "bbox": [0.0, 0.0, 600.0, 800.0],
                    "children": [
                        {
                            "id": f"/page/{page}/Text/0",
                            "block_type": "Text",
                            "bbox": [50.0, 25.0, 550.0, 40.0],
                            "html": (
                                f"<p>Journal <i>Name</i> Page {page + 2}</p>"
                            ),
                        },
                        {
                            "id": f"/page/{page}/Text/1",
                            "block_type": "Text",
                            "bbox": [50.0, 100.0, 550.0, 130.0],
                            "html": f"<p>Body {'one' if page == 0 else 'two'}.</p>",
                        },
                    ],
                }
                for page in range(2)
            ],
        },
    )
    skeleton = build_document_skeleton(marker, native)

    assert skeleton.native_body_emphasis_count == 2
    assert sum(
        occurrence.native_unit_type == "page_header"
        and occurrence.native_emphasis_count == 1
        for occurrence in skeleton.occurrences
    ) == 2

    rendered, audit, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": skeleton},
        {"marker": marker},
    )

    assert rendered == final.text
    assert audit == _audit(final)
    furniture = [
        event
        for event in events
        if event.get("outcome") == "declined"
        and event.get("reason") == "native_page_furniture"
    ]
    assert len(furniture) == 2
    assert all(event["protected_claim"] is True for event in furniture)


def _native_text_only_table_skeleton(artifact):
    skeleton = build_document_skeleton(artifact, None)
    occurrence = next(
        item for item in skeleton.occurrences if item.unit_type == "paragraph"
    )
    span = NativeEmphasisSpan(
        occurrence_id="a" * 64,
        visible_start=5,
        visible_end=8,
    )
    occurrence = replace(
        occurrence,
        unit_type="table",
        native_id="native-table-1",
        native_order=0,
        native_emphasis_occurrence_ids=(span.occurrence_id,),
        native_visible_text="Gene dpp works.",
        native_emphasis_spans=(span,),
    )
    return replace(
        skeleton,
        occurrences=(occurrence,),
        native_mapped_occurrence_count=1,
        native_body_emphasis_count=1,
        mapped_native_body_emphasis_count=1,
    )


def test_native_text_fallback_places_a_claim_from_an_unsupported_markdown_unit():
    marker = SourceArtifact.from_text("marker", "Gene dpp works.\n")
    final = SourceArtifact.from_text("grobid", "Gene dpp works.\n")

    rendered, _audit_result, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": _native_text_only_table_skeleton(marker)},
        {"marker": marker},
    )

    assert rendered == "Gene *dpp* works.\n"
    projected = next(
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    )
    assert projected["direct_native_donor"] is True
    assert projected["donor_unit_type"] == "table"


def test_native_text_fallback_abstains_when_the_claim_interval_does_not_map():
    marker = SourceArtifact.from_text("marker", "Gene dpp works.\n")
    final = SourceArtifact.from_text("grobid", "XYZ QQQ.\n")

    rendered, audit, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": _native_text_only_table_skeleton(marker)},
        {"marker": marker},
    )

    assert rendered == final.text
    assert audit == _audit(final)
    assert any(
        event.get("reason") == "final_unit_pair_unavailable"
        for event in events
    )
    assert not any(
        event.get("reason") == "donor_visible_text_unavailable"
        for event in events
    )


def test_below_floor_mappable_unit_reaches_numbered_selection():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor before.\n\nPreviously described (30).\n\nAnchor after.\n",
    )
    final = SourceArtifact.from_text(
        "grobid",
        "Anchor before.\n\nIncubate for 30 min.\n\nAnchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html="Previously described (<i>30</i>).",
        after="Anchor after.",
    )
    calls = []

    def choose_target(*, baseline_id, choices, **_kwargs):
        calls.append(choices)
        return next(
            choice["candidate_id"]
            for choice in choices
            if choice["candidate_id"] != baseline_id
        )

    rendered, _audit_result, events = project_native_emphasis(
        final.text,
        _audit(final),
        {
            "marker": build_document_skeleton(marker, native),
            "grobid": build_document_skeleton(final, None),
        },
        {"marker": marker, "grobid": final},
        style_selection_resolver=choose_target,
    )

    assert rendered == final.text.replace("30", "*30*")
    assert len(calls) == 1
    projected = next(
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    )
    assert projected["style_selection_below_deterministic_floor"] is True
    assert projected["model_selected_target"] is True


def test_numbered_selection_failure_records_unavailable_without_crashing():
    marker = SourceArtifact.from_text(
        "marker",
        "Anchor before.\n\nPreviously described (30).\n\nAnchor after.\n",
    )
    final = SourceArtifact.from_text(
        "grobid",
        "Anchor before.\n\nIncubate for 30 min.\n\nAnchor after.\n",
    )
    native = _marker_native_with_italic_paragraph(
        marker,
        before="Anchor before.",
        paragraph_html="Previously described (<i>30</i>).",
        after="Anchor after.",
    )

    def unavailable_selector(**_kwargs):
        raise RuntimeError("selection service unavailable")

    rendered, audit, events = project_native_emphasis(
        final.text,
        _audit(final),
        {
            "marker": build_document_skeleton(marker, native),
            "grobid": build_document_skeleton(final, None),
        },
        {"marker": marker, "grobid": final},
        style_selection_resolver=unavailable_selector,
    )

    assert rendered == final.text
    assert audit == _audit(final)
    assert any(
        event.get("outcome") == "declined"
        and event.get("reason") == "model_selection_unavailable"
        and event.get("style_selection_method") == "sol_numbered_choice"
        and event.get("model_selected_target") is False
        for event in events
    )


def test_symbol_only_style_interval_maps_without_a_substitution_table():
    marker = SourceArtifact.from_text("marker", "Sample genotype +/− was measured.\n")
    final = SourceArtifact.from_text("grobid", "Sample genotype +/− was measured.\n")
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [{
                    "id": "/page/0/Text/0",
                    "block_type": "Text",
                    "html": "<p>Sample genotype <i>+/−</i> was measured.</p>",
                }],
            }],
        },
    )

    rendered, _audit_result, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": build_document_skeleton(marker, native)},
        {"marker": marker},
    )

    assert rendered == "Sample genotype *+/−* was measured.\n"
    projected = next(
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    )
    assert projected["target_visible_end"] - projected["target_visible_start"] == 3


def test_non_mappable_deterministic_pair_reaches_claim_mappable_selection():
    marker = SourceArtifact.from_text(
        "marker", "Methods reviewed gene dpp and samples carefully.\n"
    )
    final = SourceArtifact.from_text(
        "grobid",
        (
            "Methods reviewed gene xyz and samples carefully.\n\n"
            "The dpp assay was performed.\n"
        ),
    )
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [{
                    "id": "/page/0/Text/0",
                    "block_type": "Text",
                    "html": "<p>Methods reviewed gene <i>dpp</i> and samples carefully.</p>",
                }],
            }],
        },
    )
    calls = []

    def choose_target(*, baseline_id, choices, **_kwargs):
        calls.append(choices)
        return next(
            choice["candidate_id"]
            for choice in choices
            if choice["candidate_id"] != baseline_id
        )

    rendered, _audit_result, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": build_document_skeleton(marker, native)},
        {"marker": marker},
        style_selection_resolver=choose_target,
    )

    assert rendered.endswith("The *dpp* assay was performed.\n")
    assert len(calls) == 1
    projected = next(
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    )
    assert projected["style_selection_below_deterministic_floor"] is True
    assert projected["model_selected_target"] is True


def test_one_donor_can_project_protected_claims_to_two_final_units():
    marker = SourceArtifact.from_text(
        "marker", "First alpha result and second beta result.\n"
    )
    final = SourceArtifact.from_text(
        "grobid", "First alpha result.\n\nSecond beta result.\n"
    )
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [{
                    "id": "/page/0/Text/0",
                    "block_type": "Text",
                    "html": (
                        "<p>First <i>alpha</i> result and second "
                        "<i>beta</i> result.</p>"
                    ),
                }],
            }],
        },
    )
    calls = []

    def choose_target(*, baseline_id, choices, **_kwargs):
        calls.append(choices)
        return next(
            choice["candidate_id"]
            for choice in choices
            if choice["candidate_id"] != baseline_id
        )

    rendered, _audit_result, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": build_document_skeleton(marker, native)},
        {"marker": marker},
        style_selection_resolver=choose_target,
    )

    assert rendered == "First *alpha* result.\n\nSecond *beta* result.\n"
    assert {
        event["positive_style_claim_id"]
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    } == {
        event["positive_style_claim_id"]
        for event in events
        if event.get("outcome") == "eligible"
    }
    assert len(calls) == 1


def test_split_donor_preserves_executable_claim_group_and_reroutes_remainder():
    marker = SourceArtifact.from_text(
        "marker", "First alpha and gamma result and second beta result.\n"
    )
    final = SourceArtifact.from_text(
        "grobid", "First alpha and gamma result.\n\nSecond beta result.\n"
    )
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [{
                    "id": "/page/0/Text/0",
                    "block_type": "Text",
                    "html": (
                        "<p>First <i>alpha</i> and <i>gamma</i> result and "
                        "second <i>beta</i> result.</p>"
                    ),
                }],
            }],
        },
    )
    calls = []

    def choose_target(*, baseline_id, choices, **_kwargs):
        calls.append(choices)
        return next(
            choice["candidate_id"]
            for choice in choices
            if choice["candidate_id"] != baseline_id
        )

    rendered, _audit_result, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": build_document_skeleton(marker, native)},
        {"marker": marker},
        style_selection_resolver=choose_target,
    )

    assert rendered == (
        "First *alpha* and *gamma* result.\n\nSecond *beta* result.\n"
    )
    assert len(calls) == 1
    selected_claim_ids = {
        event["positive_style_claim_id"]
        for event in events
        if event.get("outcome") == "projected"
        and event.get("boundary") == "open"
        and event.get("model_selected_target") is True
    }
    deterministic_claim_ids = {
        event["positive_style_claim_id"]
        for event in events
        if event.get("outcome") == "projected"
        and event.get("boundary") == "open"
        and event.get("model_selected_target") is not True
    }
    assert len(selected_claim_ids) == 1
    assert len(deterministic_claim_ids) == 2


def test_unrelated_below_floor_unit_never_reaches_numbered_selection():
    marker = SourceArtifact.from_text("marker", "Before ZZZ after.\n")
    final = SourceArtifact.from_text("grobid", "Before AAA after.\n")
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [{
                    "id": "/page/0/Text/0",
                    "block_type": "Text",
                    "html": "<p>Before <i>ZZZ</i> after.</p>",
                }],
            }],
        },
    )
    calls = []

    rendered, audit, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": build_document_skeleton(marker, native)},
        {"marker": marker},
        style_selection_resolver=lambda **kwargs: calls.append(kwargs),
    )

    assert rendered == final.text
    assert audit == _audit(final)
    assert calls == []
    assert any(
        event.get("reason") == "final_unit_pair_unavailable"
        for event in events
    )


def test_auxiliary_only_mapping_cannot_authorize_below_floor_selection():
    marker = SourceArtifact.from_text(
        "marker", "Protected ZZZ and *30*.\n"
    )
    final = SourceArtifact.from_text("grobid", "Incubate for 30 min.\n")
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [{
                    "id": "/page/0/Text/0",
                    "block_type": "Text",
                    "html": "<p>Protected <i>ZZZ</i> and 30.</p>",
                }],
            }],
        },
    )
    calls = []

    rendered, audit, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": build_document_skeleton(marker, native)},
        {"marker": marker},
        style_selection_resolver=lambda **kwargs: calls.append(kwargs),
    )

    assert rendered == final.text
    assert audit == _audit(final)
    assert calls == []
    assert any(
        event.get("protected_claim") is True
        and event.get("reason") == "final_unit_pair_unavailable"
        for event in events
    )


def test_auxiliary_mapping_cannot_keep_a_mixed_protected_pair_alive():
    marker = SourceArtifact.from_text(
        "marker", "Protected ZZZ and *30* after.\n"
    )
    final = SourceArtifact.from_text(
        "grobid", "Protected AAA and 30 after.\n"
    )
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [{
                    "id": "/page/0/Text/0",
                    "block_type": "Text",
                    "html": "<p>Protected <i>ZZZ</i> and 30 after.</p>",
                }],
            }],
        },
    )

    rendered, audit, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": build_document_skeleton(marker, native)},
        {"marker": marker},
    )

    assert rendered == final.text
    assert audit == _audit(final)
    assert not any(event.get("outcome") == "projected" for event in events)
    assert any(
        event.get("protected_claim") is True
        and event.get("reason") == "final_unit_pair_unavailable"
        for event in events
    )


def test_numbered_style_selection_accepts_more_than_seven_target_candidates():
    marker = SourceArtifact.from_text("marker", "Gene dpp.\n")
    final = SourceArtifact.from_text(
        "grobid", "\n\n".join(["Gene dpp."] * 9) + "\n"
    )
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [{
                    "id": "/page/0/Text/0",
                    "block_type": "Text",
                    "html": "<p>Gene <i>dpp</i>.</p>",
                }],
            }],
        },
    )
    calls = []

    def choose_first_target(*, baseline_id, choices, **_kwargs):
        calls.append(choices)
        return next(
            choice["candidate_id"]
            for choice in choices
            if choice["candidate_id"] != baseline_id
        )

    rendered, _audit_result, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": build_document_skeleton(marker, native)},
        {"marker": marker},
        style_selection_resolver=choose_first_target,
    )

    assert rendered == final.text.replace("dpp", "*dpp*", 1)
    assert len(calls) == 1
    assert len(calls[0]) == 10
    projected = next(
        event
        for event in events
        if event.get("outcome") == "projected" and event.get("boundary") == "open"
    )
    assert projected["style_selection_candidate_count"] == 9


def test_model_selected_order_crossing_is_diagnostic_not_a_veto():
    marker = SourceArtifact.from_text(
        "marker",
        "Alpha one.\n\nGene dpp.\n",
    )
    final = SourceArtifact.from_text(
        "grobid",
        "Gene dpp.\n\nAlpha one.\n\nGene dpp.\n",
    )
    native = _native(
        "marker",
        marker,
        {
            "block_type": "Document",
            "children": [{
                "block_type": "Page",
                "children": [
                    {
                        "id": "/page/0/Text/0",
                        "block_type": "Text",
                        "html": "<p>Alpha <i>one</i>.</p>",
                    },
                    {
                        "id": "/page/0/Text/1",
                        "block_type": "Text",
                        "html": "<p>Gene <i>dpp</i>.</p>",
                    },
                ],
            }],
        },
    )

    def choose_first_target(*, baseline_id, choices, **_kwargs):
        return next(
            choice["candidate_id"]
            for choice in choices
            if choice["candidate_id"] != baseline_id
        )

    rendered, _audit_result, events = project_native_emphasis(
        final.text,
        _audit(final),
        {"marker": build_document_skeleton(marker, native)},
        {"marker": marker},
        style_selection_resolver=choose_first_target,
    )

    assert rendered == "Gene *dpp*.\n\nAlpha *one*.\n\nGene dpp.\n"
    crossing = next(
        event
        for event in events
        if event.get("outcome") == "projected"
        and event.get("boundary") == "open"
        and event.get("style_selection_order_crossing") is True
    )
    assert crossing["model_selected_target"] is True
    assert crossing["style_selection_donor_ordinal"] == 1
    assert crossing["style_selection_target_ordinal"] == 0
    assert not any(
        event.get("reason") == "model_selection_non_monotone"
        for event in events
    )
