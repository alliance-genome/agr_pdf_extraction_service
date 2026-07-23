import hashlib

import pytest
from pydantic import ValidationError

from app.services.source_contracts import (
    CandidateGraph,
    CandidateStore,
    ConsensusContractError,
    RegionCandidateGraph,
    RegionSelectionDecision,
    SelectionDecisionResponse,
    SourceArtifact,
    SourceSpanCandidate,
    unsafe_unicode_characters,
)


def _store():
    baseline = SourceArtifact.from_text("grobid", "alpha")
    alternative = SourceArtifact.from_text("docling", "beta")
    candidates = {
        "base": SourceSpanCandidate(
            candidate_id="base", occurrence_id="base-1", structural_unit_id="u1",
            source="grobid", artifact_digest=baseline.digest, byte_start=0,
            byte_end=5, candidate_type="prose", comparison_key="alpha",
        ),
        "alt": SourceSpanCandidate(
            candidate_id="alt", occurrence_id="alt-1", structural_unit_id="u1",
            source="docling", artifact_digest=alternative.digest, byte_start=0,
            byte_end=4, candidate_type="prose", comparison_key="beta",
        ),
    }
    store = CandidateStore(
        {
            ("grobid", baseline.digest): baseline,
            ("docling", alternative.digest): alternative,
        },
        candidates,
    )
    graph = CandidateGraph(regions=(RegionCandidateGraph(
        region_id="r1", baseline_candidate_id="base",
        valid_paths=(("base",), ("alt",)),
    ),))
    return baseline, alternative, store, graph


def test_source_artifact_binds_exact_utf8_bytes_and_digest():
    artifact = SourceArtifact.from_text("marker", "Gene γ")
    assert artifact.raw_utf8 == "Gene γ".encode()
    assert artifact.digest == hashlib.sha256(artifact.raw_utf8).hexdigest()
    with pytest.raises(ConsensusContractError, match="digest mismatch"):
        SourceArtifact("marker", artifact.raw_utf8, "0" * 64)


def test_unsafe_unicode_detection_rejects_controls_and_bidi_formatting():
    assert unsafe_unicode_characters("safe\n") == ()
    assert unsafe_unicode_characters("bad\x03\u202e") == ("\x03", "\u202e")


def test_candidate_store_executes_only_an_allowed_exact_source_path():
    _baseline, _alternative, store, graph = _store()
    selected = RegionSelectionDecision(
        region_id="r1", action="select_candidates", candidate_ids=("alt",)
    )
    assert store.resolve_region(selected, region=graph.regions[0]) == b"beta"
    invalid = RegionSelectionDecision(
        region_id="r1", action="select_candidates", candidate_ids=("base", "alt")
    )
    with pytest.raises(ConsensusContractError, match="not an executable path"):
        store.resolve_region(invalid, region=graph.regions[0])


def test_model_response_contract_has_no_publication_text_field():
    with pytest.raises(ValidationError):
        RegionSelectionDecision.model_validate({
            "region_id": "r1", "action": "keep_baseline", "chosen_text": "authored"
        })
    response = SelectionDecisionResponse(decisions=(
        RegionSelectionDecision(region_id="r1", action="keep_baseline"),
    ))
    assert "text" not in response.model_dump_json()


def test_selection_request_contains_numbered_evidence_but_no_output_field():
    _baseline, _alternative, store, graph = _store()
    request = store.build_selection_request(graph)
    payload = request.model_dump()
    assert payload["regions"][0]["valid_paths"] == (("base",), ("alt",))
    assert "output" not in payload["regions"][0]
