import pytest
from unittest.mock import MagicMock
from app.services.llm_service import (
    ConflictResolutionResponse,
    ResolvedSegment,
    LLM,
    TokenAccumulator,
)


class DummyLLM(LLM):
    def __init__(self):
        self.client = MagicMock()
        self.model = "dummy-model"
        self.reasoning_effort = "medium"
        self.conflict_batch_size = 10
        self.conflict_max_workers = 4
        self.conflict_retry_rounds = 2
        self.usage = TokenAccumulator()


def test_resolve_conflicts_success():
    llm = DummyLLM()
    conflicts = [
        {"segment_id": "seg_001", "block_type": "paragraph",
         "context_before": "prev", "context_after": "next",
         "grobid": "text a", "docling": "text b", "marker": "text c"},
        {"segment_id": "seg_002", "block_type": "paragraph",
         "context_before": "prev2", "context_after": "next2",
         "grobid": "x", "docling": "y", "marker": "z"},
    ]

    message = MagicMock()
    message.parsed = ConflictResolutionResponse(
        resolved=[
            ResolvedSegment(segment_id="seg_001", action="keep", text="resolved text a"),
            ResolvedSegment(segment_id="seg_002", action="keep", text="resolved text b"),
        ]
    )
    message.refusal = None

    completion = MagicMock()
    completion.choices = [MagicMock(message=message)]
    llm.client.chat.completions.parse.return_value = completion

    result = llm.resolve_conflicts(conflicts)
    assert result == {"seg_001": "resolved text a", "seg_002": "resolved text b"}
    llm.client.chat.completions.parse.assert_called_once()


def test_resolve_conflicts_retry_on_transient_error():
    """First parse call fails, second succeeds."""
    llm = DummyLLM()
    conflicts = [
        {"segment_id": "seg_001", "block_type": "paragraph",
         "context_before": "", "context_after": "",
         "grobid": "a", "docling": "b", "marker": "c"},
    ]

    message = MagicMock()
    message.parsed = ConflictResolutionResponse(resolved=[ResolvedSegment(segment_id="seg_001", action="keep", text="fixed text")])
    message.refusal = None
    good_completion = MagicMock()
    good_completion.choices = [MagicMock(message=message)]

    llm.client.chat.completions.parse.side_effect = [Exception("boom"), good_completion]

    result = llm.resolve_conflicts(conflicts)
    assert result == {"seg_001": "fixed text"}
    assert llm.client.chat.completions.parse.call_count == 2


def test_resolve_conflicts_both_attempts_fail():
    """All batched rounds fail — should raise."""
    llm = DummyLLM()
    conflicts = [
        {"segment_id": "seg_001", "block_type": "paragraph",
         "context_before": "", "context_after": "",
         "grobid": "a", "docling": "b", "marker": "c"},
    ]

    llm.client.chat.completions.parse.side_effect = Exception("down")

    with pytest.raises(Exception, match="resolve_conflicts failed after batched retries"):
        llm.resolve_conflicts(conflicts)


def test_resolve_conflicts_missing_segment_id():
    """Response missing a requested segment_id — should retry."""
    llm = DummyLLM()
    conflicts = [
        {"segment_id": "seg_001", "block_type": "paragraph",
         "context_before": "", "context_after": "",
         "grobid": "a", "docling": "b", "marker": "c"},
        {"segment_id": "seg_002", "block_type": "paragraph",
         "context_before": "", "context_after": "",
         "grobid": "x", "docling": "y", "marker": "z"},
    ]

    # First: missing seg_002
    partial_message = MagicMock()
    partial_message.parsed = ConflictResolutionResponse(resolved=[ResolvedSegment(segment_id="seg_001", action="keep", text="text")])
    partial_message.refusal = None
    partial_completion = MagicMock()
    partial_completion.choices = [MagicMock(message=partial_message)]

    # Second: complete
    complete_message = MagicMock()
    complete_message.parsed = ConflictResolutionResponse(
        resolved=[
            ResolvedSegment(segment_id="seg_001", action="keep", text="text"),
            ResolvedSegment(segment_id="seg_002", action="keep", text="text2"),
        ]
    )
    complete_message.refusal = None
    complete_completion = MagicMock()
    complete_completion.choices = [MagicMock(message=complete_message)]

    llm.client.chat.completions.parse.side_effect = [partial_completion, complete_completion]

    result = llm.resolve_conflicts(conflicts)
    assert "seg_001" in result and "seg_002" in result


def test_resolve_conflicts_retry_on_refusal():
    llm = DummyLLM()
    conflicts = [
        {"segment_id": "seg_001", "block_type": "paragraph",
         "context_before": "", "context_after": "",
         "grobid": "a", "docling": "b", "marker": "c"},
    ]

    refusal_message = MagicMock()
    refusal_message.parsed = None
    refusal_message.refusal = "I cannot help with that request."
    refusal_completion = MagicMock()
    refusal_completion.choices = [MagicMock(message=refusal_message)]

    good_message = MagicMock()
    good_message.parsed = ConflictResolutionResponse(resolved=[ResolvedSegment(segment_id="seg_001", action="keep", text="ok")])
    good_message.refusal = None
    good_completion = MagicMock()
    good_completion.choices = [MagicMock(message=good_message)]

    llm.client.chat.completions.parse.side_effect = [refusal_completion, good_completion]
    result = llm.resolve_conflicts(conflicts)
    assert result == {"seg_001": "ok"}


# ---------------------------------------------------------------------------
# Zone resolution tests
# ---------------------------------------------------------------------------

def test_resolve_conflict_zones_single_zone():
    """Single zone with one conflict should resolve successfully."""
    llm = DummyLLM()

    zone = {
        "zone_id": "zone_0",
        "context_before": [{"segment_id": "seg_000", "text": "Before", "status": "agreed"}],
        "segments": [
            {"segment_id": "seg_001", "status": "conflict", "block_type": "paragraph",
             "grobid": "text a", "docling": "text b", "marker": "text c"},
        ],
        "context_after": [{"segment_id": "seg_002", "text": "After", "status": "agreed"}],
        "triple_indices": [1],
    }

    message = MagicMock()
    message.parsed = ConflictResolutionResponse(
        resolved=[ResolvedSegment(segment_id="seg_001", action="keep", text="resolved zone text")]
    )
    message.refusal = None
    completion = MagicMock()
    completion.choices = [MagicMock(message=message)]
    llm.client.chat.completions.parse.return_value = completion

    result, unresolved = llm.resolve_conflict_zones([zone])
    assert result == {"seg_001": "resolved zone text"}
    assert unresolved == set()


def test_resolve_conflict_zones_multiple_zones():
    """Multiple zones should all be resolved."""
    llm = DummyLLM()

    zones = [
        {
            "zone_id": "zone_0",
            "context_before": [],
            "segments": [
                {"segment_id": "seg_001", "status": "conflict", "block_type": "paragraph",
                 "grobid": "a", "docling": "b", "marker": "c"},
            ],
            "context_after": [],
            "triple_indices": [1],
        },
        {
            "zone_id": "zone_1",
            "context_before": [],
            "segments": [
                {"segment_id": "seg_005", "status": "conflict", "block_type": "paragraph",
                 "grobid": "x", "docling": "y", "marker": "z"},
            ],
            "context_after": [],
            "triple_indices": [5],
        },
    ]

    # Each call resolves one zone
    msg1 = MagicMock()
    msg1.parsed = ConflictResolutionResponse(resolved=[ResolvedSegment(segment_id="seg_001", action="keep", text="r1")])
    msg1.refusal = None
    comp1 = MagicMock()
    comp1.choices = [MagicMock(message=msg1)]

    msg2 = MagicMock()
    msg2.parsed = ConflictResolutionResponse(resolved=[ResolvedSegment(segment_id="seg_005", action="keep", text="r2")])
    msg2.refusal = None
    comp2 = MagicMock()
    comp2.choices = [MagicMock(message=msg2)]

    llm.client.chat.completions.parse.side_effect = [comp1, comp2]

    result, unresolved = llm.resolve_conflict_zones(zones)
    assert result["seg_001"] == "r1"
    assert result["seg_005"] == "r2"
    assert unresolved == set()


def test_resolve_conflict_zones_retry_on_failure():
    """Zone that fails once should be retried."""
    llm = DummyLLM()

    zone = {
        "zone_id": "zone_0",
        "context_before": [],
        "segments": [
            {"segment_id": "seg_001", "status": "conflict", "block_type": "paragraph",
             "grobid": "a", "docling": "b", "marker": "c"},
        ],
        "context_after": [],
        "triple_indices": [1],
    }

    msg = MagicMock()
    msg.parsed = ConflictResolutionResponse(resolved=[ResolvedSegment(segment_id="seg_001", action="keep", text="ok")])
    msg.refusal = None
    good_comp = MagicMock()
    good_comp.choices = [MagicMock(message=msg)]

    llm.client.chat.completions.parse.side_effect = [Exception("boom"), good_comp]

    result, unresolved = llm.resolve_conflict_zones([zone])
    assert result == {"seg_001": "ok"}
    assert unresolved == set()
    assert llm.client.chat.completions.parse.call_count == 2


def test_resolve_conflict_zones_all_fail():
    """All retries fail — returns empty resolved + unresolved IDs (no raise)."""
    llm = DummyLLM()

    zone = {
        "zone_id": "zone_0",
        "context_before": [],
        "segments": [
            {"segment_id": "seg_001", "status": "conflict", "block_type": "paragraph",
             "grobid": "a", "docling": "b", "marker": "c"},
        ],
        "context_after": [],
        "triple_indices": [1],
    }

    llm.client.chat.completions.parse.side_effect = Exception("down")

    result, unresolved = llm.resolve_conflict_zones([zone])
    assert result == {}
    assert "seg_001" in unresolved
