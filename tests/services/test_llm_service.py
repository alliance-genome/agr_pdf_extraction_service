import pytest
from unittest.mock import MagicMock
from app.services.llm_service import (
    ConflictResolutionResponse,
    MicroConflictResolutionResponse,
    ResolvedSegment,
    ResolvedMicroConflict,
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
# Micro-conflict resolution tests
# ---------------------------------------------------------------------------

def test_resolve_micro_conflicts_single_payload():
    """Single payload should resolve all micro-conflict IDs."""
    llm = DummyLLM()

    payload = {
        "segment_id": "seg_001",
        "block_type": "paragraph",
        "micro_conflicts": [
            {
                "conflict_id": "seg_001_mc_0",
                "context_before": "The result was",
                "disagreement": {"grobid": "95%", "docling": "96%", "marker": "97%"},
                "context_after": "in all runs.",
            },
        ],
    }

    message = MagicMock()
    message.parsed = MicroConflictResolutionResponse(
        resolved=[ResolvedMicroConflict(conflict_id="seg_001_mc_0", text="95.5%")]
    )
    message.refusal = None
    completion = MagicMock()
    completion.choices = [MagicMock(message=message)]
    llm.client.chat.completions.parse.return_value = completion

    result = llm.resolve_micro_conflicts(payload)
    assert result.resolved[0].conflict_id == "seg_001_mc_0"
    assert result.resolved[0].text == "95.5%"


def test_resolve_micro_conflicts_retry_on_partial():
    """Missing IDs in first response should trigger retry."""
    llm = DummyLLM()

    payload = {
        "segment_id": "seg_001",
        "block_type": "paragraph",
        "micro_conflicts": [
            {
                "conflict_id": "seg_001_mc_0",
                "context_before": "",
                "disagreement": {"grobid": "a", "docling": "b", "marker": "c"},
                "context_after": "",
            },
            {
                "conflict_id": "seg_001_mc_1",
                "context_before": "",
                "disagreement": {"grobid": "x", "docling": "y", "marker": "z"},
                "context_after": "",
            },
        ],
    }

    msg1 = MagicMock()
    msg1.parsed = MicroConflictResolutionResponse(
        resolved=[ResolvedMicroConflict(conflict_id="seg_001_mc_0", text="r1")]
    )
    msg1.refusal = None
    comp1 = MagicMock()
    comp1.choices = [MagicMock(message=msg1)]

    msg2 = MagicMock()
    msg2.parsed = MicroConflictResolutionResponse(
        resolved=[ResolvedMicroConflict(conflict_id="seg_001_mc_1", text="r2")]
    )
    msg2.refusal = None
    comp2 = MagicMock()
    comp2.choices = [MagicMock(message=msg2)]

    llm.client.chat.completions.parse.side_effect = [comp1, comp2]

    result = llm.resolve_micro_conflicts(payload)
    resolved = {item.conflict_id: item.text for item in result.resolved}
    assert resolved["seg_001_mc_0"] == "r1"
    assert resolved["seg_001_mc_1"] == "r2"


def test_resolve_micro_conflicts_all_fail():
    """All retries fail should return empty (graceful degradation, no raise)."""
    llm = DummyLLM()
    payload = {
        "segment_id": "seg_001",
        "block_type": "paragraph",
        "micro_conflicts": [
            {
                "conflict_id": "seg_001_mc_0",
                "context_before": "",
                "disagreement": {"grobid": "a", "docling": "b", "marker": "c"},
                "context_after": "",
            },
        ],
    }

    llm.client.chat.completions.parse.side_effect = Exception("down")

    result = llm.resolve_micro_conflicts(payload)
    assert len(result.resolved) == 0


# ---------------------------------------------------------------------------
# Refusal triggers retry + escalation (C-3)
# ---------------------------------------------------------------------------

def test_resolve_micro_conflicts_refusal_triggers_retry():
    """Refusal inside resolve_micro_conflicts should trigger retry/escalation."""
    llm = DummyLLM()
    payload = {
        "segment_id": "seg_001",
        "block_type": "paragraph",
        "micro_conflicts": [
            {
                "conflict_id": "seg_001_mc_0",
                "context_before": "",
                "disagreement": {"grobid": "a", "docling": "b", "marker": "c"},
                "context_after": "",
            },
        ],
    }

    # First call: model refuses
    refusal_message = MagicMock()
    refusal_message.refusal = "I cannot help with that."
    refusal_message.parsed = None
    refusal_completion = MagicMock()
    refusal_completion.choices = [MagicMock(message=refusal_message)]
    refusal_completion.usage = None

    # Second call: succeeds
    good_message = MagicMock()
    good_message.refusal = None
    good_message.parsed = MicroConflictResolutionResponse(
        resolved=[ResolvedMicroConflict(conflict_id="seg_001_mc_0", text="resolved")]
    )
    good_completion = MagicMock()
    good_completion.choices = [MagicMock(message=good_message)]
    good_completion.usage = None

    llm.client.chat.completions.parse.side_effect = [refusal_completion, good_completion]

    result = llm.resolve_micro_conflicts(payload)
    resolved = {item.conflict_id: item.text for item in result.resolved}
    assert resolved["seg_001_mc_0"] == "resolved"
    assert llm.client.chat.completions.parse.call_count == 2


# ---------------------------------------------------------------------------
# Empty text with action=drop is accepted (C-4)
# ---------------------------------------------------------------------------

def test_resolve_micro_conflicts_action_drop_accepted():
    """action=drop with empty text should resolve the conflict (intentional deletion)."""
    llm = DummyLLM()
    payload = {
        "segment_id": "seg_001",
        "block_type": "paragraph",
        "micro_conflicts": [
            {
                "conflict_id": "seg_001_mc_0",
                "context_before": "before",
                "disagreement": {"grobid": "noise", "docling": "", "marker": ""},
                "context_after": "after",
            },
        ],
    }

    message = MagicMock()
    message.refusal = None
    message.parsed = MicroConflictResolutionResponse(
        resolved=[ResolvedMicroConflict(conflict_id="seg_001_mc_0", text="", action="drop")]
    )
    completion = MagicMock()
    completion.choices = [MagicMock(message=message)]
    completion.usage = None
    llm.client.chat.completions.parse.return_value = completion

    result = llm.resolve_micro_conflicts(payload)
    resolved = {item.conflict_id: item.text for item in result.resolved}
    assert "seg_001_mc_0" in resolved
    assert resolved["seg_001_mc_0"] == ""


# ---------------------------------------------------------------------------
# Partial resolution returns on retry exhaustion (O-3)
# ---------------------------------------------------------------------------

def test_resolve_micro_conflicts_partial_on_exhaustion():
    """When retries exhaust, return partial results instead of raising."""
    llm = DummyLLM()
    payload = {
        "segment_id": "seg_001",
        "block_type": "paragraph",
        "micro_conflicts": [
            {
                "conflict_id": "seg_001_mc_0",
                "context_before": "",
                "disagreement": {"grobid": "a", "docling": "b", "marker": "c"},
                "context_after": "",
            },
            {
                "conflict_id": "seg_001_mc_1",
                "context_before": "",
                "disagreement": {"grobid": "x", "docling": "y", "marker": "z"},
                "context_after": "",
            },
        ],
    }

    # Every call only resolves mc_0, never mc_1
    msg = MagicMock()
    msg.refusal = None
    msg.parsed = MicroConflictResolutionResponse(
        resolved=[ResolvedMicroConflict(conflict_id="seg_001_mc_0", text="resolved_a")]
    )
    comp = MagicMock()
    comp.choices = [MagicMock(message=msg)]
    comp.usage = None
    llm.client.chat.completions.parse.return_value = comp

    # Should NOT raise — returns partial
    result = llm.resolve_micro_conflicts(payload)
    resolved = {item.conflict_id: item.text for item in result.resolved}
    assert "seg_001_mc_0" in resolved
    assert resolved["seg_001_mc_0"] == "resolved_a"
    # mc_1 is unresolved but no exception raised
