import pytest
from unittest.mock import MagicMock
from app.services.llm_service import (
    ConflictResolutionResponse,
    ResolvedSegment,
    LLM,
)


class DummyLLM(LLM):
    def __init__(self):
        self.client = MagicMock()
        self.model = "dummy-model"
        self.reasoning_effort = "medium"

    def create_prompt(self, g, d, m):
        return "prompt"


def test_llm_extract_success():
    llm = DummyLLM()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "merged output"
    llm.client.chat.completions.create.return_value = mock_response

    result = llm.extract("grobid", "docling", "marker")
    assert result == "merged output"


def test_llm_extract_error():
    llm = DummyLLM()
    llm.client.chat.completions.create.side_effect = Exception("fail")
    with pytest.raises(Exception) as excinfo:
        llm.extract("grobid", "docling", "marker")
    assert "Error in LLM processing" in str(excinfo.value)


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
            ResolvedSegment(segment_id="seg_001", text="resolved text a"),
            ResolvedSegment(segment_id="seg_002", text="resolved text b"),
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
    message.parsed = ConflictResolutionResponse(resolved=[ResolvedSegment(segment_id="seg_001", text="fixed text")])
    message.refusal = None
    good_completion = MagicMock()
    good_completion.choices = [MagicMock(message=message)]

    llm.client.chat.completions.parse.side_effect = [Exception("boom"), good_completion]

    result = llm.resolve_conflicts(conflicts)
    assert result == {"seg_001": "fixed text"}
    assert llm.client.chat.completions.parse.call_count == 2


def test_resolve_conflicts_both_attempts_fail():
    """Both attempts fail — should raise."""
    llm = DummyLLM()
    conflicts = [
        {"segment_id": "seg_001", "block_type": "paragraph",
         "context_before": "", "context_after": "",
         "grobid": "a", "docling": "b", "marker": "c"},
    ]

    llm.client.chat.completions.parse.side_effect = Exception("down")

    with pytest.raises(Exception, match="resolve_conflicts failed after 2 attempts"):
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
    partial_message.parsed = ConflictResolutionResponse(resolved=[ResolvedSegment(segment_id="seg_001", text="text")])
    partial_message.refusal = None
    partial_completion = MagicMock()
    partial_completion.choices = [MagicMock(message=partial_message)]

    # Second: complete
    complete_message = MagicMock()
    complete_message.parsed = ConflictResolutionResponse(
        resolved=[
            ResolvedSegment(segment_id="seg_001", text="text"),
            ResolvedSegment(segment_id="seg_002", text="text2"),
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
    good_message.parsed = ConflictResolutionResponse(resolved=[ResolvedSegment(segment_id="seg_001", text="ok")])
    good_message.refusal = None
    good_completion = MagicMock()
    good_completion.choices = [MagicMock(message=good_message)]

    llm.client.chat.completions.parse.side_effect = [refusal_completion, good_completion]
    result = llm.resolve_conflicts(conflicts)
    assert result == {"seg_001": "ok"}
