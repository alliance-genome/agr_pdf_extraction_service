import json
import pytest
from unittest.mock import MagicMock
from app.services.llm_service import LLM


class DummyLLM(LLM):
    def __init__(self):
        self.client = MagicMock()
        self.model = "dummy-model"
        self.max_tokens = 16000

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
         "grobid": "text a", "docling": "text b", "marker": "text c"},
        {"segment_id": "seg_002", "block_type": "paragraph",
         "grobid": "x", "docling": "y", "marker": "z"},
    ]

    resolved_json = json.dumps({
        "resolved": {
            "seg_001": "resolved text a",
            "seg_002": "resolved text b",
        }
    })
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = resolved_json
    llm.client.chat.completions.create.return_value = mock_response

    result = llm.resolve_conflicts(conflicts)
    assert result == {"seg_001": "resolved text a", "seg_002": "resolved text b"}


def test_resolve_conflicts_retry_on_bad_json():
    """First call returns invalid JSON, second succeeds."""
    llm = DummyLLM()
    conflicts = [
        {"segment_id": "seg_001", "block_type": "paragraph",
         "grobid": "a", "docling": "b", "marker": "c"},
    ]

    # First call: bad JSON
    bad_response = MagicMock()
    bad_response.choices = [MagicMock()]
    bad_response.choices[0].message.content = "not valid json"

    # Second call: good JSON
    good_response = MagicMock()
    good_response.choices = [MagicMock()]
    good_response.choices[0].message.content = json.dumps({
        "resolved": {"seg_001": "fixed text"}
    })

    llm.client.chat.completions.create.side_effect = [bad_response, good_response]

    result = llm.resolve_conflicts(conflicts)
    assert result == {"seg_001": "fixed text"}
    assert llm.client.chat.completions.create.call_count == 2


def test_resolve_conflicts_both_attempts_fail():
    """Both attempts fail — should raise."""
    llm = DummyLLM()
    conflicts = [
        {"segment_id": "seg_001", "block_type": "paragraph",
         "grobid": "a", "docling": "b", "marker": "c"},
    ]

    bad_response = MagicMock()
    bad_response.choices = [MagicMock()]
    bad_response.choices[0].message.content = "not json"
    llm.client.chat.completions.create.return_value = bad_response

    with pytest.raises(Exception, match="resolve_conflicts failed after 2 attempts"):
        llm.resolve_conflicts(conflicts)


def test_resolve_conflicts_missing_segment_id():
    """Response missing a requested segment_id — should retry."""
    llm = DummyLLM()
    conflicts = [
        {"segment_id": "seg_001", "block_type": "paragraph",
         "grobid": "a", "docling": "b", "marker": "c"},
        {"segment_id": "seg_002", "block_type": "paragraph",
         "grobid": "x", "docling": "y", "marker": "z"},
    ]

    # First: missing seg_002
    partial_response = MagicMock()
    partial_response.choices = [MagicMock()]
    partial_response.choices[0].message.content = json.dumps({
        "resolved": {"seg_001": "text"}
    })

    # Second: complete
    complete_response = MagicMock()
    complete_response.choices = [MagicMock()]
    complete_response.choices[0].message.content = json.dumps({
        "resolved": {"seg_001": "text", "seg_002": "text2"}
    })

    llm.client.chat.completions.create.side_effect = [partial_response, complete_response]

    result = llm.resolve_conflicts(conflicts)
    assert "seg_001" in result and "seg_002" in result
