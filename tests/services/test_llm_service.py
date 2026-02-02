import pytest
from unittest.mock import MagicMock, patch
from app.services.llm_service import LLM

class DummyLLM(LLM):
    def __init__(self):
        self.client = MagicMock()
        self.model = "dummy-model"
    def create_prompt(self, g, d, m):
        return "prompt"

def test_llm_extract_success(monkeypatch):
    llm = DummyLLM()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="merged output")]
    llm.client.messages.create.return_value = mock_message

    result = llm.extract("grobid", "docling", "marker")
    assert result == "merged output"

def test_llm_extract_error(monkeypatch):
    llm = DummyLLM()
    llm.client.messages.create.side_effect = Exception("fail")
    with pytest.raises(Exception) as excinfo:
        llm.extract("grobid", "docling", "marker")
    assert "Error in LLM processing" in str(excinfo.value)