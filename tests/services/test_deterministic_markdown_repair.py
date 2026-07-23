import hashlib

import pytest

from app.services.deterministic_markdown_repair import normalize_trailing_newline


def _audit(raw: bytes) -> list[dict]:
    return [{
        "output_byte_start": 0,
        "output_byte_end": len(raw),
        "source": "grobid",
        "artifact_digest": hashlib.sha256(raw).hexdigest(),
        "source_byte_start": 0,
        "source_byte_end": len(raw),
    }]


def test_exactly_one_terminal_lf_is_unchanged():
    text = "body\n"
    audit = _audit(text.encode())
    assert normalize_trailing_newline(text, audit) == (text, audit, False)


@pytest.mark.parametrize("suffix", ["", "\r", "\n\n", "\r\n\r\n"])
def test_terminal_newlines_are_normalized_with_a_narrow_audit_entry(suffix):
    text = "body" + suffix
    normalized, audit, changed = normalize_trailing_newline(text, _audit(text.encode()))
    assert normalized == "body\n"
    assert changed is True
    assert audit[-1]["source"] == "deterministic_markup"
    assert audit[-1]["transformation"] == "trailing_newline_normalization"
    assert audit[-1]["output_byte_start"] == 4
    assert audit[-1]["output_byte_end"] == 5


def test_normalization_rejects_an_unclippable_audit_range():
    with pytest.raises(ValueError, match="cannot clip"):
        normalize_trailing_newline(
            "body\n\n",
            [{
                "output_byte_start": 0,
                "output_byte_end": 6,
                "source_byte_start": 5,
                "source_byte_end": 6,
            }],
        )
