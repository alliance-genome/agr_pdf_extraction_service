import hashlib

import pytest

from app.services.merge_cache_policy import verify_merged_output_digest


def test_merged_download_bytes_must_match_recorded_digest():
    text = "complete Gγ1 output"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    assert verify_merged_output_digest(text, {"output_digest": digest}) == digest

    with pytest.raises(ValueError, match="does not match"):
        verify_merged_output_digest(
            text + " changed",
            {"output_digest": digest},
        )


def test_merged_download_rejects_missing_digest_metrics():
    with pytest.raises(ValueError, match="missing its output digest"):
        verify_merged_output_digest("output", None)
