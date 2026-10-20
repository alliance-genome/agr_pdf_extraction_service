import io
import json
from pathlib import Path
import pytest
from config import Config
from app.pipeline_storage import PipelineStore


class S3:
    def __init__(self):
        self.objects = {}

    def upload_file(self, path, bucket, key, **kwargs):
        self.objects[key] = Path(path).read_bytes()

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}


def test_full_checkpoint_roundtrip_and_corruption(tmp_path):
    client = S3()
    store = PipelineStore(client, "test-bucket")
    root = tmp_path / "extract"
    root.mkdir()
    (root / "source.pdf").write_bytes(b"%PDF-test")
    (root / "sources.json").write_text(
        json.dumps({"file_hash": "123", "extractions": {"marker": "Full text"}})
    )
    (root / "cache").mkdir()
    (root / "cache" / "native.json").write_text('{"structure":"intact"}')
    (root / "cache" / "image.png").write_bytes(b"pixels")
    receipt = store.checkpoint(root, "job", 1)
    state = store.restore(receipt, tmp_path / "merge")
    assert state["extractions"]["marker"] == "Full text"
    assert (tmp_path / "merge" / "cache" / "native.json").read_bytes() == (
        root / "cache" / "native.json"
    ).read_bytes()
    manifest = json.loads(client.objects[receipt["key"]])
    client.objects[manifest["files"][0]["key"]] = b"corruption"
    with pytest.raises(ValueError):
        store.restore(receipt, tmp_path / "corrupt")


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "cache/../../escape", "a\\b", "a//b"]
)
def test_rejects_unsafe_manifest_path(name):
    from app.pipeline_storage import relative_path

    with pytest.raises(ValueError):
        relative_path(name)


def test_checkpoint_bounds_and_symlinks(tmp_path, monkeypatch):
    root = tmp_path / "extract"
    root.mkdir()
    (root / "source.pdf").write_bytes(b"pdf")
    (root / "sources.json").write_text("{}")
    store = PipelineStore(S3(), "test-bucket")
    monkeypatch.setattr(Config, "PIPELINE_CHECKPOINT_MAX_BYTES", 1)
    with pytest.raises(ValueError, match="bounds"):
        store.checkpoint(root, "job", 1)
    monkeypatch.setattr(Config, "PIPELINE_CHECKPOINT_MAX_BYTES", 1000)
    (root / "link").symlink_to(root / "source.pdf")
    with pytest.raises(ValueError, match="symlink"):
        store.checkpoint(root, "job", 1)
