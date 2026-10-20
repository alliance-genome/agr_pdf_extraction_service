"""Immutable, content-addressed source/checkpoint objects for staged jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import boto3
from config import Config


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_path(value):
    p = PurePosixPath(value)
    if (
        not value
        or p.is_absolute()
        or ".." in p.parts
        or "\\" in value
        or str(p) != value
    ):
        raise ValueError("Unsafe checkpoint path")
    return p


class PipelineStore:
    def __init__(self, client=None, bucket=None):
        self.bucket = bucket or Config.PIPELINE_S3_BUCKET or Config.AUDIT_S3_BUCKET
        if not self.bucket:
            from app.services.audit_logger import _resolve_bucket_name

            self.bucket = _resolve_bucket_name(Config)
        if not self.bucket:
            raise RuntimeError(
                "Pipeline object storage is required for durable acceptance"
            )
        self.client = client or boto3.client(
            "s3", region_name=Config.AWS_DEFAULT_REGION
        )
        self.prefix = Config.PIPELINE_S3_PREFIX.strip("/")

    def put_file(self, path, prefix):
        path = Path(path)
        sha = digest(path)
        key = f"{self.prefix}/{prefix}/{sha}"
        self.client.upload_file(
            str(path), self.bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"}
        )
        return {"key": key, "sha256": sha, "size": path.stat().st_size}

    def fetch_file(self, receipt, destination):
        if (
            not isinstance(receipt.get("size"), int)
            or receipt["size"] < 0
            or receipt["size"] > Config.PIPELINE_CHECKPOINT_MAX_BYTES
            or not str(receipt.get("key", "")).startswith(self.prefix + "/")
            or receipt.get("sha256") != str(receipt.get("key", "")).rsplit("/", 1)[-1]
        ):
            raise ValueError("Invalid pipeline object receipt")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Stream with a hard bound rather than trusting the manifest or S3 metadata.
        body = self.client.get_object(Bucket=self.bucket, Key=receipt["key"])["Body"]
        size = 0
        try:
            with target.open("xb") as out:
                while chunk := body.read(1024 * 1024):
                    size += len(chunk)
                    if size > receipt["size"]:
                        raise ValueError("Pipeline object exceeds declared size")
                    out.write(chunk)
        finally:
            body.close()
        if size != receipt["size"] or digest(target) != receipt["sha256"]:
            raise ValueError("Pipeline object checksum mismatch")
        return target

    def checkpoint(self, workspace, process_id, generation):
        root = Path(workspace)
        files = []
        total = 0
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("Checkpoint cannot contain symlinks")
            if not path.is_file():
                continue
            total += path.stat().st_size
            if (
                total > Config.PIPELINE_CHECKPOINT_MAX_BYTES
                or len(files) >= Config.PIPELINE_CHECKPOINT_MAX_FILES
            ):
                raise ValueError("Checkpoint exceeds configured bounds")
            name = path.relative_to(root).as_posix()
            relative_path(name)
            files.append(
                {
                    "path": name,
                    **self.put_file(path, f"{process_id}/{generation}/files"),
                }
            )
        if not {"source.pdf", "sources.json"} <= {f["path"] for f in files}:
            raise ValueError("Incomplete extraction checkpoint")
        manifest = {
            "version": 1,
            "config_version": Config.EXTRACTION_CONFIG_VERSION,
            "merge_contract_id": Config.MERGE_CONTRACT_ID,
            "files": files,
        }
        manifest_path = root.parent / f"{generation}-manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        if manifest_path.stat().st_size > Config.PIPELINE_MANIFEST_MAX_BYTES:
            raise ValueError("Checkpoint manifest exceeds configured bound")
        return self.put_file(manifest_path, f"{process_id}/{generation}/manifest")

    def restore(self, receipt, workspace):
        root = Path(workspace)
        root.mkdir(parents=True, exist_ok=False)
        if receipt["size"] > Config.PIPELINE_MANIFEST_MAX_BYTES:
            raise ValueError("Checkpoint manifest exceeds configured bound")
        manifest_file = self.fetch_file(receipt, root / "checkpoint.json")
        manifest = json.loads(manifest_file.read_text())
        if (
            manifest.get("version") != 1
            or manifest.get("config_version") != Config.EXTRACTION_CONFIG_VERSION
            or manifest.get("merge_contract_id") != Config.MERGE_CONTRACT_ID
        ):
            raise ValueError("Unsupported checkpoint contract")
        files = manifest["files"]
        if (
            len(files) > Config.PIPELINE_CHECKPOINT_MAX_FILES
            or sum(f["size"] for f in files) > Config.PIPELINE_CHECKPOINT_MAX_BYTES
        ):
            raise ValueError("Checkpoint exceeds configured bounds")
        names = [str(relative_path(f["path"])) for f in files]
        if len(set(names)) != len(names) or not {"source.pdf", "sources.json"} <= set(
            names
        ):
            raise ValueError("Incomplete or duplicate checkpoint files")
        for name, item in zip(names, files):
            self.fetch_file(item, root / name)
        return json.loads((root / "sources.json").read_text())
