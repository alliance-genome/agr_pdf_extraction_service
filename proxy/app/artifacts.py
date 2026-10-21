"""Authenticated routes use durable receipts only; this module cannot wake EC2."""
from __future__ import annotations

import uuid
import threading
from datetime import datetime, timedelta, timezone

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from pdfx_image_metadata import normalize_image_manifest_entry, copy_image_metadata

METHODS = {"grobid", "docling", "marker", "merged", "audit", "page_provenance"}


_client_lock = threading.Lock()
_s3_client = None
_s3_client_config = None


class ArtifactStreamResponse(StreamingResponse):
    """Always release the S3 connection, including disconnect before iteration."""
    def __init__(self, body, *args, **kwargs):
        self._s3_body = body
        super().__init__(*args, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._s3_body.close()


class ArtifactReader:
    def __init__(self, status_reader):
        self.status_reader = status_reader

    def lookup(self, process_id, *, completed=True):
        try:
            uuid.UUID(process_id)
        except (ValueError, TypeError):
            raise HTTPException(400, "Invalid process_id")
        available, run = self.status_reader.lookup(process_id)
        if not available:
            raise HTTPException(503, "Durable artifact metadata unavailable")
        if run is None:
            raise HTTPException(404, "process_id not found")
        if completed and run.get("status") not in {"complete", "succeeded"}:
            raise HTTPException(409, "Job not complete yet")
        return run

    def client(self):
        if not settings.ARTIFACT_S3_BUCKET or not settings.ARTIFACT_S3_PREFIX:
            raise HTTPException(503, "Durable artifact storage unconfigured")
        global _s3_client, _s3_client_config
        config_key = (settings.EC2_REGION, max(1, settings.ARTIFACT_S3_TIMEOUT_SECONDS),
                      max(1, settings.ARTIFACT_S3_MAX_ATTEMPTS))
        # Sessions are not thread-safe. Construct a dedicated session/client once
        # under the lock; subsequent requests share the thread-safe client/pool.
        with _client_lock:
            if _s3_client is None or _s3_client_config != config_key:
                _s3_client = boto3.Session().client("s3", region_name=config_key[0], config=Config(
                    connect_timeout=config_key[1], read_timeout=config_key[1],
                    retries={"total_max_attempts": config_key[2], "mode": "standard"},
                ))
                _s3_client_config = config_key
            return _s3_client

    @staticmethod
    def key(key):
        # Accept only persisted keys within the configured artifact namespace.
        prefix = settings.ARTIFACT_S3_PREFIX + "/"
        if not isinstance(key, str) or not key.startswith(prefix) or any(
            part in {"", ".", ".."} for part in key.split("/")
        ):
            raise HTTPException(503, "Invalid durable artifact receipt")
        return key

    def stream(self, key):
        key = self.key(key)
        try:
            obj = self.client().get_object(Bucket=settings.ARTIFACT_S3_BUCKET, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                raise HTTPException(404, "Stored artifact not found") from exc
            raise HTTPException(503, "Durable artifact storage unavailable") from exc
        except Exception as exc:
            raise HTTPException(503, "Durable artifact storage unavailable") from exc
        body = obj["Body"]

        def chunks():
            try:
                while data := body.read(max(1, settings.ARTIFACT_STREAM_CHUNK_BYTES)):
                    yield data
            finally:
                body.close()

        headers = {"content-length": str(obj["ContentLength"])}
        if obj.get("ContentDisposition"):
            headers["content-disposition"] = obj["ContentDisposition"]
        return ArtifactStreamResponse(body, chunks(), media_type=obj.get("ContentType", "application/octet-stream"),
                                 headers=headers)

    def download(self, run, method):
        if method not in METHODS:
            raise HTTPException(400, "Invalid method")
        key = (run.get("artifacts_json") or {}).get(method)
        if key:
            return self.stream(key)
        if method in {"merged", "audit", "page_provenance"} and run.get("consensus_metrics_json") is not None:
            raise HTTPException(500, "Completed merge artifact is internally unavailable")
        raise HTTPException(404, "Output file not found for method")

    @staticmethod
    def images(run):
        return [image for entry in (run.get("artifacts_json") or {}).get("images", [])
                if (image := normalize_image_manifest_entry(entry))]

    def image(self, run, filename):
        for image in self.images(run):
            if image["filename"] == filename and image.get("s3_key"):
                return self.stream(image["s3_key"])
        raise HTTPException(404, "Image not found")

    def presign(self, key, ttl):
        key = self.key(key)
        try:
            return self.client().generate_presigned_url("get_object", Params={
                "Bucket": settings.ARTIFACT_S3_BUCKET, "Key": key,
            }, ExpiresIn=ttl)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(503, "Unable to generate artifact URL") from exc

    def image_urls(self, run, process_id):
        ttl = max(1, settings.IMAGE_URL_TTL_SECONDS)
        expires = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat().replace("+00:00", "Z")
        images = []
        for image in self.images(run):
            if not image.get("s3_key"):
                continue
            entry = {"filename": image["filename"], "s3_key": image["s3_key"],
                     "size_bytes": image.get("size_bytes"),
                     "url": self.presign(image["s3_key"], ttl), "expires_at": expires}
            images.append(copy_image_metadata(image, entry))
        return {"process_id": process_id, "images": images, "manifest_ttl_seconds": ttl,
                "image_retention_ttl_seconds": settings.IMAGE_RETENTION_TTL_SECONDS}

    def artifact_urls(self, run, process_id):
        def keys(value, path="$"):
            if isinstance(value, str) and "/" in value:
                yield path, value
            elif isinstance(value, dict):
                for name, child in value.items():
                    yield from keys(child, f"{path}.{name}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    yield from keys(child, f"{path}[{index}]")
        # Only actual artifact receipts, never image captions or scientific text.
        artifacts = run.get("artifacts_json") or {}
        receipts = {name: key for name, key in artifacts.items() if isinstance(key, str)}
        receipts["native"] = {
            method: {name: receipt[name] for name in ("artifact", "manifest", "page_provenance", "style") if receipt.get(name)}
            for method, receipt in (artifacts.get("native") or {}).items()
            if isinstance(receipt, dict)
        }
        receipts["images"] = [{"s3_key": image["s3_key"]} for image in self.images(run) if image.get("s3_key")]
        entries = list(keys(receipts))
        if not entries:
            raise HTTPException(404, "No artifacts available for this process")
        ttl = max(1, settings.ARTIFACT_URL_TTL_SECONDS)
        return {"process_id": process_id, "expires_in": ttl, "artifact_urls": [
            {"name": name, "s3_key": key, "url": self.presign(key, ttl)} for name, key in entries]}
