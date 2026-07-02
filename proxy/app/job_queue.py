"""Queue implementations for PDF extraction startup buffering."""

from __future__ import annotations

import base64
import asyncio
import contextlib
import json
import logging
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

import boto3
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    """Queue cannot accept new jobs."""


class QueuePayloadMissingError(Exception):
    """Queued metadata points to an S3 payload that no longer exists."""


@dataclass
class QueuedJob:
    """Serialized queue payload for replay."""

    job_id: str
    pdf_data: bytes | None
    form_fields: dict[str, Any]
    filename: str = "upload.pdf"
    authorization: str | None = None
    queued_at: float = field(default_factory=time.time)
    pdf_s3_bucket: str | None = None
    pdf_s3_key: str | None = None
    pdf_file_path: str | None = None

    def to_json(self) -> str:
        payload = {
            "job_id": self.job_id,
            "filename": self.filename,
            "authorization": self.authorization,
            "form_fields": self.form_fields,
            "queued_at": self.queued_at,
        }
        if self.pdf_s3_key:
            payload["pdf_s3_bucket"] = self.pdf_s3_bucket
            payload["pdf_s3_key"] = self.pdf_s3_key
        elif self.pdf_data is not None:
            # Backward-compatible memory queue / legacy S3 payload shape.
            payload["pdf_data_b64"] = base64.b64encode(self.pdf_data).decode("ascii")
        else:
            raise ValueError("QueuedJob requires either pdf_data or pdf_s3_key")
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "QueuedJob":
        payload = json.loads(raw)
        pdf_data = None
        if "pdf_data_b64" in payload:
            pdf_data = base64.b64decode(payload["pdf_data_b64"])
        return cls(
            job_id=str(payload["job_id"]),
            filename=str(payload.get("filename") or "upload.pdf"),
            authorization=payload.get("authorization"),
            form_fields=dict(payload.get("form_fields") or {}),
            queued_at=float(payload.get("queued_at") or time.time()),
            pdf_data=pdf_data,
            pdf_s3_bucket=payload.get("pdf_s3_bucket"),
            pdf_s3_key=payload.get("pdf_s3_key"),
        )

    @contextlib.contextmanager
    def open_pdf(self) -> Iterator[bytes | Any]:
        if self.pdf_file_path:
            with open(self.pdf_file_path, "rb") as pdf_file:
                yield pdf_file
            return
        if self.pdf_data is not None:
            yield self.pdf_data
            return
        if self.pdf_s3_key:
            self._download_pdf_from_s3()
            with open(self.pdf_file_path, "rb") as pdf_file:
                yield pdf_file
            return
        raise ValueError("QueuedJob PDF payload is not materialized")

    def _download_pdf_from_s3(self) -> None:
        if not self.pdf_s3_key:
            raise ValueError("QueuedJob has no S3 PDF payload")
        bucket = self.pdf_s3_bucket
        if not bucket:
            raise ValueError("QueuedJob S3 payload is missing a bucket")

        tmp = tempfile.NamedTemporaryFile(prefix=f"pdfx-{self.job_id}-", suffix=".pdf", delete=False)
        try:
            with tmp:
                boto3.client("s3").download_fileobj(bucket, self.pdf_s3_key, tmp)
        except ClientError as exc:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp.name)
            if _is_missing_s3_object_error(exc):
                raise QueuePayloadMissingError(
                    f"Queued job {self.job_id} is missing payload s3://{bucket}/{self.pdf_s3_key}",
                ) from exc
            raise
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp.name)
            raise
        self.pdf_file_path = tmp.name

    def cleanup(self, *, delete_remote: bool = False) -> None:
        if self.pdf_file_path:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.pdf_file_path)
            self.pdf_file_path = None
        if delete_remote and self.pdf_s3_key:
            bucket = self.pdf_s3_bucket
            if bucket:
                boto3.client("s3").delete_object(Bucket=bucket, Key=self.pdf_s3_key)
            self.pdf_s3_key = None
            self.pdf_s3_bucket = None


class BaseJobQueue:
    """Queue contract used by proxy routes."""

    @property
    def size(self) -> int:
        raise NotImplementedError

    def enqueue(
        self,
        job_id: str,
        pdf_data: bytes,
        form_fields: dict,
        filename: str = "upload.pdf",
        authorization: str | None = None,
    ) -> None:
        raise NotImplementedError

    async def enqueue_upload(
        self,
        job_id: str,
        upload_file: Any,
        form_fields: dict,
        filename: str = "upload.pdf",
        authorization: str | None = None,
    ) -> QueuedJob:
        await upload_file.seek(0)
        pdf_data = await upload_file.read()
        self.enqueue(job_id, pdf_data, form_fields, filename, authorization=authorization)
        return QueuedJob(
            job_id=job_id,
            pdf_data=pdf_data,
            form_fields=form_fields,
            filename=filename,
            authorization=authorization,
        )

    def dequeue(self) -> QueuedJob:
        raise NotImplementedError

    def drain(self) -> list[QueuedJob]:
        raise NotImplementedError

    def has_job(self, job_id: str) -> bool:
        raise NotImplementedError

    def remove_job(self, job_id: str) -> bool:
        raise NotImplementedError

    def acknowledge(self, job_id: str) -> bool:
        """Mark one queued job as handed off while keeping its payload available."""
        return False

    def oldest_age_seconds(self) -> float:
        raise NotImplementedError

    @property
    def durable(self) -> bool:
        return False


class InMemoryJobQueue(BaseJobQueue):
    """FIFO queue for extraction jobs waiting on EC2."""

    def __init__(self, max_size: int = 10):
        self._max_size = max_size
        self._queue: deque[QueuedJob] = deque()

    @property
    def size(self) -> int:
        return len(self._queue)

    def enqueue(
        self,
        job_id: str,
        pdf_data: bytes,
        form_fields: dict,
        filename: str = "upload.pdf",
        authorization: str | None = None,
    ) -> None:
        if len(self._queue) >= self._max_size:
            raise QueueFullError(f"Queue full ({self._max_size} jobs max)")
        self._queue.append(self._build_job(job_id, pdf_data, form_fields, filename, authorization))

    def _build_job(
        self,
        job_id: str,
        pdf_data: bytes,
        form_fields: dict,
        filename: str = "upload.pdf",
        authorization: str | None = None,
    ) -> QueuedJob:
        return QueuedJob(
            job_id=job_id,
            pdf_data=pdf_data,
            form_fields=form_fields,
            filename=filename,
            authorization=authorization,
        )

    async def enqueue_upload(
        self,
        job_id: str,
        upload_file: Any,
        form_fields: dict,
        filename: str = "upload.pdf",
        authorization: str | None = None,
    ) -> QueuedJob:
        await upload_file.seek(0)
        pdf_data = await upload_file.read()
        self.enqueue(job_id, pdf_data, form_fields, filename, authorization=authorization)
        return self._build_job(job_id, pdf_data, form_fields, filename, authorization)

    def dequeue(self) -> QueuedJob:
        return self._queue.popleft()

    def drain(self) -> list[QueuedJob]:
        jobs = list(self._queue)
        self._queue.clear()
        return jobs

    def has_job(self, job_id: str) -> bool:
        return any(j.job_id == job_id for j in self._queue)

    def remove_job(self, job_id: str) -> bool:
        for idx, queued_job in enumerate(self._queue):
            if queued_job.job_id == job_id:
                del self._queue[idx]
                return True
        return False

    def acknowledge(self, job_id: str) -> bool:
        return False

    def oldest_age_seconds(self) -> float:
        if not self._queue:
            return 0.0
        return max(0.0, time.time() - self._queue[0].queued_at)


class S3JobQueue(BaseJobQueue):
    """Durable queue backed by S3 objects."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "pdfx-proxy-queue",
        max_size: int = 10,
        region_name: str | None = None,
    ):
        if not bucket:
            raise ValueError("QUEUE_S3_BUCKET is required when QUEUE_BACKEND=s3")
        self._bucket = bucket
        self._prefix = prefix.strip().strip("/")
        self._max_size = max_size
        kwargs = {"region_name": region_name} if region_name else {}
        self._client = boto3.client("s3", **kwargs)

    @property
    def durable(self) -> bool:
        return True

    def _queue_prefix(self) -> str:
        return f"{self._prefix}/jobs/"

    def _payload_prefix(self) -> str:
        return f"{self._prefix}/payloads/"

    def _build_key(self, job: QueuedJob) -> str:
        ts_ms = int(job.queued_at * 1000)
        return f"{self._queue_prefix()}{ts_ms:013d}_{job.job_id}.json"

    def _build_payload_key(self, job_id: str) -> str:
        return f"{self._payload_prefix()}{job_id}.pdf"

    def _iter_keys(self) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._queue_prefix()):
            for item in page.get("Contents", []):
                key = item.get("Key")
                if key:
                    keys.append(key)
        keys.sort()
        return keys

    @property
    def size(self) -> int:
        return len(self._iter_keys())

    def enqueue(
        self,
        job_id: str,
        pdf_data: bytes,
        form_fields: dict,
        filename: str = "upload.pdf",
        authorization: str | None = None,
    ) -> None:
        if self.size >= self._max_size:
            raise QueueFullError(f"Queue full ({self._max_size} jobs max)")
        payload_key = self._build_payload_key(job_id)
        self._client.put_object(
            Bucket=self._bucket,
            Key=payload_key,
            Body=pdf_data,
            ContentType="application/pdf",
            ServerSideEncryption="AES256",
        )
        job = QueuedJob(
            job_id=job_id,
            pdf_data=None,
            form_fields=form_fields,
            filename=filename,
            authorization=authorization,
            pdf_s3_bucket=self._bucket,
            pdf_s3_key=payload_key,
        )
        try:
            self._put_job_metadata(job)
        except Exception:
            self._client.delete_object(Bucket=self._bucket, Key=payload_key)
            raise

    async def enqueue_upload(
        self,
        job_id: str,
        upload_file: Any,
        form_fields: dict,
        filename: str = "upload.pdf",
        authorization: str | None = None,
    ) -> QueuedJob:
        if self.size >= self._max_size:
            raise QueueFullError(f"Queue full ({self._max_size} jobs max)")

        payload_key = self._build_payload_key(job_id)
        await upload_file.seek(0)
        await asyncio.to_thread(
            self._client.upload_fileobj,
            upload_file.file,
            self._bucket,
            payload_key,
            ExtraArgs={
                "ContentType": "application/pdf",
                "ServerSideEncryption": "AES256",
            },
        )
        job = QueuedJob(
            job_id=job_id,
            pdf_data=None,
            form_fields=form_fields,
            filename=filename,
            authorization=authorization,
            pdf_s3_bucket=self._bucket,
            pdf_s3_key=payload_key,
        )
        try:
            self._put_job_metadata(job)
        except Exception:
            self._client.delete_object(Bucket=self._bucket, Key=payload_key)
            raise
        return job

    def _put_job_metadata(self, job: QueuedJob) -> None:
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._build_key(job),
            Body=job.to_json().encode("utf-8"),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )

    def dequeue(self) -> QueuedJob:
        keys = self._iter_keys()
        for key in keys:
            try:
                job = self._load_job(key)
            except QueuePayloadMissingError as exc:
                logger.warning("%s; deleting orphaned queue metadata %s", exc, key)
                self._delete_job_metadata(key)
                continue
            self._delete_job_metadata(key)
            return job
        raise IndexError("dequeue from empty queue")

    def drain(self) -> list[QueuedJob]:
        keys = self._iter_keys()
        jobs: list[QueuedJob] = []
        if not keys:
            return jobs

        for key in keys:
            try:
                jobs.append(self._load_job(key))
            except QueuePayloadMissingError as exc:
                logger.warning("%s; deleting orphaned queue metadata %s", exc, key)
                self._delete_job_metadata(key)

        return jobs

    def _load_job(self, key: str) -> QueuedJob:
        obj = self._client.get_object(Bucket=self._bucket, Key=key)
        raw = obj["Body"].read().decode("utf-8")
        job = QueuedJob.from_json(raw)
        if job.pdf_s3_key:
            bucket = job.pdf_s3_bucket or self._bucket
            tmp = tempfile.NamedTemporaryFile(prefix=f"pdfx-{job.job_id}-", suffix=".pdf", delete=False)
            try:
                with tmp:
                    self._client.download_fileobj(bucket, job.pdf_s3_key, tmp)
            except ClientError as exc:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp.name)
                if _is_missing_s3_object_error(exc):
                    raise QueuePayloadMissingError(
                        f"Queued job {job.job_id} is missing payload s3://{bucket}/{job.pdf_s3_key}",
                    ) from exc
                raise
            except Exception:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp.name)
                raise
            job.pdf_file_path = tmp.name
        return job

    def _delete_s3_keys(self, keys: list[str]) -> None:
        for i in range(0, len(keys), 1000):
            chunk = keys[i : i + 1000]
            if not chunk:
                continue
            self._client.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": [{"Key": key} for key in chunk], "Quiet": True},
            )

    def _delete_job_metadata(self, metadata_key: str) -> None:
        self._delete_s3_keys([metadata_key])

    def _delete_job_objects(self, metadata_key: str, job: QueuedJob) -> None:
        keys = [metadata_key]
        if job.pdf_s3_key:
            keys.append(job.pdf_s3_key)
        self._delete_s3_keys(keys)

    def has_job(self, job_id: str) -> bool:
        suffix = f"_{job_id}.json"
        return any(key.endswith(suffix) for key in self._iter_keys())

    def remove_job(self, job_id: str) -> bool:
        suffix = f"_{job_id}.json"
        for key in self._iter_keys():
            if key.endswith(suffix):
                obj = self._client.get_object(Bucket=self._bucket, Key=key)
                raw = obj["Body"].read().decode("utf-8")
                job = QueuedJob.from_json(raw)
                self._delete_job_objects(key, job)
                return True
        return False

    def acknowledge(self, job_id: str) -> bool:
        suffix = f"_{job_id}.json"
        for key in self._iter_keys():
            if key.endswith(suffix):
                self._delete_job_metadata(key)
                return True
        return False

    def oldest_age_seconds(self) -> float:
        keys = self._iter_keys()
        if not keys:
            return 0.0
        key = keys[0].rsplit("/", 1)[-1]
        try:
            ts_raw = key.split("_", 1)[0]
            queued_at = int(ts_raw) / 1000.0
        except (ValueError, IndexError):
            return 0.0
        return max(0.0, time.time() - queued_at)


def build_job_queue(max_size: int = 10) -> BaseJobQueue:
    """Create configured queue backend."""
    backend = settings.QUEUE_BACKEND
    if backend == "s3":
        if not settings.QUEUE_S3_BUCKET:
            logger.warning("QUEUE_BACKEND=s3 but QUEUE_S3_BUCKET is not set. Falling back to memory queue.")
            return InMemoryJobQueue(max_size=max_size)
        return S3JobQueue(
            bucket=settings.QUEUE_S3_BUCKET,
            prefix=settings.QUEUE_S3_PREFIX,
            max_size=max_size,
            region_name=settings.QUEUE_S3_REGION or None,
        )

    return InMemoryJobQueue(max_size=max_size)


def _is_missing_s3_object_error(exc: ClientError) -> bool:
    error = exc.response.get("Error", {})
    code = str(error.get("Code", "")).strip()
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status == 404 or code in {"404", "NoSuchKey", "NotFound"}


# Backward-compatible alias used by tests/import sites.
JobQueue = InMemoryJobQueue
