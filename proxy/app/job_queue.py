"""Queue implementations for PDF extraction startup buffering."""

from __future__ import annotations

import base64
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import boto3

from app.config import settings

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    """Queue cannot accept new jobs."""


@dataclass
class QueuedJob:
    """Serialized queue payload for replay."""

    job_id: str
    pdf_data: bytes
    form_fields: dict[str, Any]
    filename: str = "upload.pdf"
    authorization: str | None = None
    queued_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        payload = {
            "job_id": self.job_id,
            "filename": self.filename,
            "authorization": self.authorization,
            "form_fields": self.form_fields,
            "queued_at": self.queued_at,
            "pdf_data_b64": base64.b64encode(self.pdf_data).decode("ascii"),
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "QueuedJob":
        payload = json.loads(raw)
        return cls(
            job_id=str(payload["job_id"]),
            filename=str(payload.get("filename") or "upload.pdf"),
            authorization=payload.get("authorization"),
            form_fields=dict(payload.get("form_fields") or {}),
            queued_at=float(payload.get("queued_at") or time.time()),
            pdf_data=base64.b64decode(payload["pdf_data_b64"]),
        )


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

    def dequeue(self) -> QueuedJob:
        raise NotImplementedError

    def drain(self) -> list[QueuedJob]:
        raise NotImplementedError

    def has_job(self, job_id: str) -> bool:
        raise NotImplementedError

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
        self._queue.append(
            QueuedJob(
                job_id=job_id,
                pdf_data=pdf_data,
                form_fields=form_fields,
                filename=filename,
                authorization=authorization,
            )
        )

    def dequeue(self) -> QueuedJob:
        return self._queue.popleft()

    def drain(self) -> list[QueuedJob]:
        jobs = list(self._queue)
        self._queue.clear()
        return jobs

    def has_job(self, job_id: str) -> bool:
        return any(j.job_id == job_id for j in self._queue)

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

    def _build_key(self, job: QueuedJob) -> str:
        ts_ms = int(job.queued_at * 1000)
        return f"{self._queue_prefix()}{ts_ms:013d}_{job.job_id}.json"

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
        job = QueuedJob(
            job_id=job_id,
            pdf_data=pdf_data,
            form_fields=form_fields,
            filename=filename,
            authorization=authorization,
        )
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._build_key(job),
            Body=job.to_json().encode("utf-8"),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )

    def dequeue(self) -> QueuedJob:
        keys = self._iter_keys()
        if not keys:
            raise IndexError("dequeue from empty queue")
        key = keys[0]
        obj = self._client.get_object(Bucket=self._bucket, Key=key)
        raw = obj["Body"].read().decode("utf-8")
        self._client.delete_object(Bucket=self._bucket, Key=key)
        return QueuedJob.from_json(raw)

    def drain(self) -> list[QueuedJob]:
        keys = self._iter_keys()
        jobs: list[QueuedJob] = []
        if not keys:
            return jobs

        for key in keys:
            obj = self._client.get_object(Bucket=self._bucket, Key=key)
            raw = obj["Body"].read().decode("utf-8")
            jobs.append(QueuedJob.from_json(raw))

        # Delete in batches of 1000 (S3 limit).
        for i in range(0, len(keys), 1000):
            chunk = keys[i : i + 1000]
            self._client.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": [{"Key": key} for key in chunk], "Quiet": True},
            )
        return jobs

    def has_job(self, job_id: str) -> bool:
        suffix = f"_{job_id}.json"
        return any(key.endswith(suffix) for key in self._iter_keys())

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


# Backward-compatible alias used by tests/import sites.
JobQueue = InMemoryJobQueue
