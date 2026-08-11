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
    metadata_key: str | None = None
    claim_owner: str | None = None
    claim_etag: str | None = None

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

    def claim_next(self, owner_id: str, lease_seconds: int) -> QueuedJob | None:
        raise NotImplementedError

    def release_claim(self, job: QueuedJob) -> bool:
        raise NotImplementedError

    def acknowledge_claim(self, job: QueuedJob) -> bool:
        raise NotImplementedError

    def record_accepted(self, job_id: str) -> bool:
        raise NotImplementedError

    def record_cancelled(self, job_id: str, reason: str) -> None:
        raise NotImplementedError

    def record_cancel_requested(self, job_id: str, reason: str) -> None:
        raise NotImplementedError

    def record_failed(self, job_id: str, reason: str) -> None:
        raise NotImplementedError

    def get_durable_phase(self, job_id: str) -> str | None:
        raise NotImplementedError

    def get_durable_record(self, job_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def cleanup_durable_record(self, job_id: str, retention_seconds: int, *, terminal: bool) -> bool:
        raise NotImplementedError

    def list_expired_durable_records(self, retention_seconds: int, limit: int) -> list[dict[str, Any]]:
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
        self._claims: dict[str, tuple[str, float]] = {}
        self._durable_phases: dict[str, dict[str, Any]] = {}

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

    def claim_next(self, owner_id: str, lease_seconds: int) -> QueuedJob | None:
        now = time.time()
        for job in self._queue:
            claim = self._claims.get(job.job_id)
            if claim and claim[1] > now:
                continue
            self._claims[job.job_id] = (owner_id, now + max(1, lease_seconds))
            job.claim_owner = owner_id
            job.claim_etag = f"memory:{owner_id}:{now}"
            return job
        return None

    def release_claim(self, job: QueuedJob) -> bool:
        claim = self._claims.get(job.job_id)
        if not claim or claim[0] != job.claim_owner:
            return False
        self._claims.pop(job.job_id, None)
        return True

    def acknowledge_claim(self, job: QueuedJob) -> bool:
        claim = self._claims.get(job.job_id)
        if not claim or claim[0] != job.claim_owner:
            return False
        removed = self.remove_job(job.job_id)
        self._claims.pop(job.job_id, None)
        return removed

    def record_accepted(self, job_id: str) -> bool:
        existing = self._durable_phases.get(job_id)
        if existing and existing.get("phase") in {"cancel_requested", "cancelled", "failed"}:
            return False
        self._durable_phases[job_id] = {
            "process_id": job_id,
            "phase": "accepted",
            "recorded_at": time.time(),
        }
        return True

    def record_cancelled(self, job_id: str, reason: str) -> None:
        self._durable_phases[job_id] = {
            "process_id": job_id,
            "phase": "cancelled",
            "message": reason,
            "recorded_at": time.time(),
        }

    def record_cancel_requested(self, job_id: str, reason: str) -> None:
        self._durable_phases[job_id] = {
            "process_id": job_id,
            "phase": "cancel_requested",
            "message": reason,
            "recorded_at": time.time(),
        }

    def record_failed(self, job_id: str, reason: str) -> None:
        self._durable_phases[job_id] = {
            "process_id": job_id,
            "phase": "failed",
            "message": reason,
            "recorded_at": time.time(),
        }

    def get_durable_phase(self, job_id: str) -> str | None:
        record = self._durable_phases.get(job_id)
        if record:
            return str(record["phase"])
        if self.has_job(job_id):
            return "claimed" if job_id in self._claims else "queued"
        return None

    def get_durable_record(self, job_id: str) -> dict[str, Any] | None:
        record = self._durable_phases.get(job_id)
        return dict(record) if record else None

    def cleanup_durable_record(self, job_id: str, retention_seconds: int, *, terminal: bool) -> bool:
        record = self._durable_phases.get(job_id)
        if not record or not terminal:
            return False
        recorded_at = record.get("recorded_at")
        age = time.time() - float(recorded_at if recorded_at is not None else time.time())
        if age < max(0, retention_seconds):
            return False
        self._durable_phases.pop(job_id, None)
        return True

    def list_expired_durable_records(self, retention_seconds: int, limit: int) -> list[dict[str, Any]]:
        cutoff = time.time() - max(0, retention_seconds)
        records = [
            dict(record)
            for record in self._durable_phases.values()
            if float(
                record.get("recorded_at")
                if record.get("recorded_at") is not None
                else time.time()
            ) <= cutoff
        ]
        records.sort(key=lambda record: float(record.get("recorded_at") or 0))
        return records[:max(0, limit)]

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
        self._accepted_cleanup_cursor: str | None = None

    @property
    def durable(self) -> bool:
        return True

    def _queue_prefix(self) -> str:
        return f"{self._prefix}/jobs/"

    def _payload_prefix(self) -> str:
        return f"{self._prefix}/payloads/"

    def _claim_key(self, job_id: str) -> str:
        return f"{self._prefix}/claims/{job_id}.json"

    def _accepted_key(self, job_id: str) -> str:
        return f"{self._prefix}/accepted/{job_id}.json"

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
        job.metadata_key = key
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

    def claim_next(self, owner_id: str, lease_seconds: int) -> QueuedJob | None:
        """Conditionally claim the oldest available job without deleting it."""
        for metadata_key in self._iter_keys():
            suffix = metadata_key.rsplit("_", 1)[-1]
            job_id = suffix[:-5] if suffix.endswith(".json") else suffix
            claim_etag = self._acquire_claim(job_id, owner_id, lease_seconds)
            if claim_etag is None:
                continue
            placeholder = QueuedJob(job_id=job_id, pdf_data=b"", form_fields={})
            placeholder.claim_owner = owner_id
            placeholder.claim_etag = claim_etag
            try:
                job = self._load_job(metadata_key)
            except QueuePayloadMissingError as exc:
                logger.warning("%s; deleting orphaned queue metadata %s", exc, metadata_key)
                self._delete_job_metadata(metadata_key)
                self.release_claim(placeholder)
                continue
            except Exception:
                self.release_claim(placeholder)
                raise
            job.claim_owner = owner_id
            job.claim_etag = claim_etag
            return job
        return None

    def _acquire_claim(self, job_id: str, owner_id: str, lease_seconds: int) -> str | None:
        claim_key = self._claim_key(job_id)
        now = time.time()
        payload = json.dumps(
            {
                "process_id": job_id,
                "owner_id": owner_id,
                "claimed_at": now,
                "expires_at": now + max(1, lease_seconds),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        put_args = {
            "Bucket": self._bucket,
            "Key": claim_key,
            "Body": payload,
            "ContentType": "application/json",
            "ServerSideEncryption": "AES256",
        }
        try:
            response = self._client.put_object(**put_args, IfNoneMatch="*")
            return str(response.get("ETag", "")).strip('"')
        except ClientError as exc:
            if not _is_s3_precondition_error(exc):
                raise

        try:
            existing = self._client.get_object(Bucket=self._bucket, Key=claim_key)
        except ClientError as exc:
            if not _is_missing_s3_object_error(exc):
                raise
            try:
                response = self._client.put_object(**put_args, IfNoneMatch="*")
                return str(response.get("ETag", "")).strip('"')
            except ClientError as retry_exc:
                if _is_s3_precondition_error(retry_exc):
                    return None
                raise

        existing_payload = json.loads(existing["Body"].read().decode("utf-8"))
        existing_etag = str(existing.get("ETag", "")).strip('"')
        if float(existing_payload.get("expires_at") or 0) > now:
            return None
        try:
            response = self._client.put_object(**put_args, IfMatch=existing_etag)
            return str(response.get("ETag", "")).strip('"')
        except ClientError as exc:
            if _is_s3_precondition_error(exc):
                return None
            raise

    def release_claim(self, job: QueuedJob) -> bool:
        if not job.claim_owner or not job.claim_etag:
            return False
        claim_key = self._claim_key(job.job_id)
        try:
            current = self._client.get_object(Bucket=self._bucket, Key=claim_key)
            payload = json.loads(current["Body"].read().decode("utf-8"))
            current_etag = str(current.get("ETag", "")).strip('"')
            if payload.get("owner_id") != job.claim_owner or current_etag != job.claim_etag:
                return False
            self._client.delete_object(
                Bucket=self._bucket,
                Key=claim_key,
                IfMatch=job.claim_etag,
            )
            return True
        except ClientError as exc:
            if _is_missing_s3_object_error(exc) or _is_s3_precondition_error(exc):
                return False
            raise

    def acknowledge_claim(self, job: QueuedJob) -> bool:
        if not job.metadata_key or not job.claim_owner or not job.claim_etag:
            return False
        try:
            current = self._client.get_object(Bucket=self._bucket, Key=self._claim_key(job.job_id))
        except ClientError as exc:
            if _is_missing_s3_object_error(exc):
                return False
            raise
        payload = json.loads(current["Body"].read().decode("utf-8"))
        current_etag = str(current.get("ETag", "")).strip('"')
        if payload.get("owner_id") != job.claim_owner or current_etag != job.claim_etag:
            return False
        self._delete_job_metadata(job.metadata_key)
        self.release_claim(job)
        return True

    def record_accepted(self, job_id: str) -> bool:
        """Create the accepted marker without overwriting cancellation/failure intent."""
        payload = self._durable_phase_payload(job_id, "accepted")
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=self._accepted_key(job_id),
                Body=payload,
                ContentType="application/json",
                ServerSideEncryption="AES256",
                IfNoneMatch="*",
            )
            return True
        except ClientError as exc:
            if not _is_s3_precondition_error(exc):
                raise
        existing = self.get_durable_record(job_id)
        return bool(existing and existing.get("phase") == "accepted")

    def record_cancelled(self, job_id: str, reason: str) -> None:
        self._put_durable_phase(job_id, "cancelled", message=reason)

    def record_cancel_requested(self, job_id: str, reason: str) -> None:
        self._put_durable_phase(job_id, "cancel_requested", message=reason)

    def record_failed(self, job_id: str, reason: str) -> None:
        self._put_durable_phase(job_id, "failed", message=reason)

    def _put_durable_phase(self, job_id: str, phase: str, **extra: Any) -> None:
        payload = self._durable_phase_payload(job_id, phase, **extra)
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._accepted_key(job_id),
            Body=payload,
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )

    @staticmethod
    def _durable_phase_payload(job_id: str, phase: str, **extra: Any) -> bytes:
        payload = {
            "process_id": job_id,
            "phase": phase,
            "recorded_at": time.time(),
            **extra,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

    def get_durable_phase(self, job_id: str) -> str | None:
        record = self.get_durable_record(job_id)
        if record is not None:
            return str(record.get("phase") or "") or None
        if not self.has_job(job_id):
            return None
        try:
            self._client.get_object(Bucket=self._bucket, Key=self._claim_key(job_id))
            return "claimed"
        except ClientError as exc:
            if _is_missing_s3_object_error(exc):
                return "queued"
            raise

    def get_durable_record(self, job_id: str) -> dict[str, Any] | None:
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=self._accepted_key(job_id))
            payload = json.loads(obj["Body"].read().decode("utf-8"))
            return dict(payload)
        except ClientError as exc:
            if not _is_missing_s3_object_error(exc):
                raise
        return None

    def cleanup_durable_record(self, job_id: str, retention_seconds: int, *, terminal: bool) -> bool:
        record = self.get_durable_record(job_id)
        if record is None or not terminal:
            return False
        recorded_at = record.get("recorded_at")
        age = time.time() - float(recorded_at if recorded_at is not None else time.time())
        if age < max(0, retention_seconds):
            return False
        self._client.delete_object(Bucket=self._bucket, Key=self._accepted_key(job_id))
        return True

    def list_expired_durable_records(self, retention_seconds: int, limit: int) -> list[dict[str, Any]]:
        """List a bounded rotating batch for eventual terminal revalidation."""
        remaining = max(0, limit)
        if remaining == 0:
            return []
        cutoff = time.time() - max(0, retention_seconds)
        records: list[dict[str, Any]] = []
        continuation = self._accepted_cleanup_cursor
        prefix = f"{self._prefix}/accepted/"
        while remaining > 0:
            params = {
                "Bucket": self._bucket,
                "Prefix": prefix,
                "MaxKeys": min(1000, remaining),
            }
            if continuation:
                params["ContinuationToken"] = continuation
            try:
                response = self._client.list_objects_v2(**params)
            except Exception:
                if continuation:
                    self._accepted_cleanup_cursor = None
                raise
            for item in response.get("Contents", []):
                try:
                    obj = self._client.get_object(Bucket=self._bucket, Key=item["Key"])
                    record = json.loads(obj["Body"].read().decode("utf-8"))
                    recorded_at = record.get("recorded_at")
                    if float(recorded_at if recorded_at is not None else time.time()) <= cutoff:
                        records.append(dict(record))
                        remaining -= 1
                        if remaining == 0:
                            break
                except Exception as exc:
                    logger.warning("Skipping unreadable durable status record %s: %s", item.get("Key"), exc)
            if remaining == 0:
                self._accepted_cleanup_cursor = (
                    response.get("NextContinuationToken")
                    if response.get("IsTruncated")
                    else None
                )
                break
            if not response.get("IsTruncated"):
                self._accepted_cleanup_cursor = None
                break
            continuation = response.get("NextContinuationToken")
            if not continuation:
                self._accepted_cleanup_cursor = None
                break
        records.sort(key=lambda record: float(record.get("recorded_at") or 0))
        return records

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


def _is_s3_precondition_error(exc: ClientError) -> bool:
    error = exc.response.get("Error", {})
    code = str(error.get("Code", "")).strip()
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status == 412 or code in {"412", "PreconditionFailed"}


# Backward-compatible alias used by tests/import sites.
JobQueue = InMemoryJobQueue
