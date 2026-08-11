"""Tests for in-memory job queue during EC2 startup."""

import asyncio
import io
import json

import pytest
from botocore.exceptions import ClientError

from app.job_queue import JobQueue, QueuedJob, QueueFullError, S3JobQueue


class _ConditionalS3:
    """Small shared S3 model with the conditional operations used by claims."""

    def __init__(self):
        self.objects = {}
        self.etags = {}
        self.version = 0

    def _etag(self, key):
        return self.etags[key]

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        client = self

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                keys = sorted(key for bucket, key in client.objects if bucket == Bucket and key.startswith(Prefix))
                return [{"Contents": [{"Key": key} for key in keys]}] if keys else []

        return _Paginator()

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000, ContinuationToken=None):
        keys = sorted(key for bucket, key in self.objects if bucket == Bucket and key.startswith(Prefix))
        start = int(ContinuationToken or 0)
        selected = keys[start:start + MaxKeys]
        next_index = start + len(selected)
        response = {
            "Contents": [{"Key": key} for key in selected],
            "IsTruncated": next_index < len(keys),
        }
        if response["IsTruncated"]:
            response["NextContinuationToken"] = str(next_index)
        return response

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, **_kwargs):
        object_key = (Bucket, Key)
        current_etag = self.etags.get(object_key)
        if IfNoneMatch == "*" and object_key in self.objects:
            self._precondition_failed("PutObject")
        if IfMatch is not None and current_etag != IfMatch.strip('"'):
            self._precondition_failed("PutObject")
        self.version += 1
        etag = f"etag-{self.version}"
        self.objects[object_key] = Body if isinstance(Body, bytes) else bytes(Body)
        self.etags[object_key] = etag
        return {"ETag": f'"{etag}"'}

    def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):
        self.put_object(Bucket=bucket, Key=key, Body=fileobj.read(), **(ExtraArgs or {}))

    def get_object(self, Bucket, Key):
        object_key = (Bucket, Key)
        if object_key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
                "GetObject",
            )
        return {
            "Body": io.BytesIO(self.objects[object_key]),
            "ETag": f'"{self.etags[object_key]}"',
        }

    def download_fileobj(self, bucket, key, fileobj):
        fileobj.write(self.objects[(bucket, key)])

    def delete_object(self, Bucket, Key, IfMatch=None):
        object_key = (Bucket, Key)
        current_etag = self.etags.get(object_key)
        if IfMatch is not None and current_etag != IfMatch.strip('"'):
            self._precondition_failed("DeleteObject")
        self.objects.pop(object_key, None)
        self.etags.pop(object_key, None)

    def delete_objects(self, Bucket, Delete):
        for item in Delete["Objects"]:
            self.delete_object(Bucket=Bucket, Key=item["Key"])

    @staticmethod
    def _precondition_failed(operation):
        raise ClientError(
            {"Error": {"Code": "PreconditionFailed"}, "ResponseMetadata": {"HTTPStatusCode": 412}},
            operation,
        )


class TestJobQueue:
    def test_enqueue_and_dequeue(self):
        q = JobQueue(max_size=5)
        q.enqueue("job-1", b"pdf-data-1", {"merge": "true"})
        assert q.size == 1
        job = q.dequeue()
        assert job.job_id == "job-1"
        assert job.pdf_data == b"pdf-data-1"
        assert q.size == 0

    def test_fifo_order(self):
        q = JobQueue(max_size=5)
        q.enqueue("a", b"", {})
        q.enqueue("b", b"", {})
        q.enqueue("c", b"", {})
        assert q.dequeue().job_id == "a"
        assert q.dequeue().job_id == "b"
        assert q.dequeue().job_id == "c"

    def test_queue_full_raises(self):
        q = JobQueue(max_size=2)
        q.enqueue("1", b"", {})
        q.enqueue("2", b"", {})
        with pytest.raises(QueueFullError):
            q.enqueue("3", b"", {})

    def test_drain_returns_all(self):
        q = JobQueue(max_size=5)
        q.enqueue("a", b"", {})
        q.enqueue("b", b"", {})
        jobs = q.drain()
        assert len(jobs) == 2
        assert q.size == 0

    def test_has_job(self):
        q = JobQueue(max_size=5)
        q.enqueue("abc", b"", {})
        assert q.has_job("abc") is True
        assert q.has_job("xyz") is False

    def test_remove_job_removes_only_target(self):
        q = JobQueue(max_size=5)
        q.enqueue("a", b"a", {})
        q.enqueue("b", b"b", {})
        q.enqueue("c", b"c", {})

        removed = q.remove_job("b")
        assert removed is True
        assert q.size == 2
        assert q.has_job("a") is True
        assert q.has_job("b") is False
        assert q.has_job("c") is True
        assert q.dequeue().job_id == "a"
        assert q.dequeue().job_id == "c"

    def test_remove_job_returns_false_for_unknown_id(self):
        q = JobQueue(max_size=5)
        q.enqueue("a", b"a", {})
        assert q.remove_job("missing") is False
        assert q.size == 1

    def test_dequeue_empty_raises(self):
        q = JobQueue(max_size=5)
        with pytest.raises(IndexError):
            q.dequeue()

    def test_custom_filename(self):
        q = JobQueue(max_size=5)
        q.enqueue("job-1", b"data", {"merge": "true"}, filename="paper.pdf")
        job = q.dequeue()
        assert job.filename == "paper.pdf"

    def test_authorization_context_is_preserved(self):
        q = JobQueue(max_size=5)
        q.enqueue(
            "job-auth-1",
            b"data",
            {"merge": "true"},
            filename="paper.pdf",
            authorization="Bearer token-abc",
        )
        job = q.dequeue()
        assert job.authorization == "Bearer token-abc"

    def test_acceptance_cannot_overwrite_durable_cancellation_intent(self):
        q = JobQueue(max_size=5)
        q.record_cancel_requested("job-cancel-race", "User cancelled")

        assert q.record_accepted("job-cancel-race") is False
        assert q.get_durable_phase("job-cancel-race") == "cancel_requested"


class TestS3JobQueue:
    def test_two_proxy_queues_cannot_claim_the_same_job(self, monkeypatch):
        shared_s3 = _ConditionalS3()
        monkeypatch.setattr("app.job_queue.boto3.client", lambda *_args, **_kwargs: shared_s3)
        first = S3JobQueue(bucket="test-bucket", prefix="prefix")
        second = S3JobQueue(bucket="test-bucket", prefix="prefix")
        first.enqueue("job-shared", b"%PDF shared", {"merge": "true"})

        claimed = first.claim_next("proxy-a", lease_seconds=60)

        assert claimed is not None
        assert claimed.job_id == "job-shared"
        assert claimed.claim_owner == "proxy-a"
        assert second.claim_next("proxy-b", lease_seconds=60) is None

    def test_expired_claim_is_atomically_reclaimable(self, monkeypatch):
        shared_s3 = _ConditionalS3()
        monkeypatch.setattr("app.job_queue.boto3.client", lambda *_args, **_kwargs: shared_s3)
        now = {"value": 100.0}
        monkeypatch.setattr("app.job_queue.time.time", lambda: now["value"])
        first = S3JobQueue(bucket="test-bucket", prefix="prefix")
        second = S3JobQueue(bucket="test-bucket", prefix="prefix")
        first.enqueue("job-stale", b"%PDF stale", {})
        original = first.claim_next("proxy-a", lease_seconds=10)

        now["value"] = 111.0
        reclaimed = second.claim_next("proxy-b", lease_seconds=10)

        assert original is not None
        assert reclaimed is not None
        assert reclaimed.job_id == original.job_id
        assert reclaimed.claim_owner == "proxy-b"
        assert first.release_claim(original) is False
        assert second.release_claim(reclaimed) is True
        assert first.has_job("job-stale") is True

    def test_accepted_handoff_is_secret_free_before_queue_ack(self, monkeypatch):
        shared_s3 = _ConditionalS3()
        monkeypatch.setattr("app.job_queue.boto3.client", lambda *_args, **_kwargs: shared_s3)
        queue = S3JobQueue(bucket="test-bucket", prefix="prefix")
        queue.enqueue(
            "job-accepted",
            b"%PDF accepted",
            {},
            authorization="Bearer must-not-copy",
        )
        claimed = queue.claim_next("proxy-a", lease_seconds=60)

        queue.record_accepted("job-accepted")
        accepted_key = ("test-bucket", "prefix/accepted/job-accepted.json")
        accepted_payload = json.loads(shared_s3.objects[accepted_key])
        assert accepted_payload["process_id"] == "job-accepted"
        assert accepted_payload["phase"] == "accepted"
        assert "authorization" not in accepted_payload
        assert "token" not in json.dumps(accepted_payload).lower()
        assert queue.has_job("job-accepted") is True

        assert queue.acknowledge_claim(claimed) is True
        assert queue.has_job("job-accepted") is False
        assert queue.get_durable_phase("job-accepted") == "accepted"

    def test_s3_acceptance_cannot_overwrite_durable_cancellation_intent(self, monkeypatch):
        shared_s3 = _ConditionalS3()
        monkeypatch.setattr("app.job_queue.boto3.client", lambda *_args, **_kwargs: shared_s3)
        queue = S3JobQueue(bucket="test-bucket", prefix="prefix")
        queue.record_cancel_requested("job-cancel-race", "User cancelled")

        assert queue.record_accepted("job-cancel-race") is False
        assert queue.get_durable_phase("job-cancel-race") == "cancel_requested"
        assert queue.get_durable_record("job-cancel-race")["message"] == "User cancelled"

    def test_release_claim_keeps_unaccepted_job_replayable(self, monkeypatch):
        shared_s3 = _ConditionalS3()
        monkeypatch.setattr("app.job_queue.boto3.client", lambda *_args, **_kwargs: shared_s3)
        queue = S3JobQueue(bucket="test-bucket", prefix="prefix")
        queue.enqueue("job-capacity", b"%PDF capacity", {})
        claimed = queue.claim_next("proxy-a", lease_seconds=60)

        assert queue.release_claim(claimed) is True
        assert queue.has_job("job-capacity") is True
        assert queue.claim_next("proxy-b", lease_seconds=60).job_id == "job-capacity"

    def test_retention_cleanup_never_deletes_active_accepted_record(self, monkeypatch):
        shared_s3 = _ConditionalS3()
        monkeypatch.setattr("app.job_queue.boto3.client", lambda *_args, **_kwargs: shared_s3)
        now = {"value": 100.0}
        monkeypatch.setattr("app.job_queue.time.time", lambda: now["value"])
        queue = S3JobQueue(bucket="test-bucket", prefix="prefix")
        queue.record_accepted("job-active")
        now["value"] = 200.0

        assert queue.cleanup_durable_record("job-active", 60, terminal=False) is False
        assert queue.get_durable_phase("job-active") == "accepted"
        assert queue.cleanup_durable_record("job-active", 60, terminal=True) is True
        assert queue.get_durable_phase("job-active") is None

    def test_s3_expired_record_listing_is_bounded_for_eventual_revalidation(self, monkeypatch):
        shared_s3 = _ConditionalS3()
        monkeypatch.setattr("app.job_queue.boto3.client", lambda *_args, **_kwargs: shared_s3)
        now = {"value": 100.0}
        monkeypatch.setattr("app.job_queue.time.time", lambda: now["value"])
        queue = S3JobQueue(bucket="test-bucket", prefix="prefix")
        queue.record_accepted("old-a")
        queue.record_accepted("old-b")
        now["value"] = 200.0
        queue.record_accepted("active")

        first = queue.list_expired_durable_records(retention_seconds=60, limit=1)
        second = queue.list_expired_durable_records(retention_seconds=60, limit=1)

        assert {first[0]["process_id"], second[0]["process_id"]} == {"old-a", "old-b"}
        assert queue.get_durable_phase("active") == "accepted"

    def test_s3_pointer_job_open_pdf_downloads_lazily_and_cleans_remote(self, monkeypatch):
        class _FakeS3Client:
            def __init__(self):
                self.deleted = []

            def download_fileobj(self, bucket, key, fileobj):
                assert bucket == "test-bucket"
                assert key == "prefix/payloads/job-lazy.pdf"
                fileobj.write(b"%PDF lazy")

            def delete_object(self, Bucket, Key):
                self.deleted.append((Bucket, Key))

        fake_client = _FakeS3Client()
        monkeypatch.setattr("app.job_queue.boto3.client", lambda *_args, **_kwargs: fake_client)

        job = QueuedJob(
            job_id="job-lazy",
            pdf_data=None,
            form_fields={},
            pdf_s3_bucket="test-bucket",
            pdf_s3_key="prefix/payloads/job-lazy.pdf",
        )

        with job.open_pdf() as pdf_file:
            assert pdf_file.read() == b"%PDF lazy"

        assert job.pdf_file_path is not None
        job.cleanup(delete_remote=True)
        assert job.pdf_file_path is None
        assert job.pdf_s3_key is None
        assert fake_client.deleted == [("test-bucket", "prefix/payloads/job-lazy.pdf")]

    def test_enqueue_upload_stores_pdf_as_separate_s3_object(self, monkeypatch):
        class _Paginator:
            def paginate(self, **kwargs):
                return []

        class _Upload:
            def __init__(self):
                self.file = io.BytesIO(b"%PDF large-ish")

            async def seek(self, offset):
                self.file.seek(offset)

        class _FakeS3Client:
            def __init__(self):
                self.uploaded = {}
                self.metadata = {}

            def get_paginator(self, name):
                assert name == "list_objects_v2"
                return _Paginator()

            def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):
                self.uploaded[(bucket, key)] = {
                    "body": fileobj.read(),
                    "extra": ExtraArgs,
                }

            def put_object(self, Bucket, Key, Body, **kwargs):
                self.metadata[(Bucket, Key)] = {
                    "body": Body,
                    "kwargs": kwargs,
                }

            def delete_object(self, Bucket, Key):
                raise AssertionError("metadata write should not fail in this test")

        fake_client = _FakeS3Client()
        monkeypatch.setattr("app.job_queue.boto3.client", lambda *_args, **_kwargs: fake_client)

        q = S3JobQueue(bucket="test-bucket", prefix="prefix")
        job = asyncio.run(
            q.enqueue_upload(
                "job-large",
                _Upload(),
                {"merge": "true"},
                filename="large.pdf",
                authorization="Bearer token",
            )
        )

        assert job.pdf_data is None
        assert job.pdf_s3_key == "prefix/payloads/job-large.pdf"
        assert fake_client.uploaded[("test-bucket", "prefix/payloads/job-large.pdf")]["body"] == b"%PDF large-ish"

        [(bucket, metadata_key)] = fake_client.metadata.keys()
        assert bucket == "test-bucket"
        assert metadata_key.endswith("_job-large.json")
        metadata_payload = json.loads(fake_client.metadata[(bucket, metadata_key)]["body"])
        assert metadata_payload["pdf_s3_key"] == "prefix/payloads/job-large.pdf"
        assert "pdf_data_b64" not in metadata_payload

    def test_remove_job_deletes_matching_object(self, monkeypatch):
        class _Paginator:
            def paginate(self, **kwargs):
                return [
                    {
                        "Contents": [
                            {"Key": "prefix/jobs/0000000000001_job-a.json"},
                            {"Key": "prefix/jobs/0000000000002_job-b.json"},
                        ]
                    }
                ]

        class _FakeS3Client:
            def __init__(self):
                self.deleted = []

            def get_paginator(self, name):
                assert name == "list_objects_v2"
                return _Paginator()

            def get_object(self, Bucket, Key):
                payload = {
                    "job_id": "job-b",
                    "filename": "paper.pdf",
                    "form_fields": {},
                    "queued_at": 1,
                    "pdf_s3_bucket": Bucket,
                    "pdf_s3_key": "prefix/payloads/job-b.pdf",
                }
                return {"Body": io.BytesIO(json.dumps(payload).encode("utf-8"))}

            def delete_object(self, Bucket, Key):
                self.deleted.append((Bucket, Key))

            def delete_objects(self, Bucket, Delete):
                for item in Delete["Objects"]:
                    self.deleted.append((Bucket, item["Key"]))

        fake_client = _FakeS3Client()
        monkeypatch.setattr("app.job_queue.boto3.client", lambda *_args, **_kwargs: fake_client)

        q = S3JobQueue(bucket="test-bucket", prefix="prefix")
        assert q.remove_job("job-b") is True
        assert fake_client.deleted == [
            ("test-bucket", "prefix/jobs/0000000000002_job-b.json"),
            ("test-bucket", "prefix/payloads/job-b.pdf"),
        ]

    def test_drain_skips_orphaned_metadata_when_payload_is_missing(self, monkeypatch):
        class _Paginator:
            def paginate(self, **kwargs):
                return [
                    {
                        "Contents": [
                            {"Key": "prefix/jobs/0000000000001_job-missing.json"},
                            {"Key": "prefix/jobs/0000000000002_job-ok.json"},
                        ]
                    }
                ]

        class _FakeS3Client:
            def __init__(self):
                self.deleted = []

            def get_paginator(self, name):
                assert name == "list_objects_v2"
                return _Paginator()

            def get_object(self, Bucket, Key):
                job_id = "job-missing" if Key.endswith("job-missing.json") else "job-ok"
                payload = {
                    "job_id": job_id,
                    "filename": "paper.pdf",
                    "form_fields": {},
                    "queued_at": 1,
                    "pdf_s3_bucket": Bucket,
                    "pdf_s3_key": f"prefix/payloads/{job_id}.pdf",
                }
                return {"Body": io.BytesIO(json.dumps(payload).encode("utf-8"))}

            def download_fileobj(self, bucket, key, fileobj):
                if key.endswith("job-missing.pdf"):
                    raise ClientError(
                        {
                            "Error": {"Code": "404", "Message": "Not Found"},
                            "ResponseMetadata": {"HTTPStatusCode": 404},
                        },
                        "HeadObject",
                    )
                fileobj.write(b"%PDF ok")

            def delete_objects(self, Bucket, Delete):
                for item in Delete["Objects"]:
                    self.deleted.append((Bucket, item["Key"]))

        fake_client = _FakeS3Client()
        monkeypatch.setattr("app.job_queue.boto3.client", lambda *_args, **_kwargs: fake_client)

        q = S3JobQueue(bucket="test-bucket", prefix="prefix")
        jobs = q.drain()

        assert [job.job_id for job in jobs] == ["job-ok"]
        with jobs[0].open_pdf() as pdf_file:
            assert pdf_file.read() == b"%PDF ok"
        jobs[0].cleanup()
        assert fake_client.deleted == [
            ("test-bucket", "prefix/jobs/0000000000001_job-missing.json"),
        ]

        assert q.acknowledge("job-ok") is True
        assert fake_client.deleted == [
            ("test-bucket", "prefix/jobs/0000000000001_job-missing.json"),
            ("test-bucket", "prefix/jobs/0000000000002_job-ok.json"),
        ]

    def test_acknowledge_deletes_metadata_without_payload(self, monkeypatch):
        class _Paginator:
            def paginate(self, **kwargs):
                return [
                    {
                        "Contents": [
                            {"Key": "prefix/jobs/0000000000001_job-a.json"},
                        ]
                    }
                ]

        class _FakeS3Client:
            def __init__(self):
                self.deleted = []

            def get_paginator(self, name):
                assert name == "list_objects_v2"
                return _Paginator()

            def delete_objects(self, Bucket, Delete):
                for item in Delete["Objects"]:
                    self.deleted.append((Bucket, item["Key"]))

        fake_client = _FakeS3Client()
        monkeypatch.setattr("app.job_queue.boto3.client", lambda *_args, **_kwargs: fake_client)

        q = S3JobQueue(bucket="test-bucket", prefix="prefix")

        assert q.acknowledge("job-a") is True
        assert fake_client.deleted == [
            ("test-bucket", "prefix/jobs/0000000000001_job-a.json"),
        ]
