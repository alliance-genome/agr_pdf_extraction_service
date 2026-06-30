"""Tests for in-memory job queue during EC2 startup."""

import asyncio
import io
import json

import pytest
from app.job_queue import JobQueue, QueuedJob, QueueFullError, S3JobQueue


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


class TestS3JobQueue:
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
