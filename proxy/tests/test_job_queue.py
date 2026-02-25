"""Tests for in-memory job queue during EC2 startup."""

import pytest
from app.job_queue import JobQueue, QueueFullError


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
