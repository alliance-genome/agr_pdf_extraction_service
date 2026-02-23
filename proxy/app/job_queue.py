"""In-memory job queue for PDF extraction requests during EC2 startup."""

from collections import deque
from dataclasses import dataclass
from typing import Any


class QueueFullError(Exception):
    pass


@dataclass
class QueuedJob:
    job_id: str
    pdf_data: bytes
    form_fields: dict[str, Any]
    filename: str = "upload.pdf"


class JobQueue:
    """FIFO queue for extraction jobs waiting on EC2."""

    def __init__(self, max_size: int = 10):
        self._max_size = max_size
        self._queue: deque[QueuedJob] = deque()

    @property
    def size(self) -> int:
        return len(self._queue)

    def enqueue(self, job_id: str, pdf_data: bytes, form_fields: dict, filename: str = "upload.pdf") -> None:
        if len(self._queue) >= self._max_size:
            raise QueueFullError(f"Queue full ({self._max_size} jobs max)")
        self._queue.append(QueuedJob(job_id=job_id, pdf_data=pdf_data, form_fields=form_fields, filename=filename))

    def dequeue(self) -> QueuedJob:
        return self._queue.popleft()

    def drain(self) -> list[QueuedJob]:
        jobs = list(self._queue)
        self._queue.clear()
        return jobs

    def has_job(self, job_id: str) -> bool:
        return any(j.job_id == job_id for j in self._queue)
