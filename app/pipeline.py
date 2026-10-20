"""PostgreSQL-authoritative stage ownership, outbox, cancellation and recovery.

Messages are repeatable notifications. A committed row, never Redis, owns work.
All mutations lock the public run before its pipeline row, including cancellation.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import time

from sqlalchemy import select, text, case, func, and_
from sqlalchemy.orm import Session
from app.pipeline_errors import (
    OwnershipLost,
    PipelineDeadlineExceeded,
    ProviderAdmissionError,
    StageControlError,
)
from app.models import (
    ExtractionRun,
    PipelineJob,
    TERMINAL_EXTRACTION_STATUSES,
    get_engine,
)
from config import Config
from pdfx_observability import capture_failure


def now():
    return datetime.now(timezone.utc)


def aware(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class UncertainProviderOutcome(RuntimeError):
    pass


@contextmanager
def session():
    with Session(get_engine(), expire_on_commit=False) as db:
        with db.begin():
            yield db


def locked(db, process_id):
    run = db.scalar(
        select(ExtractionRun)
        .where(ExtractionRun.process_id == process_id)
        .with_for_update()
    )
    job = db.scalar(
        select(PipelineJob)
        .where(PipelineJob.process_id == process_id)
        .with_for_update()
    )
    return run, job


def owned(db, process_id, generation):
    run, job = locked(db, process_id)
    if (
        not run
        or not job
        or run.status in TERMINAL_EXTRACTION_STATUSES
        or job.state != "running"
        or job.generation != generation
    ):
        raise OwnershipLost("Stage ownership is no longer current")
    if aware(job.deadline_at) <= now():
        raise PipelineDeadlineExceeded("Pipeline deadline expired")
    return run, job


def accept(process_id, source, options, claimed_at):
    """Publish no broker messages here: durable ready row is the outbox."""
    with session() as db:
        run, existing = locked(db, process_id)
        if existing:
            return
        if not run or run.status != "submitting" or run.started_at != claimed_at:
            # SQLite drops timezone; compare normalized timestamps for local tests.
            if (
                not run
                or run.status != "submitting"
                or aware(run.started_at) != aware(claimed_at)
            ):
                raise OwnershipLost("Submission claim changed")
        timestamp = now()
        db.add(
            PipelineJob(
                process_id=process_id,
                source=source,
                options=options,
                created_at=timestamp,
                deadline_at=timestamp
                + timedelta(seconds=Config.PIPELINE_QUEUE_MAX_AGE_SECONDS),
                stage="extraction",
                state="ready",
                generation=1,
                attempts=0,
                provider_started=False,
            )
        )
        run.status = "queued"
        run.started_at = None


def snapshot(process_id):
    with session() as db:
        job = db.get(PipelineJob, process_id)
        if not job:
            return None
        return {
            col.name: getattr(job, col.name) for col in PipelineJob.__table__.columns
        }


def claim(process_id, generation, stage):
    expired = None
    result = None
    with session() as db:
        run, job = locked(db, process_id)
        if (
            not run
            or not job
            or run.status in TERMINAL_EXTRACTION_STATUSES
            or job.generation != generation
            or job.stage != stage
            or job.state != "dispatched"
        ):
            return None
        if aware(job.deadline_at) <= now():
            expired = PipelineDeadlineExceeded("Pipeline deadline expired")
            _fail(run, job, expired)
        else:
            if run.started_at is None:
                # A batch waiting behind earlier PDFs has its own queue-age
                # bound. Start execution time once, preserving it on retries
                # and across the CPU handoff.
                run.started_at = now()
                job.deadline_at = run.started_at + timedelta(
                    seconds=Config.PIPELINE_DEADLINE_SECONDS
                )
            job.state = "running"
            job.heartbeat_at = job.stage_started_at = now()
            job.attempts += 1
            run.status = "running"
            result = {
                col.name: getattr(job, col.name)
                for col in PipelineJob.__table__.columns
            }
            result["execution_started_at"] = run.started_at
    if expired:
        capture_failure(
            expired, process_id=process_id, stage=stage, generation=generation
        )
    return result


def heartbeat(process_id, generation, progress=None):
    with session() as db:
        _, job = owned(db, process_id, generation)
        job.heartbeat_at = now()
        if progress is not None:
            job.progress = progress


def handoff(process_id, generation, checkpoint):
    with session() as db:
        _, job = owned(db, process_id, generation)
        job.checkpoint = checkpoint
        job.extraction_ended_at = now()
        job.stage = "finalization"
        job.state = "ready"
        job.generation += 1
        job.attempts = 0
        job.dispatched_at = job.heartbeat_at = job.stage_started_at = None
        job.progress = {"stage": "awaiting_finalization"}


def complete(process_id, generation, result, values):
    with session() as db:
        run, job = owned(db, process_id, generation)
        for key, value in values.items():
            setattr(run, key, value)
        run.status = "succeeded"
        run.ended_at = now()
        job.state = "terminal"
        job.result = result


def _fail(run, job, exc):
    run.status = "failed"
    run.ended_at = now()
    run.error_code = type(exc).__name__
    # Keep external diagnostics bounded and avoid spilling provider payloads.
    run.error_message = str(exc)[: Config.PIPELINE_ERROR_MESSAGE_MAX_CHARS]
    job.state = "terminal"
    job.result = {
        "status": "failed",
        "process_id": str(run.process_id),
        "error": run.error_message,
    }


def fail(process_id, generation, exc, values=None):
    with session() as db:
        run, job = locked(db, process_id)
        if (
            not run
            or not job
            or run.status in TERMINAL_EXTRACTION_STATUSES
            or job.generation != generation
        ):
            return False
        _fail(run, job, exc)
        for key, value in (values or {}).items():
            setattr(run, key, value)
    capture_failure(exc, process_id=process_id, stage=job.stage, generation=generation)
    return True


def cancel(process_id, reason):
    with session() as db:
        run, job = locked(db, process_id)
        if not job:
            return None
        if run.status in TERMINAL_EXTRACTION_STATUSES:
            return False
        run.status = "cancelled"
        run.error_code = "cancelled"
        run.ended_at = now()
        run.error_message = reason
        job.state = "terminal"
        job.generation += 1
        job.result = {"status": "cancelled", "process_id": process_id}
        return True


@contextmanager
def live_lock(process_id):
    """Dedicated DB session lock; workers hold it until all computation stops."""
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        # SQLite is deliberately not a multi-process scheduler backend.
        raise RuntimeError("Stage execution requires PostgreSQL ownership locks")
    connection = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    key = f"pdfx-pipeline:{process_id}"
    acquired = False
    try:
        acquired = connection.execute(
            text("SELECT pg_try_advisory_lock(hashtextextended(:key, 0))"), {"key": key}
        ).scalar()
        yield connection if acquired else None
    finally:
        if acquired:
            try:
                connection.execute(
                    text("SELECT pg_advisory_unlock(hashtextextended(:key, 0))"),
                    {"key": key},
                )
            except Exception:
                connection.invalidate()
        connection.close()


def dispatch_once(publish):
    """One globally serialized admission pass, followed by at-least-once publish."""
    messages = []
    failures = []
    timestamp = now()
    with session() as db:
        if db.bind.dialect.name != "postgresql":
            raise RuntimeError("Dispatcher requires PostgreSQL")
        if not db.execute(
            text(
                "SELECT pg_try_advisory_xact_lock(hashtextextended('pdfx-dispatcher',0))"
            )
        ).scalar():
            return 0
        candidates = db.scalars(
            select(PipelineJob)
            .where(PipelineJob.state != "terminal")
            .order_by(
                case(
                    (
                        and_(
                            PipelineJob.stage == "extraction",
                            PipelineJob.state == "ready",
                        ),
                        1,
                    ),
                    else_=0,
                ),
                PipelineJob.created_at,
            )
            .limit(Config.PIPELINE_DISPATCH_BATCH_SIZE)
        ).all()
        # Lock run then job in a consistent order, reloading states changed by workers.
        pairs = [locked(db, j.process_id) for j in candidates]
        db.expire_all()
        for run, job in pairs:
            if run.status in TERMINAL_EXTRACTION_STATUSES:
                job.state = "terminal"
                continue
            if job.state == "running":
                if (
                    aware(job.heartbeat_at)
                    + timedelta(seconds=Config.PIPELINE_LEASE_SECONDS)
                    > timestamp
                ):
                    continue
                with live_lock(job.process_id) as abandoned:
                    if abandoned is None:
                        continue
                    if job.provider_started:
                        exc = UncertainProviderOutcome(
                            "Finalizer interrupted after provider work began; automatic replay refused"
                        )
                    elif job.attempts >= Config.PIPELINE_MAX_ATTEMPTS:
                        exc = RuntimeError("Stage recovery attempts exhausted")
                    elif aware(job.deadline_at) <= timestamp:
                        exc = PipelineDeadlineExceeded("Pipeline deadline expired")
                    else:
                        job.state = "ready"
                        job.generation += 1
                        job.heartbeat_at = None
                        continue
                    _fail(run, job, exc)
                    failures.append((job.process_id, job.stage, job.generation, exc))
            elif aware(job.deadline_at) <= timestamp:
                exc = PipelineDeadlineExceeded("Pipeline deadline expired while queued")
                _fail(run, job, exc)
                failures.append((job.process_id, job.stage, job.generation, exc))
        db.flush()
        active = PipelineJob.state.in_(["dispatched", "running"])
        active_gpu, active_cpu, downstream = db.execute(
            select(
                func.count().filter(and_(PipelineJob.stage == "extraction", active)),
                func.count().filter(and_(PipelineJob.stage == "finalization", active)),
                func.count().filter(
                    and_(
                        PipelineJob.state != "terminal",
                        (PipelineJob.stage == "finalization") | active,
                    )
                ),
            ).where(PipelineJob.state != "terminal")
        ).one()
        for _, job in pairs:
            if job.state == "ready":
                if job.stage == "extraction":
                    if (
                        active_gpu >= Config.PIPELINE_GPU_ADMISSIONS
                        or downstream >= Config.PIPELINE_DOWNSTREAM_CAP
                    ):
                        continue
                    active_gpu += 1
                    downstream += 1
                else:
                    if active_cpu >= Config.PIPELINE_FINALIZERS:
                        continue
                    active_cpu += 1
                job.state = "dispatched"
                job.dispatched_at = timestamp
                messages.append((job.process_id, job.generation, job.stage))
            elif (
                job.state == "dispatched"
                and aware(job.dispatched_at)
                + timedelta(seconds=Config.PIPELINE_REPUBLISH_SECONDS)
                <= timestamp
            ):
                job.dispatched_at = timestamp
                messages.append((job.process_id, job.generation, job.stage))
    for process_id, stage, generation, exc in failures:
        capture_failure(exc, process_id=process_id, stage=stage, generation=generation)
    for process_id, generation, stage in messages:
        # Failure here leaves a durable dispatched row for the next republish pass.
        publish(process_id, generation, stage)
    return len(messages)


# One stage per OS worker process; provider calls may use a thread pool.
_active_job = None


def set_active_job(process_id=None, generation=None):
    global _active_job
    _active_job = (process_id, generation) if process_id else None


@contextmanager
def provider_slot():
    """Cross-process cap around the actual SDK call, including its retries."""
    if _active_job is None:
        yield
        return
    process_id, generation = _active_job
    connection = None
    slot = None
    try:
        try:
            connection = (
                get_engine().connect().execution_options(isolation_level="AUTOCOMMIT")
            )
            while slot is None:
                with session() as db:
                    owned(db, process_id, generation)
                for number in range(Config.PIPELINE_PROVIDER_CONCURRENCY):
                    if connection.execute(
                        text("SELECT pg_try_advisory_lock(765123, :slot)"),
                        {"slot": number},
                    ).scalar():
                        slot = number
                        break
                if slot is None:
                    # Local admission contention must not become a scientific
                    # provider timeout/fallback. owned() and the watchdog enforce
                    # the job deadline while this call waits for an available slot.
                    time.sleep(0.1)
            with session() as db:
                _, job = owned(db, process_id, generation)
                job.provider_started = True
        except StageControlError:
            raise
        except Exception as exc:
            raise ProviderAdmissionError(
                "Provider admission could not confirm current ownership and capacity"
            ) from exc
        import sentry_sdk

        with sentry_sdk.start_span(op="pdfx.provider", description="pdfx.provider"):
            yield
    except Exception as exc:
        if not isinstance(exc, OwnershipLost):
            capture_failure(
                exc, process_id=process_id, stage="provider", generation=generation
            )
        raise
    finally:
        if slot is not None:
            try:
                connection.execute(
                    text("SELECT pg_advisory_unlock(765123, :slot)"), {"slot": slot}
                )
            except Exception:
                connection.invalidate()
        if connection is not None:
            connection.close()
