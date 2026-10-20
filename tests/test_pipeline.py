"""Real PostgreSQL concurrency/recovery tests; opt in with an isolated test URL."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import os
import uuid

import pytest
from sqlalchemy import delete
from app import pipeline
from app.models import ExtractionRun, PipelineJob, reset_db_engine
from config import Config


@pytest.fixture
def jobs(monkeypatch):
    url = os.getenv("PDFX_TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "Set PDFX_TEST_DATABASE_URL to an isolated, migrated PostgreSQL database"
        )
    monkeypatch.setenv("DATABASE_URL", url)
    reset_db_engine()
    created = []

    def make():
        pid = str(uuid.uuid4())
        created.append(pid)
        claimed = pipeline.now()
        with pipeline.session() as db:
            db.add(
                ExtractionRun(process_id=pid, status="submitting", started_at=claimed)
            )
        pipeline.accept(
            pid,
            {"sha256": "a" * 64, "size": 1, "key": "unused"},
            {"methods": ["marker"]},
            claimed,
        )
        return pid

    yield make
    with pipeline.session() as db:
        db.execute(delete(PipelineJob).where(PipelineJob.process_id.in_(created)))
        db.execute(delete(ExtractionRun).where(ExtractionRun.process_id.in_(created)))
    reset_db_engine()


def dispatch():
    messages = []
    pipeline.dispatch_once(lambda *args: messages.append(args))
    return messages


def test_duplicate_dispatchers_single_gpu_admission(jobs):
    ids = [jobs() for _ in range(3)]
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: dispatch(), range(2)))
    assert sum(map(len, results)) == 1
    assert sum(pipeline.snapshot(i)["state"] == "dispatched" for i in ids) == 1
    pid, generation, stage = next(r[0] for r in results if r)
    with ThreadPoolExecutor(2) as pool:
        claims = list(
            pool.map(lambda _: pipeline.claim(pid, generation, stage), range(2))
        )
    assert sum(c is not None for c in claims) == 1


def test_outbox_publish_failure_republishes_same_generation(jobs, monkeypatch):
    pid = jobs()

    def unavailable(*args):
        raise ConnectionError("Redis unavailable")

    with pytest.raises(ConnectionError):
        pipeline.dispatch_once(unavailable)
    job = pipeline.snapshot(pid)
    assert job["state"] == "dispatched"
    with pipeline.session() as db:
        db.get(PipelineJob, pid).dispatched_at -= timedelta(
            seconds=Config.PIPELINE_REPUBLISH_SECONDS + 1
        )
    assert dispatch() == [(pid, 1, "extraction")]


def test_handoff_overlap_backpressure_and_cancel_fence(jobs, monkeypatch):
    monkeypatch.setattr(Config, "PIPELINE_DOWNSTREAM_CAP", 2)
    first, second, third = jobs(), jobs(), jobs()
    assert dispatch()[0][0] == first
    assert pipeline.claim(first, 1, "extraction")
    pipeline.handoff(first, 1, {"key": "checkpoint"})
    messages = dispatch()
    assert set((p, s) for p, g, s in messages) == {
        (first, "finalization"),
        (second, "extraction"),
    }
    assert pipeline.claim(first, 2, "finalization")
    assert pipeline.claim(second, 1, "extraction")
    pipeline.handoff(second, 1, {"key": "checkpoint"})
    dispatch()
    assert pipeline.snapshot(third)["state"] == "ready"
    assert pipeline.cancel(first, "cancelled")
    with pytest.raises(pipeline.OwnershipLost):
        pipeline.complete(first, 2, {}, {})
    assert any(p == third for p, g, s in dispatch())


def test_lost_owner_after_provider_is_terminal_not_replayed(jobs):
    pid = jobs()
    dispatch()
    pipeline.claim(pid, 1, "extraction")
    pipeline.handoff(pid, 1, {})
    dispatch()
    pipeline.claim(pid, 2, "finalization")
    with pipeline.session() as db:
        job = db.get(PipelineJob, pid)
        job.provider_started = True
        job.heartbeat_at -= timedelta(seconds=Config.PIPELINE_LEASE_SECONDS + 1)
    assert dispatch() == []
    job = pipeline.snapshot(pid)
    assert job["state"] == "terminal"
    with pipeline.session() as db:
        assert db.get(ExtractionRun, pid).error_code == "UncertainProviderOutcome"


def test_live_lock_prevents_recovery_and_safe_recovery_fences_old_owner(jobs):
    pid = jobs()
    dispatch()
    pipeline.claim(pid, 1, "extraction")
    with pipeline.session() as db:
        db.get(PipelineJob, pid).heartbeat_at -= timedelta(
            seconds=Config.PIPELINE_LEASE_SECONDS + 1
        )
    with pipeline.live_lock(pid) as owner:
        assert owner is not None
        assert dispatch() == []
        assert pipeline.snapshot(pid)["generation"] == 1
    assert dispatch() == [(pid, 2, "extraction")]
    with pytest.raises(pipeline.OwnershipLost):
        pipeline.handoff(pid, 1, {})


def test_original_deadline_applies_to_queued_stage(jobs):
    pid = jobs()
    with pipeline.session() as db:
        db.get(PipelineJob, pid).deadline_at = pipeline.now() - timedelta(seconds=1)
    assert dispatch() == []
    assert pipeline.snapshot(pid)["state"] == "terminal"


def test_completion_is_immutable_and_input_durable_before_dispatch(jobs):
    pid = jobs()
    assert pipeline.snapshot(pid)["source"]["sha256"] == "a" * 64
    dispatch()
    pipeline.claim(pid, 1, "extraction")
    pipeline.handoff(pid, 1, {})
    dispatch()
    pipeline.claim(pid, 2, "finalization")
    pipeline.complete(pid, 2, {"status": "success"}, {})
    assert pipeline.fail(pid, 2, RuntimeError("late failure")) is False
    assert pipeline.cancel(pid, "late cancel") is False
    assert pipeline.snapshot(pid)["result"] == {"status": "success"}


def _provider_child(url, pid, generation, shared, gate):
    import os, time

    os.environ["DATABASE_URL"] = url
    os.environ["PIPELINE_PROVIDER_CONCURRENCY"] = "2"
    from config import Config

    Config.PIPELINE_PROVIDER_CONCURRENCY = 2
    from app import pipeline
    from app.models import reset_db_engine

    reset_db_engine()
    pipeline.set_active_job(pid, generation)
    gate.wait()
    for _ in range(2):
        with pipeline.provider_slot():
            with shared["lock"]:
                shared["counts"]["active"] += 1
                shared["counts"]["maximum"] = max(
                    shared["counts"]["maximum"], shared["counts"]["active"]
                )
            time.sleep(0.2)
            with shared["lock"]:
                shared["counts"]["active"] -= 1
    reset_db_engine()


def test_provider_budget_is_global_across_processes(jobs):
    import multiprocessing

    pid = jobs()
    dispatch()
    pipeline.claim(pid, 1, "extraction")
    pipeline.handoff(pid, 1, {})
    dispatch()
    pipeline.claim(pid, 2, "finalization")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager:
        shared = {"lock": manager.Lock(), "counts": manager.dict(active=0, maximum=0)}
        gate = ctx.Barrier(4)
        workers = [
            ctx.Process(
                target=_provider_child,
                args=(os.environ["PDFX_TEST_DATABASE_URL"], pid, 2, shared, gate),
            )
            for _ in range(4)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(15)
            if worker.is_alive():
                worker.kill()
                worker.join()
                pytest.fail("Provider budget test child stalled")
            assert worker.exitcode == 0
        assert shared["counts"]["maximum"] == 2
        assert shared["counts"]["active"] == 0
    assert pipeline.snapshot(pid)["provider_started"] is True


def test_interrupted_finalizer_before_provider_reuses_checkpoint(jobs):
    pid = jobs()
    dispatch()
    pipeline.claim(pid, 1, "extraction")
    pipeline.handoff(pid, 1, {"key": "durable-checkpoint"})
    dispatch()
    pipeline.claim(pid, 2, "finalization")
    with pipeline.session() as db:
        db.get(PipelineJob, pid).heartbeat_at -= timedelta(
            seconds=Config.PIPELINE_LEASE_SECONDS + 1
        )
    assert dispatch() == [(pid, 3, "finalization")]
    assert pipeline.snapshot(pid)["checkpoint"] == {"key": "durable-checkpoint"}
    assert pipeline.claim(pid, 2, "finalization") is None
    assert pipeline.claim(pid, 3, "finalization")


def _native_hang_child(url, pid, ready=None):
    import os, ctypes

    os.environ["DATABASE_URL"] = url
    from app import pipeline
    from app.models import reset_db_engine
    from app.pipeline_tasks import Watchdog

    reset_db_engine()
    with pipeline.live_lock(pid) as owner:
        assert owner is not None
        assert pipeline.claim(pid, 1, "extraction")
        with Watchdog(pid, 1, "extraction") as watchdog:
            if ready is not None:
                ready.send(watchdog.process.pid)
                ready.close()
            # PyDLL deliberately holds the GIL across this native call. A thread
            # watchdog cannot enforce this deadline; a separate process must.
            ctypes.PyDLL(None).sleep(60)


def test_external_watchdog_kills_native_hang_and_persists_deadline(jobs):
    import multiprocessing

    pid = jobs()
    dispatch()
    with pipeline.session() as db:
        db.get(PipelineJob, pid).deadline_at = pipeline.now() + timedelta(seconds=3)
    child = multiprocessing.get_context("spawn").Process(
        target=_native_hang_child, args=(os.environ["PDFX_TEST_DATABASE_URL"], pid)
    )
    child.start()
    child.join(20)
    if child.is_alive():
        child.kill()
        child.join()
        pytest.fail("External deadline monitor did not stop native hang")
    assert child.exitcode == -9
    with pipeline.session() as db:
        run = db.get(ExtractionRun, pid)
        assert run.status == "failed"
        assert run.error_code == "PipelineDeadlineExceeded"
    with pipeline.live_lock(pid) as owner:
        assert owner is not None


@pytest.mark.parametrize("target", ["watchdog", "sentinel"])
def test_monitor_death_stops_worker_holding_native_gil(jobs, target):
    import multiprocessing
    import signal
    import time
    from pathlib import Path

    pid = jobs()
    dispatch()
    ctx = multiprocessing.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    child = ctx.Process(
        target=_native_hang_child,
        args=(os.environ["PDFX_TEST_DATABASE_URL"], pid, writer),
    )
    child.start()
    writer.close()
    try:
        assert reader.poll(15), "Watchdog startup failed"
        watchdog_pid = reader.recv()
        target_pid = watchdog_pid
        if target == "sentinel":
            children = (
                Path(f"/proc/{watchdog_pid}/task/{watchdog_pid}/children")
                .read_text()
                .split()
            )
            assert len(children) == 1
            target_pid = int(children[0])
        time.sleep(0.3)  # allow the child to enter the native GIL-holding sleep
        os.kill(target_pid, signal.SIGKILL)
        child.join(15)
        assert not child.is_alive(), "Worker survived loss of stage supervision"
        assert child.exitcode == -9
        with pipeline.live_lock(pid) as owner:
            assert owner is not None
    finally:
        reader.close()
        if child.is_alive():
            child.kill()
            child.join()


def test_watchdog_ready_despite_sentry_secret_outage(jobs, monkeypatch):
    from app.pipeline_tasks import Watchdog

    # Exercise the real subprocess and boto client against a closed local port.
    for key, value in {
        "SENTRY_ENABLED": "true",
        "SENTRY_DSN": "",
        "SENTRY_RUNTIME_SECRET_ID": "unavailable-reporting",
        "SENTRY_PROJECT": "pdfx-dev",
        "AWS_ACCESS_KEY_ID": "synthetic",
        "AWS_SECRET_ACCESS_KEY": "synthetic",
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_ENDPOINT_URL_SECRETS_MANAGER": "http://127.0.0.1:1",
    }.items():
        monkeypatch.setenv(key, value)
    pid = jobs()
    dispatch()
    with pipeline.live_lock(pid):
        assert pipeline.claim(pid, 1, "extraction")
        with Watchdog(pid, 1, "extraction") as watchdog:
            assert watchdog.process.poll() is None
            assert pipeline.snapshot(pid)["state"] == "running"


def test_claim_after_deadline_reports_failure_once(jobs, monkeypatch):
    reports = []
    monkeypatch.setattr(
        pipeline, "capture_failure", lambda exc, **kw: reports.append(kw)
    )
    pid = jobs()
    dispatch()
    with pipeline.session() as db:
        db.get(PipelineJob, pid).deadline_at = pipeline.now() - timedelta(seconds=1)
    assert pipeline.claim(pid, 1, "extraction") is None
    assert pipeline.claim(pid, 1, "extraction") is None
    assert reports == [{"process_id": pid, "stage": "extraction", "generation": 1}]


def test_runtime_permission_preflight_rejects_read_only_role(jobs):
    from deploy.check_pipeline_database import check_pipeline_database
    from app.models import get_engine
    from sqlalchemy import text

    role = "pipeline_test_" + uuid.uuid4().hex
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(text(f"CREATE ROLE {role}"))
            connection.execute(
                text(f"GRANT SELECT ON extraction_run, pipeline_job TO {role}")
            )
            connection.execute(text(f"SET LOCAL ROLE {role}"))
            with pytest.raises(RuntimeError, match="pipeline_job:INSERT"):
                check_pipeline_database(connection)
            connection.execute(text("RESET ROLE"))
            connection.execute(
                text(
                    f"GRANT INSERT,UPDATE,DELETE ON extraction_run,pipeline_job TO {role}"
                )
            )
            connection.execute(text(f"SET LOCAL ROLE {role}"))
            check_pipeline_database(connection)
        finally:
            transaction.rollback()
