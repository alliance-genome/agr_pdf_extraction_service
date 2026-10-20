"""GPU-only extraction and CPU-only finalization on immutable checkpoints."""

from contextlib import contextmanager
import json
import fcntl
import logging
import os
from pathlib import Path
import shutil
import tempfile
import select
import subprocess
import threading
import sys
import time

from celery.exceptions import Ignore
from config import Config
from celery_app import celery
from app import pipeline
from app.pipeline_storage import PipelineStore
from app.services.audit_logger import AuditLogger
from pdfx_observability import stage_scope, capture_failure

logger = logging.getLogger(__name__)


class Progress:
    def __init__(self, process_id, generation):
        self.process_id, self.generation = process_id, generation

    def update_state(self, state, meta):
        pipeline.heartbeat(self.process_id, self.generation, meta)


class Watchdog:
    """A separate process can enforce deadlines even inside a hung native call."""

    def __init__(self, process_id, generation, stage):
        self.arguments = [
            sys.executable,
            "-m",
            "app.pipeline_watchdog",
            process_id,
            str(generation),
            stage,
        ]
        self.process = None
        self.expected_exit = threading.Event()

    def __enter__(self):
        self.process = subprocess.Popen(
            self.arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )
        ready = select.select(
            [self.process.stdout], [], [], Config.PIPELINE_WATCHDOG_STARTUP_SECONDS
        )[0]
        if not ready or self.process.stdout.readline() != b"READY\n":
            self.__exit__()
            raise RuntimeError("Stage ownership watchdog did not become ready")

        def monitor_exit():
            self.process.wait()
            if not self.expected_exit.is_set():
                os._exit(70)

        threading.Thread(target=monitor_exit, daemon=True).start()
        return self

    def __exit__(self, *args):
        if self.process is None:
            return
        self.expected_exit.set()
        self.process.stdin.close()
        try:
            self.process.wait(timeout=Config.PIPELINE_DB_TIMEOUT_SECONDS * 2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


@contextmanager
def workspace(process_id, generation):
    root = Path(Config.PIPELINE_WORKSPACE)
    root.mkdir(parents=True, exist_ok=True)
    # Local OS locks distinguish active attempts from leftovers after SIGKILL.
    # Serialize creation with cleanup so a new directory is never mistaken for
    # an abandoned attempt before it has acquired its lease.
    with (root / ".cleanup.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        for abandoned in root.glob("job-*"):
            lease_path = abandoned / ".lease"
            if (
                not abandoned.is_dir()
                or abandoned.is_symlink()
                or not lease_path.exists()
            ):
                continue
            with lease_path.open("a") as old_lease:
                try:
                    fcntl.flock(old_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                shutil.rmtree(abandoned)
        parent = Path(
            tempfile.mkdtemp(prefix=f"job-{process_id}-{generation}-", dir=root)
        )
        lease = (parent / ".lease").open("a")
        fcntl.flock(lease, fcntl.LOCK_EX)
    path = parent / "workspace"
    previous = Config.CACHE_FOLDER
    Config.CACHE_FOLDER = str(path / "cache")
    try:
        yield path
    finally:
        Config.CACHE_FOLDER = previous
        with (root / ".cleanup.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            shutil.rmtree(parent)
            lease.close()


def _run_stage(process_id, generation, stage):
    import celery_app as processing

    with pipeline.live_lock(process_id) as connection:
        if connection is None:
            raise Ignore()
        job = pipeline.claim(process_id, generation, stage)
        if job is None:
            raise Ignore()
        options = job["options"]
        shared = {"llm": None}
        audit = None
        pipeline.set_active_job(process_id, generation)
        from app.logging_config import set_job_context

        set_job_context(process_id, stage, generation)
        try:
            with (
                stage_scope(process_id, stage, generation, job["source"]["sha256"]),
                workspace(process_id, generation) as path,
            ):
                store = PipelineStore()
                audit = AuditLogger(
                    process_id, Config, attempt_id=f"{stage}-{generation}"
                )
                # Fenced generations have separate artifact destinations. Only the
                # winning DB transaction exposes them to download endpoints.
                audit._artifact_prefix += f"/generation-{generation}"
                if not audit.enabled:
                    raise RuntimeError("Durable audit/artifact storage is unavailable")
                progress = Progress(process_id, generation)
                elapsed = (
                    pipeline.now() - pipeline.aware(job["created_at"])
                ).total_seconds()
                started = time.monotonic() - elapsed
                common = dict(
                    audit_logger=audit,
                    extract_images=options["extract_images"],
                    review_images=options["review_images"],
                    _shared=shared,
                    job_started_monotonic=started,
                )
                with Watchdog(process_id, generation, stage):
                    if stage == "extraction":
                        path.mkdir()
                        Path(Config.CACHE_FOLDER).mkdir()
                        pdf = store.fetch_file(job["source"], path / "source.pdf")
                        source_hash = processing._get_file_hash(pdf)
                        sources = processing._extract_sources(
                            progress,
                            str(pdf),
                            options["methods"],
                            options["merge"],
                            file_hash=source_hash,
                            **common,
                        )
                        sources["file_hash"] = source_hash
                        (path / "sources.json").write_text(json.dumps(sources))
                        with stage_scope(process_id, "checkpoint", generation):
                            checkpoint = store.checkpoint(path, process_id, generation)
                        audit.log("checkpoint", "completed")
                        audit.flush(required=True)
                    else:
                        sources = store.restore(job["checkpoint"], path)
                        pdf = path / "source.pdf"
                        result = processing._finalize_sources(
                            progress,
                            str(pdf),
                            options["methods"],
                            options["merge"],
                            file_hash=sources["file_hash"],
                            source_state=sources,
                            **common,
                        )
                        pipeline.heartbeat(process_id, generation)
                        artifacts = processing._upload_artifacts(
                            audit, result, options["merge"], pdf_path=str(pdf)
                        )
                        required = {"source_pdf", *sources["methods_used"]}
                        if (
                            not required <= artifacts.keys()
                            or not set(sources["methods_used"])
                            <= artifacts.get("native", {}).keys()
                        ):
                            raise RuntimeError(
                                "Required source artifacts were not durably published"
                            )
                        if len(artifacts.get("images", [])) != len(
                            result.get("images", [])
                        ):
                            raise RuntimeError(
                                "Required image artifacts were not durably published"
                            )
                        result.update(
                            process_id=process_id,
                            artifacts_json=artifacts,
                            reference_curie=options.get("reference_curie"),
                            started_at=pipeline.aware(job["created_at"]).isoformat(),
                            ended_at=pipeline.now().isoformat(),
                            log_s3_key=audit.get_log_s3_key(),
                        )
                        audit.log("finalize", "completed")
                        audit.flush(required=True)
                if stage == "extraction":
                    pipeline.handoff(process_id, generation, checkpoint)
                    return None
                pipeline.complete(
                    process_id,
                    generation,
                    result,
                    {
                        "source_pdf_md5": sources["file_hash"],
                        "artifacts_json": artifacts,
                        "log_s3_key": audit.get_log_s3_key(),
                        "consensus_metrics_json": result.get("consensus_metrics"),
                        "llm_usage_json": result.get("llm_usage_json"),
                        "llm_cost_usd": result.get("llm_cost_usd"),
                    },
                )
                return result
        except pipeline.OwnershipLost:
            raise Ignore()
        except Exception as exc:
            values = {}
            if shared.get("llm") is not None:
                cost, usage = processing._compute_llm_accounting(shared["llm"])
                values.update(llm_cost_usd=cost, llm_usage_json=usage)
            if audit:
                values["log_s3_key"] = audit.get_log_s3_key()
                try:
                    audit.log(stage, "failed", error_code=type(exc).__name__)
                    audit.flush()
                except Exception as audit_exc:
                    capture_failure(
                        audit_exc,
                        process_id=process_id,
                        stage="publication",
                        generation=generation,
                    )
                    logger.error(
                        "Failed to flush failure audit: %s", type(audit_exc).__name__
                    )
            if not pipeline.fail(process_id, generation, exc, values):
                raise Ignore()
            return {
                "status": "failed",
                "process_id": process_id,
                "error": str(exc)[: Config.PIPELINE_ERROR_MESSAGE_MAX_CHARS],
            }
        finally:
            pipeline.set_active_job()
            set_job_context()


@celery.task(name="pdfx.extract_stage", ignore_result=True)
def extract_stage(process_id, generation):
    return _run_stage(process_id, generation, "extraction")


@celery.task(name="pdfx.finalize_stage")
def finalize_stage(process_id, generation):
    return _run_stage(process_id, generation, "finalization")
