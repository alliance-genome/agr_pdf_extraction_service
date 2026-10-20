"""Durable outbox reconciler; run as its own supervised service."""

import logging
import time
from app.pipeline_health import touch
from config import Config
from app.pipeline import dispatch_once
from pdfx_observability import initialize, capture_failure

logger = logging.getLogger(__name__)


def publish(process_id, generation, stage):
    from celery_app import celery

    task_id = (
        process_id
        if stage == "finalization"
        else f"{process_id}:extraction:{generation}"
    )
    celery.send_task(
        f"pdfx.{'extract' if stage == 'extraction' else 'finalize'}_stage",
        args=[process_id, generation],
        task_id=task_id,
        queue=f"pdfx.{stage}",
    )


def main():
    initialize("dispatcher")
    from app.logging_config import setup_logging

    setup_logging(component="dispatcher")
    last_error = None
    last_reported = 0.0
    while True:
        try:
            count = dispatch_once(publish)
            touch("dispatcher")
            last_error = None
            if count:
                logger.info("Published %d pipeline notifications", count)
        except Exception as exc:
            if (
                type(exc) is not last_error
                or time.monotonic() - last_reported
                >= Config.SENTRY_DISPATCHER_ERROR_INTERVAL_SECONDS
            ):
                capture_failure(exc, stage="dispatcher")
                last_error, last_reported = type(exc), time.monotonic()
            logger.error("Dispatcher pass failed: %s", type(exc).__name__)
        time.sleep(1)


if __name__ == "__main__":
    main()
