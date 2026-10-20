"""CPU-only worker: no GPU model preload and no GPU task queue."""

from config import Config
from pdfx_observability import initialize


def main():
    import os

    os.environ["PDFX_WORKER_ROLE"] = "finalization"
    initialize("finalization")
    from celery_app import celery

    celery.worker_main(
        [
            "worker",
            "--loglevel=info",
            "--pool=prefork",
            f"--concurrency={Config.PIPELINE_FINALIZERS}",
            "-Q",
            "pdfx.finalization",
        ]
    )


if __name__ == "__main__":
    main()
