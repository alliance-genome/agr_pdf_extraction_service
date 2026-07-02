"""Start the production Celery worker after same-process model preloading."""

from celery_app import (
    _configure_torch_for_worker_process,
    _preload_marker_models_for_worker,
    celery,
)


def main(argv=None):
    """Preload GPU models, then start Celery in the same process.

    The GPU deployment uses Celery's solo pool so task execution happens in
    this process. Keeping preload and Celery startup together lets the first
    real Marker task reuse the in-memory converter cache.
    """
    _configure_torch_for_worker_process()
    _preload_marker_models_for_worker()
    celery.worker_main(argv or [
        "worker",
        "--loglevel=info",
        "--pool=solo",
        "-Q",
        "default",
    ])


if __name__ == "__main__":
    main()
