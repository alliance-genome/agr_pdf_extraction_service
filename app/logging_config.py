import logging
import os

NOISY_LOGGERS = {
    "transformers": logging.ERROR,
    "torch": logging.ERROR,
    "tqdm": logging.ERROR,
    "urllib3": logging.WARNING,
    "PIL": logging.WARNING,
    "onnxruntime": logging.WARNING,
    "docling": logging.WARNING,
    "marker": logging.WARNING,
    "openai": logging.WARNING,
}

_configured = False

# Singleton filter instance — attached to every handler we can find.
_celery_filter = None


class _CeleryRedirectedFilter(logging.Filter):
    """Rewrite ``celery.redirected`` log records so they look clean in GELF.

    Celery captures third-party stdout/stderr (tqdm progress bars, etc.) into
    a logger named ``celery.redirected`` at WARNING level.  This filter:
    1. Renames the logger to ``pdfx.worker.stdio`` so GELF shows a proper source.
    2. Downgrades the level from WARNING to INFO (progress bars aren't warnings).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "celery.redirected":
            record.name = "pdfx.worker.stdio"
            record.levelno = logging.INFO
            record.levelname = "INFO"
        return True


def _install_filter_on_all_handlers():
    """Add _CeleryRedirectedFilter to every handler on the root logger.

    Called from setup_logging() and also from the Celery after_setup_logger
    signal so the filter covers handlers Celery adds after our initial setup.
    """
    global _celery_filter
    if _celery_filter is None:
        _celery_filter = _CeleryRedirectedFilter()

    root = logging.getLogger()
    for handler in root.handlers:
        if _celery_filter not in handler.filters:
            handler.addFilter(_celery_filter)


class MergingLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that merges per-call extra with base extra.

    Python's default LoggerAdapter.process() overwrites call-time extra with
    self.extra. This subclass merges both so per-job identity fields (from
    self.extra) and per-event fields (from the log call) are both emitted as
    GELF additional fields.
    """

    def process(self, msg, kwargs):
        extra = {**self.extra, **kwargs.get("extra", {})}
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging(component="worker"):
    """Configure structured GELF logging for the PDF extraction service.

    Args:
        component: "worker" or "app" — identifies the source container in GELF.
    """
    global _configured
    if _configured:
        return
    _configured = True

    gelf_enabled = os.environ.get("GELF_ENABLED", "false").lower() == "true"

    root = logging.getLogger()
    # Set root to DEBUG so per-publication file handlers can capture everything.
    # Console handler filters to INFO; GELF handler filters to INFO.
    root.setLevel(logging.DEBUG)

    # Console handler (keep for local dev and docker logs)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root.addHandler(console)

    # GELF handler (UDP)
    if gelf_enabled:
        try:
            from pygelf import GelfUdpHandler

            gelf_host = os.environ.get("GELF_HOST", "logs.alliancegenome.org")
            gelf_port = int(os.environ.get("GELF_PORT", "12201"))

            gelf_handler = GelfUdpHandler(
                host=gelf_host,
                port=gelf_port,
                include_extra_fields=True,
                compress=True,
                _service="pdfx",
                _component=component,
                _container_name=f"pdfx-{component}",
            )
            # Override pygelf's default domain (container ID hash) so Graylog
            # shows "pdfx-worker" or "pdfx-app" as the source instead of "unknown".
            gelf_handler.domain = f"pdfx-{component}"
            root.addHandler(gelf_handler)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to configure GELF handler, continuing with console only: %s", exc
            )

    # Install celery.redirected filter on all handlers (ours + any Celery added)
    _install_filter_on_all_handlers()

    # Suppress noisy third-party loggers
    for logger_name, level in NOISY_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(level)

    # Hook into Celery's logging setup so the filter covers handlers added later.
    # This import is safe even outside a Celery context — it's a no-op if Celery
    # isn't driving the process.
    try:
        from celery.signals import after_setup_logger, after_setup_task_logger

        @after_setup_logger.connect
        def _on_celery_logger_setup(**kwargs):
            _install_filter_on_all_handlers()

        @after_setup_task_logger.connect
        def _on_celery_task_logger_setup(**kwargs):
            _install_filter_on_all_handlers()
    except ImportError:
        pass
