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
    root.setLevel(logging.INFO)

    # Console handler (keep for local dev and docker logs)
    console = logging.StreamHandler()
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
            )
            root.addHandler(gelf_handler)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to configure GELF handler, continuing with console only: %s", exc
            )

    # Suppress noisy third-party loggers
    for logger_name, level in NOISY_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(level)
