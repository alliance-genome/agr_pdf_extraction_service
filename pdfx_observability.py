"""Content-free error reporting shared by the API, proxy and staged workers.

Only reviewed structural fields leave this process. Detailed evidence remains in
PDFX's authenticated audit store; Sentry is an error index, not a PDF archive.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import logging
import math
import os
import re
from urllib.parse import urlsplit


STAGES = frozenset(
    {
        "submission",
        "queue",
        "extraction",
        "checkpoint",
        "finalization",
        "provider",
        "publication",
        "dispatcher",
        "proxy",
        "api",
    }
)
COMPONENTS = frozenset(
    {"api", "proxy", "extraction", "finalization", "dispatcher", "worker"}
)
_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.<>-]*$")
_HEX = re.compile(r"^[a-f0-9]{32,64}$")


def _identifier(value, fallback="redacted"):
    return (
        value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else fallback
    )


def sanitize_event(event, hint=None):
    """Rebuild an allowlisted envelope, including SDK-generated transactions."""
    safe = {
        key: event[key]
        for key in (
            "event_id",
            "timestamp",
            "start_timestamp",
            "platform",
            "level",
            "type",
            "release",
            "environment",
            "sdk",
        )
        if key in event
    }
    safe["server_name"] = "pdfx"
    tags = event.get("tags") or {}
    safe["tags"] = {}
    for name, allowed in (("pdfx.component", COMPONENTS), ("pdfx.stage", STAGES)):
        if tags.get(name) in allowed:
            safe["tags"][name] = tags[name]
    for name in ("pdfx.process_hash", "pdfx.source_sha256"):
        if isinstance(tags.get(name), str) and _HEX.fullmatch(tags[name]):
            safe["tags"][name] = tags[name]
    if (
        isinstance(tags.get("pdfx.generation"), (int, str))
        and str(tags["pdfx.generation"]).isdigit()
    ):
        safe["tags"]["pdfx.generation"] = str(tags["pdfx.generation"])
    values = []
    for exception in (event.get("exception") or {}).get("values", []):
        frames = []
        for frame in (exception.get("stacktrace") or {}).get("frames", []):
            # Do not send absolute filesystem paths, source context or locals.
            filename = (
                str(frame.get("filename", "")).replace("\\", "/").rsplit("/", 1)[-1]
            )
            frames.append(
                {
                    "filename": _identifier(filename),
                    "function": _identifier(frame.get("function")),
                    "lineno": frame.get("lineno")
                    if isinstance(frame.get("lineno"), int)
                    else None,
                    "in_app": bool(frame.get("in_app")),
                }
            )
        values.append(
            {
                "type": _identifier(exception.get("type")),
                "value": "PDFX operational failure (details in audit log)",
                "stacktrace": {"frames": frames},
            }
        )
    if values:
        safe["exception"] = {"values": values}
    else:
        safe["message"] = "PDFX operational event"
    trace = (event.get("contexts") or {}).get("trace") or {}
    safe["contexts"] = {
        "trace": {
            k: trace[k] for k in ("trace_id", "span_id", "parent_span_id") if k in trace
        }
    }
    if event.get("type") == "transaction":
        stage = tags.get("pdfx.stage")
        safe["transaction"] = f"pdfx.{stage}" if stage in STAGES else "pdfx.request"
        safe["transaction_info"] = {"source": "custom"}
        safe["spans"] = []
        for span in event.get("spans", []):
            cleaned = {
                k: span[k]
                for k in (
                    "trace_id",
                    "span_id",
                    "parent_span_id",
                    "start_timestamp",
                    "timestamp",
                )
                if k in span
            }
            op = span.get("op", "")
            cleaned["op"] = (
                op if op in {f"pdfx.{s}" for s in STAGES} else "pdfx.operation"
            )
            cleaned["description"] = cleaned["op"]
            safe["spans"].append(cleaned)
    return safe


def initialize(component):
    """No DSN means disabled. Explicit integration list prevents AI payload capture."""
    if component not in COMPONENTS:
        raise ValueError("Unknown PDFX Sentry component")
    if os.getenv("SENTRY_ENABLED", "true").lower() != "true":
        return False
    dsn = os.getenv("SENTRY_DSN", "").strip()
    secret_id = os.getenv("SENTRY_RUNTIME_SECRET_ID", "").strip()
    from_secret = bool(not dsn and secret_id)
    if from_secret:
        import boto3
        from botocore.config import Config as AWSConfig
        from botocore.exceptions import BotoCoreError, ClientError

        project = os.getenv("SENTRY_PROJECT", "").strip()
        if project not in {"pdfx-dev", "pdfx-prod"}:
            raise ValueError("Set SENTRY_PROJECT explicitly to pdfx-dev or pdfx-prod")
        timeout = float(os.getenv("SENTRY_SECRET_TIMEOUT_SECONDS", "2"))
        attempts = int(os.getenv("SENTRY_SECRET_MAX_ATTEMPTS", "1"))
        if not math.isfinite(timeout) or timeout <= 0 or attempts < 1:
            raise ValueError("Invalid Sentry secret timeout or attempt limit")
        try:
            response = boto3.client(
                "secretsmanager",
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                config=AWSConfig(
                    connect_timeout=timeout,
                    read_timeout=timeout,
                    retries={"total_max_attempts": attempts},
                ),
            ).get_secret_value(SecretId=secret_id)
            dsn = json.loads(response["SecretString"])["projects"][project][
                "dsn_private_vpc"
            ]
            if not isinstance(dsn, str):
                raise TypeError("Sentry secret DSN must be a string")
        except (BotoCoreError, ClientError, KeyError, ValueError, TypeError):
            # Reporting availability must not change extraction outcomes.
            # Do not log exception details, which can contain credential context.
            logging.getLogger(__name__).warning(
                "Sentry secret unavailable or malformed; reporting disabled for this process"
            )
            return False
    if not dsn:
        return False
    try:
        parsed = urlsplit(dsn)
        # urllib validates numeric/range constraints lazily, as the SDK does.
        _ = parsed.port
        if parsed.scheme != "https" and not (
            parsed.scheme == "http"
            and os.getenv("SENTRY_ALLOW_INSECURE_DSN", "false").lower() == "true"
        ):
            raise ValueError(
                "Sentry requires HTTPS or explicit private-ingestion HTTP opt-in"
            )
    except ValueError:
        if not from_secret:
            raise
        logging.getLogger(__name__).warning(
            "Sentry secret unavailable or malformed; reporting disabled for this process"
        )
        return False
    rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
    flush = float(os.getenv("SENTRY_SHUTDOWN_TIMEOUT_SECONDS", "2"))
    if not 0 <= rate <= 1 or not math.isfinite(flush) or flush <= 0:
        raise ValueError("Invalid Sentry sample rate or shutdown timeout")
    import sentry_sdk
    from sentry_sdk.utils import BadDsn
    from sentry_sdk.integrations.dedupe import DedupeIntegration
    from sentry_sdk.integrations.excepthook import ExcepthookIntegration
    from sentry_sdk.integrations.threading import ThreadingIntegration

    integrations = [
        DedupeIntegration(),
        ExcepthookIntegration(),
        ThreadingIntegration(),
    ]
    if component == "api":
        from sentry_sdk.integrations.flask import FlaskIntegration

        integrations.append(FlaskIntegration())
    elif component == "proxy":
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        # Handled HTTP failures belong to the proxy exception/response handlers.
        # SDK middleware still owns uncaught exceptions, preserving their frames.
        integrations.extend([
            FastApiIntegration(failed_request_status_codes=set()),
            StarletteIntegration(failed_request_status_codes=set()),
        ])
    elif component in {"worker", "extraction", "finalization"}:
        from sentry_sdk.integrations.celery import CeleryIntegration

        integrations.append(CeleryIntegration(monitor_beat_tasks=False))
    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=os.getenv("SENTRY_ENVIRONMENT", "local"),
            release=os.getenv("SENTRY_RELEASE", "pdfx@unknown"),
            server_name="pdfx",
            default_integrations=False,
            auto_enabling_integrations=False,
            integrations=integrations,
            send_default_pii=False,
            include_local_variables=False,
            include_source_context=False,
            max_request_body_size="never",
            max_breadcrumbs=0,
            enable_logs=False,
            traces_sample_rate=rate,
            shutdown_timeout=flush,
            trace_propagation_targets=[],
            before_send=sanitize_event,
            before_send_transaction=sanitize_event,
        )
    except BadDsn:
        if not from_secret:
            raise
        logging.getLogger(__name__).warning(
            "Sentry secret unavailable or malformed; reporting disabled for this process"
        )
        return False
    sentry_sdk.set_tag("pdfx.component", component)
    return True


@contextmanager
def stage_scope(process_id, stage, generation=None, source_sha256=None):
    import sentry_sdk

    if stage not in STAGES:
        raise ValueError("Unknown PDFX stage")
    with sentry_sdk.isolation_scope() as scope:
        scope.set_tag("pdfx.stage", stage)
        if process_id:
            scope.set_tag(
                "pdfx.process_hash",
                hashlib.sha256(str(process_id).encode()).hexdigest(),
            )
        if generation is not None:
            scope.set_tag("pdfx.generation", generation)
        if source_sha256:
            scope.set_tag("pdfx.source_sha256", source_sha256)
        with sentry_sdk.start_transaction(name=f"pdfx.{stage}", op=f"pdfx.{stage}"):
            yield


def capture_failure(exc, *, process_id=None, stage, generation=None):
    """Report caught failures; SDK deduplication prevents a later re-raise duplicate."""
    import sentry_sdk

    # Keep the active transaction for stage workers; do not create a trace on errors.
    with sentry_sdk.new_scope() as scope:
        if stage in STAGES:
            scope.set_tag("pdfx.stage", stage)
        if process_id:
            scope.set_tag(
                "pdfx.process_hash",
                hashlib.sha256(str(process_id).encode()).hexdigest(),
            )
        if generation is not None:
            scope.set_tag("pdfx.generation", generation)
        return sentry_sdk.capture_exception(exc)
