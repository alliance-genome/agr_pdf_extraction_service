"""Read-only access to authoritative extraction status in the existing RDS table."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from app.config import settings

logger = logging.getLogger(__name__)


class RDSStatusReader:
    """Perform one bounded, read-only lookup without waking the GPU backend."""

    def __init__(self, database_url: str | None = None):
        self._database_url = database_url if database_url is not None else settings.STATUS_DATABASE_URL

    def _connect(self):
        dsn = self._database_url.replace("postgresql+psycopg2://", "postgresql://", 1)
        timeout = max(1, settings.STATUS_DB_TIMEOUT_SECONDS)
        return psycopg2.connect(
            dsn,
            connect_timeout=timeout,
            options=f"-c statement_timeout={timeout * 1000} -c default_transaction_read_only=on",
        )

    def lookup(self, process_id: str) -> tuple[bool, dict[str, Any] | None]:
        if not self._database_url:
            return False, None
        try:
            with self._connect() as connection:
                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        SELECT process_id, reference_curie, mod_abbreviation,
                               source_pdf_md5, extract_images, review_images,
                               status, started_at, ended_at, error_code,
                               error_message, artifacts_json, log_s3_key,
                               consensus_metrics_json, llm_usage_json,
                               llm_cost_usd
                          FROM extraction_run
                         WHERE process_id = %s
                        """,
                        (process_id,),
                    )
                    row = cursor.fetchone()
            return True, self._to_payload(row) if row else None
        except Exception as exc:
            logger.warning("Authoritative status lookup unavailable for process_id=%s: %s", process_id, exc)
            return False, None

    def has_active_work(self) -> tuple[bool, bool | None]:
        """Return fail-closed shared RDS work evidence for destructive actions."""
        if not self._database_url:
            return False, None
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1
                              FROM extraction_run
                             WHERE status IN ('submitting', 'queued', 'running')
                        )
                        """
                    )
                    row = cursor.fetchone()
            return True, bool(row and row[0])
        except Exception as exc:
            logger.warning("Authoritative active-work lookup unavailable: %s", exc)
            return False, None

    @staticmethod
    def _to_payload(row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        payload["process_id"] = str(payload["process_id"])
        payload["status"] = {
            "submitting": "pending",
            "submission_failed": "pending",
            "queued": "pending",
            "running": "started",
            "succeeded": "complete",
        }.get(str(payload.get("status")), str(payload.get("status")))
        for key in ("started_at", "ended_at"):
            value = payload.get(key)
            if isinstance(value, datetime):
                payload[key] = value.isoformat().replace("+00:00", "Z")
        for key in ("artifacts_json", "consensus_metrics_json", "llm_usage_json"):
            value = payload.get(key)
            if isinstance(value, str):
                try:
                    payload[key] = json.loads(value)
                except ValueError:
                    pass
        if isinstance(payload.get("llm_cost_usd"), Decimal):
            payload["llm_cost_usd"] = float(payload["llm_cost_usd"])
        error_code = payload.pop("error_code", None)
        error_message = payload.pop("error_message", None)
        if error_code or error_message:
            payload["error_code"] = error_code
        if error_message:
            message = str(error_message)
            limit = max(1, settings.STATUS_ERROR_MESSAGE_MAX_CHARS)
            if len(message) > limit:
                omitted = len(message) - limit
                message = f"{message[:limit]}... [truncated {omitted} chars]"
            payload["error"] = message
        artifacts = payload.get("artifacts_json")
        if isinstance(artifacts, dict):
            payload["available_extractors"] = sorted(
                method for method in {"grobid", "docling", "marker"} if method in artifacts
            )
            images = artifacts.get("images")
            payload["image_count"] = len(images) if isinstance(images, list) else 0
        else:
            payload["available_extractors"] = []
            payload["image_count"] = 0
        return payload
