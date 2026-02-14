import os
import json
import logging
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)



def _cfg(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)



def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")



def build_s3_client(config):
    aws_access_key_id = _cfg(config, "AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key = _cfg(config, "AWS_SECRET_ACCESS_KEY", "")
    region = _cfg(config, "AWS_DEFAULT_REGION", "us-east-1")

    if not aws_access_key_id or not aws_secret_access_key:
        logger.warning("Audit logging disabled: missing AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY")
        return None

    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region,
    )



class AuditLogger:
    def __init__(self, process_id, config):
        self.process_id = str(process_id)
        self.config = config
        self.bucket = _cfg(config, "AUDIT_S3_BUCKET")
        self.prefix = (_cfg(config, "AUDIT_S3_PREFIX", "pdfx/audit") or "pdfx/audit").strip("/")

        now = datetime.now(timezone.utc)
        self._date_path = now.strftime("%Y/%m/%d")

        self._events = []
        self._flushed = False

        self._log_s3_key = "{prefix}/{date}/{process_id}.ndjson".format(
            prefix=self.prefix,
            date=self._date_path,
            process_id=self.process_id,
        )
        self._artifact_prefix = "{prefix}/{date}/{process_id}".format(
            prefix=self.prefix,
            date=self._date_path,
            process_id=self.process_id,
        )

        self.s3_client = build_s3_client(config)
        self.enabled = bool(self.s3_client and self.bucket)

        if not self.bucket:
            logger.warning("Audit logging disabled: AUDIT_S3_BUCKET is empty")
            self.enabled = False

    def log(self, stage, status, **kwargs):
        event = {
            "ts": utc_now_iso(),
            "stage": stage,
            "status": status,
        }
        for key, value in kwargs.items():
            if value is not None:
                event[key] = value
        self._events.append(event)

    def upload_artifact(self, filename, content, subdir=None):
        if not self.enabled:
            return None

        safe_filename = os.path.basename(filename)
        if not safe_filename:
            return None

        safe_subdir = ""
        if subdir:
            pieces = [os.path.basename(piece) for piece in str(subdir).split("/") if piece and piece != "."]
            if pieces:
                safe_subdir = "/".join(pieces)

        key_prefix = self._artifact_prefix
        if safe_subdir:
            key_prefix = "{prefix}/{subdir}".format(prefix=key_prefix, subdir=safe_subdir)

        artifact_key = "{prefix}/{filename}".format(prefix=key_prefix, filename=safe_filename)

        body = content
        if isinstance(content, str):
            body = content.encode("utf-8")

        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=artifact_key,
                Body=body,
            )
            return artifact_key
        except Exception as exc:
            logger.warning("Failed to upload audit artifact %s: %s", safe_filename, exc)
            return None

    def flush(self):
        if self._flushed:
            return

        if not self.enabled:
            self._flushed = True
            return

        try:
            lines = [json.dumps(event, separators=(",", ":")) for event in self._events]
            payload = "\n".join(lines)
            if payload:
                payload += "\n"

            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=self._log_s3_key,
                Body=payload.encode("utf-8"),
                ContentType="application/x-ndjson",
            )
            self._flushed = True
        except Exception as exc:
            logger.warning("Failed to flush audit log to S3 key %s: %s", self._log_s3_key, exc)

    def get_log_s3_key(self):
        return self._log_s3_key

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.flush()
