import os
import json
import logging
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError

logger = logging.getLogger(__name__)



def _cfg(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)



def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_bucket_name(config):
    """Resolve the S3 bucket name: env var first, then SSM Parameter Store."""
    bucket = _cfg(config, "AUDIT_S3_BUCKET", "")
    if bucket:
        return bucket

    ssm_param = _cfg(config, "AUDIT_S3_BUCKET_SSM_PARAM", "/pdfx/audit-s3-bucket")
    if not ssm_param:
        return ""

    region = _cfg(config, "AWS_DEFAULT_REGION", "us-east-1")
    try:
        ssm = boto3.client("ssm", region_name=region)
        resp = ssm.get_parameter(Name=ssm_param)
        bucket = resp["Parameter"]["Value"]
        logger.info("Resolved AUDIT_S3_BUCKET from SSM %s: %s", ssm_param, bucket)
        return bucket
    except Exception as exc:
        logger.warning("Failed to read SSM parameter %s: %s", ssm_param, exc)
        return ""


def build_s3_client(config=None):
    """Build an S3 client using the default boto3 credential chain.

    On EC2 with an IAM instance profile, credentials are provided
    automatically.  Returns None if no credentials are available
    (local dev without AWS config).
    """
    region = "us-east-1"
    if config is not None:
        region = _cfg(config, "AWS_DEFAULT_REGION", "us-east-1")

    try:
        session = boto3.Session(region_name=region)
        credentials = session.get_credentials()
        if credentials is None:
            logger.warning("Audit S3 client unavailable: no AWS credentials found")
            return None
        return session.client("s3")
    except (BotoCoreError, NoCredentialsError) as exc:
        logger.warning("Audit S3 client unavailable: %s", exc)
        return None



class AuditLogger:
    def __init__(self, process_id, config):
        self.process_id = str(process_id)
        self.config = config
        self.bucket = _resolve_bucket_name(config)
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
            logger.warning("Audit logging disabled: no AUDIT_S3_BUCKET (env or SSM)")
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
