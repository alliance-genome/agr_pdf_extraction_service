"""Configuration loaded from environment variables.

All infrastructure-specific values (instance IDs, Cognito pool IDs, etc.)
are stored in AWS SSM Parameter Store under /pdfx/* and injected into the
container at launch by ECS via the task definition's "secrets" block.
"""

import os


class Settings:
    EC2_INSTANCE_ID: str = os.environ["EC2_INSTANCE_ID"]
    EC2_REGION: str = os.environ.get("EC2_REGION", "us-east-1")
    EC2_PORT: int = int(os.environ.get("EC2_PORT", "5000"))

    COGNITO_USER_POOL_ID: str = os.environ["COGNITO_USER_POOL_ID"]
    COGNITO_REGION: str = os.environ.get("COGNITO_REGION", "us-east-1")
    COGNITO_REQUIRED_SCOPE: str = os.environ.get("COGNITO_REQUIRED_SCOPE", "pdfx-api/extract")

    IDLE_TIMEOUT_MINUTES: int = int(os.environ.get("IDLE_TIMEOUT_MINUTES", "30"))
    MIN_UPTIME_MINUTES: int = int(os.environ.get("MIN_UPTIME_MINUTES", "20"))
    STARTUP_TIMEOUT_MINUTES: int = int(os.environ.get("STARTUP_TIMEOUT_MINUTES", "10"))
    HEALTH_POLL_INTERVAL_SECONDS: int = int(os.environ.get("HEALTH_POLL_INTERVAL_SECONDS", "15"))
    MAX_QUEUED_JOBS: int = int(os.environ.get("MAX_QUEUED_JOBS", "10"))

    FORWARD_TIMEOUT_SECONDS: int = int(os.environ.get("FORWARD_TIMEOUT_SECONDS", "600"))
    ALWAYS_ON_MODE: bool = os.environ.get("ALWAYS_ON_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}

    QUEUE_BACKEND: str = os.environ.get("QUEUE_BACKEND", "memory").strip().lower()
    QUEUE_S3_BUCKET: str = os.environ.get("QUEUE_S3_BUCKET", "").strip()
    QUEUE_S3_PREFIX: str = os.environ.get("QUEUE_S3_PREFIX", "pdfx-proxy-queue").strip().strip("/")
    QUEUE_S3_REGION: str = os.environ.get("QUEUE_S3_REGION", "").strip()

    STUCK_PENDING_MINUTES: int = int(os.environ.get("STUCK_PENDING_MINUTES", "20"))
    RECONCILER_INTERVAL_SECONDS: int = int(os.environ.get("RECONCILER_INTERVAL_SECONDS", "60"))
    RECONCILER_REQUEUE_ONCE: bool = os.environ.get("RECONCILER_REQUEUE_ONCE", "false").strip().lower() in {"1", "true", "yes", "on"}

    HEALTHCHECK_BEARER_TOKEN: str = os.environ.get("HEALTHCHECK_BEARER_TOKEN", "").strip()

    CANARY_INTERVAL_SECONDS: int = int(os.environ.get("CANARY_INTERVAL_SECONDS", "0"))
    CANARY_BEARER_TOKEN: str = os.environ.get("CANARY_BEARER_TOKEN", "").strip()


settings = Settings()
