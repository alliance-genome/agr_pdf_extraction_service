"""Configuration loaded from environment variables."""

import os


class Settings:
    EC2_INSTANCE_ID: str = os.environ.get("EC2_INSTANCE_ID", "i-0105504873df917c0")
    EC2_REGION: str = os.environ.get("EC2_REGION", "us-east-1")
    EC2_PORT: int = int(os.environ.get("EC2_PORT", "5000"))

    COGNITO_USER_POOL_ID: str = os.environ.get("COGNITO_USER_POOL_ID", "us-east-1_d3eK6SYpI")
    COGNITO_REGION: str = os.environ.get("COGNITO_REGION", "us-east-1")
    COGNITO_REQUIRED_SCOPE: str = os.environ.get("COGNITO_REQUIRED_SCOPE", "pdfx-api/extract")

    IDLE_TIMEOUT_MINUTES: int = int(os.environ.get("IDLE_TIMEOUT_MINUTES", "30"))
    STARTUP_TIMEOUT_MINUTES: int = int(os.environ.get("STARTUP_TIMEOUT_MINUTES", "10"))
    HEALTH_POLL_INTERVAL_SECONDS: int = int(os.environ.get("HEALTH_POLL_INTERVAL_SECONDS", "15"))
    MAX_QUEUED_JOBS: int = int(os.environ.get("MAX_QUEUED_JOBS", "10"))

    FORWARD_TIMEOUT_SECONDS: int = int(os.environ.get("FORWARD_TIMEOUT_SECONDS", "600"))


settings = Settings()
