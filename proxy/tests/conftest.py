"""Shared test fixtures — set required env vars before any app import."""

import os

# These must be set before config.py is imported (module-level os.environ[])
os.environ.setdefault("EC2_INSTANCE_ID", "i-test-instance")
os.environ.setdefault("COGNITO_USER_POOL_ID", "us-east-1_TestPool")
