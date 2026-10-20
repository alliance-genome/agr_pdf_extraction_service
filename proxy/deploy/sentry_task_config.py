"""Add non-secret Sentry configuration to a rendered ECS task definition."""

import json
import os
import sys


def configure(task, secret_id, project, environment, release):
    if secret_id == "disabled":
        secret_id = ""
    if secret_id and (project not in {"pdfx-dev", "pdfx-prod"} or not environment):
        raise ValueError(
            "Configured Sentry requires an explicit PDFX project and environment"
        )
    values = {
        "SENTRY_RUNTIME_SECRET_ID": secret_id,
        "SENTRY_PROJECT": project,
        "SENTRY_ENVIRONMENT": environment or "local",
        "SENTRY_RELEASE": f"pdfx@{release}",
        "SENTRY_ALLOW_INSECURE_DSN": "true" if secret_id else "false",
    }
    for container in task["containerDefinitions"]:
        container["environment"] = [
            v for v in container.get("environment", []) if v["name"] not in values
        ]
        container["environment"].extend(
            {"name": k, "value": v} for k, v in values.items()
        )
    return task


if __name__ == "__main__":
    result = configure(
        json.load(sys.stdin),
        os.environ.get("PDFX_SENTRY_SECRET", ""),
        os.environ.get("PDFX_SENTRY_PROJECT", ""),
        os.environ.get("PDFX_SENTRY_ENVIRONMENT", ""),
        os.environ["PDFX_SENTRY_RELEASE"],
    )
    json.dump(result, sys.stdout)
