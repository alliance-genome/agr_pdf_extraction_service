"""Scheduled PDFX GPU idle guard.

Publishes CloudWatch metrics that back alarms when the backend GPU ASG stays
running past configured thresholds.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from typing import Any

import boto3

asg = boto3.client("autoscaling")
cw = boto3.client("cloudwatch")
ec2 = boto3.client("ec2")
ssm = boto3.client("ssm")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _active_asg_instances(group: dict[str, Any]) -> list[str]:
    instances = []
    for item in group.get("Instances", []):
        state = item.get("LifecycleState", "")
        if state not in {"Terminating", "Terminated", "Detaching"}:
            instance_id = item.get("InstanceId")
            if instance_id:
                instances.append(instance_id)
    return instances


def _oldest_launch_age_minutes(instance_ids: list[str]) -> float:
    if not instance_ids:
        return 0.0
    response = ec2.describe_instances(InstanceIds=instance_ids)
    now = datetime.now(timezone.utc)
    ages = []
    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            launch_time = instance.get("LaunchTime")
            state = instance.get("State", {}).get("Name")
            if launch_time and state not in {"shutting-down", "terminated"}:
                ages.append((now - launch_time).total_seconds() / 60.0)
    return max(ages) if ages else 0.0


def _fetch_metrics(url: str, timeout_seconds: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _has_active_work(metrics: dict[str, Any]) -> bool:
    for key in ("queue_depth", "replay_inflight_count", "active_backend_jobs"):
        try:
            if int(metrics.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _put_metric(name: str, value: float, dimensions: list[dict[str, str]]) -> None:
    cw.put_metric_data(
        Namespace="PDFX/IdleGuard",
        MetricData=[
            {
                "MetricName": name,
                "Dimensions": dimensions,
                "Value": float(value),
                "Unit": "Count",
            }
        ],
    )


def _put_metrics(metrics: dict[str, float], dimensions: list[dict[str, str]]) -> None:
    cw.put_metric_data(
        Namespace="PDFX/IdleGuard",
        MetricData=[
            {
                "MetricName": name,
                "Dimensions": dimensions,
                "Value": float(value),
                "Unit": "Count",
            }
            for name, value in metrics.items()
        ],
    )


def _parse_iso_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _running_since(parameter_name: str, backend_running: bool, now: datetime) -> datetime | None:
    if not backend_running:
        try:
            ssm.delete_parameter(Name=parameter_name)
        except ssm.exceptions.ParameterNotFound:
            pass
        return None

    try:
        response = ssm.get_parameter(Name=parameter_name)
        return _parse_iso_datetime(response["Parameter"]["Value"])
    except (ssm.exceptions.ParameterNotFound, KeyError, TypeError, ValueError):
        value = now.isoformat().replace("+00:00", "Z")
        ssm.put_parameter(
            Name=parameter_name,
            Value=value,
            Type="String",
            Overwrite=True,
        )
        return now


def _dimensions(project: str, env: str, asg_name: str) -> list[dict[str, str]]:
    return [
        {"Name": "Project", "Value": project},
        {"Name": "Environment", "Value": env},
        {"Name": "BackendAsgName", "Value": asg_name},
    ]


def _publish_failure_heartbeat() -> None:
    project = os.environ.get("PROJECT_NAME", "pdfx")
    env = os.environ.get("ENVIRONMENT_NAME", "prod")
    asg_name = os.environ.get("BACKEND_ASG_NAME", "unknown")
    try:
        _put_metric("GuardCheckSucceeded", 0, _dimensions(project, env, asg_name))
    except Exception as exc:  # noqa: BLE001 - best-effort alarm signal before re-raise
        print(json.dumps({"guard_failure_heartbeat_error": str(exc)}, sort_keys=True))


def _run_check() -> dict[str, Any]:
    project = os.environ["PROJECT_NAME"]
    env = os.environ["ENVIRONMENT_NAME"]
    asg_name = os.environ["BACKEND_ASG_NAME"]
    running_since_parameter = os.environ["RUNNING_SINCE_PARAMETER_NAME"]
    metrics_url = os.environ["PROXY_METRICS_URL"]
    idle_threshold = _env_int("IDLE_ALERT_AFTER_MINUTES", 45)
    absolute_threshold = _env_int("ABSOLUTE_ALERT_AFTER_MINUTES", 240)
    metrics_timeout = _env_int("METRICS_TIMEOUT_SECONDS", 5)
    treat_metrics_failure_as_idle = os.environ.get(
        "TREAT_METRICS_FETCH_FAILURE_AS_IDLE", "true"
    ).lower() in {"1", "true", "yes", "on"}

    groups = asg.describe_auto_scaling_groups(AutoScalingGroupNames=[asg_name]).get(
        "AutoScalingGroups", []
    )
    if not groups:
        raise RuntimeError(f"Auto Scaling Group not found: {asg_name}")

    group = groups[0]
    desired = int(group.get("DesiredCapacity", 0) or 0)
    instance_ids = _active_asg_instances(group)
    oldest_age = _oldest_launch_age_minutes(instance_ids)
    now = datetime.now(timezone.utc)

    metrics: dict[str, Any] = {}
    metrics_error = None
    try:
        metrics = _fetch_metrics(metrics_url, metrics_timeout)
    except Exception as exc:  # noqa: BLE001 - report exact operational failure
        metrics_error = str(exc)

    has_active_work = _has_active_work(metrics) if metrics_error is None else True
    if metrics_error and treat_metrics_failure_as_idle:
        has_active_work = False

    backend_running = desired > 0 or bool(instance_ids)
    running_since = _running_since(running_since_parameter, backend_running, now)
    continuous_age = (
        max((now - running_since).total_seconds() / 60.0, 0.0)
        if running_since
        else 0.0
    )
    idle_too_long = backend_running and not has_active_work and continuous_age >= idle_threshold
    absolute_too_long = (
        backend_running and absolute_threshold > 0 and continuous_age >= absolute_threshold
    )

    dimensions = _dimensions(project, env, asg_name)
    _put_metrics(
        {
            "GuardCheckSucceeded": 1,
            "IdleRunningTooLong": 1 if idle_too_long else 0,
            "AbsoluteRunningTooLong": 1 if absolute_too_long else 0,
            "BackendDesiredCapacity": desired,
            "BackendOldestInstanceAgeMinutes": oldest_age,
            "BackendContinuousRunningAgeMinutes": continuous_age,
        },
        dimensions,
    )

    result = {
        "asg": asg_name,
        "desired": desired,
        "active_instance_ids": instance_ids,
        "oldest_instance_age_minutes": round(oldest_age, 2),
        "continuous_running_age_minutes": round(continuous_age, 2),
        "running_since": running_since.isoformat() if running_since else None,
        "metrics_error": metrics_error,
        "has_active_work": has_active_work,
        "idle_too_long": idle_too_long,
        "absolute_too_long": absolute_too_long,
        "metrics": metrics,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    try:
        return _run_check()
    except Exception as exc:
        print(json.dumps({"idle_guard_error": str(exc)}, sort_keys=True))
        _publish_failure_heartbeat()
        raise
