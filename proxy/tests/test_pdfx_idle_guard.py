"""Unit tests for the deployable idle guard Lambda."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "deploy" / "aws" / "lambda" / "pdfx_idle_guard.py"
)


class ParameterNotFound(Exception):
    pass


class FakeSSMExceptions:
    ParameterNotFound = ParameterNotFound


class FakeSSM:
    exceptions = FakeSSMExceptions

    def __init__(self, parameters: dict[str, str]):
        self.parameters = parameters

    def get_parameter(self, Name: str) -> dict[str, Any]:
        if Name not in self.parameters:
            raise ParameterNotFound(Name)
        return {"Parameter": {"Value": self.parameters[Name]}}

    def put_parameter(self, Name: str, Value: str, Type: str, Overwrite: bool) -> None:
        self.parameters[Name] = Value

    def delete_parameter(self, Name: str) -> None:
        if Name not in self.parameters:
            raise ParameterNotFound(Name)
        del self.parameters[Name]


class FakeASG:
    def __init__(self, activities: list[dict[str, Any]]):
        self.activities = activities

    def describe_auto_scaling_groups(self, AutoScalingGroupNames: list[str]) -> dict[str, Any]:
        return {
            "AutoScalingGroups": [
                {
                    "AutoScalingGroupName": AutoScalingGroupNames[0],
                    "DesiredCapacity": 1,
                    "Instances": [
                        {
                            "InstanceId": "i-current",
                            "LifecycleState": "InService",
                            "HealthStatus": "Healthy",
                        }
                    ],
                }
            ]
        }

    def describe_scaling_activities(
        self,
        AutoScalingGroupName: str,
        MaxRecords: int,
    ) -> dict[str, Any]:
        return {"Activities": self.activities[:MaxRecords]}


class FakeEC2:
    def __init__(self, launch_time: datetime):
        self.launch_time = launch_time

    def describe_instances(self, InstanceIds: list[str]) -> dict[str, Any]:
        return {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": InstanceIds[0],
                            "LaunchTime": self.launch_time,
                            "State": {"Name": "running"},
                        }
                    ]
                }
            ]
        }


class FakeCloudWatch:
    def __init__(self):
        self.metric_data: list[dict[str, Any]] = []

    def put_metric_data(self, Namespace: str, MetricData: list[dict[str, Any]]) -> None:
        self.metric_data.extend(MetricData)

    def value_for(self, name: str) -> float:
        for metric in self.metric_data:
            if metric["MetricName"] == name:
                return metric["Value"]
        raise AssertionError(f"Metric was not published: {name}")


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _load_guard(monkeypatch, clients: dict[str, Any]):
    import boto3

    def fake_client(name: str):
        return clients[name]

    monkeypatch.setattr(boto3, "client", fake_client)
    module_name = "_pdfx_idle_guard_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    env = {
        "PROJECT_NAME": "pdfx",
        "ENVIRONMENT_NAME": "prod",
        "BACKEND_ASG_NAME": "pdfx-backend-test",
        "RUNNING_SINCE_PARAMETER_NAME": "/pdfx/prod/idle-guard/asg-running-since",
        "IDLE_SINCE_PARAMETER_NAME": "/pdfx/prod/idle-guard/asg-idle-since",
        "PROXY_METRICS_URL": "https://example.org/api/v1/metrics",
        "IDLE_ALERT_AFTER_MINUTES": "60",
        "ABSOLUTE_ALERT_AFTER_MINUTES": "240",
        "METRICS_TIMEOUT_SECONDS": "5",
        "TREAT_METRICS_FETCH_FAILURE_AS_IDLE": "true",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        module,
        "_fetch_metrics",
        lambda url, timeout: {
            "queue_depth": 0,
            "replay_inflight_count": 0,
            "active_backend_jobs": 0,
        },
    )
    return module


def test_idle_guard_resets_when_scale_to_zero_was_missed_between_checks(monkeypatch):
    now = datetime.now(timezone.utc)
    stored_since = now - timedelta(minutes=80)
    scale_to_zero_at = now - timedelta(minutes=15)
    current_launch = now - timedelta(minutes=14)
    running_param = "/pdfx/prod/idle-guard/asg-running-since"
    idle_param = "/pdfx/prod/idle-guard/asg-idle-since"
    ssm = FakeSSM({running_param: _iso(stored_since), idle_param: _iso(stored_since)})
    cw = FakeCloudWatch()
    clients = {
        "autoscaling": FakeASG(
            [
                {
                    "StatusCode": "Successful",
                    "StartTime": scale_to_zero_at,
                    "Cause": (
                        "At 2026-06-29T22:01:42Z a user request explicitly set group "
                        "desired capacity changing the desired capacity from 1 to 0."
                    ),
                }
            ]
        ),
        "cloudwatch": cw,
        "ec2": FakeEC2(current_launch),
        "ssm": ssm,
    }
    module = _load_guard(monkeypatch, clients)

    result = module._run_check()

    assert result["reset_after_missed_stop"] is True
    assert result["idle_too_long"] is False
    assert cw.value_for("IdleRunningTooLong") == 0.0
    assert ssm.parameters[running_param] == _iso(current_launch)
    assert module._parse_iso_datetime(ssm.parameters[idle_param]) > scale_to_zero_at


def test_idle_guard_does_not_reset_for_instance_replacement_without_scale_to_zero(monkeypatch):
    now = datetime.now(timezone.utc)
    stored_since = now - timedelta(minutes=80)
    current_launch = now - timedelta(minutes=14)
    running_param = "/pdfx/prod/idle-guard/asg-running-since"
    idle_param = "/pdfx/prod/idle-guard/asg-idle-since"
    ssm = FakeSSM({running_param: _iso(stored_since), idle_param: _iso(stored_since)})
    cw = FakeCloudWatch()
    clients = {
        "autoscaling": FakeASG(
            [
                {
                    "StatusCode": "Successful",
                    "StartTime": now - timedelta(minutes=15),
                    "Cause": "Launching a new EC2 instance after an unhealthy replacement.",
                }
            ]
        ),
        "cloudwatch": cw,
        "ec2": FakeEC2(current_launch),
        "ssm": ssm,
    }
    module = _load_guard(monkeypatch, clients)

    result = module._run_check()

    assert result["reset_after_missed_stop"] is False
    assert result["idle_too_long"] is True
    assert cw.value_for("IdleRunningTooLong") == 1.0
    assert ssm.parameters[running_param] == _iso(stored_since)
    assert ssm.parameters[idle_param] == _iso(stored_since)
