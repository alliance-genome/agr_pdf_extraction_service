"""Tests for deployment template invariants."""

import json
from pathlib import Path


def test_task_definition_healthcheck_uses_python_not_curl():
    template_path = Path(__file__).resolve().parents[1] / "deploy" / "task-definition.template.json"
    template = json.loads(template_path.read_text())
    health_cmd = template["containerDefinitions"][0]["healthCheck"]["command"][1]

    assert "python -c" in health_cmd
    assert "urllib.request.urlopen" in health_cmd
    assert "/api/v1/health/live" in health_cmd
    assert "curl" not in health_cmd


def test_cloudformation_healthcheck_uses_proxy_liveness():
    template_path = Path(__file__).resolve().parents[2] / "deploy" / "aws" / "pdfx-test-mirror-stack.yaml"
    template_text = template_path.read_text()

    assert "urlopen('http://localhost:80/api/v1/health/live', timeout=3)" in template_text
    assert "urlopen('http://localhost:80/api/v1/health', timeout=3)" not in template_text


def test_idle_guard_stack_has_email_alarm_path_and_schedule():
    template_path = Path(__file__).resolve().parents[2] / "deploy" / "aws" / "pdfx-idle-guard-stack.yaml"
    template_text = template_path.read_text()

    assert "AWS::Lambda::Function" in template_text
    assert "AWS::Events::Rule" in template_text
    assert "IdleRunningTooLongAlarm" in template_text
    assert "AbsoluteRunningTooLongAlarm" in template_text
    assert "IdleGuardHeartbeatMissingAlarm" in template_text
    assert "IdleGuardLambdaErrorsAlarm" in template_text
    assert "IdleGuardLambdaThrottlesAlarm" in template_text
    assert "GuardCheckSucceeded" in template_text
    assert "RUNNING_SINCE_PARAMETER_NAME" in template_text
    assert "ssm:GetParameter" in template_text
    assert "AlarmSnsTopicArn" in template_text
    assert "PDFX/IdleGuard" in template_text
    assert "TreatMissingData: breaching" in template_text
    assert "TreatMissingData: notBreaching" in template_text


def test_deploy_script_supports_explicit_image_tag():
    script_path = Path(__file__).resolve().parents[1] / "deploy" / "deploy.sh"
    script = script_path.read_text()

    assert "--image-tag" in script
    assert 'IMAGE_TAG="latest"' in script
    assert "agr_pdfx_proxy:${IMAGE_TAG}" in script


def test_deploy_script_supports_environment_scoped_resources():
    script_path = Path(__file__).resolve().parents[1] / "deploy" / "deploy.sh"
    script = script_path.read_text()

    assert "--ssm-prefix" in script
    assert 'SSM_PREFIX="/pdfx"' in script
    assert 'TASK_FAMILY="${TASK_FAMILY:-$SERVICE_NAME}"' in script
    assert 'LOG_GROUP="${LOG_GROUP:-/ecs/$SERVICE_NAME}"' in script
    assert 'QUEUE_PREFIX="${QUEUE_PREFIX:-${SERVICE_NAME}-queue}"' in script
    assert 'AWS_CMD=(aws --region "$REGION")' in script
    assert 'AWS_CMD=(aws --profile "$PROFILE" --region "$REGION")' in script
    assert '"${AWS_CMD[@]}" ssm get-parameter' in script
    assert "read_optional_param()" in script
    assert 'ensure_ssm_param "${SSM_PREFIX}/asg-startup-replacement-attempts" "1"' in script
    assert 'if $DRY_RUN; then' in script
    assert "Would create placeholder SSM parameter" in script


def test_task_definition_is_environment_parameterized():
    template_path = Path(__file__).resolve().parents[1] / "deploy" / "task-definition.template.json"
    template_text = template_path.read_text()
    template = json.loads(template_text)

    assert template["family"] == "${TASK_FAMILY}"
    container = template["containerDefinitions"][0]
    assert container["name"] == "${CONTAINER_NAME}"
    assert {"name": "QUEUE_S3_PREFIX", "value": "${QUEUE_S3_PREFIX}"} in container["environment"]
    assert {"name": "QUEUE_S3_REGION", "value": "${QUEUE_S3_REGION}"} in container["environment"]
    assert container["secrets"][0]["valueFrom"].startswith("${SSM_PREFIX}/")
    assert {"name": "BACKEND_ASG_NAME", "valueFrom": "${SSM_PREFIX}/backend-asg-name"} in container["secrets"]
    assert {
        "name": "ASG_STARTUP_REPLACEMENT_ATTEMPTS",
        "valueFrom": "${SSM_PREFIX}/asg-startup-replacement-attempts",
    } in container["secrets"]
    assert container["logConfiguration"]["options"]["awslogs-group"] == "${LOG_GROUP}"


def test_iam_policy_is_environment_parameterized_and_allows_image_tags():
    template_path = Path(__file__).resolve().parents[1] / "deploy" / "iam-policy.template.json"
    template_text = template_path.read_text()
    template = json.loads(template_text)

    assert "parameter/${SSM_PARAMETER_RESOURCE}/*" in template_text
    assert "autoscaling:SetDesiredCapacity" in template_text
    assert "autoscaling:SetInstanceHealth" in template_text
    assert "autoscaling:DescribeAutoScalingGroups" in template_text
    assert "instance/${EC2_INSTANCE_RESOURCE}" in template_text
    assert "autoScalingGroupName/${BACKEND_ASG_RESOURCE}" in template_text

    object_statement = next(
        statement
        for statement in template["Statement"]
        if statement["Sid"] == "S3QueueReadWriteObjects"
    )
    assert "s3:PutObjectTagging" in object_statement["Action"]
