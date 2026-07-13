"""Tests for deployment template invariants."""

import json
import os
import subprocess
from pathlib import Path

import yaml


def test_task_definition_healthcheck_uses_python_not_curl():
    template_path = Path(__file__).resolve().parents[1] / "deploy" / "task-definition.template.json"
    template = json.loads(template_path.read_text())
    health_cmd = template["containerDefinitions"][0]["healthCheck"]["command"][1]

    assert "python -c" in health_cmd
    assert "urllib.request.urlopen" in health_cmd
    assert "/api/v1/health/live" in health_cmd
    assert "curl" not in health_cmd


def test_cloudformation_healthcheck_uses_proxy_liveness():
    template_path = Path(__file__).resolve().parents[2] / "deploy" / "aws" / "pdfx-stack.yaml"
    template_text = template_path.read_text()

    assert "urlopen('http://localhost:80/api/v1/health/live', timeout=3)" in template_text
    assert "urlopen('http://localhost:80/api/v1/health', timeout=3)" not in template_text
    assert "PDFX_DEPLOY_BUILD_MODE=never" in template_text
    assert "PDFX_GPU_IMAGE=" in template_text
    assert "BackendImageRepositoryName" in template_text
    assert "agr_pdfx_backend" in template_text


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
    assert "IDLE_SINCE_PARAMETER_NAME" in template_text
    assert "ASG_STATE_PARAMETER_NAME" in template_text
    assert "ssm:GetParameter" in template_text
    assert "autoscaling:DescribeScalingActivities" in template_text
    assert "AlarmSnsTopicArn" in template_text
    assert "PDFX/IdleGuard" in template_text
    assert "Default: 130" in template_text
    assert "TreatMissingData: breaching" in template_text
    assert "TreatMissingData: notBreaching" in template_text

    deploy_script = (Path(__file__).resolve().parents[2] / "deploy" / "aws" / "deploy_idle_guard.sh").read_text()
    assert 'IDLE_ALERT_AFTER_MINUTES="130"' in deploy_script
    assert "--reuse-existing-parameters" in deploy_script
    assert "read_existing_parameter" in deploy_script
    assert "reuse_existing_parameter_unless_explicit" in deploy_script
    assert "TreatMetricsFetchFailureAsIdle" in deploy_script
    assert "LambdaArtifactBucket" in deploy_script


def test_idle_guard_reuses_stack_parameters_but_keeps_explicit_overrides(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "deploy" / "aws" / "deploy_idle_guard.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aws_log = tmp_path / "aws.log"
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$FAKE_AWS_LOG"
case "$*" in
  *BackendAsgName*) echo stored-asg ;;
  *ProxyMetricsUrl*) echo https://stored.example/api/v1/metrics ;;
  *IdleAlertAfterMinutes*) echo 131 ;;
  *AbsoluteAlertAfterMinutes*) echo 1500 ;;
  *ScheduleExpression*) echo 'rate(10 minutes)' ;;
  *MetricsTimeoutSeconds*) echo 9 ;;
  *TreatMetricsFetchFailureAsIdle*) echo false ;;
  *AlarmSnsTopicArn*) echo arn:aws:sns:us-east-1:123456789012:stored-topic ;;
  *LambdaArtifactBucket*) echo stored-bucket ;;
esac
"""
    )
    fake_aws.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["FAKE_AWS_LOG"] = str(aws_log)

    result = subprocess.run(
        [
            str(script_path),
            "--reuse-existing-parameters",
            "--idle-alert-after-minutes",
            "999",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = aws_log.read_text()
    deploy_call = next(line for line in calls.splitlines() if "cloudformation deploy" in line)
    assert "BackendAsgName=stored-asg" in deploy_call
    assert "ProxyMetricsUrl=https://stored.example/api/v1/metrics" in deploy_call
    assert "IdleAlertAfterMinutes=999" in deploy_call
    assert "AbsoluteAlertAfterMinutes=1500" in deploy_call
    assert "ScheduleExpression=rate(10 minutes)" in deploy_call
    assert "MetricsTimeoutSeconds=9" in deploy_call
    assert "TreatMetricsFetchFailureAsIdle=false" in deploy_call
    assert "AlarmSnsTopicArn=arn:aws:sns:us-east-1:123456789012:stored-topic" in deploy_call
    assert "LambdaArtifactBucket=stored-bucket" in deploy_call
    assert "--tags Team=specialists Project=pdfx" in deploy_call


def test_deploy_script_supports_explicit_image_tag():
    script_path = Path(__file__).resolve().parents[1] / "deploy" / "deploy.sh"
    script = script_path.read_text()

    assert "--image-tag" in script
    assert 'IMAGE_TAG="latest"' in script
    assert "agr_pdfx_proxy:${IMAGE_TAG}" in script


def test_proxy_idle_timeout_defaults_to_two_hours():
    config_path = Path(__file__).resolve().parents[1] / "app" / "config.py"
    config_text = config_path.read_text()
    stack_path = Path(__file__).resolve().parents[2] / "deploy" / "aws" / "pdfx-stack.yaml"
    stack_text = stack_path.read_text()

    assert 'IDLE_TIMEOUT_MINUTES", "120"' in config_text
    assert "IdleTimeoutMinutes:" in stack_text
    assert 'Default: "120"' in stack_text


def test_backend_deploy_supports_prebuilt_gpu_image():
    script_path = Path(__file__).resolve().parents[2] / "deploy" / "deploy.sh"
    script = script_path.read_text()
    compose_path = Path(__file__).resolve().parents[2] / "deploy" / "docker-compose.gpu.yml"
    compose_text = compose_path.read_text()
    prebuilt_compose_path = Path(__file__).resolve().parents[2] / "deploy" / "docker-compose.gpu.prebuilt.yml"
    prebuilt_compose_text = prebuilt_compose_path.read_text()

    assert "PDFX_DEPLOY_BUILD_MODE" in script
    assert "PDFX_DEPLOY_PULL_IMAGES" in script
    assert "SHOULD_PULL_PREBUILT_GPU_IMAGE" in script
    assert 'docker-compose.gpu.prebuilt.yml" -p "$PROJECT_NAME"' in script
    assert '[ "${PDFX_DEPLOY_PULL_IMAGES}" = "always" ]' in script
    assert '[ "${PDFX_DEPLOY_PULL_IMAGES}" = "auto" ] && [ "${PDFX_DEPLOY_BUILD_MODE}" = "never" ]' in script
    assert "docker compose \"${COMPOSE_ARGS[@]}\" pull app worker" in script
    assert "Waiting for NVIDIA container runtime" in script
    assert "PDFX_NVIDIA_PROBE_TIMEOUT_SECONDS" in script
    assert "docker run --rm --gpus all --entrypoint nvidia-smi" in script
    assert "PDFX_WORKER_CUDA_PROBE_TIMEOUT_SECONDS" in script
    assert "GPU worker CUDA" in script
    assert "PDFX_FLASK_HEALTH_TIMEOUT_SECONDS" in script
    assert "wait_for_flask_health" in script
    assert "torch.cuda.mem_get_info(0)" in script
    assert "PDFX_PREWARM_MODELS" in script
    assert "Prewarming Marker models into persistent cache" in script
    assert "from marker.models import create_model_dict" in script
    assert "marker_worker_ready.json" in script
    assert "PDFX_GPU_IMAGE" in compose_text
    assert "image: ${PDFX_GPU_IMAGE:-pdfx-gpu}" in compose_text
    assert "PDFX_MARKER_READY_FILE" in compose_text
    assert "PDFX_HEALTH_REQUIRE_MARKER_READY" in compose_text
    assert "PDFX_WORKER_PRELOAD_MARKER_MODELS" in compose_text
    assert "../app:/app/app:ro" not in compose_text
    assert "../celery_app.py:/app/celery_app.py:ro" not in compose_text
    assert "../config.py:/app/config.py:ro" not in compose_text
    assert "../app:/app/app:ro" not in prebuilt_compose_text
    assert "../celery_app.py:/app/celery_app.py:ro" not in prebuilt_compose_text
    assert "../config.py:/app/config.py:ro" not in prebuilt_compose_text


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
    assert '"minimumHealthyPercent":100' in script
    assert '"maximumPercent":200' in script
    assert '--deployment-configuration "$DEPLOYMENT_CONFIGURATION"' in script
    assert 'ACTIVE_TASK_DEF=$("${AWS_CMD[@]}" ecs describe-services' in script
    assert 'PRIMARY_ROLLOUT_STATE=$("${AWS_CMD[@]}" ecs describe-services' in script
    assert '"$ACTIVE_TASK_DEF" != "$TASK_DEF_ARN"' in script
    assert 'case "$PRIMARY_ROLLOUT_STATE" in' in script
    assert "ECS_ROLLOUT_VERIFY_ATTEMPTS" in script
    assert "ECS_ROLLOUT_VERIFY_DELAY_SECONDS" in script


def test_deploy_script_fails_when_ecs_stabilizes_after_rollback(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "proxy" / "deploy" / "deploy.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
args="$*"
case "$args" in
  *"ssm get-parameter"*"/aws-account-id"*) echo 123456789012 ;;
  *"ssm get-parameter"*"/ec2-instance-id"*) echo i-0123456789abcdef0 ;;
  *"ssm get-parameter"*"/execution-role-name"*) echo execution-role ;;
  *"ssm get-parameter"*"/task-role-name"*) echo task-role ;;
  *"ssm get-parameter"*"/audit-s3-bucket"*) echo audit-bucket ;;
  *"ssm get-parameter"*"/backend-asg-name"*) echo pdfx-backend ;;
  *"ssm get-parameter"*) echo configured ;;
  *"ecs register-task-definition"*) echo arn:aws:ecs:us-east-1:123456789012:task-definition/pdfx-proxy:52 ;;
  *"ecs update-service"*) echo '{}' ;;
  *"ecs wait services-stable"*) exit 0 ;;
  *"ecs describe-services"*"taskDefinition"*) echo arn:aws:ecs:us-east-1:123456789012:task-definition/pdfx-proxy:51 ;;
  *"ecs describe-services"*"rolloutState"*) echo COMPLETED ;;
  *) echo "unexpected aws invocation: $args" >&2; exit 90 ;;
esac
"""
    )
    fake_aws.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [str(script_path), "--image-tag", "new-release"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "ECS stabilized on a different task definition" in result.stderr
    assert "pdfx-proxy:52" in result.stderr
    assert "pdfx-proxy:51" in result.stderr


def test_deploy_script_waits_for_rollout_state_to_catch_up(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "proxy" / "deploy" / "deploy.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    rollout_state_file = tmp_path / "rollout-state-count"
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
args="$*"
case "$args" in
  *"ssm get-parameter"*"/aws-account-id"*) echo 123456789012 ;;
  *"ssm get-parameter"*"/ec2-instance-id"*) echo i-0123456789abcdef0 ;;
  *"ssm get-parameter"*"/execution-role-name"*) echo execution-role ;;
  *"ssm get-parameter"*"/task-role-name"*) echo task-role ;;
  *"ssm get-parameter"*"/audit-s3-bucket"*) echo audit-bucket ;;
  *"ssm get-parameter"*"/backend-asg-name"*) echo pdfx-backend ;;
  *"ssm get-parameter"*) echo configured ;;
  *"ecs register-task-definition"*) echo arn:aws:ecs:us-east-1:123456789012:task-definition/pdfx-proxy:52 ;;
  *"ecs update-service"*) echo '{}' ;;
  *"ecs wait services-stable"*) exit 0 ;;
  *"ecs describe-services"*"taskDefinition"*) echo arn:aws:ecs:us-east-1:123456789012:task-definition/pdfx-proxy:52 ;;
  *"ecs describe-services"*"rolloutState"*)
    count=$(cat "$FAKE_ROLLOUT_STATE_FILE" 2>/dev/null || echo 0)
    count=$((count + 1))
    printf '%s\n' "$count" > "$FAKE_ROLLOUT_STATE_FILE"
    if ((count == 1)); then echo IN_PROGRESS; else echo COMPLETED; fi
    ;;
  *) echo "unexpected aws invocation: $args" >&2; exit 90 ;;
esac
"""
    )
    fake_aws.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_ROLLOUT_STATE_FILE": str(rollout_state_file),
            "ECS_ROLLOUT_VERIFY_ATTEMPTS": "3",
            "ECS_ROLLOUT_VERIFY_DELAY_SECONDS": "0",
        }
    )

    result = subprocess.run(
        [str(script_path), "--image-tag", "new-release"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Rollout state is still IN_PROGRESS" in result.stdout
    assert "Service is stable on" in result.stdout
    assert rollout_state_file.read_text().strip() == "2"


def test_backend_release_publication_restores_ssm_pair_on_partial_write(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    workflow_path = repo_root / ".github" / "workflows" / "main-build-and-deploy.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    publish_script = next(
        step["run"]
        for step in workflow["jobs"]["deploy-prod"]["steps"]
        if step.get("name") == "Publish matched backend AMI and image tag"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aws_log = tmp_path / "aws.log"
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_AWS_LOG"
case "$*" in
  *"ssm get-parameter"*"/backend-ami"*) echo ami-00000000000000001 ;;
  *"ssm get-parameter"*"/backend-image-tag"*) echo old-tag ;;
  *"ssm put-parameter"*"--value new-tag"*) exit 42 ;;
  *"ssm put-parameter"*) exit 0 ;;
  *) echo "unexpected aws invocation: $*" >&2; exit 90 ;;
esac
"""
    )
    fake_aws.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_AWS_LOG": str(aws_log),
            "AWS_REGION": "us-east-1",
            "SSM_PREFIX": "/pdfx",
            "NEW_AMI_ID": "ami-00000000000000002",
            "IMAGE_TAG": "new-tag",
        }
    )

    result = subprocess.run(
        ["bash", "-c", publish_script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 42
    assert "restoring the previous SSM pair" in result.stderr
    calls = aws_log.read_text()
    assert "--value ami-00000000000000002" in calls
    assert "--value new-tag" in calls
    assert "--value ami-00000000000000001" in calls
    assert "--value old-tag" in calls


def test_task_definition_is_environment_parameterized():
    template_path = Path(__file__).resolve().parents[1] / "deploy" / "task-definition.template.json"
    template_text = template_path.read_text()
    template = json.loads(template_text)

    assert template["family"] == "${TASK_FAMILY}"
    assert int(template["memory"]) >= 2048
    assert template["ephemeralStorage"] == {"sizeInGiB": 50}
    container = template["containerDefinitions"][0]
    assert container["name"] == "${CONTAINER_NAME}"
    assert {"name": "QUEUE_S3_PREFIX", "value": "${QUEUE_S3_PREFIX}"} in container["environment"]
    assert {"name": "QUEUE_S3_REGION", "value": "${QUEUE_S3_REGION}"} in container["environment"]
    assert {"name": "MAX_UPLOAD_BYTES", "value": "524288000"} in container["environment"]
    assert {"name": "MAX_MULTIPART_OVERHEAD_BYTES", "value": "10485760"} in container["environment"]
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

    gh_policy_path = Path(__file__).resolve().parents[2] / "deploy" / "aws" / "github-actions-deploy-policy.json"
    gh_policy_text = gh_policy_path.read_text()
    gh_policy = json.loads(gh_policy_text)
    assert "repository/agr_pdfx_proxy" in gh_policy_text
    assert "repository/agr_pdfx_backend" in gh_policy_text
    profile_read = next(
        statement
        for statement in gh_policy["Statement"]
        if statement["Sid"] == "ReadBackendInstanceProfile"
    )
    assert profile_read["Action"] == "iam:GetInstanceProfile"
    assert profile_read["Resource"] == (
        "arn:aws:iam::100225593120:instance-profile/pdfx-ec2-profile"
    )

    stack_path = Path(__file__).resolve().parents[2] / "deploy" / "aws" / "pdfx-stack.yaml"
    stack_text = stack_path.read_text()
    assert "ecr:GetAuthorizationToken" in stack_text
    assert "ecr:GetDownloadUrlForLayer" in stack_text
    assert "repository/${BackendImageRepositoryName}" in stack_text
    assert "QueueRetentionDays" in stack_text
    assert "pdfx-expire-proxy-queue" in stack_text
    assert "ProxyTaskEphemeralStorageGiB" in stack_text
    assert "EphemeralStorage:" in stack_text
