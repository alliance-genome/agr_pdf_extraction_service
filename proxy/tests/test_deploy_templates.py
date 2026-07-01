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
    assert "TreatMissingData: breaching" in template_text
    assert "TreatMissingData: notBreaching" in template_text


def test_deploy_script_supports_explicit_image_tag():
    script_path = Path(__file__).resolve().parents[1] / "deploy" / "deploy.sh"
    script = script_path.read_text()

    assert "--image-tag" in script
    assert 'IMAGE_TAG="latest"' in script
    assert "agr_pdfx_proxy:${IMAGE_TAG}" in script


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
    assert "torch.cuda.mem_get_info(0)" in script
    assert "PDFX_PREWARM_MODELS" in script
    assert "Prewarming Marker models into persistent cache" in script
    assert "from marker.models import create_model_dict" in script
    assert "PDFX_GPU_IMAGE" in compose_text
    assert "image: ${PDFX_GPU_IMAGE:-pdfx-gpu}" in compose_text
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
    assert "repository/agr_pdfx_proxy" in gh_policy_text
    assert "repository/agr_pdfx_backend" in gh_policy_text

    stack_path = Path(__file__).resolve().parents[2] / "deploy" / "aws" / "pdfx-stack.yaml"
    stack_text = stack_path.read_text()
    assert "ecr:GetAuthorizationToken" in stack_text
    assert "ecr:GetDownloadUrlForLayer" in stack_text
    assert "repository/${BackendImageRepositoryName}" in stack_text
    assert "QueueRetentionDays" in stack_text
    assert "pdfx-expire-proxy-queue" in stack_text
    assert "ProxyTaskEphemeralStorageGiB" in stack_text
    assert "EphemeralStorage:" in stack_text
