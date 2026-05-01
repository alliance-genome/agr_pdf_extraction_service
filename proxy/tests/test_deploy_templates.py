"""Tests for deployment template invariants."""

import json
from pathlib import Path


def test_task_definition_healthcheck_uses_python_not_curl():
    template_path = Path(__file__).resolve().parents[1] / "deploy" / "task-definition.template.json"
    template = json.loads(template_path.read_text())
    health_cmd = template["containerDefinitions"][0]["healthCheck"]["command"][1]

    assert "python -c" in health_cmd
    assert "urllib.request.urlopen" in health_cmd
    assert "curl" not in health_cmd


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
    assert container["logConfiguration"]["options"]["awslogs-group"] == "${LOG_GROUP}"


def test_iam_policy_is_environment_parameterized_and_allows_image_tags():
    template_path = Path(__file__).resolve().parents[1] / "deploy" / "iam-policy.template.json"
    template_text = template_path.read_text()
    template = json.loads(template_text)

    assert "parameter/${SSM_PARAMETER_RESOURCE}/*" in template_text

    object_statement = next(
        statement
        for statement in template["Statement"]
        if statement["Sid"] == "S3QueueReadWriteObjects"
    )
    assert "s3:PutObjectTagging" in object_statement["Action"]
