"""Tests for the PDFX AWS stack template."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


STACK_PATH = Path(__file__).resolve().parents[1] / "deploy" / "aws" / "pdfx-stack.yaml"
RUNBOOK_PATH = Path(__file__).resolve().parents[1] / "deploy" / "aws" / "pdfx.md"
GPU_COMPOSE_PATH = Path(__file__).resolve().parents[1] / "deploy" / "docker-compose.gpu.yml"
GPU_PREBUILT_COMPOSE_PATH = Path(__file__).resolve().parents[1] / "deploy" / "docker-compose.gpu.prebuilt.yml"
GPU_DOCKERFILE_PATH = Path(__file__).resolve().parents[1] / "deploy" / "Dockerfile.gpu"
GPU_CONSTRAINTS_PATH = Path(__file__).resolve().parents[1] / "deploy" / "gpu-constraints.txt"
REQUIREMENTS_PATH = Path(__file__).resolve().parents[1] / "requirements.txt"


def test_pdfx_stack_uses_canonical_resources():
    template = STACK_PATH.read_text()

    assert "Default: pdfx" in template
    assert "Default: pdfx-backend" in template
    assert "Default: agr-pdf-extraction-benchmark" in template
    assert "Default: pdfx-proxy" in template
    assert "Default: sg-21ac675b" in template
    assert "Default: sg-006b41eff1820ad53" in template
    assert "BackendLaunchTemplate:" in template
    assert "BackendAutoScalingGroup:" in template
    assert "Type: AWS::EC2::LaunchTemplate" in template
    assert "Type: AWS::AutoScaling::AutoScalingGroup" in template
    assert "HealthCheckType: EBS" in template
    assert "MaxSize: !Ref BackendMaxSize" in template
    assert "StartupTimeoutMinutes:" in template
    assert 'Default: "30"' in template
    assert "Name: !Sub \"/${SsmParameterPath}/ec2-instance-id\"" in template
    assert "Name: !Sub \"/${SsmParameterPath}/backend-asg-name\"" in template
    assert "Name: !Sub \"/${SsmParameterPath}/asg-startup-replacement-attempts\"" in template
    assert "ValueFrom: !Ref SsmEc2InstanceId" in template
    assert "ValueFrom: !Ref SsmBackendAsgName" in template
    assert "ValueFrom: !Ref SsmAsgStartupReplacementAttempts" in template
    assert "BackendInstance:" not in template
    assert "BackendWarmPoolMinSize:" in template
    assert "Default: 1" in template
    assert "MinimumHealthyPercent: 100" in template
    assert "DeploymentCircuitBreaker:" in template
    assert "Rollback: true" in template


def test_pdfx_stack_supports_image_retention_and_tagged_uploads():
    template = STACK_PATH.read_text()

    assert "pdfx-expire-extracted-images" in template
    assert "pdfx-artifact-type" in template
    assert "extracted-image" in template
    assert "pdfx-retention" in template
    assert "temporary" in template
    assert "pdfx-expire-proxy-queue" in template
    assert "ExpirationInDays: !Ref QueueRetentionDays" in template
    assert 'Prefix: !Sub "${QueuePrefix}/"' in template
    assert template.count("s3:PutObjectTagging") >= 2


def test_pdfx_bootstrap_scrubs_storage_env():
    template = STACK_PATH.read_text()

    assert "Scrub production storage values" in template
    assert "AUDIT_S3_BUCKET|AUDIT_S3_BUCKET_SSM_PARAM|AUDIT_S3_PREFIX" in template
    assert "AUDIT_S3_BUCKET=${AuditBucket}" in template
    assert "AUDIT_S3_BUCKET_SSM_PARAM=/${SsmParameterPath}/audit-s3-bucket" in template


def test_pdfx_bootstrap_supports_branch_tag_or_sha_checkout():
    template = STACK_PATH.read_text()

    assert "dnf install -y docker git jq awscli" in template
    assert "dnf install -y docker git jq awscli curl" not in template
    assert "docker compose version" in template
    assert 'git clone "${BackendGitRepositoryUrl}" "$SERVICE_DIR"' in template
    assert 'Reusing existing backend checkout at $SERVICE_DIR' in template
    assert 'git -C "$SERVICE_DIR" fetch --all --tags --prune' in template
    assert 'git -C "$SERVICE_DIR" checkout "${BackendGitRef}"' in template
    assert 'git -C "$SERVICE_DIR" reset --hard "origin/${BackendGitRef}"' in template
    assert 'BACKEND_IMAGE_URI="${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/${BackendImageRepositoryName}:${BackendImageTag}"' in template
    assert 'PDFX_GPU_IMAGE="$BACKEND_IMAGE_URI"' in template
    assert "PDFX_DEPLOY_BUILD_MODE=never" in template
    assert "PDFX_DEPLOY_PULL_IMAGES=auto" in template
    assert 'git clone --branch "${BackendGitRef}"' not in template


def test_deploy_script_does_not_force_rebuild_by_default():
    deploy_script = (Path(__file__).resolve().parents[1] / "deploy" / "deploy.sh").read_text()

    assert 'PDFX_DEPLOY_BUILD_MODE="${PDFX_DEPLOY_BUILD_MODE:-auto}"' in deploy_script
    assert "rebuild)" in deploy_script
    assert "BUILD_ARGS=(--build)" in deploy_script
    assert 'docker compose "${COMPOSE_ARGS[@]}" up -d "${BUILD_ARGS[@]}"' in deploy_script
    assert 'up -d --build' not in deploy_script


def test_pdfx_runbook_documents_safe_bootstrap_path():
    runbook = RUNBOOK_PATH.read_text()

    assert "/pdfx/backend-env" in runbook
    assert "/pdfx/backend-asg-name" in runbook
    assert "Current Account Migration Note" in runbook
    assert "first deploy of this template as a new `pdfx` stack" in runbook
    assert "collide with those existing names" in runbook
    assert "retains its bootstrapped git" in runbook
    assert "fresh one pulls" in runbook
    assert "remove legacy toggles such as `MARKER_EXTRACT_IMAGES`" in runbook
    assert "DeployBackendOnBoot=true" in runbook
    assert "--ssm-prefix /pdfx" in runbook
    assert "prebuilt `agr_pdfx_backend` ECR image" in runbook
    assert "docker-compose.gpu.prebuilt.yml" in runbook


def test_pdfx_stack_has_backend_resilience_alarms():
    template = STACK_PATH.read_text()

    assert "ProxyStartupTimeoutMetricFilter" in template
    assert "ProxyBackendReplacementMetricFilter" in template
    assert "pdfx-startup-timeouts" in template
    assert "pdfx-backend-replacements" in template
    assert "AlarmSnsTopicArn" in template


def test_gpu_compose_runs_image_baked_source():
    compose = GPU_COMPOSE_PATH.read_text()
    prebuilt_compose = GPU_PREBUILT_COMPOSE_PATH.read_text()

    assert "image: ${PDFX_GPU_IMAGE:-pdfx-gpu}" in compose
    assert "dockerfile: deploy/Dockerfile.gpu" in compose
    assert "../app:/app/app:ro" not in compose
    assert "../celery_app.py:/app/celery_app.py:ro" not in compose
    assert "../config.py:/app/config.py:ro" not in compose
    assert "../app:/app/app:ro" not in prebuilt_compose
    assert "../celery_app.py:/app/celery_app.py:ro" not in prebuilt_compose
    assert "../config.py:/app/config.py:ro" not in prebuilt_compose
    assert "/usr/local/lib/python3.11/site-packages/rapidocr/models" in compose
    assert "/usr/local/lib/python3.11/dist-packages/rapidocr/models" in compose
    assert "/usr/local/lib/python3.11/site-packages/rapidocr/models" in prebuilt_compose
    assert "/usr/local/lib/python3.11/dist-packages/rapidocr/models" in prebuilt_compose


def test_prebuilt_gpu_compose_render_omits_source_bind_mounts(tmp_path):
    if not shutil.which("docker"):
        pytest.skip("docker CLI is not available")

    repo = tmp_path / "repo"
    deploy_dir = repo / "deploy"
    deploy_dir.mkdir(parents=True)
    (repo / ".env").write_text("")
    for source in (GPU_COMPOSE_PATH, GPU_PREBUILT_COMPOSE_PATH):
        shutil.copy(source, deploy_dir / source.name)

    env = os.environ.copy()
    env["PDFX_GPU_IMAGE"] = "example.com/pdfx:tag"
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.gpu.yml",
            "-f",
            "docker-compose.gpu.prebuilt.yml",
            "-p",
            "pdfx",
            "config",
        ],
        cwd=deploy_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "example.com/pdfx:tag" in result.stdout
    assert "/app/app" not in result.stdout
    assert "/app/celery_app.py" not in result.stdout
    assert "/app/config.py" not in result.stdout


def test_gpu_dockerfile_preserves_cuda_pytorch_layer():
    dockerfile = GPU_DOCKERFILE_PATH.read_text()
    constraints = GPU_CONSTRAINTS_PATH.read_text()
    requirements_install = dockerfile.split("COPY requirements.txt deploy/gpu-constraints.txt ./", 1)[1]

    assert "deploy/gpu-constraints.txt" in dockerfile
    assert "torch==2.11.0+cu128" in dockerfile
    assert "torchvision==0.26.0+cu128" in dockerfile
    assert "--ignore-installed 'blinker>=1.9.0'" in dockerfile
    assert "-c gpu-constraints.txt" in requirements_install
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in requirements_install
    assert "--ignore-installed" not in requirements_install
    assert "--force-reinstall \\\n    torch" not in requirements_install
    assert "torch==2.11.0+cu128" in constraints
    assert "torchvision==0.26.0+cu128" in constraints


def test_gpu_compose_keeps_web_app_cpu_only_and_worker_ocr_overridable():
    compose = GPU_COMPOSE_PATH.read_text()

    app_section = compose.split("  app:", 1)[1].split("  worker:", 1)[0]
    worker_section = compose.split("  worker:", 1)[1].split("  grobid:", 1)[0]

    assert "DOCLING_DEVICE: cpu" in app_section
    assert 'CUDA_VISIBLE_DEVICES: ""' in app_section
    assert 'DOCLING_DEVICE: "cuda"' in worker_section
    assert "DOCLING_RAPIDOCR_BACKEND:" not in worker_section
    assert "DOCLING_RAPIDOCR_MODEL_TYPE:" not in worker_section
    assert "DOCLING_RAPIDOCR_USE_CUDA:" not in worker_section


def test_marker_requirement_uses_modern_converter_api_version():
    requirements = REQUIREMENTS_PATH.read_text()

    assert "marker-pdf>=1.10.2,<1.11" in requirements


def test_docling_ocr_backend_has_cpu_onnxruntime_requirement():
    requirements = REQUIREMENTS_PATH.read_text()

    assert "onnxruntime>=1.18,<2" in requirements
