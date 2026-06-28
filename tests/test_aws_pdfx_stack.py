"""Tests for the PDFX AWS stack template."""

from pathlib import Path


STACK_PATH = Path(__file__).resolve().parents[1] / "deploy" / "aws" / "pdfx-stack.yaml"
RUNBOOK_PATH = Path(__file__).resolve().parents[1] / "deploy" / "aws" / "pdfx.md"
GPU_COMPOSE_PATH = Path(__file__).resolve().parents[1] / "deploy" / "docker-compose.gpu.yml"
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
    assert "PDFX_DEPLOY_BUILD_MODE=auto GPU_MODE=on ./deploy.sh" in template
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
    assert "remove legacy toggles such as `MARKER_EXTRACT_IMAGES`" in runbook
    assert "DeployBackendOnBoot=true" in runbook
    assert "--ssm-prefix /pdfx" in runbook
    assert "of cloning an EBS volume" in runbook


def test_pdfx_stack_has_backend_resilience_alarms():
    template = STACK_PATH.read_text()

    assert "ProxyStartupTimeoutMetricFilter" in template
    assert "ProxyBackendReplacementMetricFilter" in template
    assert "pdfx-startup-timeouts" in template
    assert "pdfx-backend-replacements" in template
    assert "AlarmSnsTopicArn" in template


def test_gpu_compose_builds_local_image_for_backend():
    compose = GPU_COMPOSE_PATH.read_text()

    assert "image: pdfx-gpu" in compose
    assert "dockerfile: deploy/Dockerfile.gpu" in compose
    assert "/usr/local/lib/python3.11/site-packages/rapidocr/models" in compose
    assert "/usr/local/lib/python3.11/dist-packages/rapidocr/models" in compose


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
