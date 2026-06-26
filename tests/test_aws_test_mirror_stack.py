"""Tests for the PDFX AWS test mirror template."""

from pathlib import Path


STACK_PATH = Path(__file__).resolve().parents[1] / "deploy" / "aws" / "pdfx-test-mirror-stack.yaml"
RUNBOOK_PATH = Path(__file__).resolve().parents[1] / "deploy" / "aws" / "pdfx-test-mirror.md"
GPU_COMPOSE_PATH = Path(__file__).resolve().parents[1] / "deploy" / "docker-compose.gpu.yml"
REQUIREMENTS_PATH = Path(__file__).resolve().parents[1] / "requirements.txt"


def test_test_mirror_stack_uses_separate_environment_resources():
    template = STACK_PATH.read_text()

    assert "Default: pdfx-test" in template
    assert "Default: agr-pdf-extraction-test" in template
    assert "Default: pdfx-proxy-test" in template
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
    assert "/pdfx/ec2-instance-id" not in template
    assert "agr-pdf-extraction-benchmark" not in template
    assert "MinimumHealthyPercent: 100" in template
    assert "DeploymentCircuitBreaker:" in template
    assert "Rollback: true" in template


def test_test_mirror_stack_supports_image_retention_and_tagged_uploads():
    template = STACK_PATH.read_text()

    assert "pdfx-expire-extracted-images" in template
    assert "pdfx-artifact-type" in template
    assert "extracted-image" in template
    assert "pdfx-retention" in template
    assert "temporary" in template
    assert template.count("s3:PutObjectTagging") >= 2


def test_test_mirror_bootstrap_scrubs_prod_storage_env():
    template = STACK_PATH.read_text()

    assert "Scrub production storage values" in template
    assert "AUDIT_S3_BUCKET|AUDIT_S3_BUCKET_SSM_PARAM|AUDIT_S3_PREFIX" in template
    assert "AUDIT_S3_BUCKET=${AuditBucket}" in template
    assert "AUDIT_S3_BUCKET_SSM_PARAM=/${SsmParameterPath}/audit-s3-bucket" in template


def test_test_mirror_bootstrap_supports_branch_tag_or_sha_checkout():
    template = STACK_PATH.read_text()

    assert "dnf install -y docker git jq awscli" in template
    assert "dnf install -y docker git jq awscli curl" not in template
    assert "docker compose version" in template
    assert 'git clone "${BackendGitRepositoryUrl}" "$SERVICE_DIR"' in template
    assert 'git -C "$SERVICE_DIR" fetch --all --tags --prune' in template
    assert 'git -C "$SERVICE_DIR" checkout "${BackendGitRef}"' in template
    assert 'git clone --branch "${BackendGitRef}"' not in template


def test_test_mirror_runbook_documents_safe_bootstrap_path():
    runbook = RUNBOOK_PATH.read_text()

    assert "/pdfx-test/backend-env" in runbook
    assert "/pdfx-test/backend-asg-name" in runbook
    assert "remove legacy toggles such as `MARKER_EXTRACT_IMAGES`" in runbook
    assert "DeployBackendOnBoot=true" in runbook
    assert "--ssm-prefix /pdfx-test" in runbook
    assert "does not clone the production EBS volume" in runbook


def test_test_mirror_stack_has_backend_resilience_alarms():
    template = STACK_PATH.read_text()

    assert "ProxyStartupTimeoutMetricFilter" in template
    assert "ProxyBackendReplacementMetricFilter" in template
    assert "pdfx-${EnvironmentName}-startup-timeouts" in template
    assert "pdfx-${EnvironmentName}-backend-replacements" in template
    assert "AlarmSnsTopicArn" in template


def test_gpu_compose_builds_local_image_for_test_backend():
    compose = GPU_COMPOSE_PATH.read_text()

    assert "image: pdfx-gpu" in compose
    assert "dockerfile: deploy/Dockerfile.gpu" in compose


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
