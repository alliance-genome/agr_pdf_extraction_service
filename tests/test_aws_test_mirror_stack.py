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
    assert "Name: !Sub \"/${SsmParameterPath}/ec2-instance-id\"" in template
    assert "ValueFrom: !Ref SsmEc2InstanceId" in template
    assert "/pdfx/ec2-instance-id" not in template
    assert "agr-pdf-extraction-benchmark" not in template


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
    assert "remove legacy toggles such as `MARKER_EXTRACT_IMAGES`" in runbook
    assert "DeployBackendOnBoot=true" in runbook
    assert "--ssm-prefix /pdfx-test" in runbook
    assert "does not clone the production EBS volume" in runbook


def test_gpu_compose_builds_local_image_for_test_backend():
    compose = GPU_COMPOSE_PATH.read_text()

    assert "image: pdfx-gpu" in compose
    assert "dockerfile: deploy/Dockerfile.gpu" in compose


def test_marker_requirement_uses_modern_converter_api_version():
    requirements = REQUIREMENTS_PATH.read_text()

    assert "marker-pdf>=1.10.2,<1.11" in requirements
