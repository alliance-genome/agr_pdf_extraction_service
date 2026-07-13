"""Tests for the main branch deployment workflow."""

from pathlib import Path
import re

import yaml


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "main-build-and-deploy.yml"
)
PACKER_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "aws"
    / "ami"
    / "pdfx-backend.pkr.hcl"
)


def _load_workflow():
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def _backend_path_pattern():
    detector = next(
        step
        for step in _load_workflow()["jobs"]["on-merge"]["steps"]
        if step.get("id") == "changed"
    )["run"]
    match = re.search(
        r"if grep -Eq '([^']*deploy/Dockerfile\\\.gpu[^']*)' <<< \"\$changed_files\"; then",
        detector,
    )
    assert match is not None
    return re.compile(match.group(1))


def test_workflow_supports_merged_prs_and_manual_recovery():
    workflow = WORKFLOW_PATH.read_text()

    assert "pull_request:" in workflow
    assert "types: [closed]" in workflow
    assert "github.event.pull_request.merged == true" in workflow
    assert "workflow_dispatch:" in workflow
    assert "force_proxy_deploy:" in workflow
    assert "force_backend_bake:" in workflow
    assert "backend_build_subnet_id:" in workflow
    assert "default: subnet-81c95ee4" in workflow
    assert "no-deploy" in workflow


def test_packer_allows_two_hours_for_large_ami_registration():
    template = PACKER_TEMPLATE_PATH.read_text()

    polling = re.search(r"aws_polling\s*\{(?P<body>.*?)\}", template, re.DOTALL)
    assert polling is not None
    assert re.search(r"delay_seconds\s*=\s*20\b", polling.group("body"))
    assert re.search(r"max_attempts\s*=\s*360\b", polling.group("body"))


def test_workflow_uses_one_environment_scoped_oidc_release_job():
    data = _load_workflow()
    jobs = data["jobs"]

    environments = {
        name: job["environment"]
        for name, job in jobs.items()
        if "environment" in job
    }
    assert environments == {"deploy-prod": "prod"}

    id_token_writers = {
        name
        for name, job in jobs.items()
        if (job.get("permissions") or {}).get("id-token") == "write"
    }
    assert id_token_writers == {"deploy-prod"}

    deploy_prod = jobs["deploy-prod"]
    assert "build-proxy-image" in (deploy_prod.get("needs") or [])
    assert "always()" in deploy_prod["if"]

    step_uses = [
        step.get("uses", "")
        for job in jobs.values()
        for step in (job.get("steps") or [])
    ]
    assert step_uses.count("aws-actions/configure-aws-credentials@v5") == 1
    assert step_uses.count("docker/build-push-action@v6") == 2
    assert "actions/upload-artifact@v4" in step_uses
    assert "actions/download-artifact@v5" in step_uses

    workflow_text = WORKFLOW_PATH.read_text()
    assert "role-to-assume: ${{ secrets.GH_ACTIONS_AWS_ROLE }}" in workflow_text
    assert "file: ./deploy/Dockerfile.gpu" in workflow_text
    assert 'backend_git_ref="${SOURCE_REF}"' in workflow_text
    assert "BUILD_SUBNET_ID: ${{ inputs.backend_build_subnet_id || 'subnet-81c95ee4' }}" in workflow_text


def test_change_scope_avoids_unrelated_builds_and_covers_backend_inputs():
    data = _load_workflow()
    scope_steps = data["jobs"]["on-merge"]["steps"]
    detector = next(step for step in scope_steps if step.get("id") == "changed")
    detector_script = detector["run"]

    assert "proxy-changed" in detector_script
    assert "backend-changed" in detector_script
    assert "idle-guard-changed" in detector_script
    assert "main-stack-changed" in detector_script
    assert "gh api --paginate" in detector_script
    assert 'pulls/${PR_NUMBER}/files?per_page=100' in detector_script
    assert "git diff --name-only HEAD~1 HEAD" not in detector_script
    assert "proxy/app/" in detector_script
    assert "proxy/deploy/" in detector_script
    assert "alembic/" in detector_script
    assert "deploy/gpu-constraints" in detector_script
    assert "deploy/deploy" in detector_script
    assert "deploy/aws/ami/(pdfx-backend" in detector_script

    jobs = data["jobs"]
    assert jobs["build-proxy-image"]["if"] == "needs.on-merge.outputs.proxy-changed == 'true'"
    backend_build = next(
        step
        for step in jobs["deploy-prod"]["steps"]
        if step.get("name") == "Build and push immutable backend image tag"
    )
    assert "backend-changed" in backend_build["if"]
    assert "idle-guard-deploy-required" in jobs
    assert "main-stack-reconcile-required" in jobs
    assert "--reuse-existing-parameters" in WORKFLOW_PATH.read_text()


def test_backend_change_scope_only_includes_runtime_and_bake_inputs():
    pattern = _backend_path_pattern()

    triggering_paths = (
        ".dockerignore",
        "requirements.txt",
        "alembic/versions/example.py",
        "app/main.py",
        "deploy/Dockerfile.gpu",
        "deploy/docker-compose.gpu.yml",
        "deploy/aws/ami/pdfx-backend.pkr.hcl",
        "deploy/aws/ami/provision.sh",
    )
    ignored_paths = (
        "README.md",
        "deploy/aws/ami/README.md",
        "deploy/aws/ami/tests/test_prune_amis.sh",
        "deploy/aws/ami/prune_amis.sh",
        ".github/workflows/main-build-and-deploy.yml",
    )

    assert all(pattern.search(path) for path in triggering_paths)
    assert not any(pattern.search(path) for path in ignored_paths)


def test_proxy_latest_moves_only_after_rollout_and_public_smoke_test():
    steps = _load_workflow()["jobs"]["deploy-prod"]["steps"]
    step_names = [step.get("name", "") for step in steps]

    deploy_idx = step_names.index("Register task definition and roll ECS service")
    smoke_idx = step_names.index("Smoke test the public proxy")
    latest_idx = step_names.index("Promote proxy :latest to the deployed image")

    assert deploy_idx < smoke_idx < latest_idx
    assert steps[latest_idx]["if"] == (
        "success() && needs.on-merge.outputs.proxy-changed == 'true'"
    )
    assert "/api/v1/health/live" in steps[smoke_idx]["run"]
    assert "/api/v1/metrics" in steps[smoke_idx]["run"]


def test_backend_latest_moves_only_after_ami_and_ssm_publication():
    steps = _load_workflow()["jobs"]["deploy-prod"]["steps"]
    step_names = [step.get("name", "") for step in steps]

    build_idx = step_names.index("Build and push immutable backend image tag")
    bake_idx = step_names.index("Bake backend AMI")
    publish_idx = step_names.index("Publish matched backend AMI and image tag")
    prune_idx = step_names.index("Prune old baked AMIs")
    latest_idx = step_names.index(
        "Promote backend :latest after successful AMI publication"
    )

    assert build_idx < bake_idx < publish_idx < prune_idx < latest_idx
    assert steps[latest_idx]["if"] == (
        "success() && needs.on-merge.outputs.backend-changed == 'true'"
    )
    bake_script = steps[bake_idx]["run"]
    assert "aws ec2 wait image-available" in bake_script
    assert 'echo "ami-id=${AMI_ID}" >> "$GITHUB_OUTPUT"' in bake_script
    publish_script = steps[publish_idx]["run"]
    assert "backend-ami" in publish_script
    assert "--data-type aws:ec2:image" in publish_script
    assert "backend-image-tag" in publish_script
    assert "rollback_release_pair" in publish_script
    assert '"$PUBLISHED_AMI_ID" != "$NEW_AMI_ID"' in publish_script
