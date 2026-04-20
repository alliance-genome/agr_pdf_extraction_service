"""Tests for the main branch deployment workflow."""

from pathlib import Path


def test_workflow_triggers_only_for_merged_prs_to_main():
    workflow_path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "main-build-and-deploy.yml"
    workflow = workflow_path.read_text()

    assert "pull_request:" in workflow
    assert "types: [closed]" in workflow
    assert "branches:" in workflow
    assert "- main" in workflow
    assert "github.event.pull_request.merged == true" in workflow
    assert "no-deploy" in workflow


def test_workflow_uses_oidc_and_promotes_same_image_across_envs():
    workflow_path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "main-build-and-deploy.yml"
    workflow = workflow_path.read_text()

    for env_name in ("dev", "stage", "prod"):
        assert f"environment: {env_name}" in workflow

    assert "aws-actions/configure-aws-credentials@v5" in workflow
    assert "role-to-assume: ${{ secrets.GH_ACTIONS_AWS_ROLE }}" in workflow
    assert workflow.count("docker/build-push-action@v6") == 1
    assert "actions/upload-artifact@v4" in workflow
    assert "actions/download-artifact@v5" in workflow
    assert "docker load --input" in workflow
    assert "/agr_pdfx_proxy:${{ env.IMAGE_TAG }}" in workflow
    assert "./deploy.sh --region \"${AWS_REGION}\" --image-tag \"${IMAGE_TAG}\"" in workflow
