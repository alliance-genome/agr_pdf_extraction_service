"""Tests for the main branch deployment workflow."""

from pathlib import Path

import yaml


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "main-build-and-deploy.yml"
)


def _load_workflow():
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def test_workflow_triggers_only_for_merged_prs_to_main():
    workflow = WORKFLOW_PATH.read_text()

    assert "pull_request:" in workflow
    assert "types: [closed]" in workflow
    assert "branches:" in workflow
    assert "- main" in workflow
    assert "github.event.pull_request.merged == true" in workflow
    assert "no-deploy" in workflow


def test_workflow_deploys_to_single_prod_environment_with_oidc():
    data = _load_workflow()
    jobs = data["jobs"]

    environments = {
        name: job["environment"]
        for name, job in jobs.items()
        if "environment" in job
    }
    assert environments == {"deploy-prod": "prod"}, (
        f"expected a single deploy-prod job bound to the 'prod' environment, got {environments}"
    )

    id_token_writers = {
        name
        for name, job in jobs.items()
        if (job.get("permissions") or {}).get("id-token") == "write"
    }
    assert id_token_writers == {"deploy-prod"}

    deploy_prod = jobs["deploy-prod"]
    needs = deploy_prod.get("needs") or []
    assert "build-proxy-image" in needs
    assert "deploy-dev" not in needs
    assert "deploy-stage" not in needs

    for forbidden_job in ("deploy-dev", "deploy-stage"):
        assert forbidden_job not in jobs, (
            f"{forbidden_job} reappeared; this workflow is intentionally single-environment"
        )

    step_uses = [
        step.get("uses", "")
        for job in jobs.values()
        for step in (job.get("steps") or [])
    ]
    assert step_uses.count("aws-actions/configure-aws-credentials@v5") == 1
    assert step_uses.count("docker/build-push-action@v6") == 1
    assert "actions/upload-artifact@v4" in step_uses
    assert "actions/download-artifact@v5" in step_uses

    workflow_text = WORKFLOW_PATH.read_text()
    assert "role-to-assume: ${{ secrets.GH_ACTIONS_AWS_ROLE }}" in workflow_text
    assert "docker load --input" in workflow_text
    assert "/agr_pdfx_proxy:${{ env.IMAGE_TAG }}" in workflow_text
    assert "./deploy.sh --region \"${AWS_REGION}\" --image-tag \"${IMAGE_TAG}\"" in workflow_text


def test_latest_tag_is_promoted_only_after_ecs_rollout_succeeds():
    data = _load_workflow()
    steps = data["jobs"]["deploy-prod"]["steps"]

    step_names = [step.get("name", "") for step in steps]
    deploy_idx = next(
        i for i, name in enumerate(step_names) if "Register task definition" in name
    )
    latest_idx = next(
        i for i, name in enumerate(step_names) if "Promote :latest" in name
    )

    assert latest_idx > deploy_idx, (
        ":latest must be pushed after the ECS rollout step so a failed rollout "
        "does not leave :latest pointing at an image that never went live"
    )
    assert steps[latest_idx].get("if") == "success()"

    for earlier_step in steps[:latest_idx]:
        run_block = earlier_step.get("run", "") or ""
        assert "agr_pdfx_proxy:latest" not in run_block, (
            f"found :latest push in step {earlier_step.get('name')!r} before ECS rollout"
        )
