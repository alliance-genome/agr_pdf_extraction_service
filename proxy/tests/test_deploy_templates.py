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
