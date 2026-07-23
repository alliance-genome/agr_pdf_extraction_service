import pytest

from config import Config
from app.services.model_policy import (
    ALLOWED_RUNTIME_MODELS,
    RuntimeModelPolicyError,
    resolve_runtime_model,
    resolved_runtime_model_map,
    validate_runtime_model_policy,
)


def test_runtime_model_map_contains_only_three_exact_5_6_routes():
    role_map = resolved_runtime_model_map()

    assert role_map == {
        "source_selection": {
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
        },
        "hard_selection": {
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
        },
        "image_text_review": {
            "model": "gpt-5.6-luna",
            "reasoning_effort": "medium",
        },
    }
    assert {item["model"] for item in role_map.values()} == ALLOWED_RUNTIME_MODELS


@pytest.mark.parametrize(
    "role", ["source_selection", "hard_selection", "image_text_review"]
)
def test_every_route_rejects_old_runtime_model(role):
    with pytest.raises(RuntimeModelPolicyError, match="disallowed runtime model"):
        resolve_runtime_model(role, model="gpt-5.5")


def test_route_rejects_wrong_5_6_tier_and_reasoning():
    with pytest.raises(RuntimeModelPolicyError, match="requires 'gpt-5.6-terra'"):
        resolve_runtime_model("source_selection", model="gpt-5.6-sol")
    with pytest.raises(RuntimeModelPolicyError, match="requires 'high' reasoning"):
        resolve_runtime_model("hard_selection", reasoning_effort="medium")


def test_environment_override_fails_policy(monkeypatch):
    monkeypatch.setattr(Config, "SOURCE_SELECTION_MODEL", "gpt-5.5")
    with pytest.raises(RuntimeModelPolicyError, match="disallowed runtime model"):
        resolved_runtime_model_map()


def test_startup_policy_rejects_incomplete_pricing(monkeypatch):
    monkeypatch.setattr(
        Config,
        "LLM_PRICING",
        {"gpt-5.6-terra": Config.LLM_PRICING["gpt-5.6-terra"]},
    )
    with pytest.raises(RuntimeModelPolicyError, match="pricing must cover exactly"):
        validate_runtime_model_policy()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), 601])
def test_startup_policy_rejects_unbounded_openai_timeout(monkeypatch, timeout):
    monkeypatch.setattr(Config, "LLM_OPENAI_TIMEOUT_SECONDS", timeout)
    with pytest.raises(RuntimeModelPolicyError, match="TIMEOUT_SECONDS"):
        validate_runtime_model_policy()


@pytest.mark.parametrize("retries", [-1, 3])
def test_startup_policy_rejects_excessive_transport_retries(monkeypatch, retries):
    monkeypatch.setattr(Config, "LLM_OPENAI_MAX_RETRIES", retries)
    with pytest.raises(RuntimeModelPolicyError, match="MAX_RETRIES"):
        validate_runtime_model_policy()


@pytest.mark.parametrize("threshold", [0, -1, float("inf")])
def test_startup_policy_rejects_invalid_cost_alert_threshold(
    monkeypatch, threshold
):
    monkeypatch.setattr(Config, "LLM_COST_ALERT_USD_PER_JOB", threshold)
    with pytest.raises(RuntimeModelPolicyError, match="COST_ALERT_USD_PER_JOB"):
        validate_runtime_model_policy()
