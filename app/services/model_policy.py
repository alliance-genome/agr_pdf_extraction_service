"""Exact GPT-5.6 policy for the three reachable PDFX model routes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from config import Config


ModelRole = Literal["source_selection", "hard_selection", "image_text_review"]
GPT_5_6_SOL = "gpt-5.6-sol"
GPT_5_6_TERRA = "gpt-5.6-terra"
GPT_5_6_LUNA = "gpt-5.6-luna"
ALLOWED_RUNTIME_MODELS = frozenset({GPT_5_6_SOL, GPT_5_6_TERRA, GPT_5_6_LUNA})
ALLOWED_REASONING_EFFORTS = frozenset({"medium", "high"})
MAX_BOUNDED_ID_CHOICES = 64
MAX_BOUNDED_TARGET_CHOICES = MAX_BOUNDED_ID_CHOICES - 1

_ROLE_CONFIG = {
    "source_selection": (
        "SOURCE_SELECTION_MODEL",
        "SOURCE_SELECTION_REASONING",
        GPT_5_6_TERRA,
        "medium",
    ),
    "hard_selection": (
        "HARD_SELECTION_MODEL",
        "HARD_SELECTION_REASONING",
        GPT_5_6_SOL,
        "high",
    ),
    "image_text_review": (
        "IMAGE_TEXT_REVIEW_MODEL",
        "IMAGE_TEXT_REVIEW_REASONING",
        GPT_5_6_LUNA,
        "medium",
    ),
}


class RuntimeModelPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedRuntimeModel:
    role: ModelRole
    model: str
    reasoning_effort: str


def resolve_runtime_model(
    role: ModelRole,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> ResolvedRuntimeModel:
    try:
        model_attr, effort_attr, required_model, required_effort = _ROLE_CONFIG[role]
    except KeyError as exc:
        raise RuntimeModelPolicyError(f"unknown model role: {role!r}") from exc
    resolved_model = model or str(getattr(Config, model_attr, "")).strip()
    resolved_effort = reasoning_effort or str(getattr(Config, effort_attr, "")).strip()
    if resolved_model not in ALLOWED_RUNTIME_MODELS:
        raise RuntimeModelPolicyError(
            f"role {role!r} resolved disallowed runtime model {resolved_model!r}"
        )
    if resolved_model != required_model:
        raise RuntimeModelPolicyError(
            f"role {role!r} requires {required_model!r}, got {resolved_model!r}"
        )
    if resolved_effort not in ALLOWED_REASONING_EFFORTS:
        raise RuntimeModelPolicyError(
            f"role {role!r} resolved invalid reasoning effort {resolved_effort!r}"
        )
    if resolved_effort != required_effort:
        raise RuntimeModelPolicyError(
            f"role {role!r} requires {required_effort!r} reasoning, got {resolved_effort!r}"
        )
    return ResolvedRuntimeModel(role, resolved_model, resolved_effort)


def resolved_runtime_model_map() -> dict[str, dict[str, str]]:
    return {
        role: {
            "model": resolved.model,
            "reasoning_effort": resolved.reasoning_effort,
        }
        for role in _ROLE_CONFIG
        for resolved in (resolve_runtime_model(role),)
    }


def validate_runtime_model_policy() -> None:
    resolved_runtime_model_map()
    timeout = Config.LLM_OPENAI_TIMEOUT_SECONDS
    if not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise RuntimeModelPolicyError(
            "LLM_OPENAI_TIMEOUT_SECONDS must be finite and between 0 and 600"
        )
    if not 0 <= Config.LLM_OPENAI_MAX_RETRIES <= 2:
        raise RuntimeModelPolicyError("LLM_OPENAI_MAX_RETRIES must be between 0 and 2")
    cost_alert = Config.LLM_COST_ALERT_USD_PER_JOB
    if not math.isfinite(cost_alert) or cost_alert <= 0:
        raise RuntimeModelPolicyError(
            "LLM_COST_ALERT_USD_PER_JOB must be finite and greater than zero"
        )
    if frozenset(Config.LLM_PRICING) != ALLOWED_RUNTIME_MODELS:
        raise RuntimeModelPolicyError(
            "LLM pricing must cover exactly the release-approved runtime models"
        )
    for model, rates in Config.LLM_PRICING.items():
        keys = ("input", "output", "cached_input")
        if any(key not in rates or rates[key] < 0 for key in keys):
            raise RuntimeModelPolicyError(f"LLM pricing for {model!r} is invalid")
