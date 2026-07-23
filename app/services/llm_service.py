"""Selection-only OpenAI client for source-backed PDFX decisions."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import Config
from app.prompts import render_prompt
from app.services.source_contracts import (
    CandidateGraph,
    CandidateSelectionRequest,
    CandidateStore,
    RegionSelectionDecision,
    SelectionDecisionResponse,
)
from app.services.model_policy import (
    MAX_BOUNDED_ID_CHOICES,
    MAX_BOUNDED_TARGET_CHOICES,
    resolve_runtime_model,
)


logger = logging.getLogger(__name__)


class CandidateSelectionFailure(RuntimeError):
    """A bounded selector failure that must retain the source baseline."""

    def __init__(
        self,
        reason: Literal[
            "terra_refusal",
            "terra_timeout",
            "no_valid_terra_selection",
        ],
    ):
        super().__init__(reason)
        self.reason = reason


class TokenAccumulator:
    """Per-job, thread-safe token accounting."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[tuple[str, str], dict[str, int]] = {}

    def record(self, usage, call_type: str, model: str) -> None:
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
        with self._lock:
            item = self._data.setdefault(
                (call_type, model),
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "calls": 0,
                },
            )
            item["prompt_tokens"] += prompt
            item["completion_tokens"] += completion
            item["cached_tokens"] += cached
            item["calls"] += 1

    def summary(self) -> dict:
        with self._lock:
            items = {key: dict(value) for key, value in self._data.items()}
        call_models: dict[str, set[str]] = {}
        for call_type, model in items:
            call_models.setdefault(call_type, set()).add(model)
        breakdown = {}
        for (call_type, model), item in items.items():
            key = call_type if len(call_models[call_type]) == 1 else f"{call_type}:{model}"
            breakdown[key] = {"model": model, **item}
        prompt = sum(item["prompt_tokens"] for item in items.values())
        completion = sum(item["completion_tokens"] for item in items.values())
        cached = sum(item["cached_tokens"] for item in items.values())
        return {
            "total_prompt_tokens": prompt,
            "total_completion_tokens": completion,
            "total_cached_tokens": cached,
            "total_tokens": prompt + completion,
            "breakdown": breakdown,
        }

    def tokens_for_types(self, *call_types: str) -> int:
        wanted = set(call_types)
        with self._lock:
            return sum(
                item["prompt_tokens"] + item["completion_tokens"]
                for (call_type, _model), item in self._data.items()
                if call_type in wanted
            )


def compute_cost(summary: dict, pricing: dict) -> tuple[float, dict]:
    """Compute cost from observed usage without inventing missing rates."""

    total = 0.0
    breakdown = {}
    for call_type, item in summary.get("breakdown", {}).items():
        model = item["model"]
        rates = pricing.get(model)
        if rates is None:
            raise ValueError(f"No pricing entry for runtime model {model!r}")
        uncached = max(0, item["prompt_tokens"] - item["cached_tokens"])
        cost = round(
            uncached / 1_000_000 * rates["input"]
            + item["cached_tokens"] / 1_000_000 * rates["cached_input"]
            + item["completion_tokens"] / 1_000_000 * rates["output"],
            6,
        )
        total += cost
        breakdown[call_type] = {**item, "cost_usd": cost}
    total = round(total, 6)
    primary = ""
    if breakdown:
        primary = max(breakdown.values(), key=lambda item: item["calls"])["model"]
    return total, {
        "model_primary": primary,
        "total_prompt_tokens": summary["total_prompt_tokens"],
        "total_completion_tokens": summary["total_completion_tokens"],
        "total_cached_tokens": summary["total_cached_tokens"],
        "total_tokens": summary["total_tokens"],
        "estimated_cost_usd": total,
        "pricing": {
            model: rates
            for model, rates in pricing.items()
            if any(item["model"] == model for item in breakdown.values())
        },
        "breakdown": breakdown,
    }


class _NumberedRegionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    region_id: str = Field(min_length=1)
    choice: int = Field(ge=0, le=MAX_BOUNDED_TARGET_CHOICES)


class _NumberedSelectionResponse(BaseModel):
    """The only publication-merge response: a digest and numbers."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions: tuple[_NumberedRegionDecision, ...]

    @model_validator(mode="after")
    def unique_regions(self):
        region_ids = [decision.region_id for decision in self.decisions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("numbered response repeats a region")
        return self


class ImageTextReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    classification: Literal[
        "scientific_figure",
        "table_or_equation",
        "publisher_logo",
        "badge_or_ui",
        "cover_art",
        "decorative",
        "unknown",
    ]
    is_scientific_figure: bool
    confidence: float = 0.0
    figure_label: str | None = None
    figure_number: str | None = None
    needs_vision_review: bool = False
    reason: str = ""


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, _ModelRefusal):
        return "terra_refusal"
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.casefold():
        return "terra_timeout"
    return "no_valid_terra_selection"


class _ModelRefusal(ValueError):
    pass


class LLM:
    """One bounded client for numbered source choices and image classification."""

    def __init__(
        self,
        api_key,
        openai_timeout_seconds: float | None = None,
        openai_max_retries: int | None = None,
    ):
        timeout = float(
            Config.LLM_OPENAI_TIMEOUT_SECONDS
            if openai_timeout_seconds is None
            else openai_timeout_seconds
        )
        retries = int(
            Config.LLM_OPENAI_MAX_RETRIES
            if openai_max_retries is None
            else openai_max_retries
        )
        if not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise ValueError("OpenAI timeout must be finite and between 0 and 600 seconds")
        if not 0 <= retries <= 2:
            raise ValueError("OpenAI max_retries must be between 0 and 2")
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=retries)
        self.openai_timeout_seconds = timeout
        self.openai_max_retries = retries
        self.usage = TokenAccumulator()
        self.selection_call_traces: list[dict] = []

    @staticmethod
    def _selection_request_trace(payload: dict) -> dict:
        """Retain reconstructable choice evidence without duplicating article text."""

        regions = []
        for region in payload.get("regions", []):
            candidates = []
            for candidate in region.get("candidates", []):
                display = candidate.get("display", "")
                candidates.append(
                    {
                        key: value
                        for key, value in candidate.items()
                        if key != "display"
                    }
                    | {
                        "display_sha256": hashlib.sha256(
                            display.encode("utf-8")
                        ).hexdigest(),
                        "display_utf8_bytes": len(display.encode("utf-8")),
                    }
                )
            regions.append(
                {
                    "region_id": region.get("region_id"),
                    "keep_baseline_choice": region.get("keep_baseline_choice"),
                    "baseline_candidate_id": region.get("baseline_candidate_id"),
                    "candidates": candidates,
                    "path_choices": region.get("path_choices", []),
                }
            )
        return {
            "request_sha256": payload.get("request_sha256"),
            "regions": regions,
        }

    @staticmethod
    def _single_call_usage(usage) -> dict:
        details = getattr(usage, "prompt_tokens_details", None)
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "cached_tokens": getattr(details, "cached_tokens", 0) or 0
            if details
            else 0,
        }

    @staticmethod
    def _numbered_request(request: CandidateSelectionRequest) -> tuple[dict, dict]:
        core = {"regions": []}
        choices: dict[str, dict[int, tuple[str, ...]]] = {}
        for region in request.regions:
            baseline = (region.baseline_candidate_id,)
            paths = [path for path in region.valid_paths if path != baseline]
            choices[region.region_id] = {
                index: path for index, path in enumerate(paths, start=1)
            }
            core["regions"].append(
                {
                    "region_id": region.region_id,
                    "keep_baseline_choice": 0,
                    "baseline_candidate_id": region.baseline_candidate_id,
                    "candidates": [
                        candidate.model_dump(mode="json")
                        for candidate in region.candidates
                    ],
                    "path_choices": [
                        {"choice": index, "candidate_ids": list(path)}
                        for index, path in choices[region.region_id].items()
                    ],
                }
            )
        canonical = json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        return {"request_sha256": digest, **core}, choices

    def resolve_candidate_selections(
        self,
        candidate_store: CandidateStore,
        graph: CandidateGraph,
        *,
        use_sol: bool = False,
        raise_on_failure: bool = False,
    ) -> SelectionDecisionResponse:
        """Select numbered executable paths; never accept publication text."""

        started = time.monotonic()
        trace: dict | None = None
        completion = None
        try:
            request = candidate_store.build_selection_request(graph)
            if not request.regions:
                return SelectionDecisionResponse(decisions=())
            payload, choices = self._numbered_request(request)
            if all(not region_choices for region_choices in choices.values()):
                return SelectionDecisionResponse(
                    decisions=tuple(
                        RegionSelectionDecision(
                            region_id=region.region_id,
                            action="keep_baseline",
                        )
                        for region in request.regions
                    )
                )

            runtime = resolve_runtime_model(
                "hard_selection" if use_sol else "source_selection"
            )
            call_type = "source_path_selection_sol" if use_sol else "source_path_selection_terra"
            trace = {
                "tier": "sol" if use_sol else "terra",
                "call_type": call_type,
                "model": runtime.model,
                "reasoning_effort": runtime.reasoning_effort,
                "request": self._selection_request_trace(payload),
                "timeout_seconds": self.openai_timeout_seconds,
                "max_retries": self.openai_max_retries,
            }
            completion = self.client.chat.completions.parse(
                model=runtime.model,
                reasoning_effort=runtime.reasoning_effort,
                messages=[
                    {"role": "system", "content": render_prompt("source_selection")},
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                ],
                response_format=_NumberedSelectionResponse,
            )
            self.usage.record(completion.usage, call_type, runtime.model)
            message = completion.choices[0].message
            if getattr(message, "refusal", None):
                raise _ModelRefusal("model refused source-path selection")
            if not message.parsed:
                raise ValueError("model returned no parsed source-path selection")
            parsed = _NumberedSelectionResponse.model_validate(
                message.parsed.model_dump(mode="python")
            )
            if parsed.request_sha256 != payload["request_sha256"]:
                raise ValueError("source-path response digest mismatch")
            expected_ids = {region.region_id for region in request.regions}
            if {decision.region_id for decision in parsed.decisions} != expected_ids:
                raise ValueError("source-path response region set mismatch")

            decisions = []
            for decision in parsed.decisions:
                if decision.choice == 0:
                    decisions.append(
                        RegionSelectionDecision(
                            region_id=decision.region_id,
                            action="keep_baseline",
                        )
                    )
                    continue
                try:
                    selected = choices[decision.region_id][decision.choice]
                except KeyError as exc:
                    raise ValueError("source-path response choice is not executable") from exc
                decisions.append(
                    RegionSelectionDecision(
                        region_id=decision.region_id,
                        action="select_candidates",
                        candidate_ids=selected,
                    )
                )
            trace.update(
                {
                    "outcome": "valid",
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                    "usage": self._single_call_usage(completion.usage),
                    "response": parsed.model_dump(mode="json"),
                }
            )
            self.selection_call_traces.append(trace)
            return SelectionDecisionResponse(decisions=tuple(decisions))
        except Exception as exc:
            failure = CandidateSelectionFailure(_failure_reason(exc))
            if trace is not None:
                trace.update(
                    {
                        "outcome": failure.reason,
                        "elapsed_ms": round(
                            (time.monotonic() - started) * 1000, 3
                        ),
                        "error_type": type(exc).__name__,
                    }
                )
                if completion is not None:
                    trace["usage"] = self._single_call_usage(completion.usage)
                self.selection_call_traces.append(trace)
            logger.warning(
                "%s source-path selection failed; retaining baseline: %s",
                "Sol" if use_sol else "Terra",
                type(exc).__name__,
            )
            if raise_on_failure:
                raise failure from exc
            return SelectionDecisionResponse(decisions=())

    def resolve_bounded_id_choice(
        self,
        *,
        reason: str,
        baseline_id: str,
        choices: list[dict],
    ) -> str:
        """Use Sol/high to return one existing application-owned ID by number."""

        if not 1 <= len(choices) <= MAX_BOUNDED_ID_CHOICES:
            raise ValueError(
                "bounded ID choice requires between one and "
                f"{MAX_BOUNDED_ID_CHOICES} choices"
            )
        choice_ids = [item.get("candidate_id") for item in choices]
        if (
            any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in choice_ids)
            or len(choice_ids) != len(set(choice_ids))
            or baseline_id not in choice_ids
        ):
            raise ValueError("bounded ID choices are invalid")
        ordered_ids = [baseline_id] + sorted(
            candidate_id for candidate_id in choice_ids if candidate_id != baseline_id
        )
        by_id = {item["candidate_id"]: dict(item) for item in choices}
        numbered = {index: candidate_id for index, candidate_id in enumerate(ordered_ids[1:], 1)}
        core = {
            "regions": [{
                "region_id": reason,
                "selection_reason": reason,
                "keep_baseline_choice": 0,
                "baseline_candidate_id": baseline_id,
                "candidates": [
                    {
                        **by_id[candidate_id],
                        "display": str(by_id[candidate_id].get("display", "")),
                    }
                    for candidate_id in ordered_ids
                ],
                "path_choices": [
                    {"choice": number, "candidate_ids": [candidate_id]}
                    for number, candidate_id in numbered.items()
                ],
            }]
        }
        canonical = json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = {"request_sha256": hashlib.sha256(canonical).hexdigest(), **core}
        started = time.monotonic()
        trace = {
            "tier": "sol",
            "call_type": "bounded_id_selection_sol",
            "model": None,
            "reasoning_effort": None,
            "request": self._selection_request_trace(payload),
            "timeout_seconds": self.openai_timeout_seconds,
            "max_retries": self.openai_max_retries,
        }
        completion = None
        try:
            runtime = resolve_runtime_model("hard_selection")
            trace["model"] = runtime.model
            trace["reasoning_effort"] = runtime.reasoning_effort
            completion = self.client.chat.completions.parse(
                model=runtime.model,
                reasoning_effort=runtime.reasoning_effort,
                messages=[
                    {"role": "system", "content": render_prompt("source_selection")},
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                ],
                response_format=_NumberedSelectionResponse,
            )
            self.usage.record(completion.usage, "bounded_id_selection_sol", runtime.model)
            message = completion.choices[0].message
            if getattr(message, "refusal", None):
                raise _ModelRefusal("model refused bounded ID selection")
            if not message.parsed:
                raise ValueError("model returned no bounded ID selection")
            parsed = _NumberedSelectionResponse.model_validate(
                message.parsed.model_dump(mode="python")
            )
            if parsed.request_sha256 != payload["request_sha256"]:
                raise ValueError("bounded ID response digest mismatch")
            if len(parsed.decisions) != 1 or parsed.decisions[0].region_id != reason:
                raise ValueError("bounded ID response region mismatch")
            choice = parsed.decisions[0].choice
            selected_id = baseline_id if choice == 0 else numbered.get(choice)
            if selected_id is None:
                raise ValueError("bounded ID response choice is not executable")
            trace.update(
                {
                    "outcome": "valid",
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                    "usage": self._single_call_usage(completion.usage),
                    "response": parsed.model_dump(mode="json"),
                }
            )
            self.selection_call_traces.append(trace)
            return selected_id
        except Exception as exc:
            failure = CandidateSelectionFailure(_failure_reason(exc))
            trace.update(
                {
                    "outcome": failure.reason,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                    "error_type": type(exc).__name__,
                }
            )
            if completion is not None:
                trace["usage"] = self._single_call_usage(completion.usage)
            self.selection_call_traces.append(trace)
            raise failure from exc

    def resolve_bounded_id_choice_with_receipt(
        self,
        *,
        reason: str,
        baseline_id: str,
        choices: list[dict],
    ) -> dict[str, object]:
        """Return one existing ID together with its executable numeric trace binding."""

        trace_count = len(self.selection_call_traces)
        selected_id = self.resolve_bounded_id_choice(
            reason=reason,
            baseline_id=baseline_id,
            choices=choices,
        )
        if len(self.selection_call_traces) != trace_count + 1:
            raise RuntimeError("bounded ID selection did not emit exactly one trace")
        trace = self.selection_call_traces[-1]
        request = trace.get("request")
        response = trace.get("response")
        decisions = response.get("decisions") if isinstance(response, dict) else None
        if (
            trace.get("call_type") != "bounded_id_selection_sol"
            or trace.get("outcome") != "valid"
            or not isinstance(request, dict)
            or not isinstance(request.get("request_sha256"), str)
            or not isinstance(decisions, list)
            or len(decisions) != 1
            or decisions[0].get("region_id") != reason
            or type(decisions[0].get("choice")) is not int
        ):
            raise RuntimeError("bounded ID selection trace is incomplete")
        return {
            "selected_candidate_id": selected_id,
            "request_sha256": request["request_sha256"],
            "response_choice": decisions[0]["choice"],
            "model": trace.get("model"),
            "reasoning_effort": trace.get("reasoning_effort"),
        }

    def review_image_context(self, image: dict) -> ImageTextReviewResponse:
        """Classify image metadata; the response never enters publication text."""

        runtime = resolve_runtime_model("image_text_review")
        payload = {
            key: image.get(key)
            for key in (
                "filename",
                "page_index",
                "marker_image_type",
                "marker_image_index",
                "block_id",
                "group_id",
                "bbox",
                "image_width",
                "image_height",
                "diagnostic_flags",
                "alt_text",
                "caption_text",
                "nearby_text",
            )
        }
        completion = self.client.chat.completions.parse(
            model=runtime.model,
            reasoning_effort=runtime.reasoning_effort,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify metadata for one scientific-PDF image. You do not see "
                        "pixels. Identify scientific figures conservatively; logos, badges, "
                        "covers, UI, and decoration are not figures. Return labels only when "
                        "explicitly present in supplied text."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format=ImageTextReviewResponse,
        )
        self.usage.record(completion.usage, "image_text_review", runtime.model)
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise ValueError("model refused image metadata classification")
        if not message.parsed:
            raise ValueError("model returned no image metadata classification")
        return ImageTextReviewResponse.model_validate(
            message.parsed.model_dump(mode="python")
        )
