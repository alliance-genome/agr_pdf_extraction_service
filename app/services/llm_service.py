"""OpenAI-backed LLM service used by consensus and hierarchy resolution flows."""

import json
import logging
import threading
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from pydantic import BaseModel

from config import Config
from app.services.pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)

# Reasoning effort escalation order for retries (lightest → heaviest)
REASONING_ESCALATION_ORDER = ["low", "medium", "high"]


class TokenAccumulator:
    """Thread-safe accumulator for OpenAI token usage across LLM calls.

    Records prompt_tokens, completion_tokens, and cached_tokens per
    (call_type, model) key. One instance per LLM (i.e. per extraction run).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[tuple[str, str], dict] = {}

    def record(self, usage, call_type: str, model: str) -> None:
        """Record token usage from an OpenAI completion response."""
        if usage is None:
            return

        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            cached = getattr(details, "cached_tokens", 0) or 0

        key = (call_type, model)
        with self._lock:
            if key not in self._data:
                self._data[key] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "calls": 0,
                }
            entry = self._data[key]
            entry["prompt_tokens"] += prompt
            entry["completion_tokens"] += completion
            entry["cached_tokens"] += cached
            entry["calls"] += 1

    def summary(self) -> dict:
        """Return a full summary of accumulated usage."""
        with self._lock:
            total_prompt = 0
            total_completion = 0
            total_cached = 0
            breakdown = {}

            # Check for call_type collisions (same call_type, different models)
            call_type_models: dict[str, set[str]] = {}
            for (call_type, model) in self._data:
                call_type_models.setdefault(call_type, set()).add(model)

            for (call_type, model), entry in self._data.items():
                total_prompt += entry["prompt_tokens"]
                total_completion += entry["completion_tokens"]
                total_cached += entry["cached_tokens"]

                # Use "call_type:model" key if multiple models share a call_type
                key = call_type
                if len(call_type_models.get(call_type, set())) > 1:
                    key = f"{call_type}:{model}"

                breakdown[key] = {
                    "model": model,
                    "prompt_tokens": entry["prompt_tokens"],
                    "completion_tokens": entry["completion_tokens"],
                    "cached_tokens": entry["cached_tokens"],
                    "calls": entry["calls"],
                }

            return {
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_cached_tokens": total_cached,
                "total_tokens": total_prompt + total_completion,
                "breakdown": breakdown,
            }

    def tokens_for_types(self, *call_types: str) -> int:
        """Return total tokens (prompt + completion) for given call types."""
        with self._lock:
            total = 0
            for (ct, _model), entry in self._data.items():
                if ct in call_types:
                    total += entry["prompt_tokens"] + entry["completion_tokens"]
            return total


def compute_cost(summary: dict, pricing: dict) -> tuple[float, dict]:
    """Compute USD cost from a TokenAccumulator summary and pricing dict.

    Args:
        summary: Output of TokenAccumulator.summary().
        pricing: Dict mapping model name to {input, output, cached_input} rates per 1M tokens.

    Returns:
        (total_cost_usd, usage_json) — total_cost_usd is a rounded float;
        usage_json is the full breakdown dict ready for DB storage.
    """
    breakdown = summary.get("breakdown", {})
    total_cost = 0.0
    cost_breakdown = {}

    for call_type, entry in breakdown.items():
        model = entry["model"]
        rates = pricing.get(model)
        if rates is None:
            logger.warning(
                "No pricing entry for model %r — cost will be underreported. "
                "Add it to LLM_PRICING in config.py.",
                model,
            )
            rates = {"input": 0, "output": 0, "cached_input": 0}

        prompt = entry["prompt_tokens"]
        cached = entry["cached_tokens"]
        uncached = max(0, prompt - cached)
        completion = entry["completion_tokens"]

        cost = (
            (uncached / 1_000_000) * rates["input"]
            + (cached / 1_000_000) * rates["cached_input"]
            + (completion / 1_000_000) * rates["output"]
        )
        cost = round(cost, 6)
        total_cost += cost

        cost_breakdown[call_type] = {
            "model": model,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_tokens": cached,
            "calls": entry["calls"],
            "cost_usd": cost,
        }

    # Determine primary model (the one with most calls)
    primary_model = ""
    if cost_breakdown:
        primary_model = max(
            cost_breakdown.values(), key=lambda v: v["calls"],
        )["model"]

    total_cost = round(total_cost, 6)

    usage_json = {
        "model_primary": primary_model,
        "total_prompt_tokens": summary["total_prompt_tokens"],
        "total_completion_tokens": summary["total_completion_tokens"],
        "total_cached_tokens": summary["total_cached_tokens"],
        "total_tokens": summary["total_tokens"],
        "estimated_cost_usd": total_cost,
        "pricing": {
            model: rates
            for model, rates in pricing.items()
            if any(e["model"] == model for e in cost_breakdown.values())
        },
        "breakdown": cost_breakdown,
    }

    return total_cost, usage_json


class ResolvedSegment(BaseModel):
    """A single resolved conflict segment.

    The ``action`` field makes the LLM's intent explicit:
    - "keep"  → resolved text is provided (use it)
    - "drop"  → segment should be excluded from final output (text must be empty)

    This eliminates the ambiguity of empty-string responses where we previously
    could not distinguish "intentionally dropped" from "failed to produce output".
    """

    segment_id: str
    action: Literal["keep", "drop"]
    text: str


class ConflictResolutionResponse(BaseModel):
    """Structured conflict-resolution response.

    Uses a list of {segment_id, text} objects instead of a dict because
    OpenAI's strict structured output mode does not support dynamic keys
    (dict[str, str] generates additionalProperties which is rejected).
    """

    resolved: list[ResolvedSegment]


class ResolvedMicroConflict(BaseModel):
    """A single resolved micro-conflict span."""

    conflict_id: str
    text: str = ""
    action: Literal["keep", "drop"] = "keep"


class MicroConflictResolutionResponse(BaseModel):
    """Structured output for micro-conflict resolution."""

    resolved: list[ResolvedMicroConflict]


class RescueSegmentResponse(BaseModel):
    """Response from a rescue resolution call for a single failed segment."""

    resolved_text: str  # The resolved text, or empty string if intentionally dropped
    is_intentionally_empty: bool  # True if the LLM determined this segment SHOULD be empty
    explanation: str  # Why the segment was resolved this way (ALWAYS required)


class HeaderDecision(BaseModel):
    """Classification decision for a single heading line."""

    heading_index: int  # 0-based index in the list of extracted headers
    original_text: str  # exact heading text (for validation/debugging)
    action: Literal["keep_level", "set_level", "demote_to_text"]
    new_level: int | None = None  # 1-6, required when action=set_level


class HeaderHierarchyResponse(BaseModel):
    """Structured output for header hierarchy resolution."""

    decisions: list[HeaderDecision]
    detected_title: str | None = None  # paper title found in opening text but not in headings


class AlignmentTieBreakResponse(BaseModel):
    """Structured output for alignment tie-break selection."""

    choice: Literal["candidate_a", "candidate_b"]
    confidence: float = 0.0
    explanation: str = ""


class LLM(PDFExtractor):
    def __init__(
        self,
        api_key,
        model="gpt-5.2",
        reasoning_effort="low",
        conflict_batch_size: int = 500,
        conflict_max_workers: int = 100,
        conflict_retry_rounds: int = 2,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.conflict_batch_size = max(1, int(conflict_batch_size))
        self.conflict_max_workers = max(1, int(conflict_max_workers))
        self.conflict_retry_rounds = max(1, int(conflict_retry_rounds))
        self.usage = TokenAccumulator()

    @staticmethod
    def _next_reasoning_effort(current_effort: str) -> str | None:
        """Return the next higher reasoning effort, or None if already at highest."""
        try:
            idx = REASONING_ESCALATION_ORDER.index(current_effort)
            if idx + 1 < len(REASONING_ESCALATION_ORDER):
                return REASONING_ESCALATION_ORDER[idx + 1]
        except ValueError:
            pass
        return None

    def _resolve_conflict_batch(self, batch: list[dict]) -> tuple[dict[str, str], set[str]]:
        """Resolve one conflict batch, returning (resolved_map, unresolved_ids)."""
        use_model = Config.LLM_MODEL_CONFLICT_BATCH
        use_reasoning = Config.LLM_REASONING_CONFLICT_BATCH or self.reasoning_effort
        expected_ids = {c["segment_id"] for c in batch}
        prompt_payload = json.dumps({"conflicts": batch})
        system_msg = (
            "You are resolving conflicts between three PDF extraction tools "
            "(GROBID, Docling, Marker) that processed the same scientific paper. "
            "Each conflict has the same passage as seen by each tool. "
            "For each conflict, pick the most accurate and complete version, "
            "or merge the best parts from each.\n\n"
            "IMPORTANT: Each conflict includes 'context_before' and 'context_after' "
            "fields. These are the surrounding text (read-only). Do NOT repeat or "
            "include this context in your output. Your resolved text must flow "
            "naturally between context_before and context_after, but you must only "
            "return the text for the conflict segment itself.\n\n"
            "SPECIAL CHARACTERS:\n"
            "- Use actual Unicode Greek letters (α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, "
            "ν, ξ, π, ρ, σ, τ, υ, φ, χ, ψ, ω and uppercase equivalents), NOT LaTeX "
            "notation ($\\alpha$, $\\beta$, etc.).\n"
            "- Example: write β-ENaC, not $\\beta$-ENaC. Write Aβ42, not A$\\beta$42.\n"
            "- Exception: preserve LaTeX only inside actual math expressions "
            "(e.g. $E = mc^2$ or $$\\sum_{i=1}^n x_i$$).\n"
            "- Use Unicode superscripts/subscripts where appropriate (e.g. Ca²⁺, Na⁺, "
            "μm, °C) rather than LaTeX equivalents.\n\n"
            "Return a JSON object with a 'resolved' key containing a list of "
            "objects, each with 'segment_id', 'action', and 'text' keys.\n"
            "- Set action=\"keep\" and provide the resolved text for segments "
            "that should appear in the final document.\n"
            "- Set action=\"drop\" with empty text only for segments that are "
            "artifacts, duplicates, or noise that should be excluded.\n"
            "The 'text' value is the resolved markdown for that segment."
        )

        completion = self.client.chat.completions.parse(
            model=use_model,
            reasoning_effort=use_reasoning,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_payload},
            ],
            response_format=ConflictResolutionResponse,
        )

        usage = completion.usage
        self.usage.record(usage, "conflict_batch", use_model)
        if usage:
            logger.info(
                "LLM conflict resolution: model=%s, tokens=%d (prompt=%d, completion=%d)",
                use_model, usage.total_tokens, usage.prompt_tokens, usage.completion_tokens,
                extra={
                    "_event": "llm_resolve_complete",
                    "_llm_model": use_model,
                    "_llm_tokens_used": usage.total_tokens,
                },
            )

        message = completion.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValueError(f"Model refused: {refusal}")

        if not message.parsed:
            raise ValueError("No parsed response from model")

        raw_map = {seg.segment_id: seg for seg in message.parsed.resolved}
        resolved: dict[str, str] = {}
        unresolved = set(expected_ids)
        for seg_id in expected_ids:
            seg_result = raw_map.get(seg_id)
            if seg_result is None:
                continue  # stays in unresolved

            if seg_result.action == "keep":
                if seg_result.text.strip():
                    resolved[seg_id] = seg_result.text.strip()
                    unresolved.discard(seg_id)
                # else: keep+empty → stays unresolved for retry
            elif seg_result.action == "drop":
                if seg_result.text.strip():
                    # Said drop but provided text — use it
                    resolved[seg_id] = seg_result.text.strip()
                    unresolved.discard(seg_id)
                else:
                    resolved[seg_id] = ""
                    unresolved.discard(seg_id)

        return resolved, unresolved

    def _run_conflict_batches(self, batches: list[list[dict]]) -> tuple[dict[str, str], set[str]]:
        """Run conflict batches in parallel and aggregate resolved/unresolved IDs."""
        resolved: dict[str, str] = {}
        unresolved: set[str] = set()
        if not batches:
            return resolved, unresolved

        max_workers = min(self.conflict_max_workers, len(batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._resolve_conflict_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                batch = futures[future]
                batch_ids = {c["segment_id"] for c in batch}
                try:
                    batch_resolved, batch_unresolved = future.result()
                    resolved.update(batch_resolved)
                    unresolved.update(batch_unresolved)
                except Exception as exc:
                    logger.warning(
                        "resolve_conflicts batch failed for %d segments: %s",
                        len(batch), exc,
                    )
                    unresolved.update(batch_ids)
        return resolved, unresolved

    def resolve_conflicts(self, conflicts: list[dict]) -> dict[str, str]:
        """Resolve conflicts in parallel batches with targeted retries."""
        if not conflicts:
            return {}

        by_id = {c["segment_id"]: c for c in conflicts}
        pending_ids = list(by_id.keys())
        resolved_all: dict[str, str] = {}
        last_error: Exception | None = None

        for round_idx in range(self.conflict_retry_rounds + 1):
            if not pending_ids:
                break

            # Halve batch size on retries: smaller batches reduce blast radius
            # and give the LLM fewer segments to juggle, improving success rate.
            round_size = self.conflict_batch_size if round_idx == 0 else max(1, self.conflict_batch_size // 2)
            round_conflicts = [by_id[sid] for sid in pending_ids]
            batches = [
                round_conflicts[i:i + round_size]
                for i in range(0, len(round_conflicts), round_size)
            ]
            logger.info(
                "resolve_conflicts round %d: %d segments in %d batch(es)",
                round_idx + 1, len(round_conflicts), len(batches),
            )

            round_resolved, round_unresolved = self._run_conflict_batches(batches)
            resolved_all.update(round_resolved)
            pending_ids = [sid for sid in pending_ids if sid in round_unresolved]

            if pending_ids and round_idx < self.conflict_retry_rounds:
                logger.warning(
                    "resolve_conflicts round %d left %d unresolved segments; retrying only failed IDs",
                    round_idx + 1, len(pending_ids),
                )
                continue

            if pending_ids:
                last_error = ValueError(f"Unresolved segment_ids after retries: {sorted(pending_ids)}")

        if pending_ids:
            raise Exception(
                "resolve_conflicts failed after batched retries"
                + (f": {last_error}" if last_error else "")
            )

        return {c["segment_id"]: resolved_all[c["segment_id"]] for c in conflicts}

    def resolve_micro_conflicts(
        self,
        payload: dict,
        model: str | None = None,
    ) -> MicroConflictResolutionResponse:
        """Resolve one segment's micro-conflicts with targeted retries/escalation."""
        micro_conflicts = payload.get("micro_conflicts") or []
        if not micro_conflicts:
            return MicroConflictResolutionResponse(resolved=[])

        expected_ids = [mc.get("conflict_id", "") for mc in micro_conflicts if mc.get("conflict_id")]
        if not expected_ids:
            return MicroConflictResolutionResponse(resolved=[])

        resolved_map: dict[str, str] = {}
        pending_ids = list(expected_ids)
        current_model = model or Config.LLM_MODEL_ZONE_RESOLUTION
        use_reasoning = Config.LLM_REASONING_ZONE_RESOLUTION or self.reasoning_effort
        system_msg = (
            "You are resolving small disagreements between three PDF extraction tools "
            "(GROBID, Docling, Marker) that processed the same scientific paper.\n\n"
            "You will receive one or more MICRO-CONFLICTS. Each shows a small region "
            "where the tools disagree, surrounded by context where they agree.\n\n"
            "FORMAT:\n"
            "- conflict_id: unique identifier\n"
            "- context_before: agreed text before the disagreement (read-only)\n"
            "- disagreement: what each extractor has (some may be empty)\n"
            "- context_after: agreed text after the disagreement (read-only)\n\n"
            "RULES:\n"
            "1. Pick the most accurate version or merge the best parts.\n"
            "2. Return ONLY the resolved text for the disagreement span.\n"
            "3. Do NOT repeat context_before or context_after.\n"
            "4. Preserve ALL numbers exactly as they appear in sources.\n"
            "5. Use actual Unicode Greek letters (α, β, γ, δ, ε, μ, etc.), NOT LaTeX "
            "notation ($\\alpha$, $\\beta$, etc.). Example: β-ENaC not $\\beta$-ENaC.\n"
            "6. When sources disagree on a number, pick one verbatim — do NOT invent.\n"
            "7. Prefer selecting from source text over rewriting.\n"
            "8. For table cell disagreements, preserve markdown table formatting.\n"
            "9. For equation disagreements, preserve LaTeX/mathematical notation.\n"
            "10. Use Unicode superscripts/subscripts where appropriate (Ca²⁺, Na⁺, μm, °C).\n\n"
            "Return JSON: {\"resolved\": [{\"conflict_id\": \"...\", \"text\": \"...\", \"action\": \"keep\"}, ...]}\n"
            "Set action=\"keep\" (default) with resolved text for normal spans.\n"
            "Set action=\"drop\" with empty text to intentionally delete a span."
        )

        for attempt in range(self.conflict_retry_rounds + 1):
            if not pending_ids:
                break

            attempt_payload = dict(payload)
            attempt_payload["micro_conflicts"] = [
                mc for mc in micro_conflicts if mc.get("conflict_id") in set(pending_ids)
            ]
            try:
                completion = self.client.chat.completions.parse(
                    model=current_model,
                    reasoning_effort=use_reasoning,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": json.dumps(attempt_payload)},
                    ],
                    response_format=MicroConflictResolutionResponse,
                )

                usage = completion.usage
                self.usage.record(usage, "micro_conflict", current_model)
                if usage:
                    logger.info(
                        "LLM micro-conflict resolution: model=%s, tokens=%d (prompt=%d, completion=%d)",
                        current_model,
                        usage.total_tokens,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                    )

                message = completion.choices[0].message
                refusal = getattr(message, "refusal", None)
                if refusal:
                    raise ValueError(f"Model refused: {refusal}")
                if not message.parsed:
                    raise ValueError("No parsed response from model")

                for item in message.parsed.resolved:
                    conflict_id = (item.conflict_id or "").strip()
                    text = (item.text or "").strip()
                    if conflict_id in pending_ids and (text or item.action == "drop"):
                        resolved_map[conflict_id] = text  # empty string for drops

            except Exception as exc:
                logger.warning(
                    "Micro-conflict call failed (attempt=%d, model=%s, reasoning=%s): %s",
                    attempt + 1, current_model, use_reasoning, exc,
                )
                if attempt < self.conflict_retry_rounds:
                    next_reasoning = self._next_reasoning_effort(use_reasoning)
                    if next_reasoning:
                        use_reasoning = next_reasoning
                    continue
                break

            pending_ids = [cid for cid in pending_ids if cid not in resolved_map]
            if pending_ids and attempt < self.conflict_retry_rounds:
                next_reasoning = self._next_reasoning_effort(use_reasoning)
                if next_reasoning:
                    logger.info(
                        "Escalating micro-conflict reasoning: %s -> %s (model=%s)",
                        use_reasoning, next_reasoning, current_model,
                    )
                    use_reasoning = next_reasoning

        if pending_ids:
            logger.warning(
                "Unresolved micro_conflicts after %d retries: %s",
                self.conflict_retry_rounds + 1, sorted(pending_ids),
            )

        resolved_items = [
            ResolvedMicroConflict(conflict_id=cid, text=resolved_map.get(cid, ""), action="keep" if resolved_map.get(cid) else "drop")
            for cid in expected_ids
            if cid in resolved_map
        ]
        return MicroConflictResolutionResponse(resolved=resolved_items)

    def rescue_single_segment(
        self,
        seg: dict,
        zone: dict,
        neighboring_resolved: dict[str, str],
        extra_flanking: list[dict],
        model: str | None = None,
        reasoning_effort: str | None = None,
        reason: str | None = None,
        previous_text: str | None = None,
        novel_numbers: list[str] | None = None,
    ) -> RescueSegmentResponse | None:
        """Focused rescue call for a single segment that zone resolution returned empty.

        This gives the LLM a second chance with:
        - A clear explanation of what happened (zone resolution returned empty)
        - Just the ONE segment to focus on (not the whole zone)
        - Extra flanking context (more than the zone had)
        - Neighboring resolved segments so it sees document flow
        - An explicit ask: provide text OR explain why it should be empty

        Args:
            model: Override model for this call (defaults to self.model).
            reasoning_effort: Override reasoning effort for this call.

        Returns a RescueSegmentResponse, or None if the API call itself fails.
        """
        use_model = model or self.model
        use_reasoning = reasoning_effort or self.reasoning_effort
        seg_id = seg["segment_id"]
        seg_status = seg["status"]

        # Build the source texts display
        source_lines = []
        has_any_extractor_text = False
        for source_name in ("grobid", "docling", "marker"):
            text = (seg.get(source_name) or "").strip()
            if text:
                has_any_extractor_text = True
                source_lines.append(f"### {source_name.upper()}\n{text}")
            else:
                source_lines.append(f"### {source_name.upper()}\n(no output from this extractor)")
        if not has_any_extractor_text and seg_status == "gap":
            gap_text = (seg.get("text") or "").strip()
            gap_source = (seg.get("gap_source") or "unknown").strip()
            if gap_text:
                source_lines.append(f"### GAP ({gap_source})\n{gap_text}")
        sources_display = "\n\n".join(source_lines)

        # Build neighboring context display
        context_lines = []
        for ctx in extra_flanking:
            ctx_text = ctx.get("text", "").strip()
            if ctx_text:
                label = ctx.get("segment_id", "?")
                context_lines.append(f"[{label}]: {ctx_text[:500]}")
        for nid, ntext in neighboring_resolved.items():
            if ntext.strip():
                context_lines.append(f"[{nid} — already resolved]: {ntext[:500]}")
        context_display = (
            "\n".join(context_lines) if context_lines
            else "(no surrounding context available)"
        )

        reason = (reason or "empty").strip().lower()
        if reason == "numeric_integrity":
            numeric_note = ""
            if novel_numbers:
                numeric_note = (
                    "\n\nNUMERIC INTEGRITY ISSUE:\n"
                    "A previous resolution for this segment introduced number(s) not present in any input source.\n"
                    f"Novel numbers detected: {', '.join(novel_numbers)}\n"
                )
            prev_note = ""
            if previous_text and previous_text.strip():
                prev_note = f"\n\nPREVIOUS OUTPUT (do not trust blindly):\n{previous_text.strip()}\n"

            system_msg = (
                "You are resolving a SINGLE segment from a scientific paper where PDF extraction "
                "tools (GROBID, Docling, Marker) produced different text.\n\n"
                "IMPORTANT CONTEXT: This segment was previously resolved, but the output triggered a "
                "numeric-integrity guard (it contained numbers not present in any source).\n"
                "This is a CRITICAL scientific-accuracy issue.\n"
                f"{numeric_note}"
                f"{prev_note}\n"
                "RULES:\n"
                "1. Do NOT introduce any new numbers. Every number in your output must appear in at least one source.\n"
                "2. You MAY delete numbers that appear to be extractor artifacts (but explain why).\n"
                "3. If sources disagree on a number, choose one that appears in a source and explain which source you followed.\n"
                "4. If status is conflict/near_agree, you must provide non-empty resolved_text.\n"
                "5. If status is gap, you may set is_intentionally_empty=true ONLY if you justify why it should be dropped.\n"
                "6. Use actual Unicode Greek letters (α, β, γ, δ, ε, μ, etc.), NOT LaTeX "
                "notation ($\\alpha$, $\\beta$, etc.). Example: β-ENaC not $\\beta$-ENaC. "
                "Only preserve LaTeX inside actual math expressions.\n"
                "You MUST provide an explanation in ALL cases.\n\n"
                "SURROUNDING DOCUMENT CONTEXT (for understanding flow — do NOT repeat these):\n"
                f"{context_display}\n\n"
                f"SEGMENT TO RESOLVE ({seg_id}, status: {seg_status}):\n\n"
                f"{sources_display}"
            )
        else:
            system_msg = (
                "You are resolving a SINGLE segment from a scientific paper where three PDF extraction "
                "tools (GROBID, Docling, Marker) produced different text.\n\n"
                "IMPORTANT CONTEXT: This segment was previously sent to you as part of a larger conflict "
                "zone, and you returned EMPTY text for it. We need you to look at it again carefully.\n\n"
                "You have two valid options:\n"
                "1. PROVIDE RESOLVED TEXT: Pick the best version from the sources below, merge them, "
                "   or clean one up. Set is_intentionally_empty=false and put the text in resolved_text.\n"
                "2. EXPLAIN WHY IT SHOULD BE EMPTY: If this segment genuinely should not appear in the "
                "   final document (it's a duplicate of nearby text, a page artifact, metadata noise, "
                "   a figure label that doesn't belong inline, etc.), set is_intentionally_empty=true, "
                "   leave resolved_text as empty string, and explain your reasoning in the explanation field.\n\n"
                "SPECIAL CHARACTERS:\n"
                "- Use actual Unicode Greek letters (α, β, γ, δ, ε, μ, etc.), NOT LaTeX "
                "notation ($\\alpha$, $\\beta$, etc.). Example: β-ENaC not $\\beta$-ENaC.\n"
                "- Use Unicode superscripts/subscripts where appropriate (Ca²⁺, Na⁺, μm, °C).\n"
                "- Only preserve LaTeX inside actual math expressions.\n\n"
                "You MUST provide an explanation in ALL cases — even when providing resolved text, briefly "
                "explain what you did (e.g., 'Chose Docling version as it was most complete').\n\n"
                "SURROUNDING DOCUMENT CONTEXT (for understanding flow — do NOT repeat these):\n"
                f"{context_display}\n\n"
                f"SEGMENT TO RESOLVE ({seg_id}, status: {seg_status}):\n\n"
                f"{sources_display}"
            )

        try:
            completion = self.client.chat.completions.parse(
                model=use_model,
                reasoning_effort=use_reasoning,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": (
                        f"Please resolve segment {seg_id}. Either provide the resolved text "
                        f"or explain why this segment should be empty in the final document."
                    )},
                ],
                response_format=RescueSegmentResponse,
            )

            usage = completion.usage
            self.usage.record(usage, "rescue", use_model)
            if usage:
                logger.info(
                    "LLM rescue call: seg=%s, tokens=%d (prompt=%d, completion=%d)",
                    seg_id, usage.total_tokens, usage.prompt_tokens, usage.completion_tokens,
                )

            message = completion.choices[0].message
            if message.parsed:
                return message.parsed

            logger.warning("Rescue call for %s returned no parsed response", seg_id)
            return None

        except Exception as exc:
            logger.warning("Rescue call for %s failed: %s", seg_id, exc)
            return None

    def resolve_alignment_tiebreak(
        self,
        payload: dict,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict:
        """Choose between two candidate alignments for ambiguous close-score cases."""
        use_model = model or Config.LLM_MODEL_ZONE_RESOLUTION
        use_reasoning = reasoning_effort or Config.LLM_REASONING_ZONE_RESOLUTION or self.reasoning_effort
        system_msg = (
            "You are selecting between two candidate alignments for the same local document region.\n"
            "Choose only the better candidate ID. Do not rewrite any text.\n"
            "Return JSON with keys: choice, confidence, explanation.\n"
            "- choice must be exactly 'candidate_a' or 'candidate_b'.\n"
            "- Prefer semantic coherence and scientific fidelity."
        )

        last_error = None
        for attempt in range(self.conflict_retry_rounds + 1):
            try:
                completion = self.client.chat.completions.parse(
                    model=use_model,
                    reasoning_effort=use_reasoning,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                    response_format=AlignmentTieBreakResponse,
                )

                usage = completion.usage
                self.usage.record(usage, "alignment_tiebreak", use_model)
                message = completion.choices[0].message
                refusal = getattr(message, "refusal", None)
                if refusal:
                    raise ValueError(f"Model refused alignment tie-break: {refusal}")
                if not message.parsed:
                    raise ValueError("No parsed alignment tie-break response")

                return {
                    "choice": message.parsed.choice,
                    "confidence": float(message.parsed.confidence),
                    "explanation": message.parsed.explanation,
                }
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "resolve_alignment_tiebreak attempt %d failed (model=%s, reasoning=%s): %s",
                    attempt + 1, use_model, use_reasoning, exc,
                )
                if attempt < self.conflict_retry_rounds:
                    next_reasoning = self._next_reasoning_effort(use_reasoning)
                    if next_reasoning:
                        use_reasoning = next_reasoning
                    continue
                break

        raise Exception(
            f"resolve_alignment_tiebreak failed after {self.conflict_retry_rounds + 1} attempts: {last_error}"
        )

    def resolve_header_hierarchy(
        self, headers: list[dict], model: str | None = None,
        reasoning_effort: str | None = None, opening_text: str | None = None,
    ) -> HeaderHierarchyResponse:
        """Classify heading lines for proper hierarchy via structured LLM output.

        Args:
            headers: List of {index, text, content_preview} dicts.
            model: Override model (defaults to self.model).
            reasoning_effort: Override reasoning effort (defaults to self.reasoning_effort).
            opening_text: First ~1000 chars of the document, used to detect
                the paper title when it appears as plain text rather than a heading.

        Returns:
            HeaderHierarchyResponse with a decision per heading.

        Raises:
            Exception if both attempts fail.
        """
        use_model = model or self.model
        use_reasoning = reasoning_effort or self.reasoning_effort

        system_msg = (
            "You are an expert at analyzing scientific paper structure. "
            "Given a list of headings extracted from a merged PDF document, "
            "assign the correct MARKDOWN HEADING LEVEL to each one.\n\n"

            "CRITICAL — DO NOT MODIFY HEADING TEXT:\n"
            "- You are ONLY assigning heading levels (1-6) or demoting to plain text.\n"
            "- You must NEVER rename, rephrase, reword, or invent headings.\n"
            "- The original_text you return MUST be the EXACT text from the input.\n"
            "- Do NOT convert Unicode characters to LaTeX. If the input has β, return β — "
            "NOT $\\beta$. If it has α, return α — NOT $\\alpha$. Copy characters exactly.\n"
            "- Every paper is different. Section names vary widely across journals "
            "and disciplines. Work with what the paper actually contains.\n\n"

            "YOUR TASK — LEVEL ASSIGNMENT:\n"
            "For each heading in the input, decide its correct level based on its "
            "role in THIS specific paper's structure:\n\n"

            "Level 1 — Paper title:\n"
            "- If one of the headings IS the paper title, assign it level 1.\n"
            "- If NONE of the headings contain the paper title (it may appear as "
            "plain text in the opening_text instead), do NOT force any heading to "
            "level 1. Instead, set the detected_title field to the EXACT title "
            "text as it appears in opening_text. In this case, 0 headings get level 1.\n\n"

            "Level 2 — Major top-level sections:\n"
            "- The primary structural divisions of the paper.\n"
            "- Examples might include things like an abstract, introduction, methods, "
            "results, discussion, references, etc. — but every paper is different. "
            "Use the actual headings present in THIS paper.\n\n"

            "Level 3+ — Subsections:\n"
            "- Headings nested under a top-level section.\n"
            "- Numbered subsections (e.g., '2.1. Something') are typically one level "
            "deeper than their parent numbered section.\n"
            "- Sub-subsections (e.g., '2.1.1.') go one level deeper still.\n"
            "- Use the content_preview to help determine context when a heading's "
            "role is ambiguous.\n\n"

            "Demote to text (demote_to_text):\n"
            "- Lines that are NOT real section headings but were incorrectly "
            "extracted as headings by the PDF parser.\n"
            "- Common examples: DOI lines, journal URLs, copyright notices, "
            "email addresses, ORCID identifiers, page numbers.\n"
            "- Only demote when you are confident the line is metadata, not a "
            "section heading.\n\n"

            "ACTIONS:\n"
            "- 'keep_level': the current heading level is already correct.\n"
            "- 'set_level': change to new_level (1-6).\n"
            "- 'demote_to_text': strip heading markers, make plain text.\n\n"

            "DETECTED TITLE:\n"
            "- If the paper title is NOT among the headings but IS visible in the "
            "opening_text, set detected_title to the EXACT title string.\n"
            "- Copy the title VERBATIM from the opening_text — do not rephrase.\n"
            "- If the title IS already one of the headings, leave detected_title null.\n\n"

            "STRUCTURAL RULES:\n"
            "- Either exactly one heading gets level 1, OR zero headings get level 1 "
            "and detected_title is set (the title will be inserted separately).\n"
            "- Return a decision for EVERY heading in the input, in the same order."
        )

        user_content = json.dumps(headers)
        if opening_text:
            user_content = (
                f"OPENING TEXT (first ~1000 chars of the document):\n"
                f"{opening_text}\n\n"
                f"HEADINGS TO CLASSIFY:\n{user_content}"
            )

        last_error = None
        for attempt in range(2):
            try:
                completion = self.client.chat.completions.parse(
                    model=use_model,
                    reasoning_effort=use_reasoning,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=HeaderHierarchyResponse,
                )

                usage = completion.usage
                self.usage.record(usage, "header_hierarchy", use_model)
                if usage:
                    logger.info(
                        "LLM header hierarchy: model=%s, tokens=%d (prompt=%d, completion=%d)",
                        use_model, usage.total_tokens, usage.prompt_tokens, usage.completion_tokens,
                        extra={
                            "_event": "llm_hierarchy_complete",
                            "_llm_model": use_model,
                            "_llm_tokens_used": usage.total_tokens,
                        },
                    )

                message = completion.choices[0].message
                refusal = getattr(message, "refusal", None)
                if refusal:
                    raise ValueError(f"Model refused: {refusal}")

                if not message.parsed:
                    raise ValueError("No parsed response from model")

                return message.parsed

            except Exception as e:
                last_error = e
                if attempt == 0:
                    logger.warning(
                        "resolve_header_hierarchy attempt %d failed: %s — retrying",
                        attempt + 1, e,
                    )
                    continue
                break

        raise Exception(f"resolve_header_hierarchy failed after 2 attempts: {last_error}")
