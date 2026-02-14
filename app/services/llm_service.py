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

# Model tier ordering for retry escalation (cheapest → strongest)
MODEL_TIER_ORDER = ["gpt-5-mini", "gpt-5.2"]


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


class LLM(PDFExtractor):
    def __init__(
        self,
        api_key,
        model="gpt-5.2",
        reasoning_effort="medium",
        conflict_batch_size: int = 10,
        conflict_max_workers: int = 4,
        conflict_retry_rounds: int = 2,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.conflict_batch_size = max(1, int(conflict_batch_size))
        self.conflict_max_workers = max(1, int(conflict_max_workers))
        self.conflict_retry_rounds = max(1, int(conflict_retry_rounds))
        self.usage = TokenAccumulator()

    def _estimate_zone_tokens(self, zone: dict) -> int:
        """Estimate token count for a zone payload (~4 chars per token + system message overhead)."""
        return len(json.dumps(zone)) // 4 + 500

    def _select_zone_model(self, zone: dict) -> str:
        """Select the appropriate model for a conflict zone based on size."""
        est_tokens = self._estimate_zone_tokens(zone)
        if est_tokens > Config.ZONE_ESCALATION_THRESHOLD:
            return Config.ZONE_ESCALATION_MODEL
        return Config.LLM_MODEL_ZONE_RESOLUTION

    @staticmethod
    def _next_tier_model(current_model: str) -> str | None:
        """Return the next stronger model in the tier order, or None if already at strongest."""
        try:
            idx = MODEL_TIER_ORDER.index(current_model)
            if idx + 1 < len(MODEL_TIER_ORDER):
                return MODEL_TIER_ORDER[idx + 1]
        except ValueError:
            pass
        return None

    def extract(self, grobid_md, docling_md, marker_md):
        """Full-document LLM merge (fallback path)."""
        prompt = self.create_prompt(grobid_md, docling_md, marker_md)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = response.usage
            self.usage.record(usage, "full_merge", self.model)
            if usage:
                logger.info(
                    "LLM full merge complete: model=%s, tokens=%d (prompt=%d, completion=%d)",
                    self.model, usage.total_tokens, usage.prompt_tokens, usage.completion_tokens,
                    extra={
                        "_event": "llm_resolve_complete",
                        "_llm_model": self.model,
                        "_llm_tokens_used": usage.total_tokens,
                    },
                )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error in LLM processing: {str(e)}")

    def _resolve_conflict_batch(self, batch: list[dict]) -> tuple[dict[str, str], set[str]]:
        """Resolve one conflict batch, returning (resolved_map, unresolved_ids)."""
        use_model = Config.LLM_MODEL_CONFLICT_BATCH
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
            reasoning_effort=self.reasoning_effort,
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

    def _resolve_single_zone(self, zone: dict, model: str | None = None) -> tuple[dict[str, str], set[str]]:
        """Resolve one conflict zone, returning (resolved_map, unresolved_ids).

        Args:
            zone: The conflict zone dict to resolve.
            model: Override model for this call (defaults to self.model).
        """
        use_model = model or self.model

        # Collect expected segment_ids (conflict, gap, and near_agree — not pre_resolved or agreed)
        expected_ids = {
            seg["segment_id"]
            for seg in zone["segments"]
            if seg["status"] in ("conflict", "gap", "near_agree")
        }
        if not expected_ids:
            return {}, set()

        system_msg = (
            "You are resolving conflicts between PDF extraction tools (GROBID, Docling, Marker) "
            "that processed the same scientific paper.\n\n"
            "You will receive a \"conflict zone\" — consecutive segments where extractors disagreed "
            "or produced slightly different text, surrounded by agreed-upon context that is already settled.\n\n"
            "ZONE STRUCTURE:\n"
            "- context_before: agreed segments BEFORE the zone (read-only, do NOT repeat)\n"
            "- segments: the zone's segments in document order. Each has a status:\n"
            "    \"conflict\" — extractors disagree significantly. Pick the best version or merge.\n"
            "    \"near_agree\" — extractors mostly agree but text differs slightly (formatting, "
            "accents, punctuation, whitespace). All extractor versions are provided. Pick the "
            "cleanest, most complete version or merge the best parts. The \"current_choice\" "
            "field shows what the programmatic pipeline chose — you may keep it or improve it.\n"
            "    \"gap\" — only one extractor has text. Keep it, drop it (empty string), "
            "or clean it up.\n"
            "    \"pre_resolved\" — already resolved, included for context only. Do NOT "
            "return text for these.\n"
            "    \"agreed\" — already agreed, included for context only. Do NOT "
            "return text for these.\n"
            "- context_after: agreed segments AFTER the zone (read-only, do NOT repeat)\n\n"
            "RULES:\n"
            "1. Return a resolution for every \"conflict\", \"near_agree\", and \"gap\" segment.\n"
            "2. Do NOT return entries for \"pre_resolved\", \"agreed\", or context segments.\n"
            "3. Your resolved segments must flow naturally in document order between "
            "context_before and context_after.\n"
            "4. Preserve markdown formatting (bold, headers, italics) where appropriate.\n"
            "5. For \"near_agree\" segments, prefer the version with correct accented characters, "
            "complete text, and proper formatting.\n\n"
            "ACTION FIELD (required for every segment):\n"
            "- Set action=\"keep\" and provide the resolved text for segments that belong in "
            "the final document. This applies to ALL conflict and near_agree segments — you "
            "must ALWAYS resolve these with action=\"keep\".\n"
            "- Set action=\"drop\" with empty text ONLY for \"gap\" segments that are artifacts, "
            "duplicates, or page noise that should be excluded. NEVER drop a conflict or "
            "near_agree segment.\n\n"
            "Return JSON: {\"resolved\": [{\"segment_id\": \"...\", \"action\": \"keep\"|\"drop\", \"text\": \"...\"}, ...]}"
        )

        prompt_payload = json.dumps(zone)

        completion = self.client.chat.completions.parse(
            model=use_model,
            reasoning_effort=self.reasoning_effort,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_payload},
            ],
            response_format=ConflictResolutionResponse,
        )

        usage = completion.usage
        self.usage.record(usage, "zone_resolution", use_model)
        if usage:
            logger.info(
                "LLM zone resolution: zone=%s, model=%s, tokens=%d (prompt=%d, completion=%d)",
                zone.get("zone_id", "?"),
                use_model, usage.total_tokens, usage.prompt_tokens, usage.completion_tokens,
                extra={
                    "_event": "llm_zone_resolve_complete",
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

        # Build lookup preserving the full ResolvedSegment (with action field)
        raw_map = {seg.segment_id: seg for seg in message.parsed.resolved}
        # Map segment_id → original status for drop validation
        seg_status_map = {
            seg["segment_id"]: seg["status"]
            for seg in zone["segments"]
            if seg["status"] in ("conflict", "gap", "near_agree")
        }
        resolved: dict[str, str] = {}
        unresolved = set(expected_ids)

        for seg_id in expected_ids:
            seg_result = raw_map.get(seg_id)

            if seg_result is None:
                # Segment missing entirely — LLM omitted it
                logger.warning("LLM omitted segment %s from zone response — unresolved", seg_id)
                continue  # stays in unresolved set

            if seg_result.action == "keep":
                if seg_result.text.strip():
                    # Normal resolution — text provided
                    resolved[seg_id] = seg_result.text.strip()
                    unresolved.discard(seg_id)
                else:
                    # Error: said "keep" but gave empty text — treat as unresolved
                    logger.warning(
                        "LLM returned action='keep' but empty text for %s — treating as unresolved",
                        seg_id,
                    )
                    # stays in unresolved set — will trigger retry

            elif seg_result.action == "drop":
                orig_status = seg_status_map.get(seg_id, "conflict")
                if seg_result.text.strip():
                    # Suspicious: said "drop" but provided text — use the text
                    logger.warning(
                        "LLM returned action='drop' but provided text for %s — using text as 'keep'",
                        seg_id,
                    )
                    resolved[seg_id] = seg_result.text.strip()
                    unresolved.discard(seg_id)
                elif orig_status == "gap":
                    # Intentional drop of gap segment — legitimate empty
                    resolved[seg_id] = ""
                    unresolved.discard(seg_id)
                else:
                    # Invalid: tried to drop a conflict/near_agree segment — treat as unresolved
                    logger.warning(
                        "LLM tried to drop %s segment %s — treating as unresolved for retry/escalation",
                        orig_status, seg_id,
                    )
                    # stays in unresolved set — will trigger retry with escalated model

        return resolved, unresolved

    def rescue_single_segment(
        self,
        seg: dict,
        zone: dict,
        neighboring_resolved: dict[str, str],
        extra_flanking: list[dict],
        model: str | None = None,
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

        Returns a RescueSegmentResponse, or None if the API call itself fails.
        """
        use_model = model or self.model
        seg_id = seg["segment_id"]
        seg_status = seg["status"]

        # Build the source texts display
        source_lines = []
        for source_name in ("grobid", "docling", "marker"):
            text = (seg.get(source_name) or "").strip()
            if text:
                source_lines.append(f"### {source_name.upper()}\n{text}")
            else:
                source_lines.append(f"### {source_name.upper()}\n(no output from this extractor)")
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
                reasoning_effort=self.reasoning_effort,
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

    def resolve_conflict_zones(
        self, zones: list[dict],
    ) -> tuple[dict[str, str], set[str]]:
        """Resolve conflict zones in parallel with per-zone model selection and retry escalation.

        Each zone gets a model selected by _select_zone_model() based on its
        estimated token count. On failure or unresolved segments, the model is
        escalated to the next tier (mini→5.2) before retrying.

        Returns:
            (resolved_all, unresolved_ids) — resolved_all maps segment_id to
            resolved text for every segment the LLM handled; unresolved_ids
            contains segment IDs from zones that still had failures after all
            retry rounds.  The caller is responsible for deciding how to handle
            unresolved segments (rescue, fallback, skip).
        """
        if not zones:
            return {}, set()

        resolved_all: dict[str, str] = {}
        pending_zones = list(zones)

        def _zkey(z: dict) -> str:
            """Consistent zone key for model tracking."""
            return z.get("zone_id") or str(id(z))

        # Track per-zone model selection for escalation across rounds
        zone_models: dict[str, str] = {}
        for zone in zones:
            zid = _zkey(zone)
            selected = self._select_zone_model(zone)
            zone_models[zid] = selected
            est_tokens = self._estimate_zone_tokens(zone)
            logger.info(
                "Zone %s: est_tokens=%d, selected_model=%s",
                zid, est_tokens, selected,
                extra={
                    "_event": "zone_model_selected",
                    "_zone_id": zid,
                    "_est_tokens": est_tokens,
                    "_selected_model": selected,
                },
            )

        for round_idx in range(self.conflict_retry_rounds + 1):
            if not pending_zones:
                break

            logger.info(
                "resolve_conflict_zones round %d: %d zone(s)",
                round_idx + 1, len(pending_zones),
            )

            round_resolved: dict[str, str] = {}
            failed_zones: list[dict] = []

            max_workers = min(self.conflict_max_workers, len(pending_zones))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for zone in pending_zones:
                    zid = _zkey(zone)
                    use_model = zone_models.get(zid, self.model)
                    futures[executor.submit(self._resolve_single_zone, zone, model=use_model)] = zone

                for future in as_completed(futures):
                    zone = futures[future]
                    zid = _zkey(zone)
                    current_model = zone_models.get(zid, self.model)
                    try:
                        zone_resolved, zone_unresolved = future.result()
                        round_resolved.update(zone_resolved)
                        if zone_unresolved:
                            logger.warning(
                                "Zone %s left %d unresolved segments (model=%s, attempt=%d)",
                                zid, len(zone_unresolved), current_model, round_idx + 1,
                                extra={
                                    "_event": "zone_resolve_partial",
                                    "_zone_id": zid,
                                    "_model": current_model,
                                    "_attempt": round_idx + 1,
                                    "_unresolved_count": len(zone_unresolved),
                                },
                            )
                            # Escalate model for next round
                            next_model = self._next_tier_model(current_model)
                            if next_model and round_idx < self.conflict_retry_rounds:
                                logger.info(
                                    "Escalating zone %s: %s → %s",
                                    zid, current_model, next_model,
                                    extra={
                                        "_event": "zone_model_escalated",
                                        "_zone_id": zid,
                                        "_escalated_from": current_model,
                                        "_escalated_to": next_model,
                                    },
                                )
                                zone_models[zid] = next_model
                            failed_zones.append(zone)
                    except Exception as exc:
                        logger.warning(
                            "resolve_conflict_zones zone %s failed (model=%s, attempt=%d): %s",
                            zid, current_model, round_idx + 1, exc,
                            extra={
                                "_event": "zone_resolve_failed",
                                "_zone_id": zid,
                                "_model": current_model,
                                "_attempt": round_idx + 1,
                            },
                        )
                        # Escalate model for next round
                        next_model = self._next_tier_model(current_model)
                        if next_model and round_idx < self.conflict_retry_rounds:
                            logger.info(
                                "Escalating zone %s: %s → %s",
                                zid, current_model, next_model,
                                extra={
                                    "_event": "zone_model_escalated",
                                    "_zone_id": zid,
                                    "_escalated_from": current_model,
                                    "_escalated_to": next_model,
                                },
                            )
                            zone_models[zid] = next_model
                        failed_zones.append(zone)

            resolved_all.update(round_resolved)
            pending_zones = failed_zones

            if pending_zones and round_idx < self.conflict_retry_rounds:
                logger.warning(
                    "resolve_conflict_zones round %d: %d zone(s) need retry",
                    round_idx + 1, len(pending_zones),
                )
                continue

        # Collect only the truly unresolved segment IDs (not already in resolved_all)
        unresolved_ids: set[str] = set()
        if pending_zones:
            for z in pending_zones:
                for seg in z["segments"]:
                    sid = seg["segment_id"]
                    if seg["status"] in ("conflict", "near_agree", "gap") and sid not in resolved_all:
                        unresolved_ids.add(sid)
            failed_zone_ids = [z.get("zone_id", "?") for z in pending_zones]
            logger.warning(
                "resolve_conflict_zones: %d zone(s) unresolved after retries: %s (%d segments)",
                len(pending_zones), failed_zone_ids, len(unresolved_ids),
            )

        return resolved_all, unresolved_ids

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
            "- No level jumps > 1 (e.g., going from level 2 to level 4 "
            "without a level 3 in between is invalid).\n"
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

    def create_prompt(self, grobid_md, docling_md, marker_md):
        return f"""You are processing a scientific article that has been extracted using three different tools: GROBID, Docling, and Marker.
Each tool produces different quality outputs with varying levels of detail.

Your task is to:
1. Merge the three markdown extractions into a single, well-structured document
2. Identify and clearly mark the following sections:
   - Title
   - Authors (with affiliations if available)
   - Abstract
   - Keywords (if present)
   - Introduction
   - Methodology/Methods
   - Results
   - Discussion
   - Conclusion
   - References
   - Any other relevant sections

3. Extract and list:
   - All tables (preserve structure)
   - All figures/images (note their captions and references)
   - All equations (preserve formatting)

4. Resolve conflicts between the three extractions by choosing the most complete and accurate version
5. Maintain academic formatting and citation styles

Here are the three extractions:

## GROBID Extraction:
{grobid_md}

## Docling Extraction:
{docling_md}

## Marker Extraction:
{marker_md}

Please provide a single, well-structured markdown document with clear section headers and all elements properly organized."""
