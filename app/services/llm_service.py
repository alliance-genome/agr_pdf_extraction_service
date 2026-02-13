import json
import logging
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from app.services.pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)


class ResolvedSegment(BaseModel):
    """A single resolved conflict segment."""

    segment_id: str
    text: str


class ConflictResolutionResponse(BaseModel):
    """Structured conflict-resolution response.

    Uses a list of {segment_id, text} objects instead of a dict because
    OpenAI's strict structured output mode does not support dynamic keys
    (dict[str, str] generates additionalProperties which is rejected).
    """

    resolved: list[ResolvedSegment]


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
    def __init__(self, api_key, model="gpt-5.2", reasoning_effort="medium"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort

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

    def resolve_conflicts(self, conflicts: list[dict]) -> dict[str, str]:
        """
        Resolve specific conflict blocks via the LLM using structured parsing.

        Args:
            conflicts: List of conflict bundles, each with segment_id,
                       block_type, context_before, context_after,
                       grobid, docling, marker text.

        Returns:
            Dict mapping segment_id -> resolved markdown text.

        Raises:
            Exception if both attempts fail (caller handles fallback).
        """
        expected_ids = {c["segment_id"] for c in conflicts}
        prompt_payload = json.dumps({"conflicts": conflicts})

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
            "objects, each with 'segment_id' and 'text' keys. The 'text' value "
            "is the resolved markdown for that segment."
        )

        last_error = None
        for attempt in range(2):  # retry once on transient failure
            try:
                completion = self.client.chat.completions.parse(
                    model=self.model,
                    reasoning_effort=self.reasoning_effort,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt_payload},
                    ],
                    response_format=ConflictResolutionResponse,
                )

                usage = completion.usage
                if usage:
                    logger.info(
                        "LLM conflict resolution: model=%s, tokens=%d (prompt=%d, completion=%d)",
                        self.model, usage.total_tokens, usage.prompt_tokens, usage.completion_tokens,
                        extra={
                            "_event": "llm_resolve_complete",
                            "_llm_model": self.model,
                            "_llm_tokens_used": usage.total_tokens,
                        },
                    )

                message = completion.choices[0].message
                refusal = getattr(message, "refusal", None)
                if refusal:
                    raise ValueError(f"Model refused: {refusal}")

                if not message.parsed:
                    raise ValueError("No parsed response from model")

                # Convert list of ResolvedSegment to dict
                resolved = {seg.segment_id: seg.text for seg in message.parsed.resolved}

                # Validate: all expected segment_ids present with non-empty values
                missing = expected_ids - set(resolved.keys())
                if missing:
                    raise ValueError(f"Missing segment_ids in response: {missing}")

                empty = [sid for sid in expected_ids if not resolved.get(sid, "").strip()]
                if empty:
                    raise ValueError(f"Empty values for segment_ids: {empty}")

                return {sid: resolved[sid] for sid in expected_ids}

            except Exception as e:
                last_error = e
                if attempt == 0:
                    logger.warning(
                        "resolve_conflicts attempt %d failed: %s — retrying",
                        attempt + 1, e,
                    )
                    continue
                break

        raise Exception(f"resolve_conflicts failed after 2 attempts: {last_error}")

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
