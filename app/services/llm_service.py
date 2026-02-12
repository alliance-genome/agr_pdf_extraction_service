import json
import logging

from openai import OpenAI

from app.services.pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)


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
        Resolve specific conflict blocks via the LLM.

        Args:
            conflicts: List of conflict bundles, each with segment_id,
                       block_type, grobid, docling, marker text.

        Returns:
            Dict mapping segment_id -> resolved markdown text.

        Raises:
            Exception if both attempts fail (caller handles fallback).
        """
        expected_ids = {c["segment_id"] for c in conflicts}
        prompt_payload = json.dumps({"conflicts": conflicts})

        system_msg = (
            "You are resolving conflicts between three PDF extraction tools. "
            "Return valid JSON only. The JSON must have a key \"resolved\" "
            "containing an object where each key is a segment_id and each "
            "value is the resolved markdown text for that segment."
        )

        last_error = None
        for attempt in range(2):  # retry once on failure
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    reasoning_effort=self.reasoning_effort,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt_payload},
                    ],
                    response_format={"type": "json_object"},
                )

                usage = response.usage
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

                raw = response.choices[0].message.content
                parsed = json.loads(raw)
                resolved = parsed.get("resolved", parsed)

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
