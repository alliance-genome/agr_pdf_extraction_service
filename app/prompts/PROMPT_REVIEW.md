# Prompt Improvement Analysis (Read-Only)

> **Date**: 2026-02-17
> **Scope**: All 7 YAML prompts + `_shared.yaml`
> **Method**: Anthropic prompt engineering checklist (10 techniques)
> **Status**: Recommendations only — no files modified

---

## Executive Summary

The prompts are already well above average. They have clear constraints, specific rules, and good domain context. The three highest-impact opportunities are:

1. **Add examples** — None of the 7 prompts have input/output examples. For complex merge tasks (conflict_batch, micro_conflict), even one example would significantly reduce ambiguity.
2. **Document response schema in-prompt** — Several prompts describe a JSON format that is enforced by Pydantic structured output, but rescue_numeric and rescue_general don't describe their fields at all. The model produces better-quality field content when it can read the field descriptions.
3. **Tighten vague language** — A few phrases like "carefully merge," "clearly more complete," and "substantially duplicates" leave room for inconsistent interpretation.

---

## Per-Prompt Analysis

### 1. `conflict_batch.yaml`

**Strengths**:
- Clear section headers (FLANKING CONTEXT, SINGLE-SOURCE SEGMENTS)
- Explicit keep/drop action semantics
- Good constraint: "Do NOT unnecessarily reword"

**Recommendations**:

| # | Priority | Issue | Suggestion |
|---|----------|-------|------------|
| 1 | **High** | No examples | Add 1-2 examples: (a) a 3-way merge where sources have complementary content, (b) a flanking-context duplicate that should be dropped. Even one example anchors the model's understanding of merge granularity. |
| 2 | **Medium** | `action="drop"` says "empty text" but code accepts drop+text | The code at line 339-341 treats drop+text as "use the text." The prompt says `Set action="drop" with empty text only for segments that are duplicates...`. This mismatch means the model might provide useful text but set action=drop, and the code happens to recover — but the intent is ambiguous. Consider documenting this fallback behavior or making the prompt match the code's actual tolerance. |
| 3 | **Medium** | "carefully merge the best parts" is vague | What does "best" mean? Consider: "prefer the version with more complete sentences, correct punctuation, and intact special characters. When two sources each have content the other lacks, combine them." |
| 4 | **Low** | "clearly more complete or accurate" — no definition of "clearly" | Could add: "If one source has full sentences where others have fragments, or preserves numbers/references that others dropped, prefer it." |
| 5 | **Low** | No extractor reliability guidance | If there's a known reliability hierarchy (e.g., GROBID tends to be better for references, Marker for tables), a brief note could help the model make better tiebreak decisions. |

---

### 2. `micro_conflict.yaml`

**Strengths**:
- 8 well-numbered rules — actionable and specific
- Good domain-specific rules (table formatting, LaTeX notation)
- Rule 6 ("Prefer selecting from source text over rewriting") is excellent

**Recommendations**:

| # | Priority | Issue | Suggestion |
|---|----------|-------|------------|
| 1 | **High** | No examples | Add one example showing: a disagreement span with 3 extractor variants, context_before/after, and the resolved output. Show why one variant was chosen (e.g., "GROBID had the correct Greek letter, Marker truncated it"). |
| 2 | **Medium** | "Pick the most accurate version" — how to judge accuracy? | Consider: "Pick the version that most likely matches the original paper. Indicators of accuracy: complete words (not truncated), correct Unicode characters, numbers matching surrounding context, proper sentence boundaries." |
| 3 | **Low** | FORMAT section describes fields but doesn't show structure | A brief JSON snippet of what the input looks like would help: `{"conflict_id": "...", "context_before": "...", "disagreement": {"grobid": "...", "docling": "...", "marker": "..."}, "context_after": "..."}` |

---

### 3. `rescue_numeric.yaml`

**Strengths**:
- Extremely clear numeric constraint: "Every number in your output must appear in at least one source"
- Good escalation context ("previously resolved, but triggered a numeric-integrity guard")
- Dynamic template with well-named placeholders

**Recommendations**:

| # | Priority | Issue | Suggestion |
|---|----------|-------|------------|
| 1 | **High** | Response fields not described | The Pydantic model expects `resolved_text`, `is_intentionally_empty`, `explanation` — but the prompt never names these fields. The model fills them more accurately when it knows their semantics. Add: "Return JSON with: `resolved_text` (the corrected text, or empty string), `is_intentionally_empty` (true only if segment should be excluded), `explanation` (required — describe what you changed and why)." |
| 2 | **Medium** | Rules 4-5 reference "status" values without context | "If status is conflict/near_agree" and "If status is gap" — the model receives `seg_status` in the template but may not understand what these statuses mean. A one-line glossary would help: "Status values: conflict = all 3 extractors disagree, near_agree = 2 agree + 1 differs, gap = only 1 extractor produced content." |
| 3 | **Low** | No example of a numeric integrity fix | One before/after example showing a novel number being corrected back to source values would make the constraint concrete. |

---

### 4. `rescue_general.yaml`

**Strengths**:
- Clear two-option structure (provide text OR explain empty)
- Good parenthetical examples for empty justification
- "You MUST provide an explanation in ALL cases" — unambiguous

**Recommendations**:

| # | Priority | Issue | Suggestion |
|---|----------|-------|------------|
| 1 | **High** | Response fields not described (same as rescue_numeric) | Add the same `resolved_text` / `is_intentionally_empty` / `explanation` field descriptions. |
| 2 | **Low** | "Pick the best version from the sources below, merge them, or clean one up" | Three options in one sentence — could be clearer as a prioritized list: "(1) If one source is clearly superior, use it. (2) If sources have complementary content, merge them. (3) If one source is mostly correct but has extraction artifacts, clean it up." |

---

### 5. `gap_resolution.yaml`

**Strengths**:
- Well-separated KEEP/DROP criteria with bullet lists
- Good edge case: "content that spans a segment boundary"
- Clear "only return the text for the GAP segment itself" constraint

**Recommendations**:

| # | Priority | Issue | Suggestion |
|---|----------|-------|------------|
| 1 | **Medium** | "substantially duplicates" vs conflict_batch's "exactly duplicates" | `conflict_batch.yaml` was intentionally changed from "substantially" to "exactly." Should gap_resolution match? If "substantially" is intentional here (since GAP segments from a single extractor might paraphrase), document why the threshold differs. |
| 2 | **Medium** | "a list of objects" for what is always a single segment | `resolve_gap()` always sends exactly one segment, but the prompt says "Return a JSON object with a 'resolved' key containing a list of objects." The list framing is technically correct (matches Pydantic) but could confuse the model into thinking there might be multiple segments. Consider: "Return a JSON object with a 'resolved' key containing a list with exactly one object..." |
| 3 | **Low** | No example | A brief example showing a GAP segment that's a flanking-context duplicate (→ drop) vs. one with unique content (→ keep) would help. |

---

### 6. `alignment_tiebreak.yaml`

**Strengths**:
- Priority-ordered decision criteria (5 levels) — excellent structure
- Good confidence scale explanation (0.5 = coin flip, 1.0 = obvious)
- Clear "Do NOT rewrite any text — just choose" constraint
- Response format well-specified

**Recommendations**:

| # | Priority | Issue | Suggestion |
|---|----------|-------|------------|
| 1 | **Low** | "may include" for candidate fields | The prompt lists candidate fields (dp_score, sample_columns, text, etc.) with "Candidates may include." If certain fields are always present, saying so helps the model know what to rely on. |
| 2 | **Low** | "Semantic coherence" is somewhat abstract | Could add: "e.g., a sentence should not be split across two alignment columns if it belongs together." |

This is the strongest prompt. No high-priority changes.

---

### 7. `header_hierarchy.yaml`

**Strengths**:
- Very thorough level-by-level guidance
- "Every paper is different" — excellent guard against over-rigid behavior
- CRITICAL section on preserving heading text is well-emphasized
- Smart title detection fallback (detected_title when title isn't a heading)

**Recommendations**:

| # | Priority | Issue | Suggestion |
|---|----------|-------|------------|
| 1 | **Medium** | No example | This prompt is the longest (~53 lines) and most complex. A single example showing 5-6 headings with their assigned levels would anchor understanding. E.g., "Introduction" → L2, "2.1 Cell Culture" → L3, "https://doi.org/..." → demote_to_text. |
| 2 | **Low** | Action values are listed but the JSON response structure isn't fully shown | The Pydantic model has `heading_index`, `original_text`, `action`, `new_level`, and the wrapper has `decisions` + `detected_title`. Showing a brief JSON example of the expected output would help, especially for the `detected_title` edge case. |

---

### 8. `_shared.yaml`

**Strengths**:
- `special_characters`: Excellent concrete examples (beta-ENaC, Abeta42, Ca2+)
- `extraction_errors`: Good coverage of common PDF extraction artifacts
- Clear "Do NOT fix errors that appear to be in the original paper" constraint

**Recommendations**:

| # | Priority | Issue | Suggestion |
|---|----------|-------|------------|
| 1 | **Low** | `extraction_errors` could add one more common pattern | Consider adding: "Dropped or swapped characters in chemical formulas: 'H2O2' becoming 'H202' (digit 2 vs letter O)" — this is a frequent scientific PDF artifact. |
| 2 | **Low** | `special_characters` LaTeX exception could be more precise | "Preserve LaTeX only inside actual math expressions" — could clarify what counts as an "actual math expression" vs. inline use: "Actual math expressions are delimited by `$...$` or `$$...$$` and contain operators, equations, or summations — not just a single Greek letter." |

---

## Cross-Cutting Observations

### A. Examples Are the Single Biggest Gap

None of the 7 prompts include examples. For Anthropic's models, examples are one of the highest-leverage techniques — they disambiguate complex instructions more effectively than additional rules. The prompts most likely to benefit:

1. **conflict_batch** — 3-way merge is inherently ambiguous; an example sets the bar for merge quality
2. **micro_conflict** — token-level resolution needs a concrete demonstration
3. **header_hierarchy** — level assignment with edge cases (title detection, demotion) is hard to convey purely through rules

The prompts least in need of examples:
- **alignment_tiebreak** — binary choice with clear criteria
- **rescue_numeric** — highly constrained (numbers must come from sources)

### B. Structured Output + Prompt Description Duality

All 7 prompts use Pydantic `response_format` for structured output. This means the JSON schema is enforced by the API — the model cannot deviate. However:

- **conflict_batch, micro_conflict, gap_resolution, alignment_tiebreak** describe their JSON output in the prompt text. This is good — it gives the model semantic context for what each field means.
- **rescue_numeric and rescue_general** do NOT describe their response fields (`resolved_text`, `is_intentionally_empty`, `explanation`). The model infers field meaning from Pydantic field names alone, which usually works but is less reliable than explicit descriptions.

**Recommendation**: Add response field descriptions to rescue_numeric and rescue_general to match the other prompts.

### C. Consistency Across Prompts

| Pattern | Current State | Suggestion |
|---------|--------------|------------|
| "Do NOT" vs "Do not" | Mixed across prompts | Standardize to "Do NOT" (capitalized) for prohibitions — it's more visually prominent and already used in most prompts |
| "substantially duplicates" vs "exactly duplicates" | gap_resolution uses "substantially," conflict_batch uses "exactly" | Decide which threshold is intended for each and document the distinction |
| Extractor names | Always "GROBID, Docling, Marker" | Consistent (good) |
| Section headers | ALL CAPS with colon (e.g., "RULES:", "FLANKING CONTEXT AND DUPLICATION:") | Consistent (good) |

### D. Chain of Thought Consideration

These prompts use structured output, so CoT reasoning would need to be a dedicated field in the Pydantic model. The rescue prompts already have `explanation` (which serves this purpose). For conflict_batch and gap_resolution, there's no explanation field — adding one would require a Pydantic model change.

**Recommendation**: If you find the model making poor merge decisions in conflict_batch, adding an optional `reasoning` field to `ResolvedSegment` (and requesting it in the prompt) would be the highest-impact change. This forces the model to articulate its merge logic before committing to output. However, this adds tokens and cost to every response — only worth it if merge quality is a live problem.

### E. Role/Persona

Most prompts use a functional opener ("You are resolving conflicts between...") rather than an expertise-framed role ("You are an expert in scientific document reconstruction..."). For these highly constrained tasks with structured output, the functional opener is actually more effective — it immediately establishes the task rather than a persona. No change recommended.

---

## Priority Summary

### Do First (High Impact, Low Risk)
1. Add response field descriptions to `rescue_numeric.yaml` and `rescue_general.yaml`
2. Add 1 example to `conflict_batch.yaml` showing a 3-way merge
3. Add 1 example to `micro_conflict.yaml` showing a disagreement resolution

### Do When Convenient (Medium Impact)
4. Resolve "substantially" vs "exactly" duplicates consistency between gap_resolution and conflict_batch
5. Add status value glossary to rescue_numeric (conflict/near_agree/gap)
6. Clarify "carefully merge the best parts" in conflict_batch with specific guidance
7. Add 1 example to `header_hierarchy.yaml` showing level assignments
8. Clarify gap_resolution's "list of objects" framing for single-segment calls

### Polish (Low Impact)
9. Standardize "Do NOT" capitalization across all prompts
10. Add "actual math expression" clarification to special_characters
11. Add alignment_tiebreak candidate field presence guarantees
12. Add chemical formula artifact example to extraction_errors
