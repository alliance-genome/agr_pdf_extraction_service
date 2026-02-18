# Prompt Improvement Follow-Up Analysis (Read-Only)

> **Date**: 2026-02-17
> **Scope**: All 7 YAML prompts + `_shared.yaml` (post-improvement pass)
> **Method**: Anthropic prompt engineering checklist (10 techniques)
> **Status**: Recommendations only — no files modified
> **Predecessor**: `PROMPT_REVIEW.md` (all 12 recommendations applied)

---

## Executive Summary

All 12 recommendations from `PROMPT_REVIEW.md` have been applied. The prompts are now substantially stronger: every prompt except `rescue_general.yaml` has at least one concrete example, response formats are documented, vague language has been tightened, and cross-prompt consistency has improved. This follow-up surfaces **residual issues and new observations** that emerged after the improvements.

**Overall quality**: High. These are production-ready prompts. The findings below are polish-level — none are blocking issues.

**Top 3 findings**:
1. `rescue_general.yaml` is the only prompt still missing an example
2. `alignment_tiebreak.yaml` example mixes fields from two different usage modes (global vs local)
3. `rescue_general.yaml` lacks a STATUS VALUES glossary that `rescue_numeric.yaml` has (inconsistency)

---

## Automated Check Results

| File | XML Tags | Variables | Notes |
|------|----------|-----------|-------|
| `_shared.yaml` | None | `$E` (false positive — LaTeX `$E = mc^2$`) | Clean |
| `conflict_batch.yaml` | `<sup>` (HTML in examples, not structure) | `<sup>` (false positive — HTML tag) | Clean |
| `micro_conflict.yaml` | None | None | Clean |
| `rescue_numeric.yaml` | None | 5 dynamic: `{numeric_note}`, `{prev_note}`, `{context_display}`, `{seg_id}`, `{seg_status}`, `{sources_display}` + `{{...}}` for JSON example | All correct, well-named |
| `rescue_general.yaml` | None | 4 dynamic: `{context_display}`, `{seg_id}`, `{seg_status}`, `{sources_display}` | All correct |
| `gap_resolution.yaml` | None | None | Static prompt, correct |
| `alignment_tiebreak.yaml` | None | None | Static prompt, correct |
| `header_hierarchy.yaml` | None | None | Static prompt, correct |

No structural issues detected. All variable naming is consistent and descriptive. The `{{`/`}}` escaping in `rescue_numeric.yaml` for JSON within a dynamic template is correct.

---

## Per-Prompt Analysis

### 1. `_shared.yaml` — Solid

**Changes verified**: Character confusion pattern added (H2O2/RNAi), LaTeX exception clarified.

| # | Priority | Finding | Detail |
|---|----------|---------|--------|
| 1 | **Low** | `extraction_errors` could add header/footer artifacts | Page numbers, running headers, and footer lines that leak into body text are a common PDF extraction artifact not currently listed. E.g., "Introduction 42 Results" where "42" is a page number injected between sections. |
| 2 | **Low** | "etc." pattern in character confusion example | The character confusion bullet ends with "etc." which is correct per project convention. No issue. |

No changes recommended. Both shared sections are strong.

---

### 2. `conflict_batch.yaml` — Strong

**Changes verified**: 2 examples added, vague language tightened, drop+text fallback documented.

| # | Priority | Finding | Detail |
|---|----------|---------|--------|
| 1 | **Low** | Both examples show "pick best single source" — no true 3-way merge | Example 1 picks Marker's version entirely. Example 2 drops. Neither shows combining text from 2-3 sources (e.g., GROBID's sentence structure + Docling's reference list + Marker's special characters). A future 3rd example could demonstrate actual merging. |
| 2 | **Low** | "flows naturally" remains slightly vague | "Ensure your resolved text flows naturally between context_before and context_after" — could be tightened to "transitions grammatically and logically" but the surrounding constraints make intent clear enough. |
| 3 | **Info** | Grammar nit in paragraph 2 | "preserves numbers/references that others dropped" — subject is "one source" from earlier in the sentence, so the verb should be "preserves" (which it is). Reads correctly, just a long sentence with distant subject. |

**Verdict**: Production-ready. The examples anchor the task effectively even without a true 3-way merge demonstration.

---

### 3. `micro_conflict.yaml` — Strong

**Changes verified**: 1 example with 2 micro-conflicts, accuracy indicators added to Rule 1.

| # | Priority | Finding | Detail |
|---|----------|---------|--------|
| 1 | **Low** | No example of `action="drop"` | The response format mentions `action="drop"` to delete a span, but the example only shows `action="keep"`. Drop is rare in micro-conflicts (the surrounding agreed text exists by definition), so this is an edge case — but documenting when it applies would be useful. |
| 2 | **Low** | Example input shows `segment_id` but response doesn't include it | The example has `segment_id: "seg_007"` as context, and the response JSON has `conflict_id` (not `segment_id`). This matches the `MicroConflictResolutionResponse` Pydantic model correctly. Just noting that the structural relationship (micro-conflicts *within* a segment) is implied, not explicitly stated. |

**Verdict**: Production-ready. Example quality is high — covers character fidelity and formatting convention decisions.

---

### 4. `rescue_numeric.yaml` — Strong

**Changes verified**: RESPONSE FORMAT section added, STATUS VALUES glossary added, example with novel number correction added, `{{`/`}}` escaping correct.

| # | Priority | Finding | Detail |
|---|----------|---------|--------|
| 1 | **Low** | `{prev_note}` concatenated directly after `{numeric_note}` with no separator | Line 15: `{numeric_note}{prev_note}` — if both are non-empty, they'll run together unless the calling code includes trailing whitespace/newline in each value. Verify the calling code handles this. |
| 2 | **Low** | Example shows only the non-empty case | No example of `is_intentionally_empty=true`. This is acceptable because rescue_numeric is called when numbers need fixing — emptiness is a rare edge case — and the rules already clearly describe when it's allowed (Rule 5). |

**Verdict**: Production-ready. The STATUS VALUES glossary and RESPONSE FORMAT section significantly improve clarity.

---

### 5. `rescue_general.yaml` — Good (one gap)

**Changes verified**: RESPONSE FORMAT section added, prioritized option list replaces vague "pick the best."

| # | Priority | Finding | Detail |
|---|----------|---------|--------|
| 1 | **Medium** | **No example** — the only prompt without one | Every other prompt now has at least one example. An example showing the "explain why empty" case (e.g., a segment that's a page artifact) would be most useful, since the "provide text" case is straightforward. |
| 2 | **Medium** | No STATUS VALUES glossary | `rescue_numeric.yaml` defines conflict/near_agree/gap, but `rescue_general.yaml` also receives `{seg_status}` and doesn't explain what the values mean. Copy the same 3-line glossary for consistency. |
| 3 | **Low** | "genuinely should not appear" — slightly vague | "If this segment genuinely should not appear in the final document" — the parenthetical examples that follow ("duplicate of nearby text, a page artifact, metadata noise, a figure label that doesn't belong inline, etc.") are strong enough to clarify intent. |

**Verdict**: Functional but less robust than sibling prompts. Adding an example and status glossary would bring it to parity.

---

### 6. `gap_resolution.yaml` — Excellent

**Changes verified**: 3 examples added, "exactly duplicates" threshold, "list with exactly one object" clarification.

| # | Priority | Finding | Detail |
|---|----------|---------|--------|
| 1 | **Low** | Example 2 relies on domain knowledge ("**BC**" → "JBC") | The OCR fix assumes the model knows JBC = Journal of Biological Chemistry. This is reasonable for a scientific PDF extraction context — the model has this knowledge — but worth noting. |
| 2 | **Info** | 3 examples cover keep, keep-with-cleanup, and drop | Excellent coverage of the action spectrum. |

**Verdict**: Strongest prompt. Three diverse examples with clear "Why" explanations make this a model for how the other prompts should look.

---

### 7. `alignment_tiebreak.yaml` — Strong (one issue)

**Changes verified**: 1 example added, explicit field lists per mode, "semantic coherence" clarified.

| # | Priority | Finding | Detail |
|---|----------|---------|--------|
| 1 | **Medium** | **Example mixes fields from both usage modes** | The example's candidate_a has `mode: 3, dp_score: 0.847, description: "merge split fragments", sample_columns: ...`. Checking `arbitration.py`: global mode sends `{mode, dp_score, sample_columns}` (lines 130-134), local repair sends `{description, score, text, anchor_text}` (lines 188-194). The example includes `description` alongside `mode`/`dp_score`/`sample_columns` — this combination never occurs in practice. The `description` field should be removed from the example, or a second example should show local repair separately. |
| 2 | **Low** | No example of the local repair arbitration mode | The example only covers global alignment. A second example with `description`, `score`, `text`, and `anchor_text` fields would demonstrate both modes. |
| 3 | **Low** | Confidence scale has endpoints defined but no mid-range guidance | 0.5 = coin flip, 1.0 = obvious. What does 0.7 mean? The model handles continuous scales well, so this is minor. |

**Verdict**: Strong, but the field mismatch in the example could cause the model to expect fields that won't be present (or vice versa) in one of the two modes. Worth fixing.

---

### 8. `header_hierarchy.yaml` — Strong

**Changes verified**: 1 comprehensive example with 9 headings, JSON output matching Pydantic model.

| # | Priority | Finding | Detail |
|---|----------|---------|--------|
| 1 | **Low** | No example of the `detected_title` case | The example has `detected_title: null` (title IS among the headings). The alternate path — title found in `opening_text` but not in headings — is described in rules but not demonstrated. This is the more complex case. |
| 2 | **Low** | Example heading [1] may appear truncated | "Binding site for the small molecule ENaC activator" — if the full paper title is longer, this could look like a truncation. If this IS the full title, no issue. |
| 3 | **Info** | Longest prompt at ~78 lines (with example) | Length is justified by task complexity. The example alone is ~25 lines and provides essential anchor for level assignment logic. |

**Verdict**: Production-ready. The single comprehensive example effectively covers set_level (multiple levels), demote_to_text, and the detected_title=null case.

---

## Cross-Cutting Observations

### A. Example Quality Assessment (New)

All added examples share these strengths:
- **Real data**: Based on the ENaC paper test run (run_b2cdfe36), not synthetic
- **"Why" explanations**: Every example includes reasoning that models the decision process
- **Pydantic-matched output**: JSON examples use exact field names from response models
- **Edge cases covered**: OCR fixes, flanking duplicates, Unicode vs LaTeX

| Prompt | Examples | Coverage | Gap |
|--------|----------|----------|-----|
| `conflict_batch` | 2 | keep (merge) + drop (duplicate) | No true 3-way merge |
| `micro_conflict` | 1 (2 conflicts) | Character fidelity + formatting | No drop case |
| `rescue_numeric` | 1 | Novel number correction | No empty case |
| `rescue_general` | **0** | — | **No example at all** |
| `gap_resolution` | 3 | keep + keep-with-cleanup + drop | Full coverage |
| `alignment_tiebreak` | 1 | Global mode tiebreak | No local repair; field mismatch |
| `header_hierarchy` | 1 (9 headings) | set_level + demote_to_text | No detected_title case |

### B. Consistency Audit

| Pattern | Status | Notes |
|---------|--------|-------|
| "Do NOT" capitalization | **Mostly consistent** | All prompts use "Do NOT" for prohibitions. Good. |
| "exactly duplicates" threshold | **Fixed** | Both conflict_batch and gap_resolution now use "exactly." |
| RESPONSE FORMAT sections | **5 of 7** | rescue_numeric, rescue_general have explicit sections. conflict_batch, micro_conflict, gap_resolution describe format inline. alignment_tiebreak has a named section. header_hierarchy describes format in the example. All adequate. |
| STATUS VALUES glossary | **1 of 2 rescue prompts** | rescue_numeric has it, rescue_general does not. |
| Section header style | **Consistent** | ALL CAPS with colon (e.g., "RULES:", "FLANKING CONTEXT AND DUPLICATION:"). |
| Example section headers | **Minor inconsistency** | gap_resolution and conflict_batch use "EXAMPLES:" (plural). micro_conflict, rescue_numeric, alignment_tiebreak, header_hierarchy use "EXAMPLE:" (singular). All acceptable — matches actual count. |

### C. Techniques Not Needed (Confirmed)

| Technique | Verdict | Rationale |
|-----------|---------|-----------|
| **XML tags** | Skip | YAML structure + section headers + Pydantic structured output make XML tags unnecessary overhead. |
| **Chain of thought** | Skip (mostly) | Rescue prompts have `explanation` field (implicit CoT). Other prompts produce decisions directly — adding a reasoning field would require Pydantic model changes and increase token cost. Only worth it if quality issues emerge. |
| **Task decomposition** | Skip | Each prompt has a single, focused purpose. The pipeline already chains prompts (conflict_batch → rescue → etc.). |
| **Role/persona enrichment** | Skip | Functional openers ("You are resolving...") are more effective than expertise framing for these constrained structured-output tasks. |

### D. Template Rendering Safety (New Observation)

Static prompts (conflict_batch, micro_conflict, gap_resolution, alignment_tiebreak, header_hierarchy) bypass `format_map()` entirely when no kwargs are passed — so literal `{` and `}` in JSON examples are safe. Dynamic prompts (rescue_numeric, rescue_general) correctly use `{{`/`}}` for literal braces. The `_SafeFormatDict` fallback also protects against missing keys. This system is sound.

---

## Priority Summary

### Recommended (Medium Impact, Low Risk)

| # | File | Action |
|---|------|--------|
| 1 | `rescue_general.yaml` | Add 1 example showing the "explain why empty" case (page artifact or duplicate) |
| 2 | `rescue_general.yaml` | Add STATUS VALUES glossary (copy from rescue_numeric) |
| 3 | `alignment_tiebreak.yaml` | Fix example: remove `description` field from candidates (it belongs to local repair mode, not global) OR add a second example for local repair mode |

### Consider (Low Impact, Polish)

| # | File | Action |
|---|------|--------|
| 4 | `conflict_batch.yaml` | Future: add a 3rd example showing true multi-source merge (combining content from 2-3 sources) |
| 5 | `header_hierarchy.yaml` | Future: add a 2nd example showing `detected_title` non-null case |
| 6 | `_shared.yaml` | Future: add header/footer artifact pattern to extraction_errors |
| 7 | `alignment_tiebreak.yaml` | Future: add local repair arbitration example with `description`, `score`, `text`, `anchor_text` fields |

### No Changes Needed

- `micro_conflict.yaml` — strong as-is
- `rescue_numeric.yaml` — strong as-is
- `gap_resolution.yaml` — excellent, model prompt for the others
