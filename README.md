# AGR PDF Extraction Service

A production service for extracting structured text from scientific PDFs. Built for the [Alliance of Genome Resources](https://www.alliancegenome.org/).

The service runs three independent extraction engines on each PDF, then merges their outputs through a multi-stage consensus pipeline to produce a single, high-quality markdown document. The merging process uses a 3-way Needleman-Wunsch dynamic programming alignment to match content blocks across extractors, identifies where they agree and disagree, extracts word-level micro-conflicts from disagreement spans using a Numba-JIT-compiled token-level DP kernel, and resolves only those narrow spans through a layered approach that starts with deterministic heuristics and escalates to an LLM only when needed.

## Table of Contents

- [How It Works -- Overview](#how-it-works----overview)
- [The Three Extraction Engines](#the-three-extraction-engines)
- [The Consensus Pipeline](#the-consensus-pipeline)
  - [Step 1: Parse](#step-1-parse--break-each-output-into-blocks)
  - [Step 2: Align](#step-2-align--3-way-needleman-wunsch-dp-alignment)
  - [Step 3: Classify](#step-3-classify--determine-agreement-or-disagreement)
  - [Step 4: Guard Gates](#step-4-guard-gates--document-health-assessment)
  - [Step 5: Extract Micro-Conflicts](#step-5-extract-micro-conflicts--numba-jit-token-level-dp)
  - [Step 6: Resolve Conflicts](#step-6-resolve-conflicts--layered-resolution)
  - [Step 7: Assemble and Clean](#step-7-assemble-and-clean--build-the-final-document)
  - [Step 8: Heading Hierarchy](#step-8-heading-hierarchy--fix-section-structure)
- [The Alignment Subpackage](#the-alignment-subpackage)
  - [Scoring System](#scoring-system)
  - [Anchor-Windowed Partitioning](#anchor-windowed-partitioning)
  - [DP3 Recurrence](#dp3-recurrence)
  - [Traceback and Column Construction](#traceback-and-column-construction)
  - [Split/Merge Repair](#splitmerge-repair)
  - [Close-Score Arbitration](#close-score-arbitration)
- [LLM Integration and Cost Tracking](#llm-integration-and-cost-tracking)
- [Quality Metrics and Degradation Grading](#quality-metrics-and-degradation-grading)
- [Using the Service](#using-the-service)
  - [Web Interface](#web-interface)
  - [REST API](#rest-api)
  - [API Endpoints Reference](#api-endpoints-reference)
- [Deployment](#deployment)
  - [Docker Quick Start](#docker-quick-start)
  - [Architecture](#architecture)
  - [Management Commands](#management-commands)
  - [Durable Storage](#durable-storage)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)

---

## How It Works -- Overview

When you submit a PDF, the service:

1. **Extracts** the document with three independent engines (GROBID, Docling, Marker), each producing a markdown version of the paper
2. **Parses** each markdown output into typed blocks (headings, paragraphs, tables, equations, figures, citations) and normalizes extractor-specific artifacts
3. **Aligns** blocks across the three sources using anchor-windowed 3-way Needleman-Wunsch DP alignment with composable scoring
4. **Classifies** each aligned triple as AGREE_EXACT, AGREE_NEAR, GAP, or CONFLICT
5. **Extracts micro-conflicts** -- word-level disagreement spans within each CONFLICT segment using a Numba-JIT-compiled 3-way token-level DP kernel, preserving surrounding majority-agreed tokens
6. **Resolves** only the narrow disagreement spans through a layered approach: median-source selection, majority vote, per-segment micro-conflict LLM, three-tier rescue
7. **Assembles** the final merged document with proper heading hierarchy restored by LLM

The result is a single markdown file that is more accurate than any individual extractor could produce alone.

```
                            +------------+
                  PDF ------> | GROBID   |------> Markdown A
                  PDF ------> | Docling  |------> Markdown B
                  PDF ------> | Marker   |------> Markdown C
                            +------------+
                                  |
                                  v
                     +------------------------------+
                     |     Consensus Pipeline        |
                     |                               |
                     |  1. Parse into blocks          |
                     |  2. 3-way DP alignment         |
                     |  3. Classify agreement          |
                     |  4. Guard gates                 |
                     |  5. Token-level micro-conflicts |
                     |  6. Layered resolution           |
                     |  7. Assemble + dedup             |
                     |  8. Heading hierarchy            |
                     +------------------------------+
                                  |
                                  v
                         Final Merged Markdown
```

---

## The Three Extraction Engines

Each engine has different strengths. Using all three together compensates for individual weaknesses.

| Engine | How It Works | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **GROBID** | Machine-learning model trained specifically on scientific papers; uses Conditional Random Fields (CRF) | Excellent at identifying paper structure (title, authors, abstract, sections, references); understands scientific document layout | Can struggle with complex tables, figures, and non-standard layouts |
| **Docling** | IBM's document understanding toolkit; uses deep-learning vision models | Strong table extraction; good at preserving document formatting and structure | May occasionally merge adjacent sections or miss section boundaries |
| **Marker** | Vision-language model pipeline; "sees" the PDF like a human would | Handles complex layouts, multi-column text, and embedded equations well; good image extraction | Can sometimes hallucinate text in low-quality scans; may split sections differently |

All three engines run in parallel to minimize extraction time. GROBID is HTTP-based (runs as a separate service), while Docling and Marker run in-process and benefit from GPU acceleration when available.

---

## The Consensus Pipeline

The consensus pipeline is the core algorithm that merges three independent extractions into one accurate output. It is orchestrated by `consensus_service.py` and delegates to specialized modules for each stage.

### Step 1: Parse -- Break Each Output into Blocks

Each extractor's markdown output is first run through **source-level normalization** (stripping HTML tags, stray image links, bold/italic/code markup, and web links so comparisons are fair), then parsed into individual **blocks** using a mistune-based AST parser. A block is a logical unit of content:

- **Heading** -- Section titles (e.g., `## Introduction`, `### Methods`), with heading level recorded
- **Paragraph** -- Body text
- **Table** -- Data tables
- **Equation** -- Mathematical expressions
- **Figure reference** -- Image captions or figure descriptions
- **Citation list** -- Reference entries

Each block records its type, raw text, normalized text (for comparison), heading level (if applicable), position in the document (`order_index`), source extractor name, and original source markdown.

**Sparse extractor exclusion:** If one extractor produced dramatically fewer blocks than the others (less than 30% of the maximum), it is excluded from alignment entirely. This prevents a partially-failed extraction from poisoning the merge.

### Step 2: Align -- 3-Way Needleman-Wunsch DP Alignment

The pipeline needs to figure out which block from GROBID corresponds to which block from Docling and Marker. This is a three-way sequence alignment problem -- extractors may split paragraphs differently, skip sections, or order content slightly differently.

The alignment uses a **full 3-way Needleman-Wunsch dynamic programming algorithm** (implemented in `app/services/alignment/dp3.py`) that considers all three source sequences simultaneously. This is not a pairwise algorithm run twice -- it operates in a true 3-dimensional DP tensor.

**How it works at a high level:**

1. **Anchor partitioning** (optional, enabled by default): Before running the expensive full 3-way DP, the system identifies strong anchor blocks (H1/H2 headings by default) using a lightweight mini-DP pass. These anchors partition the document into smaller windows that are aligned independently, reducing the cubic complexity of the full DP.

2. **Per-window 3-way DP**: Within each window, the algorithm fills a 4-dimensional tensor of shape `(n_g+1) x (n_d+1) x (n_m+1) x 8` where `n_g`, `n_d`, `n_m` are the block counts for GROBID, Docling, and Marker respectively, and the 8 modes are the 7 legal transitions plus a START state.

3. **Traceback**: The optimal alignment path is reconstructed from backpointers, producing a sequence of `AlignmentColumn` objects -- each representing a group of up to three blocks (one from each extractor) that correspond to the same content.

4. **Repair pass**: Adjacent split/merge motifs (where one extractor split a paragraph that another kept whole) are detected and repaired by virtual concatenation scoring.

5. **Arbitration**: When two alignment paths score nearly identically (within a configurable delta), the system uses semantic reranking and optionally LLM tiebreaking to choose between them.

The result is a list of **aligned triples** -- groups of up to three blocks representing the same piece of content, each with a confidence score.

See [The Alignment Subpackage](#the-alignment-subpackage) for full technical details.

### Step 3: Classify -- Determine Agreement or Disagreement

Each aligned triple is classified into one of four categories:

| Classification | What It Means | What Happens |
|---------------|---------------|--------------|
| **AGREE_EXACT** | Two or more extractors produced identical text (after normalization) | Text is accepted as-is -- no LLM needed |
| **AGREE_NEAR** | Text is very similar (>92% token overlap AND >90% character-level match) with no differences in numbers or citations | Best version is selected automatically -- no LLM needed |
| **GAP** | A block appears in only one extractor (the others have nothing at that position) | Usually kept as-is, since one extractor found content the others missed |
| **CONFLICT** | The extractors disagree and the differences are significant enough to matter | Sent to the conflict resolution pipeline |

**Numeric and citation guardrails for AGREE_NEAR:** If two blocks are textually similar but differ in numbers (e.g., "p < 0.05" vs "p < 0.5") or citation references (e.g., "[1,2]" vs "[1,3]"), the block is escalated to CONFLICT. This prevents silently accepting corrupted data values.

**Strict numeric near mode** (default: enabled via `CONSENSUS_STRICT_NUMERIC_NEAR=true`): When enabled, AGREE_NEAR is disabled for any block containing numbers -- such blocks must be AGREE_EXACT or they become CONFLICT. This is the safest option for scientific PDFs but increases LLM usage.

**Table and equation escalation** (default: enabled via `CONSENSUS_ALWAYS_ESCALATE_TABLES=true`): Tables and equations are always escalated to CONFLICT regardless of similarity, since even minor formatting differences can change their meaning.

**Source preference for AGREE_NEAR:** When the best version needs to be selected, the pipeline uses a source preference table:

| Block Type | Preferred Extractor | Why |
|-----------|-------------------|-----|
| Headings | GROBID | Best at identifying paper structure |
| Paragraphs | Marker | Best at preserving complete body text |
| Tables | Docling | Strongest table extraction |
| Equations | Docling | Good structured content handling |
| Figure references | Marker | Best at image/figure context |
| Citation lists | GROBID | Trained specifically on reference parsing |

### Step 3b: GAP Deduplication

Before evaluating guard gates, the pipeline removes duplicate GAP blocks using a local window comparison. Each GAP block is compared against nearby GAP blocks (within a 3-block window). If two GAPs are 85% or more similar, the shorter duplicate is removed.

### Step 4: Guard Gates -- Document Health Assessment

Before resolving conflicts, the pipeline evaluates the overall difficulty of the document through several guard gates:

| Guard Gate | Threshold | What Triggers It |
|------------|-----------|-----------------|
| **Overall conflict ratio** | > adaptive threshold (default 40%) | Too many total disagreements across all block types |
| **Textual conflict ratio** | > 40% of text blocks are CONFLICT | Too many disagreements in body text |
| **Structured conflict ratio** | > 85% of tables/equations are CONFLICT | Extractors fundamentally disagree on structured content |
| **Alignment confidence** | < 0.50 | The block-matching step had very low confidence |

The only guard gate that causes a **hard failure** is alignment confidence -- if the block-matching step had very low confidence, the extraction fails with a clear error rather than producing unreliable output. All other guard metrics are logged as telemetry for monitoring but do not block the pipeline.

**Localized conflict relief:** If conflicts are clustered in one part of the document rather than spread throughout, this is noted in telemetry metrics. Localized conflicts within a span smaller than 35% of the document, with fewer than 25 conflict blocks, receive a configurable relief adjustment.

### Step 5: Extract Micro-Conflicts -- Numba-JIT Token-Level DP

Before sending anything to the LLM, the pipeline performs **3-way token-level dynamic programming alignment** on each CONFLICT segment to isolate exactly where the extractors disagree. This produces **micro-conflicts** -- small contiguous spans of disagreement surrounded by tokens where all extractors agree.

The extraction uses the same 3-way Needleman-Wunsch algorithm family as block-level alignment, but operates at the **token level** with a **Numba-JIT-compiled kernel** (`_token_dp_kernel` in `dp3.py`, decorated with `@numba.njit(nogil=True, cache=True)`). This gives ~50-100x speedup over pure Python and releases the GIL, enabling parallel token alignment across segments via `ThreadPoolExecutor`.

The process for each CONFLICT segment:

1. **Tokenize** all extractor texts (word-level split with sentence-end punctuation separated). Token streams are capped at 500 tokens per source.
2. **Normalize** for comparison -- strip markdown formatting, normalize Unicode dashes/quotes, collapse whitespace.
3. **Run 3-way token DP** -- the Numba kernel fills a 3D score tensor and backpointer array. Returns aligned token columns.
4. **Majority vote** at each position -- if 2+ extractors agree on a token, it is accepted as agreed.
5. **Extract conflict spans** -- contiguous runs of positions where no majority exists become individual micro-conflicts.
6. **Coalesce** nearby micro-conflicts (within a configurable gap, default 8 tokens) to avoid excessive fragmentation.
7. **Add context** -- each micro-conflict carries surrounding agreed tokens as context for the LLM (capped at 30 tokens).

**High-divergence fallback:** If the majority-agree ratio for a segment is below a threshold (default 40%), or a conflict span exceeds 12 tokens, or the segment has fewer than 10 tokens, the segment is treated as fully divergent and goes through full-segment resolution rather than micro-conflict resolution.

**Parallelism:** Token alignment runs in parallel threads because the Numba kernel releases the GIL. The block-level DP alignment is inherently sequential (single 3D tensor fill), but each window from anchor partitioning can be aligned independently.

### Step 6: Resolve Conflicts -- Layered Resolution

All conflicts are resolved through a **layered approach**, starting with the cheapest method and escalating only when needed:

#### Layer 1: Median-Source Selection (No LLM)

For blocks where all three extractors have output and the micro-conflict count is below a threshold (default 20), the pipeline picks the text that is most similar to the other two -- the "median" source. If the median source has at least 60% pairwise similarity to the others, it is accepted without any LLM call.

#### Layer 2: Per-Segment Micro-Conflict LLM Resolution

For remaining conflicts, the pipeline sends only the micro-conflict spans (from Step 5) to the LLM, rather than entire document zones. The LLM sees each disagreement span with its surrounding agreed context and resolves just that narrow region.

The LLM is instructed to:
- Pick the most accurate version or merge the best parts of the disagreement span
- Return ONLY the resolved text for the disagreement span (not the surrounding context)
- Preserve all numbers, Greek letters, subscripts, and math symbols exactly as they appear in sources
- Never fabricate content that does not appear in any extractor

The LLM responds with a structured `action` per conflict: **keep** (use resolved text) or **drop** (remove the span entirely, for artifact/noise). The resolved micro-conflict texts are stitched back into the majority-agreed token stream to reconstruct the full segment.

**Reasoning escalation:** If the initial LLM call fails to resolve a micro-conflict, it is retried with increased reasoning effort (up to the configured number of retry rounds, default 2).

**Numeric integrity guard:** After LLM resolution, the pipeline checks whether the output contains any number that did not appear anywhere in the source extractor texts. If novel numbers are detected, the segment is retried with a stricter instruction requiring explicit numeric-integrity justification. If the LLM still invents numbers, the segment falls back to best-source and the paper's quality score is heavily penalized.

**Post-resolution validation:** Every resolved segment goes through three checks:
1. **Dropped numbers check** -- numbers present in source but missing from output
2. **Content similarity check** -- output must be reasonably similar to at least one source
3. **Numeric truncation check** -- detects numbers that were shortened (e.g., "0.003" becoming "0.03")

#### Layer 3: Rescue Resolution (Three Tiers)

If any segments remain unresolved after micro-conflict resolution (the LLM returned empty or failed to parse), the pipeline attempts rescue resolution:

**Tier 1 -- Focused LLM retry with enriched context:** The unresolved segment is re-sent to the LLM with more surrounding context (5 blocks on each side) and any already-resolved neighboring segments. The LLM can also determine that a segment genuinely should be empty (e.g., duplicate heading, page artifact) by setting `is_intentionally_empty=true` with a required explanation.

**Tier 2 -- Best-source fallback (no LLM):** If the LLM retry also fails, the pipeline falls back to a deterministic heuristic -- containment check (if one extractor's text fully contains the others, that version is selected) or longest version. These segments are tracked as "degraded" in quality metrics.

**Tier 3 -- Skip:** If all extractor sources are empty for this segment, it is dropped entirely.

### Step 7: Assemble and Clean -- Build the Final Document

The resolved blocks are reassembled into a single markdown document in the correct order, then cleaned:

- **Post-assembly paragraph deduplication** -- removes blocks that appear more than once (common when extractors overlap at section boundaries)
- **HTML cleanup** -- strips leftover HTML tags, comments, and span elements
- **Whitespace normalization** -- ensures consistent spacing between sections

### Step 8: Heading Hierarchy -- Fix Section Structure

PDF extractors often produce "flat" headings -- every heading at the same level (e.g., all `##`) regardless of the actual document structure. This step uses the LLM to restore proper heading hierarchy.

The LLM receives a numbered list of all headings in order, along with a short content preview after each one. It assigns appropriate levels (H1 through H6) based on the logical structure of the paper.

The LLM can take three actions per heading:
- **set_level** -- Change to a specific level (e.g., H2 to H4)
- **keep_level** -- Leave the heading level as-is
- **demote_to_text** -- Remove heading formatting entirely (for lines that are not real headings)

**Strict validation:**
- The LLM cannot change heading text -- only the level
- The paper title must be H1
- Maximum one H1 heading
- No heading level jumps greater than allowed limits
- Demotion count is capped
- If validation fails, the original heading levels are kept unchanged

A post-hierarchy deduplication pass catches any duplicates exposed by heading demotion. Finally, a **QA gate** runs global duplicate detection with numeric and citation guardrails on the assembled document.

---

## The Alignment Subpackage

The alignment system lives in `app/services/alignment/` and consists of nine modules that implement a full 3-way sequence alignment pipeline. This section describes each component in detail.

### Scoring System

**Module:** `alignment/scoring.py`

The scoring system uses a composable `ScoreConfig` dataclass that operates in a normalized space (approximately -1 to +1). Every block-pair comparison produces a `PairScoreComponents` breakdown.

**Lexical similarity** is a weighted blend of three rapidfuzz metrics:

| Metric | Weight | What It Captures |
|--------|--------|-----------------|
| `token_set_ratio` | 55% | Overlap regardless of word order |
| `token_sort_ratio` | 30% | Overlap with word-order sensitivity |
| `ratio` (raw) | 15% | Strict character-level edit distance |

The blend ensures that tiny text fragments do not score near-perfect against long blocks (the raw ratio penalizes length mismatches).

**On top of the lexical score, the following adjustments are applied:**

| Adjustment | Value | When Applied |
|-----------|-------|-------------|
| Family match bonus | +0.06 | Both blocks are the same non-paragraph type (heading-heading, table-table, etc.) |
| Cross-family penalty | -0.55 | Incompatible block types (e.g., table vs heading) |
| Heading pair bonus | +0.10 | Both blocks are headings |
| Heading level bonus | up to +0.06 | Heading level delta is small (scaled by `1.0 - 0.25 * delta`) |
| Heading mismatch penalty | -0.30 | One block is a heading, the other is not |
| Length dampening (strong) | 0.20x-1.0x | Length ratio below 15% (short fragment vs long block) |
| Length dampening (mild) | 0.80x | Length ratio below 35% |
| Weak match penalty | variable | Lexical score below 0.60 threshold |
| Numeric mismatch penalty | -0.18 | Blocks contain different numeric tokens |
| Citation mismatch penalty | -0.12 | Blocks contain different citation references |

A **family whitelist** allows heading-paragraph and paragraph-citation mismatches without the full cross-family penalty, since extractors commonly disagree on the boundary between these types.

**Transition scoring** for the 3-way DP sums the pairwise scores across all present block combinations in a given transition (up to 3 pairs for a 111 match, 1 pair for a 2-source partial match).

### Anchor-Windowed Partitioning

**Module:** `alignment/partitioning.py`

Full 3-way DP has O(n_g * n_d * n_m) time complexity, which becomes expensive for long documents. Anchor partitioning reduces this by splitting the document into smaller windows.

**How it works:**

1. **Identify anchor candidates** -- blocks that are H1 or H2 headings (configurable via `max_heading_level`), and optionally tables/figures as secondary anchors.
2. **Run a mini 3-way DP** on just the anchor blocks to find their optimal alignment.
3. **Select strong anchors** -- aligned anchor triples where all three sources match with a score above `min_anchor_score` (default 0.72) and no ambiguity (no alternative alignment within `ambiguity_delta` of 0.03).
4. **Partition** the full block sequences at anchor boundaries, creating windows of blocks between consecutive anchors.
5. **Conservation invariant check** -- every block must appear in exactly one window after partitioning.
6. **Align each window independently** using the full 3-way DP.

If fewer than 2 sources have anchor candidates, partitioning is skipped and the full sequence is aligned as one window.

### DP3 Recurrence

**Module:** `alignment/dp3.py`

The core 3-way global alignment uses a 4-dimensional DP tensor:

```
scores[i][j][k][m] = best score aligning:
  - first i blocks of GROBID
  - first j blocks of Docling
  - first k blocks of Marker
  - with last transition being mode m
```

**Seven transition modes** (named by which sources consume a block):

| Mode | GROBID | Docling | Marker | Meaning |
|------|--------|---------|--------|---------|
| 111 | consume | consume | consume | All three sources match at this position |
| 110 | consume | consume | gap | GROBID and Docling match; Marker has a gap |
| 101 | consume | gap | consume | GROBID and Marker match; Docling has a gap |
| 011 | gap | consume | consume | Docling and Marker match; GROBID has a gap |
| 100 | consume | gap | gap | Only GROBID advances |
| 010 | gap | consume | gap | Only Docling advances |
| 001 | gap | gap | consume | Only Marker advances |

Plus a **START mode** (mode index 7) that acts as the initial state.

**Affine gap penalties** are used -- opening a new gap in a source costs `gap_open` (-0.40 by default), while extending an existing gap costs only `gap_extend` (-0.15). This encourages keeping gaps contiguous rather than scattered, which produces more biologically meaningful alignments.

The DP recurrence at each cell considers all 7 transitions from all 8 previous modes (7 transitions + START), taking the maximum. Pairwise scores are precomputed and cached. The final best score and mode are tracked across the last cell of the tensor.

**Token-level DP** reuses the same algorithm family but with a Numba-JIT-compiled kernel (`_token_dp_kernel`, decorated with `@numba.njit(nogil=True, cache=True)`) that operates on integer token IDs rather than Block objects. The Numba kernel:
- Releases the GIL (`nogil=True`), enabling parallel execution across threads
- Caches compiled code (`cache=True`) for fast subsequent invocations
- Uses numpy arrays directly for the DP tensor
- Caps token streams at 500 tokens per source to bound memory and runtime

### Traceback and Column Construction

**Modules:** `alignment/traceback.py`, `alignment/triples.py`

After the DP tensor is filled, traceback reconstructs the optimal alignment path by following backpointers from the final cell back to the origin. The result is a sequence of `AlignmentColumn` objects, each containing:

- `grobid_block`, `docling_block`, `marker_block` -- the block from each source (or `None` for gaps)
- `transition` -- which mode produced this column (e.g., "111", "110", "001")
- `local_score` -- the transition score at this position
- `cumulative_score` -- running total
- `reason` -- audit-friendly explanation string (via `alignment/telemetry.py`)
- `metadata` -- additional scoring details

A **monotonicity assertion** verifies that block indices increase along the alignment path.

The `build_aligned_triples` function converts columns into `AlignedTriple` objects with per-column confidence scores and a global alignment confidence (mean of all column confidences).

### Split/Merge Repair

**Module:** `alignment/repair.py`

PDF extractors sometimes split a single paragraph into two blocks, or merge two paragraphs into one. The repair pass detects and fixes these **split/merge motifs** in the alignment output.

**Motifs detected:**
- **Match + adjacent gap** -- a column with blocks from all sources, followed by a column with a block from only one source (suggesting one extractor split the content)
- **Gap + adjacent match** -- the reverse pattern

**Repair process:**
1. Virtually concatenate the gap block with its neighbor's block from the same source.
2. Score the concatenated text against the anchor blocks from other sources.
3. Accept the repair if the score exceeds the acceptance threshold (0.68) with at least 0.10 improvement over the original.
4. Use arbitration for close decisions.
5. Remove any columns that become fully empty after repair.

### Close-Score Arbitration

**Module:** `alignment/arbitration.py`

When two alignment candidates score within `ambiguity_delta` (default 0.03) of each other, the system cannot confidently choose based on DP score alone. The arbitration system provides two tiebreaking layers:

1. **Semantic rerank** (enabled by default): Compute the average lexical similarity across all columns for each candidate alignment. If one candidate is better by at least `semantic_margin` (0.02), prefer it.

2. **LLM tiebreak** (configurable, default disabled in production): Send a summary of the two candidate alignments to the LLM with the surrounding columns for context. The LLM returns a structured `AlignmentTieBreakResponse` choosing candidate A or B with an explanation.

The same arbitration logic is used for both end-mode selection in the main DP and repair candidate decisions.

An `ArbitrationContext` object tracks LLM tiebreak call counts for telemetry.

---

## LLM Integration and Cost Tracking

**Module:** `app/services/llm_service.py`

The LLM service wraps OpenAI API calls with structured Pydantic response models, retry logic, and thread-safe token accounting.

### LLM Call Types

| Call Type | Purpose | Default Model | Reasoning |
|-----------|---------|---------------|-----------|
| `micro_conflict` | Resolve word-level disagreement spans | gpt-5.2 | low |
| `rescue` | Retry unresolved segments with enriched context | gpt-5.2 | low |
| `header_hierarchy` | Restore heading levels | gpt-5.2 | low |
| `conflict_batch` | Batched full-segment conflict resolution | gpt-5.2 | low |
| `alignment_tiebreak` | Break close-score alignment ties | gpt-5.2 | low |

Each call type can independently override both model and reasoning effort via environment variables (e.g., `LLM_MODEL_ZONE_RESOLUTION`, `LLM_REASONING_ZONE_RESOLUTION`). Falls back to `LLM_MODEL` / `LLM_REASONING_EFFORT` when not set.

### Token Accumulator

The `TokenAccumulator` class provides thread-safe per-(call_type, model) token usage tracking:

- Records `prompt_tokens`, `completion_tokens`, `cached_tokens`, `reasoning_tokens` per call
- Keyed by `(call_type, model)` tuples
- Thread-safe via `threading.Lock`
- `compute_cost()` computes total USD cost from token counts using the `LLM_PRICING` dict in config
- `summary()` returns a serializable breakdown by call type and model

### Cost Tracking in Database

Token usage and cost are persisted on the `ExtractionRun` database row:
- `llm_usage_json` (JSONB) -- full breakdown by call type and model with token counts
- `llm_cost_usd` (Numeric) -- total estimated USD cost

Cost is recorded on both success and failure paths (partial LLM work before a failure is still tracked).

### LLM Pricing

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Cached Input (per 1M tokens) |
|-------|----------------------|----------------------|---------------------------|
| gpt-5.2 | $1.75 | $14.00 | $0.175 |
| gpt-5-mini | $0.25 | $2.00 | $0.025 |
| gpt-4.1 | $2.00 | $8.00 | $0.50 |
| gpt-4.1-mini | $0.40 | $1.60 | $0.10 |

### Unicode Guardrails

All LLM prompts include explicit instructions to preserve Greek letters (alpha, beta, gamma, delta, etc.), superscripts, subscripts, mathematical symbols, and special Unicode characters exactly as they appear in the source text.

---

## Quality Metrics and Degradation Grading

**Module:** `app/services/degradation_metrics.py`

Every extraction produces both top-level consensus metrics and detailed degradation metrics.

### Quality Score and Grade

The quality score is a weighted sum based on how each segment was resolved:

| Resolution Method | Quality Weight | Tier |
|-------------------|---------------|------|
| `agree_exact` | 1.00 | high |
| `agree_near` | 1.00 | high |
| `median_source` | 1.00 | high |
| `majority_vote` | 1.00 | high |
| `micro_conflict_llm` | 1.00 | high |
| `llm_conflict` | 1.00 | high |
| `llm_gap` | 1.00 | high |
| `llm_rescue_resolved` | 0.95 | medium_high |
| `llm_conflict_rescue_resolved` | 0.95 | medium_high |
| `llm_rescue_intentional_drop` | 0.95 | medium_high |
| `llm_conflict_rescue_intentional_drop` | 0.95 | medium_high |
| `deterministic_two_source` | 0.85 | medium |
| `llm_conflict_fallback_best_source` | 0.40 | low |

**Quality Grade:**

| Grade | Score Range |
|-------|------------|
| A | >= 0.95 |
| B | >= 0.85 |
| C | >= 0.70 |
| D | >= 0.50 |
| F | < 0.50 |

A paper-level penalty is applied when the numeric integrity guard fires (best-source fallback due to LLM inventing numbers).

### Degradation Metrics Object

The `degradation_metrics` object in the response includes:

| Field | Description |
|-------|-------------|
| `quality_score` / `quality_grade` | Overall quality (0.0-1.0) and letter grade |
| `resolution_summary` | Breakdown by method and quality tier |
| `degraded_segments` | Segments where LLM failed and deterministic fallback was used |
| `rescue_segments` | Segments that needed rescue retry, with explanations |
| `confidence_distribution` | Statistical spread: mean, median, min, max, std_dev, p10/p25/p75/p90, percentage below 50%/25% |
| `section_risk` | Per-section risk assessment keyed by resolved heading hierarchy paths (e.g., "Methods > RNA Extraction"). Reports risk level: none/low/medium/high |
| `risk_flags` | Boolean alerts: `high_degradation_rate` (>10%), `low_confidence_cluster` (3+ consecutive low-confidence), `degradation_concentrated` (>50% in one section), `high_risk_top_level_heading`, `numeric_integrity` |
| `token_efficiency` | Token usage breakdown: micro-conflict tokens, rescue tokens, total consensus tokens |

---

## Using the Service

### Web Interface

A browser-based interface is available at `http://localhost:5000`. Upload a PDF, select extraction methods, and view or download results. The web UI runs extraction synchronously and is best for manual testing and small-scale use.

### REST API

All extraction jobs run **asynchronously** -- you submit a PDF, get a `process_id`, and poll for results. This is the recommended interface for batch processing.

#### Submit an Extraction

```bash
curl -X POST http://localhost:5000/api/v1/extract \
  -F "file=@paper.pdf" \
  -F "methods=grobid,docling,marker" \
  -F "merge=true"
```

Returns a `process_id` (HTTP 202) that you use to check status and download results.

Optional fields:
- `reference_curie` -- link to an Alliance reference identifier (e.g., `AGRKB:101000000000001`)
- `mod_abbreviation` -- MOD abbreviation (e.g., `WB`, `FB`, `SGD`)
- `clear_cache` -- `true` to clear any existing cached results for this PDF before processing

#### Poll for Status

```bash
curl http://localhost:5000/api/v1/extract/{process_id}
```

**Status progression:** `pending` -> `started` -> `progress` -> `complete`

When complete, the response includes consensus metrics, degradation metrics, LLM cost, download paths, and artifact references.

#### Download Results

```bash
# Download the merged output
curl -O http://localhost:5000/api/v1/extract/{process_id}/download/merged

# Download individual extractor outputs
curl -O http://localhost:5000/api/v1/extract/{process_id}/download/grobid
curl -O http://localhost:5000/api/v1/extract/{process_id}/download/docling
curl -O http://localhost:5000/api/v1/extract/{process_id}/download/marker
```

Downloads try local cache first, then fall back to S3 artifact redirect.

#### Get Durable S3 URLs

```bash
curl http://localhost:5000/api/v1/extract/{process_id}/artifacts/urls
```

Returns pre-signed S3 URLs (1-hour expiry) for all artifacts.

### API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Service health check (GROBID, Redis, Celery worker status) |
| `/api/v1/extractions` | GET | List extraction runs (filterable by status, MOD, reference; paginated) |
| `/api/v1/extract` | POST | Submit a PDF for extraction (returns 202) |
| `/api/v1/extract/{id}` | GET | Poll extraction status and results |
| `/api/v1/extract/{id}/download/{method}` | GET | Download output (grobid, docling, marker, merged, audit) |
| `/api/v1/extract/{id}/images` | GET | List extracted images |
| `/api/v1/extract/{id}/images/{file}` | GET | Download a specific extracted image |
| `/api/v1/extract/{id}/logs` | GET | Get pre-signed URL for the NDJSON audit log |
| `/api/v1/extract/{id}/artifacts` | GET | Get artifact metadata |
| `/api/v1/extract/{id}/artifacts/urls` | GET | Get pre-signed S3 URLs for all artifacts |

---

## Deployment

### Docker Quick Start

```bash
# 1. Clone
git clone https://github.com/alliance-genome/agr_pdf_extraction_service.git
cd agr_pdf_extraction_service

# 2. Configure
cp .env.example .env
# Edit .env -- set at minimum: OPENAI_API_KEY

# 3. Deploy
cd deploy && ./deploy.sh
```

The deploy script creates data directories, builds containers, starts all services, and runs health checks. On first run, Docling and Marker download ML models (~2-5 GB), which takes several minutes.

**After deployment:**

| Endpoint | URL |
|----------|-----|
| Web UI | http://localhost:5000 |
| Swagger UI | http://localhost:5000/docs |
| OpenAPI spec | http://localhost:5000/openapi.yaml |
| Health check | http://localhost:5000/api/v1/health |

### Architecture

```
+-----------------------------------------------------------+
|                    Docker Compose Stack                     |
|                                                            |
|  +------------+   +-----------+   +-----------+           |
|  | Flask +    |   | Celery    |   | GROBID    |           |
|  | Gunicorn   |-->| Worker    |-->| 0.8.2-CRF |           |
|  | (Web+API)  |   | (GPU)     |   +-----------+           |
|  | CPU only   |   | (Jobs)    |                            |
|  +------------+   +-----------+   +-----------+           |
|       |                |          | PostgreSQL|           |
|       |                |--------->| 16-alpine |           |
|       |--------------->|          +-----------+           |
|                                   +-----------+           |
|                                   | Redis 7   |           |
|                                   | (Queue)   |           |
|                                   +-----------+           |
+-----------------------------------------------------------+
```

| Container | Image | Purpose |
|-----------|-------|---------|
| `pdfx-app` | pdfx-gpu | Flask + Gunicorn -- serves the web UI and REST API (CPU-only, no GPU) |
| `pdfx-worker` | pdfx-gpu | Celery worker -- runs PDF extraction jobs with GPU access (NVIDIA A10G) |
| `pdfx-grobid` | grobid/grobid:0.8.2-crf | GROBID CRF -- scientific PDF structure extraction (6 GB memory limit, JVM 2-4 GB) |
| `pdfx-postgres` | postgres:16-alpine | PostgreSQL -- durable extraction run tracking |
| `pdfx-redis` | redis:7-alpine | Redis 7 -- Celery job queue (db 0) and result backend (db 1) |

**Key architectural decisions:**

- The Flask app container runs **CPU-only** (`CUDA_VISIBLE_DEVICES=""`) to avoid GPU memory contention with the worker.
- The Celery worker runs with `--pool=solo` (single process) because Docling and Marker use internal threading and manage their own GPU memory.
- Code is mounted read-only (`app:/app/app:ro`) for hot-reload during development, while data directories are shared read-write.
- All containers use **GELF logging** to `logs.alliancegenome.org:12201` for centralized log aggregation.
- Health checks with generous start periods (120s) accommodate ML model loading time.

### Management Commands

```bash
cd deploy

./manage.sh status         # Container status + health checks
./manage.sh logs           # Follow all logs
./manage.sh logs worker    # Follow worker logs only
./manage.sh restart        # Restart all services
./manage.sh rebuild        # Rebuild images and redeploy
./manage.sh shell          # Open bash in the app container
./manage.sh worker-status  # Show Celery worker info
./manage.sh test paper.pdf # Quick extraction test
./manage.sh cleanup        # Stop and remove volumes
```

### Durable Storage

All extraction outputs are stored in S3 via the audit trail, and tracked in the PostgreSQL `extraction_run` table.

| What | Storage | Durability |
|------|---------|------------|
| Extraction status and metadata | PostgreSQL | Permanent |
| Markdown outputs and source PDFs | S3 (pre-signed URLs via API) | Permanent |
| Run logs (NDJSON format) | S3 | Permanent |
| Extracted images | S3 | Permanent |
| Local cache (fast access) | Docker volume | Cleared on cleanup |

**Prerequisites for durable storage:**

1. **IAM role** -- The EC2 instance needs an IAM instance profile with `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` on the audit bucket, and `ssm:GetParameter` on `/pdfx/*`
2. **IMDSv2 hop limit** -- Set to 2 so Docker containers can reach the EC2 metadata service for IAM credentials (`aws ec2 modify-instance-metadata-options --instance-id <id> --http-put-response-hop-limit 2`)
3. **SSM parameter** -- Store the bucket name at `/pdfx/audit-s3-bucket` in Parameter Store
4. **No AWS keys needed** -- The service uses the default boto3 credential chain (instance profile on EC2, env vars or config for local dev)

Without S3 access, extraction still works but outputs are only in local cache.

---

## Configuration Reference

All settings live in `config.py` with sensible defaults. Override via environment variables or `.env` file.

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required for consensus merge) |

### Extractors

| Variable | Default | Description |
|----------|---------|-------------|
| `GROBID_URL` | `http://localhost:8070` | GROBID server URL |
| `GROBID_REQUEST_TIMEOUT` | `120` | GROBID request timeout in seconds |
| `GROBID_INCLUDE_COORDINATES` | `false` | Include coordinate data in GROBID output |
| `GROBID_INCLUDE_RAW_CITATIONS` | `false` | Include raw citation strings |
| `DOCLING_DEVICE` | `cpu` | Docling device (`cpu` or `cuda` for GPU) |
| `MARKER_DEVICE` | `cpu` | Marker device (`cpu` or `auto` for GPU) |
| `MARKER_EXTRACT_IMAGES` | `true` | Extract images from PDFs |

### LLM Model Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-5.2` | Base model for all LLM calls |
| `LLM_REASONING_EFFORT` | `low` | Default reasoning effort for LLM calls |
| `LLM_MODEL_ZONE_RESOLUTION` | `gpt-5.2` | Model for micro-conflict resolution |
| `LLM_REASONING_ZONE_RESOLUTION` | _(empty, falls back to LLM_REASONING_EFFORT)_ | Reasoning effort for micro-conflict resolution |
| `LLM_MODEL_GENERAL_RESCUE` | `gpt-5.2` | Model for general rescue resolution |
| `LLM_REASONING_GENERAL_RESCUE` | _(empty)_ | Reasoning effort for general rescue |
| `LLM_MODEL_NUMERIC_RESCUE` | `gpt-5.2` | Model for numeric integrity rescue |
| `LLM_REASONING_NUMERIC_RESCUE` | _(empty)_ | Reasoning effort for numeric rescue |
| `LLM_MODEL_CONFLICT_BATCH` | `gpt-5.2` | Model for batched conflict resolution |
| `LLM_REASONING_CONFLICT_BATCH` | _(empty)_ | Reasoning effort for batched conflicts |
| `HIERARCHY_LLM_MODEL` | `gpt-5.2` | Model for heading hierarchy resolution |
| `HIERARCHY_LLM_REASONING` | `low` | Reasoning effort for heading hierarchy |
| `LLM_CONFLICT_BATCH_SIZE` | `500` | Number of conflicts per batch in batched resolution |
| `LLM_CONFLICT_MAX_WORKERS` | `100` | Max parallel workers for batched conflict resolution |
| `LLM_CONFLICT_RETRY_ROUNDS` | `2` | Number of retry rounds for unresolved micro-conflicts |

### Consensus Pipeline

| Variable | Default | Description |
|----------|---------|-------------|
| `CONSENSUS_ENABLED` | `true` | Enable the consensus merge pipeline |
| `CONSENSUS_NEAR_THRESHOLD` | `0.92` | Token similarity threshold for AGREE_NEAR classification |
| `CONSENSUS_LEVENSHTEIN_THRESHOLD` | `0.90` | Character-level similarity threshold for AGREE_NEAR |
| `CONSENSUS_CONFLICT_RATIO_FALLBACK` | `0.40` | Overall conflict ratio threshold (telemetry only) |
| `CONSENSUS_CONFLICT_RATIO_TEXTUAL_FALLBACK` | `0.40` | Text-block conflict ratio (telemetry only) |
| `CONSENSUS_CONFLICT_RATIO_STRUCTURED_FALLBACK` | `0.85` | Table/equation conflict ratio (telemetry only) |
| `CONSENSUS_ALIGNMENT_CONFIDENCE_MIN` | `0.50` | Alignment confidence below this fails the extraction |
| `CONSENSUS_LAYERED_ENABLED` | `true` | Enable layered conflict resolver (median-source + LLM) |
| `CONSENSUS_LAYERED_MEDIUM_SIM_THRESHOLD` | `0.60` | Minimum pairwise similarity for median-source selection |
| `CONSENSUS_MEDIAN_SOURCE_MAX_MICRO_CONFLICTS` | `20` | Max micro-conflicts for median-source to apply |
| `CONSENSUS_ALWAYS_ESCALATE_TABLES` | `true` | Always send tables/equations to LLM, even if extractors agree |
| `CONSENSUS_STRICT_NUMERIC_NEAR` | `true` | Disable AGREE_NEAR for any numeric-bearing block (must be AGREE_EXACT or CONFLICT) |
| `CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES` | `true` | Flag global duplicates in QA gate |
| `CONSENSUS_HIERARCHY_ENABLED` | `true` | Enable heading hierarchy resolution step |

### Alignment

| Variable | Default | Description |
|----------|---------|-------------|
| `CONSENSUS_ALIGNMENT_ANCHOR_PARTITIONING_ENABLED` | `true` | Enable anchor-windowed partitioning before DP |
| `CONSENSUS_ALIGNMENT_ANCHOR_MIN_SCORE` | `0.72` | Minimum score for a block to be a partition anchor |
| `CONSENSUS_ALIGNMENT_ANCHOR_INCLUDE_STRUCTURAL` | `false` | Include tables/figures as secondary anchors |
| `CONSENSUS_ALIGNMENT_ANCHOR_MAX_HEADING_LEVEL` | `2` | Maximum heading level for anchors (1=H1 only, 2=H1+H2) |
| `CONSENSUS_ALIGNMENT_AMBIGUITY_DELTA` | `0.03` | Score difference within which two alignments are "tied" |
| `CONSENSUS_ALIGNMENT_SEMANTIC_RERANK_ENABLED` | `true` | Enable semantic reranking for close-score tiebreaking |
| `CONSENSUS_ALIGNMENT_SEMANTIC_MARGIN` | `0.02` | Minimum semantic advantage to break a tie |
| `CONSENSUS_ALIGNMENT_LLM_TIEBREAK_ENABLED` | `true` | Enable LLM tiebreaking for alignment ambiguity |

### Localized Conflict

| Variable | Default | Description |
|----------|---------|-------------|
| `CONSENSUS_LOCALIZED_CONFLICT_SPAN_MAX` | `0.35` | Max document span ratio for localized conflict relief |
| `CONSENSUS_LOCALIZED_CONFLICT_RELIEF` | `0.15` | How much to relax conflict ratio when conflicts are localized |
| `CONSENSUS_LOCALIZED_CONFLICT_MAX_BLOCKS` | `25` | Max conflict blocks for localized relief to apply |

### Micro-Conflict Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `MICRO_CONFLICT_CONTEXT_CAP` | `30` | Max context tokens to include before/after each micro-conflict span |
| `MICRO_CONFLICT_HIGH_DIVERGENCE_RATIO_THRESHOLD` | `0.40` | If majority-agree ratio is below this, treat segment as fully divergent |
| `MICRO_CONFLICT_HIGH_DIVERGENCE_SPAN_THRESHOLD` | `12` | If a conflict span exceeds this many tokens, treat as high-divergence |
| `MICRO_CONFLICT_COALESCE_GAP` | `8` | Merge nearby micro-conflicts within this many tokens of each other |
| `MICRO_CONFLICT_HIGH_DIVERGENCE_MIN_TOKENS` | `10` | Minimum token count for high-divergence detection to apply |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://pdfx:pdfx@localhost:5432/pdfx` | PostgreSQL connection string |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Redis broker URL |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/1` | Redis result backend |
| `UPLOAD_FOLDER` | `./uploaded_pdfs` | Upload directory |
| `CACHE_FOLDER` | `./extraction_cache` | Local cache directory |
| `MAX_CONTENT_LENGTH` | `104857600` | Max upload size in bytes (100 MB) |
| `EXTRACTION_CONFIG_VERSION` | `4` | Bump to invalidate cached outputs |
| `AUDIT_S3_BUCKET` | _(empty)_ | S3 bucket for durable artifact storage; resolved from SSM if unset |
| `AUDIT_S3_BUCKET_SSM_PARAM` | `/pdfx/audit-s3-bucket` | SSM parameter name for bucket resolution |
| `AUDIT_S3_PREFIX` | `pdfx/audit` | S3 key prefix for audit artifacts |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for SSM and S3 clients |

---

## Project Structure

```
agr_pdf_extraction_service/
|-- config.py                         # Central configuration (all env vars)
|-- celery_app.py                     # Celery app + extract_pdf background task
|-- run.py                            # Dev server entry point
|-- benchmark_tiered.py               # Benchmark runner
|-- requirements.txt                  # Python dependencies
|-- .env.example                      # Environment variable template
|-- app/
|   |-- __init__.py                   # Flask app factory
|   |-- api.py                        # REST API v1 blueprint (11 endpoints)
|   |-- server.py                     # Web UI routes
|   |-- models.py                     # SQLAlchemy models (ExtractionRun)
|   |-- utils.py                      # File hashing, cache paths, image dirs
|   |-- logging_config.py             # GELF + file logging configuration
|   |-- openapi.yaml                  # OpenAPI 3.0 specification
|   |-- services/
|   |   |-- pdf_extractor.py          # Abstract base class for extractors
|   |   |-- grobid_service.py         # GROBID extractor (HTTP client)
|   |   |-- docling_service.py        # Docling extractor (in-process, GPU)
|   |   |-- marker_service.py         # Marker extractor (in-process, GPU)
|   |   |-- llm_service.py            # LLM client, TokenAccumulator, cost tracking
|   |   |-- audit_logger.py           # S3 audit trail + NDJSON log upload
|   |   |-- consensus_service.py      # Pipeline orchestrator (merge_with_consensus)
|   |   |-- consensus_models.py       # Block, AlignedTriple, MicroConflict data classes
|   |   |-- consensus_parsing_alignment.py  # Markdown parsing (mistune) + alignment orchestration
|   |   |-- consensus_classification_assembly.py  # Triple classification, assembly, dedup
|   |   |-- consensus_micro_conflicts.py    # Token-level DP micro-conflict extraction
|   |   |-- consensus_resolution.py         # Layered conflict + rescue resolution
|   |   |-- consensus_hierarchy_qa.py       # Heading hierarchy + QA gates
|   |   |-- consensus_reporting.py          # Metrics computation + audit entry builders
|   |   |-- consensus_pipeline_steps.py     # Backwards-compatible re-export layer
|   |   |-- degradation_metrics.py          # Quality scoring, grading, section risk
|   |   +-- alignment/                      # 3-way DP alignment subpackage
|   |       |-- __init__.py                 # Package exports
|   |       |-- dp3.py                      # 3-way Needleman-Wunsch DP (block + token level)
|   |       |-- scoring.py                  # Composable pair scoring (rapidfuzz + penalties)
|   |       |-- partitioning.py             # Anchor-windowed document partitioning
|   |       |-- traceback.py                # DP traceback + AlignmentColumn dataclass
|   |       |-- triples.py                  # Column-to-AlignedTriple conversion
|   |       |-- repair.py                   # Split/merge motif repair
|   |       |-- arbitration.py              # Close-score tiebreaking (semantic + LLM)
|   |       +-- telemetry.py                # Audit-friendly reason string formatting
|   +-- templates/
|       +-- index.html                # Web UI template
|-- tests/                            # Test suite
+-- deploy/
    |-- Dockerfile                    # CPU container image
    |-- Dockerfile.gpu                # GPU container image (CUDA 12.8 + Python 3.11)
    |-- docker-compose.yml            # CPU deployment stack
    |-- docker-compose.gpu.yml        # GPU deployment stack (5 services)
    |-- deploy.sh                     # One-command deployment
    +-- manage.sh                     # Management commands
```

## Deployment Target

Currently deployed on **AWS EC2** (g5.2xlarge):
- NVIDIA A10G GPU (24 GB VRAM), 8 vCPU, 32 GB RAM
- CUDA 12.8 runtime + Python 3.11
- Docker + Docker Compose with NVIDIA Container Toolkit
- GPU-accelerated Docling (CUDA) and Marker (auto-detect) inference
- Numba-JIT-compiled token alignment kernel

## License

MIT License
