# AGR PDF Extraction Service

A production service for extracting structured text from scientific PDFs. Built for the [Alliance of Genome Resources](https://www.alliancegenome.org/).

The service runs three independent extraction engines on each PDF, then intelligently merges their outputs to produce a single, high-quality markdown document. The merging process identifies where the extractors agree and disagree, resolves disagreements automatically, and uses an AI language model only when necessary — reducing cost by 65-72% compared to sending everything to the AI.

## Table of Contents

- [How It Works — Overview](#how-it-works--overview)
- [The Three Extraction Engines](#the-three-extraction-engines)
- [The Consensus Pipeline](#the-consensus-pipeline)
  - [Step 1: Parse](#step-1-parse--break-each-output-into-blocks)
  - [Step 2: Align](#step-2-align--match-blocks-across-extractors)
  - [Step 3: Classify](#step-3-classify--determine-agreement-or-disagreement)
  - [Step 4: Guard Gates](#step-4-guard-gates--decide-how-to-resolve-conflicts)
  - [Step 5: Resolve Conflicts](#step-5-resolve-conflicts--four-layers-of-resolution)
  - [Step 6: Assemble and Clean](#step-6-assemble-and-clean--build-the-final-document)
  - [Step 7: Heading Hierarchy](#step-7-heading-hierarchy--fix-section-structure)
- [Two-Tier Model Selection](#two-tier-model-selection)
- [Quality Metrics](#quality-metrics)
- [Cost Tracking](#cost-tracking)
- [Using the Service](#using-the-service)
  - [Web Interface](#web-interface)
  - [REST API](#rest-api)
  - [API Endpoints Reference](#api-endpoints-reference)
- [Deployment](#deployment)
  - [Docker Quick Start](#docker-quick-start)
  - [Architecture](#architecture)
  - [Management Commands](#management-commands)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)

---

## How It Works — Overview

When you submit a PDF, the service:

1. **Extracts** the document with three independent engines (GROBID, Docling, Marker), each producing a markdown version of the paper
2. **Compares** the three outputs block by block, looking for agreement and disagreement
3. **Keeps** the parts where extractors agree (no AI needed — this is typically 60-85% of the document)
4. **Resolves** the parts where extractors disagree, using a layered approach that starts with simple heuristics and escalates to an AI language model only when needed
5. **Assembles** the final merged document with proper heading structure

The result is a single markdown file that is more accurate than any individual extractor could produce alone.

```
                            ┌──────────┐
                  PDF ─────▶│  GROBID   │────▶ Markdown A
                  PDF ─────▶│  Docling  │────▶ Markdown B
                  PDF ─────▶│  Marker   │────▶ Markdown C
                            └──────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │   Consensus Pipeline    │
                     │                         │
                     │  1. Parse into blocks    │
                     │  2. Align across sources │
                     │  3. Classify agreement   │
                     │  4. Resolve conflicts    │
                     │  5. Assemble output      │
                     │  6. Fix heading levels   │
                     └────────────────────────┘
                                  │
                                  ▼
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

All three engines run in parallel to minimize extraction time.

---

## The Consensus Pipeline

The consensus pipeline is the core algorithm that merges three independent extractions into one accurate output. It works in seven steps.

### Step 1: Parse — Break Each Output into Blocks

Each extractor's markdown output is split into individual **blocks**. A block is a logical unit of content:

- **Heading** — Section titles (e.g., "## Introduction", "### Methods")
- **Paragraph** — Body text
- **Table** — Data tables
- **Equation** — Mathematical expressions
- **Figure reference** — Image captions or figure descriptions
- **Citation list** — Reference entries

Each block records its type, text content, position in the document, and which extractor produced it.

Before parsing, each output goes through **source-level normalization** — removing extractor-specific artifacts like HTML tags, stray formatting, inconsistent whitespace, web links (converted to plain text), and broken image references so that comparisons are fair.

**Example:** A paper's Methods section might produce these blocks:

```
Block 1:  [heading]    "## Methods"
Block 2:  [paragraph]  "Drosophila melanogaster strains were maintained at 25°C..."
Block 3:  [paragraph]  "Total RNA was extracted using TRIzol reagent (Invitrogen)..."
Block 4:  [table]      "| Gene | Primer Sequence | Tm (°C) | ..."
Block 5:  [heading]    "## Results"
```

### Step 2: Align — Match Blocks Across Extractors

The pipeline needs to figure out which block from GROBID corresponds to which block from Docling and Marker. This is not always obvious — extractors may split paragraphs differently, skip sections, or order content slightly differently.

The alignment uses the [**Hungarian algorithm**](https://en.wikipedia.org/wiki/Hungarian_algorithm) (a well-known optimization method for finding optimal one-to-one assignments) to find the best matching across extractors. It scores each potential match on three factors:

| Factor | Weight | What It Measures |
|--------|--------|-----------------|
| **Heading match** | 40% | Do both blocks have the same section heading nearby? |
| **Content similarity** | 40% | How similar is the actual text? (measured with fuzzy string matching) |
| **Position proximity** | 20% | Are they in roughly the same location in the document? |

**Example:** Consider how three extractors might handle the same paragraph about fly strains:

```
GROBID  block 4:  "Drosophila melanogaster strains were maintained at 25°C on standard..."
Docling block 5:  "Drosophila melanogaster strains were maintained at 25°C on standard..."
Marker  block 3:  "D. melanogaster strains were maintained at 25 °C on standard..."
```

Even though the blocks are at different positions (4, 5, 3) and Marker abbreviated the genus name, the algorithm matches them into a single **aligned triple** because the content similarity is high (40%), they're all under the "Methods" heading (40%), and their relative positions are close (20%).

The result is a list of these aligned triples — groups of up to three blocks (one from each extractor) that represent the same piece of content. The alignment also computes a **confidence score** (0.0 to 1.0) reflecting how well the blocks matched overall.

**Sparse extractor exclusion:** If one extractor produced dramatically fewer blocks than the others (less than 30% of the maximum), it is excluded from alignment. This prevents a partially-failed extraction from poisoning the merge.

### Step 3: Classify — Determine Agreement or Disagreement

Each aligned triple is classified into one of four categories:

| Classification | What It Means | What Happens |
|---------------|---------------|--------------|
| **AGREE_EXACT** | Two or more extractors produced identical text (after normalization) | Text is accepted as-is — no AI needed |
| **AGREE_NEAR** | Text is very similar (>92% token overlap AND >90% character-level match) with no differences in numbers or citations | Best version is selected automatically — no AI needed |
| **GAP** | A block appears in only one extractor (the others have nothing at that position) | Usually kept as-is, since one extractor found content the others missed |
| **CONFLICT** | The extractors disagree and the differences are significant enough to matter | Sent to the conflict resolution pipeline (see next step) |

**Examples of each classification:**

```
AGREE_EXACT — All three say the same thing:
  GROBID:  "Flies were raised at 25°C on standard cornmeal medium."
  Docling: "Flies were raised at 25°C on standard cornmeal medium."
  Marker:  "Flies were raised at 25°C on standard cornmeal medium."
  → Accepted as-is. ✓

AGREE_NEAR — Very similar, minor wording difference:
  GROBID:  "RNA was extracted using TRIzol reagent (Invitrogen, Cat. #15596026)."
  Docling: "RNA was extracted using TRIzol reagent (Invitrogen, Cat. No. 15596026)."
  → 96% similar, same numbers and citations → AGREE_NEAR. ✓

GAP — Only one extractor found this content:
  GROBID:  (nothing)
  Docling: (nothing)
  Marker:  "Supplementary Table S1 contains the full list of primers used."
  → Kept from Marker — it found content the others missed. ✓

CONFLICT — Each extractor captured the table differently:
  GROBID:  "| Gene | Expression | | dpp | 2.5-fold |"
  Docling: "| Gene | Fold Change | p-value |\n| dpp | 2.5 | 0.003 |"
  Marker:  "Gene Expression dpp 2.5-fold (p < 0.003)"
  → Very different structure and content → sent to AI for resolution. ⚠
```

**How the "best" version is selected for AGREE_NEAR:** When two or more extractors agree, the pipeline picks the version from the extractor that is strongest for that type of content, using a source preference table:

| Block Type | Preferred Extractor | Why |
|-----------|-------------------|-----|
| Headings | GROBID | Best at identifying paper structure |
| Paragraphs | Marker | Best at preserving complete body text |
| Tables | Docling | Strongest table extraction |
| Equations | Docling | Good structured content handling |
| Figure references | Marker | Best at image/figure context |
| Citation lists | GROBID | Trained specifically on reference parsing |

Only the agreeing extractors are considered — the outlier (if any) is excluded. If the preferred extractor isn't among the agreeing pair, the first agreeing extractor's version is used.

**Numeric and citation guardrails:** AGREE_NEAR has strict protections — if two blocks are textually similar but differ in numbers (e.g., "p < 0.05" vs "p < 0.5") or citation references (e.g., "[1,2]" vs "[1,3]"), the block is escalated to CONFLICT. This prevents silently accepting corrupted data values.

There is one narrow exception: if the blocks are extremely similar (>97% token overlap AND >95% character match), the numeric difference is tiny (≤2 numbers, ≤1 citation), and the differing numbers are tied to Figure/Table/Section references (not data values), the block can still qualify as AGREE_NEAR. This prevents spurious conflicts from extractors formatting "Figure 1" vs "Fig. 1" differently while still catching real data discrepancies.

**Table and equation escalation:** Tables and equations are optionally always escalated to CONFLICT regardless of similarity, since even minor formatting differences in these structured elements can change their meaning.

### Step 3b: GAP Deduplication

Before evaluating guard gates, the pipeline removes duplicate GAP blocks in two passes:

1. **Local dedup** — Compares each GAP block against nearby GAP blocks (within a 3-block window). If two GAPs are ≥85% similar, the duplicate is removed.
2. **Global dedup** — Compares each remaining GAP block against ALL other blocks (regardless of classification or distance). If a GAP's text is ≥85% similar to another block, or if one fully contains the other (≥90% containment), the GAP is removed.

This prevents the same content from appearing twice in the final output when multiple extractors captured it at slightly different positions.

### Step 4: Guard Gates — Decide How to Resolve Conflicts

Before resolving conflicts, the pipeline evaluates the overall difficulty of the document through several **guard gates**. These decide whether conflicts can be resolved zone-by-zone (efficient) or whether the entire document needs to be sent to the AI (expensive but thorough).

| Guard Gate | Threshold | What Triggers It |
|------------|-----------|-----------------|
| **Overall conflict ratio** | > adaptive threshold (default 40%) | Too many total disagreements across all block types |
| **Textual conflict ratio** | > 40% of text blocks are CONFLICT | Too many disagreements in body text for zone-based resolution |
| **Structured conflict ratio** | > 85% of tables/equations are CONFLICT | Extractors fundamentally disagree on structured content |
| **Alignment confidence** | < 0.50 | The block-matching step had very low confidence, suggesting the extractors produced very different document structures |

If any guard gate triggers, the pipeline falls back to sending the **entire document** to the AI for a full-document merge. This is the safety net — it costs more but gives the AI the most context to work with.

**Two-source conflict exclusion:** When computing conflict ratios for guard gate decisions, the pipeline excludes conflicts where only two extractors produced output (the third had nothing at that position). These "two-source conflicts" are simpler to resolve (just pick the better of two versions) and shouldn't trigger the full-document fallback.

**Localized conflict relief:** If conflicts are clustered in one part of the document (e.g., a difficult table section) rather than spread throughout, the overall conflict ratio threshold is relaxed by up to +0.15 (capped at 0.95). For this relief to apply, conflicts must span ≤35% of the document and involve no more than 25 conflict blocks. This prevents a single tricky section from forcing a full-document merge when the rest of the paper merged cleanly.

### Step 5: Resolve Conflicts — Four Layers of Resolution

Conflicts that pass the guard gates are resolved through a **layered approach**, starting with the cheapest method and escalating only when needed:

#### Layer 1: Median-Source Selection

For blocks where all three extractors have output, the pipeline picks the text that is most similar to the other two — the "median" source. If the median source has at least 60% similarity to the others, it is accepted without any AI call.

*Why this works:* If two extractors say roughly the same thing and one is an outlier, the consensus of two is usually correct.

**Example:**

```
GROBID:  "The dsRNA was injected into third-instar larvae (n = 45)."
Docling: "The dsRNA was injected into third-instar larvae (n = 45)."
Marker:  "The dsRNA was injected into third instar larvae (n=45)."
         ← minor formatting differences (hyphen, spacing)

Median-source picks GROBID or Docling (they're most similar to each other).
No AI call needed — resolved programmatically.
```

#### Layer 2: Zone-Based LLM Resolution

Remaining conflicts are grouped into **conflict zones** — contiguous stretches of the document where disagreements occur, along with a few surrounding "context" blocks so the AI understands what comes before and after.

Each zone is sent to the AI language model, which sees all extractor versions side by side and attempts to select the best text for each conflicting segment. The AI is instructed to:
- Attempt to choose the most complete and accurate version
- Never fabricate content that doesn't appear in any extractor
- Preserve all data values, citations, and scientific terminology exactly

The AI does not have access to the original PDF — it can only compare what the three extractors produced and use its judgment to pick the version that appears most complete and internally consistent.

**Example of what the AI sees for a conflict zone:**

```
── Context (agreed, for reference) ──────────────────────
seg_010 [AGREE_EXACT]: "## Results"
seg_011 [AGREE_NEAR]:  "We first examined the expression pattern of dpp in wing discs."

── Conflict (AI must resolve) ──────────────────────────
seg_012 [CONFLICT]:
  GROBID:  "dpp-lacZ expression was detected in a broad stripe along the
            A/P boundary (Fig. 2A, B; Table 1)"
  Docling: "dpp-lacZ expression was detected in a broad stripe along the
            anterior-posterior boundary (Figure 2A, B; Table 1)"
  Marker:  "dpp-lacZ expression was detected in a broad stripe along the
            A/P boundary (Fig. 2A,B, Table 1)"

── Context (agreed) ────────────────────────────────────
seg_013 [AGREE_NEAR]:  "This pattern was consistent across all genotypes tested."
```

**How the AI responds — structured keep/drop decisions:**

For every segment in the zone, the AI must return a structured decision with an explicit `action` field:

| Action | Meaning | When It's Used |
|--------|---------|---------------|
| **keep** | Include this text in the final document | ALL conflict and near_agree segments — the AI must always provide resolved text for these |
| **drop** | Exclude this segment entirely | ONLY for gap segments that are artifacts, duplicates, or page noise (e.g., a stray header repeated from a previous page) |

The AI is **never allowed** to drop a conflict or near_agree segment — those always need resolved text. If the AI says "keep" but provides empty text, or tries to drop a conflict segment, the segment is treated as **unresolved** and sent to rescue resolution (see Layer 3 below).

This structured response eliminates ambiguity: previously, an empty string could mean "I couldn't resolve this" or "this segment should genuinely be empty." The explicit action field makes the AI's intent clear.

#### Layer 3: Rescue Resolution (Three Tiers)

If any segments remain unresolved after zone-based resolution (the AI returned empty or failed to parse), the pipeline attempts **rescue resolution** — a series of fallback strategies for that specific segment:

**Tier 1 — Focused AI retry with enriched context:**

The single unresolved segment is re-sent to the AI, but this time with more surrounding context (5 blocks on each side instead of the usual 2) and any already-resolved neighboring segments. The AI is told that it previously returned empty for this segment, and is asked to either provide resolved text or explain why the segment should genuinely be empty.

```
── Surrounding context (5 blocks, for understanding flow) ───────
seg_038 [resolved]: "Worms were scored daily for survival..."
seg_039 [resolved]: "Statistical analysis was performed using..."
seg_040 [resolved]: "| Strain | Median Lifespan | Max Lifespan |..."
seg_041 [resolved]: "All DSV-fed groups showed altered survival curves..."

── Segment to rescue (previously returned empty) ───────────────
seg_042 [CONFLICT — unresolved]:
  GROBID:  "Table 2. Phenotype summary"
  Docling: "Table 2. Phenotype summary for RNAi knockdowns
            in larval wing discs (n = 30 per strain)"
  Marker:  "Table 2. Phenotype summary for RNAi knockdowns"

── More surrounding context ────────────────────────────────────
seg_043 [resolved]: "Gene expression values were normalized to..."
seg_044 [resolved]: "Among the proteostasis-related genes..."

AI response → resolved_text: "Table 2. Phenotype summary for RNAi
  knockdowns in larval wing discs (n = 30 per strain)"
  explanation: "Chose Docling — most complete caption with sample size."
```

**The "intentionally empty" path:** The AI can also determine that a segment genuinely should not appear in the final document. In this case it sets `is_intentionally_empty=true` and must explain why:

```
── Segment to rescue (previously returned empty) ───────────────
seg_078 [GAP — unresolved]:
  GROBID:  (nothing)
  Docling: (nothing)
  Marker:  "## Methods"

── Surrounding context ──────────────────────────────────────────
seg_077 [resolved]: "## Materials and Methods"
seg_079 [resolved]: "Drosophila melanogaster strains were maintained..."

AI response → resolved_text: ""
  is_intentionally_empty: true
  explanation: "Duplicate heading — 'Methods' is a repeat of the
    already-resolved 'Materials and Methods' heading at seg_077.
    Marker extracted it as a separate block but it refers to
    the same section."
```

Valid reasons for intentionally empty include: duplicate of nearby text, page artifact (headers/footers repeated on each page), metadata noise (page numbers, running titles), or figure labels that don't belong inline. The explanation is always required and is recorded in the audit log.

**Tier 2 — Best-source fallback (no AI):**

If the AI retry also fails, the pipeline falls back to a deterministic heuristic — no AI involved. It checks whether one extractor's text fully contains the others (a containment check), and if so, that version is clearly the most complete. Otherwise, the longest version is used. These segments are tracked as "degraded" in the detailed degradation metrics (separate from the top-level consensus metrics shown above).

```
seg_042 still unresolved after Tier 1:
  GROBID:  "Table 2. Phenotype summary"
  Docling: "Table 2. Phenotype summary for RNAi knockdowns in larval wing discs"
  Marker:  "Table 2. Phenotype summary for RNAi knockdowns"

Containment check: GROBID's text is contained within Docling's text ✓
→ Docling selected as best source (most complete version).
```

**Tier 3 — Skip:**

If all extractor sources are empty for this segment (all three produced nothing), the segment is dropped entirely. This is rare and typically means the segment was an artifact of the alignment step rather than real content.

#### Layer 4: Full-Document LLM Fallback

If guard gates triggered earlier, or if rescue still fails for critical content, the entire document is sent to the strongest AI model (GPT-5.2) for a complete merge. This is the most expensive path but gives the AI maximum context to work with.

### Step 6: Assemble and Clean — Build the Final Document

The resolved blocks are reassembled into a single markdown document in the correct order. The output goes through several cleaning passes:

- **Deduplication** — Removes blocks that appear more than once (common when extractors overlap at section boundaries)
- **HTML cleanup** — Strips leftover HTML tags, comments, and span elements
- **Whitespace normalization** — Ensures consistent spacing between sections

### Step 7: Heading Hierarchy — Fix Section Structure

PDF extractors often produce "flat" headings — every heading at the same level (e.g., all `##`) regardless of the actual document structure. The final step uses the AI to restore the proper heading hierarchy.

The AI receives a numbered list of all headings in order, along with a short content preview after each one (so it can understand what each section contains). It then assigns appropriate levels (H1 through H6) based on the logical structure of the paper.

**Real example** from a paper about the Alliance of Genome Resources:

```
BEFORE (flat — all ## from extractors):

  ## Abstract
  ## Introduction
  ## The web portal
  ## Community homepages
  ## Xenopus in the Alliance
  ## New gene page section: paralogy
  ## JBrowse sequence detail widget
  ## Model organism BLAST
  ## AllianceMine
  ## SimpleMine
  ## Harmonized data models
  ## Persistent store architecture
  ## Security, stability, and backups
  ## Literature acquisition
  ## APIs
  ## Outreach and interactions
  ## The Alliance help desk
  ## Online documentation
  ...

AFTER (proper hierarchy restored):

  # Updates to the Alliance of Genome Resources central infrastructure
  ## Abstract
  ## Introduction
  ## Methods / Methodology
    ### Harmonized data models
    ### Persistent store architecture
    ### Security, stability, and backups
    ### Literature acquisition (ABC)
    ### APIs
  ## Results
    ### The web portal
      #### Community homepages
      #### Xenopus in the Alliance
      #### New gene page section: paralogy
      #### JBrowse sequence detail widget
      #### Model organism BLAST (SequenceServer)
      #### AllianceMine
      #### SimpleMine
  ## Discussion
    ### Outreach and interactions
      #### The Alliance help desk
      #### Online documentation
  ...
```

The AI recognized that "Community homepages", "Xenopus in the Alliance", etc. are subsections under "The web portal" (which is itself under "Results"), and that "Harmonized data models", "Persistent store architecture", etc. belong under "Methods". It also detected the paper title from the opening text and added it as H1.

The AI can take three actions per heading:
- **set_level** — Change to a specific level (e.g., H2 → H4)
- **keep_level** — Leave the heading level as-is
- **demote_to_text** — Remove heading formatting entirely (for lines that aren't real headings)

This step has strict validation:
- The AI cannot change heading text — only the level
- The title must be H1 (or detected in the opening text)
- If validation fails, the original heading levels are kept unchanged

---

## Two-Tier Model Selection

The service uses two AI models to balance cost and quality:

| Task | Model | Why |
|------|-------|-----|
| **Zone resolution** (resolving individual conflict sections) | GPT-5-mini | Handles most conflicts well at lower cost |
| **Large zone resolution** (zones with >20,000 estimated tokens) | GPT-5.2 | Complex/large zones need the stronger model |
| **Heading hierarchy** | GPT-5.2 | Needs holistic document understanding |
| **Full-document merge** (fallback) | GPT-5.2 | Always uses the strongest model for full-document passes |

If GPT-5-mini fails to resolve a zone, it is automatically retried with GPT-5.2.

---

## Quality Metrics

Every extraction produces a **consensus metrics** report that tells you how the merge went:

| Metric | What It Tells You |
|--------|------------------|
| `total_blocks` | Total content blocks identified across all extractors |
| `agree_exact` | Blocks where extractors produced identical text |
| `agree_near` | Blocks where extractors were very similar (accepted automatically) |
| `gap` | Blocks found by only one extractor |
| `conflict` | Blocks where extractors disagreed (resolved by the pipeline) |
| `conflict_ratio` | Fraction of blocks that were conflicts (lower is better; typical range 0.10-0.35) |
| `alignment_confidence` | How well the blocks matched across extractors (0.0-1.0; higher is better) |
| `fallback_triggered` | Whether the full-document AI merge was needed (false is better — means zone-based resolution handled everything) |

**Example metrics from a typical paper:**

```json
{
  "total_blocks": 174,
  "agree_exact": 89,
  "agree_near": 52,
  "gap": 15,
  "conflict": 18,
  "conflict_ratio": 0.11,
  "alignment_confidence": 0.82,
  "fallback_triggered": false,
  "fallback_reason": null,

  "degradation_metrics": {
    "quality_score": 0.943,
    "quality_grade": "A",

    "resolution_summary": {
      "total_resolved_segments": 18,
      "by_quality_tier": {
        "high":        { "count": 12, "pct": 66.7 },
        "medium_high": { "count": 5,  "pct": 27.8 },
        "medium":      { "count": 1,  "pct": 5.6 }
      }
    },

    "degraded_segments": {
      "count": 1,
      "pct_of_total": 0.6,
      "pct_of_resolved": 5.6,
      "segment_ids": ["seg_042"]
    },

    "rescue_segments": {
      "count": 2,
      "details": [
        {
          "segment_id": "seg_078",
          "method": "llm_conflict_rescue_intentional_drop",
          "rescue_explanation": "Duplicate heading — repeat of 'Materials and Methods'"
        },
        {
          "segment_id": "seg_055",
          "method": "llm_conflict_rescue_resolved",
          "rescue_explanation": "Chose Docling — most complete table caption"
        }
      ]
    },

    "confidence_distribution": {
      "mean": 0.87,
      "median": 0.91,
      "min": 0.42,
      "max": 1.0,
      "std_dev": 0.15,
      "p10": 0.52,
      "p25": 0.78,
      "p75": 0.96,
      "p90": 1.0,
      "below_50_pct": 5.6,
      "below_25_pct": 0.0
    },

    "section_risk": {
      "Materials and Methods": {
        "heading_level": 2,
        "total_segments": 31,
        "degraded": 0,
        "pct_degraded": 0.0,
        "risk": "none"
      },
      "Materials and Methods > RNA Extraction": {
        "heading_level": 3,
        "total_segments": 6,
        "degraded": 2,
        "pct_degraded": 33.3,
        "risk": "high"
      }
    },

    "risk_flags": {
      "high_degradation_rate": false,
      "low_confidence_cluster": false,
      "degradation_concentrated": false,
      "high_risk_top_level_heading": false
    },

    "token_efficiency": {
      "zone_resolution_tokens": 12400,
      "rescue_call_tokens": 1800,
      "total_consensus_tokens": 14200,
      "estimated_full_merge_tokens": 54200,
      "tokens_saved": 40000,
      "savings_pct": 73.8
    }
  }
}
```

This paper had 174 content blocks. 141 (81%) were resolved automatically without AI. Only 18 conflicts (11%) needed AI resolution, and zone-based resolution handled them all (no full-document fallback needed).

The nested `degradation_metrics` object provides deeper insight into how the merge went:

| Sub-object | What It Tells You |
|-----------|------------------|
| `quality_score` / `quality_grade` | Overall quality grade (A/B/C/D/F) based on resolution methods and confidence |
| `resolution_summary` | How conflicts were resolved, grouped by quality tier (high = exact/near agreement, medium = LLM fallback) |
| `degraded_segments` | Segments where AI resolution failed and a deterministic fallback was used — these are lower confidence |
| `rescue_segments` | Segments that needed rescue (Tier 1 retry) — includes the AI's explanation for each decision |
| `confidence_distribution` | Statistical spread of resolution confidence scores — includes mean, median, percentiles (p10/p25/p75/p90), std_dev, and percentage below 50%/25% confidence |
| `section_risk` | Per-section degradation summary keyed by the paper's **actual resolved headings** (post heading-hierarchy fix). Keys are hierarchical paths like `Materials and Methods > RNA Extraction`, and each entry reports degraded counts and a risk level (none/low/medium/high) based on `% degraded` in that section. |
| `risk_flags` | Boolean alerts — `high_degradation_rate` (>10% of blocks degraded), `low_confidence_cluster` (3+ consecutive low-confidence segments), `degradation_concentrated` (>50% of degraded segments in a single zone), `high_risk_top_level_heading` (any H1/H2 section has `risk=high`) |
| `token_efficiency` | Token usage breakdown — zone resolution tokens, rescue tokens, total, estimated full-merge tokens, tokens saved, and savings percentage |

**Resolution method vocabulary:** Each resolved segment is tagged with a `method` string that describes exactly how it was resolved. These appear in `resolution_summary.by_method` and in per-segment audit entries.

| Method | Weight | What Happened |
|--------|--------|---------------|
| `median_source` | 1.0 | Median-source heuristic picked the consensus version (no AI) |
| `llm_conflict` | 1.0 | AI resolved a conflict segment in zone resolution |
| `llm_near_agree` | 1.0 | AI resolved a near-agree segment in zone resolution |
| `llm_gap` | 1.0 | AI resolved a gap segment in zone resolution |
| `llm_rescue_resolved` | 0.95 | AI provided text on rescue retry |
| `llm_conflict_rescue_resolved` | 0.95 | AI provided text on rescue retry (conflict segment) |
| `llm_near_agree_rescue_resolved` | 0.95 | AI provided text on rescue retry (near-agree segment) |
| `llm_rescue_intentional_drop` | 0.95 | AI determined segment should be empty on rescue retry (with explanation) |
| `llm_conflict_rescue_intentional_drop` | 0.95 | AI determined segment should be empty on rescue retry (with explanation) |
| `llm_near_agree_rescue_intentional_drop` | 0.95 | AI determined segment should be empty on rescue retry (with explanation) |
| `deterministic_two_source` | 0.85 | Only two extractors present; deterministic pick (no AI) |
| `llm_conflict_fallback_best_source` | 0.40 | AI failed; fell back to longest/most-complete extractor version |
| `llm_near_agree_fallback_best_source` | 0.40 | AI failed; fell back to longest/most-complete extractor version |
| `zone_fallback_best_source` | 0.40 | Zone resolution failed entirely; deterministic fallback |

The weight column shows each method's contribution to the `quality_score` calculation. Direct resolutions (AI or median-source) contribute full weight (1.0). Rescue resolutions contribute 0.95 (slightly lower since they needed a retry). Deterministic two-source picks contribute 0.85. Degraded fallbacks (where AI failed entirely) contribute only 0.40. Agreed blocks (AGREE_EXACT, AGREE_NEAR) that needed no conflict resolution contribute 1.0 each.

**Interpreting the numbers:**
- **conflict_ratio < 0.20**: Excellent — extractors mostly agree. High confidence in output.
- **conflict_ratio 0.20-0.40**: Normal — some complex content needed AI resolution. Still reliable.
- **conflict_ratio > 0.40**: High — may trigger full-document merge. The paper likely has complex layout, many tables, or unusual formatting.
- **alignment_confidence > 0.70**: Good structural agreement between extractors.
- **alignment_confidence < 0.50**: Extractors produced very different structures — the merge may be less reliable.

---

## Cost Tracking

Each extraction run tracks AI usage and estimated cost:

| Field | Description |
|-------|-------------|
| `llm_cost_usd` | Total estimated cost in US dollars for this extraction |
| `llm_usage_json` | Detailed breakdown by call type and model, including token counts |

The cost breakdown shows token usage per call type (zone_resolution, header_hierarchy, full_merge, rescue, conflict_batch) and per model (gpt-5-mini, gpt-5.2), so you can see exactly where the AI budget was spent.

**Typical costs per paper:** $0.02-$0.20 depending on paper length and complexity.

---

## Using the Service

### Web Interface

A browser-based interface is available at `http://localhost:5000`. Upload a PDF, select extraction methods, and view or download results. The web UI runs extraction synchronously and is best for manual testing and small-scale use.

### REST API

All extraction jobs run **asynchronously** — you submit a PDF, get a job ID, and poll for results. This is the recommended interface for batch processing.

#### Submit an Extraction

```bash
curl -X POST http://localhost:5000/api/v1/extract \
  -F "file=@paper.pdf" \
  -F "methods=grobid,docling,marker" \
  -F "merge=true"
```

Returns a `process_id` that you use to check status and download results.

Optional fields:
- `reference_curie` — Link to an Alliance reference identifier (e.g., `AGRKB:101000000000001`)
- `mod_abbreviation` — MOD abbreviation (e.g., `WB`, `FB`, `SGD`)

#### Poll for Status

```bash
curl http://localhost:5000/api/v1/extract/{process_id}
```

**Status progression:** `pending` → `started` → `progress` → `complete`

When complete, the response includes extraction previews, consensus metrics, download paths, and cost information.

#### Download Results

```bash
# Download the merged output
curl -O http://localhost:5000/api/v1/extract/{process_id}/download/merged

# Download individual extractor outputs
curl -O http://localhost:5000/api/v1/extract/{process_id}/download/grobid
curl -O http://localhost:5000/api/v1/extract/{process_id}/download/docling
curl -O http://localhost:5000/api/v1/extract/{process_id}/download/marker
```

#### Get Durable S3 URLs

For long-term access, use the artifacts endpoint to get pre-signed S3 URLs:

```bash
curl http://localhost:5000/api/v1/extract/{process_id}/artifacts/urls
```

### API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Service health check (GROBID, Redis, Celery status) |
| `/api/v1/extract` | POST | Submit a PDF for extraction |
| `/api/v1/extract/{id}` | GET | Poll extraction status and results |
| `/api/v1/extract/{id}/download/{method}` | GET | Download output (grobid, docling, marker, merged) |
| `/api/v1/extract/{id}/images` | GET | List extracted images |
| `/api/v1/extract/{id}/images/{file}` | GET | Download a specific extracted image |
| `/api/v1/extract/{id}/logs` | GET | Get pre-signed URL for the NDJSON audit log |
| `/api/v1/extract/{id}/artifacts` | GET | Get artifact metadata |
| `/api/v1/extract/{id}/artifacts/urls` | GET | Get pre-signed S3 URLs for all artifacts |
| `/api/v1/extractions` | GET | List extraction runs (filterable by status, MOD, reference) |

Full interactive API documentation (Swagger UI) is available at `/docs` when the service is running.

---

## Deployment

### Docker Quick Start

```bash
# 1. Clone
git clone https://github.com/alliance-genome/agr_pdf_extraction_service.git
cd agr_pdf_extraction_service

# 2. Configure
cp .env.example .env
# Edit .env — set at minimum: OPENAI_API_KEY

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
┌─────────────────────────────────────────────────────┐
│                  Docker Compose                      │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌────────┐          │
│  │ Flask +   │   │ Celery   │   │ GROBID │          │
│  │ Gunicorn  │──▶│ Worker   │──▶│ (CRF)  │          │
│  │ (Web+API) │   │ (Jobs)   │   └────────┘          │
│  └──────────┘   └──────────┘                        │
│       │              │          ┌────────┐          │
│       │              │          │Postgres│          │
│       │              └─────────▶│ (DB)   │          │
│       └────────────────────────▶│        │          │
│                                 └────────┘          │
│                                 ┌────────┐          │
│                                 │ Redis  │          │
│                                 │ (Queue)│          │
│                                 └────────┘          │
└─────────────────────────────────────────────────────┘
```

| Container | Purpose |
|-----------|---------|
| `pdfx-app` | Flask + Gunicorn — serves the web UI and REST API |
| `pdfx-worker` | Celery worker — runs PDF extraction jobs in the background |
| `pdfx-grobid` | GROBID 0.8.2 CRF — scientific PDF structure extraction |
| `pdfx-postgres` | PostgreSQL — durable extraction run tracking |
| `pdfx-redis` | Redis 7 — Celery job queue and result backend |

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

**Prerequisites for durable storage:** AWS credentials must be configured in `.env` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`). Without them, extraction still works but outputs are only in local cache.

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
| `DOCLING_DEVICE` | `cpu` | Docling device (`cpu` or `auto` for GPU) |
| `MARKER_DEVICE` | `cpu` | Marker device (`cpu` or `auto` for GPU) |
| `MARKER_EXTRACT_IMAGES` | `true` | Extract images from PDFs |

### AI Model Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-5.2` | Base model (used for full-document merge) |
| `LLM_MODEL_ZONE_RESOLUTION` | `gpt-5-mini` | Model for zone-based conflict resolution |
| `LLM_MODEL_FULL_MERGE` | `gpt-5.2` | Model for full-document merge fallback |
| `LLM_MODEL_RESCUE` | `gpt-5-mini` | Model for rescue resolution attempts |
| `LLM_MODEL_CONFLICT_BATCH` | `gpt-5.2` | Model for batched conflict resolution |
| `HIERARCHY_LLM_MODEL` | `gpt-5.2` | Model for heading hierarchy resolution |
| `HIERARCHY_LLM_REASONING` | `medium` | Reasoning effort for heading hierarchy calls |
| `LLM_REASONING_EFFORT` | `medium` | Default reasoning effort for LLM calls |
| `ZONE_ESCALATION_THRESHOLD` | `20000` | Token threshold — zones above this use the escalation model |
| `ZONE_ESCALATION_MODEL` | `gpt-5.2` | Model used when a zone exceeds the escalation threshold |
| `LLM_CONFLICT_BATCH_SIZE` | `10` | Number of conflicts per batch in batched resolution |
| `LLM_CONFLICT_MAX_WORKERS` | `4` | Max parallel workers for batched conflict resolution |
| `LLM_CONFLICT_RETRY_ROUNDS` | `2` | Number of retry rounds for unresolved conflicts |

### Consensus Pipeline

| Variable | Default | Description |
|----------|---------|-------------|
| `CONSENSUS_ENABLED` | `true` | Enable the consensus merge pipeline |
| `CONSENSUS_NEAR_THRESHOLD` | `0.92` | Token similarity threshold for AGREE_NEAR classification |
| `CONSENSUS_LEVENSHTEIN_THRESHOLD` | `0.90` | Character-level similarity threshold for AGREE_NEAR |
| `CONSENSUS_CONFLICT_RATIO_FALLBACK` | `0.4` | Overall conflict ratio that triggers full-LLM fallback |
| `CONSENSUS_CONFLICT_RATIO_TEXTUAL_FALLBACK` | `0.4` | Text-block conflict ratio for fallback |
| `CONSENSUS_CONFLICT_RATIO_STRUCTURED_FALLBACK` | `0.85` | Table/equation conflict ratio for fallback |
| `CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK` | `0.5` | Alignment confidence below this triggers fallback |
| `CONSENSUS_LAYERED_ENABLED` | `true` | Enable layered conflict resolver (median-source + LLM) |
| `CONSENSUS_LAYERED_MEDIUM_SIM_THRESHOLD` | `0.60` | Minimum pairwise similarity for median-source selection |
| `CONSENSUS_ALWAYS_ESCALATE_TABLES` | `true` | Always send tables/equations to AI, even if extractors agree |
| `CONSENSUS_LOCALIZED_CONFLICT_SPAN_MAX` | `0.35` | Max document span ratio for localized conflict relief |
| `CONSENSUS_LOCALIZED_CONFLICT_RELIEF` | `0.15` | How much to relax conflict ratio when conflicts are localized |
| `CONSENSUS_LOCALIZED_CONFLICT_MAX_BLOCKS` | `25` | Max conflict blocks for localized relief to apply |
| `CONSENSUS_ZONE_FLANKING_COUNT` | `2` | Number of context blocks on each side of a conflict zone |
| `CONSENSUS_HIERARCHY_ENABLED` | `true` | Enable heading hierarchy resolution step |
| `CONSENSUS_FAIL_ON_GLOBAL_DUPLICATES` | `true` | Flag global duplicates in QA gate |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://pdfx:pdfx@localhost:5432/pdfx` | PostgreSQL connection string |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Redis broker URL |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/1` | Redis result backend |
| `CACHE_FOLDER` | `./extraction_cache` | Local cache directory |
| `AUDIT_S3_BUCKET` | `agr-pdf-extraction-benchmark` | S3 bucket for durable artifact storage |
| `EXTRACTION_CONFIG_VERSION` | `4` | Bump to invalidate cached outputs |

---

## Project Structure

```
agr_pdf_extraction_service/
├── config.py                    # Central configuration
├── celery_app.py                # Celery app + extract_pdf background task
├── run.py                       # Dev server entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── api.py                   # REST API v1 blueprint
│   ├── openapi.yaml             # OpenAPI 3.0 specification
│   ├── server.py                # Web UI routes
│   ├── utils.py                 # File hashing, cache paths
│   ├── models.py                # SQLAlchemy models (ExtractionRun)
│   ├── services/
│   │   ├── pdf_extractor.py     # Abstract base class for extractors
│   │   ├── grobid_service.py    # GROBID extractor
│   │   ├── docling_service.py   # Docling extractor
│   │   ├── marker_service.py    # Marker extractor
│   │   ├── llm_service.py       # LLM service (model selection, cost tracking)
│   │   ├── consensus_service.py # Consensus merge pipeline
│   │   └── degradation_metrics.py # Quality scoring
│   └── templates/
│       └── index.html           # Web UI
├── tests/                       # Test suite
└── deploy/
    ├── Dockerfile               # Container image
    ├── docker-compose.yml       # Service stack
    ├── deploy.sh                # One-command deployment
    └── manage.sh                # Management commands
```

## Deployment Target

Currently deployed on **AWS EC2** (g5.2xlarge):
- NVIDIA A10G GPU, 8 vCPU, 32 GB RAM
- Docker + Docker Compose
- GPU-accelerated Docling and Marker inference

## License

MIT License
