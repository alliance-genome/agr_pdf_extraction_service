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

Before parsing, each output goes through **source-level normalization** — removing extractor-specific artifacts like HTML tags, stray formatting, and inconsistent whitespace so that comparisons are fair.

### Step 2: Align — Match Blocks Across Extractors

The pipeline needs to figure out which block from GROBID corresponds to which block from Docling and Marker. This is not always obvious — extractors may split paragraphs differently, skip sections, or order content slightly differently.

The alignment uses the **Hungarian algorithm** (a well-known optimization method) to find the best one-to-one matching across extractors. It scores each potential match on three factors:

| Factor | Weight | What It Measures |
|--------|--------|-----------------|
| **Heading match** | 40% | Do both blocks have the same section heading nearby? |
| **Content similarity** | 40% | How similar is the actual text? (measured with fuzzy string matching) |
| **Position proximity** | 20% | Are they in roughly the same location in the document? |

The result is a list of **aligned triples** — groups of up to three blocks (one from each extractor) that represent the same piece of content. The alignment also computes a **confidence score** (0.0 to 1.0) reflecting how well the blocks matched overall.

**Sparse extractor exclusion:** If one extractor produced dramatically fewer blocks than the others (less than 30% of the maximum), it is excluded from alignment. This prevents a partially-failed extraction from poisoning the merge.

### Step 3: Classify — Determine Agreement or Disagreement

Each aligned triple is classified into one of four categories:

| Classification | What It Means | What Happens |
|---------------|---------------|--------------|
| **AGREE_EXACT** | Two or more extractors produced identical text (after normalization) | Text is accepted as-is — no AI needed |
| **AGREE_NEAR** | Text is very similar (>92% token overlap AND >90% character-level match) with no differences in numbers or citations | Best version is selected automatically — no AI needed |
| **GAP** | A block appears in only one extractor (the others have nothing at that position) | Usually kept as-is, since one extractor found content the others missed |
| **CONFLICT** | The extractors disagree and the differences are significant enough to matter | Sent to the conflict resolution pipeline (see next step) |

**Numeric and citation guardrails:** AGREE_NEAR has strict protections — if two blocks are textually similar but differ in any numbers (e.g., "p < 0.05" vs "p < 0.5") or citation references (e.g., "[1,2]" vs "[1,3]"), the block is escalated to CONFLICT. This prevents silently accepting corrupted data values.

**Table and equation escalation:** Tables and equations are optionally always escalated to CONFLICT regardless of similarity, since even minor formatting differences in these structured elements can change their meaning.

### Step 4: Guard Gates — Decide How to Resolve Conflicts

Before resolving conflicts, the pipeline evaluates the overall difficulty of the document through several **guard gates**. These decide whether conflicts can be resolved zone-by-zone (efficient) or whether the entire document needs to be sent to the AI (expensive but thorough).

| Guard Gate | Threshold | What Triggers It |
|------------|-----------|-----------------|
| **Textual conflict ratio** | > 40% of text blocks are CONFLICT | Too many disagreements for zone-based resolution to be reliable |
| **Structured conflict ratio** | > 85% of tables/equations are CONFLICT | Extractors fundamentally disagree on structured content |
| **Alignment confidence** | < 0.50 | The block-matching step had very low confidence, suggesting the extractors produced very different document structures |

If any guard gate triggers, the pipeline falls back to sending the **entire document** to the AI for a full-document merge. This is the safety net — it costs more but ensures quality.

**Localized conflict relief:** If conflicts are clustered in one part of the document (e.g., a difficult table section) rather than spread throughout, the thresholds are relaxed. This prevents a single tricky section from forcing a full-document merge when the rest of the paper merged cleanly.

### Step 5: Resolve Conflicts — Four Layers of Resolution

Conflicts that pass the guard gates are resolved through a **layered approach**, starting with the cheapest method and escalating only when needed:

#### Layer 1: Median-Source Selection

For blocks where all three extractors have output, the pipeline picks the text that is most similar to the other two — the "median" source. If the median source has at least 60% similarity to the others, it is accepted without any AI call.

*Why this works:* If two extractors say roughly the same thing and one is an outlier, the consensus of two is usually correct.

#### Layer 2: Zone-Based LLM Resolution

Remaining conflicts are grouped into **conflict zones** — contiguous stretches of the document where disagreements occur, along with a few surrounding "context" blocks so the AI understands what comes before and after.

Each zone is sent to the AI language model, which sees all extractor versions side by side and selects the best text for each conflicting segment. The AI is instructed to:
- Choose the most complete and accurate version
- Never fabricate content that doesn't appear in any extractor
- Preserve all data values, citations, and scientific terminology exactly

#### Layer 3: Rescue Resolution (Three Tiers)

If any segments remain unresolved after zone-based resolution, the pipeline attempts **rescue resolution** with increasingly aggressive strategies:

| Rescue Tier | Strategy |
|-------------|----------|
| **Tier 1** | Re-send the segment with expanded context (more surrounding blocks) |
| **Tier 2** | Re-send with even more context and a stronger prompt emphasizing the specific failure |
| **Tier 3** | Last-resort containment check — if one extractor's text contains the others, pick the longest version |

#### Layer 4: Full-Document LLM Fallback

If guard gates triggered earlier, or if rescue still fails for critical content, the entire document is sent to the strongest AI model (GPT-5.2) for a complete merge. This is expensive but guarantees a result.

### Step 6: Assemble and Clean — Build the Final Document

The resolved blocks are reassembled into a single markdown document in the correct order. The output goes through several cleaning passes:

- **Deduplication** — Removes blocks that appear more than once (common when extractors overlap at section boundaries)
- **HTML cleanup** — Strips leftover HTML tags, comments, and span elements
- **Link cleanup** — Converts web links to plain text (keeping the visible text, removing URLs)
- **Whitespace normalization** — Ensures consistent spacing between sections
- **Image reference cleanup** — Removes broken image references that don't correspond to actual extracted images

### Step 7: Heading Hierarchy — Fix Section Structure

PDF extractors often produce "flat" headings — every heading at the same level (e.g., all `##`) regardless of the actual document structure. The final step uses the AI to restore the proper heading hierarchy.

The AI receives a list of all headings in order and assigns appropriate levels (H1 through H6) based on the logical structure of the paper. For example:

```
Before:  ## Abstract, ## Introduction, ## Methods, ## Cell Culture, ## Results
After:   ## Abstract, ## Introduction, ## Methods, ### Cell Culture, ## Results
```

This step has strict validation:
- The AI cannot change heading text — only the level
- The title must be H1 (or detected in the opening text)
- No heading can jump more than one level (e.g., H2 directly to H4 is rejected)
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
| `HIERARCHY_LLM_MODEL` | `gpt-5.2` | Model for heading hierarchy resolution |
| `ZONE_ESCALATION_THRESHOLD` | `20000` | Token threshold — zones above this use the escalation model |
| `ZONE_ESCALATION_MODEL` | `gpt-5.2` | Model used when a zone exceeds the escalation threshold |

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
| `CONSENSUS_ALWAYS_ESCALATE_TABLES` | `true` | Always send tables/equations to AI, even if extractors agree |

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

Designed for **FlySQL servers** (on-premise, Alliance of Genome Resources):
- 48 CPUs, 256GB RAM, no GPU
- Docker + Docker Compose
- VPN connectivity to Alliance network

GPU deployment (for faster Docling/Marker inference) is supported via `deploy/Dockerfile.gpu` and `deploy/docker-compose.gpu.yml`.

## License

MIT License
