# AGR PDF Extraction Service

Production service for extracting structured content from scientific PDFs using multiple extraction engines. Built for the [Alliance of Genome Resources](https://www.alliancegenome.org/).

Runs three extractors in parallel — **GROBID**, **Docling**, and **Marker** — and merges their outputs using a selective consensus pipeline backed by an LLM (GPT 5.2). The consensus pipeline identifies agreement across extractors programmatically and only sends disagreements to the LLM, saving 50-70% of token usage. Designed for CPU-first deployment on on-premise servers via Docker Compose.

## Architecture

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
│       │              └─────────▶│ Redis  │          │
│       └────────────────────────▶│ (Queue)│          │
│                                 └────────┘          │
└─────────────────────────────────────────────────────┘
```

| Service | Container | Purpose |
|---------|-----------|---------|
| **app** | `pdfx-app` | Flask + Gunicorn — serves the web UI and REST API |
| **worker** | `pdfx-worker` | Celery worker — runs PDF extraction jobs in the background |
| **grobid** | `pdfx-grobid` | GROBID 0.8.2 CRF — scientific PDF structure extraction via HTTP |
| **redis** | `pdfx-redis` | Redis 7 — Celery job queue and result backend |

## Quick Start (Docker)

```bash
# 1. Clone the repository
git clone https://github.com/alliance-genome/agr_pdf_extraction_service.git
cd agr_pdf_extraction_service

# 2. Configure environment
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY for LLM merge

# 3. Deploy
cd deploy
./deploy.sh
```

The deploy script creates data directories, builds containers, starts all services, and runs health checks. On first run, Docling and Marker will download ML models (~2-5 GB) which takes several minutes.

**Endpoints after deployment:**

| Endpoint | URL |
|----------|-----|
| Web UI | http://localhost:5000 |
| REST API | http://localhost:5000/api/v1/ |
| Health check | http://localhost:5000/api/v1/health |

## REST API

All extraction jobs run asynchronously via Celery. Submit a PDF, get a `job_id`, poll for results.

### `POST /api/v1/extract`

Submit a PDF for extraction. Returns `202 Accepted` with a job ID.

**Request** (multipart/form-data):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | *(required)* | PDF file to extract |
| `methods` | string | `grobid,docling,marker` | Comma-separated extractors to run |
| `merge` | string | `false` | Set to `true` to merge outputs with LLM |

```bash
# Extract with all methods
curl -X POST http://localhost:5000/api/v1/extract \
  -F "file=@paper.pdf" \
  -F "methods=grobid,docling,marker" \
  -F "merge=true"
```

**Response** (`202`):
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "queued",
  "methods": ["grobid", "docling", "marker"],
  "merge": true
}
```

### `GET /api/v1/extract/{job_id}`

Poll job status. Returns results when complete.

**Status progression**: `pending` → `started` → `progress` → `complete`

```bash
curl http://localhost:5000/api/v1/extract/a1b2c3d4-...
```

**Response when complete** (`200`):
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "complete",
  "result": {
    "status": "success",
    "file_hash": "abc123...",
    "methods_used": ["grobid", "docling", "marker"],
    "cached_methods": [],
    "extractions": {
      "grobid": "First 500 chars of extracted text...",
      "docling": "...",
      "marker": "..."
    },
    "merged_output": "First 1000 chars of merged text...",
    "download_paths": {
      "grobid": "/path/to/v3_abc123_grobid.md",
      "docling": "/path/to/v3_abc123_docling.md",
      "marker": "/path/to/v3_abc123_marker.md"
    },
    "consensus_metrics": {
      "total_blocks": 42,
      "agree_exact": 20,
      "agree_near": 10,
      "gap": 5,
      "conflict": 7,
      "conflict_ratio": 0.19,
      "alignment_confidence": 0.85,
      "fallback_triggered": false
    }
  }
}
```

### `GET /api/v1/extract/{job_id}/download/{method}`

Download the full markdown output for a completed job.

| Method | Description |
|--------|-------------|
| `grobid` | GROBID extraction output |
| `docling` | Docling extraction output |
| `marker` | Marker extraction output |
| `merged` | LLM-merged output (if merge was requested) |

```bash
curl -O http://localhost:5000/api/v1/extract/a1b2c3d4-.../download/grobid
```

### `GET /api/v1/extractions`

List all extraction runs with optional filtering and pagination.

**Query parameters:**

| Parameter | Description |
|-----------|-------------|
| `status` | Filter by status (`queued`, `running`, `succeeded`, `failed`) |
| `reference_curie` | Filter by reference curie |
| `mod_abbreviation` | Filter by MOD abbreviation |
| `limit` | Max results (default 50, max 200) |
| `offset` | Skip first N results (default 0) |

```bash
# List recent extractions
curl http://localhost:5000/api/v1/extractions

# Filter by status
curl http://localhost:5000/api/v1/extractions?status=succeeded&limit=10
```

### `GET /api/v1/extract/{process_id}/logs`

Return a pre-signed URL to the NDJSON run log stored in S3.

### `GET /api/v1/extract/{process_id}/artifacts`

Return the artifact keys recorded for a run (markdown outputs, source PDF copy, extracted image copies).

### `GET /api/v1/extract/{process_id}/artifacts/urls`

Return pre-signed S3 URLs for every artifact key stored on the run, including nested keys such as image artifacts.

### `GET /api/v1/health`

Service health check. Returns component status for GROBID, Redis, and Celery workers.

```bash
curl http://localhost:5000/api/v1/health
```

**Response** (`200` or `503`):
```json
{
  "status": "ok",
  "checks": {
    "service": "ok",
    "grobid": "ok",
    "redis": "ok",
    "workers": 1
  }
}
```

## Web UI

A browser-based interface is available at `http://localhost:5000`. Upload a PDF, select extraction methods, and view/download results. The web UI uses synchronous extraction (blocks until complete) and is intended for manual testing and small-scale use.

## Standalone Deployment (Temporary — Pre-ABC Integration)

The service can run standalone on FlySQL servers without ABC Literature System integration. In this mode, all extraction outputs are stored durably in S3 via the audit trail, and tracked in the Postgres `extraction_run` table.

**Recommended workflow:**

```bash
# 1. Submit a PDF for extraction
curl -X POST http://localhost:5000/api/v1/extract \
  -F "file=@paper.pdf" -F "methods=grobid,docling,marker" -F "merge=true"
# → Returns process_id

# 2. Poll until complete
curl http://localhost:5000/api/v1/extract/{process_id}

# 3. Get pre-signed S3 URLs for all outputs (durable)
curl http://localhost:5000/api/v1/extract/{process_id}/artifacts/urls

# 4. Browse past extractions
curl http://localhost:5000/api/v1/extractions
```

### Endpoint Durability

| Endpoint | Durable? | Backend |
|----------|----------|---------|
| `GET /extract/{id}` | Yes | Postgres `extraction_run` table |
| `GET /extract/{id}/logs` | Yes | S3 pre-signed URL |
| `GET /extract/{id}/artifacts` | Yes | Postgres `artifacts_json` column |
| `GET /extract/{id}/artifacts/urls` | Yes | S3 pre-signed URLs |
| `GET /extractions` | Yes | Postgres query |
| `GET /extract/{id}/download/{method}` | Yes* | Local cache, S3 fallback |
| `GET /extract/{id}/images/{file}` | Yes* | Local cache, S3 fallback |

*Download and image endpoints try local cache first for speed, then fall back to S3 artifacts if the cache has been cleared.

**Prerequisites for durable storage:**
- Postgres must be running (included in docker-compose)
- AWS credentials must be configured in `.env` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- Without AWS credentials, extraction still works but outputs are only in local cache (not durable)

## Configuration

All settings live in `config.py` with sensible defaults. Override via environment variables or `.env` file. See `.env.example` for the full list.

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required for LLM merge) |

### Extractors

| Variable | Default | Description |
|----------|---------|-------------|
| `GROBID_URL` | `http://localhost:8070` | GROBID server URL (set by Docker Compose) |
| `GROBID_REQUEST_TIMEOUT` | `120` | GROBID request timeout in seconds |
| `GROBID_INCLUDE_COORDINATES` | `false` | Include bounding box coordinates |
| `GROBID_INCLUDE_RAW_CITATIONS` | `false` | Include raw citation strings |
| `DOCLING_DEVICE` | `cpu` | Docling device (`cpu` or `auto`) |
| `MARKER_DEVICE` | `cpu` | Marker device (`cpu` or `auto`) |
| `MARKER_EXTRACT_IMAGES` | `true` | Extract images from PDFs |

### LLM Merge

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-5.2` | OpenAI model for merging |
| `LLM_MAX_TOKENS` | `16000` | Max tokens for merge output |

### Consensus Pipeline

| Variable | Default | Description |
|----------|---------|-------------|
| `CONSENSUS_ENABLED` | `true` | Enable selective consensus merge pipeline |
| `CONSENSUS_NEAR_THRESHOLD` | `0.92` | Token similarity threshold for AGREE_NEAR |
| `CONSENSUS_LEVENSHTEIN_THRESHOLD` | `0.90` | Levenshtein similarity threshold for AGREE_NEAR |
| `CONSENSUS_CONFLICT_RATIO_FALLBACK` | `0.4` | Conflict ratio above this triggers full-LLM fallback |
| `CONSENSUS_ALIGNMENT_CONFIDENCE_FALLBACK` | `0.5` | Alignment confidence below this triggers fallback |
| `CONSENSUS_ALWAYS_ESCALATE_TABLES` | `true` | Always send tables/equations to LLM |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Redis broker URL |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/1` | Redis result backend |
| `CACHE_FOLDER` | `./extraction_cache` | Cache directory for outputs |
| `UPLOAD_FOLDER` | `./uploaded_pdfs` | Upload directory |
| `MAX_CONTENT_LENGTH` | `104857600` | Max upload size in bytes (100MB) |
| `EXTRACTION_CONFIG_VERSION` | `3` | Bump to invalidate cached outputs |

## Management

Use `deploy/manage.sh` for day-to-day operations:

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

## Local Development (without Docker)

```bash
# 1. Prerequisites: Python 3.11+, Redis running locally, GROBID running on port 8070

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 4. Start Redis (if not already running)
redis-server &

# 5. Start Celery worker (in a separate terminal)
celery -A celery_app worker --loglevel=info

# 6. Start Flask dev server
python run.py
```

## Running Tests

```bash
python3 -m pytest
```

## Project Structure

```
agr_pdf_extraction_service/
├── config.py                   # Central configuration (all settings)
├── celery_app.py               # Celery app + extract_pdf background task
├── run.py                      # Dev server entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── app/
│   ├── __init__.py             # Flask app factory
│   ├── api.py                  # REST API v1 blueprint
│   ├── server.py               # Web UI routes
│   ├── utils.py                # File hashing, cache paths
│   ├── services/
│   │   ├── pdf_extractor.py    # Abstract base class
│   │   ├── grobid_service.py   # GROBID extractor
│   │   ├── docling_service.py  # Docling extractor
│   │   ├── marker_service.py   # Marker extractor
│   │   ├── llm_service.py       # LLM merge service
│   │   └── consensus_service.py # Selective consensus merge pipeline
│   └── templates/
│       └── index.html          # Web UI
├── tests/
│   ├── test_server.py
│   ├── test_utils.py
│   └── services/               # Per-service tests
└── deploy/
    ├── Dockerfile              # CPU-only, python:3.11-slim
    ├── docker-compose.yml      # 4-service stack
    ├── deploy.sh               # One-command deployment
    └── manage.sh               # Management commands
```

## Deployment Target

Designed for **FlySQL servers** (on-premise, Alliance of Genome Resources):
- 48 CPUs, 256GB RAM, no GPU
- Docker + Docker Compose
- VPN connectivity to Alliance network

## License

MIT License
