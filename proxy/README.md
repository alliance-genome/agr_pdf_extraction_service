# PDFX Proxy

A lightweight FastAPI proxy that sits in front of the GPU-based PDF Extraction Service. It handles authentication, auto-starts the EC2 GPU instance on demand, queues jobs during startup, and forwards requests once the backend is ready. Designed to run on AWS Fargate so the expensive GPU instance can be stopped when idle.

## Why a Proxy?

The PDF extraction backend runs on a GPU instance (g5.2xlarge) that costs ~$1/hour. Leaving it running 24/7 is wasteful when jobs arrive intermittently. The proxy solves this by:

1. Running cheaply on Fargate (256 CPU / 512 MB — pennies/hour)
2. Auto-starting the GPU instance when a job arrives
3. Queuing jobs while EC2 boots (~2-3 minutes), with optional durable S3 queue
4. Replaying queued jobs once the backend is healthy
5. Auto-stopping the GPU instance after an idle timeout

Callers talk to the proxy at a stable endpoint and never need to know whether the GPU instance is running.

## Architecture

```
                     ┌──────────────────────┐
                     │   Cognito (Auth)      │
                     └──────────┬───────────┘
                                │ JWT
                     ┌──────────▼───────────┐
   Client ──────────>│   PDFX Proxy         │
   (Curation UI)     │   (Fargate)          │
                     │                      │
                     │  - Auth validation    │
                     │  - EC2 lifecycle      │
                     │  - Job queue          │
                     │  - Request forwarding │
                     └──────────┬───────────┘
                                │ HTTP (private IP)
                     ┌──────────▼───────────┐
                     │   GPU Backend         │
                     │   (EC2 g5.2xlarge)    │
                     │                      │
                     │  - GROBID / Docling   │
                     │  - Marker / LLM Merge │
                     │  - PostgreSQL / Redis │
                     └──────────────────────┘
```

## API Endpoints

All endpoints except `/api/v1/health`, `/api/v1/health/deep`, and `/api/v1/metrics` require a Cognito Bearer token.

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/v1/health` | GET | No | Proxy health + EC2 state (degraded unless worker is ready) |
| `/api/v1/health/deep` | GET | No | Deep probe: proxy auth validation + downstream status round-trip |
| `/api/v1/metrics` | GET | No | Queue/replay/lifecycle/canary metrics for alerting |
| `/api/v1/status` | GET | Yes | EC2 state, idle time, active/queued job counts |
| `/api/v1/wake` | POST | Yes | Start the GPU instance (idempotent) |
| `/api/v1/extract` | POST | Yes | Submit a PDF for extraction |
| `/api/v1/extract/{id}` | GET | Yes | Poll job status with granular progress |
| `/api/v1/extract/{id}/cancel` | POST | Yes | Request cancellation of a queued/running extraction job |
| `/api/v1/extract/{id}/download/{method}` | GET | Yes | Download backend extraction output |

### Submit Extraction (`POST /api/v1/extract`)

Accepts the same multipart form fields as the backend (`file`, `methods`, `merge`, `clear_cache`, `clear_cache_scope`, `reference_curie`, `mod_abbreviation`).

**If EC2 is running:** the request is forwarded immediately and the backend's 202 response is returned.

**If EC2 is stopped:** the job is queued, EC2 is started, and a 202 response is returned:

```json
{
  "process_id": "...",
  "status": "queued",
  "state": "stopped",
  "message": "EC2 is starting. Job queued. Poll GET /api/v1/extract/{process_id} for status.",
  "retry_after": 30,
  "progress": {
    "stage": "ec2_starting",
    "stage_display": "Spinning up GPU instance",
    "stages_completed": [],
    "stages_pending": [],
    "stages_total": 0,
    "stages_done": 0,
    "percent": 0
  }
}
```

Once EC2 is healthy, all queued jobs are automatically replayed to the backend. Replay failures are marked as explicit `failed` states (no infinite pending loop).

### Poll Status (`GET /api/v1/extract/{id}`)

Returns granular progress through the extraction pipeline:

**While EC2 is starting (job queued locally):**
```json
{
  "process_id": "...",
  "status": "queued",
  "progress": {
    "stage": "ec2_starting",
    "stage_display": "Spinning up GPU instance",
    "percent": 0
  }
}
```

**Once the backend is processing:**

The proxy forwards the request to EC2 and returns the backend's response verbatim, which includes stage-by-stage progress:

```json
{
  "process_id": "...",
  "status": "progress",
  "progress": {
    "stage": "docling",
    "stage_display": "Running DOCLING extraction",
    "stages_completed": ["initializing", "grobid"],
    "stages_done": 2,
    "stages_total": 6,
    "stages_pending": ["marker", "llm_merge", "finalizing"],
    "percent": 33
  }
}
```

**Canonical stages** (in pipeline order):

| Stage | Display Text | Source |
|-------|-------------|--------|
| `ec2_starting` | Spinning up GPU instance | Proxy |
| `initializing` | Initializing extraction job | Backend |
| `grobid` | Running GROBID extraction | Backend (conditional) |
| `docling` | Running Docling extraction | Backend (conditional) |
| `marker` | Running Marker extraction | Backend (conditional) |
| `llm_merge` | Merging extraction outputs with LLM | Backend (conditional) |
| `finalizing` | Uploading artifacts and finalizing | Backend |

Extraction stages are dynamic based on the `methods` parameter. `llm_merge` only appears when `merge=true`.

## EC2 Lifecycle State Machine

```
  STOPPED ──(job arrives)──> STARTING ──(health check OK)──> READY
     ^                                                         │
     │                                                    (job running)
     │                                                         │
     └──(idle timeout)──── READY <──(job done)──── BUSY ◄──────┘
```

| State | Meaning |
|-------|---------|
| `STOPPED` | EC2 is off. Jobs trigger a start. |
| `STARTING` | EC2 is booting. Jobs are queued in memory. |
| `READY` | EC2 is healthy. Requests are forwarded. |
| `BUSY` | At least one job is in flight. Idle timer is paused. |

The idle monitor checks every 60 seconds. The worker is only eligible for stop when all guards pass:
- no queued jobs
- no replay-inflight jobs
- no tracked active backend jobs
- minimum uptime (`MIN_UPTIME_MINUTES`) has elapsed
- `ALWAYS_ON_MODE` is disabled

When guards pass and idle exceeds `IDLE_TIMEOUT_MINUTES`, the instance is stopped automatically.

On proxy startup, `sync_state_from_ec2()` checks the actual EC2 state so the proxy's internal state matches reality.

## Configuration

All settings come from environment variables. In production, values are injected from AWS SSM Parameter Store via the ECS task definition's `secrets` block.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `EC2_INSTANCE_ID` | Yes | — | The managed GPU instance ID |
| `COGNITO_USER_POOL_ID` | Yes | — | Cognito user pool for JWT validation |
| `EC2_REGION` | No | `us-east-1` | AWS region for EC2 API calls |
| `EC2_PORT` | No | `5000` | Port the backend listens on |
| `COGNITO_REGION` | No | `us-east-1` | AWS region for Cognito |
| `COGNITO_REQUIRED_SCOPE` | No | `pdfx-api/extract` | OAuth scope accepted in the JWT |
| `COGNITO_ACCEPTED_SCOPES` | No | — | Comma-separated additional scopes that also grant access |
| `COGNITO_ACCEPTED_CLIENT_IDS` | No | — | Comma-separated Cognito app client_ids accepted without requiring a PDFX-specific scope (e.g. the CurationAPI-Admin M2M client) |
| `IDLE_TIMEOUT_MINUTES` | No | `30` | Minutes of inactivity before EC2 is stopped |
| `MIN_UPTIME_MINUTES` | No | `20` | Minimum uptime after wake before idle stop is allowed |
| `STARTUP_TIMEOUT_MINUTES` | No | `10` | Max minutes to wait for EC2 health check |
| `HEALTH_POLL_INTERVAL_SECONDS` | No | `15` | Seconds between EC2 health polls during startup |
| `MAX_QUEUED_JOBS` | No | `10` | Max jobs to hold in memory during startup |
| `FORWARD_TIMEOUT_SECONDS` | No | `600` | Timeout for forwarded HTTP requests to EC2 |
| `ALWAYS_ON_MODE` | No | `false` | Emergency mode that disables idle auto-stop |
| `QUEUE_BACKEND` | No | `memory` | Queue backend: `memory` or `s3` |
| `QUEUE_S3_BUCKET` | No | — | S3 bucket for durable queue (`QUEUE_BACKEND=s3`) |
| `QUEUE_S3_PREFIX` | No | `pdfx-proxy-queue` | S3 prefix for durable queue jobs |
| `QUEUE_S3_REGION` | No | — | Optional S3 region override |
| `STUCK_PENDING_MINUTES` | No | `20` | Age threshold for stale pending/running jobs |
| `RECONCILER_INTERVAL_SECONDS` | No | `60` | Background reconciler interval |
| `RECONCILER_REQUEUE_ONCE` | No | `false` | Optional one-time requeue of stale jobs |
| `HEALTHCHECK_BEARER_TOKEN` | No | — | Token used by `/api/v1/health/deep` |
| `CANARY_INTERVAL_SECONDS` | No | `0` | Downstream canary interval (0 disables) |
| `CANARY_BEARER_TOKEN` | No | — | Token used for canary downstream probe |

## Deployment

The proxy is deployed as an ECS Fargate service behind an ALB.

### Prerequisites

- ECR repository for the proxy image
- ECS cluster with Fargate capacity
- Cognito user pool with a resource server and `pdfx-api/extract` scope
- SSM parameters under `/pdfx/*` for configuration values
- IAM roles: execution role (ECR pull + SSM read + CloudWatch Logs) and task role (EC2 start/stop/describe + SSM read)

### Build and Deploy

```bash
# Run from repo root
cd proxy

# Build and push to ECR
docker build -t agr_pdfx_proxy .
docker tag agr_pdfx_proxy:latest <account>.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_proxy:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_proxy:latest

# Register the ECS task definition and roll ECS service (reads SSM parameters automatically)
cd deploy
./deploy.sh --profile <profile>

# Or dry-run to inspect the generated task definition
./deploy.sh --profile <profile> --dry-run

# Optional: register task definition only (skip ECS service update)
./deploy.sh --profile <profile> --no-update-service
```

### IAM Permissions (Task Role)

The task role needs:
- `ec2:StartInstances` / `ec2:StopInstances` — scoped to the managed instance
- `ec2:DescribeInstances` — for state polling
- `ssm:GetParameters` — scoped to `/pdfx/*`

See `deploy/iam-policy.template.json` for the full policy.

## Project Structure

```
proxy/
├── app/
│   ├── main.py            # FastAPI routes (health, status, wake, extract, poll, download)
│   ├── config.py           # Settings from environment variables
│   ├── auth.py             # Cognito JWT validation
│   ├── ec2_manager.py      # EC2 start/stop/describe via boto3
│   ├── state_machine.py    # InstanceState enum + LifecycleManager
│   └── job_queue.py        # Queue backends (in-memory + optional durable S3)
├── tests/
│   ├── test_main.py        # Integration tests for all routes
│   ├── test_auth.py        # Cognito token validation tests
│   ├── test_ec2_manager.py # EC2 manager tests (mocked boto3)
│   ├── test_job_queue.py   # Queue behavior tests
│   └── test_state_machine.py # State machine transition tests
├── deploy/
│   ├── deploy.sh           # SSM-aware ECS task definition registration
│   ├── task-definition.template.json
│   └── iam-policy.template.json
├── Dockerfile              # Python 3.11-slim + uvicorn
├── requirements.txt
└── .env.example
```

## Running Tests

```bash
cd proxy
pip install -r requirements.txt
pip install pytest
python -m pytest tests/ -v
```

All tests use mocked singletons (no real AWS calls or Cognito validation).

## Local Development

```bash
cd proxy
cp .env.example .env
# Fill in EC2_INSTANCE_ID and COGNITO_USER_POOL_ID

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

The proxy will sync with the actual EC2 instance state on startup.

## Accepting Shared M2M Admin Tokens

By default the proxy only accepts tokens carrying the `pdfx-api/extract` scope.
Backend services that already hold a shared admin token (for example the
CurationAPI-Admin M2M client used by AGR and the A-team) can be admitted
without being reissued a PDFX-specific token by setting either:

- `COGNITO_ACCEPTED_CLIENT_IDS` — comma-separated Cognito app client_ids
  whose tokens are accepted regardless of scope. This is the recommended
  mechanism for the CurationAPI-Admin client: drop the client's app client_id
  into this list per environment (dev / stage / prod).
- `COGNITO_ACCEPTED_SCOPES` — comma-separated additional scopes that also
  grant access. Useful if a shared client always issues tokens with a known
  admin scope.

JWT signature, issuer, and expiry are still verified for every request;
only the authorization check (scope vs. client_id) is relaxed.

### Provisioning per environment

These two settings are sourced from optional SSM parameters by `deploy.sh`:

| SSM parameter | Maps to env var |
|---------------|-----------------|
| `/pdfx/cognito-accepted-scopes` | `COGNITO_ACCEPTED_SCOPES` |
| `/pdfx/cognito-accepted-client-ids` | `COGNITO_ACCEPTED_CLIENT_IDS` |

If a parameter is absent, `deploy.sh` defaults the value to an empty string
(prints a notice and continues), so the first deployment after this feature
ships does not require any new SSM keys to be present. To enable shared M2M
access, create or update the SSM parameter and redeploy:

```bash
aws ssm put-parameter \
  --name /pdfx/cognito-accepted-client-ids \
  --type String \
  --overwrite \
  --value "<curation-admin-client-id>[,<other-client-id>...]"

cd proxy/deploy && ./deploy.sh --profile <profile>
```

## Operational Fallbacks

### Always-On Worker Window
During high-throughput curation windows, set `ALWAYS_ON_MODE=true` and redeploy proxy. This disables idle auto-stop until reverted.
