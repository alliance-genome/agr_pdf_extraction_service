# PDFX Proxy

A lightweight FastAPI proxy that sits in front of the GPU-based PDF Extraction Service. It handles authentication, auto-starts the EC2 GPU instance on demand, queues jobs during startup, and forwards requests once the backend is ready. Designed to run on AWS Fargate so the expensive GPU instance can be stopped when idle.

## Why a Proxy?

The PDF extraction backend runs on a GPU instance (currently g5.4xlarge). Leaving it running 24/7 is wasteful when jobs arrive intermittently. The proxy solves this by:

1. Running cheaply on Fargate (256 CPU / 2048 MB — pennies/hour)
2. Auto-starting the GPU instance when a job arrives
3. Queuing jobs while EC2 boots (~2-3 minutes), with optional durable S3 queue
4. Claiming and replaying durable jobs once the backend is healthy
5. Reading authoritative job status from the existing RDS table without waking EC2
6. Auto-stopping the GPU instance after an idle timeout

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
                    │   (EC2 g5.4xlarge)    │
                     │                      │
                     │  - GROBID / Docling   │
                     │  - Marker / LLM Merge │
                     │  - PostgreSQL / Redis │
                     └──────────────────────┘
```

## API Endpoints

All endpoints except `/api/v1/health/live`, `/api/v1/health`, `/api/v1/health/deep`, and `/api/v1/metrics` require a Cognito Bearer token.

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/v1/health/live` | GET | No | Proxy-only container liveness probe |
| `/api/v1/health` | GET | No | Proxy health + EC2 state (`healthy`, `busy`, or degraded/stopped state) |
| `/api/v1/health/deep` | GET | No | Deep probe: proxy auth validation + downstream status round-trip |
| `/api/v1/metrics` | GET | No | Queue/replay/lifecycle/canary metrics for alerting |
| `/api/v1/status` | GET | Yes | Passive EC2 state, idle time, active/queued job counts |
| `/api/v1/status?wake=true` | GET | Yes | Status plus explicit backend wake if stopped |
| `/api/v1/wake` | POST | Yes | Start the GPU instance (idempotent) |
| `/api/v1/extract` | POST | Yes | Submit a PDF for extraction |
| `/api/v1/extract/{id}` | GET | Yes | Poll job status with granular progress |
| `/api/v1/extract/{id}/cancel` | POST | Yes | Request cancellation of a queued/running extraction job |
| `/api/v1/extract/{id}/download/{method}` | GET | Yes | Download backend extraction output |

### Submit Extraction (`POST /api/v1/extract`)

Accepts the same multipart form fields as the backend (`file`, `methods`, `merge`, `clear_cache`, `clear_cache_scope`, `reference_curie`, `mod_abbreviation`).

**For every accepted upload:** the proxy first writes the PDF and queue metadata
to the configured queue. In production, one ECS task then conditionally claims
that S3 record and forwards it to the backend. This same durable-first path is
used whether EC2 is already ready or still starting.

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

Once EC2 is healthy, queued jobs are automatically replayed one at a time.
Transient startup, capacity, network, and backend 5xx failures release the
claim and retain the queue record. Only proven permanent submission rejection,
explicit cancellation, or eligible retention cleanup removes unaccepted work.

In S3 queue mode, the proxy stores the PDF upload as a separate object under
`<QUEUE_S3_PREFIX>/payloads/` and stores only small job metadata under
`<QUEUE_S3_PREFIX>/jobs/`. This avoids base64-encoding large PDFs into queue
JSON and keeps Fargate memory bounded while the GPU instance wakes up.
Per-job leases live under `<QUEUE_S3_PREFIX>/claims/`; a conditional S3 write
prevents two overlapping ECS tasks from replaying one record. After the backend
accepts the stable process ID, the proxy writes a secret-free handoff marker
under `<QUEUE_S3_PREFIX>/accepted/` before acknowledging queue metadata.
Uploads larger than `MAX_UPLOAD_BYTES` are rejected before backend wake/replay
so oversized submissions do not fill the durable queue or start the GPU. When
`Content-Length` is present, the proxy rejects grossly oversized requests before
multipart parsing, allowing `MAX_MULTIPART_OVERHEAD_BYTES` for boundaries and
form fields.
`RECONCILER_REQUEUE_ONCE` is disabled by default. Receiver-side idempotency
means an ambiguous at-least-once replay with the same process ID cannot publish
a second Celery task or reset a terminal RDS row.

### Poll Status (`GET /api/v1/extract/{id}`)

Returns granular progress through the extraction pipeline:

Celery reports `PENDING` for both a queued task and an unknown task ID. The
proxy therefore returns an untracked backend `pending` response to the caller
without treating that arbitrary ID as active work. Known submitted jobs remain
tracked. An unchanged bare `pending` fallback cannot renew the reconciler's
stale deadline, while database-backed queued jobs and concrete running/progress
states retain their existing protection.

Durable queued, claimed, and accepted phases are checked before an ID is called
unknown. The proxy then reads the existing `extraction_run` row directly from
RDS using a bounded read-only session. A reachable RDS row is authoritative,
including its original terminal error. If RDS is temporarily unreachable, a
durably accepted ID returns HTTP 200 `pending` with the same process ID instead
of 404. A truly unknown ID stays 404 and does not wake the GPU backend.

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

**While the backend worker is busy with another PDF:**
```json
{
  "process_id": "...",
  "status": "queued",
  "progress": {
    "stage": "queued",
    "stage_display": "PDFX worker busy; job waiting in queue",
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
| `queued` | PDFX worker busy; job waiting in queue | Proxy |
| `initializing` | Initializing extraction job | Backend |
| `grobid` | Running GROBID extraction | Backend (conditional) |
| `docling` | Running Docling extraction | Backend (conditional) |
| `marker` | Running Marker extraction | Backend (conditional) |
| `llm_merge` | Merging extraction outputs with LLM | Backend (conditional) |
| `finalizing` | Uploading artifacts and finalizing | Backend |

Extraction stages are dynamic based on the `methods` parameter. `llm_merge` only appears when `merge=true`.

## Backend Lifecycle State Machine

```
  STOPPED ──(job arrives)──> STARTING ──(health check OK)──> READY
     ^                                                         │
     │                                                    (job running)
     │                                                         │
     └──(idle timeout)──── READY <──(job done)──── BUSY ◄──────┘
```

| State | Meaning |
|-------|---------|
| `STOPPED` | Backend capacity is off. Jobs trigger EC2 start or ASG desired capacity 1. |
| `STARTING` | Backend is booting. Jobs are queued in memory or durable S3. |
| `READY` | Backend is healthy. Requests are forwarded. |
| `BUSY` | At least one job is in flight. Idle timer is paused. |

The idle monitor checks every 60 seconds. The worker is only eligible for stop when all guards pass:
- no queued jobs
- no replay-inflight jobs
- no tracked active backend jobs
- minimum uptime (`MIN_UPTIME_MINUTES`) has elapsed
- `ALWAYS_ON_MODE` is disabled

When guards pass and idle exceeds `IDLE_TIMEOUT_MINUTES`, the backend is stopped automatically. In legacy mode this calls `StopInstances`; in Auto Scaling mode it sets the backend ASG desired capacity to `0`.

On proxy startup, `sync_state_from_ec2()` checks the actual EC2 state so the
proxy's internal state matches reality. `ensure_running()` and startup sync
share one transition lock and one monotonically increasing startup generation.
Each monitor owns a generation and exact instance ID; after every health/AWS
boundary it revalidates both before it can mark or stop an instance. Establishing
READY invalidates older monitors, so an obsolete deadline cannot replace the
current healthy backend.

Preferred production mode is `BACKEND_ASG_NAME`. The proxy scales the ASG to desired capacity `1` on wake and discovers the current healthy instance private IP from the ASG. If the backend fails to become healthy before `STARTUP_TIMEOUT_MINUTES`, the proxy marks the current ASG instance `Unhealthy` so EC2 Auto Scaling replaces it from the launch template, then keeps queued replay waiting for up to `ASG_STARTUP_REPLACEMENT_ATTEMPTS` replacement attempts. Keep the ASG `MaxSize` at `1` for strict cost control, or `2` only when deliberately testing launch-before-terminate behavior.

## Configuration

All settings come from environment variables. In production, values are injected from AWS SSM Parameter Store via the ECS task definition's `secrets` block.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BACKEND_ASG_NAME` | Conditional | — | Preferred managed backend Auto Scaling group name. Required unless `EC2_INSTANCE_ID` is set |
| `EC2_INSTANCE_ID` | Conditional | — | Legacy managed GPU instance ID. Required unless `BACKEND_ASG_NAME` is set |
| `COGNITO_USER_POOL_ID` | Yes | — | Cognito user pool for JWT validation |
| `EC2_REGION` | No | `us-east-1` | AWS region for EC2 API calls |
| `EC2_PORT` | No | `5000` | Port the backend listens on |
| `COGNITO_REGION` | No | `us-east-1` | AWS region for Cognito |
| `COGNITO_REQUIRED_SCOPE` | No | `pdfx-api/extract` | OAuth scope accepted in the JWT |
| `COGNITO_ACCEPTED_SCOPES` | No | — | Comma-separated additional scopes that also grant access |
| `COGNITO_ACCEPTED_CLIENT_IDS` | No | — | Comma-separated Cognito app client_ids accepted without requiring a PDFX-specific scope (e.g. the CurationAPI-Admin M2M client) |
| `IDLE_TIMEOUT_MINUTES` | No | `120` | Minutes of inactivity before EC2 is stopped |
| `MIN_UPTIME_MINUTES` | No | `20` | Minimum uptime after wake before idle stop is allowed |
| `STARTUP_TIMEOUT_MINUTES` | No | `30` | Max minutes to wait for EC2 health check |
| `ASG_STARTUP_REPLACEMENT_ATTEMPTS` | No | `1` | Extra ASG replacement attempts to wait through after startup timeout |
| `HEALTH_POLL_INTERVAL_SECONDS` | No | `15` | Seconds between EC2 health polls during startup |
| `MAX_QUEUED_JOBS` | No | `10` | Max queued jobs during startup |
| `MAX_UPLOAD_BYTES` | No | `524288000` | Max PDF upload size accepted by the proxy (500 MiB) |
| `MAX_MULTIPART_OVERHEAD_BYTES` | No | `10485760` | Multipart overhead allowance for early request-size rejection |
| `FORWARD_TIMEOUT_SECONDS` | No | `600` | Timeout for forwarded HTTP requests to EC2 |
| `PROXY_BACKEND_READY_TIMEOUT_SECONDS` | No | `30` | Max seconds artifact/image proxy routes wait for a refreshed or waking backend before returning 503 |
| `PROXY_BACKEND_READY_POLL_SECONDS` | No | `2` | Seconds between backend readiness checks while artifact/image proxy routes wait |
| `ALWAYS_ON_MODE` | No | `false` | Emergency mode that disables idle auto-stop |
| `QUEUE_BACKEND` | No | `memory` | Queue backend: `memory` or `s3` |
| `QUEUE_S3_BUCKET` | No | — | S3 bucket for durable queue (`QUEUE_BACKEND=s3`) |
| `QUEUE_S3_PREFIX` | No | `pdfx-proxy-queue` | S3 prefix for durable queue metadata and PDF payload objects |
| `QUEUE_S3_REGION` | No | — | Optional S3 region override |
| `QUEUE_CLAIM_TTL_SECONDS` | No | `900` | Lease duration for one ECS replay owner; expired claims are conditionally reclaimable after task loss |
| `REPLAY_RETRY_DELAY_SECONDS` | No | `30` | Backoff before durable queued work retries a transient backend handoff without waiting for new traffic |
| `ACCEPTED_STATUS_RETENTION_SECONDS` | No | `604800` | Minimum age before a caller-proven terminal accepted/status marker may be removed; active markers are preserved |
| `ACCEPTED_CLEANUP_BATCH_SIZE` | No | `25` | Maximum expired accepted/status markers revalidated against RDS per reconciler pass |
| `STATUS_DATABASE_URL` | Production | — | Existing backend RDS URL used only by read-only status sessions; injected from SSM in ECS |
| `STATUS_DB_TIMEOUT_SECONDS` | No | `5` | Connection and statement timeout for the always-on authoritative status lookup |
| `STATUS_ERROR_MESSAGE_MAX_CHARS` | No | `4000` | Maximum error text returned by direct RDS status, matching the backend response bound |
| `SHARED_RUNNING_MAX_AGE_MINUTES` | No | `60` | Maximum age of an unchanged RDS `running` row that can block exact-target startup replacement; exceeds the 35-minute backend hard limit |
| `PROXY_SHUTDOWN_GRACE_SECONDS` | No | `90` | Grace for an in-progress replay handoff before cancellation releases its claim |
| `STUCK_PENDING_MINUTES` | No | `20` | Age threshold for stale pending/running jobs |
| `RECONCILER_INTERVAL_SECONDS` | No | `60` | Background reconciler interval |
| `RECONCILER_REQUEUE_ONCE` | No | `false` | Optional one-time requeue of stale jobs |
| `HEALTHCHECK_BEARER_TOKEN` | No | — | Token used by `/api/v1/health/deep` |
| `CANARY_INTERVAL_SECONDS` | No | `0` | Downstream canary interval (0 disables) |
| `CANARY_BEARER_TOKEN` | No | — | Token used for canary downstream probe |

## Deployment

The proxy is deployed as an ECS Fargate service behind an ALB.

Production allocates 50 GiB of Fargate ephemeral storage. Durable S3 queue
uploads spool at most the configured 500 MiB request limit at a time, leaving
ample scratch headroom during burst replay and overlapping task deployments.

### Prerequisites

- ECR repository for the proxy image
- ECS cluster with Fargate capacity
- Cognito user pool with a resource server and `pdfx-api/extract` scope
- SSM parameters under the target environment prefix (`/pdfx/*` for prod), including the existing `database-url` SecureString
- IAM roles: execution role (ECR pull + SSM read + CloudWatch Logs) and task role (Auto Scaling lifecycle + EC2 describe + SSM read). Legacy single-instance deployments also need EC2 start/stop on the managed instance.
- Network access from the proxy task security group to the RDS security group on TCP 5432. The proxy image includes the RDS CA bundle and opens each status session with transaction read-only enforcement.

### Build and Deploy

```bash
# Run from repo root
cd proxy

# Build and push to ECR
docker build -t agr_pdfx_proxy .
docker tag agr_pdfx_proxy:latest <account>.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_proxy:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_proxy:latest

# Register the ECS task definition and roll ECS service (reads /pdfx SSM parameters automatically)
cd deploy
./deploy.sh --profile <profile>

# Optional: deploy an immutable image tag instead of :latest
./deploy.sh --profile <profile> --image-tag <git-sha-or-release-tag>

# Or dry-run to inspect the generated task definition
./deploy.sh --profile <profile> --dry-run

# Optional: register task definition only (skip ECS service update)
./deploy.sh --profile <profile> --no-update-service
```

### Deploying with explicit names

To override the default PDFX SSM parameters or ECS resources, pass the names
explicitly:

```bash
cd proxy/deploy
./deploy.sh \
  --profile ctabone \
  --region us-east-1 \
  --cluster pdfx-proxy \
  --service pdfx-proxy \
  --ssm-prefix /pdfx \
  --image-tag <image-tag>
```

By default the task family, container name, log group, and queue prefix follow
`--service`. Override them with `--task-family`, `--container-name`,
`--log-group`, or `--queue-prefix` only when the ECS service was provisioned
with different names.

### GitHub Actions auto-deploy

The repo-level workflow at `.github/workflows/main-build-and-deploy.yml`
automates the manual steps above when a PR is merged into `main`.

- Trigger: `pull_request.closed` on `main`, guarded by `github.event.pull_request.merged == true`
- Escape hatch: add the `no-deploy` label to the PR to skip the deployment job
- Manual recovery: `workflow_dispatch` can force a proxy deploy, a backend
  image+AMI bake, or both from the selected ref. Backend bakes can also override
  the temporary Packer subnet when an AZ has insufficient GPU capacity.
- Target: the canonical `pdfx` environment.
- Approval gate: the `deploy-prod` job is attached to the GitHub Actions
  `prod` environment, so required reviewers can block production rollout
  until explicitly approved
- Path-aware releases: proxy inputs build and roll only the proxy; backend
  inputs build the backend image and publish a baked AMI; idle-guard changes
  emit explicit operator deployment instructions. Documentation-only merges
  do not roll production.

Required GitHub setup:

- Create a GitHub Actions environment named `prod`
- Store `GH_ACTIONS_AWS_ROLE` as an **environment** secret on `prod` (not a
  repository secret) pointing at the AWS role ARN below. Keeping it scoped
  to the environment is what lets required-reviewer protection actually gate
  access to the role.
- Configure required reviewers on the `prod` environment if you want a
  manual approval gate before the deploy job runs

The proxy image promotion sequence inside the deploy job is:

1. push `agr_pdfx_proxy:<merge-commit-sha>` to ECR (immutable artifact)
2. run `proxy/deploy/deploy.sh --image-tag <merge-commit-sha>` to register a
   new ECS task definition and roll the `pdfx-proxy` service
3. run public liveness, health, and metrics smoke checks
4. only after rollout and smoke checks succeed, re-tag the same image as
   `:latest` and push

This keeps `:latest` pointing at the most recent image that actually rolled
out to prod — if ECS rollout fails, `:latest` does not move.

Backend releases use the same environment-scoped AWS role. The immutable
backend image is built only for backend-affecting changes. CI then bakes and
validates the AMI, publishes and verifies `/pdfx/backend-ami` plus the mirrored
`/pdfx/backend-image-tag` release record, prunes old AMIs, and only then promotes
the backend image to `:latest`. The AMI carries its authoritative immutable tag
in `/opt/pdfx/backend-image-tag`; the SSM tag remains a legacy bootstrap
fallback until the corresponding launch-template update is applied. A failed
bake or pair publication therefore cannot move backend `:latest`.

`proxy/deploy/deploy.sh` also enforces `minimumHealthyPercent=100` and
`maximumPercent=200` on every ECS update. With one desired task, ECS must keep
the old target healthy until the replacement passes its checks.

The assumed AWS role needs enough access to:

- authenticate to and push images into the `agr_pdfx_proxy` ECR repository
- read the target SSM prefix from Parameter Store (`/pdfx/*` for prod)
- register ECS task definitions and update the `pdfx-proxy` ECS service
- pass the ECS execution and task roles referenced by the task definition

If you want `deploy.sh` to auto-create the optional
`/<ssm-prefix>/backend-asg-name`,
`/<ssm-prefix>/cognito-accepted-scopes` and
`/<ssm-prefix>/cognito-accepted-client-ids` placeholders when missing, also
grant `ssm:PutParameter` on the selected prefix.

### IAM Permissions (Task Role)

The task role needs:
- `autoscaling:SetDesiredCapacity` — scale backend ASG desired capacity between 0 and 1
- `autoscaling:SetInstanceHealth` — mark failed-startup backend instances unhealthy for replacement
- `autoscaling:DescribeAutoScalingGroups` — discover the current backend instance
- `ec2:StartInstances` / `ec2:StopInstances` — legacy single-instance mode only, scoped to the configured instance ARN
- `ec2:DescribeInstances` — for state polling
- `ssm:GetParameters` — scoped to the selected SSM prefix

See `deploy/iam-policy.template.json` for the full policy.

## Project Structure

```
proxy/
├── app/
│   ├── main.py            # FastAPI routes (health, status, wake, extract, poll, download)
│   ├── config.py           # Settings from environment variables
│   ├── auth.py             # Cognito JWT validation
│   ├── ec2_manager.py      # Backend lifecycle via EC2 or Auto Scaling APIs
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
# Fill in BACKEND_ASG_NAME or EC2_INSTANCE_ID, plus COGNITO_USER_POOL_ID

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

The proxy will sync with the actual backend state on startup.

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

Both settings are injected into the ECS task via the task definition's
`secrets` block, sourced from these SSM parameters in the selected
environment prefix:

| SSM parameter | Maps to env var |
|---------------|-----------------|
| `/<ssm-prefix>/cognito-accepted-scopes` | `COGNITO_ACCEPTED_SCOPES` |
| `/<ssm-prefix>/cognito-accepted-client-ids` | `COGNITO_ACCEPTED_CLIENT_IDS` |

`deploy.sh` ensures both parameters exist before registering the task
definition: if either is missing it creates a String parameter with a
single-space placeholder (SSM does not allow empty String values; the
proxy's config layer `.strip()`s the placeholder back to `""`, leaving
the allow-list inactive). Existing values are never overwritten.

This means the operator running `deploy.sh` needs `ssm:PutParameter` on the
selected SSM prefix in addition to `ssm:GetParameter` (the ECS task execution
role only needs `ssm:GetParameters`, which is already granted by
`iam-policy.template.json`).

To enable shared M2M access, populate the parameter and redeploy:

```bash
aws ssm put-parameter \
  --name /pdfx/cognito-accepted-client-ids \
  --type String \
  --overwrite \
  --value "<curation-admin-client-id>[,<other-client-id>...]"

cd proxy/deploy && ./deploy.sh --profile <profile>
```

Use `/pdfx/cognito-accepted-client-ids` and
`./deploy.sh --ssm-prefix /pdfx ...` for the PDFX stack.

## Operational Fallbacks

### Always-On Worker Window
During high-throughput curation windows, set `ALWAYS_ON_MODE=true` and redeploy proxy. This disables idle auto-stop until reverted.
