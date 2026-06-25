# PDFX Proxy

A lightweight FastAPI proxy that sits in front of the GPU-based PDF Extraction Service. It handles authentication, auto-starts the EC2 GPU instance on demand, queues jobs during startup, and forwards requests once the backend is ready. Designed to run on AWS Fargate so the expensive GPU instance can be stopped when idle.

## Why a Proxy?

The PDF extraction backend runs on a GPU instance (currently g5.4xlarge). Leaving it running 24/7 is wasteful when jobs arrive intermittently. The proxy solves this by:

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
                    │   (EC2 g5.4xlarge)    │
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

On proxy startup, `sync_state_from_ec2()` checks the actual EC2 state so the proxy's internal state matches reality.

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
| `IDLE_TIMEOUT_MINUTES` | No | `30` | Minutes of inactivity before EC2 is stopped |
| `MIN_UPTIME_MINUTES` | No | `20` | Minimum uptime after wake before idle stop is allowed |
| `STARTUP_TIMEOUT_MINUTES` | No | `30` | Max minutes to wait for EC2 health check |
| `ASG_STARTUP_REPLACEMENT_ATTEMPTS` | No | `1` | Extra ASG replacement attempts to wait through after startup timeout |
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
- SSM parameters under the target environment prefix (`/pdfx/*` for prod)
- IAM roles: execution role (ECR pull + SSM read + CloudWatch Logs) and task role (Auto Scaling lifecycle + EC2 describe + SSM read). Legacy single-instance deployments also need EC2 start/stop on the managed instance.

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

### Deploying to a non-prod mirror

For a test mirror with separate SSM parameters and ECS resources, pass the
environment-specific names explicitly:

```bash
cd proxy/deploy
./deploy.sh \
  --profile ctabone \
  --region us-east-1 \
  --cluster pdfx-proxy-test \
  --service pdfx-proxy-test \
  --ssm-prefix /pdfx-test \
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
- Target: a single `prod` environment. The `pdfx-test` mirror is deployed
  manually so production releases remain approval-gated and single-purpose.
- Approval gate: the `deploy-prod` job is attached to the GitHub Actions
  `prod` environment, so required reviewers can block production rollout
  until explicitly approved
- Image handling: the workflow builds the proxy image once, uploads it as a
  GitHub Actions artifact, and the deploy job downloads that exact archive,
  verifies its checksum, pushes it to ECR as
  `agr_pdfx_proxy:<merge-commit-sha>` plus `:latest`, then runs
  `proxy/deploy/deploy.sh --image-tag <merge-commit-sha>`

Required GitHub setup:

- Create a GitHub Actions environment named `prod`
- Store `GH_ACTIONS_AWS_ROLE` as an **environment** secret on `prod` (not a
  repository secret) pointing at the AWS role ARN below. Keeping it scoped
  to the environment is what lets required-reviewer protection actually gate
  access to the role.
- Configure required reviewers on the `prod` environment if you want a
  manual approval gate before the deploy job runs

The image promotion sequence inside the deploy job is:

1. push `agr_pdfx_proxy:<merge-commit-sha>` to ECR (immutable artifact)
2. run `proxy/deploy/deploy.sh --image-tag <merge-commit-sha>` to register a
   new ECS task definition and roll the `pdfx-proxy` service
3. only after step 2 succeeds, re-tag the same image as `:latest` and push

This keeps `:latest` pointing at the most recent image that actually rolled
out to prod — if ECS rollout fails, `:latest` does not move.

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

Use `/pdfx-test/cognito-accepted-client-ids` and
`./deploy.sh --ssm-prefix /pdfx-test ...` for the test mirror.

## Operational Fallbacks

### Always-On Worker Window
During high-throughput curation windows, set `ALWAYS_ON_MODE=true` and redeploy proxy. This disables idle auto-stop until reverted.
