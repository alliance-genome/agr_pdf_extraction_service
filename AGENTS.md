# PDFX Agent Notes

This file is safe to commit. Keep host-specific credentials, SSH key paths,
account IDs, one-off IP addresses, tokens, and private operator notes out of
this file. Put local access details in ignored local notes such as
`SSH_ACCESS.md`.

## Architecture

PDFX is an on-demand PDF extraction service with a small always-on proxy and a
GPU backend that should run only when there is work.

- Clients submit PDFs to the Fargate proxy (`pdfx-proxy`; image repo
  `agr_pdfx_proxy`) over HTTPS.
- The proxy validates auth, stores queued uploads durably in S3, wakes the GPU
  backend Auto Scaling Group, and replays queued jobs when the backend is ready.
- The backend is a GPU EC2 host running Docker Compose: nginx, Flask/Gunicorn,
  Celery, Redis, PostgreSQL, GROBID, Docling, and Marker.
- The backend image (`agr_pdfx_backend`) is prebuilt with CUDA/PyTorch so EC2
  boot does not pip-install multi-GB GPU dependencies.
- Audit logs and extracted artifacts live in S3. Extracted image artifacts,
  image manifests, and durable queue payloads have lifecycle cleanup.

## Operating Principles

- Keep the GPU backend cost-safe. Desired capacity should return to zero after
  work finishes, with only stopped warm-pool capacity left behind.
- Treat a deep health check that says the backend is stopped as normal when the
  queue is empty. Liveness for the proxy should not depend on a running backend.
- Do not use `test` names, labels, statuses, or user-visible strings for the
  production pipeline. If a production resource still has a historical `test`
  name, migrate it rather than spreading the name further.
- Prefer IaC and repo scripts over manual console changes. When an emergency
  live hotfix is necessary, backfill the same behavior into templates, docs, and
  tests before calling the work done.
- Keep PDFs out of Fargate memory. The proxy should stream/spool uploads and put
  them in the S3 queue; the backend should read from disk/container volumes.
- Large PDF support is intentionally 500 MiB. If the limit changes, update all
  relevant layers together: Flask config, backend nginx config, Compose env,
  ECS/Fargate scratch capacity, docs, and tests.
- Keep LLM calls bounded. PDFX consensus merge fans out multiple OpenAI calls;
  one slow request can hold the whole Celery task at `llm_merge`. Use
  `LLM_OPENAI_TIMEOUT_SECONDS` and `LLM_OPENAI_MAX_RETRIES` rather than letting
  the OpenAI client wait indefinitely. A timed-out segment should degrade through
  the existing retry/fallback paths.

## Deploy Model

- Proxy deploys are ECS task-definition updates from `proxy/deploy/deploy.sh`.
  Task definitions should include S3 queue settings and enough ephemeral storage
  for multipart upload spooling.
- Backend deploys are image-based. GPU Compose should run app and worker code
  baked into the backend image, not source bind mounts over `/app`.
- The merge workflow builds immutable proxy/backend image tags and then promotes
  `:latest`. A running or stopped warm-pool backend will not use a newly pushed
  backend image until it boots or is replaced.
- For urgent backend fixes, build/push a SHA-tagged backend image, update the
  launch/deploy path deliberately, and verify the backend actually pulled that
  tag. Do not assume `:latest` promotion alone changed an existing instance.

## Alerts And Metrics

Watch the cost and cold-start alarms before and after infrastructure changes:

- Backend startup timeouts and replacement requests indicate cold-start churn.
- Idle-running-too-long and absolute-running-too-long indicate the GPU backend
  may be burning money.
- Idle-guard heartbeat, Lambda errors, and Lambda throttles confirm the cost
  guard is alive.
- Do not route idle/cost guard alarms to OOM-named SNS topics. OOM naming should
  be reserved for alerts that actually measure memory kills or out-of-memory
  events.

Useful proxy metrics include queue depth, oldest pending job age, startup
timeout count, backend replacement count, backend state, and active job counts.

## Known Failure Patterns

- Cold-start loops happen when the backend installs CUDA/PyTorch during boot.
  The fix is a prebuilt backend image, not a longer timeout by itself.
- Stopped warm-pool instances can power off while cloud-init user data is still
  running. Keep backend bootstrap work in the resume-safe
  `pdfx-backend-bootstrap.service`, not only inline cloud-init commands.
- Browser `NetworkError when attempting to fetch resource` on large PDFs can be
  a request-size cap at nginx/Flask or an upload path timeout. Check limits
  before assuming extractor failure.
- Stale S3 queue metadata pointing at a deleted payload can block replay. The
  queue should delete proven-orphaned metadata and continue replaying other
  jobs.
- Backend `worker_not_ready` with healthy EC2 status checks usually means the
  application stack is still starting, not hardware failure.
- A backend job stuck around `llm_merge` with low CPU and no fresh worker logs
  is usually a slow/blocked OpenAI request inside the parallel consensus
  resolver. Check `/api/v1/extract/<process_id>` for progress, worker logs for
  the last LLM segment, and the OpenAI timeout/retry config before assuming GPU
  failure.
- ALB-wide timeout changes affect other services on shared load balancers. Only
  change shared ALB attributes when logs prove that is the failing hop.

## Safe AWS Checks

Use the operator-provided AWS profile and region for the environment. Keep real
account IDs and private IPs out of committed notes.

```bash
aws ssm get-parameter \
  --name /<ssm-prefix>/backend-asg-name \
  --query Parameter.Value \
  --output text

aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names <backend-asg-name>

aws autoscaling describe-warm-pool \
  --auto-scaling-group-name <backend-asg-name>

aws ecs describe-services \
  --cluster <proxy-cluster> \
  --services <proxy-service>

aws logs tail /ecs/<proxy-service> --since 2h --format short
```

## Validation

Prefer focused checks; this repo has both backend `app` and proxy `app`
packages, so unscoped `pytest` can collect under the wrong import path unless
the environment is prepared carefully.

```bash
python3 -m pytest -q proxy/tests
python3 -m pytest -q tests/test_aws_pdfx_stack.py
bash -n deploy/deploy.sh
bash -n deploy/aws/deploy_idle_guard.sh
python3 -m py_compile deploy/aws/lambda/pdfx_idle_guard.py
python3 -m compileall -q proxy/app
git diff --check
```

For upload fixes, perform an end-to-end production-path submission after deploy
with a non-sensitive large PDF or an operator-approved sample. Verify the client
gets a process ID, the proxy queue drains, the backend accepts the job, and the
GPU backend stops or returns to warm-pool idle afterward.

## Security Notes

- Never commit SSH key paths, credentials, bearer tokens, private sample files,
  raw SSM SecureString values, account IDs, public IPs tied to emergency access,
  or screenshots/logs containing secrets.
- Prefer SSM/ECS secret injection and IAM roles over environment values copied
  into docs.
- If auth fails locally, do not rewrite credentials or start a new login flow
  unless the operator asks for it. The credential may simply be locked.
