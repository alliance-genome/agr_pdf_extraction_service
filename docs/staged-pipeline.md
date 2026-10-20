# Durable extraction and finalization

The backend uses PostgreSQL as the authority for each public `process_id`.
Redis/Celery messages are repeatable notifications, not job state. Submission
stores the PDF in S3 and commits a ready pipeline record before returning 202.
An independent dispatcher admits extraction and finalization work with bounded
concurrency. The dispatcher indexes and counts only unfinished work, excluding terminal
history from its admission scan. Redis can be unavailable during acceptance; dispatch resumes when
it returns. PostgreSQL and S3 must be available.

The solo GPU worker retains its preloaded models and runs only extraction. It
writes an immutable checkpoint containing the complete source PDF, extractor
Markdown, native artifacts, style/page receipts, images, and source state. The
checkpoint manifest binds file sizes and SHA-256 checksums. CPU prefork workers
restore that checkpoint and run the same merge, provenance, validation and
artifact publication code. They never invoke a GPU extractor. The next PDF can
use the GPU while earlier PDFs finalize.

Each attempt has a separate temporary workspace and artifact destination.
Identical PDFs cannot clear or overwrite another job's cache. Cross-job runtime
extraction/merge cache reuse is intentionally removed; model caches remain
persistent. `clear_cache` options are accepted but every new job already starts
with an isolated empty runtime cache. This trades duplicate-PDF cache hits for
reliable concurrent ownership.

## Ownership and recovery

A generation number fences stale notifications. Each stage also holds a live
PostgreSQL session advisory lock. The dispatcher recovers a stale heartbeat only
when the live lock has been released. A separate watchdog process enforces the
job deadline and cancellation even if a native extension holds the worker's
Python GIL. A separate dedicated sentinel supervises watchdog death; both
monitors share the worker-owned stop pipe and use exact-process handles.
Worker services must retain Compose `init: true`: Linux protects namespace
PID 1 from same-namespace kill signals, so the actual worker must be a child
of the container init process. Init also reaps orphaned monitors.
Losing the ownership connection or either monitor stops that worker
process. A GPU worker restart reloads its models; a finalizer restart affects
only its prefork child.

Before provider work begins, a lost finalizer can safely restore its retained
checkpoint. Once any provider call may have started, an interrupted finalizer
fails with `UncertainProviderOutcome`; it is never automatically replayed as a
whole phase. Provider requests already sent cannot be recalled, so cancellation
or a process crash can leave uncertain usage. Existing bounded per-segment
provider retry and deterministic source fallback behavior is preserved.
PostgreSQL advisory slots cap actual provider calls across all finalizers,
including each SDK call's internal retries.

Cancellation commits terminal state and advances the generation before stopping
work. Stale workers cannot publish a successful terminal result. Downloads use
only the winning generation's artifact keys. A job is complete only after all
required artifacts are durable and its database completion transaction commits.
Unknown process IDs return 404; unavailable status storage returns 503 rather
than an invented pending state. Historical database results remain readable through their retained S3 artifact keys.
Downloads and image listings no longer depend on Redis or a worker-local cache.

The default limits are one admitted GPU extraction, two finalizers, four
unfinished downstream/admitted jobs, and four provider calls. Limits, timeouts,
checkpoint sizes, recovery attempts and container resources are documented in
`.env.example`. A separate backend queue-age limit (24 hours by default) applies until the
first extraction claim. At that claim, the execution deadline starts once and
covers both stages, retries and handoff waits. Queue time does not consume the
scientific finalization budget. The proxy's pre-backend durable queue retains
its separate existing age limit.
Health requires fresh dispatcher, GPU-worker and finalizer heartbeats, and idle
accounting includes queued finalization even when the GPU queue is empty.

## Deployment

Keep the PR's `no-deploy` label through merge: this change requires an ordered
release, and the existing merge workflow otherwise deploys the proxy before
backend schema preparation. Do not enable automatic deployment for this PR.
The new proxy requires `pipeline_job`; deploy it only after draining old work,
applying the migration/grants and starting the staged backend. Mixed old/new
workers and pre-migration proxy startup are deliberately unsupported.

1. Drain the existing proxy/backend queue and all active jobs. Do not mix old
   monolithic workers with the staged scheduler for in-flight work.
2. Configure the pipeline S3 bucket/prefix and lifecycle retention. The AWS
   template supplies these settings and a configurable retention period. Keep
   checkpoint retention longer than the maximum job lifetime plus recovery
   margin. Persistent PostgreSQL must outlive a backend replacement for durable
   recovery; production uses the external database Compose overlay.
3. Apply the Alembic migration before starting the new API, proxy, dispatcher or workers.
   The additive `pipeline_job` table references existing public run records.
   Grant the backend runtime role SELECT, INSERT and UPDATE on both
   `extraction_run` and `pipeline_job` (or verify existing default privileges).
   Grant the proxy role SELECT on both tables. The proxy
   still enforces read-only transactions. `deploy.sh` checks runtime privileges
   after migration and fails before starting services if grants are missing.
4. Deploy one immutable backend image to app, worker, finalizer and dispatcher,
   then deploy the matching proxy. The GPU
   Compose file grants GPU access only to the extraction worker. Do not mount
   source over the image. Check readiness and submit both text and scanned PDFs.
5. Verify durable artifacts, both queues, Sentry and idle shutdown. Existing
   `deploy/aws/ami/warmup.py` checks every requested extractor and real OCR. AMI
   baking uses the isolated `bake-s3.yml` emulator overlay; that overlay is never
   a production storage configuration.

Rollback requires draining staged jobs before stopping the dispatcher/workers.
Do not start the old monolithic worker on unfinished staged records. Retain the
additive table and artifacts for investigation. No production infrastructure or
application rollout is implied by creating the implementation PR.

## Sentry and logging

Use separate `pdfx-dev` and `pdfx-prod` projects in the self-hosted Sentry
organization. API, proxy, GPU extraction, CPU finalization, provider calls and
dispatcher failures carry component/stage tags, environment and immutable image
release. Job correlation uses SHA-256 of the public process ID. Worker logs
include both `process_id` and `process_hash`, allowing a Sentry issue to be
matched to the authenticated job audit and central logs.

Configure `SENTRY_DSN` directly or `SENTRY_RUNTIME_SECRET_ID` with an explicit
`SENTRY_PROJECT`. The secret has a `projects` mapping, each entry containing
`dsn_private_vpc`. AWS deployment parameters grant read access only to that
secret and allow private ingestion only from the backend and proxy security
groups. Proxy deployments load non-secret references from SSM. Apply this
infrastructure configuration before rollout; installing the SDK alone does not
activate reporting. Private HTTP ingestion requires the explicit opt-in;
otherwise HTTPS is required. An empty DSN/reference or `SENTRY_ENABLED=false`
disables reporting. Preserve the kill switch during emergency troubleshooting.
Secret lookup uses bounded connect/read timeouts and attempts; AWS lookup
failures disable reporting in that process and emit a content-free log warning.
They do not fail extraction. Malformed secret contents also disable reporting. Invalid explicit
project/transport settings still fail
explicitly. After a lookup outage, restart the affected service to reinitialize
reporting; watchdog processes also attempt initialization for each new stage.

The shared sanitizer rebuilds events from structural fields only. It excludes
PDF text, prompts, responses, request bodies/headers, credentials, raw URLs,
locals, source context, arbitrary extras and breadcrumbs. Exception messages
are generic; detailed failure evidence stays in authenticated audits/logs.
Tracing is sampled and stripped of payload data. Ordinary stopped-backend
health responses and user cancellations are not errors. Caught terminal job
failures and partial extractor/provider failures are explicitly reported.
Repeated dispatcher outages are coalesced with a configurable interval.

For activation verification, send bounded synthetic failures in the development
project, inspect stage/component/release/job-hash tags and redaction, confirm
absence from production, then verify each deployed component's ingestion.
Configure project issue-owner notification rules and test their routing. A
successful SDK send alone does not prove an event was stored or an alert was
received.

Operational choices: a stage stops immediately if its independent ownership
monitor cannot confirm database ownership. This intentionally favors fencing
and bounded native work over continuing through transient database errors;
post-provider interruptions remain explicit uncertain outcomes. Reporting
service outages are different and do not stop work. Invalid explicit Sentry project or transport configuration remains a startup
error; malformed runtime-secret contents disable reporting with a warning.

Global provider admission waits are bounded by the durable job deadline, not a
separate short local timeout that could trigger scientific fallback without a
provider call. Idle accounting uses nonterminal durable jobs when available;
broker queued/unacked counts remain diagnostics and do not extend the idle
lifetime of completed or cancelled work. The watchdog refreshes extraction
liveness independently of the worker GIL during active native work.

Admission database/ownership failures propagate through all scientific model,
source-selection, title/style and outer merge handlers as operational control
errors. They fail the stage explicitly; only actual provider-response failures
retain the existing scientific retry/fallback behavior.
