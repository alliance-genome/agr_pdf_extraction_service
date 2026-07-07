# PDFX Backend: Capacity Resilience & Cost-Aware Cold Start

**Date:** 2026-07-07
**Status:** Approved design (Opus design review incorporated) — pending implementation plan
**Scope:** `agr_pdf_extraction_service` GPU backend (ASG `pdfx-backend`), its AMI, and the CI that produces it. The Fargate proxy behavior is unchanged except where noted.

> **Review note:** This spec incorporates an independent design review (2026-07-07). Key changes from the first draft: the g6/L4 rollout is **staged** (§5), the boot fast-path is pinned to an **immutable image digest** rather than a `:latest` tag (§4B/§6), the bake runs a **real sample extraction** so Docling/GROBID/rapidocr caches are truly baked (§4), IAM and the `resolve:ssm` permission dependency are spelled out (§7), and several safety details (N1–N6) are folded into §5–§9.

---

## 1. Background & motivation

On 2026-07-07 the PDF extraction service was reported unreachable by a curator, with several uploads failing. Investigation (AWS ASG activity, ECS events, proxy CloudWatch logs, EC2 console output, live health probes) established the following chain, entirely on the GPU backend — the Fargate proxy never recycled:

1. The backend had been idle-scaled to zero. A job arrived and the proxy scaled the ASG `pdfx-backend` from 0→1.
2. The **warm-pool fast start failed with "Insufficient g5.2xlarge capacity."** A stopped warm-pool instance can only restart in its own AZ (us-east-1d at the time), which was capacity-constrained. Fresh launches in another AZ also hit intermittent capacity errors before one succeeded.
3. Because the warm path failed, the ASG launched a **fresh** instance that had to run the full bootstrap (install Docker, `git clone`, pull the ECR GPU image, `deploy.sh` + model prewarm). Console output showed containers did not even start until **~22 minutes** into boot.
4. The proxy's **30-minute startup timeout** fired before the worker was ready. The proxy marked the instance unhealthy; the ASG **replaced** it — and the replacement had to cold-start from scratch again, extending the outage.
5. Net curator impact: **~15 uploads failed** (9 as proxy progress-timeouts, 6 as `502` replay failures) over roughly a 10-minute window.

### Root causes

- **Single instance type, no fallback.** The launch template used only `g5.2xlarge` on-demand. When that one capacity pool was tight, there was nothing to fall back to.
- **Slow fresh bootstrap racing a fixed timeout.** A from-scratch boot (~22 min) is close enough to the 30-min startup timeout that a warm-pool miss cascades into a force-replace loop, doubling the outage.
- **Warm pool is inherently single-AZ and "sticky."** It pins one stopped instance of one type in one AZ; it is fragile to that AZ's capacity and also keeps whatever (possibly expensive) type it last held.

### Confirmed facts used in this design

- All five us-east-1 AZs (a/b/c/d/f) offer `g5.2xlarge`, `g5.4xlarge`, and `g6.2xlarge`; `g6e.2xlarge` is offered in four (no us-east-1f).
- The ASG spans all five AZs at the subnet level in production, but the CloudFormation template (`deploy/aws/pdfx-stack.yaml`) pins a single subnet — i.e. **production has drifted from the template**.
- There are **no capacity reservations** in the account.
- CI (`.github/workflows/main-build-and-deploy.yml`) already assumes an AWS role via OIDC (`secrets.GH_ACTIONS_AWS_ROLE`) and builds/pushes `agr_pdfx_backend` from `deploy/Dockerfile.gpu` in the `deploy-prod` job, tagged with the immutable merge SHA and re-pointing `:latest` on every merge.
- `deploy/deploy.sh` exposes the knobs a bake needs (`GPU_MODE`, `PDFX_DEPLOY_BUILD_MODE=never`, `PDFX_DEPLOY_PULL_IMAGES=always|never`, `PDFX_PREWARM_MODELS=marker|off`), and blocks on full worker health (`wait_for_flask_health` + `probe_worker_cuda`, up to 1800s). **Its prewarm step loads Marker models only** — Docling/rapidocr/GROBID caches are populated lazily on first use, and worker "ready" gates on the Marker readiness file alone.

---

## 2. Goals & non-goals

### Goals

1. **Capacity resilience** — survive a single-instance-type or single-AZ capacity shortage without a slow fallback or a replace loop.
2. **Fast cold start** — reduce time-to-ready from ~22 min to ~2–4 min by baking Docker images and *all* ML model caches into the AMI.
3. **Cost-aware selection** — always prefer the cheapest eligible GPU type; only land on a pricier type when the cheaper pools are genuinely empty; do not stay "stuck" on an expensive type after a job finishes.
4. **Reproducible IaC** — the AMI build is defined in-repo and can be re-run / updated on demand and automatically.

### Non-goals

- No Spot instances (interruptions would fail in-flight extractions). On-demand only.
- No change to the consensus/extraction pipeline or the proxy's queueing/replay logic.
- No special-case "is this instance expensive?" runtime logic (see §5 — terminating every idle instance and re-selecting cheapest achieves the goal without it).
- This spec stops at an open PR. Applying anything to production is a separate, human-driven step (§8).

---

## 3. Behavior summary (target end state)

- On job arrival, the proxy scales the ASG 0→1 (unchanged). The ASG's **MixedInstancesPolicy** picks the cheapest available *eligible* type across all AZs.
- The instance boots from a **pre-baked AMI**: Docker images and all ML model caches are already on disk, so it only fetches secrets from SSM and starts the compose stack (~2–4 min).
- On idle, the proxy scales desired back to 0. With **no warm pool**, that terminates the instance.
- The next wake re-runs the allocation from scratch — so the service re-attempts the cheapest eligible type every time and never stays pinned to an expensive one.

---

## 4. Component A — Baked AMI via Packer

**Location:** `deploy/aws/ami/`

- `pdfx-backend.pkr.hcl` — Packer template using the `amazon-ebs` builder.
  - Source AMI: the Deep Learning Base OSS NVIDIA Driver GPU AMI (same family as prod today; parameterized).
  - Build instance type: **g6.2xlarge** — this also serves as the standing L4/Ada smoke test (the bake fails if the stack can't run on L4), which de-risks the §5 Phase 2 addition of g6 to the run mix.
  - Region: us-east-1. Encrypted EBS. Root volume sized to hold images + model caches (≥ current 200 GiB unless measured smaller).
  - Variables: backend image repo + **immutable tag** to bake (the merge SHA, not `latest`), base AMI id, retention count.
- `provision.sh` — thin wrapper invoked by Packer that **reuses the production boot path** so bake and boot cannot diverge:
  1. Install Docker, the compose plugin, and confirm the NVIDIA container runtime (same steps as the launch-template bootstrap).
  2. `git clone`/checkout the target ref.
  3. ECR login **via the build instance's instance profile** (no static credentials on disk — see §7) and run `deploy.sh` with `GPU_MODE=on PDFX_DEPLOY_BUILD_MODE=never PDFX_DEPLOY_PULL_IMAGES=always PDFX_PREWARM_MODELS=marker`, which pulls the target backend image tag plus all compose images (postgres, redis, grobid, worker, app, nginx) and prewarms Marker.
  4. **Run one real sample extraction** end-to-end against a small bundled test PDF, so the lazily-loaded **Docling, rapidocr, and GROBID** caches populate in addition to Marker. This makes the bake a *true* warm cache rather than Marker-only.
  5. **Assert every mounted cache dir is non-empty** (`data/models`, `data/model_cache`, `data/rapidocr_models`, and the Docling/HuggingFace cache — confirm Docling writes under the bind-mounted `HF_HOME=/app/data/models` so it persists into the AMI) before proceeding. Fail the bake if any is empty.
  6. Bring the stack down (`docker compose down`) but leave images and model caches on disk.
- **Bake `.env` (N2):** `deploy.sh` runs the full stack and blocks on real worker health, so the bake needs a `.env` sufficient for `create_app()` and the worker to reach health — **placeholders, never real secrets** (e.g. dummy LLM keys if `create_app()` hard-requires them). The real `/pdfx/backend-env` SecretString is **never** fetched during the bake.
- **Bake hygiene (security-critical):** before snapshot, remove anything secret or identity-bearing: the bake `.env`, ECR docker credentials (`~/.docker/config.json`), any `~/.aws` static credentials, cloud-init state, SSH host keys, `/etc/machine-id`, bootstrap/deploy logs, and shell history. Secrets remain in SSM and are fetched at **boot**, exactly as today.
- **Marker file:** write `/opt/pdfx/baked.json` recording base AMI id, backend image **tag and digest**, a model manifest (which caches were populated), and a UTC timestamp. Used by the boot fast-path (§4B) and for observability.
- **Output:** an AMI tagged `Project=pdfx`, `Role=backend-baked`, and the backend image tag; the build publishes the AMI id and the immutable image tag to SSM (§6).

### 4B. Boot fast-path (launch-template user-data change)

The launch-template user-data (in `pdfx-stack.yaml`) gains a branch at the top of the bootstrap, extracted into a small **unit-testable shell function** (`should_use_baked_fastpath`):

- Determine the digest of the backend image to be run: `aws ecr describe-images` for the pinned immutable tag (from `/pdfx/backend-image-tag`, §6).
- If `/opt/pdfx/baked.json` exists **and** its recorded image **digest equals** that digest: run `deploy.sh` with `PDFX_DEPLOY_PULL_IMAGES=never PDFX_PREWARM_MODELS=off` — images and all caches are present, so this is a fast `compose up`. Secrets are still fetched from SSM at boot.
- Otherwise (marker missing, or digest mismatch): **fall back to today's full pull+prewarm path.** A stale or missing AMI therefore degrades to current behavior rather than failing. Because the check is on the immutable **digest** (not `:latest`), this safety net is real, not a tautology.

---

## 5. Component B — MixedInstancesPolicy, multi-AZ, and terminate-on-idle

**Location:** `deploy/aws/pdfx-stack.yaml`

### MixedInstancesPolicy (staged g6 rollout)

Replace the ASG's single `LaunchTemplate` with a `MixedInstancesPolicy`.

- **Phase 1 (this work):** `LaunchTemplate.Overrides = [g5.2xlarge, g5.4xlarge]` — proven A10G types only. This already yields multi-type × multi-AZ resilience (2 types × 5 AZs), the baked fast boot, and terminate-on-idle. `g6.2xlarge` is deliberately **excluded** until L4 is validated.
- **Phase 2 (separate follow-up, tracked in Jira):** add `g6.2xlarge` to `Overrides` after the L4 validation gate (§8) passes. Because the allocation strategy is `lowest-price`, g6 cannot be "present but deprioritized" — so it must stay out of the list until proven, then be added.
- `InstancesDistribution`:
  - `OnDemandBaseCapacity: 0`, `OnDemandPercentageAboveBaseCapacity: 100` (100% on-demand, no Spot).
  - `OnDemandAllocationStrategy: lowest-price` — picks the cheapest *eligible* type first (Phase 1: g5.2xlarge, then g5.4xlarge on capacity fallback), which is exactly the "always cheapest, fall back on capacity" behavior.

> **Why staged (design-review B2):** `lowest-price` always launches the cheapest type first and ignores list order. So the instant `g6.2xlarge` (the cheapest of the three) is in `Overrides`, 100% of launches go to g6. There is no "in the list but not preferred" state under `lowest-price`. Adding g6 before validating L4 would send all traffic to a potentially-incompatible type. Phase 1 ships without g6; Phase 2 adds it once the bake + a real g6 extraction confirm compatibility.

### Multi-AZ

- Change `VPCZoneIdentifier` from the single `BackendSubnetId` to a **list of subnets across all five AZs** (introduce a `BackendSubnetIds` `List<AWS::EC2::Subnet::Id>` parameter, defaulting to the five production subnets). This gives the allocation strategy room to dodge a constrained AZ. (The group's `MaxSize` stays 1, so there is no meaningful AZ-rebalance churn.)

### Terminate-on-idle / no warm pool

- Default `BackendWarmPoolMinSize: 0`, disabling the warm pool. The warm-pool CloudFormation resource stays behind its existing `EnableBackendWarmPool` condition so it can be re-enabled deliberately, but it is **off by default**.
- No new proxy code is required. `ec2_manager.stop_instance()` already calls `set_desired_capacity(0)`; with no warm pool, that scale-in **terminates** the instance, and the next wake launches fresh via `lowest-price`. This is the entire "always try cheapest / don't stay stuck on expensive" mechanism — achieved by removing the sticky warm pool once boot is fast, not by inspecting instance types at runtime.

### Startup timeout (N3)

- Raise `startup-timeout-minutes` from 30 to **~40–45**. Fast-boot makes this irrelevant on the happy path, but the **fallback** path (§4B) is still the ~22–25 min full bootstrap, and the incident showed 30 min is close enough to trigger a force-replace loop. The larger margin is cheap insurance for the fallback path, decoupled from making fast-boot reliable.

---

## 6. Component C — AMI & image-tag wiring, lifecycle

- New SSM String parameters, created by the stack with seed values:
  - **`/pdfx/backend-ami`** — the AMI id to launch.
  - **`/pdfx/backend-image-tag`** — the immutable backend image tag the baked AMI was built against (and that the boot path pulls/verifies). This replaces the launch template's use of `:latest`, so the AMI and the image it runs are always a **matched pair**.
- The launch template sets `ImageId: resolve:ssm:/pdfx/backend-ami`. EC2 resolves this **at instance launch**, so a rebake that updates the parameter is picked up on the next wake with **no CloudFormation deploy and no launch-template version bump** — in steady state. (The *initial* switch to `resolve:ssm` is itself one new LT version; see §8.)
- The bake publishes the new AMI id and immutable image tag to these parameters as its final step, **only after a successful `RegisterImage`**, so the ASG never resolves an empty/invalid value.
- **Lifecycle (N1):** the bake deregisters older baked AMIs and deletes their snapshots, keeping the most recent N (default 3), selected by the `Role=backend-baked` tag. The prune **must never** deregister the AMI currently referenced by `/pdfx/backend-ami` or one just published by a concurrent run.

---

## 7. Component D — CI auto-rebake

**Location:** `.github/workflows/main-build-and-deploy.yml`

- Add a job `bake-backend-ami` with `needs: deploy-prod` (runs after the backend image is built, pushed, and promoted).
- The job installs Packer, assumes the existing OIDC role, and runs the bake against the **immutable** backend image tag just published, then updates `/pdfx/backend-ami` and `/pdfx/backend-image-tag` and prunes old AMIs.
- **Concurrency (N1):** the job carries its own `concurrency: group` so two near-simultaneous backend merges cannot race the AMI prune or the SSM updates; the parameters must end on the newest AMI.
- **Change-gated:** the job runs only when backend-affecting paths changed in the merged PR — `deploy/Dockerfile.gpu`, `requirements.txt`, `app/**`, `celery_app.py`, `config.py`, `deploy/docker-compose*.yml`, and model-related config. Proxy-only or docs-only PRs skip the ~15–20 min GPU bake. The gate uses a path filter on the merge diff; if detection is inconclusive, default to **running** the bake (safe over stale).
- **IAM (B4):** the OIDC role (`GH_ACTIONS_AWS_ROLE`) needs the permissions Packer's `amazon-ebs` builder actually uses, enumerated from AWS's published minimal Packer policy — including temporary key pair and security group create/authorize/revoke (or reuse of existing ones), `ec2:Describe{Images,Instances,InstanceStatus,Subnets,Vpcs,SecurityGroups,Snapshots,Regions,ImageAttribute}`, `RunInstances`/`StopInstances`/`TerminateInstances`, `CreateImage`/`RegisterImage`/`DeregisterImage`/`CreateSnapshot`/`DeleteSnapshot`, `CreateTags` on the image and snapshot, and — because the build instance authenticates to ECR via an instance profile — `iam:PassRole` plus `ec2:AssociateIamInstanceProfile`/`ReplaceIamInstanceProfileAssociation`. Plus `ssm:PutParameter` on `/pdfx/backend-ami` and `/pdfx/backend-image-tag`. Changes land in `deploy/aws/github-actions-deploy-policy.json` and are documented in `deploy/aws/ami/README.md`.
- **`resolve:ssm` permission dependency (B4):** a launch template whose `ImageId` is `resolve:ssm:...` requires the launching principal to hold `ssm:GetParameters`, and EC2 Auto Scaling **validates this with a RunInstances dry-run at `UpdateAutoScalingGroup` time**. The operator/role that applies the ASG change must therefore have `ssm:GetParameters` on `/pdfx/backend-ami` (the existing policy grants `/pdfx/*`, which covers it) — the README calls out verifying this end-to-end with a dry-run before the cutover, since a missing grant fails the ASG update, not just a launch.

---

## 8. Rollout & validation (human-driven; not performed by this work)

This work lands as an open PR only. The following are documented for the operator and **not executed** as part of the change:

1. **L4/Ada compatibility gate (Phase 2 prerequisite).** The bake runs on g6.2xlarge, so a successful bake is already a strong L4 signal. Before adding `g6.2xlarge` to the run mix (§5 Phase 2), also boot an instance from the baked AMI on g6.2xlarge and run a real extraction to confirm the CUDA 12.8 / PyTorch / Marker / Docling stack behaves correctly on L4 end-to-end. This is the follow-up tracked in Jira.
2. **Template drift & apply path (N4).** Production `pdfx-backend` is not cleanly CloudFormation-managed (template pins one subnet; prod spans five). **Prefer reconciling/importing the resources into the stack** over direct CLI edits, because direct edits widen the drift and complicate the eventual import. If a direct CLI/console apply is unavoidable, mirror the identical changes into `pdfx-stack.yaml` in the same PR. Note that the **initial** switch to `resolve:ssm` ImageId requires one new launch-template version (only subsequent rebakes avoid a version bump). Document the apply order: publish a baked AMI → set `/pdfx/backend-ami` + `/pdfx/backend-image-tag` → dry-run the `UpdateAutoScalingGroup` (verifies `ssm:GetParameters`) → apply the ASG change. Choosing and executing the path is left to the operator.
3. **Backout.** Re-enable the warm pool (`BackendWarmPoolMinSize: 1`) and point `/pdfx/backend-ami` back at a known-good AMI (or the DL base AMI, which triggers the boot fallback path) to restore prior behavior.

### Rejected alternatives (N5)

- **On-Demand Capacity Reservation (ODCR).** A single reserved g6.2xlarge in one AZ would guarantee the common-case wake with none of the multi-type juggling. **Rejected on cost:** you pay ~24/7 (~$400–500/mo) for a mostly-idle service, which undercuts the scale-to-zero design. It remains the obvious "buy your way out of capacity risk" lever if the multi-type approach ever proves insufficient.

### Residual failure mode (N6)

- If **all** eligible type × AZ pools are simultaneously exhausted (a regional GPU crunch), no instance launches and the proxy hits its startup timeout. The **S3 durable proxy queue + replay is the backstop**: queued jobs must survive a wake that never succeeds and re-drive on the next wake rather than being lost. Implementation must confirm this holds (it is existing proxy behavior; the plan will include a check).

---

## 9. Testing

- **Packer:** `packer validate` on the template; `provision.sh` is lint-clean (`shellcheck`) and shares its core with `deploy.sh`.
- **CloudFormation:** `aws cloudformation validate-template` and `cfn-lint` on `pdfx-stack.yaml`; confirm the `MixedInstancesPolicy` (Phase-1 override list), `resolve:ssm` ImageId, subnet list, and warm-pool condition render as intended.
- **Boot fast-path logic:** unit tests for the extracted `should_use_baked_fastpath` function — present+digest-match (fast path), present+digest-mismatch (fallback), and absent (fallback).
- **CI gate:** verify the path filter fires the bake for a backend-affecting change and skips it for a proxy/docs-only change; verify the concurrency guard.
- **End-to-end (manual, pre-cutover):** bake an AMI, launch from it, confirm sub-5-min ready and a successful extraction that exercises Marker **and** Docling **and** GROBID (proving the caches are truly baked) — including on g6.2xlarge for the Phase 2 compatibility gate.

---

## 10. Files added / changed

```
deploy/aws/ami/pdfx-backend.pkr.hcl          # NEW  Packer template (builds on g6.2xlarge)
deploy/aws/ami/provision.sh                  # NEW  bake provisioner (wraps deploy.sh) + sample extraction + cache asserts + hygiene
deploy/aws/ami/test-sample.pdf               # NEW  small PDF used to warm Docling/GROBID/rapidocr caches during bake
deploy/aws/ami/README.md                     # NEW  how to bake/redo, IAM, params, rollout/backout, resolve:ssm dry-run
deploy/aws/pdfx-stack.yaml                   # EDIT MixedInstancesPolicy (Phase-1 g5 only), multi-AZ subnets, /pdfx/backend-ami + /pdfx/backend-image-tag + resolve:ssm, warm pool default 0, startup-timeout 40-45, boot fast-path user-data (should_use_baked_fastpath)
deploy/aws/github-actions-deploy-policy.json # EDIT extra IAM for the bake job + PassRole + ssm:PutParameter
.github/workflows/main-build-and-deploy.yml  # EDIT change-gated, concurrency-guarded bake-backend-ami job
docs/superpowers/specs/2026-07-07-pdfx-capacity-cost-design.md  # NEW  this spec
```

---

## 11. Open questions / risks

- **L4 compatibility** is expected (CUDA 12.8 supports Ada) and the g6.2xlarge bake is a standing smoke test, but the run-mix addition of g6 waits on the §8 Phase 2 validation.
- **Bake duration & cost:** each auto-rebake launches a GPU instance for ~15–20 min (now a bit longer because of the sample extraction). The change-gate limits this to backend-affecting merges; the AMI lifecycle prune bounds storage.
- **Sample-extraction fidelity:** the bundled test PDF must exercise Docling and GROBID paths (not a trivial one-line PDF) so their caches actually populate; the cache-non-empty asserts (§4 step 5) are the guardrail.
- **`resolve:ssm` invariant:** `/pdfx/backend-ami` and `/pdfx/backend-image-tag` must always hold valid values; the bake updates them only after a successful `RegisterImage`, and the stack seeds them with valid values.
