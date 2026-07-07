# PDFX Backend: Capacity Resilience & Cost-Aware Cold Start

**Date:** 2026-07-07
**Status:** Approved design — pending implementation plan
**Scope:** `agr_pdf_extraction_service` GPU backend (ASG `pdfx-backend`), its AMI, and the CI that produces it. The Fargate proxy behavior is unchanged except where noted.

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
- CI (`.github/workflows/main-build-and-deploy.yml`) already assumes an AWS role via OIDC (`secrets.GH_ACTIONS_AWS_ROLE`) and builds/pushes `agr_pdfx_backend` from `deploy/Dockerfile.gpu` in the `deploy-prod` job.
- `deploy/deploy.sh` already exposes the knobs a bake needs: `GPU_MODE`, `PDFX_DEPLOY_BUILD_MODE=never`, `PDFX_DEPLOY_PULL_IMAGES=always|never`, `PDFX_PREWARM_MODELS=marker|off`.

---

## 2. Goals & non-goals

### Goals

1. **Capacity resilience** — survive a single-instance-type or single-AZ capacity shortage without a slow fallback or a replace loop.
2. **Fast cold start** — reduce time-to-ready from ~22 min to ~2–4 min by baking Docker images and ML models into the AMI.
3. **Cost-aware selection** — always prefer the cheapest eligible GPU type; only land on a pricier type when the cheaper pools are genuinely empty; do not stay "stuck" on an expensive type after a job finishes.
4. **Reproducible IaC** — the AMI build is defined in-repo and can be re-run / updated on demand and automatically.

### Non-goals

- No Spot instances (interruptions would fail in-flight extractions). On-demand only.
- No change to the consensus/extraction pipeline or the proxy's queueing/replay logic.
- No special-case "is this instance expensive?" runtime logic (see §5 — terminating every idle instance and re-selecting cheapest achieves the goal without it).
- This spec stops at an open PR. Applying anything to production is a separate, human-driven step (§8).

---

## 3. Behavior summary (target end state)

- On job arrival, the proxy scales the ASG 0→1 (unchanged). The ASG's **MixedInstancesPolicy** picks the cheapest available type across all AZs.
- The instance boots from a **pre-baked AMI**: Docker images and ML model caches are already on disk, so it only fetches secrets from SSM and starts the compose stack (~2–4 min).
- On idle, the proxy scales desired back to 0. With **no warm pool**, that terminates the instance.
- The next wake re-runs the `lowest-price` allocation from scratch — so the service re-attempts the cheapest type every time and never stays pinned to an expensive one.

---

## 4. Component A — Baked AMI via Packer

**Location:** `deploy/aws/ami/`

- `pdfx-backend.pkr.hcl` — Packer template using the `amazon-ebs` builder.
  - Source AMI: the Deep Learning Base OSS NVIDIA Driver GPU AMI (same family as prod today; parameterized).
  - Build instance type: **g6.2xlarge** (also serves as the L4/Ada compatibility check — see §8).
  - Region: us-east-1. Encrypted EBS. Root volume sized to hold images + model caches (≥ current 200 GiB unless measured smaller).
  - Variables: backend image repo/tag to bake, base AMI id, retention count.
- `provision.sh` — thin wrapper invoked by Packer that **reuses the production boot path** so bake and boot cannot diverge:
  1. Install Docker, the compose plugin, and confirm the NVIDIA container runtime (same steps as the launch-template bootstrap).
  2. `git clone`/checkout the target ref.
  3. ECR login and run `deploy.sh` with `GPU_MODE=on PDFX_DEPLOY_BUILD_MODE=never PDFX_DEPLOY_PULL_IMAGES=always PDFX_PREWARM_MODELS=marker`, which pulls the target backend image tag plus all compose images (postgres, redis, grobid, worker, app, nginx) and prewarms Marker/Docling/rapidocr model caches into the persistent model dirs.
  4. Bring the stack down (`docker compose down`) but leave images and model caches on disk.
- **Bake hygiene (security-critical):** the bake **must not** fetch the real `/pdfx/backend-env` SecretString and **must not** leave any secret or host identity in the image. Before snapshot:
  - Do not write the production `.env`; use a minimal dummy env sufficient for model prewarm only.
  - Remove ECR docker credentials (`~/.docker/config.json`), cloud-init state, SSH host keys, `/etc/machine-id`, bootstrap/deploy logs, shell history, and any `.env`.
  - Secrets remain in SSM and are fetched at **boot**, exactly as today.
- **Marker file:** write `/opt/pdfx/baked.json` recording base AMI id, backend image tag/SHA, a model manifest (which model dirs/versions were prewarmed), and a UTC timestamp. Used by the boot fast-path (§4B) and for observability.
- **Output:** an AMI tagged `Project=pdfx`, `Role=backend-baked`, with the backend image tag; the build publishes the AMI id to SSM (§6).

### 4B. Boot fast-path (launch-template user-data change)

The launch-template user-data (in `pdfx-stack.yaml`) gains a branch at the top of the bootstrap:

- If `/opt/pdfx/baked.json` exists **and** its recorded backend image tag matches the desired tag: run `deploy.sh` with `PDFX_DEPLOY_PULL_IMAGES=never PDFX_PREWARM_MODELS=off` — images and models are already present, so this is a fast `compose up`. Secrets are still fetched from SSM at boot.
- Otherwise (marker missing or tag mismatch): **fall back to today's full pull+prewarm path.** A stale or missing AMI therefore degrades to current behavior rather than failing — this is the primary safety net.

---

## 5. Component B — MixedInstancesPolicy, multi-AZ, and terminate-on-idle

**Location:** `deploy/aws/pdfx-stack.yaml`

### MixedInstancesPolicy

Replace the ASG's single `LaunchTemplate` with a `MixedInstancesPolicy`:

- `LaunchTemplate.Overrides`: **g6.2xlarge, g5.2xlarge, g5.4xlarge** (optionally `g6e.2xlarge` last). All are single-GPU, 24 GB-class (g6e is 48 GB), interchangeable for this workload.
- `InstancesDistribution`:
  - `OnDemandBaseCapacity: 0`, `OnDemandPercentageAboveBaseCapacity: 100` (100% on-demand, no Spot).
  - `OnDemandAllocationStrategy: lowest-price` — picks the cheapest type first (g6.2xlarge is currently the cheapest and typically most available), falling back to the next only on capacity errors.

### Multi-AZ

- Change `VPCZoneIdentifier` from the single `BackendSubnetId` to a **list of subnets across all five AZs** (introduce a `BackendSubnetIds` `List<AWS::EC2::Subnet::Id>` parameter, defaulting to the five production subnets). This gives the allocation strategy room to dodge a constrained AZ.

### Terminate-on-idle / no warm pool

- Default `BackendWarmPoolMinSize: 0`, disabling the warm pool. The warm-pool CloudFormation resource stays behind its existing `EnableBackendWarmPool` condition so it can be re-enabled deliberately, but it is **off by default**.
- No new proxy code is required. The proxy already scales desired capacity to 0 on idle; with no warm pool, that scale-in **terminates** the instance, and the next wake launches fresh via `lowest-price`. This is the entire "always try cheapest / don't stay stuck on expensive" mechanism — achieved by removing the sticky warm pool once boot is fast, not by inspecting instance types at runtime.
- Because boot is now fast, the 30-min `startup-timeout-minutes` is comfortable margin rather than a race. Leave it as-is for now (revisit only if fast-boot proves reliably sub-5-min in practice).

---

## 6. Component C — AMI wiring & lifecycle

- New SSM String parameter **`/pdfx/backend-ami`**, created by the stack with a seed value (current DL base AMI or first baked AMI).
- The launch template sets `ImageId: resolve:ssm:/pdfx/backend-ami`. EC2 resolves this **at instance launch**, so a rebake that updates the parameter is picked up on the next wake with **no CloudFormation deploy and no launch-template version bump**.
- The bake publishes the new AMI id to `/pdfx/backend-ami` as its final step.
- **Lifecycle:** the bake deregisters older baked AMIs and deletes their snapshots, keeping the most recent N (default 3), so repeated auto-rebakes don't accumulate cost. Selection is by the `Role=backend-baked` tag.

---

## 7. Component D — CI auto-rebake

**Location:** `.github/workflows/main-build-and-deploy.yml`

- Add a job `bake-backend-ami` with `needs: deploy-prod` (runs after the backend image is built, pushed, and promoted to `:latest`).
- The job installs Packer, assumes the existing OIDC role, and runs the bake against the just-published backend image tag, then updates `/pdfx/backend-ami` and prunes old AMIs.
- **Change-gated:** the job runs only when backend-affecting paths changed in the merged PR — `deploy/Dockerfile.gpu`, `requirements.txt`, `app/**`, `celery_app.py`, `config.py`, `deploy/docker-compose*.yml`, and model-related config. Proxy-only or docs-only PRs skip the ~15–20 min GPU bake. The gate uses a path filter on the merge diff; if detection is inconclusive, default to **running** the bake (safe over stale).
- **IAM:** the OIDC role (`GH_ACTIONS_AWS_ROLE`) needs additional permissions for the bake: `ec2:RunInstances`/`CreateTags`/`CreateImage`/`RegisterImage`/`DeregisterImage`/`CreateSnapshot`/`DeleteSnapshot`/`DescribeImages`/`DescribeInstances`/`StopInstances`/`TerminateInstances` (scoped to the bake, ideally via a tag/instance-profile condition) and `ssm:PutParameter` on `/pdfx/backend-ami`. These additions are documented in `deploy/aws/ami/README.md` and reflected in `deploy/aws/github-actions-deploy-policy.json`.

---

## 8. Rollout & validation (human-driven; not performed by this work)

This work lands as an open PR only. The following are documented for the operator and **not executed** as part of the change:

1. **L4/Ada compatibility gate.** The first bake runs on g6.2xlarge; before g6 enters the prod instance mix, run a real test extraction on a g6.2xlarge booted from the baked AMI to confirm the CUDA 12.8 image and PyTorch/Marker/Docling stack run correctly on L4. Only after that passes should `lowest-price` be allowed to prefer g6.
2. **Template drift.** Production `pdfx-backend` is not cleanly CloudFormation-managed (template pins one subnet; prod spans five). The README documents the two apply options — reconcile/import into the stack, or apply the MixedInstancesPolicy + subnet + warm-pool changes directly to the live ASG via CLI/console — and the order (publish a baked AMI → set `/pdfx/backend-ami` → update the ASG). Choosing and executing the apply path is left to the operator.
3. **Backout.** Re-enabling the warm pool (`BackendWarmPoolMinSize: 1`) and pointing `/pdfx/backend-ami` back at a known-good AMI (or the DL base AMI, which triggers the boot fallback path) restores prior behavior.

---

## 9. Testing

- **Packer:** `packer validate` on the template; `provision.sh` is lint-clean (`shellcheck`) and shares its core with `deploy.sh`.
- **CloudFormation:** `aws cloudformation validate-template` and `cfn-lint` on `pdfx-stack.yaml`; confirm the `MixedInstancesPolicy`, `resolve:ssm` ImageId, subnet list, and warm-pool condition render as intended.
- **User-data marker logic:** a small, extracted, unit-testable shell function for the "baked marker present & tag matches?" decision, with tests for present-match (fast path), present-mismatch (fallback), and absent (fallback).
- **CI gate:** verify the path filter fires the bake for a backend-affecting change and skips it for a proxy/docs-only change.
- **End-to-end (manual, pre-cutover):** bake an AMI, launch an instance from it, confirm sub-5-min ready and a successful extraction — including on g6.2xlarge for the compatibility gate.

---

## 10. Files added / changed

```
deploy/aws/ami/pdfx-backend.pkr.hcl          # NEW  Packer template
deploy/aws/ami/provision.sh                  # NEW  bake provisioner (wraps deploy.sh) + hygiene
deploy/aws/ami/README.md                     # NEW  how to bake/redo, IAM, params, rollout/backout
deploy/aws/pdfx-stack.yaml                   # EDIT MixedInstancesPolicy, multi-AZ subnets, /pdfx/backend-ami + resolve:ssm, warm pool default 0, boot fast-path user-data
deploy/aws/github-actions-deploy-policy.json # EDIT extra IAM for the bake job
.github/workflows/main-build-and-deploy.yml  # EDIT change-gated bake-backend-ami job
docs/superpowers/specs/2026-07-07-pdfx-capacity-cost-design.md  # NEW  this spec
```

---

## 11. Open questions / risks

- **L4 compatibility** is expected (CUDA 12.8 supports Ada) but must be validated by the §8 gate before g6 is preferred in prod.
- **Bake duration & cost:** each auto-rebake launches a GPU instance for ~15–20 min. The change-gate limits this to backend-affecting merges; the AMI lifecycle prune bounds storage.
- **Model-cache paths:** the exact set of directories to bake (`data/models`, `data/model_cache`, `data/rapidocr_models`, and any Marker/HuggingFace cache under the worker) must be confirmed against `docker-compose.gpu.prebuilt.yml` volume mounts during implementation so the prewarmed caches actually persist into the AMI and are found at boot.
- **`resolve:ssm` ImageId** requires the SSM parameter to always hold a valid AMI id; the bake updates it only after a successful `RegisterImage`, and the seed value is a valid AMI, so the ASG never resolves an empty/invalid parameter.
