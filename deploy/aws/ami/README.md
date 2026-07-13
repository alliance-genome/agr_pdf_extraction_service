# PDFX Backend Baked AMI — Operator Guide

How to bake the PDFX GPU-backend AMI, seed and maintain its SSM parameters, verify
the `resolve:ssm` permission chain, roll the change out to production, and back it
out. This is the runbook for whoever applies this work to prod.

> **Design docs**
> - Spec: [`docs/superpowers/specs/2026-07-07-pdfx-capacity-cost-design.md`](../../../docs/superpowers/specs/2026-07-07-pdfx-capacity-cost-design.md)
> - Plan: [`docs/superpowers/plans/2026-07-07-pdfx-capacity-cost-resilience.md`](../../../docs/superpowers/plans/2026-07-07-pdfx-capacity-cost-resilience.md)

## What this is

The backend previously launched a single `g5.2xlarge` on-demand instance behind a
sticky, single-AZ warm pool and cold-started from scratch (~22 min: install Docker,
`git clone`, pull the ECR image, `deploy.sh` + model prewarm). A GPU-capacity
shortage in the warm pool's AZ cascaded into a force-replace loop and a curator-facing
outage. This design replaces that with two pieces:

- **A pre-baked AMI** (this directory, built by Packer). It ships the Docker images,
  **all** ML model caches (Marker, Docling, rapidocr, GROBID), and the RDS CA bundle
  already on disk. To avoid the slow first-touch EBS-snapshot lazy-load (reading the
  baked ~17 GB off a cold volume is ~9 MB/s → ~12 min), the launch template sets an
  **EBS `VolumeInitializationRate`** (provisioned-init, default 300 MiB/s) so the root
  volume hydrates from its snapshot at boot; a fresh instance then fetches secrets from
  SSM and runs `compose up` (~4–5 min). Boot uses `PDFX_DEPLOY_PULL_IMAGES=never`
  `PDFX_PREWARM_MODELS=off` — no digest fast-path, no fallback path.
- **An external metadata DB (RDS Postgres)** instead of a bundled `postgres` container:
  moves the single `extraction_run` table off the GPU cold-start path (no local DB to
  boot/migrate on every wake) and makes it durable across scale-to-zero. Wired via
  `deploy.sh PDFX_EXTERNAL_DB=true` + `deploy/docker-compose.external-db.yml`. Runtime
  app/worker connect as a **least-privilege** app role (`/pdfx/database-url`); `alembic`
  migrations run as the master role (`/pdfx/database-url-migrate`); both use TLS
  `sslmode=verify-full` with the baked `/opt/pdfx/rds-ca.pem`.
- **A `MixedInstancesPolicy`** on the ASG (`deploy/aws/pdfx-stack.yaml`): on-demand,
  `lowest-price`, `g5.2xlarge` + `g5.4xlarge` across all five AZs, warm pool off by
  default. Idle scale-in terminates the instance, so each wake re-selects the cheapest
  eligible type and never stays pinned to an expensive one.

### Files in this directory

| File | Purpose |
|------|---------|
| `pdfx-backend.pkr.hcl` | Packer `amazon-ebs` template. Builds on `g6.2xlarge` (standing L4/Ada smoke test). |
| `provision.sh` | Bake provisioner: reuses `deploy.sh`, runs one real sample extraction to warm all caches, asserts caches non-empty, bakes the RDS CA bundle to `/opt/pdfx/rds-ca.pem`, scrubs secrets/identity before snapshot. |
| `test-sample.pdf` | Small multi-element PDF (heading + paragraph + table) used to warm Docling/GROBID/rapidocr caches during the bake. |
| `prune_amis.sh` | Deregisters old `Role=backend-baked` AMIs (keep last `KEEP_N`, default 3); never touches the AMI referenced by `/pdfx/backend-ami`. |
| `tests/` | Shell unit tests for the prune selection logic. |

Run the unit tests any time:

```bash
bash deploy/aws/ami/tests/test_prune_amis.sh
```

## Manual bake

CI bakes automatically on backend-affecting merges inside the environment-gated
`deploy-prod` job in `.github/workflows/main-build-and-deploy.yml`. Bake by hand only for out-of-band
recovery or to produce a known-good AMI for backout.

Requires an AWS session (env credentials or `AWS_PROFILE=ctabone`) for a principal
holding the bake IAM below, plus a local Packer install. Packer uses the **ambient**
AWS credentials — there is no `--profile` flag on `packer build`; export
`AWS_PROFILE=ctabone` (or assume the CI role) first.

```bash
cd deploy/aws/ami
packer init pdfx-backend.pkr.hcl
packer build \
  -var region=us-east-1 \
  -var base_ami_id=ami-00c6ddd550364d6c3 \
  -var backend_image_repo=100225593120.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_backend \
  -var backend_image_tag=<immutable-merge-sha> \
  -var iam_instance_profile=pdfx-ec2-profile \
  -var subnet_id=subnet-81c95ee4 \
  pdfx-backend.pkr.hcl
```

- `backend_image_tag` is the **immutable merge SHA** — never `latest`. The baked AMI
  and the image it runs are a matched pair.
- Optional vars with defaults: `build_instance_type=g6.2xlarge` (the bake doubles as
  the standing L4 smoke test — it fails if the stack can't run on L4), and
  `root_volume_size=200`.
- Packer allows up to two hours for AWS to register the encrypted 200 GB AMI
  snapshot. The GPU build instance is already stopped during this wait; do not
  cancel a healthy bake merely because snapshot progress advances slowly.
- The build writes `manifest.json`; read the new AMI id with:

  ```bash
  jq -r '.builds[-1].artifact_id' manifest.json | cut -d':' -f2
  ```

To only validate the template (no instance launched, no AMI produced):

```bash
packer validate \
  -var base_ami_id=ami-00c6ddd550364d6c3 \
  -var backend_image_repo=100225593120.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_backend \
  -var backend_image_tag=validate-only \
  -var iam_instance_profile=pdfx-ec2-profile \
  -var subnet_id=subnet-81c95ee4 \
  pdfx-backend.pkr.hcl
```

A manual bake does **not** publish to SSM (only CI's `deploy-prod` release job runs the
`aws ssm put-parameter` step). After a manual bake, set the SSM params yourself with
the `put-parameter` commands below.

CI defaults its temporary Packer instance to `subnet-81c95ee4` (`us-east-1c`). For a
capacity error in that AZ, manually dispatch **Main Build and Deploy** with
`force_backend_bake=true` and set `backend_build_subnet_id` to another approved public
PDFX subnet. The override changes only the disposable build box; it does not change the
production backend ASG subnets.

## SSM parameters

Parameters that wire the AMI, image, and DB to the launch template / boot:

- **`/pdfx/backend-ami`** — the AMI id the launch template resolves at instance launch
  (`ImageId: resolve:ssm:/pdfx/backend-ami`). **Must be created with
  `--data-type aws:ec2:image`** — the LT `resolve:ssm` `ImageId` rejects a plain-String
  parameter (`SsmInvalidParameter: … supported: aws:ec2:image`).
- **`/pdfx/backend-image-tag`** — a mirrored release record and legacy bootstrap
  fallback. New baked AMIs carry the authoritative immutable tag in
  `/opt/pdfx/backend-image-tag`, so the image and its tag cannot be split by two
  independent SSM reads.
- **`/pdfx/database-url`** (SecureString) — runtime DB URL (least-priv `pdfx_app` role,
  `sslmode=verify-full`); consumed by the compose override.
- **`/pdfx/database-url-migrate`** (SecureString) — migration DB URL (master role);
  used only by `deploy.sh`'s `alembic upgrade head`, never by the runtime containers.

### One-time seed (before first use)

These values are mutated by the **bake**, not by CloudFormation. They are intentionally
**not** declared as `AWS::SSM::Parameter` resources with a `Value:` in
`pdfx-stack.yaml`: a managed `Value:` is reasserted on **every** stack update, which
would silently revert the live baked AMI/tag back to the template's seed and drop every
wake to the slow full-bootstrap fallback (design-review B3). Seed them once, out of
band, before the first ASG scale-out:

```bash
aws ssm put-parameter --profile ctabone --region us-east-1 --overwrite \
  --name /pdfx/backend-ami --value <baked-ami-id> --type String --data-type aws:ec2:image
aws ssm put-parameter --profile ctabone --region us-east-1 --overwrite \
  --name /pdfx/backend-image-tag --type String --value <immutable-tag>
```

`resolve:ssm` fails a launch if the AMI parameter is empty/invalid. Until the
launch-template bootstrap with baked-marker support is applied, both release parameters
must hold valid, matched values. The boot path has **no pull+prewarm fallback** — it
uses the baked images/models directly — so `/pdfx/backend-ami` must point at a **real
baked AMI** (not the DL base) and the RDS URLs must be seeded.

### Written by the bake

CI's environment-gated backend release path records the previous pair, overwrites both
parameters only after the image is available, re-reads and verifies the exact requested
values, and restores the previous pair if publication fails. The baked marker is the
runtime source of truth after the corresponding launch-template bootstrap is applied;
the SSM image tag remains useful for release visibility and legacy fallback. The prune
step keeps the newest 3 baked AMIs and never deregisters the AMI referenced by
`/pdfx/backend-ami`.

> **Clobber warning.** A CloudFormation redeploy must **never** re-introduce
> `/pdfx/backend-ami` or `/pdfx/backend-image-tag` as managed-`Value:` resources.
> Doing so re-asserts a stale value on every stack update and reverts the live baked
> AMI/tag. The stack may only *read* them via `resolve:ssm`.

Verify the current values any time:

```bash
aws ssm get-parameters --profile ctabone --region us-east-1 \
  --names /pdfx/backend-ami /pdfx/backend-image-tag \
  --query 'Parameters[].[Name,Value]' --output text
```

## IAM

Bake permissions live in `deploy/aws/github-actions-deploy-policy.json` (the CI OIDC
role, `secrets.GH_ACTIONS_AWS_ROLE`). The bake-specific grants are:

- **`PackerBakeEc2`** — the EC2 actions Packer's `amazon-ebs` builder uses: run/stop/
  terminate instances; create/register/deregister image; create/delete snapshot;
  create tags; temporary key pair and security group create/authorize/revoke;
  `AssociateIamInstanceProfile` / `ReplaceIamInstanceProfileAssociation`; and the
  `ec2:Describe*` reads (`Images`, `Instances`, `InstanceStatus`, `Subnets`, `Vpcs`,
  `SecurityGroups`, `Snapshots`, `Regions`, `ImageAttribute`, `KeyPairs`, `Tags`).
- **`ReadBackendInstanceProfile`** — `iam:GetInstanceProfile` restricted to
  `arn:aws:iam::100225593120:instance-profile/pdfx-ec2-profile`, which Packer validates
  before it launches the temporary build box.
- **`PassBackendInstanceProfileRole`** — `iam:PassRole` on
  `arn:aws:iam::100225593120:role/pdfx-ec2-role`, conditioned on
  `iam:PassedToService = ec2.amazonaws.com`, so the build box can be launched with an
  instance profile.

SSM read (`ssm:GetParameters` on `/pdfx/*`, statement `SsmReadPdfxParams`) and write
(`ssm:PutParameter` on `/pdfx/*`, statement `SsmPutPdfxPlaceholders`) already existed —
they cover both the `resolve:ssm` resolution and the bake's SSM updates, so no new SSM
grant was needed.

**Reused instance profile.** The bake box authenticates to ECR using the **existing**
`pdfx-ec2-profile` (role `pdfx-ec2-role`, both defined in `pdfx-stack.yaml`) — passed via
the `-var iam_instance_profile=pdfx-ec2-profile` above. That role already grants
`ecr:GetAuthorizationToken` plus pull on `agr_pdfx_backend`, which is exactly what the
build box needs. No new instance profile or role is created; no static credentials are
baked into the AMI (`provision.sh` scrubs `~/.docker/config.json`, `~/.aws`, host keys,
`/etc/machine-id`, cloud-init state, and shell history before snapshot).

## Apply path (production is drifted from the template)

Production `pdfx-backend` is not cleanly CloudFormation-managed — the template
historically pinned a single subnet while prod spans five AZs. **Prefer reconciling /
importing the resources into the stack** over direct CLI/console edits, because direct
edits widen the drift and complicate the eventual import. If a direct apply is
unavoidable, mirror the identical changes into `pdfx-stack.yaml` in the same PR.

Apply in this order:

1. **Reconcile the launch-template bootstrap** so it reads the baked
   `/opt/pdfx/backend-image-tag` marker, with SSM retained as the legacy fallback.
2. **Publish a baked AMI** (manual bake above, or let CI's backend release path run).
3. **Set both SSM params** — `/pdfx/backend-ami` to the new AMI id and
   `/pdfx/backend-image-tag` to its immutable tag (CI does this; verify with the
   `get-parameters` command above).
4. **Dry-run the `resolve:ssm` permission chain BEFORE cutover** (see below).
5. **Apply the ASG change** (`UpdateAutoScalingGroup`, or the stack update). The
   **initial** switch to a `resolve:ssm` `ImageId` creates **one new launch-template
   version**; subsequent rebakes only mutate the SSM value and are picked up at the
   next launch with **no** LT version bump and no CloudFormation deploy.

### `resolve:ssm` dry-run check

A launch template whose `ImageId` is `resolve:ssm:...` requires the launching principal
to hold `ssm:GetParameters` on that parameter. EC2 Auto Scaling validates this with an
internal **RunInstances dry-run at `UpdateAutoScalingGroup` time** — so a missing grant
fails the whole ASG update, not just an individual launch. Verify the chain end-to-end
**before** the cutover, using the same RunInstances dry-run EC2 performs:

```bash
# Same validation EC2 Auto Scaling runs internally at UpdateAutoScalingGroup time.
aws ec2 run-instances --dry-run --profile ctabone --region us-east-1 \
  --launch-template LaunchTemplateName=pdfx-backend,Version='$Latest' \
  --instance-type g5.2xlarge

# And confirm the applying principal can read the parameter directly.
aws ssm get-parameters --profile ctabone --region us-east-1 \
  --names /pdfx/backend-ami --query 'Parameters[].Value' --output text
```

- `DryRunOperation` (`Request would have succeeded, but DryRun flag is set`) means the
  `resolve:ssm` `ImageId` resolved and the credentials are sufficient — safe to apply.
- `UnauthorizedOperation` citing `ssm:GetParameters` means the applying principal is
  missing the SSM read grant. Fix that first — otherwise the real
  `UpdateAutoScalingGroup` will fail. (The CI role already has `ssm:GetParameters` on
  `/pdfx/*`; confirm whichever principal actually applies the ASG change does too.)

## Backout

Restore prior behavior with two independent levers (either or both):

1. **Re-enable the warm pool.** Set `BackendWarmPoolMinSize=1`. This flips the
   `EnableBackendWarmPool` condition (`!Not [!Equals [BackendWarmPoolMinSize, 0]]`),
   recreating the `AWS::AutoScaling::WarmPool` resource.
2. **Point the AMI at a known-good build.** Overwrite `/pdfx/backend-ami` with a
   previously-good baked AMI id. The boot has no base-AMI fallback, so it must be a real
   baked AMI:

   ```bash
   aws ssm put-parameter --profile ctabone --region us-east-1 --overwrite \
     --name /pdfx/backend-ami --value <known-good-baked-ami-id> --type String --data-type aws:ec2:image
   ```

Because `/pdfx/backend-ami` is resolved at launch, the next wake picks up the change
with no launch-template version bump and no CloudFormation deploy.

## Phase 2 — g6.2xlarge added (KANBAN-1411) ✅

`g6.2xlarge` (L4/Ada) is now in the ASG `MixedInstancesPolicy.Overrides`. Because the
allocation strategy is `lowest-price` and g6.2xlarge is the cheapest of the three, it is
now the preferred type with g5.2xlarge / g5.4xlarge as capacity fallback — more capacity
resilience **and** lower cost.

L4 was validated end-to-end before adding it: green g6.2xlarge bakes plus multiple
g6.2xlarge boots from the baked AMI running real extractions (Marker + Docling + GROBID)
on the CUDA 12.8 / PyTorch stack. Current Overrides:

```yaml
          Overrides:
            - InstanceType: g5.2xlarge
            - InstanceType: g5.4xlarge
            - InstanceType: g6.2xlarge   # KANBAN-1411: preferred (cheapest), L4-validated
```
