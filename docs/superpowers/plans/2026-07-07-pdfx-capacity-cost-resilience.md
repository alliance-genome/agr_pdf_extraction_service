# PDFX Backend Capacity Resilience & Cost-Aware Cold Start — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the PDFX GPU backend survive GPU-capacity shortages and cold-start faster by baking a ready-to-run AMI, launching from a multi-type/multi-AZ on-demand instance mix, and terminating idle instances so each wake re-selects the cheapest type.

**Architecture:** A Packer bake produces an AMI with all Docker images and ML model caches on disk; its id + the immutable backend image tag are published to SSM. The CloudFormation launch template resolves the AMI from SSM at launch and boots via a fast-path that skips image pulls/model prewarm when the baked marker's image digest matches. The ASG uses a MixedInstancesPolicy (on-demand, `lowest-price`, g5.2xlarge + g5.4xlarge across all AZs) with no warm pool, so idle scale-in terminates and the next wake re-runs allocation. CI auto-rebakes on backend-affecting merges.

**Tech Stack:** Packer (`amazon-ebs`), AWS CloudFormation, AWS Auto Scaling MixedInstancesPolicy, GitHub Actions (OIDC), Bash, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-07-07-pdfx-capacity-cost-design.md`

## Global Constraints

- **Branch:** work on `pdfx-capacity-cost-resilience`. This is a secured repo: **never** use `git commit --no-verify` or `git push --no-verify`; if a hook fails, stop and ask.
- **On-demand only.** No Spot anywhere.
- **Phase 1 instance mix is `g5.2xlarge` + `g5.4xlarge` only.** Do **not** add `g6.2xlarge` in this plan — it is deliberately deferred to KANBAN-1411 until L4 is validated, because `lowest-price` cannot keep a cheaper type deprioritized.
- **No secrets in the AMI.** The bake never fetches `/pdfx/backend-env`; it scrubs all creds/identity before snapshot. Secrets stay in SSM and are fetched at boot.
- **Region:** `us-east-1`. **Account:** `100225593120`. **SSM prefix:** `/pdfx`.
- **AMI/image are a matched pair:** boot pins the immutable backend image tag from `/pdfx/backend-image-tag`; the fast-path compares the image **digest**, never `:latest`.
- **AMI prune keeps the last 3** baked AMIs and must never deregister the AMI referenced by `/pdfx/backend-ami` or one just published.
- **Change-gate defaults to running the bake** when backend-vs-other detection is inconclusive.
- Match existing file conventions in `deploy/` (bash `set -euo pipefail`, `#!/usr/bin/env bash`).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `deploy/aws/ami/lib/baked_fastpath.sh` (NEW) | Pure function `should_use_baked_fastpath` deciding fast vs fallback boot; baked into AMI at `/opt/pdfx/`. |
| `deploy/aws/ami/tests/test_baked_fastpath.sh` (NEW) | Unit tests for the fast-path function (stubbed `aws`). |
| `deploy/aws/ami/prune_amis.sh` (NEW) | Deregister old baked AMIs keeping last N, never the referenced one. |
| `deploy/aws/ami/tests/test_prune_amis.sh` (NEW) | Unit tests for prune selection (stubbed `aws`). |
| `deploy/aws/ami/provision.sh` (NEW) | Packer provisioner: wraps `deploy.sh`, runs a sample extraction, asserts caches, hygiene. |
| `deploy/aws/ami/test-sample.pdf` (NEW) | Small multi-element PDF used to warm Docling/GROBID/rapidocr caches during bake. |
| `deploy/aws/ami/pdfx-backend.pkr.hcl` (NEW) | Packer template (`amazon-ebs`, builds on g6.2xlarge). |
| `deploy/aws/ami/README.md` (NEW) | Operator docs: how to bake, IAM, params, rollout, backout. |
| `deploy/aws/pdfx-stack.yaml` (EDIT) | SSM params, `resolve:ssm` AMI, image-tag pinning, boot fast-path user-data, MixedInstancesPolicy, multi-AZ subnets, warm-pool default 0, startup timeout. |
| `deploy/aws/github-actions-deploy-policy.json` (EDIT) | Add EC2/Packer + PassRole IAM for the bake. |
| `.github/workflows/main-build-and-deploy.yml` (EDIT) | Add change-gated, concurrency-guarded `bake-backend-ami` job. |

**Task order & dependencies:** Tasks 1–2 are self-contained scripts with unit tests. Task 3 (provision) + Task 4 (Packer) build the bake. Tasks 5–6 change CloudFormation. Task 7 (IAM) + Task 8 (CI) wire the auto-rebake. Task 9 documents. Each task ends green and committed.

---

### Task 1: Boot fast-path decision function

Decide at boot whether the baked AMI is usable (fast `compose up`) or stale/missing (full pull+prewarm fallback), based on the marker's recorded image digest vs. the digest of the tag to be run.

**Files:**
- Create: `deploy/aws/ami/lib/baked_fastpath.sh`
- Test: `deploy/aws/ami/tests/test_baked_fastpath.sh`

**Interfaces:**
- Produces: `should_use_baked_fastpath <marker_path> <image_repo> <image_tag>` → exit `0` (use fast path) or `1` (fall back). Reads the running image digest via `aws ecr describe-images`. `baked.json` schema: `{"backend_image_repo": "...", "backend_image_tag": "...", "backend_image_digest": "sha256:...", "base_ami_id": "...", "baked_at": "..."}`.

- [ ] **Step 1: Write the failing test**

```bash
# deploy/aws/ami/tests/test_baked_fastpath.sh
#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../lib/baked_fastpath.sh"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fail=0
check() { if [ "$1" = "$2" ]; then echo "ok: $3"; else echo "FAIL: $3 (got '$1' want '$2')"; fail=1; fi; }

# Stub `aws ecr describe-images` to return a fixed digest for tag sha-AAA.
aws() {
  if [ "$1" = "ecr" ] && [ "$2" = "describe-images" ]; then
    echo "sha256:DDDAAA"; return 0
  fi
  return 1
}

# Case 1: marker present, digest matches -> fast path (0)
cat > "$TMP/baked.json" <<'JSON'
{"backend_image_repo":"acct.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_backend","backend_image_tag":"sha-AAA","backend_image_digest":"sha256:DDDAAA","base_ami_id":"ami-x","baked_at":"2026-07-07T00:00:00Z"}
JSON
should_use_baked_fastpath "$TMP/baked.json" "acct.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_backend" "sha-AAA"; check "$?" "0" "present+digest-match -> fast path"

# Case 2: marker present, digest mismatch -> fallback (1)
should_use_baked_fastpath "$TMP/baked.json" "acct.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_backend" "sha-BBB-different"; \
  # aws stub still returns DDDAAA for any tag; simulate mismatch by editing marker digest
sed -i 's/DDDAAA/OLDDIGEST/' "$TMP/baked.json"
should_use_baked_fastpath "$TMP/baked.json" "acct.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_backend" "sha-AAA"; check "$?" "1" "present+digest-mismatch -> fallback"

# Case 3: marker absent -> fallback (1)
should_use_baked_fastpath "$TMP/nope.json" "repo" "sha-AAA"; check "$?" "1" "absent marker -> fallback"

exit $fail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash deploy/aws/ami/tests/test_baked_fastpath.sh`
Expected: FAIL — `baked_fastpath.sh: No such file or directory` (function not defined yet).

- [ ] **Step 3: Write the minimal implementation**

```bash
# deploy/aws/ami/lib/baked_fastpath.sh
#!/usr/bin/env bash
# Decide whether a baked AMI can be used for a fast boot.
# Usage: should_use_baked_fastpath <marker_path> <image_repo> <image_tag>
# Exit 0 => use fast path (images/models baked & digest matches).
# Exit 1 => fall back to full pull+prewarm (marker missing or digest mismatch).
should_use_baked_fastpath() {
  local marker_path="$1" image_repo="$2" image_tag="$3"
  [ -f "$marker_path" ] || { echo "baked marker $marker_path absent; using full bootstrap" >&2; return 1; }

  local baked_digest running_digest
  baked_digest="$(sed -n 's/.*"backend_image_digest":"\([^"]*\)".*/\1/p' "$marker_path")"
  [ -n "$baked_digest" ] || { echo "baked marker missing digest; using full bootstrap" >&2; return 1; }

  # Digest of the image that boot will actually run (the pinned immutable tag).
  running_digest="$(aws ecr describe-images \
    --repository-name "${image_repo##*/}" \
    --image-ids imageTag="$image_tag" \
    --query 'imageDetails[0].imageDigest' --output text 2>/dev/null)"

  if [ -n "$running_digest" ] && [ "$running_digest" = "$baked_digest" ]; then
    echo "baked digest matches $running_digest; using fast path" >&2
    return 0
  fi
  echo "baked digest ($baked_digest) != running ($running_digest); using full bootstrap" >&2
  return 1
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bash deploy/aws/ami/tests/test_baked_fastpath.sh`
Expected: three `ok:` lines, exit 0.

- [ ] **Step 5: Lint**

Run: `shellcheck deploy/aws/ami/lib/baked_fastpath.sh deploy/aws/ami/tests/test_baked_fastpath.sh`
Expected: no output (clean). If `shellcheck` is absent: `sudo apt-get install -y shellcheck`.

- [ ] **Step 6: Commit**

```bash
git add deploy/aws/ami/lib/baked_fastpath.sh deploy/aws/ami/tests/test_baked_fastpath.sh
git commit -m "Add baked-AMI boot fast-path decision function + tests"
```

---

### Task 2: Baked-AMI prune script

Keep AMI storage bounded: deregister old `Role=backend-baked` AMIs beyond the newest N, and never touch the currently-referenced AMI.

**Files:**
- Create: `deploy/aws/ami/prune_amis.sh`
- Test: `deploy/aws/ami/tests/test_prune_amis.sh`

**Interfaces:**
- Produces: `select_amis_to_deregister <keep_n> <protected_ami_id> <newline-list "creationDate ami-id">` → prints ami-ids to deregister, one per line. The CLI wrapper (`main`) resolves inputs from AWS and calls `aws ec2 deregister-image` + snapshot cleanup unless `DRY_RUN=1`.

- [ ] **Step 1: Write the failing test**

```bash
# deploy/aws/ami/tests/test_prune_amis.sh
#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../prune_amis.sh"
fail=0
check() { if [ "$1" = "$2" ]; then echo "ok: $3"; else echo "FAIL: $3 (got '$1' want '$2')"; fail=1; fi; }

# 4 AMIs oldest->newest; keep 2; protect ami-3 (would-be-pruned but referenced).
LIST=$'2026-07-01 ami-1\n2026-07-02 ami-2\n2026-07-03 ami-3\n2026-07-04 ami-4'
OUT="$(select_amis_to_deregister 2 ami-3 "$LIST" | sort | tr '\n' ' ' | sed 's/ $//')"
# Newest 2 (ami-4, ami-3) kept; ami-3 also protected; so only ami-1, ami-2 pruned.
check "$OUT" "ami-1 ami-2" "keep newest 2, protect referenced"

# Protect a newest AMI (no-op on protection since already kept).
OUT2="$(select_amis_to_deregister 1 ami-4 "$LIST" | sort | tr '\n' ' ' | sed 's/ $//')"
# keep newest 1 (ami-4); prune ami-1,ami-2,ami-3 (none protected among pruned except... ami-4 protected already kept)
check "$OUT2" "ami-1 ami-2 ami-3" "keep newest 1, protected already newest"

# Protect an AMI that WOULD otherwise be pruned (exercises the grep -vx protect branch).
OUT3="$(select_amis_to_deregister 2 ami-1 "$LIST" | sort | tr '\n' ' ' | sed 's/ $//')"
# keep newest 2 (ami-4, ami-3); ami-1 & ami-2 would prune, but ami-1 is protected -> only ami-2.
check "$OUT3" "ami-2" "protect a would-be-pruned AMI"

exit $fail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash deploy/aws/ami/tests/test_prune_amis.sh`
Expected: FAIL — `select_amis_to_deregister: command not found`.

- [ ] **Step 3: Write the minimal implementation**

```bash
# deploy/aws/ami/prune_amis.sh
#!/usr/bin/env bash
set -euo pipefail

# Pure selection: given keep_n, a protected ami id, and a newline list of
# "creationDate ami-id" (any order), print ami-ids to deregister.
select_amis_to_deregister() {
  local keep_n="$1" protected="$2" list="$3"
  # Sort by date desc, drop the newest keep_n, then exclude the protected id.
  printf '%s\n' "$list" | sort -r | awk 'NF' | tail -n +"$((keep_n + 1))" \
    | awk '{print $2}' | grep -vx "$protected" || true
}

main() {
  local region="${AWS_REGION:-us-east-1}" ssm_prefix="${SSM_PREFIX:-/pdfx}" keep_n="${KEEP_N:-3}"
  local protected
  protected="$(aws ssm get-parameter --region "$region" --name "${ssm_prefix}/backend-ami" \
    --query 'Parameter.Value' --output text 2>/dev/null || echo "none")"
  local list
  list="$(aws ec2 describe-images --region "$region" --owners self \
    --filters 'Name=tag:Role,Values=backend-baked' \
    --query 'Images[].[CreationDate,ImageId]' --output text)"
  local to_prune
  to_prune="$(select_amis_to_deregister "$keep_n" "$protected" "$list")"
  [ -n "$to_prune" ] || { echo "Nothing to prune."; return 0; }
  while read -r ami; do
    [ -n "$ami" ] || continue
    local snaps
    snaps="$(aws ec2 describe-images --region "$region" --image-ids "$ami" \
      --query 'Images[0].BlockDeviceMappings[].Ebs.SnapshotId' --output text 2>/dev/null || true)"
    if [ "${DRY_RUN:-0}" = "1" ]; then
      echo "[dry-run] would deregister $ami and snapshots: $snaps"
    else
      echo "Deregistering $ami"
      aws ec2 deregister-image --region "$region" --image-id "$ami"
      for s in $snaps; do aws ec2 delete-snapshot --region "$region" --snapshot-id "$s" || true; done
    fi
  done <<< "$to_prune"
}

# Only run main when executed, not when sourced by tests.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then main "$@"; fi
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bash deploy/aws/ami/tests/test_prune_amis.sh`
Expected: two `ok:` lines, exit 0.

- [ ] **Step 5: Lint**

Run: `shellcheck deploy/aws/ami/prune_amis.sh deploy/aws/ami/tests/test_prune_amis.sh`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add deploy/aws/ami/prune_amis.sh deploy/aws/ami/tests/test_prune_amis.sh
git commit -m "Add baked-AMI prune script (keep last N, never the referenced AMI) + tests"
```

---

### Task 3: Bake provisioner + sample PDF

The Packer provisioner that turns a base DL AMI into a ready-to-run PDFX backend image: reuse `deploy.sh`, run a real extraction to warm every cache, assert caches, then scrub secrets/identity.

**Files:**
- Create: `deploy/aws/ami/provision.sh`
- Create: `deploy/aws/ami/test-sample.pdf` (a real small PDF with a heading, a paragraph, and a small table so Docling + GROBID + Marker all engage)
- Test: reuse of the extracted `assert_caches_nonempty` function via a tiny inline test.

**Interfaces:**
- Consumes: `deploy/deploy.sh` (env: `GPU_MODE`, `PDFX_DEPLOY_BUILD_MODE`, `PDFX_DEPLOY_PULL_IMAGES`, `PDFX_PREWARM_MODELS`); `deploy/aws/ami/lib/baked_fastpath.sh` (copied into the AMI).
- Produces: an AMI filesystem with images + caches on disk, `/opt/pdfx/baked.json`, and `/opt/pdfx/baked_fastpath.sh`. Environment expected from Packer: `BACKEND_IMAGE_REPO`, `BACKEND_IMAGE_TAG`, `BASE_AMI_ID`, `AWS_REGION`.

- [ ] **Step 1: Add the sample PDF**

Create a small real PDF (heading + paragraph + a 2×2 table) at `deploy/aws/ami/test-sample.pdf`. Generate it deterministically so it can be regenerated:

```bash
python3 - <<'PY'
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table
c = canvas.Canvas("deploy/aws/ami/test-sample.pdf", pagesize=letter)
c.setFont("Helvetica-Bold", 16); c.drawString(72, 720, "PDFX Bake Warm-Up Document")
c.setFont("Helvetica", 11)
c.drawString(72, 690, "This paragraph exercises Marker, Docling, and GROBID extraction paths during the AMI bake.")
t = Table([["Gene", "Value"], ["abcA", "1.23"]]); t.wrapOn(c, 400, 100); t.drawOn(c, 72, 620)
c.showPage(); c.save()
print("wrote deploy/aws/ami/test-sample.pdf")
PY
```
(If `reportlab` is unavailable: `pip install reportlab`. Commit the resulting binary PDF.)

- [ ] **Step 2: Write the provisioner with a testable cache-assert function**

```bash
# deploy/aws/ami/provision.sh
#!/usr/bin/env bash
set -euo pipefail

# --- testable helper: fail if any required cache dir is empty ---
assert_caches_nonempty() {
  # Gate only on dirs known to populate: HF_HOME/Transformers/Torch all redirect to
  # data/models, and any extraction runs OCR so data/rapidocr_models fills. data/model_cache
  # (/root/.cache) is NOT asserted -- nothing reliably writes it, so gating on it would fail a
  # healthy bake (design-review N2).
  local root="$1" rc=0 d
  for d in data/models data/rapidocr_models; do
    if [ -z "$(find "$root/$d" -type f -print -quit 2>/dev/null)" ]; then
      echo "ERROR: cache dir '$d' is empty after warm-up extraction" >&2; rc=1
    fi
  done
  if [ ! -f "$root/data/cache/marker_worker_ready.json" ]; then
    echo "ERROR: marker ready file missing" >&2; rc=1
  fi
  return $rc
}

main() {
  : "${BACKEND_IMAGE_REPO:?}"; : "${BACKEND_IMAGE_TAG:?}"; : "${AWS_REGION:=us-east-1}"; : "${BASE_AMI_ID:=unknown}"
  local SERVICE_DIR=/home/ec2-user/agr_pdf_extraction_service

  sudo dnf install -y docker git jq awscli
  sudo systemctl enable --now docker

  # Fresh checkout at the ref being baked (Packer sets BACKEND_GIT_REF; default = tag).
  sudo rm -rf "$SERVICE_DIR"
  sudo -u ec2-user git clone https://github.com/alliance-genome/agr_pdf_extraction_service.git "$SERVICE_DIR"
  sudo -u ec2-user git -C "$SERVICE_DIR" checkout "${BACKEND_GIT_REF:-$BACKEND_IMAGE_TAG}" || true

  cd "$SERVICE_DIR"
  mkdir -p data/cache data/uploads data/models data/model_cache data/rapidocr_models logs
  # Placeholder .env sufficient for create_app()+worker health, NEVER real secrets (spec N2).
  # DB/Redis/Celery URLs come from compose x-shared-env (docker-compose.gpu.yml lines 15-19),
  # which overrides env_file -- so no credentialed URI needs to live in this .env at all
  # (keeps gitleaks/TruffleHog clean and avoids drifting from the compose defaults).
  cat > .env <<'ENV'
FLASK_ENV=production
OPENAI_API_KEY=unused-during-bake
ENV

  # ECR auth via the build instance's INSTANCE PROFILE (no static creds baked).
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "${BACKEND_IMAGE_REPO%%/*}"

  # Reuse the production deploy path to pull all images + prewarm Marker.
  ( cd deploy && \
    PDFX_GPU_IMAGE="${BACKEND_IMAGE_REPO}:${BACKEND_IMAGE_TAG}" \
    PDFX_DEPLOY_BUILD_MODE=never PDFX_DEPLOY_PULL_IMAGES=always \
    PDFX_PREWARM_MODELS=marker GPU_MODE=on ./deploy.sh )

  # Warm the LAZY caches (Docling/rapidocr/GROBID) with one real extraction.
  # merge=false: skip the LLM merge (needs a real key) but still run grobid+docling+marker,
  # which is what warms the caches. Without it the run ends 'failed' (api.py merge default=true).
  cp "$SERVICE_DIR/deploy/aws/ami/test-sample.pdf" /tmp/sample.pdf
  local pid
  pid="$(curl -sS -X POST http://127.0.0.1:5000/api/v1/extract \
    -F 'file=@/tmp/sample.pdf' -F 'merge=false' | jq -r '.process_id')"
  echo "warm-up extraction process_id=$pid"
  for _ in $(seq 1 120); do
    status="$(curl -sS "http://127.0.0.1:5000/api/v1/extract/${pid}" | jq -r '.status')"
    [ "$status" = "complete" ] && break
    [ "$status" = "failed" ] && { echo "ERROR: warm-up extraction failed"; exit 1; }
    sleep 5
  done

  assert_caches_nonempty "$SERVICE_DIR"

  ( cd deploy && GPU_MODE=on ./manage.sh down 2>/dev/null || docker compose -f docker-compose.gpu.yml -p pdfx down )

  # --- write marker for the boot fast-path ---
  local digest
  digest="$(aws ecr describe-images --region "$AWS_REGION" --repository-name "${BACKEND_IMAGE_REPO##*/}" \
    --image-ids imageTag="$BACKEND_IMAGE_TAG" --query 'imageDetails[0].imageDigest' --output text)"
  sudo mkdir -p /opt/pdfx
  sudo cp deploy/aws/ami/lib/baked_fastpath.sh /opt/pdfx/baked_fastpath.sh
  printf '{"backend_image_repo":"%s","backend_image_tag":"%s","backend_image_digest":"%s","base_ami_id":"%s","baked_at":"%s"}\n' \
    "$BACKEND_IMAGE_REPO" "$BACKEND_IMAGE_TAG" "$digest" "$BASE_AMI_ID" "$(date -u +%FT%TZ)" \
    | sudo tee /opt/pdfx/baked.json >/dev/null

  # --- hygiene: strip all secrets/identity before snapshot ---
  rm -f "$SERVICE_DIR/.env"
  rm -f "$HOME/.docker/config.json" /home/ec2-user/.docker/config.json 2>/dev/null || true
  rm -rf "$HOME/.aws" /home/ec2-user/.aws 2>/dev/null || true
  sudo rm -rf /var/lib/cloud/instances/* /var/log/pdfx-*.log
  sudo rm -f /etc/ssh/ssh_host_* /etc/machine-id
  sudo truncate -s 0 /etc/machine-id 2>/dev/null || true
  cat /dev/null > "$HOME/.bash_history" 2>/dev/null || true
  echo "provision complete"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then main "$@"; fi
```

- [ ] **Step 3: Unit-test the cache-assert helper**

```bash
bash -c '
source deploy/aws/ami/provision.sh
TMP="$(mktemp -d)"; mkdir -p "$TMP"/data/{models,model_cache,rapidocr_models,cache}
# empty -> fails
if assert_caches_nonempty "$TMP"; then echo "FAIL: empty should error"; else echo "ok: empty errors"; fi
# populate -> passes
echo x > "$TMP/data/models/m"; echo x > "$TMP/data/model_cache/c"; echo x > "$TMP/data/rapidocr_models/r"; echo "{}" > "$TMP/data/cache/marker_worker_ready.json"
if assert_caches_nonempty "$TMP"; then echo "ok: populated passes"; else echo "FAIL: populated should pass"; fi
rm -rf "$TMP"'
```
Expected: `ok: empty errors` then `ok: populated passes`.

- [ ] **Step 4: Lint**

Run: `shellcheck deploy/aws/ami/provision.sh`
Expected: clean (add `# shellcheck disable=` only with a written reason).

- [ ] **Step 5: Commit**

```bash
git add deploy/aws/ami/provision.sh deploy/aws/ami/test-sample.pdf
git commit -m "Add PDFX AMI bake provisioner (reuse deploy.sh + warm-up extraction + cache asserts + hygiene) and sample PDF"
```

---

### Task 4: Packer template

Define the `amazon-ebs` build that runs `provision.sh` on a g6.2xlarge (standing L4 smoke test) and produces the tagged baked AMI.

**Files:**
- Create: `deploy/aws/ami/pdfx-backend.pkr.hcl`

**Interfaces:**
- Consumes: `provision.sh`, `lib/baked_fastpath.sh`, `test-sample.pdf`.
- Produces: an AMI named `pdfx-backend-baked-<image_tag>-<timestamp>`, tagged `Project=pdfx`, `Role=backend-baked`, `BackendImageTag=<tag>`. Build variables: `region`, `base_ami_id`, `backend_image_repo`, `backend_image_tag`, `build_instance_type`, `iam_instance_profile`, `subnet_id`, `root_volume_size`.

- [ ] **Step 1: Write the template**

```hcl
# deploy/aws/ami/pdfx-backend.pkr.hcl
packer {
  required_plugins {
    amazon = { source = "github.com/hashicorp/amazon", version = ">= 1.3.0" }
  }
}

variable "region"                { type = string  default = "us-east-1" }
variable "base_ami_id"           { type = string }
variable "backend_image_repo"    { type = string } # 100225593120.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_backend
variable "backend_image_tag"     { type = string } # immutable merge SHA
variable "build_instance_type"   { type = string  default = "g6.2xlarge" }
variable "iam_instance_profile"  { type = string } # profile granting ECR read on the build box
variable "subnet_id"             { type = string }
variable "root_volume_size"      { type = number  default = 200 }

locals { ts = formatdate("YYYYMMDD-hhmmss", timestamp()) }

source "amazon-ebs" "pdfx_backend" {
  region               = var.region
  source_ami           = var.base_ami_id
  instance_type        = var.build_instance_type
  ssh_username         = "ec2-user"
  iam_instance_profile = var.iam_instance_profile
  subnet_id            = var.subnet_id
  associate_public_ip_address = true
  ami_name             = "pdfx-backend-baked-${var.backend_image_tag}-${local.ts}"

  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = {
    Project         = "pdfx"
    Role            = "backend-baked"
    BackendImageTag = var.backend_image_tag
    BaseAmiId       = var.base_ami_id
  }
}

build {
  sources = ["source.amazon-ebs.pdfx_backend"]

  # Ship the repo scripts the provisioner needs.
  provisioner "file" {
    source      = "${path.root}/../../.."   # repo root
    destination = "/tmp/repo"
  }

  provisioner "shell" {
    environment_vars = [
      "BACKEND_IMAGE_REPO=${var.backend_image_repo}",
      "BACKEND_IMAGE_TAG=${var.backend_image_tag}",
      "BASE_AMI_ID=${var.base_ami_id}",
      "AWS_REGION=${var.region}",
      "BACKEND_GIT_REF=${var.backend_image_tag}",
    ]
    inline = [
      "sudo cp -r /tmp/repo /home/ec2-user/agr_pdf_extraction_service_src || true",
      "cd /home/ec2-user/agr_pdf_extraction_service_src || cd /tmp/repo",
      "chmod +x deploy/aws/ami/provision.sh",
      "sudo -E deploy/aws/ami/provision.sh",
    ]
  }

  # Emit the built AMI id to manifest.json for the CI job to read (robust vs. -machine-readable).
  post-processor "manifest" {
    output     = "manifest.json"
    strip_path = true
  }
}
```

- [ ] **Step 2: Format check**

Run: `packer fmt -check deploy/aws/ami/pdfx-backend.pkr.hcl`
Expected: no diff (exit 0). If it reformats, re-run without `-check` and re-stage.
(If `packer` absent: install per https://developer.hashicorp.com/packer/install.)

- [ ] **Step 3: Init plugins + validate**

Run:
```bash
cd deploy/aws/ami
packer init pdfx-backend.pkr.hcl
packer validate \
  -var base_ami_id=ami-00c6ddd550364d6c3 \
  -var backend_image_repo=100225593120.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_backend \
  -var backend_image_tag=validate-only \
  -var iam_instance_profile=pdfx-ec2-profile \
  -var subnet_id=subnet-af62dca3 \
  pdfx-backend.pkr.hcl
cd -
```
Expected: `The configuration is valid.`

- [ ] **Step 4: Commit**

```bash
git add deploy/aws/ami/pdfx-backend.pkr.hcl
git commit -m "Add Packer template to bake PDFX backend AMI on g6.2xlarge"
```

---

### Task 5: CloudFormation — SSM AMI/image-tag params, resolve:ssm, boot fast-path

Wire the launch template to resolve the AMI from SSM, pin the immutable image tag, and branch the bootstrap through the baked fast-path with a fallback to the current full path.

**Files:**
- Modify: `deploy/aws/pdfx-stack.yaml` (Parameters ~77–132; `SsmParameters` block ~656+; `BackendLaunchTemplate` UserData ~426–540)

**Interfaces:**
- Consumes: SSM `/pdfx/backend-ami`, `/pdfx/backend-image-tag`; `/opt/pdfx/baked_fastpath.sh` baked into the AMI.
- Produces: an ASG that launches the SSM-resolved AMI and boots fast when the baked digest matches.

- [ ] **Step 1: Seed the two SSM parameters OUT OF BAND (do NOT CFN-manage their Value)**

`/pdfx/backend-ami` and `/pdfx/backend-image-tag` hold values the **bake** mutates. Do **not** create them as `AWS::SSM::Parameter` resources with a `Value:` — CloudFormation reasserts a managed `Value` on every stack update, which would silently revert the live baked AMI/tag back to the seed and drop every wake to the slow full-bootstrap fallback (design-review B3). Instead they are seeded once via CLI (documented in Task 9) and thereafter owned by the bake; the launch template only *reads* them via `resolve:ssm`.

This step adds **no** CFN resource. It is a deliberate no-op in the template — the seed lives in the Task 9 runbook, and for a brand-new environment must run before the first ASG scale-out. (`BackendImageTag` already exists as a Parameter at `pdfx-stack.yaml:157`; do not re-add it.)

- [ ] **Step 2: Point the launch template ImageId at SSM**

Change `BackendLaunchTemplate.LaunchTemplateData.ImageId` from `!Ref BackendAmiId` to the launch-time SSM resolution:

```yaml
        ImageId: !Sub "resolve:ssm:/${SsmParameterPath}/backend-ami"
```

- [ ] **Step 3: Insert the boot fast-path branch into UserData**

In the `pdfx-backend-bootstrap.sh` heredoc (before the `deploy.sh` invocation near line ~520), replace the fixed deploy call with a branch that uses the baked marker when valid. Insert:

```bash
              # --- baked-AMI fast path (spec 4B) ---
              BACKEND_IMAGE_REPO="${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/${BackendImageRepositoryName}"
              PINNED_TAG="$(aws ssm get-parameter --region "${AWS::Region}" \
                --name "/${SsmParameterPath}/backend-image-tag" --query Parameter.Value --output text 2>/dev/null || echo "${BackendImageTag}")"
              USE_FAST_PATH=0
              if [ -f /opt/pdfx/baked_fastpath.sh ]; then
                . /opt/pdfx/baked_fastpath.sh
                if should_use_baked_fastpath /opt/pdfx/baked.json "$BACKEND_IMAGE_REPO" "$PINNED_TAG"; then USE_FAST_PATH=1; fi
              fi

              cd deploy
              # NOTE: this whole heredoc is inside CFN Fn::Sub, so shell vars MUST use bare $
              # (not ${...}, which CFN would try to resolve as a template variable and fail
              # validate-template -- design-review B1). Only real CFN refs use ${...} here.
              BACKEND_IMAGE_URI="$BACKEND_IMAGE_REPO:$PINNED_TAG"
              aws ecr get-login-password --region "${AWS::Region}" \
                | docker login --username AWS --password-stdin "${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com"
              if [ "$USE_FAST_PATH" = "1" ]; then
                echo "Booting via baked fast path (images/models present)"
                PDFX_GPU_IMAGE="$BACKEND_IMAGE_URI" PDFX_DEPLOY_BUILD_MODE=never \
                  PDFX_DEPLOY_PULL_IMAGES=never PDFX_PREWARM_MODELS=off GPU_MODE=on ./deploy.sh
              else
                echo "Baked marker absent/mismatched; full pull+prewarm fallback"
                PDFX_GPU_IMAGE="$BACKEND_IMAGE_URI" PDFX_DEPLOY_BUILD_MODE=never \
                  PDFX_DEPLOY_PULL_IMAGES=auto PDFX_PREWARM_MODELS=auto GPU_MODE=on ./deploy.sh
              fi
```

Remove the now-superseded original single `PDFX_GPU_IMAGE=... ./deploy.sh` block (the one using `${BackendImageTag}` directly).

- [ ] **Step 4: Validate the template renders**

Run:
```bash
aws cloudformation validate-template --profile ctabone --region us-east-1 \
  --template-body file://deploy/aws/pdfx-stack.yaml >/dev/null && echo "validate OK"
cfn-lint deploy/aws/pdfx-stack.yaml
```
Expected: `validate OK`, and cfn-lint clean (or only pre-existing warnings unrelated to this change). If `cfn-lint` absent: `pip install cfn-lint`.

- [ ] **Step 5: Assert the wiring is present**

Run:
```bash
grep -q 'resolve:ssm:/${SsmParameterPath}/backend-ami' deploy/aws/pdfx-stack.yaml && echo "ImageId wired"
grep -q 'should_use_baked_fastpath' deploy/aws/pdfx-stack.yaml && echo "fast-path wired"
grep -q '/backend-image-tag' deploy/aws/pdfx-stack.yaml && echo "image-tag read wired"
```
Expected: all three echoes print.

- [ ] **Step 6: Commit**

```bash
git add deploy/aws/pdfx-stack.yaml
git commit -m "Wire backend ASG to SSM-resolved baked AMI + immutable image tag + boot fast-path"
```

---

### Task 6: CloudFormation — MixedInstancesPolicy, multi-AZ, warm pool off, startup timeout

Replace the single-type launch with an on-demand `lowest-price` mix of g5.2xlarge + g5.4xlarge across all AZs, disable the warm pool, and widen the startup timeout for the fallback path.

**Files:**
- Modify: `deploy/aws/pdfx-stack.yaml` (Parameters ~104–130; `BackendAutoScalingGroup` ~542–566; `BackendWarmPool` ~567–579; `startup-timeout-minutes` param/SSM)

**Interfaces:**
- Consumes: `BackendLaunchTemplate` (from Task 5).
- Produces: an ASG spanning five subnets with a two-type on-demand mix and no warm pool by default.

- [ ] **Step 1: Add a multi-AZ subnet list parameter (with a Default) and retire the single-subnet param**

In Parameters, add `BackendSubnetIds` **with a Default** (spec §5 wants the five prod subnets by default; a `List<...::Subnet::Id>` with no default forces every deploy to pass it and breaks existing tooling — design-review B4):

```yaml
  BackendSubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
    Default: subnet-3ebf4477,subnet-df7c7487,subnet-81c95ee4,subnet-ff838bd5,subnet-af62dca3
    Description: Subnets across all AZs (1a,1b,1c,1d,1f) the backend ASG may launch into.
```

The existing `BackendSubnetId` (param at `:31`) is referenced only by the ASG `VPCZoneIdentifier` at `:552`. Step 3 replaces that reference, so **delete the `BackendSubnetId` parameter** to avoid a cfn-lint W2001 unused-parameter warning. First confirm it has no other references:

```bash
grep -n 'BackendSubnetId\b' deploy/aws/pdfx-stack.yaml   # expect only the param def and the :552 usage
```

- [ ] **Step 2: Default the warm pool off and widen startup timeout**

```yaml
  BackendWarmPoolMinSize:
    Type: Number
    Default: 0          # was 1 — no warm pool; fast baked boot + terminate-on-idle
    MinValue: 0
    MaxValue: 1
    Description: Stopped warm-pool count. 0 = terminate on idle and re-select cheapest on next wake.
```
And **edit** the existing `StartupTimeoutMinutes` (already at `:214`) — keep it `Type: String` like its sibling timeout params (it feeds a String SSM value); only change the default from `"30"` to `"45"` (margin so the full-bootstrap fallback can't cascade into a replace loop — design-review N5):

```yaml
  StartupTimeoutMinutes:
    Type: String
    Default: "45"
```

- [ ] **Step 3: Convert the ASG to a MixedInstancesPolicy across subnets**

Replace `BackendAutoScalingGroup.Properties.LaunchTemplate` and `VPCZoneIdentifier` with:

```yaml
      VPCZoneIdentifier: !Ref BackendSubnetIds
      MixedInstancesPolicy:
        LaunchTemplate:
          LaunchTemplateSpecification:
            LaunchTemplateId: !Ref BackendLaunchTemplate
            Version: !GetAtt BackendLaunchTemplate.LatestVersionNumber
          Overrides:
            - InstanceType: g5.2xlarge
            - InstanceType: g5.4xlarge
            # g6.2xlarge intentionally deferred to KANBAN-1411 (needs L4 validation;
            # lowest-price cannot keep a cheaper type deprioritized).
        InstancesDistribution:
          OnDemandBaseCapacity: 0
          OnDemandPercentageAboveBaseCapacity: 100
          OnDemandAllocationStrategy: lowest-price
```

(Delete the now-replaced top-level `LaunchTemplate:` block on the ASG so only the `MixedInstancesPolicy` remains.)

- [ ] **Step 4: Validate**

Run:
```bash
aws cloudformation validate-template --profile ctabone --region us-east-1 \
  --template-body file://deploy/aws/pdfx-stack.yaml >/dev/null && echo "validate OK"
cfn-lint deploy/aws/pdfx-stack.yaml
```
Expected: `validate OK`; cfn-lint clean.

- [ ] **Step 5: Assert the policy shape**

Run:
```bash
grep -q 'OnDemandAllocationStrategy: lowest-price' deploy/aws/pdfx-stack.yaml && echo "lowest-price set"
grep -q 'InstanceType: g5.4xlarge' deploy/aws/pdfx-stack.yaml && echo "g5.4xlarge in mix"
! grep -q 'InstanceType: g6' deploy/aws/pdfx-stack.yaml && echo "no g6 (correct for Phase 1)"
```
Expected: `lowest-price set`, `g5.4xlarge in mix`, `no g6 (correct for Phase 1)`.

- [ ] **Step 6: Commit**

```bash
git add deploy/aws/pdfx-stack.yaml
git commit -m "Backend ASG: on-demand lowest-price g5.2xlarge+g5.4xlarge across all AZs, warm pool off, startup timeout 45m"
```

---

### Task 7: IAM policy additions for the bake

Grant the CI OIDC role the EC2/Packer + PassRole permissions the bake needs. (SSM read/write on `/pdfx/*` already exist, so `resolve:ssm` and the SSM updates are already covered.)

**Files:**
- Modify: `deploy/aws/github-actions-deploy-policy.json`

- [ ] **Step 1: Add the bake statements**

Append two statements to the `Statement` array:

```json
    {
      "Sid": "PackerBakeEc2",
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances", "ec2:StopInstances", "ec2:TerminateInstances",
        "ec2:CreateImage", "ec2:RegisterImage", "ec2:DeregisterImage",
        "ec2:CreateSnapshot", "ec2:DeleteSnapshot", "ec2:CreateTags",
        "ec2:CreateKeyPair", "ec2:DeleteKeyPair",
        "ec2:CreateSecurityGroup", "ec2:DeleteSecurityGroup",
        "ec2:AuthorizeSecurityGroupIngress", "ec2:RevokeSecurityGroupIngress",
        "ec2:AssociateIamInstanceProfile", "ec2:ReplaceIamInstanceProfileAssociation",
        "ec2:DescribeImages", "ec2:DescribeInstances", "ec2:DescribeInstanceStatus",
        "ec2:DescribeSubnets", "ec2:DescribeVpcs", "ec2:DescribeSecurityGroups",
        "ec2:DescribeSnapshots", "ec2:DescribeRegions", "ec2:DescribeImageAttribute",
        "ec2:DescribeKeyPairs", "ec2:DescribeTags"
      ],
      "Resource": "*"
    },
    {
      "Sid": "PassBackendInstanceProfileRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::100225593120:role/pdfx-ec2-role",
      "Condition": { "StringEquals": { "iam:PassedToService": "ec2.amazonaws.com" } }
    }
```

The bake reuses the existing backend instance profile `pdfx-ec2-profile` (role `pdfx-ec2-role`, defined in `pdfx-stack.yaml`), which already grants `ecr:GetAuthorizationToken` + pull on `agr_pdfx_backend` — exactly what the build box needs for ECR auth. No new instance profile/role is created.

- [ ] **Step 2: Validate JSON + policy grammar**

Run:
```bash
jq empty deploy/aws/github-actions-deploy-policy.json && echo "valid JSON"
aws accessanalyzer validate-policy --profile ctabone --region us-east-1 \
  --policy-type IDENTITY_POLICY \
  --policy-document file://deploy/aws/github-actions-deploy-policy.json \
  --query 'findings[?findingType==`ERROR`]' --output text
```
Expected: `valid JSON`; the validate-policy ERROR query prints nothing (no errors). (Skip the second command if access-analyzer isn't permitted; `jq empty` is the gate.)

- [ ] **Step 3: Commit**

```bash
git add deploy/aws/github-actions-deploy-policy.json
git commit -m "IAM: add EC2/Packer bake permissions + PassRole for backend instance profile"
```

---

### Task 8: CI — change-gated, concurrency-guarded auto-rebake job

Add a job that bakes a new AMI after a backend-affecting merge and publishes it to SSM.

**Files:**
- Modify: `.github/workflows/main-build-and-deploy.yml`

**Interfaces:**
- Consumes: `deploy-prod` outputs (`IMAGE_TAG`), the Packer template, `provision.sh`, `prune_amis.sh`.

- [ ] **Step 1: Add a change-detection output to `on-merge`**

In the `on-merge` job, add these steps **after** the existing `Capture deployment metadata` (`id: meta`) step — the checkout's `ref: ${{ steps.meta.outputs.source-ref }}` is empty if placed before `meta` runs:

```yaml
      - name: Check out repository code
        uses: actions/checkout@v5
        with:
          ref: ${{ steps.meta.outputs.source-ref }}
          fetch-depth: 2
      - name: Detect backend-affecting changes
        id: changed
        run: |
          if git diff --name-only HEAD~1 HEAD | grep -E '^(deploy/Dockerfile\.gpu|requirements\.txt|app/|celery_app\.py|config\.py|deploy/docker-compose.*\.yml|deploy/aws/ami/)' >/dev/null; then
            echo "backend-changed=true" >> "$GITHUB_OUTPUT"
          elif git diff --name-only HEAD~1 HEAD >/dev/null 2>&1; then
            echo "backend-changed=false" >> "$GITHUB_OUTPUT"
          else
            echo "backend-changed=true" >> "$GITHUB_OUTPUT"   # inconclusive -> run (safe over stale)
          fi
```
Add to the `on-merge` job `outputs:` block: `backend-changed: ${{ steps.changed.outputs.backend-changed }}`.

- [ ] **Step 2: Add the `bake-backend-ami` job**

Append after `deploy-prod`:

```yaml
  bake-backend-ami:
    name: Bake backend AMI
    needs: [on-merge, deploy-prod]
    if: needs.on-merge.outputs.backend-changed == 'true'
    permissions:
      contents: read
      id-token: write
    runs-on: ubuntu-24.04
    concurrency:
      group: ${{ github.workflow }}-bake-backend-ami
      cancel-in-progress: false
    env:
      AWS_REGION: us-east-1
      IMAGE_TAG: ${{ needs.on-merge.outputs.image-tag }}
      BACKEND_IMAGE_REPO: 100225593120.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_backend
      BASE_AMI_ID: ami-00c6ddd550364d6c3
      BUILD_SUBNET_ID: subnet-af62dca3
      IAM_INSTANCE_PROFILE: pdfx-ec2-profile
      SSM_PREFIX: /pdfx
    steps:
      - uses: actions/checkout@v5
        with:
          ref: ${{ needs.on-merge.outputs.source-ref }}
      - name: AWS credentials configuration
        uses: aws-actions/configure-aws-credentials@v5
        with:
          role-to-assume: ${{ secrets.GH_ACTIONS_AWS_ROLE }}
          role-session-name: gh-actions-${{ github.run_id }}-pdfx-bake
          aws-region: ${{ env.AWS_REGION }}
      - name: Set up Packer
        uses: hashicorp/setup-packer@v3
        with:
          version: latest
      - name: Bake AMI
        working-directory: deploy/aws/ami
        run: |
          packer init pdfx-backend.pkr.hcl
          packer build \
            -var region="${AWS_REGION}" \
            -var base_ami_id="${BASE_AMI_ID}" \
            -var backend_image_repo="${BACKEND_IMAGE_REPO}" \
            -var backend_image_tag="${IMAGE_TAG}" \
            -var iam_instance_profile="${IAM_INSTANCE_PROFILE}" \
            -var subnet_id="${BUILD_SUBNET_ID}" \
            pdfx-backend.pkr.hcl
          # manifest.json artifact_id is "region:ami-xxxx"; take the ami id.
          AMI_ID="$(jq -r '.builds[-1].artifact_id' manifest.json | cut -d':' -f2)"
          echo "Baked AMI: $AMI_ID"
          test -n "$AMI_ID" && [ "$AMI_ID" != "null" ]
          aws ssm put-parameter --region "${AWS_REGION}" --overwrite \
            --name "${SSM_PREFIX}/backend-ami" --type String --value "$AMI_ID"
          aws ssm put-parameter --region "${AWS_REGION}" --overwrite \
            --name "${SSM_PREFIX}/backend-image-tag" --type String --value "${IMAGE_TAG}"
      - name: Prune old baked AMIs
        run: KEEP_N=3 SSM_PREFIX=/pdfx AWS_REGION=us-east-1 bash deploy/aws/ami/prune_amis.sh
```

- [ ] **Step 3: Lint the workflow**

Run: `actionlint .github/workflows/main-build-and-deploy.yml`
Expected: no errors. (If `actionlint` absent: `go install github.com/rhysd/actionlint/cmd/actionlint@latest` or use the docker image; at minimum run `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/main-build-and-deploy.yml'))" && echo YAML-OK`.)

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/main-build-and-deploy.yml
git commit -m "CI: change-gated, concurrency-guarded backend AMI auto-rebake job"
```

---

### Task 9: Operator README

Document baking, IAM, params, the `resolve:ssm` dry-run check, rollout, and backout for whoever applies this to prod.

**Files:**
- Create: `deploy/aws/ami/README.md`

- [ ] **Step 1: Write the README**

Include, as concrete runnable sections:
- **What this is:** the baked-AMI + MixedInstancesPolicy design (link the spec).
- **Manual bake:** the exact `packer init`/`packer build` command from Task 8 with `-var` values.
- **One-time SSM seed (before first use):** these two params are intentionally **not** CloudFormation-managed values (a managed `Value:` would be reasserted on every stack update, reverting the live baked AMI/tag — design-review B3). Seed them once:
  ```bash
  aws ssm put-parameter --profile ctabone --region us-east-1 --overwrite \
    --name /pdfx/backend-ami --type String --value <seed-or-baked-ami-id>
  aws ssm put-parameter --profile ctabone --region us-east-1 --overwrite \
    --name /pdfx/backend-image-tag --type String --value <immutable-tag>
  ```
- **Params written by the bake:** `/pdfx/backend-ami`, `/pdfx/backend-image-tag` (updated only after a successful `RegisterImage`). Warn readers that a CloudFormation redeploy must **not** re-introduce these as managed-`Value` resources.
- **IAM:** the added statements and the existing `pdfx-ec2-profile` (role `pdfx-ec2-role`) reused for ECR auth on the build box.
- **Apply path (prod is drifted from the template):** prefer stack reconcile/import; if applying directly, mirror edits into `pdfx-stack.yaml`. Order: publish baked AMI → set both SSM params → **dry-run** `aws autoscaling update-auto-scaling-group` (this is where EC2 validates `ssm:GetParameters` for the `resolve:ssm` ImageId) → apply. Note the initial `resolve:ssm` switch creates one new launch-template version.
- **Backout:** re-enable warm pool (`BackendWarmPoolMinSize=1`) and point `/pdfx/backend-ami` at a known-good AMI (or the DL base AMI → boot fallback path).
- **Phase 2 (KANBAN-1411):** add `g6.2xlarge` to `Overrides` after booting a g6 from the baked AMI and running a real extraction.

- [ ] **Step 2: Commit**

```bash
git add deploy/aws/ami/README.md
git commit -m "Add PDFX AMI bake + rollout/backout operator README"
```

---

## Final verification (after all tasks)

- [ ] Run all shell unit tests: `bash deploy/aws/ami/tests/test_baked_fastpath.sh && bash deploy/aws/ami/tests/test_prune_amis.sh`
- [ ] `packer validate` (Task 4 Step 3) and `cfn-lint deploy/aws/pdfx-stack.yaml` both pass.
- [ ] `jq empty deploy/aws/github-actions-deploy-policy.json` and workflow lint pass.
- [ ] Confirm **no g6 in the instance mix** (Phase 1): `! grep -En 'InstanceType:[[:space:]]*g6' deploy/aws/pdfx-stack.yaml` (scoped to the override list so it doesn't match the explanatory `g6.2xlarge` comment the plan adds).
- [ ] Push the branch and open a PR; do **not** apply to prod (that's the operator's step per the spec).

## Spec coverage self-check

- Baked AMI (Packer, reuse deploy.sh, warm-up extraction, cache asserts, hygiene, marker) → Tasks 3, 4.
- Boot fast-path with digest match + fallback → Tasks 1, 5.
- MixedInstancesPolicy (g5-only, on-demand lowest-price), multi-AZ, warm pool off, startup timeout 45 → Task 6.
- AMI + image-tag SSM wiring, resolve:ssm, lifecycle prune → Tasks 2, 5, 8.
- CI auto-rebake, change-gate, concurrency, IAM → Tasks 7, 8.
- Rollout/backout, resolve:ssm dry-run, Phase 2 pointer → Task 9.
