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

  # SC2015: intentional best-effort teardown -- if manage.sh down is unavailable/fails,
  # fall back to `docker compose down`; both are idempotent cleanup, so C-runs-when-A-true is fine.
  # shellcheck disable=SC2015
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
