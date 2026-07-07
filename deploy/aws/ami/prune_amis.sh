#!/usr/bin/env bash
set -euo pipefail

# Pure selection: given keep_n, a protected ami id, and a newline list of
# "creationDate ami-id" (any order), print ami-ids to deregister.
select_amis_to_deregister() {
  local keep_n="$1" protected="$2" list="$3"
  # Sort by date desc, drop the newest keep_n, then exclude the protected id.
  printf '%s\n' "$list" | sort -r | awk 'NF' | tail -n +"$((keep_n + 1))" \
    | awk '{print $2}' | grep -Fvx "$protected" || true
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
