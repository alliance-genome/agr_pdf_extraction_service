#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/../lib/baked_fastpath.sh"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fail=0
check() { if [ "$1" = "$2" ]; then echo "ok: $3"; else echo "FAIL: $3 (got '$1' want '$2')"; fail=1; fi; }

# Stub `aws ecr describe-images` to return a fixed digest for tag sha-AAA.
# shellcheck disable=SC2317  # invoked indirectly by the sourced function under test
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
