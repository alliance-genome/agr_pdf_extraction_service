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
