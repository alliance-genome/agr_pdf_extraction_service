#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-}"
ALERT_TOPIC_SSM_PARAM="${ALERT_TOPIC_SSM_PARAM:-/pdfx/alerts/sns_topic_arn}"
DEDUPE_SECONDS="${DEDUPE_SECONDS:-300}"
HOST_ID="${HOST_ID:-$(hostname -f 2>/dev/null || hostname)}"

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI is required" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker CLI is required" >&2
  exit 1
fi
if ! command -v journalctl >/dev/null 2>&1; then
  echo "ERROR: journalctl is required" >&2
  exit 1
fi

if [[ -z "$AWS_REGION" ]]; then
  AWS_REGION="$(aws configure get region 2>/dev/null || true)"
fi
if [[ -z "$AWS_REGION" ]]; then
  AWS_REGION="us-east-1"
fi

SNS_TOPIC_ARN="$(aws ssm get-parameter --name "$ALERT_TOPIC_SSM_PARAM" --query 'Parameter.Value' --output text --region "$AWS_REGION")"

LAST_KEY=""
LAST_TS=0

publish_alert() {
  local source="$1"
  local key="$2"
  local body="$3"
  local now
  now="$(date +%s)"

  if [[ "$key" == "$LAST_KEY" ]] && (( now - LAST_TS < DEDUPE_SECONDS )); then
    return 0
  fi
  LAST_KEY="$key"
  LAST_TS="$now"

  local subject="[PDFX][${HOST_ID}] OOM alert (${source})"
  aws sns publish \
    --region "$AWS_REGION" \
    --topic-arn "$SNS_TOPIC_ARN" \
    --subject "$subject" \
    --message "$body" >/dev/null
}

monitor_docker_oom() {
  while true; do
    docker events --filter event=oom --format '{{.Time}}|{{.Actor.ID}}|{{.Actor.Attributes.name}}' |
      while IFS='|' read -r ts container_id container_name; do
        [[ -z "$ts" ]] && continue
        local_name="${container_name:-unknown-container}"
        local_id="${container_id:-unknown-id}"
        msg="time=${ts}
host=${HOST_ID}
source=docker-events
container=${local_name}
container_id=${local_id}
message=Docker reported an OOM event for container ${local_name}"
        publish_alert "docker" "docker:${local_id}" "$msg"
      done
    sleep 2
  done
}

monitor_kernel_oom() {
  while true; do
    journalctl -kf -o cat |
      while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        if [[ "$line" =~ [Oo]ut[[:space:]]of[[:space:]]memory|oom-kill|Killed[[:space:]]process ]]; then
          local_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
          msg="time=${local_ts}
host=${HOST_ID}
source=kernel-log
message=${line}"
          publish_alert "kernel" "kernel:${line}" "$msg"
        fi
      done
    sleep 2
  done
}

monitor_docker_oom &
PID_DOCKER=$!
monitor_kernel_oom &
PID_KERNEL=$!

wait -n "$PID_DOCKER" "$PID_KERNEL"
