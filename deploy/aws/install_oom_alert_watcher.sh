#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Install and enable systemd OOM watcher service on a Linux host.

Usage:
  sudo deploy/aws/install_oom_alert_watcher.sh \
    [--region us-east-1] \
    [--profile <aws-profile>] \
    [--aws-credentials-file /path/to/credentials] \
    [--aws-config-file /path/to/config] \
    [--ssm-param /pdfx/alerts/sns_topic_arn] \
    [--dedupe-seconds 300]
USAGE
}

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE_NAME="${AWS_PROFILE:-}"
AWS_CREDENTIALS_FILE="${AWS_SHARED_CREDENTIALS_FILE:-}"
AWS_CONFIG_FILE_PATH="${AWS_CONFIG_FILE:-}"
SSM_PARAM="/pdfx/alerts/sns_topic_arn"
DEDUPE_SECONDS="300"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      AWS_REGION="$2"; shift 2 ;;
    --profile)
      AWS_PROFILE_NAME="$2"; shift 2 ;;
    --aws-credentials-file)
      AWS_CREDENTIALS_FILE="$2"; shift 2 ;;
    --aws-config-file)
      AWS_CONFIG_FILE_PATH="$2"; shift 2 ;;
    --ssm-param)
      SSM_PARAM="$2"; shift 2 ;;
    --dedupe-seconds)
      DEDUPE_SECONDS="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: run this script as root (sudo)" >&2
  exit 1
fi

if [[ -n "$AWS_PROFILE_NAME" ]]; then
  source_user="${SUDO_USER:-$USER}"
  source_home="$(getent passwd "$source_user" | cut -d: -f6 || true)"
  if [[ -z "$AWS_CREDENTIALS_FILE" ]] && [[ -n "$source_home" ]] && [[ -f "$source_home/.aws/credentials" ]]; then
    AWS_CREDENTIALS_FILE="$source_home/.aws/credentials"
  fi
  if [[ -z "$AWS_CONFIG_FILE_PATH" ]] && [[ -n "$source_home" ]] && [[ -f "$source_home/.aws/config" ]]; then
    AWS_CONFIG_FILE_PATH="$source_home/.aws/config"
  fi
fi

SCRIPT_SRC="deploy/aws/scripts/pdfx-oom-alert-watcher.sh"
UNIT_SRC="deploy/aws/systemd/pdfx-oom-alert-watcher.service"
SCRIPT_DST="/usr/local/bin/pdfx-oom-alert-watcher.sh"
UNIT_DST="/etc/systemd/system/pdfx-oom-alert-watcher.service"
ENV_FILE="/etc/default/pdfx-oom-alert-watcher"

install -m 0755 "$SCRIPT_SRC" "$SCRIPT_DST"
install -m 0644 "$UNIT_SRC" "$UNIT_DST"

cat > "$ENV_FILE" <<EOF
AWS_REGION=${AWS_REGION}
ALERT_TOPIC_SSM_PARAM=${SSM_PARAM}
DEDUPE_SECONDS=${DEDUPE_SECONDS}
EOF
if [[ -n "$AWS_PROFILE_NAME" ]]; then
  echo "AWS_PROFILE=${AWS_PROFILE_NAME}" >> "$ENV_FILE"
fi
if [[ -n "$AWS_CREDENTIALS_FILE" ]]; then
  echo "AWS_SHARED_CREDENTIALS_FILE=${AWS_CREDENTIALS_FILE}" >> "$ENV_FILE"
fi
if [[ -n "$AWS_CONFIG_FILE_PATH" ]]; then
  echo "AWS_CONFIG_FILE=${AWS_CONFIG_FILE_PATH}" >> "$ENV_FILE"
fi
chmod 0644 "$ENV_FILE"

systemctl daemon-reload
systemctl enable --now pdfx-oom-alert-watcher.service
systemctl --no-pager --full status pdfx-oom-alert-watcher.service || true

echo
echo "Installed. Tail logs with: journalctl -u pdfx-oom-alert-watcher -f"
