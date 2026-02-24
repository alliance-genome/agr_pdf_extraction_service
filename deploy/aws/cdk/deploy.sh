#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Deploy OOM alert stack via CDK.

Usage:
  deploy/aws/cdk/deploy.sh \
    --email <alert-email> \
    [--project pdfx] \
    [--env dev] \
    [--region us-east-1] \
    [--profile <aws-profile>] \
    [--ssm-prefix /pdfx/alerts] \
    [--instance-role-name ec2-role-name]
USAGE
}

PROJECT_NAME="pdfx"
ENV_NAME="dev"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-}"
SSM_PREFIX="/pdfx/alerts"
INSTANCE_ROLE_NAME=""
ALERT_EMAIL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --email)
      ALERT_EMAIL="$2"; shift 2 ;;
    --project)
      PROJECT_NAME="$2"; shift 2 ;;
    --env)
      ENV_NAME="$2"; shift 2 ;;
    --region)
      AWS_REGION="$2"; shift 2 ;;
    --profile)
      AWS_PROFILE="$2"; shift 2 ;;
    --ssm-prefix)
      SSM_PREFIX="$2"; shift 2 ;;
    --instance-role-name)
      INSTANCE_ROLE_NAME="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$ALERT_EMAIL" ]]; then
  echo "ERROR: --email is required" >&2
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d node_modules ]]; then
  npm install
fi

export CDK_DEFAULT_REGION="$AWS_REGION"
if [[ -n "$AWS_PROFILE" ]]; then
  export AWS_PROFILE
fi

npx cdk deploy OomAlertsStack \
  --require-approval never \
  --parameters ProjectName="$PROJECT_NAME" \
  --parameters EnvironmentName="$ENV_NAME" \
  --parameters AlertEmail="$ALERT_EMAIL" \
  --parameters AlertSsmPrefix="$SSM_PREFIX" \
  --parameters InstanceRoleName="$INSTANCE_ROLE_NAME"
