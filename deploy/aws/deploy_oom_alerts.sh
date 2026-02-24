#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Deploy parameterized OOM alert infrastructure (SNS + SSM + optional IAM role policy).

Usage:
  deploy/aws/deploy_oom_alerts.sh \
    --email <alert-email> \
    [--project pdfx] \
    [--env dev] \
    [--region us-east-1] \
    [--profile <aws-profile>] \
    [--ssm-prefix /pdfx/alerts] \
    [--instance-role-name ec2-role-name]

Notes:
  - If --instance-role-name is provided, this deploy needs CAPABILITY_NAMED_IAM.
  - SNS email subscriptions require recipient confirmation before delivery begins.
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

STACK_NAME="${PROJECT_NAME}-${ENV_NAME}-oom-alerts"
TEMPLATE_FILE="deploy/aws/oom-alerts-stack.yaml"

AWS_ARGS=(--region "$AWS_REGION")
if [[ -n "$AWS_PROFILE" ]]; then
  AWS_ARGS+=(--profile "$AWS_PROFILE")
fi

PARAMS=(
  "ProjectName=${PROJECT_NAME}"
  "EnvironmentName=${ENV_NAME}"
  "AlertEmail=${ALERT_EMAIL}"
  "AlertSsmPrefix=${SSM_PREFIX}"
  "InstanceRoleName=${INSTANCE_ROLE_NAME}"
)

echo "Deploying stack: $STACK_NAME"
aws "${AWS_ARGS[@]}" cloudformation deploy \
  --stack-name "$STACK_NAME" \
  --template-file "$TEMPLATE_FILE" \
  --parameter-overrides "${PARAMS[@]}" \
  --capabilities CAPABILITY_NAMED_IAM

echo
echo "Stack outputs:"
aws "${AWS_ARGS[@]}" cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
  --output table

echo
echo "Next step: confirm the SNS subscription email sent to $ALERT_EMAIL"
