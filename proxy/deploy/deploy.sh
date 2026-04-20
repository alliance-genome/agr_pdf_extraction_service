#!/usr/bin/env bash
# deploy.sh — Resolve SSM parameters, register ECS task definition,
# and roll the ECS service.
#
# Usage:
#   ./deploy.sh [--profile PROFILE] [--region REGION] [--cluster NAME] [--service NAME]
#               [--dry-run] [--no-update-service] [--no-wait]
#
# Reads infrastructure values from AWS SSM Parameter Store under /pdfx/*,
# substitutes them into the template files, and registers the task definition.
# By default, it also updates the ECS service to the new revision and waits
# for service stability.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REGION="us-east-1"
PROFILE=""
DRY_RUN=false
CLUSTER_NAME="pdfx-proxy"
SERVICE_NAME="pdfx-proxy"
UPDATE_SERVICE=true
WAIT_FOR_STABLE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile) PROFILE="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --cluster) CLUSTER_NAME="$2"; shift 2 ;;
        --service) SERVICE_NAME="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --no-update-service) UPDATE_SERVICE=false; shift ;;
        --no-wait) WAIT_FOR_STABLE=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

AWS_CMD="aws"
[[ -n "$PROFILE" ]] && AWS_CMD="aws --profile $PROFILE"

echo "==> Reading SSM parameters from /pdfx/..."
read_param() {
    $AWS_CMD ssm get-parameter --name "$1" --query "Parameter.Value" --output text
}

# Idempotently ensure an SSM String parameter exists so the ECS task
# definition's `secrets` block can resolve it. Existing values are never
# touched (no --overwrite). Created with a single space because SSM rejects
# empty-string values; the proxy's config layer .strip()s the result back to
# "" so the auth allow-lists remain inactive until an operator updates the
# parameter with real values.
ensure_ssm_param() {
    local name="$1"
    if $AWS_CMD ssm get-parameter --name "$name" >/dev/null 2>&1; then
        return 0
    fi
    echo "    Creating placeholder SSM parameter ${name} (was missing)"
    $AWS_CMD ssm put-parameter \
        --name "$name" \
        --type "String" \
        --value " " \
        >/dev/null
}

AWS_ACCOUNT_ID=$(read_param /pdfx/aws-account-id)
EC2_INSTANCE_ID=$(read_param /pdfx/ec2-instance-id)
EXECUTION_ROLE_NAME=$(read_param /pdfx/execution-role-name)
TASK_ROLE_NAME=$(read_param /pdfx/task-role-name)
QUEUE_S3_BUCKET=$(read_param /pdfx/audit-s3-bucket)

# Ensure optional auth allow-list parameters exist so the task definition's
# `secrets` references resolve on first deploy. Operator must have
# ssm:PutParameter for /pdfx/* (in addition to ssm:GetParameter) for these
# placeholder writes; populate real values via `aws ssm put-parameter
# --overwrite` per environment when the allow-list should be enabled.
ensure_ssm_param /pdfx/cognito-accepted-scopes
ensure_ssm_param /pdfx/cognito-accepted-client-ids

EXECUTION_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${EXECUTION_ROLE_NAME}"
TASK_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${TASK_ROLE_NAME}"
ECR_IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/agr_pdfx_proxy:latest"

echo "    Account:        ${AWS_ACCOUNT_ID}"
echo "    EC2 Instance:   ${EC2_INSTANCE_ID}"
echo "    Execution Role: ${EXECUTION_ROLE_ARN}"
echo "    Task Role:      ${TASK_ROLE_ARN}"
echo "    ECR Image:      ${ECR_IMAGE}"
echo "    Queue Bucket:   ${QUEUE_S3_BUCKET}"
echo "    ECS Cluster:    ${CLUSTER_NAME}"
echo "    ECS Service:    ${SERVICE_NAME}"

# --- Generate task definition ---
echo "==> Generating task definition..."
TASK_DEF=$(sed \
    -e "s|\${EXECUTION_ROLE_ARN}|${EXECUTION_ROLE_ARN}|g" \
    -e "s|\${TASK_ROLE_ARN}|${TASK_ROLE_ARN}|g" \
    -e "s|\${ECR_IMAGE}|${ECR_IMAGE}|g" \
    -e "s|\${QUEUE_S3_BUCKET}|${QUEUE_S3_BUCKET}|g" \
    "${SCRIPT_DIR}/task-definition.template.json")

# --- Generate IAM policy ---
echo "==> Generating IAM policy..."
IAM_POLICY=$(sed \
    -e "s|\${AWS_ACCOUNT_ID}|${AWS_ACCOUNT_ID}|g" \
    -e "s|\${EC2_INSTANCE_ID}|${EC2_INSTANCE_ID}|g" \
    -e "s|\${QUEUE_S3_BUCKET}|${QUEUE_S3_BUCKET}|g" \
    -e "s|\${REGION}|${REGION}|g" \
    "${SCRIPT_DIR}/iam-policy.template.json")

if $DRY_RUN; then
    echo ""
    echo "=== Task Definition (dry-run) ==="
    echo "$TASK_DEF" | python3 -m json.tool
    echo ""
    echo "=== IAM Policy (dry-run) ==="
    echo "$IAM_POLICY" | python3 -m json.tool
    echo ""
    echo "Dry run complete. No changes made."
    exit 0
fi

# --- Register task definition ---
echo "==> Registering ECS task definition..."
TASK_DEF_ARN=$($AWS_CMD ecs register-task-definition \
    --region "$REGION" \
    --cli-input-json "$TASK_DEF" \
    --query "taskDefinition.taskDefinitionArn" \
    --output text)

echo "    Registered: ${TASK_DEF_ARN}"

if ! $UPDATE_SERVICE; then
    echo "==> Skipping ECS service update (--no-update-service)."
    echo "==> Done."
    exit 0
fi

echo "==> Updating ECS service to new task definition..."
$AWS_CMD ecs update-service \
    --region "$REGION" \
    --cluster "$CLUSTER_NAME" \
    --service "$SERVICE_NAME" \
    --task-definition "$TASK_DEF_ARN" \
    --force-new-deployment \
    --query "service.{serviceName:serviceName,taskDefinition:taskDefinition}" \
    --output json

if $WAIT_FOR_STABLE; then
    echo "==> Waiting for service stability..."
    $AWS_CMD ecs wait services-stable \
        --region "$REGION" \
        --cluster "$CLUSTER_NAME" \
        --services "$SERVICE_NAME"
    echo "==> Service is stable."
else
    echo "==> Skipping service wait (--no-wait)."
fi

echo "==> Done."
