#!/usr/bin/env bash
# deploy.sh — Resolve SSM parameters and register the ECS task definition.
#
# Usage:
#   ./deploy.sh [--profile PROFILE] [--dry-run]
#
# Reads infrastructure values from AWS SSM Parameter Store under /pdfx/*,
# substitutes them into the template files, and registers the task definition.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REGION="us-east-1"
PROFILE=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile) PROFILE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

AWS_CMD="aws"
[[ -n "$PROFILE" ]] && AWS_CMD="aws --profile $PROFILE"

echo "==> Reading SSM parameters from /pdfx/..."
read_param() {
    $AWS_CMD ssm get-parameter --name "$1" --query "Parameter.Value" --output text
}

AWS_ACCOUNT_ID=$(read_param /pdfx/aws-account-id)
EC2_INSTANCE_ID=$(read_param /pdfx/ec2-instance-id)
EXECUTION_ROLE_NAME=$(read_param /pdfx/execution-role-name)
TASK_ROLE_NAME=$(read_param /pdfx/task-role-name)
QUEUE_S3_BUCKET=$(read_param /pdfx/audit-s3-bucket)

EXECUTION_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${EXECUTION_ROLE_NAME}"
TASK_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${TASK_ROLE_NAME}"
ECR_IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/agr_pdfx_proxy:latest"

echo "    Account:        ${AWS_ACCOUNT_ID}"
echo "    EC2 Instance:   ${EC2_INSTANCE_ID}"
echo "    Execution Role: ${EXECUTION_ROLE_ARN}"
echo "    Task Role:      ${TASK_ROLE_ARN}"
echo "    ECR Image:      ${ECR_IMAGE}"
echo "    Queue Bucket:   ${QUEUE_S3_BUCKET}"

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
$AWS_CMD ecs register-task-definition \
    --region "$REGION" \
    --cli-input-json "$TASK_DEF" \
    --query "taskDefinition.taskDefinitionArn" \
    --output text

echo "==> Done."
