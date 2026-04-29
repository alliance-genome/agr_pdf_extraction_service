#!/usr/bin/env bash
# deploy.sh — Resolve SSM parameters, register ECS task definition,
# and roll the ECS service.
#
# Usage:
#   ./deploy.sh [--profile PROFILE] [--region REGION] [--cluster NAME] [--service NAME]
#               [--ssm-prefix PREFIX] [--image-tag TAG] [--dry-run]
#               [--no-update-service] [--no-wait]
#
# Reads infrastructure values from AWS SSM Parameter Store under the selected
# prefix, substitutes them into the template files, and registers the task
# definition. By default, it also updates the ECS service to the new revision
# and waits for service stability.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REGION="us-east-1"
PROFILE=""
IMAGE_TAG="latest"
DRY_RUN=false
CLUSTER_NAME="pdfx-proxy"
SERVICE_NAME="pdfx-proxy"
SSM_PREFIX="/pdfx"
TASK_FAMILY=""
CONTAINER_NAME=""
LOG_GROUP=""
QUEUE_PREFIX=""
UPDATE_SERVICE=true
WAIT_FOR_STABLE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile) PROFILE="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --cluster) CLUSTER_NAME="$2"; shift 2 ;;
        --service) SERVICE_NAME="$2"; shift 2 ;;
        --ssm-prefix) SSM_PREFIX="$2"; shift 2 ;;
        --task-family) TASK_FAMILY="$2"; shift 2 ;;
        --container-name) CONTAINER_NAME="$2"; shift 2 ;;
        --log-group) LOG_GROUP="$2"; shift 2 ;;
        --queue-prefix) QUEUE_PREFIX="$2"; shift 2 ;;
        --image-tag) IMAGE_TAG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --no-update-service) UPDATE_SERVICE=false; shift ;;
        --no-wait) WAIT_FOR_STABLE=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

AWS_CMD=(aws --region "$REGION")
[[ -n "$PROFILE" ]] && AWS_CMD=(aws --profile "$PROFILE" --region "$REGION")

normalize_ssm_prefix() {
    local prefix="$1"
    prefix="/${prefix#/}"
    prefix="${prefix%/}"
    if [[ -z "$prefix" || "$prefix" == "/" ]]; then
        echo "ERROR: --ssm-prefix must not be empty" >&2
        return 1
    fi
    echo "$prefix"
}

SSM_PREFIX="$(normalize_ssm_prefix "$SSM_PREFIX")"
SSM_PARAMETER_RESOURCE="${SSM_PREFIX#/}"

TASK_FAMILY="${TASK_FAMILY:-$SERVICE_NAME}"
CONTAINER_NAME="${CONTAINER_NAME:-$SERVICE_NAME}"
LOG_GROUP="${LOG_GROUP:-/ecs/$SERVICE_NAME}"
QUEUE_PREFIX="${QUEUE_PREFIX:-${SERVICE_NAME}-queue}"

echo "==> Reading SSM parameters from ${SSM_PREFIX}/..."
read_param() {
    "${AWS_CMD[@]}" ssm get-parameter --name "$1" --query "Parameter.Value" --output text
}

# Idempotently ensure an SSM String parameter exists so the ECS task
# definition's `secrets` block can resolve it. Existing values are never
# touched (no --overwrite). Created with a single space because SSM rejects
# empty-string values; the proxy's config layer .strip()s the result back to
# "" so the auth allow-lists remain inactive until an operator updates the
# parameter with real values.
#
# Only ParameterNotFound triggers the placeholder write. Other errors
# (IAM denied, throttle, network) are surfaced so a misconfigured deploy
# fails loudly with a useful message instead of being misinterpreted as
# "param missing → create".
ensure_ssm_param() {
    local name="$1"
    local err_file
    err_file=$(mktemp)
    if "${AWS_CMD[@]}" ssm get-parameter --name "$name" >/dev/null 2>"$err_file"; then
        rm -f "$err_file"
        return 0
    fi
    if ! grep -q 'ParameterNotFound' "$err_file"; then
        echo "ERROR: failed to read SSM parameter ${name}:" >&2
        cat "$err_file" >&2
        rm -f "$err_file"
        return 1
    fi
    rm -f "$err_file"
    if $DRY_RUN; then
        echo "    Would create placeholder SSM parameter ${name} (was missing)"
        return 0
    fi
    echo "    Creating placeholder SSM parameter ${name} (was missing)"
    "${AWS_CMD[@]}" ssm put-parameter \
        --name "$name" \
        --type "String" \
        --value " " \
        >/dev/null
}

AWS_ACCOUNT_ID=$(read_param "${SSM_PREFIX}/aws-account-id")
EC2_INSTANCE_ID=$(read_param "${SSM_PREFIX}/ec2-instance-id")
EXECUTION_ROLE_NAME=$(read_param "${SSM_PREFIX}/execution-role-name")
TASK_ROLE_NAME=$(read_param "${SSM_PREFIX}/task-role-name")
QUEUE_S3_BUCKET=$(read_param "${SSM_PREFIX}/audit-s3-bucket")

# Ensure optional auth allow-list parameters exist so the task definition's
# `secrets` references resolve on first deploy. Operator must have
# ssm:PutParameter for the environment's SSM prefix (in addition to
# ssm:GetParameter) for these
# placeholder writes; populate real values via `aws ssm put-parameter
# --overwrite` per environment when the allow-list should be enabled.
ensure_ssm_param "${SSM_PREFIX}/cognito-accepted-scopes"
ensure_ssm_param "${SSM_PREFIX}/cognito-accepted-client-ids"

EXECUTION_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${EXECUTION_ROLE_NAME}"
TASK_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${TASK_ROLE_NAME}"
ECR_IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/agr_pdfx_proxy:${IMAGE_TAG}"

echo "    Account:        ${AWS_ACCOUNT_ID}"
echo "    EC2 Instance:   ${EC2_INSTANCE_ID}"
echo "    Execution Role: ${EXECUTION_ROLE_ARN}"
echo "    Task Role:      ${TASK_ROLE_ARN}"
echo "    ECR Image:      ${ECR_IMAGE}"
echo "    Image Tag:      ${IMAGE_TAG}"
echo "    Queue Bucket:   ${QUEUE_S3_BUCKET}"
echo "    SSM Prefix:     ${SSM_PREFIX}"
echo "    ECS Cluster:    ${CLUSTER_NAME}"
echo "    ECS Service:    ${SERVICE_NAME}"
echo "    Task Family:    ${TASK_FAMILY}"
echo "    Container:      ${CONTAINER_NAME}"
echo "    Log Group:      ${LOG_GROUP}"
echo "    Queue Prefix:   ${QUEUE_PREFIX}"

# --- Generate task definition ---
echo "==> Generating task definition..."
TASK_DEF=$(sed \
    -e "s|\${TASK_FAMILY}|${TASK_FAMILY}|g" \
    -e "s|\${CONTAINER_NAME}|${CONTAINER_NAME}|g" \
    -e "s|\${EXECUTION_ROLE_ARN}|${EXECUTION_ROLE_ARN}|g" \
    -e "s|\${TASK_ROLE_ARN}|${TASK_ROLE_ARN}|g" \
    -e "s|\${ECR_IMAGE}|${ECR_IMAGE}|g" \
    -e "s|\${QUEUE_S3_BUCKET}|${QUEUE_S3_BUCKET}|g" \
    -e "s|\${QUEUE_S3_PREFIX}|${QUEUE_PREFIX}|g" \
    -e "s|\${QUEUE_S3_REGION}|${REGION}|g" \
    -e "s|\${SSM_PREFIX}|${SSM_PREFIX}|g" \
    -e "s|\${LOG_GROUP}|${LOG_GROUP}|g" \
    -e "s|\${LOG_REGION}|${REGION}|g" \
    "${SCRIPT_DIR}/task-definition.template.json")

# --- Generate IAM policy ---
echo "==> Generating IAM policy..."
IAM_POLICY=$(sed \
    -e "s|\${AWS_ACCOUNT_ID}|${AWS_ACCOUNT_ID}|g" \
    -e "s|\${EC2_INSTANCE_ID}|${EC2_INSTANCE_ID}|g" \
    -e "s|\${QUEUE_S3_BUCKET}|${QUEUE_S3_BUCKET}|g" \
    -e "s|\${REGION}|${REGION}|g" \
    -e "s|\${SSM_PARAMETER_RESOURCE}|${SSM_PARAMETER_RESOURCE}|g" \
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
TASK_DEF_ARN=$("${AWS_CMD[@]}" ecs register-task-definition \
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
"${AWS_CMD[@]}" ecs update-service \
    --cluster "$CLUSTER_NAME" \
    --service "$SERVICE_NAME" \
    --task-definition "$TASK_DEF_ARN" \
    --force-new-deployment \
    --query "service.{serviceName:serviceName,taskDefinition:taskDefinition}" \
    --output json

if $WAIT_FOR_STABLE; then
    echo "==> Waiting for service stability..."
    "${AWS_CMD[@]}" ecs wait services-stable \
        --cluster "$CLUSTER_NAME" \
        --services "$SERVICE_NAME"
    echo "==> Service is stable."
else
    echo "==> Skipping service wait (--no-wait)."
fi

echo "==> Done."
