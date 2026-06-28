#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Deploy the PDFX GPU idle guard (scheduled Lambda + CloudWatch alarms).

Usage:
  deploy/aws/deploy_idle_guard.sh \
    --alarm-topic-arn <sns-topic-arn> \
    --artifact-bucket <s3-bucket> \
    [--backend-asg-name pdfx-backend] \
    [--proxy-metrics-url https://pdfx.alliancegenome.org/api/v1/metrics] \
    [--idle-alert-after-minutes 45] \
    [--absolute-alert-after-minutes 240] \
    [--project pdfx] \
    [--env prod] \
    [--region us-east-1] \
    [--profile <aws-profile>]

The idle alarm fires when the backend ASG has been running longer than the
idle threshold while proxy metrics report no queued, replaying, or active work.
The absolute alarm is a wider cap for any unusually long runtime.
USAGE
}

PROJECT_NAME="pdfx"
ENV_NAME="prod"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-}"
STACK_NAME=""
BACKEND_ASG_NAME="pdfx-backend"
PROXY_METRICS_URL="https://pdfx.alliancegenome.org/api/v1/metrics"
IDLE_ALERT_AFTER_MINUTES="45"
ABSOLUTE_ALERT_AFTER_MINUTES="240"
SCHEDULE_EXPRESSION="rate(5 minutes)"
METRICS_TIMEOUT_SECONDS="5"
TREAT_METRICS_FETCH_FAILURE_AS_IDLE="true"
ALARM_TOPIC_ARN=""
ARTIFACT_BUCKET=""
ARTIFACT_PREFIX="pdfx-idle-guard/lambda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --alarm-topic-arn)
      ALARM_TOPIC_ARN="$2"; shift 2 ;;
    --artifact-bucket)
      ARTIFACT_BUCKET="$2"; shift 2 ;;
    --artifact-prefix)
      ARTIFACT_PREFIX="$2"; shift 2 ;;
    --backend-asg-name)
      BACKEND_ASG_NAME="$2"; shift 2 ;;
    --proxy-metrics-url)
      PROXY_METRICS_URL="$2"; shift 2 ;;
    --idle-alert-after-minutes)
      IDLE_ALERT_AFTER_MINUTES="$2"; shift 2 ;;
    --absolute-alert-after-minutes)
      ABSOLUTE_ALERT_AFTER_MINUTES="$2"; shift 2 ;;
    --schedule-expression)
      SCHEDULE_EXPRESSION="$2"; shift 2 ;;
    --metrics-timeout-seconds)
      METRICS_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --treat-metrics-fetch-failure-as-idle)
      TREAT_METRICS_FETCH_FAILURE_AS_IDLE="$2"; shift 2 ;;
    --project)
      PROJECT_NAME="$2"; shift 2 ;;
    --env)
      ENV_NAME="$2"; shift 2 ;;
    --region)
      AWS_REGION="$2"; shift 2 ;;
    --profile)
      AWS_PROFILE="$2"; shift 2 ;;
    --stack-name)
      STACK_NAME="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$ALARM_TOPIC_ARN" ]]; then
  echo "ERROR: --alarm-topic-arn is required" >&2
  usage
  exit 2
fi

if [[ -z "$ARTIFACT_BUCKET" ]]; then
  echo "ERROR: --artifact-bucket is required" >&2
  usage
  exit 2
fi

if [[ -z "$STACK_NAME" ]]; then
  STACK_NAME="${PROJECT_NAME}-${ENV_NAME}-idle-guard"
fi

TEMPLATE_FILE="deploy/aws/pdfx-idle-guard-stack.yaml"
LAMBDA_SOURCE="deploy/aws/lambda/pdfx_idle_guard.py"

AWS_ARGS=(--region "$AWS_REGION")
if [[ -n "$AWS_PROFILE" ]]; then
  AWS_ARGS+=(--profile "$AWS_PROFILE")
fi

PARAMS=(
  "ProjectName=${PROJECT_NAME}"
  "EnvironmentName=${ENV_NAME}"
  "BackendAsgName=${BACKEND_ASG_NAME}"
  "ProxyMetricsUrl=${PROXY_METRICS_URL}"
  "IdleAlertAfterMinutes=${IDLE_ALERT_AFTER_MINUTES}"
  "AbsoluteAlertAfterMinutes=${ABSOLUTE_ALERT_AFTER_MINUTES}"
  "ScheduleExpression=${SCHEDULE_EXPRESSION}"
  "MetricsTimeoutSeconds=${METRICS_TIMEOUT_SECONDS}"
  "TreatMetricsFetchFailureAsIdle=${TREAT_METRICS_FETCH_FAILURE_AS_IDLE}"
  "AlarmSnsTopicArn=${ALARM_TOPIC_ARN}"
)

ARTIFACT_HASH="$(sha256sum "$LAMBDA_SOURCE" | awk '{print $1}')"
ARTIFACT_KEY="${ARTIFACT_PREFIX%/}/${ARTIFACT_HASH}.zip"
TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

cp "$LAMBDA_SOURCE" "$TMP_DIR/index.py"
(cd "$TMP_DIR" && zip -q idle-guard.zip index.py)

echo "Uploading Lambda artifact: s3://${ARTIFACT_BUCKET}/${ARTIFACT_KEY}"
aws "${AWS_ARGS[@]}" s3 cp "$TMP_DIR/idle-guard.zip" "s3://${ARTIFACT_BUCKET}/${ARTIFACT_KEY}"

PARAMS+=(
  "LambdaArtifactBucket=${ARTIFACT_BUCKET}"
  "LambdaArtifactKey=${ARTIFACT_KEY}"
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
