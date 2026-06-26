# AWS Infrastructure

This folder contains AWS infrastructure for the PDF extraction service.

## Test Mirror

`pdfx-test-mirror-stack.yaml` creates a production-shaped test environment:
GPU backend launch template and capped Auto Scaling Group, Fargate proxy, S3
audit bucket, environment-scoped SSM parameters, CloudWatch alarms, ALB routing,
and DNS.

Start with the runbook in `deploy/aws/pdfx-test-mirror.md`.

The backend ASG defaults to `MinSize=0`, `DesiredCapacity=0`, and `MaxSize=1`.
The proxy scales it to one instance on demand and marks failed-startup
instances unhealthy so Auto Scaling replaces them from the launch template.
An optional stopped warm pool can be enabled with `BackendWarmPoolMinSize=1`
when lower wake latency is worth the idle EBS cost.

The proxy deployer supports this environment with:

```bash
cd proxy/deploy
./deploy.sh \
  --profile ctabone \
  --region us-east-1 \
  --cluster pdfx-proxy-test \
  --service pdfx-proxy-test \
  --ssm-prefix /pdfx-test \
  --image-tag <image-tag>
```

## GitHub Actions Deploy Policy

`github-actions-deploy-policy.json` mirrors the managed IAM policy attached to
the production GitHub Actions deploy role. The deploy workflow must be able to
write `/pdfx/*` placeholder parameters before it registers a new proxy task
definition, because ASG-aware proxy revisions require both the legacy EC2
instance parameter and the backend ASG parameter to exist.

## GPU Idle Guard Alerts

`pdfx-idle-guard-stack.yaml` creates a scheduled Lambda and CloudWatch alarms
that alert if the GPU backend ASG stays running longer than expected. The
guard checks the backend ASG plus the public proxy `/api/v1/metrics` endpoint.
It publishes custom metrics under `PDFX/IdleGuard` and sends alarm/OK
notifications to an SNS topic.

Default production-oriented behavior:

- check every 5 minutes
- track continuous backend ASG runtime in SSM Parameter Store, so the clock
  does not reset if Auto Scaling replaces an EC2 instance
- alert after 45 minutes of continuous backend runtime with no queued,
  replaying, or active backend work
- alert after 240 minutes of continuous backend runtime regardless of apparent
  work
- alert if the guard Lambda errors, is throttled, or misses two consecutive
  `GuardCheckSucceeded` heartbeats
- treat proxy metrics fetch failures as idle, so a running backend plus a
  broken metrics endpoint does not silently hide spend

The existing `pdfx-dev-oom-alerts` SNS topic is already subscribed to
`ctabone@morgan.harvard.edu` and can be reused for email delivery:

```bash
deploy/aws/deploy_idle_guard.sh \
  --profile ctabone \
  --region us-east-1 \
  --project pdfx \
  --env prod \
  --backend-asg-name pdfx-backend-test \
  --proxy-metrics-url https://pdfx.alliancegenome.org/api/v1/metrics \
  --artifact-bucket agr-pdf-extraction-benchmark \
  --alarm-topic-arn arn:aws:sns:us-east-1:100225593120:pdfx-dev-oom-alerts
```

For Slack delivery, connect the same SNS topic to the team's AWS Chatbot Slack
channel configuration, or add that topic ARN to an existing Chatbot
configuration. Keep email subscribed as the fallback path.

## AWS OOM Alerts (IaC)

This folder also provides a parameterized, redeployable OOM alerting setup for
the PDF extraction service.

## What It Creates

- SNS topic for OOM alerts
- Email subscription endpoint
- SSM parameters for watcher configuration
- Optional IAM policy attachment to an existing EC2 instance role

## Deploy Infrastructure (CloudFormation Script)

```bash
deploy/aws/deploy_oom_alerts.sh \
  --email <alert-email> \
  --project pdfx \
  --env dev \
  --region us-east-1 \
  --profile <aws-profile> \
  --ssm-prefix /pdfx/alerts \
  --instance-role-name <ec2-role-name>
```

Then confirm the SNS email subscription from the message sent by AWS.

## Deploy Infrastructure (CDK App)

```bash
deploy/aws/cdk/deploy.sh \
  --email <alert-email> \
  --project pdfx \
  --env dev \
  --region us-east-1 \
  --profile <aws-profile> \
  --ssm-prefix /pdfx/alerts \
  --instance-role-name <ec2-role-name>
```

CDK source is under `deploy/aws/cdk/`.

## Install Host Watcher

```bash
sudo deploy/aws/install_oom_alert_watcher.sh \
  --region us-east-1 \
  --profile <aws-profile> \
  --aws-credentials-file ~/.aws/credentials \
  --aws-config-file ~/.aws/config \
  --ssm-param /pdfx/alerts/sns_topic_arn
```

## What The Watcher Monitors

- Docker OOM events (`docker events --filter event=oom`)
- Kernel OOM signals via `journalctl -k`

When an OOM is detected, it publishes an SNS message containing host and event context.

## Sensitive Data Guidance

- This SNS email path does not require secrets.
- Keep operational config in SSM Parameter Store (`/pdfx/alerts/*`).
- If you add webhook-based alerting later, keep webhook credentials in Secrets Manager and store only the secret ARN in SSM.

## CDK Notes

- The CDK app currently targets the same resources and parameters as the CloudFormation template.
- Teams can standardize on CDK deployment while preserving compatibility with raw CloudFormation for emergency/recovery workflows.
