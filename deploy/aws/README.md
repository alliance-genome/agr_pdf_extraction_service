# AWS OOM Alerts (IaC)

This folder provides a parameterized, redeployable OOM alerting setup for the PDF extraction service.

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
