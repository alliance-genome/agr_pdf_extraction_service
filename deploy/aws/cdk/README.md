# PDFX OOM Alerts CDK App

This CDK app deploys the same resources as `deploy/aws/oom-alerts-stack.yaml`:

- SNS topic for OOM alerts
- Email subscription
- SSM parameters for watcher configuration
- Optional IAM inline policy attachment to an existing instance role

## Deploy

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

Then confirm the SNS subscription email.

## CDK Commands

```bash
cd deploy/aws/cdk
npm install
npx cdk synth
npx cdk diff
```
