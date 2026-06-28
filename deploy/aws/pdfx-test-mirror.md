# PDFX Test Mirror

This runbook creates a test sibling for the production PDF extraction service
without reusing production mutable state. It mirrors the runtime shape:

- GPU backend managed by an EC2 Auto Scaling Group (`g5.4xlarge`, same Deep Learning GPU AMI family)
- Docker Compose backend stack on the EC2 host
- Fargate proxy in front of the backend
- S3-backed durable proxy queue
- S3 audit/artifact bucket with tagged temporary image lifecycle
- Cognito-protected public hostname through the shared `agr-services` ALB
- environment-scoped SSM parameters under `/pdfx-test/*`
- CloudWatch alarms for backend startup timeouts and ASG replacement requests
- optional GPU idle guard alerts from `deploy/aws/pdfx-idle-guard-stack.yaml`

The template is `deploy/aws/pdfx-test-mirror-stack.yaml`.

## Production Inventory Used

These values were discovered from the current production account and are set
as template defaults so the test stack can be a direct mirror:

| Area | Production value mirrored by default |
| --- | --- |
| Account / region | `100225593120`, `us-east-1` |
| VPC | `vpc-55522232` |
| Backend subnet | `subnet-af62dca3` |
| Fargate subnets | `subnet-04262fc338f638054`, `subnet-0d4703177afb1797d` |
| Fargate default SG | `sg-21ac675b` |
| Backend default/VPN SGs | `sg-21ac675b`, `sg-006b41eff1820ad53` |
| ALB | `agr-services-1974539419.us-east-1.elb.amazonaws.com` |
| ALB HTTPS listener | `arn:aws:elasticloadbalancing:us-east-1:100225593120:listener/app/agr-services/e75691865c2bbfb1/c9cd61e1f8237d0e` |
| Public hosted zone | `Z3IZ3D6V94JEC2` (`alliancegenome.org`) |
| Backend AMI | `ami-00c6ddd550364d6c3` |
| Backend instance type | `g5.4xlarge` |
| EC2 key pair | `pedro-benchmark-key` |

The test stack intentionally does not clone the production EBS volume or reuse
the production audit bucket. That avoids copying uploads, caches, logs, and
secrets into the test environment.

## Before Deploying

Create the backend environment file as a SecureString. Start from production's
backend `.env`, but remove legacy toggles such as `MARKER_EXTRACT_IMAGES`; image
extraction is now request-scoped through `extract_images`. Also remove any
production storage overrides such as `AUDIT_S3_BUCKET`, `AUDIT_S3_PREFIX`,
`AUDIT_S3_BUCKET_SSM_PARAM`, `IMAGE_URL_TTL_SECONDS`,
`IMAGE_RETENTION_TTL_SECONDS`, and `AWS_DEFAULT_REGION`. The bootstrap scrubs
and replaces those values as a second guardrail.

The bootstrap appends the test audit values automatically, so they do not need
to be present in the SecureString:

```bash
aws ssm put-parameter \
  --profile ctabone \
  --region us-east-1 \
  --name /pdfx-test/backend-env \
  --type SecureString \
  --overwrite \
  --value "$(cat backend.env)"
```

Fetch the production Cognito pool ID without writing it into the repo:

```bash
COGNITO_USER_POOL_ID="$(
  aws ssm get-parameter \
    --profile ctabone \
    --region us-east-1 \
    --name /pdfx/cognito-user-pool-id \
    --query Parameter.Value \
    --output text
)"
```

## Deploy The Mirror Stack

This creates the test backend launch template and Auto Scaling Group, proxy
service, S3 bucket, IAM roles, ALB rule, DNS record, CloudWatch alarms, and
`/pdfx-test/*` parameters. It does not modify production resources other than
adding a new host rule to the shared ALB and a new DNS record.

```bash
aws cloudformation deploy \
  --profile ctabone \
  --region us-east-1 \
  --stack-name pdfx-test-mirror \
  --template-file deploy/aws/pdfx-test-mirror-stack.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    CognitoUserPoolId="$COGNITO_USER_POOL_ID" \
    DeployBackendOnBoot=true \
    BackendMaxSize=1 \
    BackendGitRef=main
```

For testing a feature branch, push the branch first and set
`BackendGitRef=<branch-or-sha>`.

The backend ASG defaults to `BackendMinSize=0`, `BackendDesiredCapacity=0`,
`BackendMaxSize=1`, and `BackendWarmPoolMinSize=1`. The proxy scales desired
capacity to `1` on wake and back to `0` after idle shutdown. The warm pool keeps
one stopped, already-bootstrapped backend instance so Docker images and ML model
caches survive idle shutdown without paying for a running GPU. Keep
`BackendMaxSize=1` for strict cost control. Use `BackendMaxSize=2` only for
controlled testing of launch-before-terminate behavior.

The proxy's bounded replacement wait defaults to
`AsgStartupReplacementAttempts=1`, published to
`/<ssm-prefix>/asg-startup-replacement-attempts`.

## Deploy Idle Guard Alerts

Use `deploy/aws/pdfx-idle-guard-stack.yaml` when you want a separate
belt-and-suspenders alert if a backend ASG stays running after it should have
scaled down. The guard stores the continuous ASG runtime in SSM Parameter Store,
emits a `GuardCheckSucceeded` heartbeat, and alarms if the guard itself stops
checking. For the test mirror:

```bash
deploy/aws/deploy_idle_guard.sh \
  --profile ctabone \
  --region us-east-1 \
  --project pdfx \
  --env test \
  --backend-asg-name pdfx-backend-test \
  --proxy-metrics-url https://pdfx-test.alliancegenome.org/api/v1/metrics \
  --artifact-bucket agr-pdf-extraction-test \
  --alarm-topic-arn <sns-topic-arn>
```

For production, use the production proxy metrics URL and a confirmed SNS topic
such as `pdfx-dev-oom-alerts`, which currently emails
`ctabone@morgan.harvard.edu`.

If cold boot time becomes too expensive, set `BackendWarmPoolMinSize=1`.
That keeps one pre-initialized backend instance in the ASG warm pool in
`Stopped` state, so you pay for EBS while idle but not for running GPU compute.

## Deploy Proxy Changes To Test

The proxy deploy script now supports environment-scoped resources. After
building and pushing an image tag to ECR, roll only the test service:

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

The defaults still target production (`pdfx-proxy` and `/pdfx`) for existing
release workflows.

The CloudFormation stack uses `__PDFX_EMPTY__` as the disabled placeholder for
optional auth allow-lists because SSM String parameters cannot be empty; the
proxy ignores that exact placeholder.

## Backend ASG Operations

Get the backend ASG name and scale it up manually when needed:

```bash
TEST_BACKEND_ASG="$(
  aws ssm get-parameter \
    --profile ctabone \
    --region us-east-1 \
    --name /pdfx-test/backend-asg-name \
    --query Parameter.Value \
    --output text
)"

aws autoscaling update-auto-scaling-group \
  --profile ctabone \
  --region us-east-1 \
  --auto-scaling-group-name "$TEST_BACKEND_ASG" \
  --desired-capacity 1
```

Find the current instance and check the bootstrap log:

```bash
TEST_INSTANCE_ID="$(
  aws autoscaling describe-auto-scaling-groups \
    --profile ctabone \
    --region us-east-1 \
    --auto-scaling-group-names "$TEST_BACKEND_ASG" \
    --query 'AutoScalingGroups[0].Instances[?LifecycleState==`InService` || starts_with(LifecycleState, `Pending`)].InstanceId | [0]' \
    --output text
)"

aws ec2 describe-instances \
  --profile ctabone \
  --region us-east-1 \
  --instance-ids "$TEST_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text

ssh ec2-user@<test-public-ip> 'sudo tail -n 200 /var/log/pdfx-bootstrap.log'
```

Redeploy the backend after SSHing to the host:

```bash
cd /home/ec2-user/agr_pdf_extraction_service
git fetch --all --prune
git checkout <branch-or-sha>
git pull --ff-only || true
cd deploy
PDFX_DEPLOY_BUILD_MODE=rebuild GPU_MODE=on ./deploy.sh
```

Use `PDFX_DEPLOY_BUILD_MODE=rebuild` only when dependencies or the Dockerfile
changed. The default `auto` mode lets Docker Compose reuse the existing image
and build only if the image is missing.

If a backend startup fails, the proxy marks the ASG instance unhealthy with
`SetInstanceHealth`. Auto Scaling then terminates it and launches a replacement
from the launch template while respecting `BackendMaxSize`. The proxy keeps
queued replay waiting for `ASG_STARTUP_REPLACEMENT_ATTEMPTS` replacement
attempts before failing queued work.

Scale the backend back down when testing is complete:

```bash
aws autoscaling update-auto-scaling-group \
  --profile ctabone \
  --region us-east-1 \
  --auto-scaling-group-name "$TEST_BACKEND_ASG" \
  --desired-capacity 0
```

## Promote The Pattern To Production

1. Deploy this stack to the test mirror first and validate `/api/v1/health`,
   `/api/v1/health/deep`, and a small extraction.
2. Create a production CloudFormation stack using the same launch template,
   ASG, SSM, IAM, alarm, and proxy resources, with production values:
   `EnvironmentName=prod`, `SsmParameterPath=pdfx`,
   `AuditBucketName=agr-pdf-extraction-benchmark`, and
   `DomainName=pdfx.alliancegenome.org`.
3. Keep production `BackendMinSize=0`, `BackendDesiredCapacity=0`,
   `BackendMaxSize=1` unless a planned maintenance window explicitly tests
   `BackendMaxSize=2`.
4. Roll the production proxy image after `/pdfx/backend-asg-name` exists.
   The legacy `/pdfx/ec2-instance-id` parameter can stay as a blank
   placeholder for rollback compatibility.
5. Validate from AI Curation using its configured Cognito auth path, then run
   a small PDF extraction smoke.

## Image Artifact Retention Notes

The test IAM role includes `s3:PutObjectTagging`, and the test bucket expires
objects tagged:

- `pdfx-artifact-type=extracted-image`
- `pdfx-retention=temporary`

Production currently has no bucket lifecycle rule. Before enabling the image
manifest feature in production, apply an equivalent lifecycle rule to the prod
bucket and make sure the prod EC2 role can write object tags.
