# PDFX AWS Stack

This runbook manages the canonical PDFX AWS service:

- GPU backend managed by an EC2 Auto Scaling Group (`g5.4xlarge`, same Deep Learning GPU AMI family)
- Docker Compose backend stack on the EC2 host, using a prebuilt GPU image from ECR
- Fargate proxy in front of the backend
- S3-backed durable proxy queue
- S3 audit/artifact bucket with tagged temporary image lifecycle
- Cognito-protected public hostname through the shared `agr-services` ALB
- SSM parameters under `/pdfx/*`
- CloudWatch alarms for backend startup timeouts and ASG replacement requests
- optional GPU idle guard alerts from `deploy/aws/pdfx-idle-guard-stack.yaml`

The template is `deploy/aws/pdfx-stack.yaml`.

## AWS Inventory Used

These values were discovered from the current AWS account and are set as
template defaults:

| Area | Default value |
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
| Proxy image repo | `agr_pdfx_proxy` |
| Backend image repo | `agr_pdfx_backend` |

The backend AMI provides the host NVIDIA/Docker baseline. Python, PyTorch, CUDA
runtime wheels, Docling, Marker, and application dependencies belong in the
prebuilt `agr_pdfx_backend` ECR image. Backend user data should pull that image
and run with `PDFX_DEPLOY_BUILD_MODE=never`; it must not pip-install PyTorch or
CUDA dependencies during normal production boot. Warm-pool reuse preserves
Docker images and model caches across idle shutdowns.

When `PDFX_GPU_IMAGE` and `PDFX_DEPLOY_BUILD_MODE=never` are set, `deploy.sh`
also applies `docker-compose.gpu.prebuilt.yml`. That override keeps persistent
data/model/log mounts but removes source-code bind mounts, so production runs
the app code baked into the ECR image instead of overlaying a potentially
different checkout.

GPU deploys are intentionally gated on CUDA being usable inside containers.
`deploy.sh` probes the configured GPU image with `docker run --gpus all` before
starting the Compose stack and then probes `torch.cuda` inside `pdfx-worker`
after startup. If the host can run `nvidia-smi` but the worker probe fails, the
backend is not ready; rerun the guarded deploy or restart app/worker after the
NVIDIA container runtime is ready.

Production-style prebuilt GPU deploys also prewarm Marker model artifacts into
the persistent host cache volumes by default (`PDFX_PREWARM_MODELS=auto` with
`PDFX_DEPLOY_BUILD_MODE=never`). This makes stopped warm-pool preparation pay
the model download/load cost before curator traffic reaches the worker. Set
`PDFX_PREWARM_MODELS=off` only for controlled debugging where first-job model
download latency is acceptable.

## Current Account Migration Note

In the current AWS account, several canonical `pdfx` resources already exist
outside this CloudFormation template, including the S3 audit bucket, proxy ECS
cluster/service, proxy target group, proxy security group, and canonical IAM
roles/profile. A first deploy of this template as a new `pdfx` stack would
collide with those existing names unless the resources are imported into the
stack or the template is adapted to reference them as existing inputs.

Use this template as the canonical shape for new environments and for planned
CloudFormation adoption. For the current account migration, create or import the
backend resources carefully, update `/pdfx/backend-asg-name` and the proxy task
role policy together, then retire the old suffixed resources after
`https://pdfx.alliancegenome.org/api/v1/health/deep` is healthy.

Production `pdfx.alliancegenome.org` must not point at resources whose names,
tags, stack names, SSM paths, queues, or buckets contain a non-production suffix
such as `-test`. If `/pdfx/backend-asg-name` resolves to anything other than a
production backend such as `pdfx-backend`, treat that as a migration bug.

## Before Deploying

Create the backend environment file as a SecureString. Start from the current
backend `.env`, but remove legacy toggles such as `MARKER_EXTRACT_IMAGES`; image
extraction is now request-scoped through `extract_images`. Also remove any
storage overrides such as `AUDIT_S3_BUCKET`, `AUDIT_S3_PREFIX`,
`AUDIT_S3_BUCKET_SSM_PARAM`, `IMAGE_URL_TTL_SECONDS`,
`IMAGE_RETENTION_TTL_SECONDS`, and `AWS_DEFAULT_REGION`. The bootstrap scrubs
and replaces those values as a second guardrail.

The bootstrap appends the PDFX audit values automatically, so they do not need
to be present in the SecureString:

```bash
aws ssm put-parameter \
  --profile ctabone \
  --region us-east-1 \
  --name /pdfx/backend-env \
  --type SecureString \
  --overwrite \
  --value "$(cat backend.env)"
```

Fetch the Cognito pool ID without writing it into the repo:

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

## Deploy The PDFX Stack

This creates the backend launch template and Auto Scaling Group, proxy
service, S3 bucket, IAM roles, ALB rule, DNS record, CloudWatch alarms, and
`/pdfx/*` parameters.

Only run this as a create/update when those named resources are either absent,
already managed by the same stack, or part of a planned CloudFormation import.

```bash
aws cloudformation deploy \
  --profile ctabone \
  --region us-east-1 \
  --stack-name pdfx \
  --template-file deploy/aws/pdfx-stack.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    CognitoUserPoolId="$COGNITO_USER_POOL_ID" \
    DeployBackendOnBoot=true \
    BackendMaxSize=1 \
    BackendGitRef=main
```

For a feature branch, push the branch first and set
`BackendGitRef=<branch-or-sha>`.

The backend ASG defaults to `BackendMinSize=0`, `BackendDesiredCapacity=0`,
`BackendMaxSize=1`, and `BackendWarmPoolMinSize=1`. The proxy scales desired
capacity to `1` on wake and back to `0` after idle shutdown. The warm pool keeps
one stopped, already-bootstrapped backend instance so Docker images and ML model
caches survive idle shutdown without paying for a running GPU. Keep
in mind that the stopped warm-pool instance also retains its bootstrapped git
ref and Docker image; after dependency or Dockerfile changes, push a new
`agr_pdfx_backend` image and terminate the warm instance so a fresh one pulls
the new tag. Keep `BackendMaxSize=1` for strict cost control. Use
`BackendMaxSize=2` only for controlled validation of launch-before-terminate
behavior.

The launch template installs a resume-safe systemd bootstrap service
(`pdfx-backend-bootstrap.service`) before doing any long-running package,
checkout, image pull, or deploy work. This matters for stopped warm-pool
instances: AWS can stop a newly prepared instance while cloud-init is still
running. On the next real start, cloud-init may not rerun user data, but the
enabled systemd service starts again and completes the idempotent deploy.

Durable proxy queue metadata and PDF payload objects live under `QueuePrefix`
in the audit bucket and expire after `QueueRetentionDays` as a safety net.
Normal replay, cancellation, and failure paths should delete queue payloads
sooner; lifecycle cleanup is for abandoned clients or interrupted cleanup paths.
The proxy enforces `MAX_UPLOAD_BYTES=524288000` before backend wake/replay so
oversized requests do not fill S3 queue storage or start the GPU only to fail at
the backend upload cap. `MAX_MULTIPART_OVERHEAD_BYTES=10485760` gives the
early `Content-Length` guard room for multipart boundaries and small form
fields while still rejecting grossly oversized requests before body parsing.

The proxy's bounded replacement wait defaults to
`AsgStartupReplacementAttempts=1`, published to
`/<ssm-prefix>/asg-startup-replacement-attempts`.

## Deploy Idle Guard Alerts

Use `deploy/aws/pdfx-idle-guard-stack.yaml` when you want a separate
belt-and-suspenders alert if a backend ASG stays running after it should have
scaled down. The guard stores the continuous ASG runtime in SSM Parameter Store,
resets that runtime when the monitored ASG name changes, emits a
`GuardCheckSucceeded` heartbeat, and alarms if the guard itself stops checking.
For the PDFX stack:

```bash
deploy/aws/deploy_idle_guard.sh \
  --profile ctabone \
  --region us-east-1 \
  --project pdfx \
  --env prod \
  --backend-asg-name pdfx-backend \
  --proxy-metrics-url https://pdfx.alliancegenome.org/api/v1/metrics \
  --artifact-bucket agr-pdf-extraction-benchmark \
  --alarm-topic-arn <sns-topic-arn>
```

Use a confirmed SNS topic with an idle-guard/cost-guard name when alarm actions
should notify an operator.

Keep `BackendWarmPoolMinSize=1` when production traffic expects low wake
latency. That keeps one pre-initialized backend instance in the ASG warm pool in
`Stopped` state, so you pay for EBS while idle but not for running GPU compute.

## Deploy Proxy Changes

After building and pushing an image tag to ECR, roll the PDFX service:

```bash
cd proxy/deploy

./deploy.sh \
  --profile ctabone \
  --region us-east-1 \
  --cluster pdfx-proxy \
  --service pdfx-proxy \
  --ssm-prefix /pdfx \
  --image-tag <image-tag>
```

The deploy defaults target `pdfx-proxy` and `/pdfx`.

The CloudFormation stack uses `__PDFX_EMPTY__` as the disabled placeholder for
optional auth allow-lists because SSM String parameters cannot be empty; the
proxy ignores that exact placeholder.

## Backend ASG Operations

Get the backend ASG name and scale it up manually when needed:

```bash
PDFX_BACKEND_ASG="$(
  aws ssm get-parameter \
    --profile ctabone \
    --region us-east-1 \
    --name /pdfx/backend-asg-name \
    --query Parameter.Value \
    --output text
)"

aws autoscaling update-auto-scaling-group \
  --profile ctabone \
  --region us-east-1 \
  --auto-scaling-group-name "$PDFX_BACKEND_ASG" \
  --desired-capacity 1
```

Find the current instance and check the bootstrap log:

```bash
PDFX_INSTANCE_ID="$(
  aws autoscaling describe-auto-scaling-groups \
    --profile ctabone \
    --region us-east-1 \
    --auto-scaling-group-names "$PDFX_BACKEND_ASG" \
    --query 'AutoScalingGroups[0].Instances[?LifecycleState==`InService` || starts_with(LifecycleState, `Pending`)].InstanceId | [0]' \
    --output text
)"

aws ec2 describe-instances \
  --profile ctabone \
  --region us-east-1 \
  --instance-ids "$PDFX_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text

ssh ec2-user@<pdfx-public-ip> 'sudo tail -n 200 /var/log/pdfx-bootstrap.log'
```

Redeploy the backend after SSHing to the host:

```bash
cd /home/ec2-user/agr_pdf_extraction_service
git fetch --all --prune
git checkout <branch-or-sha>
git pull --ff-only || true
cd deploy
PDFX_GPU_IMAGE=100225593120.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_backend:<image-tag> \
  PDFX_DEPLOY_BUILD_MODE=never \
  PDFX_DEPLOY_PULL_IMAGES=auto \
  GPU_MODE=on \
  ./deploy.sh
```

Use `PDFX_DEPLOY_BUILD_MODE=rebuild` only for emergency local diagnosis. Normal
production starts must use the prebuilt ECR image, otherwise each replacement
can spend the startup window downloading and installing PyTorch/CUDA again.
In prebuilt mode, `deploy.sh` includes `docker-compose.gpu.prebuilt.yml` so the
running containers do not bind-mount checkout source over the image.

If a backend startup fails, the proxy marks the ASG instance unhealthy with
`SetInstanceHealth`. Auto Scaling then terminates it and launches a replacement
from the launch template while respecting `BackendMaxSize`. The proxy keeps
queued replay waiting for `ASG_STARTUP_REPLACEMENT_ATTEMPTS` replacement
attempts before failing queued work.

Scale the backend back down when validation is complete:

```bash
aws autoscaling update-auto-scaling-group \
  --profile ctabone \
  --region us-east-1 \
  --auto-scaling-group-name "$PDFX_BACKEND_ASG" \
  --desired-capacity 0
```

## Operational Notes

1. Keep `BackendMinSize=0`, `BackendDesiredCapacity=0`, and `BackendMaxSize=1`
   unless a planned maintenance window explicitly tests `BackendMaxSize=2`.
2. Roll the proxy image after `/pdfx/backend-asg-name` exists. The legacy
   `/pdfx/ec2-instance-id` parameter can stay as a blank placeholder for
   rollback compatibility.
3. Validate from AI Curation using its configured Cognito auth path, then run a
   small PDF extraction smoke.

## Image Artifact Retention Notes

The PDFX IAM role includes `s3:PutObjectTagging`, and the bucket expires
objects tagged:

- `pdfx-artifact-type=extracted-image`
- `pdfx-retention=temporary`

Before enabling the image manifest feature, make sure the active bucket has the
same lifecycle rule and that the EC2 role can write object tags.
