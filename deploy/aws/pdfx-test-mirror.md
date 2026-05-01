# PDFX Test Mirror

This runbook creates a test sibling for the production PDF extraction service
without reusing production mutable state. It mirrors the runtime shape:

- GPU backend on EC2 (`g5.2xlarge`, same Deep Learning GPU AMI family)
- Docker Compose backend stack on the EC2 host
- Fargate proxy in front of the backend
- S3-backed durable proxy queue
- S3 audit/artifact bucket with tagged temporary image lifecycle
- Cognito-protected public hostname through the shared `agr-services` ALB
- environment-scoped SSM parameters under `/pdfx-test/*`

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
| Backend instance type | `g5.2xlarge` |
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

This creates the test EC2 instance, proxy service, S3 bucket, IAM roles, ALB
rule, DNS record, and `/pdfx-test/*` parameters. It does not modify production
resources other than adding a new host rule to the shared ALB and a new DNS
record.

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
    BackendGitRef=main
```

For testing a feature branch, push the branch first and set
`BackendGitRef=<branch-or-sha>`. If you want to create the infrastructure
without immediately building the backend image, omit `DeployBackendOnBoot=true`;
the instance will provision host basics and then stop itself.

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

## Backend Host Operations

Get the test instance ID and start it when needed:

```bash
TEST_INSTANCE_ID="$(
  aws ssm get-parameter \
    --profile ctabone \
    --region us-east-1 \
    --name /pdfx-test/ec2-instance-id \
    --query Parameter.Value \
    --output text
)"

aws ec2 start-instances \
  --profile ctabone \
  --region us-east-1 \
  --instance-ids "$TEST_INSTANCE_ID"
```

Check the bootstrap log:

```bash
ssh ec2-user@<test-public-ip> 'sudo tail -n 200 /var/log/pdfx-bootstrap.log'
```

Redeploy the backend after SSHing to the host:

```bash
cd /home/ec2-user/agr_pdf_extraction_service
git fetch --all --prune
git checkout <branch-or-sha>
git pull --ff-only || true
cd deploy
GPU_MODE=on ./deploy.sh
```

## Image Artifact Retention Notes

The test IAM role includes `s3:PutObjectTagging`, and the test bucket expires
objects tagged:

- `pdfx-artifact-type=extracted-image`
- `pdfx-retention=temporary`

Production currently has no bucket lifecycle rule. Before enabling the image
manifest feature in production, apply an equivalent lifecycle rule to the prod
bucket and make sure the prod EC2 role can write object tags.
