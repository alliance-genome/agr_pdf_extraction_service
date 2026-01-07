# AWS Access Guide for PDF Extraction Benchmarking

**For:** Pedro Assis (pedroh@stanford.edu)
**Project:** KANBAN-874 - PDF Extraction Tool Benchmarking

---

## S3 Bucket Overview

All test PDFs are stored in a private S3 bucket:

| Setting | Value |
|---------|-------|
| **Bucket Name** | `agr-pdf-extraction-benchmark` |
| **Region** | `us-east-1` |
| **Public Access** | Blocked (private) |
| **Total Files** | 56 PDFs |
| **Total Size** | ~274 MB |

### Bucket Structure

```
s3://agr-pdf-extraction-benchmark/
├── test_pdfs/
│   ├── core/                    # 12 curated PDFs (~61 MB)
│   │   ├── AGRKB_101000000645569_mbc-31-1411.pdf
│   │   ├── AGRKB_101000000662381_13059_2020_Article_1948.pdf
│   │   └── ... (10 more files)
│   └── extended/                # 42 GROBID failure PDFs (~213 MB)
│       ├── AGRKB_101000000990125_main.pdf
│       ├── AGRKB_101000000990125_supplement_mmc1.pdf
│       └── ... (40 more files)
```

---

## IAM Policy Required

To access this bucket, you need an IAM policy attached to your AWS user/role:

### Option 1: Minimal Read-Only Access (Recommended for Benchmarking)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PDFBenchmarkReadAccess",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::agr-pdf-extraction-benchmark",
                "arn:aws:s3:::agr-pdf-extraction-benchmark/*"
            ]
        }
    ]
}
```

### Option 2: Full Access (If You Need to Upload Results)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PDFBenchmarkFullAccess",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::agr-pdf-extraction-benchmark",
                "arn:aws:s3:::agr-pdf-extraction-benchmark/*"
            ]
        }
    ]
}
```

---

## AWS CLI Setup

### 1. Install AWS CLI

```bash
# macOS
brew install awscli

# Ubuntu/Debian
sudo apt install awscli

# Or via pip
pip install awscli
```

### 2. Configure Credentials

```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Default region: us-east-1
# Default output format: json
```

Or create `~/.aws/credentials` manually:

```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

---

## Accessing Test PDFs

### List All Files

```bash
# List everything in the bucket
aws s3 ls s3://agr-pdf-extraction-benchmark/ --recursive

# List just core test set
aws s3 ls s3://agr-pdf-extraction-benchmark/test_pdfs/core/

# List extended set (GROBID failures)
aws s3 ls s3://agr-pdf-extraction-benchmark/test_pdfs/extended/
```

### Download Files

```bash
# Download entire bucket
aws s3 sync s3://agr-pdf-extraction-benchmark/ ./benchmark_data/

# Download just core PDFs
aws s3 sync s3://agr-pdf-extraction-benchmark/test_pdfs/core/ ./core_pdfs/

# Download just extended PDFs
aws s3 sync s3://agr-pdf-extraction-benchmark/test_pdfs/extended/ ./extended_pdfs/

# Download a single file
aws s3 cp s3://agr-pdf-extraction-benchmark/test_pdfs/core/AGRKB_101000000645569_mbc-31-1411.pdf ./
```

### Python Access (boto3)

```python
import boto3
from pathlib import Path

s3 = boto3.client('s3')
bucket = 'agr-pdf-extraction-benchmark'

# List all PDFs
response = s3.list_objects_v2(Bucket=bucket, Prefix='test_pdfs/')
for obj in response.get('Contents', []):
    print(obj['Key'])

# Download a PDF
def download_pdf(key, local_path):
    s3.download_file(bucket, key, local_path)

# Download all core PDFs
def download_all_core(output_dir='./core_pdfs'):
    Path(output_dir).mkdir(exist_ok=True)
    response = s3.list_objects_v2(Bucket=bucket, Prefix='test_pdfs/core/')
    for obj in response.get('Contents', []):
        filename = obj['Key'].split('/')[-1]
        if filename.endswith('.pdf'):
            local_path = f"{output_dir}/{filename}"
            print(f"Downloading {filename}...")
            s3.download_file(bucket, obj['Key'], local_path)
```

---

## Test PDF Sets

### Core Set (12 PDFs, ~61 MB)

Carefully selected PDFs covering all 7 MODs with diverse content types:
- All 7 Model Organism Databases represented
- 10 unique journals
- Mix of tables, figures, reviews, research articles
- 2 known GROBID failure cases included
- Size range: 0.92 - 10.23 MB

### Extended Set (42 PDFs, ~213 MB)

GROBID failure cases from SCRUM-5561:
- 13 main papers that failed GROBID extraction
- 29 supplementary materials
- All appear to be valid PDFs
- Real-world edge cases for robustness testing

**File Naming Convention:**
- Main papers: `AGRKB_<ID>_<filename>.pdf`
- Supplements: `AGRKB_<ID>_supplement_<filename>.pdf`

---

## Troubleshooting

### "Access Denied" Error

1. Verify your AWS credentials are configured:
   ```bash
   aws sts get-caller-identity
   ```

2. Check if the IAM policy is attached to your user/role

3. Ensure you're using the correct AWS region:
   ```bash
   aws s3 ls s3://agr-pdf-extraction-benchmark/ --region us-east-1
   ```

### "NoSuchBucket" Error

The bucket name is exactly `agr-pdf-extraction-benchmark` (no typos, all lowercase).

### Slow Downloads

For faster bulk downloads, use `aws s3 sync` which parallelizes transfers:
```bash
aws s3 sync s3://agr-pdf-extraction-benchmark/test_pdfs/ ./test_pdfs/ --only-show-errors
```

---

## Contact

- **AWS Access Issues:** Chris Tabone (Alliance)
- **Project Questions:** Chris Tabone (Alliance)
- **Benchmarking Questions:** Pedro Assis (SGD)

---

## Quick Reference

```bash
# Verify access
aws s3 ls s3://agr-pdf-extraction-benchmark/

# Download everything
aws s3 sync s3://agr-pdf-extraction-benchmark/ ./pdf_benchmark/

# Count files
aws s3 ls s3://agr-pdf-extraction-benchmark/ --recursive | wc -l

# Get total size
aws s3 ls s3://agr-pdf-extraction-benchmark/ --recursive --summarize | tail -2
```
