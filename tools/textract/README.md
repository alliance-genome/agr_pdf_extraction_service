# AWS Textract - Cloud-Based Document Intelligence

## Overview

AWS Textract is a fully managed machine learning service by Amazon Web Services that automatically extracts text, handwriting, tables, forms, and other structured data from scanned documents. It goes beyond simple optical character recognition (OCR) to identify, understand, and extract data from forms and tables without manual intervention or custom code.

### What AWS Textract Does

- **Text Extraction**: Detects and extracts printed text and handwriting from documents
- **Form Processing**: Identifies key-value pairs in forms (e.g., "Name: John Doe")
- **Table Extraction**: Detects tables and preserves their structure, including rows, columns, and cells
- **Layout Analysis**: Understands document layout and reading order for text linearization
- **Document Analysis**: Processes complex documents including PDFs, images (PNG, JPEG, TIFF)
- **Query-Based Extraction**: Extract specific information using natural language queries
- **Expense Analysis**: Extract data from invoices and receipts

## Key Features

- **High Accuracy**: Industry-leading accuracy for text and data extraction
- **No Training Required**: Pre-trained models ready to use immediately
- **Scalable**: Handles single documents or batch processing at scale
- **Multi-Format Support**: Works with PDF, PNG, JPEG, TIFF files
- **Cloud-Based**: No infrastructure to manage, fully serverless
- **Integration**: Easy integration with other AWS services (S3, Lambda, etc.)

## AWS Permissions & IAM Requirements

### Required IAM Permissions

To use AWS Textract, your IAM user or role needs the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "textract:DetectDocumentText",
        "textract:AnalyzeDocument",
        "textract:AnalyzeExpense",
        "textract:StartDocumentAnalysis",
        "textract:GetDocumentAnalysis"
      ],
      "Resource": "*"
    }
  ]
}
```

### S3 Access (Required for S3-based documents)

If analyzing documents from S3, you also need S3 permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name/*",
        "arn:aws:s3:::your-bucket-name"
      ]
    }
  ]
}
```

### Managed Policy

Alternatively, use the AWS managed policy for full Textract access:
- `arn:aws:iam::aws:policy/AmazonTextractFullAccess`

## Installation

### Install boto3 (AWS SDK for Python)

```bash
pip install boto3
```

### Install amazon-textract-textractor (Recommended)

The Textractor library provides a higher-level, more convenient interface:

```bash
pip install amazon-textract-textractor
```

### Additional Dependencies

For advanced features like pandas DataFrame export:

```bash
pip install amazon-textract-textractor[pandas]
```

## AWS Credentials Configuration

### Option 1: Environment Variables

```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

### Option 2: AWS Credentials File

Create `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

Create `~/.aws/config`:

```ini
[default]
region = us-east-1
```

### Option 3: IAM Roles (Recommended for EC2/Lambda)

When running on AWS infrastructure, use IAM roles instead of credentials.

## Python Usage Examples

### 1. Using Boto3 (Low-Level API)

#### Analyze Document from S3

```python
import boto3

textract = boto3.client('textract', region_name='us-east-1')

response = textract.analyze_document(
    Document={
        'S3Object': {
            'Bucket': 'your-bucket-name',
            'Name': 'path/to/document.pdf'
        }
    },
    FeatureTypes=['FORMS', 'TABLES']
)

# Process blocks
for block in response['Blocks']:
    if block['BlockType'] == 'LINE':
        print(f"Text: {block['Text']}")
    elif block['BlockType'] == 'TABLE':
        print("Found a table")
    elif block['BlockType'] == 'KEY_VALUE_SET':
        print(f"Form field: {block.get('Text', 'N/A')}")
```

#### Analyze Local Document (Bytes)

```python
import boto3

textract = boto3.client('textract', region_name='us-east-1')

with open('document.pdf', 'rb') as document:
    image_bytes = document.read()

response = textract.analyze_document(
    Document={'Bytes': image_bytes},
    FeatureTypes=['FORMS', 'TABLES']
)

for block in response['Blocks']:
    if block['BlockType'] == 'LINE':
        print(f"Line: {block['Text']}")
```

### 2. Using Textractor (High-Level API - Recommended)

#### Basic Document Analysis

```python
from textractor import Textractor

# Initialize Textractor
extractor = Textractor(region_name="us-east-1")

# Analyze document from S3
document = extractor.analyze_document(
    file_source="s3://your-bucket-name/path/to/document.pdf",
    features=["FORMS", "TABLES"]
)

# Extract text
print(document.text)

# Access forms
for page in document.pages:
    for field in page.form.fields:
        print(f"{field.key}: {field.value}")

# Access tables
for page in document.pages:
    for table in page.tables:
        # Convert to pandas DataFrame
        df = table.to_pandas()
        print(df)
```

#### Analyze Local File

```python
from textractor import Textractor

extractor = Textractor(region_name="us-east-1")

# Analyze local file
document = extractor.analyze_document(
    file_source="path/to/local/document.pdf",
    features=["FORMS", "TABLES", "LAYOUT"]
)

# Get all text
full_text = document.text

# Get tables as list of DataFrames
tables = [table.to_pandas() for table in document.tables]
```

#### Using Queries for Targeted Extraction

```python
from textractor import Textractor
from textractor.data.constants import TextractFeatures

extractor = Textractor(region_name="us-east-1")

document = extractor.analyze_document(
    file_source="s3://bucket/invoice.pdf",
    features=[TextractFeatures.QUERIES],
    queries=["What is the invoice number?", "What is the total amount?"]
)

# Access query results
for page in document.pages:
    for query_result in page.queries:
        print(f"Question: {query_result.query}")
        print(f"Answer: {query_result.answer}")
```

## API Comparison

### DetectDocumentText
- **Purpose**: Simple text extraction only
- **Features**: Detects lines and words of text
- **Use Case**: Basic OCR, no tables or forms needed
- **Cost**: Lowest

### AnalyzeDocument
- **Purpose**: Advanced analysis with structure
- **Features**: Text + Tables + Forms + Layout
- **Use Case**: Complex documents with structured data
- **Cost**: Higher (based on features used)

### AnalyzeExpense
- **Purpose**: Specialized for invoices/receipts
- **Features**: Extracts line items, vendor info, amounts
- **Use Case**: Financial document processing
- **Cost**: Specialized pricing

## Cost Considerations

### Pricing Model (as of 2025)

AWS Textract pricing is based on the number of pages processed and features used:

- **DetectDocumentText**: ~$1.50 per 1,000 pages
- **AnalyzeDocument (Tables/Forms)**: ~$15-$50 per 1,000 pages depending on features
- **AnalyzeExpense**: ~$50 per 1,000 pages
- **Queries**: Additional cost per query per page

### Cost Optimization Tips

1. **Use DetectDocumentText** when you only need plain text extraction
2. **Request only needed features** (don't request TABLES if not needed)
3. **Batch processing** for large volumes (asynchronous API)
4. **Use S3 for large files** instead of bytes (more efficient)
5. **Consider AWS Free Tier**: 1,000 pages/month free for DetectDocumentText (first 3 months)

### Estimated Costs for Benchmarking

For 100 PDF documents:
- **Basic text extraction**: $0.15
- **Full analysis (tables + forms)**: $1.50 - $5.00
- **With queries**: $5.00 - $10.00

**Important**: Always check current AWS pricing at https://aws.amazon.com/textract/pricing/

## Best Practices

1. **Use Textractor library** for production code (better error handling, cleaner API)
2. **Handle exceptions** - network issues, throttling, invalid documents
3. **Use asynchronous APIs** for large documents (>5 pages or >10 MB)
4. **Store results** - Textract doesn't retain analysis results
5. **Validate document format** before processing (supported formats only)
6. **Monitor costs** using AWS Cost Explorer and set billing alerts
7. **Use S3 for large-scale processing** instead of direct bytes upload

## Limitations

- **File size**: Max 5 MB for synchronous APIs, 500 MB for async
- **Page limit**: 3,000 pages max per document
- **Supported formats**: PDF, PNG, JPEG, TIFF only
- **Processing time**: Synchronous APIs timeout after ~60 seconds
- **Rate limits**: Default limits apply (can request increases)

## Error Handling

```python
from textractor import Textractor
from textractor.exceptions import TextractorError
from botocore.exceptions import ClientError

try:
    extractor = Textractor(region_name="us-east-1")
    document = extractor.analyze_document(
        file_source="path/to/document.pdf",
        features=["TABLES", "FORMS"]
    )
    print(document.text)

except TextractorError as e:
    print(f"Textractor error: {e}")
except ClientError as e:
    print(f"AWS error: {e.response['Error']['Message']}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Additional Resources

- **AWS Textract Documentation**: https://docs.aws.amazon.com/textract/
- **Textractor GitHub**: https://github.com/aws-samples/amazon-textract-textractor
- **Boto3 Textract Reference**: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/textract.html
- **AWS Textract Pricing**: https://aws.amazon.com/textract/pricing/
- **Textractor Documentation**: https://aws-samples.github.io/amazon-textract-textractor/

## Support

For issues with:
- **AWS Textract service**: Open AWS Support ticket
- **Textractor library**: https://github.com/aws-samples/amazon-textract-textractor/issues
- **Boto3**: https://github.com/boto/boto3/issues
