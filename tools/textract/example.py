#!/usr/bin/env python3
"""
AWS Textract Example Script

Demonstrates basic PDF text extraction using AWS Textract with both
boto3 (low-level) and textractor (high-level) approaches.

Requirements:
    pip install boto3 amazon-textract-textractor

AWS Credentials:
    Ensure AWS credentials are configured via:
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - ~/.aws/credentials file
    - IAM role (if running on EC2/Lambda)
"""

import sys
import os
from pathlib import Path


def example_boto3_local_file(file_path):
    """
    Example 1: Extract text from a local PDF using boto3 (low-level API)

    Note: File must be < 5MB for synchronous processing
    """
    print("\n" + "="*80)
    print("Example 1: Boto3 - Local File (Basic Text Extraction)")
    print("="*80)

    try:
        import boto3
        from botocore.exceptions import ClientError

        # Initialize Textract client
        textract = boto3.client('textract', region_name='us-east-1')

        # Read the document
        with open(file_path, 'rb') as document:
            image_bytes = document.read()

        # Call DetectDocumentText (simple text extraction, cheapest option)
        print(f"Processing: {file_path}")
        response = textract.detect_document_text(
            Document={'Bytes': image_bytes}
        )

        # Extract all text lines
        print("\nExtracted Text:")
        print("-" * 80)
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                print(block['Text'])

        print(f"\nTotal blocks detected: {len(response['Blocks'])}")

    except ClientError as e:
        print(f"AWS Error: {e.response['Error']['Message']}")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")


def example_boto3_with_tables_forms(file_path):
    """
    Example 2: Extract text, tables, and forms using boto3

    This uses AnalyzeDocument which is more expensive but extracts structured data
    """
    print("\n" + "="*80)
    print("Example 2: Boto3 - Advanced Analysis (Text + Tables + Forms)")
    print("="*80)

    try:
        import boto3
        from botocore.exceptions import ClientError

        textract = boto3.client('textract', region_name='us-east-1')

        with open(file_path, 'rb') as document:
            image_bytes = document.read()

        print(f"Processing: {file_path}")
        response = textract.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['TABLES', 'FORMS']
        )

        # Count different block types
        line_count = 0
        table_count = 0
        key_value_count = 0

        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                line_count += 1
            elif block['BlockType'] == 'TABLE':
                table_count += 1
            elif block['BlockType'] == 'KEY_VALUE_SET':
                key_value_count += 1

        print(f"\nResults:")
        print(f"  Lines of text: {line_count}")
        print(f"  Tables found: {table_count}")
        print(f"  Form fields: {key_value_count}")

        # Display first few lines
        print("\nFirst 5 lines of text:")
        print("-" * 80)
        count = 0
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                print(block['Text'])
                count += 1
                if count >= 5:
                    break

    except ClientError as e:
        print(f"AWS Error: {e.response['Error']['Message']}")
    except Exception as e:
        print(f"Error: {e}")


def example_textractor_local_file(file_path):
    """
    Example 3: Extract text using Textractor (high-level API, recommended)

    Textractor provides a cleaner interface and better error handling
    """
    print("\n" + "="*80)
    print("Example 3: Textractor - Local File (Recommended Approach)")
    print("="*80)

    try:
        from textractor import Textractor
        from textractor.exceptions import TextractorError

        # Initialize Textractor
        extractor = Textractor(region_name="us-east-1")

        print(f"Processing: {file_path}")

        # Analyze document with tables and forms
        document = extractor.analyze_document(
            file_source=file_path,
            features=["TABLES", "FORMS"]
        )

        # Get all text
        print("\nFull Document Text:")
        print("-" * 80)
        print(document.text[:1000])  # First 1000 chars
        if len(document.text) > 1000:
            print(f"\n... (showing first 1000 of {len(document.text)} characters)")

        # Process forms
        print("\nForm Fields Found:")
        print("-" * 80)
        for page in document.pages:
            for field in page.form.fields[:5]:  # Show first 5 fields
                key = field.key.text if field.key else "N/A"
                value = field.value.text if field.value else "N/A"
                print(f"  {key}: {value}")

        # Process tables
        print("\nTables Found:")
        print("-" * 80)
        for i, table in enumerate(document.tables):
            print(f"\nTable {i+1}: {table.n_rows} rows x {table.n_cols} columns")
            # Print first 3 rows
            for row_idx, row in enumerate(table.rows[:3]):
                cells = [cell.text for cell in row.cells]
                print(f"  Row {row_idx+1}: {cells}")
            if table.n_rows > 3:
                print(f"  ... ({table.n_rows - 3} more rows)")

    except TextractorError as e:
        print(f"Textractor Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


def example_textractor_s3(s3_uri):
    """
    Example 4: Extract text from a PDF stored in S3

    Args:
        s3_uri: S3 URI in format 's3://bucket-name/path/to/file.pdf'
    """
    print("\n" + "="*80)
    print("Example 4: Textractor - S3 File")
    print("="*80)

    try:
        from textractor import Textractor
        from textractor.exceptions import TextractorError

        extractor = Textractor(region_name="us-east-1")

        print(f"Processing S3 file: {s3_uri}")

        document = extractor.analyze_document(
            file_source=s3_uri,
            features=["TABLES", "FORMS"]
        )

        print(f"\nDocument processed successfully!")
        print(f"  Total pages: {len(document.pages)}")
        print(f"  Total text length: {len(document.text)} characters")
        print(f"  Tables found: {len(document.tables)}")

        # Show first 500 characters
        print("\nDocument preview:")
        print("-" * 80)
        print(document.text[:500])

    except TextractorError as e:
        print(f"Textractor Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function to run examples"""

    print("\n" + "="*80)
    print("AWS Textract PDF Extraction Examples")
    print("="*80)

    # Check for command-line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use a sample file path (update this to your test file)
        file_path = "sample_document.pdf"
        print(f"\nUsage: python {sys.argv[0]} <path_to_pdf>")
        print(f"No file specified, using default: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"\nError: File not found: {file_path}")
        print("\nPlease provide a valid PDF file path:")
        print(f"  python {sys.argv[0]} /path/to/your/document.pdf")
        return 1

    # Check file size (warn if > 5MB)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 5:
        print(f"\nWarning: File size is {file_size_mb:.2f}MB")
        print("Files > 5MB require asynchronous processing (not shown in these examples)")

    # Run examples
    try:
        # Example 1: Basic text extraction with boto3
        example_boto3_local_file(file_path)

        # Example 2: Advanced analysis with boto3
        example_boto3_with_tables_forms(file_path)

        # Example 3: Textractor (recommended)
        example_textractor_local_file(file_path)

        # Example 4: S3 (uncomment and provide S3 URI to test)
        # example_textractor_s3("s3://your-bucket/your-document.pdf")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Try with your own PDF files")
    print("  2. Experiment with different features (TABLES, FORMS, LAYOUT)")
    print("  3. Export tables to pandas DataFrames: table.to_pandas()")
    print("  4. Check AWS costs in your billing dashboard")
    print("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
