# Marker PDF to Markdown Converter

## What is Marker?

Marker is a high-performance PDF to Markdown converter that uses advanced machine learning models to accurately extract text, tables, images, and complex layouts from PDF documents. It provides superior accuracy compared to traditional OCR tools, especially for complex documents with tables, forms, mathematical equations, and mixed content.

Marker can convert PDFs to:
- **Markdown** - Clean, readable markdown format
- **JSON** - Structured data with bounding boxes and layout information
- **HTML** - For tables and structured content

## Key Features

- **High Accuracy OCR** - Advanced text recognition models
- **Table Extraction** - Accurately extracts and formats tables, including cross-page tables
- **Layout Detection** - Preserves document structure and formatting
- **Math Equation Support** - Converts inline math to proper format
- **Form Value Extraction** - Extracts data from form fields
- **Handwriting Recognition** - Can recognize handwritten text
- **LLM Enhancement** - Optional LLM integration for improved accuracy on complex documents
- **Multi-GPU Support** - Leverage multiple GPUs for batch processing
- **REST API** - Run as a FastAPI server for web-based access
- **Batch Processing** - Process multiple PDFs in parallel

## Installation

### Basic Installation

```bash
pip install marker-pdf
```

### Server Dependencies (for REST API)

```bash
pip install -U uvicorn fastapi python-multipart
```

## GPU Requirements

Marker can run on both CPU and GPU, but GPU is recommended for better performance:

- **GPU (CUDA)**:
  - ~3.5GB VRAM average per worker
  - ~5GB VRAM peak per worker
  - Supports NVIDIA GPUs with CUDA
  - Can use multiple GPUs for batch processing
  - Supports Flash Attention 2 for performance optimization

- **CPU**:
  - Works but slower
  - Uses float32 instead of float16

### Device Configuration

Marker automatically detects available GPUs and falls back to CPU if no GPU is available. You can also manually configure device settings (see Python API examples).

## Python Usage

### Basic PDF to Markdown Conversion

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Initialize converter with model artifacts
converter = PdfConverter(
    artifact_dict=create_model_dict(),
)

# Convert the PDF
rendered = converter("/path/to/document.pdf")

# Extract text and images
text, file_ext, images = text_from_rendered(rendered)

# Access metadata
metadata = rendered.metadata
print(f"Converted {len(metadata['page_stats'])} pages")

# Save output
with open("output.md", "w") as f:
    f.write(text)

# Save images
import os
for img_name, img in images.items():
    img.save(os.path.join("output", img_name), "PNG")
```

### LLM-Enhanced Conversion (Higher Accuracy)

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

# Configure with LLM support for better accuracy
config = {
    "use_llm": True,
    "gemini_api_key": "your-api-key-here",
    "output_format": "markdown",
}

config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)

rendered = converter("/path/to/complex_document.pdf")
text, _, images = text_from_rendered(rendered)

# LLM mode improves:
# - Table formatting and cross-page merging
# - Inline math conversion
# - Form value extraction
# - Handwriting recognition
```

### GPU/CPU Configuration

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
import torch

# Custom device and dtype configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

artifact_dict = create_model_dict(
    device=device,
    dtype=dtype,
    attention_implementation="flash_attention_2"  # Optional: Use Flash Attention
)

converter = PdfConverter(
    artifact_dict=artifact_dict,
)

rendered = converter("/path/to/document.pdf")
```

### Table Extraction

```python
from marker.converters.table import TableConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
import json

# Configure for table extraction
config = {
    "use_llm": True,
    "gemini_api_key": "your-api-key",
    "output_format": "json",  # Get cell bounding boxes
    "force_layout_block": "Table",  # Assume every page is a table
}

config_parser = ConfigParser(config)

converter = TableConverter(
    artifact_dict=create_model_dict(),
    config=config_parser.generate_config_dict(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)

rendered = converter("/path/to/document_with_tables.pdf")
text, _, images = text_from_rendered(rendered)

# JSON output includes table structure and cell positions
tables = json.loads(text)
for page in tables["pages"]:
    for child in page["children"]:
        if child["block_type"] == "Table":
            print(f"Table found: {child['id']}")
            print(f"HTML: {child['html']}")
```

## CLI Usage

### Single File Conversion

```bash
# Basic conversion
marker_single /path/to/document.pdf

# Convert specific page ranges
marker_single document.pdf --page_range "0,5-10,20"

# Force OCR on all pages
marker_single document.pdf --force_ocr

# Specify output format and directory
marker_single document.pdf --output_format json --output_dir ./output

# Use LLM for higher accuracy (requires API key)
export GOOGLE_API_KEY="your-api-key"
marker_single document.pdf --use_llm

# Custom block correction with LLM
marker_single document.pdf --use_llm --block_correction_prompt "Format all dates as YYYY-MM-DD"

# Paginated output (separate pages)
marker_single document.pdf --paginate_output

# Debug mode with detailed output
marker_single document.pdf --debug --output_dir ./debug_output
```

### Batch Conversion

```bash
# Convert all PDFs in a folder
marker /path/to/pdf_folder --output_dir /path/to/output

# Control parallelism (default: auto-detected)
marker /path/to/pdf_folder --workers 4

# Batch convert with LLM mode
marker /path/to/pdf_folder --use_llm --output_format json

# Convert with page range applied to all files
marker /path/to/pdf_folder --page_range "0-5"

# Extract tables only from multiple files
marker /path/to/pdf_folder \
  --converter_cls marker.converters.table.TableConverter \
  --use_llm \
  --output_format json

# Disable multiprocessing for debugging
marker /path/to/pdf_folder --disable_multiprocessing
```

### Multi-GPU Processing

```bash
# Use 4 GPUs with 15 workers total
NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert /input /output

# Each GPU will run approximately 15/4 = 3-4 workers
# Peak VRAM usage: ~5GB per worker
# Average VRAM usage: ~3.5GB per worker

# Single GPU with multiple workers
NUM_DEVICES=1 NUM_WORKERS=4 marker_chunk_convert /input /output
```

### Table Extraction CLI

```bash
# Extract tables with LLM enhancement
marker_single invoice.pdf \
  --use_llm \
  --force_layout_block Table \
  --converter_cls marker.converters.table.TableConverter \
  --output_format json

# Table-only conversion without layout detection
marker_single data.pdf \
  --converter_cls marker.converters.table.TableConverter \
  --output_format json \
  --output_dir ./tables
```

## REST API Server

### Starting the Server

```bash
# Install server dependencies
pip install -U uvicorn fastapi python-multipart

# Start server on default port 8000
marker_server

# Custom host and port
marker_server --port 8001 --host 0.0.0.0

# API documentation available at: http://localhost:8001/docs
```

### API Usage with cURL

#### Convert from File Path

```bash
curl -X POST http://localhost:8001/marker \
  -H "Content-Type: application/json" \
  -d '{
    "filepath": "/path/to/document.pdf",
    "output_format": "markdown",
    "force_ocr": false,
    "paginate_output": true
  }' | jq '.'
```

#### Upload and Convert File

```bash
# Upload and convert
curl -X POST http://localhost:8001/marker/upload \
  -F "file=@document.pdf" \
  -F "output_format=json" \
  -F "force_ocr=true" \
  -F "page_range=0-10" | jq '.success'

# Save output to file
curl -X POST http://localhost:8001/marker/upload \
  -F "file=@document.pdf" \
  -F "output_format=markdown" \
  --output response.json

# Extract only the markdown content
curl -X POST http://localhost:8001/marker/upload \
  -F "file=@document.pdf" \
  -F "output_format=markdown" | jq -r '.output' > output.md
```

## Models Included

Marker uses several specialized models in the artifact dictionary:

- **layout_model**: Page layout detection
- **recognition_model**: Text recognition (OCR)
- **table_rec_model**: Table structure recognition
- **detection_model**: Text line detection
- **ocr_error_model**: OCR quality assessment

## Performance Considerations

- **GPU recommended** for production use
- **Flash Attention 2** can improve performance on compatible GPUs
- **Multi-GPU support** for high-throughput batch processing
- **Worker parallelism** can be tuned based on available resources
- **LLM mode** increases accuracy but adds processing time and cost

## Output Formats

- **markdown**: Clean markdown text (default)
- **json**: Structured data with bounding boxes and layout information
- **HTML**: For tables and structured content (via table converter)

## Use Cases

- Document digitization and archival
- Academic paper processing
- Form data extraction
- Table extraction from reports
- Invoice and receipt processing
- Technical documentation conversion
- Legal document processing

## Resources

- GitHub Repository: https://github.com/datalab-to/marker
- Documentation: https://context7.com/datalab-to/marker
