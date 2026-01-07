# GROBID - PDF Extraction Tool

## What is GROBID?

GROBID (GeneRation Of BIbliographic Data) is a machine learning software for extracting information from scholarly documents. It specializes in parsing scientific PDFs and converting them into structured TEI XML format, extracting:

- Document metadata (title, authors, affiliations, abstract)
- Full text body with section structure
- Bibliographic references
- Figures and tables
- Formulas
- Citations and reference linking

GROBID is particularly effective for scientific and academic publications, using trained models to understand the structure of scholarly documents.

## Key Features

- **High Accuracy**: Machine learning models trained specifically on scientific literature
- **TEI XML Output**: Structured output in TEI (Text Encoding Initiative) format
- **Coordinate Extraction**: Can provide PDF coordinates for extracted elements
- **Batch Processing**: Process multiple PDFs efficiently
- **REST API**: Simple HTTP API for integration
- **Docker Support**: Easy deployment with Docker containers
- **GPU Support**: Optional GPU acceleration for improved performance

## Installation

### Docker Installation (Recommended)

The easiest way to run GROBID is using Docker:

#### 1. Pull the Docker image

```bash
docker pull grobid/grobid:0.8.2
```

#### 2. Run the GROBID service (CPU version)

```bash
docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2
```

#### 3. Run with GPU support (for better performance)

If you have a GPU available:

```bash
docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2
```

#### 4. Run with custom configuration

Mount a custom configuration file:

```bash
docker run --rm --init --ulimit core=0 -p 8070:8070 \
  -v /path/to/local/grobid.yaml:/opt/grobid/grobid-home/config/grobid.yaml:ro \
  grobid/grobid:0.8.2
```

The service will be available at `http://localhost:8070`

### Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3'
services:
  grobid:
    image: grobid/grobid:0.8.2
    ports:
      - "8070:8070"
    init: true
    ulimits:
      core: 0
```

Start with:
```bash
docker-compose up -d
```

## Python Client Usage

### Using Python Requests Library

Install the requests library:

```bash
pip install requests
```

### Basic PDF Extraction Example

```python
import requests

# URL of the GROBID service
grobid_url = "http://localhost:8070/api/processFulltextDocument"

# Path to your PDF file
pdf_path = "path/to/your/document.pdf"

# Send the PDF to GROBID
with open(pdf_path, 'rb') as pdf_file:
    files = {'input': pdf_file}
    response = requests.post(grobid_url, files=files)

# Get the TEI XML output
if response.status_code == 200:
    tei_xml = response.text
    print(tei_xml)
else:
    print(f"Error: {response.status_code}")
```

### Extract with TEI Coordinates

Get coordinates for specific elements (useful for locating text in the PDF):

```python
import requests

grobid_url = "http://localhost:8070/api/processFulltextDocument"
pdf_path = "path/to/your/document.pdf"

with open(pdf_path, 'rb') as pdf_file:
    files = {'input': pdf_file}
    # Request coordinates for figures, references, and formulas
    data = {
        'teiCoordinates': ['figure', 'biblStruct', 'formula', 'ref', 'persName']
    }
    response = requests.post(grobid_url, files=files, data=data)

if response.status_code == 200:
    tei_xml_with_coords = response.text
    # Save to file
    with open('output.tei.xml', 'w', encoding='utf-8') as f:
        f.write(tei_xml_with_coords)
```

### Extract with Raw Citations

Include the original citation strings:

```python
import requests

grobid_url = "http://localhost:8070/api/processFulltextDocument"
pdf_path = "path/to/your/document.pdf"

with open(pdf_path, 'rb') as pdf_file:
    files = {'input': pdf_file}
    data = {'includeRawCitations': '1'}
    response = requests.post(grobid_url, files=files, data=data)

if response.status_code == 200:
    tei_xml = response.text
    print(tei_xml)
```

## API Endpoints

GROBID provides several endpoints for different processing needs:

| Endpoint | Purpose |
|----------|---------|
| `/api/processFulltextDocument` | Extract full text including header, body, and citations |
| `/api/processHeaderDocument` | Extract only header metadata (title, authors, abstract) |
| `/api/processReferences` | Extract and parse bibliographic references only |
| `/api/processCitation` | Parse a single citation string |
| `/api/isalive` | Health check endpoint |

### Health Check

```bash
curl http://localhost:8070/api/isalive
```

## Command-Line Usage with cURL

### Extract Full Text

```bash
curl -v --form input=@./document.pdf http://localhost:8070/api/processFulltextDocument
```

### Extract with Coordinates

```bash
curl -v --form input=@./document.pdf \
  --form teiCoordinates=figure \
  --form teiCoordinates=biblStruct \
  --form teiCoordinates=formula \
  http://localhost:8070/api/processFulltextDocument
```

### Extract Only Header

```bash
curl -v --form input=@./document.pdf http://localhost:8070/api/processHeaderDocument
```

## Batch Processing

For processing multiple PDFs, you can use the GROBID batch mode (requires building from source):

```bash
java -Xmx4G -jar grobid-core-0.8.2-onejar.jar \
  -gH grobid-home \
  -dIn /path/to/input/directory \
  -dOut /path/to/output/directory \
  -exe processFullText
```

Or use a simple Python script to loop through files:

```python
import os
import requests
from pathlib import Path

grobid_url = "http://localhost:8070/api/processFulltextDocument"
input_dir = "input_pdfs"
output_dir = "output_tei"

Path(output_dir).mkdir(exist_ok=True)

for pdf_file in Path(input_dir).glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")

    with open(pdf_file, 'rb') as f:
        files = {'input': f}
        response = requests.post(grobid_url, files=files)

    if response.status_code == 200:
        output_path = Path(output_dir) / f"{pdf_file.stem}.tei.xml"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"  ✓ Saved to {output_path}")
    else:
        print(f"  ✗ Error processing {pdf_file.name}: {response.status_code}")
```

## Output Format

GROBID outputs TEI XML format, which includes:

- `<teiHeader>`: Document metadata (title, authors, affiliations, abstract)
- `<text><body>`: Full text content with section structure
- `<div>`: Sections and subsections
- `<figure>`: Figures and tables
- `<formula>`: Mathematical formulas
- `<biblStruct>`: Structured bibliographic references

### Parsing TEI XML Output

```python
from xml.etree import ElementTree as ET

# Parse the TEI XML
tree = ET.fromstring(tei_xml)

# Define TEI namespace
ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

# Extract title
title = tree.find('.//tei:titleStmt/tei:title', ns)
if title is not None:
    print(f"Title: {title.text}")

# Extract abstract
abstract = tree.find('.//tei:abstract', ns)
if abstract is not None:
    print(f"Abstract: {abstract.text}")

# Extract authors
authors = tree.findall('.//tei:author/tei:persName', ns)
for author in authors:
    forename = author.find('tei:forename', ns)
    surname = author.find('tei:surname', ns)
    if forename is not None and surname is not None:
        print(f"Author: {forename.text} {surname.text}")
```

## Performance Considerations

- **Memory**: Allocate sufficient memory (4GB+ recommended for full text processing)
- **GPU**: Use GPU version for better performance with large batches
- **Concurrent Requests**: GROBID can handle concurrent requests, but consider rate limiting
- **PDF Quality**: Better quality PDFs produce better extraction results
- **Model Selection**: GROBID uses different models for different document sections

## Troubleshooting

### Service not responding
- Check if Docker container is running: `docker ps`
- Check logs: `docker logs <container_id>`
- Verify port 8070 is not in use: `netstat -an | grep 8070`

### Poor extraction quality
- Ensure PDF is text-based (not scanned images)
- Try with GPU support for better accuracy
- Check PDF is a scholarly document (GROBID is optimized for academic papers)

### Memory issues
- Increase Docker memory limit
- Process files individually rather than in large batches
- Use lower memory settings for simpler documents

## Additional Resources

- **Official Documentation**: https://grobid.readthedocs.io/
- **GitHub Repository**: https://github.com/kermitt2/grobid
- **Docker Hub**: https://hub.docker.com/r/grobid/grobid
- **TEI Format**: https://tei-c.org/

## License

GROBID is distributed under Apache 2.0 license.
