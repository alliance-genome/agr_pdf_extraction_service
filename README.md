# PDF Extraction Benchmarking Toolkit

A benchmarking workspace for evaluating PDF extraction tools for scientific literature processing in the Alliance of Genome Resources AI curation pipeline.

## Project Goal

Evaluate and compare PDF extraction tools to determine the best approach for processing ~2 million scientific PDFs in the Alliance literature corpus. We need to understand how each tool performs, what parameters can be tuned, and which output formats work best for downstream AI processing.

## Tools Being Evaluated

| Tool | Type | GPU Support | Output Formats |
|------|------|-------------|----------------|
| **[GROBID](tools/grobid/)** | Self-hosted (Docker) | No | TEI-XML, BibTeX |
| **[Docling](tools/docling/)** | Self-hosted (pip) | Optional | Markdown, JSON, HTML |
| **[AWS Textract](tools/textract/)** | Cloud API | N/A | JSON (structured) |
| **[Marker](tools/marker/)** | Self-hosted (pip) | Yes (recommended) | Markdown, JSON |

Each tool has its own directory under `tools/` with:
- **README.md** - Overview, installation instructions, configuration options
- **example.py** - Working extraction script to get started

---

## Repository Structure

```
pdf_extraction_benchmark/
├── README.md                    # This file
├── AWS_ACCESS_GUIDE.md          # S3 access and AWS CLI setup
│
├── tools/                       # Extraction tool documentation & examples
│   ├── grobid/
│   │   ├── README.md            # Docker setup, API usage, config options
│   │   └── example.py           # Python extraction example
│   ├── docling/
│   │   ├── README.md            # pip install, pipeline options, GPU acceleration
│   │   └── example.py
│   ├── textract/
│   │   ├── README.md            # AWS setup, API options, cost considerations
│   │   └── example.py
│   └── marker/
│       ├── README.md            # GPU setup, batch processing, quality settings
│       └── example.py
│
├── test_pdfs/                   # Download from S3 (not in Git)
│   ├── README.md                # Full manifest of test documents
│   ├── core/                    # 12 curated PDFs (~61 MB)
│   └── extended/                # 42 GROBID failure cases (~213 MB)
│
├── benchmarks/
│   └── README.md                # Metrics to capture, comparison ideas
│
└── results/                     # Your benchmark outputs go here
```

---

## Getting Started

### 1. Download Test PDFs from S3

Test PDFs are stored in a private S3 bucket (~274 MB total):

```bash
# Download all test PDFs
aws s3 sync s3://agr-pdf-extraction-benchmark/test_pdfs/ ./test_pdfs/

# Or just the core set to start (~61 MB)
aws s3 sync s3://agr-pdf-extraction-benchmark/test_pdfs/core/ ./test_pdfs/core/
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib boto3
```

### 3. Try the Tools

Each tool directory has an `example.py` you can run immediately:

```bash
# Pick a test PDF
PDF="test_pdfs/core/AGRKB_101000000645569_mbc-31-1411.pdf"

# Try each tool
python tools/grobid/example.py $PDF
python tools/docling/example.py $PDF
python tools/textract/example.py $PDF
python tools/marker/example.py $PDF
```

See individual `tools/*/README.md` files for installation and configuration details.

---

## Test PDFs

### Core Set (`test_pdfs/core/`) - 12 PDFs

A curated selection covering:
- All 7 Model Organism Databases (MODs)
- 10 unique journals
- Mix of tables, figures, reviews, research articles
- 2 known GROBID failure cases
- Size range: 0.92 - 10.23 MB

### Extended Set (`test_pdfs/extended/`) - 42 PDFs

PDFs from SCRUM-5561 that caused GROBID failures:
- 13 main papers + 29 supplementary files
- Known problematic documents (complex layouts, scanned content, etc.)
- Good for stress-testing extraction tools

See **[test_pdfs/README.md](test_pdfs/README.md)** for the complete manifest with file sizes, journals, and known issues.

---

## What to Explore

This is an exploratory project. The goal is to understand each tool's capabilities and find what works best for our use case. Some areas to investigate:

### Performance
- Processing speed (time per PDF, pages per second)
- Memory usage
- GPU vs CPU performance (where applicable)
- Batch processing capabilities
- Cost (for Textract)

### Extraction Quality
- Text accuracy and completeness
- Structure preservation (sections, headings, paragraphs)
- Table extraction and formatting
- Figure/image handling
- Reference/bibliography parsing
- Metadata extraction (title, authors, abstract, DOI)

### Configuration & Tuning
- What parameters can be adjusted?
- Quality vs speed tradeoffs
- Output format options (Markdown, JSON, XML, etc.)
- Which format is easiest for downstream AI processing?

### Robustness
- How do tools handle problematic PDFs?
- Failure rates on the extended test set
- Error messages and recovery options

### Output Comparison
- Compare the same PDF across all 4 tools
- Which preserves structure best?
- Which handles tables best?
- Which is most "AI-ready" for LLM consumption?

---

## EC2 Instance

A GPU instance (g4dn.xlarge) is available for running benchmarks:

- **Instance ID:** `i-084041f4ad289ed85`
- **Specs:** 4 vCPU, 16GB RAM, T4 GPU (16GB VRAM)
- **Private IP:** `172.31.92.31` (connect via Alliance VPN)
- **Pre-installed:** Docker, Python 3, NVIDIA drivers, CUDA

```bash
# Start the instance
aws ec2 start-instances --instance-ids i-084041f4ad289ed85

# SSH in (after connecting to VPN)
ssh -i pedro-benchmark-key.pem ec2-user@172.31.92.31

# Stop when done
aws ec2 stop-instances --instance-ids i-084041f4ad289ed85
```

---

## Deliverables

When you've explored the tools, we're looking for:

1. **Findings** - What works, what doesn't, surprises discovered
2. **Recommendations** - Which tool(s) should we use for the AI pipeline?
3. **Sample outputs** - Examples of extraction results for comparison
4. **Any scripts/tools** you create along the way

Format is flexible - could be a report, Jupyter notebooks, comparison spreadsheets, whatever communicates the findings best.

---

## Related Resources

**S3 Buckets:**
- `s3://agr-pdf-extraction-benchmark/` - Test PDFs for this project
- `s3://agr-literature/prod/reference/documents/` - Full literature corpus (~2M PDFs)

**Jira:**
- KANBAN-874 - This benchmarking project
- SCRUM-5561 - Original GROBID benchmarking (source of extended test set)

**Contact:** Chris Tabone (Alliance)
