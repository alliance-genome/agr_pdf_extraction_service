# Docling - PDF Document Extraction Tool

## What is Docling?

Docling is a comprehensive document processing SDK and CLI tool developed by IBM Research that parses diverse document formats—including PDF, DOCX, PPTX, XLSX, HTML, audio, and more—into a unified document representation for downstream AI workflows.

### Key Features

- **Advanced PDF Understanding**: Layout analysis, reading order detection, table structure extraction, OCR, formula recognition, and image classification
- **Multiple Document Formats**: PDF, DOCX, PPTX, XLSX, HTML, audio, and more
- **Privacy-Focused**: Runs entirely locally for air-gapped and sensitive data environments (no external API calls required)
- **AI Integration**: Seamlessly integrates with LangChain, LlamaIndex, Haystack, and Crew AI frameworks
- **Multiple Output Formats**: Markdown, HTML, JSON, and DocTags
- **Visual Language Models**: Supports VLMs like GraniteDocling for end-to-end document understanding
- **OCR Support**: Multiple OCR engines including EasyOCR with multi-language support
- **Table Extraction**: Advanced table structure detection and export to DataFrames
- **Hardware Acceleration**: Support for CPU, CUDA, and MPS devices
- **Batch Processing**: Process multiple documents efficiently

## Installation

### Basic Installation

```bash
pip install docling
```

### Installation with VLM Support

For Visual Language Model support (recommended for image-based document processing):

```bash
pip install docling[vlm]
```

## Basic Usage

### Simple Document Conversion

```python
from docling.document_converter import DocumentConverter

# Convert from URL or local file path
source = "https://arxiv.org/pdf/2408.09869"  # or "/path/to/document.pdf"
converter = DocumentConverter()
result = converter.convert(source)

# Export to Markdown
markdown_output = result.document.export_to_markdown()
print(markdown_output)
```

### Converting to Different Formats

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")

# Export to Markdown
markdown = result.document.export_to_markdown()

# Export to JSON
json_dict = result.document.export_to_dict()

# Export to HTML
html = result.document.export_to_html()

# Export to DocTags
doctags = result.document.export_to_doctags()

# Check conversion status
print(result.status)  # ConversionStatus.SUCCESS
print(result.input.file)  # Source file path
```

## Advanced Features

### Configure OCR and Table Extraction

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    TableFormerMode,
)
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

# Configure PDF pipeline with advanced options
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.do_code_enrichment = True
pipeline_options.do_formula_enrichment = True
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True

# Configure OCR with specific languages
pipeline_options.ocr_options = EasyOcrOptions(
    lang=["en", "de", "fr"],
    confidence_threshold=0.5
)

# Configure table extraction mode
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
pipeline_options.table_structure_options.do_cell_matching = True

# Set up hardware acceleration
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4,
    device=AcceleratorDevice.AUTO  # or CPU, CUDA, MPS
)

# Create converter with pipeline options
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)
```

### Navigate Document Structure

```python
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel

converter = DocumentConverter()
result = converter.convert("document.pdf")
doc = result.document

# Access all text items
for item in doc.iterate_items():
    if item.label == DocItemLabel.TEXT or item.label == DocItemLabel.PARAGRAPH:
        print(f"Text: {item.text}")

# Get all section headers
headers = [item for item in doc.iterate_items() if item.label == DocItemLabel.SECTION_HEADER]
for header in headers:
    print(f"Section: {header.text}")

# Access tables and export to DataFrames
for table in doc.tables:
    df = table.export_to_dataframe()
    print(f"Table shape: {df.shape}")

# Access pictures/figures
for picture in doc.pictures:
    print(f"Picture: {picture.caption if hasattr(picture, 'caption') else 'No caption'}")

# Get document metadata
print(f"Page count: {result.input.page_count}")
print(f"File size: {result.input.filesize}")
print(f"Format: {result.input.format}")
```

## Command Line Interface

Docling also provides a CLI for document conversion:

```bash
# Basic conversion
docling https://arxiv.org/pdf/2206.01062

# Convert with output to specific directory
docling document.pdf --output ./output

# Batch convert multiple files
docling file1.pdf file2.docx file3.pptx --output ./converted

# Export to specific format
docling document.pdf --to markdown --output output.md
docling document.pdf --to json --output output.json
docling document.pdf --to html --output output.html

# Enable OCR
docling scanned.pdf --ocr --ocr-engine easyocr

# Configure table extraction
docling document.pdf --table-structure-mode accurate

# Set hardware acceleration
docling document.pdf --device cuda
```

## Use Cases

- **RAG Applications**: Extract structured content for Retrieval-Augmented Generation systems
- **Document Understanding**: Parse and analyze complex document layouts
- **Knowledge Extraction**: Extract tables, formulas, and structured data from PDFs
- **Content Migration**: Convert legacy documents to modern formats
- **Data Pipeline Integration**: Process documents as part of automated workflows
- **Research Paper Analysis**: Extract structured information from academic papers

## Integration with AI Frameworks

Docling integrates seamlessly with popular AI frameworks:

- **LangChain**: Use Docling documents in LangChain pipelines
- **LlamaIndex**: Integrate with LlamaIndex for document indexing
- **Haystack**: Process documents in Haystack pipelines
- **Crew AI**: Use in multi-agent AI workflows

## Documentation and Resources

- **GitHub**: https://github.com/docling-project/docling
- **Official Documentation**: https://docling-project.github.io/docling/
- **Example Notebooks**: Check the `docs/examples/` directory in the GitHub repository

## Performance Considerations

- **Hardware Acceleration**: Use GPU (CUDA/MPS) for faster processing of large documents
- **Multi-threading**: Configure `num_threads` for parallel processing
- **Batch Processing**: Process multiple documents in a single batch for better efficiency
- **Format-Specific Options**: Customize pipeline options based on document type (scanned vs. programmatic PDFs)

## License

Docling is developed by IBM Research. Check the GitHub repository for specific license information.
