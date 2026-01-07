# Benchmarking Methodology

This document describes the methodology for evaluating PDF extraction tools.

## Overview

Each tool will be evaluated on the same set of test PDFs, measuring both quantitative metrics (speed, accuracy) and qualitative aspects (output quality, structure preservation).

## Metrics Framework

### 1. Performance Metrics

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **Processing Time** | Time to extract text from PDF | `time.time()` before/after |
| **Pages/Second** | Throughput rate | Total pages / total time |
| **Memory Usage** | Peak RAM consumption | `tracemalloc` or `memory_profiler` |
| **CPU Usage** | Processor utilization | `psutil` |
| **Cost** | $ per document (cloud services) | API pricing × calls |

### 2. Extraction Quality Metrics

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **Text Completeness** | % of text successfully extracted | Compare to ground truth |
| **Character Accuracy** | Character-level accuracy | Levenshtein distance |
| **Word Accuracy** | Word-level accuracy | Word error rate (WER) |
| **Section Detection** | Accuracy of section identification | F1 score vs ground truth |

### 3. Structure Preservation

| Component | Evaluation Criteria |
|-----------|---------------------|
| **Headings** | Correct hierarchy (H1, H2, H3) |
| **Paragraphs** | Proper paragraph breaks |
| **Lists** | Bulleted/numbered list detection |
| **Tables** | Row/column structure preserved |
| **Figures** | Image extraction, caption association |
| **Equations** | Mathematical notation handling |

### 4. Metadata Extraction

| Field | Description |
|-------|-------------|
| **Title** | Paper title accuracy |
| **Authors** | Author list completeness |
| **Abstract** | Abstract text extraction |
| **Keywords** | Keyword/subject extraction |
| **References** | Bibliography parsing |
| **DOI/PMID** | Identifier extraction |

### 5. Robustness Metrics

| Metric | Description |
|--------|-------------|
| **Failure Rate** | % of PDFs that fail to process |
| **Error Types** | Categorization of failure modes |
| **Recovery** | Graceful handling of partial failures |
| **Edge Cases** | Performance on scanned/complex PDFs |

## Benchmark Scripts

### Basic Benchmark Template

```python
import time
import tracemalloc
from pathlib import Path

def benchmark_tool(tool_name, extract_func, pdf_path):
    """Run benchmark for a single tool on a single PDF."""
    results = {
        'tool': tool_name,
        'pdf': pdf_path,
        'success': False,
        'time_seconds': None,
        'memory_mb': None,
        'error': None,
        'output': None
    }

    tracemalloc.start()
    start_time = time.time()

    try:
        output = extract_func(pdf_path)
        results['success'] = True
        results['output'] = output
    except Exception as e:
        results['error'] = str(e)

    results['time_seconds'] = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    results['memory_mb'] = peak / 1024 / 1024
    tracemalloc.stop()

    return results

def run_benchmarks(pdf_dir, output_file):
    """Run all benchmarks on all PDFs."""
    from tools.grobid.example import extract_with_grobid
    from tools.docling.example import extract_with_docling
    from tools.textract.example import extract_with_textract
    from tools.marker.example import extract_with_marker

    tools = {
        'grobid': extract_with_grobid,
        'docling': extract_with_docling,
        'textract': extract_with_textract,
        'marker': extract_with_marker
    }

    results = []
    for pdf_path in Path(pdf_dir).glob('*.pdf'):
        for tool_name, extract_func in tools.items():
            result = benchmark_tool(tool_name, extract_func, str(pdf_path))
            results.append(result)
            print(f"{tool_name} on {pdf_path.name}: {result['time_seconds']:.2f}s")

    # Save results
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results
```

### Accuracy Evaluation

```python
from difflib import SequenceMatcher

def calculate_accuracy(extracted_text, ground_truth):
    """Calculate text extraction accuracy."""
    # Character-level similarity
    char_similarity = SequenceMatcher(None, extracted_text, ground_truth).ratio()

    # Word-level accuracy
    extracted_words = extracted_text.split()
    truth_words = ground_truth.split()
    word_similarity = SequenceMatcher(None, extracted_words, truth_words).ratio()

    return {
        'character_accuracy': char_similarity,
        'word_accuracy': word_similarity
    }
```

## Test Categories

### Category A: Standard Research Papers
- Clean PDFs from major publishers
- Expected: High accuracy across all tools

### Category B: Complex Layouts
- Multi-column, extensive tables
- Tests structure preservation

### Category C: Figure-Heavy Papers
- Many images, charts, diagrams
- Tests image handling and caption extraction

### Category D: GROBID Failures
- PDFs that GROBID failed to process
- Tests robustness and error handling

### Category E: Edge Cases
- Scanned documents (OCR required)
- Non-standard fonts
- Large file sizes

## Output Format

### Per-PDF Results
```json
{
  "pdf": "AGRKB_101000000645569_mbc-31-1411.pdf",
  "results": {
    "grobid": {
      "success": true,
      "time_seconds": 2.34,
      "memory_mb": 156.7,
      "text_length": 45023,
      "sections_found": 8,
      "tables_found": 3,
      "figures_found": 5,
      "references_found": 42
    },
    "docling": { ... },
    "textract": { ... },
    "marker": { ... }
  }
}
```

### Summary Report
```json
{
  "summary": {
    "total_pdfs": 12,
    "by_tool": {
      "grobid": {
        "success_rate": 0.83,
        "avg_time": 3.2,
        "avg_memory": 180.5
      },
      ...
    }
  }
}
```

## Recommended Workflow

1. **Week 1:** Create ground truth annotations for 3-5 representative PDFs
2. **Week 2:** Implement benchmark harness and run initial tests
3. **Week 3:** Full benchmark run on all test PDFs
4. **Week 4:** Analyze results and generate comparison report

## Tools for Analysis

- **pandas** - Data manipulation and analysis
- **matplotlib/seaborn** - Visualization
- **scikit-learn** - Accuracy metrics (precision, recall, F1)
- **difflib** - Text comparison
