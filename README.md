# PDF Extraction Benchmark

This project benchmarks and compares multiple scientific PDF extraction tools—**GROBID**, **Docling**, **Marker**, and an LLM-based merger—by providing a web interface to upload PDFs, extract structured content, and download or view results.

## Features

- Upload scientific PDFs and extract content using multiple tools.
- Compare outputs from GROBID, Docling, and Marker.
- Merge extractions using a Large Language Model (Anthropic Claude).
- Caching for fast repeated access.
- Download individual or merged extractions in Markdown format.
- Simple web interface (Flask-based).

## Requirements

- Python 3.8+
- Linux (recommended)
- [Anthropic API key](https://docs.anthropic.com/claude/docs/quickstart) (for LLM merging)
- System dependencies for extraction tools (see below)

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/pdf_extraction_benchmark.git
    cd pdf_extraction_benchmark
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Python dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```

4. **Install system dependencies for extraction tools:**
    - **GROBID:** Requires Java and GROBID server running (see [GROBID docs](https://github.com/kermitt2/grobid)). We recommend using docker. Make sure the server is running: ```bash
    docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2
    ```. The default GROBID URL is set to http://localhost:8070. For a custom location, change grobid.py.
    - **Docling** and **Marker:** May require additional binaries or Python packages. See their respective documentation.

5. **Set your Anthropic API key (for LLM merging):**
    - **Setting an env variable:** ```bash
    export ANTHROPIC_API_KEY=sk-...your-key...
    ```
    - **Editting config.py:** ```bash
    ANTHROPIC_API_KEY = "sk-...your-key..."
    ```

## Usage

1. **Start the Flask server:**
    ```bash
    python server.py
    ```

2. **Open your browser and go to:**
    ```
    http://localhost:5000
    ```

3. **Upload a PDF, select extraction methods, and process.**
    - Download or view the extracted Markdown outputs.
    - Optionally merge outputs using the LLM.

## Running tests

```bash
python3 -m pytest
```

## Notes

- Ensure all extraction tools are installed and accessible.
- The LLM merge feature requires a valid Anthropic API key.
- For production, set `debug=False` in `run.py`.

## License

MIT License

---

**Questions?**  
Open an issue or contact the maintainer.