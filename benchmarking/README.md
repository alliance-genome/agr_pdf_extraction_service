# PDFX corpus benchmark

This harness submits a locked PDF corpus to an isolated PDFX endpoint and saves
per-paper decision evidence plus Markdown, CSV, JSON, and JSONL reports. It does
not write to ABC Literature.

Run a ten-paper pilot:

```bash
python3 -m benchmarking.pdfx_benchmark run \
  --base-url "$ISOLATED_PDFX_URL" \
  --corpus-dir logs/pdfx-benchmark-corpus \
  --output-dir logs/pdfx-benchmark-results \
  --limit 10 \
  --bearer-token-env PDFX_BENCHMARK_TOKEN
```

Regenerate reports without API or model calls:

```bash
python3 -m benchmarking.pdfx_benchmark analyze \
  --results-dir logs/pdfx-benchmark-results
```

The result directory contains `summary.md`, `cases.csv`, `cases.jsonl`,
`decision-events.jsonl`, and a digest manifest. Article text stays in the
ignored per-case artifact directories; model-call metrics contain hashes and
source spans rather than duplicated text.
