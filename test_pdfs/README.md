# Test PDF Collections

This directory contains curated PDF samples for benchmarking PDF extraction tools.

## Download Test PDFs

PDFs are stored in S3 (too large for Git). Download with:

```bash
# Download all test PDFs (~274 MB)
aws s3 sync s3://agr-pdf-extraction-benchmark/test_pdfs/ ./

# Download just core set (~61 MB)
aws s3 sync s3://agr-pdf-extraction-benchmark/test_pdfs/core/ ./core/

# Download just extended set (~213 MB)
aws s3 sync s3://agr-pdf-extraction-benchmark/test_pdfs/extended/ ./extended/
```

See [AWS_ACCESS_GUIDE.md](../AWS_ACCESS_GUIDE.md) for AWS credentials setup.

---

## Directory Structure

```
test_pdfs/
├── core/           # 12 carefully selected PDFs from Alliance corpus
├── extended/       # Additional problem PDFs (GROBID failures, edge cases)
└── README.md       # This file
```

---

## Core Test Set (`core/`)

12 PDFs selected to maximize variety across organisms, journals, content types, and extraction challenges.

### Summary

| Metric | Value |
|--------|-------|
| Total Files | 12 PDFs |
| Total Size | ~61 MB |
| MOD Coverage | All 7 (FlyBase, MGI, RGD, SGD, WormBase, Xenbase, ZFIN) |
| GROBID Failures | 2 papers |
| Open Access | All (PMC available) |

### PDF Details

| # | AGRKB ID | Journal | MODs | Size | Special Features |
|---|----------|---------|------|------|------------------|
| 1 | 101000000645569 | Mol Biol Cell | XB, ZFIN | 2.15 MB | ⚠️ GROBID failure |
| 2 | 101000000993203 | Genetics | 5 MODs | 1.45 MB | ⚠️ GROBID failure, Multi-MOD |
| 3 | 101000000662381 | Genome Biol | ZFIN | 4.57 MB | 📊 Tables, supplements |
| 4 | 101000000986757 | Sci Adv | ZFIN | 5.11 MB | 🖼️ Rich figures, multimedia |
| 5 | 101000001051113 | Mol Metab | RGD, MGI | 4.08 MB | 🖼️ Figures |
| 6 | 101000001193578 | Neural Regen Res | MGI | 10.23 MB | 📊 Tables |
| 7 | 101000001183100 | Fly (Austin) | FB | 0.92 MB | Smallest file (review) |
| 8 | 101000001190073 | Gut Microbes | WB | 9.17 MB | Diverse supplements |
| 9 | 101000001197127 | Synth Syst Biotechnol | SGD | 6.77 MB | Tables + figures |
| 10 | 101000001031176 | Elife | MGI, RGD | 7.13 MB | 📁 Large (92 files total) |

### Selection Criteria

**Diversity:**
- ✅ All 7 Model Organism Databases represented
- ✅ 10 unique journals
- ✅ Size range: 0.92 - 10.23 MB
- ✅ Mix of tables, figures, reviews, research articles

**Challenges:**
- ✅ 2 papers where GROBID failed but PDFs appear valid
- ✅ Complex layouts with tables
- ✅ Rich figure content

---

## Extended Test Set (`extended/`)

GROBID failure cases from SCRUM-5561 ticket - PDFs that appear valid but failed extraction.

### Summary

| Metric | Value |
|--------|-------|
| Total Files | 42 PDFs (13 main + 29 supplements) |
| Total Size | ~213 MB |
| Papers | 13 unique papers |
| Source | SCRUM-5561 GROBID failures |

### Papers Included

| AGRKB ID | Title | Main | Supplements |
|----------|-------|------|-------------|
| 101000000990125 | The small molecule activator S3969... | 4.6 MB | 1 |
| 101000000991131 | Revealing mitf functions... | 9.8 MB | 3 |
| 101000001190066 | Glial subtype-specific modulation... | 1.5 MB | 20 |
| 101000001192774 | Multiplex metabolic engineering... | 3.4 MB | 0 |
| 101000001193732 | Distinct systemic impacts of Aβ42... | 21.5 MB | 0 |
| 101000001194885 | Hypoimmune CD19 CAR T cells... | 5.1 MB | 1 |
| 101000001195772 | Integrins coordinate basal surface... | 52.0 MB | 0 |
| 101000001196773 | The mind of a predatory worm | 1.0 MB | 0 |
| 101000001198025 | The inhibitory effect of tyrosine... | 2.2 MB | 0 |
| 101000001198055 | Strain-specific effects of Desulfovibrio... | 1.6 MB | 0 |
| 101000001198164 | Decomposed Linear Dynamical Systems... | 5.6 MB | 3 |
| 101000001199967 | Multiple cis-regulatory modules... | 3.8 MB | 0 |
| 101000001200143 | Gal4 drivers of the geosmin receptor... | 2.1 MB | 1 |

### File Naming Convention

Files are named to group main papers with their supplements:
- **Main papers**: `AGRKB_<ID>_<filename>.pdf`
- **Supplements**: `AGRKB_<ID>_supplement_<filename>.pdf`

Example:
```
AGRKB_101000001190066_main.pdf           # Main paper
AGRKB_101000001190066_supplement_mmc1.pdf  # Supplement 1
AGRKB_101000001190066_supplement_mmc2.pdf  # Supplement 2
```

### Why These Papers?

All 13 papers failed GROBID extraction despite appearing to be valid PDFs. They represent real-world edge cases for benchmarking extraction tool robustness

---

## Downloading Additional PDFs

PDFs are stored in `s3://agr-literature/prod/reference/documents/` with paths based on MD5 hash.

### S3 Path Format
```
s3://agr-literature/prod/reference/documents/{md5[0]}/{md5[1]}/{md5[2]}/{md5[3]}/{md5}.gz
```

### Quick Download Example
```bash
# Given MD5: 395a9bcea1168b2adfed49720c4bcff1
aws s3 cp s3://agr-literature/prod/reference/documents/3/9/5/a/395a9bcea1168b2adfed49720c4bcff1.gz - | gunzip > output.pdf
```

Contact Chris Tabone for database access if you need to query additional paper metadata.
