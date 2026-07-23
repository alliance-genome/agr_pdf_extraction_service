# AGR PDF Extraction Service

PDFX turns scientific PDFs into Alliance Markdown. It runs GROBID, Docling,
and Marker independently, preserves each extractor's native result, and builds
one merged document from exact source-backed spans.

The model is a selector, never an author: PDFX gives it numbered paths that
already exist in extractor output, receives integer choices, and performs the
byte assembly itself. No model response can insert publication text.

## Architecture

- The always-on Fargate proxy accepts PDFs, durably queues uploads in S3, and
  wakes the GPU backend Auto Scaling Group.
- The GPU EC2 backend runs nginx, Flask/Gunicorn, Celery, Redis, PostgreSQL,
  GROBID, Docling, and Marker with Docker Compose.
- The proxy replays queued work only after the backend and Marker worker report
  ready.
- Completed artifacts and audit logs are stored in S3. The GPU backend returns
  to zero desired capacity when idle.

See [deploy/aws/pdfx.md](deploy/aws/pdfx.md) for the AWS runbook.

## Extraction contract

Each requested extractor is independent. A job continues when one or two
extractors fail as long as at least one produces a complete, usable artifact.
The result reports both `available_extractors` and `failed_extractors`.

Native artifacts are committed manifest-last and bind:

- the source PDF digest;
- the exact Markdown bytes and digest;
- the extractor-native bytes and digest;
- the extractor name and pinned package version;
- page-coverage evidence when the native format can prove it.

The native formats are:

| Extractor | Markdown source | Native evidence |
|---|---|---|
| GROBID | Official Alliance TEI-to-Markdown converter | TEI XML |
| Docling | `DocumentConverter` Markdown export | Docling JSON |
| Marker | Marker Markdown renderer | Marker JSON renderer output |

Runtime package pins live in [requirements.txt](requirements.txt). Change the
adapter, native-artifact contract, package pin, and tests together.

GROBID TEI does not currently provide a reliable complete page inventory to
this adapter. Its manifest records page coverage as unavailable, so a
GROBID-baseline delivery is labeled `failsafe` rather than `qualified`.

## Source-backed merge

The merger has one runtime implementation:

1. Validate completed source artifacts and choose a structurally complete
   baseline.
2. Align source structural units and construct bounded executable paths.
3. Reject paths that lose baseline coverage, overlap source scope, introduce
   unsafe Unicode, or increase repeated content beyond every source.
4. Resolve exact agreements in the application.
5. Ask Terra to choose a numbered path for ordinary unresolved regions.
6. Ask Sol to choose a numbered path for protected-payload and skeleton
   conflicts, or after a typed Terra invalid response, refusal, or timeout.
   Requests remain individually bounded, but there is no per-job Sol-region
   quota that can silently skip otherwise valid work.
7. Assemble only exact source bytes, preserve source-backed italics, and
   normalize the terminal newline.
8. Validate the exact output with the pinned Alliance parser and its actual
   downstream reader, then commit the output, metrics, audit, and manifest.

If a selection call fails or no path is safe, the verified complete baseline
span is retained. This is the delivery safety property of the one merge
implementation; it is not an alternate merger or an old compatibility path.
The whole job fails only when no extractor produces nonempty structurally and
byte-safe source text. If a complete source has an Alliance heading or table
structure error, the same renderer changes only the required structural
markers, preserves the source-backed publication text, and labels unresolved
selection or coverage evidence as `failsafe`; it is never mislabeled
`qualified`.

### Alliance Markdown

PDFX validates the exact merged bytes with
`agr-abc-document-parsers==1.6.0`. It does not ask a model to repair headings or
rewrite prose, and it does not run a normalizing emitter over arbitrary
extractor content. Validator and downstream-reader receipts are bound to the
output SHA-256 in merge metrics and the commit manifest.

Italics are publication data. When aligned sources show an unambiguous
formatting-only omission, PDFX copies the complete italic-bearing span from the
source artifact and records the exact byte provenance. It never synthesizes
emphasis markers around model-authored text.

### Duplicate-content protection

Alternative paths must preserve monotonic, non-overlapping source scope. A
narrow scope-spill guard rejects obvious passage movement from neighboring
regions. After assembly, paragraph and token-shingle diagnostics reject any
output repetition beyond the maximum observed in the source artifacts; the
provenance validator also forbids reusing one source occurrence.

## Model policy

Every reachable model route is fixed to the GPT-5.6 series:

| Role | Model | Reasoning | Purpose |
|---|---|---|---|
| `source_selection` | `gpt-5.6-terra` | medium | Ordinary numbered source-path choices |
| `hard_selection` | `gpt-5.6-sol` | high | Bounded hard-region choice |
| `image_text_review` | `gpt-5.6-luna` | medium | Text-only image artifact metadata review |

Startup rejects a missing, older, cross-role, or incorrectly tiered model.
Selection responses contain only the request digest, region IDs, and integer
choices. OpenAI calls are bounded by `LLM_OPENAI_TIMEOUT_SECONDS` and
`LLM_OPENAI_MAX_RETRIES`.

## REST API

Swagger UI is served at `/docs`; the OpenAPI document is at `/openapi.yaml`.

Typical flow:

```bash
curl -X POST https://HOST/api/v1/extract \
  -H "Authorization: Bearer $PDFX_TOKEN" \
  -F "file=@paper.pdf" \
  -F 'methods=["grobid","docling","marker"]' \
  -F "merge=true"

curl -H "Authorization: Bearer $PDFX_TOKEN" \
  https://HOST/api/v1/extract/PROCESS_ID

curl -H "Authorization: Bearer $PDFX_TOKEN" \
  https://HOST/api/v1/extract/PROCESS_ID/download/merged
```

Important endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Shallow service health |
| `GET` | `/api/v1/health?deep=true` | Dependency and backend readiness |
| `GET` | `/api/v1/config` | Exact merge contract and model routes |
| `POST` | `/api/v1/extract` | Submit a PDF |
| `GET` | `/api/v1/extract/{process_id}` | Poll status and progress |
| `POST` | `/api/v1/extract/{process_id}/cancel` | Cancel active work |
| `GET` | `/api/v1/extract/{process_id}/download/{method}` | Download verified output |
| `GET` | `/api/v1/extract/{process_id}/logs` | Obtain an audit-log URL |
| `GET` | `/api/v1/extract/{process_id}/artifacts/urls` | Obtain artifact URLs |

Merged downloads are served from a locally verified bundle or the exact durable
S3 artifact uploaded during finalization. A merge job is not marked successful
unless that durable upload succeeds, so a completed job does not depend on the
GPU host's local cache remaining present.

## Configuration

Copy `.env.example` to `.env`. Secrets belong in local environment files,
SSM, or ECS secret injection and must never be committed.

Key settings:

| Variable | Default | Meaning |
|---|---:|---|
| `OPENAI_API_KEY` | empty | Enables model selection and image review |
| `SOURCE_SELECTION_MODEL` | `gpt-5.6-terra` | Ordinary source selector |
| `SOURCE_SELECTION_REASONING` | `medium` | Required Terra reasoning |
| `HARD_SELECTION_MODEL` | `gpt-5.6-sol` | Hard-region selector |
| `HARD_SELECTION_REASONING` | `high` | Required Sol reasoning |
| `IMAGE_TEXT_REVIEW_MODEL` | `gpt-5.6-luna` | Image metadata reviewer |
| `IMAGE_TEXT_REVIEW_REASONING` | `medium` | Required Luna reasoning |
| `LLM_OPENAI_TIMEOUT_SECONDS` | `180` | Per-call timeout |
| `LLM_OPENAI_MAX_RETRIES` | `1` | OpenAI client retries, maximum 2 |
| `LLM_COST_ALERT_USD_PER_JOB` | `2.0` | Non-blocking per-job cost warning threshold |
| `TASK_SOFT_TIME_LIMIT_SECONDS` | `1800` | Overall Celery soft deadline |
| `TASK_HARD_TIME_LIMIT_SECONDS` | `2100` | Overall Celery hard deadline |
| `EXTRACTION_FINALIZATION_RESERVE_SECONDS` | `300` | Recovery stops to preserve finalization time |
| `MAX_CONTENT_LENGTH` | `500 MiB` | Backend upload limit |
| `PDFX_MARKER_READY_FILE` | cache path | Worker-process Marker readiness receipt |

## Development and validation

The backend image carries the production dependency set. Focus tests carefully
because both backend and proxy expose an `app` package.

```bash
python3 -m pytest -q proxy/tests
python3 -m pytest -q tests/test_aws_pdfx_stack.py
bash -n deploy/deploy.sh
bash -n deploy/aws/deploy_idle_guard.sh
python3 -m py_compile deploy/aws/lambda/pdfx_idle_guard.py
python3 -m compileall -q proxy/app
git diff --check
```

For backend application tests, run `python -m pytest -q tests` in the pinned
backend image or an environment installed from `requirements.txt`.

## Deployment

Proxy deployments update the ECS task definition through
`proxy/deploy/deploy.sh`. Backend deployments use a prebuilt immutable GPU
image through `deploy/deploy.sh`; production Compose must run code baked into
that image, not a source bind mount.

Do not stop or replace a production backend while jobs are active unless the
operator explicitly authorizes an interrupting hotfix. After deployment,
verify the exact image tag, CUDA in the worker container, Marker readiness, an
end-to-end extraction, queue drain, and return to cost-safe idle capacity.

## Security

- Never commit PDFs, credentials, tokens, private host details, account IDs,
  SSH key paths, or raw secure parameter values.
- Keep host access notes in ignored files such as `SSH_ACCESS.md`.
- Prefer IAM roles, SSM, and ECS secret injection over copied environment
  values.

## License

See [LICENSE](LICENSE).
