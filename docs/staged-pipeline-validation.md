# Staged pipeline validation

Validation used an isolated temporary g6.4xlarge (16 vCPU, 64 GiB RAM, NVIDIA
L4), PostgreSQL 16, Redis 7 and real S3. The serial baseline was main commit
`670163951fcba80a003e1e63cdfc7180566a7fc2`. Both versions reused the same baked
model caches and ran text/all-three-extractor and scanned/Docling warm-ups.
No production queue, database, bucket or Auto Scaling Group was modified.

## Scheduling experiment

Each burst submitted four alternating text and scanned fixtures. Both versions
used real extractors and deterministic source-backed merge, with no provider
key. A lab-only 20-second wait before artifact publication represented slow
downstream work. This measures scheduling overlap, not LLM quality or a promised
production speedup.

| Run | Four-job makespan | Per-job median latency | Per-job p95 latency |
|---|---:|---:|---:|
| Serial baseline 1 | 92.85 s | 57.63 s | 88.35 s |
| Serial baseline 2 | 91.86 s | 57.14 s | 87.51 s |
| Staged candidate 1 | 53.35 s | 40.92 s | 51.74 s |
| Staged candidate 2 | 53.15 s | 39.27 s | 51.31 s |

All 16 jobs completed. Stage timestamps demonstrate extraction of later PDFs
while earlier PDFs finalize. These are four-job descriptive percentiles, not
population estimates. Coarse GPU telemetry missed the tiny fixtures' short
kernels; it does not establish a GPU utilization percentage improvement. Peak
sampled GPU allocation was 4,184 MiB. Production sizing and longer-document
throughput require representative workload validation.

## Correctness and recovery evidence

- Fresh schema migration, and upgrade of the baseline database with its ten
  existing runs: process IDs, terminal states and artifact metadata preserved.
  Runtime permission checks reject a SELECT-only application role.
- Real PostgreSQL tests cover concurrent dispatchers, duplicate claims, fencing,
  publish failure, backpressure, deadlines, terminal immutability, cross-process
  provider budgets and recovery. A native call deliberately holding the GIL was
  killed by the external watchdog and its deadline persisted.
- Redis stopped during acceptance, duplicate submission, then Redis restarted
  with an empty queue: the same durable job completed.
- Six-job saturation with finalizers stopped: four downstream jobs admitted,
  two held ready. Admission resumed after finalizers restarted; queued
  cancellation stayed terminal.
- GPU worker killed before checkpoint: safe extraction retry completed.
  Finalizer killed before provider work: checkpoint restored and completed.
  Finalizer killed after the production provider-slot boundary was entered:
  `UncertainProviderOutcome`, with no automatic replay. The boundary test was
  synthetic; it did not send a paid request.
- Corrupt S3 checkpoint: checksum rejection and explicit failure. Cancellation
  during finalization: terminal cancellation remained immutable.
- EC2 host reboot with a retained checkpoint: same generation and extraction
  end time, followed by successful finalization. This proves recovery with the
  same EBS database volume and S3; it does not prove instance termination or
  replacement recovery. Production requires its external PostgreSQL database.
- A three-page table-and-figure PDF submitted concurrently with images on/off:
  both completed all three extractors with fresh caches. Native byte digests,
  style/page receipts and required S3 objects verified; one image retained only
  for the image-enabled job. No cross-job cache interference.
- Completed status, all six text/audit/page download methods and image download
  remained usable with Redis stopped.
- The isolated AMI S3 emulator overlay passed the unchanged text/all-extractor
  and scanned/OCR warm-up contract. Its temporary volumes were removed.
- Dedicated development Sentry ingestion verified for API, proxy, extraction,
  finalization and dispatcher: expected tags/release, absent from production,
  content sentinel removed. Real Flask/FastAPI SDK integration tests cover
  unhandled-error correlation, redaction and one-event reporting. Caught real
  checkpoint and uncertain-provider failures also reached development Sentry.

The backend suite runs in the deployed Python/CUDA dependency image; proxy tests
run separately because both packages are named `app`. PostgreSQL tests opt in
through `PDFX_TEST_DATABASE_URL`, pointing only to a migrated disposable test
DB. Scientific merge, native artifact and page-provenance regression suites
remain part of backend validation. Docker Compose, shell syntax and
CloudFormation validation accompany the runtime tests.

See [the rollout guide](staged-pipeline.md) for drain/migration order, runtime
grants, Sentry activation and rollback constraints. This branch does not deploy
or activate production Sentry by itself.
