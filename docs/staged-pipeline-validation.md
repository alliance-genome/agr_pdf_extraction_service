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

A separate two-job probe used unique 24-page synthetic PDFs with Marker and
Docling, no merge/provider calls, and 100 ms GPU sampling. Both jobs completed.
It recorded 762 samples, 148 with nonzero GPU utilization, a 100% peak and
7,882 MiB peak GPU allocation. The sample mean was 14.3% across the whole probe.
This confirms GPU execution on a longer fixture; it is not a baseline
comparison or evidence of saturated GPU use.

## Correctness and recovery evidence

- Fresh schema migration, and upgrade of the baseline database with its ten
  existing runs: process IDs, terminal states and artifact metadata preserved.
  Runtime permission checks reject a SELECT-only application role and accept
  SELECT/INSERT/UPDATE without DELETE. A separate real-PostgreSQL proxy test
  verifies joined status, queued-finalizer accounting, read-only enforcement
  and unknown/unavailable state before migration.
- A simulated 30-job batch spanning an hour retained each job's full execution
  budget at first claim; the CPU handoff did not reset that budget. Separate
  queue-age expiration remains bounded.
- Admission connection, ownership-query and slot-query faults propagate through
  source/page/bounded-choice, title/style, outer merge and image-review handlers
  rather than publishing scientific fallbacks. Actual provider failures retain
  their prior tested fallback behavior. Malformed Sentry secrets disable reporting.
- Real PostgreSQL tests cover concurrent dispatchers, duplicate claims, fencing,
  publish failure, backpressure, deadlines, terminal immutability, cross-process
  provider budgets and recovery. A native call deliberately holding the GIL was
  killed by the external watchdog and its deadline persisted. Killing either
  watchdog or sentinel while a worker held the native GIL also terminated
  that worker and released ownership. Actual Compose GPU-worker tests verify
  `init: true`, worker PID greater than 1, container restart and released locks
  after native-GIL hangs under deadline expiry, watchdog death and cancellation.
  Active extraction health remained fresh after 22 seconds holding the GIL.
  Terminal jobs reported zero durable active work despite stale broker unacked
  messages. A reporting-secret endpoint outage
  did not prevent watchdog readiness.
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
- Selective scientific audit upload failures (exception, empty receipt and
  missing file) prevent completion, independently of the NDJSON job audit.
- Completed status, all six text/audit/page download methods and image download
  remained usable with Redis stopped.
- The isolated AMI S3 emulator overlay passed the unchanged text/all-extractor
  and scanned/OCR warm-up contract. Its temporary volumes were removed.
- Dedicated development Sentry ingestion verified for API, proxy, extraction,
  finalization and dispatcher: expected tags/release, absent from production,
  content sentinel removed. Real Flask/FastAPI SDK integration tests cover
  unhandled-error correlation, redaction and one-event reporting. Caught real
  checkpoint and uncertain-provider failures also reached development Sentry.

The AWS dependency-image suite passed 558 tests before final review fixes.
The final admission/error and queue-budget fixes were additionally validated
in a local Python 3.11 CUDA dependency image built from the available pinned
backend base: 576 passed, 24 skipped, with 17 real-PostgreSQL pipeline tests
run separately. Subsequent malformed-secret DSN hardening passed all 21 Sentry
tests in that same dependency image. The final proxy run passed 213 tests,
including its opt-in PostgreSQL case covering bounded replacement protection
for never-claimed jobs and retained protection after first claim. Proxy tests
run separately because both packages are named `app`. PostgreSQL tests opt in
through `PDFX_TEST_DATABASE_URL`, pointing only to a migrated disposable test
DB. Scientific merge, native artifact and page-provenance regression suites
remain part of backend validation. Docker Compose, shell syntax and
CloudFormation validation accompany the runtime tests.

See [the rollout guide](staged-pipeline.md) for drain/migration order, runtime
grants, Sentry activation and rollback constraints. This branch does not deploy
or activate production Sentry by itself.
