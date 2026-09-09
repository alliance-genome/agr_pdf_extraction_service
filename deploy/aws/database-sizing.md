# Backend database sizing

The backend metadata database uses `db.t4g.small` (2 GiB RAM). The reference
CloudFormation parameter is `BackendDbInstanceClass` in `pdfx-stack.yaml`.
Production was created through the AWS CLI and is not managed by this reference
stack. Do not deploy the stack over the existing database without first
reconciling resource ownership and configuration.

The September 2026 move from `db.t4g.micro` adds memory headroom after repeated
80 MiB freeable-memory alarms. CPU, connection counts, and storage latency were
low; this change does not increase GPU extraction concurrency.

For an authorized production resize:

1. Check both proxy and backend health. Wait until the durable proxy queue,
   backend active/queued runs, and broker queued/unacknowledged counts are zero.
   Unknown or unreachable health is not evidence of an idle service.
2. Inspect RDS status and pending modifications. Applying immediately must not
   accidentally apply unrelated pending changes.
3. Modify only the existing database's instance class through the AWS CLI.
   RDS class changes interrupt database availability; use an idle window.
4. Wait for RDS to report the requested class, `available`, and no pending
   modification. Verify backend and proxy database connectivity, then check
   fresh memory metrics and alarm recovery.

An idle check does not prevent new submissions during the resize. Observe the
queue and service recovery throughout the operation.
