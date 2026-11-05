# Merge rejection audit

Finalization writes content-free diagnostics into its existing S3 NDJSON audit
log on the Alliance-validation fallback path:

- `merge_validation/rejected` records the initial Alliance validation report,
  including rule IDs, parser identity, and output digest.
- `source_delivery/rejected` records each excluded or rejected fallback source
  (`grobid`, `docling`, or `marker`) and reason codes. These distinguish missing
  prerequisites, Alliance validation, parser mismatch, repetition, document
  safety, and candidate exceptions. Exceptions retain the operation and class,
  never their message or traceback.
- `source_delivery/accepted` identifies a successful fallback and its validator
  report, preserving rejected attempts even when the job recovers.

Candidate exception operations distinguish baseline selection, baseline merging,
skeleton rendering, and candidate validation. The pipeline's existing failure
handler flushes these events with the terminal failure. Audit sink failures do
not change selection results. No document text, model responses, or credentials
are included in these events.

The terminal `ConsensusContractError` alone does not imply an Alliance schema
violation in every source: repetition and document safety also reject candidates.
Inspect the preceding events to determine which source failed which check.
