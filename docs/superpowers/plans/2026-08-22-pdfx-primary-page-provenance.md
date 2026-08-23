# PDFX Primary Page-Provenance Goal

**Date:** 2026-08-22

**Status:** Complete in PDFX production; parser 1.7.2 released, PDFX PR #48
deployed, and all three replacement canaries verified

**PDFX implementation base:** `origin/main` at `9747452`

**Parser implementation base:** `agr_abc_document_parsers` `origin/main` at `a37100e` / `v1.6.0`

**Reference evidence only:** PDFX PR #42 and commit `05687ea`

**Official goal:** Active for the complete parser/PDFX/review/release/deploy
sequence in this document. This document remains the scope and acceptance
authority for that goal.

### Implementation evidence ledger

- Parser PR #2 merged as `7efc257bb858449fab9e4d96f17cfa031a9402cb`
  and established the additive API in 1.7.0. Production canary inspection then
  found that a generated `Figure Legends` heading could remain residual and
  receive the wrong Luna page after merge interleaving defeated PDFX's
  adjacency ownership rule. Parser PR #3 merged as
  `37a63876db1bd5345a5664df8e980fd57031cc95`, and tag `v1.7.1` binds generated
  figure/reference headings to their first emitted entry without changing
  Markdown bytes. Replacement canaries then exposed missing native provenance
  for generated Acknowledgments, Funding, and Availability headings. Parser PR
  #4 merged as `1875dcabf0ff690ac501c9279f4e3ce210647cfb`; tag `v1.7.2`
  corrects those headings without changing Markdown bytes. PDFX now pins
  `agr-abc-document-parsers==1.7.2`.
- Parser source digest is pinned by PDFX as
  `fea04b17244c8a262867852daa2b3a5c922e529f858403caececaf3f671e3bab`.
- Parser validation: 543 passed, 4 skipped, 3 deselected; repeated required
  Sol/xhigh `$max-review-skill` and bounded Claude reviews accepted the final
  parser change with no blocker, material correction, or production-code
  change remaining.
- The bounded PDFX 1.7.2 integration passes 22 focused parser-policy,
  GROBID, and native-manifest tests against the public wheel; 489 backend tests
  pass with 7 skips when the host-only Marker module is excluded; all 189 proxy
  tests pass. A synthetic parser-1.7.1 GROBID manifest is rejected by the 1.7.2
  runtime pin, so no extraction-config version bump is required. Scoped Ruff,
  `py_compile`, and `git diff --check` pass.
- PDFX validation after the first bounded Claude correction round: 188 focused
  tests passed with 6 skips; 487 backend tests passed with 6 skips when the
  host-only Marker module was excluded; 189 proxy tests passed; scoped Ruff
  and `git diff --check` passed.
- The excluded Marker renderer invariant was executed separately against the
  pinned Marker 1.10.2 production GPU/Torch image: the real four-page
  `Document` fixture passed exactly (table; list with link/image; blank page;
  terminal page).
- The first local Sol/max PDFX review accepted after corrections. The first
  Claude round then identified four concrete separation/binding/test gaps;
  only those four were implemented: GROBID coverage separation, real Marker
  dual-render proof, complete deterministic-ownership coverage, and
  caller-authoritative final sidecar digest validation.
- The repeated bounded Claude collaboration found one further Material
  Docling defect against pinned `docling-core==2.87.1`: a primary provenance
  order such as `1,2,1,3` can satisfy the transition-count check while placing
  revisited page-1 bytes in a range labelled page 2. Before merge, PDFX must
  validate the primary native page order with the exact explicit content-layer
  and picture-traversal settings used by the Markdown export, and fail closed
  to the existing residual path on any decrease. This is the only supported
  Material correction from that review; blank-page recovery and generalized
  provenance changes remain out of scope.
- That correction now passes 27 focused Docling/page-provenance tests,
  including a real `docling-core==2.87.1` subprocess fixture; the broader
  backend suite passes 490 tests with 6 existing host-only Marker skips, and
  the proxy suite passes all 189 tests. Scoped Ruff, `py_compile`, and
  `git diff --check` pass.
- The required repeated GPT-5.6 Sol/xhigh `$max-review-skill` gate returned
  `Accept with follow-ups`, with no supported Blocker, Material correction, or
  High-value simplification remaining. Its only gates are commit/push, parser
  publication, and the preserved deployment canaries.
- GitHub's Claude workflow hit its hard ten-minute cancellation twice without
  posting a verdict. A final bounded manual Claude Opus/xhigh review of pushed
  commit `0df7932` independently reran the 27 focused, 490 backend, and 189
  proxy tests and returned `Accept with follow-ups`, with no supported Blocker,
  Material correction, or High-value simplification. Its non-blocking notes
  remain recorded on PR #46 and do not justify another code round under the
  Section 11 stop rules.
- Parser 1.7.0 is published on PyPI after Valerio (`@valearna`) approved the
  change and added PyPI account `ctabone` as an owner. PyPI records the exact
  verified wheel SHA-256
  `287d32db68410c36c5d7d947867adad3d67fae817e949bac1a1270cbed07345b`
  and sdist SHA-256
  `0ffa35f760d1adc7777eb8092f49c7352a05505494f78cd2473bf463fbd20870`.
  A fresh no-cache wheel install from the public Simple Index returned version
  1.7.0 and the PDFX-pinned implementation digest
  `192f912fff47fe79e6a3118a60530cfd00a07944a06c2394ced59fa47e82095c`.
- Parser 1.7.1 is also published and verified from the public PyPI Simple
  Index. Its wheel SHA-256 is
  `cf0e5e9c06dc0aefaeecf80e9402ac8d4ae7cab0df72d670e03a47a97afb890a`,
  its sdist SHA-256 is
  `e14b4f54fa950036cf4c967dbd670e3405ce3f84990f7ae2b2a99c677cecb2f6`,
  and a fresh install produced implementation digest
  `41ce835298863d25a30c733cd245580f3d782eb943470fda096e5389e5914ad2`.
- Parser 1.7.2 is published and verified from its public PyPI wheel. Its wheel
  SHA-256 is
  `eff7ab73b2a181a9b0775c1080e747a38f3984da14696a312dc9ddf30e3765ce`,
  its sdist SHA-256 is
  `7e3363d34a0a389e330beb3c26293c03c4799bf9a41b966b4060ca279257a83f`,
  and the public wheel produced the pinned implementation digest above.
- All three exact Debbie PDFs were recovered from AI Curation production
  storage and durably preserved under
  `s3://agr-pdf-extraction-benchmark/pdfx/canaries/2026-08-23/source/`.
  Their exact page counts are 51, 27, and 25; the earlier 24-page note was
  incorrect. The 1.7.0 deployment exposed the Figure Legends defect; the 1.7.1
  replacement canaries exposed the remaining generated Acknowledgments,
  Funding, and Availability defects. All three were rerun successfully after
  the bounded 1.7.2 PDFX integration, as recorded below.
- PDFX PR #48 merged as
  `94dd55556235079a8b5ddac5ce49c5374678d766` and its complete deployment
  workflow succeeded:
  <https://github.com/alliance-genome/agr_pdf_extraction_service/pull/48> and
  <https://github.com/alliance-genome/agr_pdf_extraction_service/actions/runs/32625401103>.
  The immutable backend image tag is the merge SHA above, with ECR index digest
  `sha256:11348566173518811dfc556b2237859ee277fa7313ef155a7c7cf46320285f14`.
  The matched AMI is `ami-06ade111a5fc13fa4`; the two SSM publication pointers
  were updated atomically to that AMI and image tag. Live inspection verified
  parser distribution/policy version 1.7.2, implementation digest
  `fea04b17244c8a262867852daa2b3a5c922e529f858403caececaf3f671e3bab`,
  NVIDIA L4/CUDA readiness, and the exact immutable container tag.
- The exact parser-1.7.2 production canaries completed successfully:

  | PDF | Process ID | Docling source bytes (`direct` / residual) | Final page methods (bytes / ranges) | Page Luna / fallback | Existing merge Sol usage / cost |
  | --- | --- | ---: | --- | --- | ---: |
  | `8395208_J390188.pdf` (51 pages) | `a057665d-7785-4fcc-9b2c-6b08252c8bac` | 149,651 / 0 | `direct` 176,534 / 218; `deterministic_owner` 616 / 244 | 0 batches / 0 bytes | 187,231 tokens / $1.206545 |
  | `8395484_J390190.pdf` (27 pages) | `7e1e09c2-8c1a-4f56-a70e-ab9f7e8e79f1` | 79,474 / 0 | `direct` 49,153 / 270; `native_start_page` 13,173 / 73; `aligned_agreement` 1,081 / 4; `deterministic_owner` 303 / 250 | 0 batches / 0 bytes | 114,151 tokens / $0.782150 |
  | `8394599_J390144.pdf` (25 pages) | `ff06f11b-7319-45b8-b708-e47356b66b6f` | 0 / 78,400 (`unsafe_docling_page_transition`) | `direct` 45,542 / 413; `native_start_page` 12,810 / 73; `aligned_agreement` 2,969 / 7; `deterministic_owner` 451 / 393 | 0 batches / 0 bytes | 137,802 tokens / $0.842905 |

  Sparse method counters plus `llm_batch_count=0`, empty `llm_outcomes`, and
  `fallback_used=false` prove zero page-selection LLM and fallback ranges/bytes;
  the nonzero costs above are the pre-existing document-merge Sol calls, not
  page-number review.
- Exact heading inspection proves the reported regressions are corrected
  programmatically: the 27-page final Acknowledgments heading is direct page 9;
  the 25-page final Figure Legends, Funding, and Availability headings are
  direct pages 5, 13, and 13. References begin on pages 14, 10, and 13. The
  corresponding GROBID source headings retain their direct/native candidates;
  no heading needed Luna.
- Every current GROBID Markdown artifact is byte-identical to the 1.7.2 parser
  replay. Native mapping covers 99,330/100,756 bytes (98.585%),
  62,384/63,465 bytes (98.297%), and 60,119/61,304 bytes (98.067%). All
  coordinate-bearing `biblStruct` records map to distinct emitted reference
  ranges: 142/142, 47/47, and 61/61.
- Independent validators bound each source map to the exact PDF, native
  artifact, Markdown, range partition, and record digest, and bound each final
  map to the exact merged bytes, audit, merge contract, and source-map digests.
  Official ABC validation/readback succeeds for all three GROBID parser outputs
  and all three final merged outputs. The 51-page final retains only the same
  pre-existing nonfatal S01/S02/S08 warnings as its 1.7.1 output; the other two
  final outputs are validator-clean. Docling and Marker page capture preserves
  their raw pre-feature Markdown bytes exactly, including pre-existing raw
  Marker schema diagnostics; those source audit artifacts are not reclassified
  as final ABC output.
- Authenticated public downloads of `merged` and `page_provenance` returned HTTP
  200 and matched their durable S3 objects byte-for-byte. Merged/sidecar
  SHA-256 pairs are respectively
  `08255ac55ce9dbe731bee742ce944ef83112b24fb670bd80e36a813d282ff915` /
  `78b2110eee386290c87435f232fe98d7255e1a59b21e7727d4c17eea28167b23`,
  `fae1b7dd742a902d353c44339a7c22cbfd65fc1d120d423f125a1a0f908205ec` /
  `89a7d8d9969f14d779dd43222894d1849c7cc4bcd0e24132e962d0aa49f486da`,
  and
  `b94bbdc9e7da1fa0b57460c55bba60a4479ff06d305da3868b941363e50e10fb` /
  `5197d54e6a9bca051300f28988c406c8fa41d15be1f1052f99e5ecddea808858`.
- The deterministic masked holdout covered all three captures, all three
  extractors, and every observed TEI kind in 36 one-range bounded requests.
  Luna/medium made 35 valid choices and all 35 were correct; one abstract case
  returned an invalid structured response and therefore produced no accepted
  model choice. PDFX's tested fail-closed path records and deterministically
  resolves such a missing choice. No wrong LLM page was accepted. The
  content-free evidence digest is
  `2b0206fbdaa67e7b05b6ba0de01fe8dccad4fd578be506bb2f0a0f95cee7bf38`
  and request-set digest is
  `0c7773a2e2be1783b13dd6b55f1cf69439540d5f5887bf9b757acac4816a9dc0`.
  The exact production canaries had zero real residual model selections, so
  the complete real-selection inspection set was empty.
- Three preliminary submissions used a JSON-like `methods` form value instead
  of the API's required comma-separated value and failed at HTTP 400 before
  extraction. They are not canaries and do not indicate a 1.7.2 runtime
  failure. Correctly encoded fresh process IDs are the three recorded above.
  After evidence collection the durable queue and active-job counts were zero,
  the ASG desired capacity was returned to zero, and the GPU instance
  terminated.

## 1. Goal and Non-Negotiable Contract

Produce reliable, one-based primary PDF page numbers for every range of final
PDFX merged Markdown while preserving the exact publication Markdown bytes.
Page information is delivered in a digest-bound JSON sidecar and later
consumed by AI Curation in a separate PR.

The official AGR ABC Markdown schema in
`agr_abc_document_parsers/src/agr_abc_document_parsers/MARKDOWN_SCHEMA.md` is
authoritative. This work must not add inline page comments, page markers,
attributes, or any other new Markdown syntax.

Blocking invariants:

- Existing and provenance-aware TEI conversion produce byte-identical AGR ABC
  Markdown.
- The exact parser output passes the package's `validate_markdown()` and is
  readable by `read_markdown()` with the same semantic document model.
- PDFX merged Markdown remains byte-identical before and after page-sidecar
  generation and continues through its existing exact validator/reader gates.
- Page projection adds no semantic Markdown reread, fuzzy alignment, or
  heuristic publication-role regex.
- Every final page-provenance range has exactly one integer `page_number`
  between 1 and the source PDF page count.

## 2. Evidence and Design Decision

The design is grounded in the preserved Debbie production runs:

- `8395208_J390188.pdf`: 51 pages.
- `8395484_J390190.pdf`: 27 pages.
- `8394599_J390144.pdf`: 25 pages (corrected from the earlier 24-page note).

Read-only replay established:

- Docling's existing full-document serializer can expose all page transitions.
  Removing a transient sentinel reproduced current Markdown byte-for-byte for
  all three captures.
- Marker's pinned Markdown renderer supports paginated output with native page
  IDs in the same render pass.
- GROBID already returns one-based `coords` for requested TEI elements, but the
  current Alliance TEI converter discards them. A one-pass trace prototype
  recovered approximately 97–98% of GROBID Markdown bytes and every formatted
  reference in both Debbie captures.
- The existing merge audit already partitions final output into exact source
  byte spans, so final page mapping can be an interval join rather than a new
  Markdown parse.

Therefore page evidence is captured at each extractor's existing Markdown
emission boundary, bound to the exact source artifact, and translated through
the existing merge audit. PR #42's late structural rescans and inline marker
insertion are not reused.

## 3. Page Semantics and Resolution Order

`page_number` means the page where the represented Markdown range begins.

For native evidence spanning multiple pages, select the first page in native
emission/coordinate order and retain every observed page in `candidate_pages`.

Resolve each final range in this order:

1. Exact single-page extractor evidence.
2. First page of exact multi-page extractor evidence.
3. Static ownership for deterministic markup and formatting bytes.
4. Unanimous page evidence from alternative candidates already aligned by the
   existing merge graph.
5. Identical preceding and following native page anchors.
6. Bounded GPT-5.6 Luna/medium selection for every remaining
   publication-text range.
7. If the model is unavailable, invalid, refuses, or times out, choose the
   highest-ranked candidate page; break ties by nearest byte anchor and then
   lower page number. Use page 1 only if no page evidence exists anywhere.

Evidence tiers are categorical rather than invented numeric confidence:

- `direct`
- `native_start_page`
- `deterministic_owner`
- `aligned_agreement`
- `llm_selected`
- `deterministic_fallback`

The LLM may select only a supplied page choice. It cannot edit Markdown,
invent a page, or override direct native evidence.

## 4. PR 1: Additive TEI Provenance API

Repository: `agr_abc_document_parsers`

Branch: `fix/tei-markdown-page-provenance`

Add this explicitly TEI-specific Python interface:

```python
convert_tei_to_markdown_with_provenance(tei_xml: bytes) -> MarkdownEmission
```

`MarkdownEmission` contains the exact Markdown string plus non-overlapping
UTF-8 byte spans. Each `MarkdownSourceSpan` contains:

- `byte_start`
- `byte_end`
- ordered `page_numbers`
- optional TEI `xml:id` as `native_id`
- semantic `kind`

Implementation requirements:

1. Parse `coords` into ordered, unique, positive page numbers.
2. Carry optional source provenance for TEI-derived title/heading, paragraph,
   figure, table, formula, list, acknowledgment, and reference structures.
3. Instrument the existing emitter to record byte spans while producing the
   same lines. Do not emit hidden trace tokens and do not parse the generated
   Markdown again.
4. Keep `convert_xml_to_markdown()`, JATS conversion, and all existing public
   behavior unchanged.
5. Request additional supported GROBID coordinates for `title`, `affiliation`,
   and `note`, retaining the existing `p`, `head`, `figure`, `biblStruct`,
   `formula`, `ref`, and `persName` requests.
6. Do not enable GROBID sentence segmentation in this change. Primary-page
   semantics do not require character-level splitting inside a cross-page
   paragraph.
7. Release and tag `agr-abc-document-parsers==1.7.2`; PDFX pins that exact
   version. Version 1.7.1 binds generated figure/reference headings to their
   first emitted entry; version 1.7.2 adds native provenance for generated
   Acknowledgments, Funding, and Availability headings.

Parser acceptance criteria:

- [x] Existing conversion APIs return byte-identical Markdown for all current
  fixtures.
- [x] The new TEI API's Markdown equals `convert_xml_to_markdown(...,
  source_format="tei")` exactly.
- [x] New output passes the official ABC `validate_markdown()` contract.
- [x] Reading old and new output with `read_markdown()` produces equal document
  models.
- [x] Provenance spans are in bounds, ordered, and non-overlapping.
- [x] Multi-page coordinate order is preserved.
- [x] Focused fixtures cover title, body paragraphs, headings, tables, figures,
  formulas, lists, acknowledgments, and references.
- [x] Every coordinate-bearing `biblStruct` maps to its emitted reference line.
- [x] Existing pytest, Ruff, formatting, and mypy checks pass.

## 5. PR 2: PDFX Source Page Maps

Repository: `agr_pdf_extraction_service`

Branch: `fix/primary-page-provenance-sidecar`

Add a dedicated `page_provenance` service. Do not extend
`document_skeleton.py` into another page mapper.

### 5.1 Docling

1. Export the full document once using a PDF-digest-derived,
   collision-checked page-break sentinel.
2. Remove the sentinel with an exact linear state machine.
3. Validate transition count and order against the native page inventory.
4. Preserve global serialization; never concatenate per-page exports.
5. If known nested-group or skipped-page behavior makes a boundary unsafe,
   leave the affected source range residual rather than guessing.
6. Pin identical explicit content-layer and picture-traversal settings on the
   Markdown export and its native-order validation walk; do not rely on
   third-party defaults remaining equal.
7. Validate the ordered primary `prov[0].page_no` sequence used by the
   serializer. If it ever decreases, preserve the exact Markdown and leave the
   complete Docling source map residual rather than emitting plausible but
   wrong `direct` page evidence.
8. Do not use secondary coordinates from a multi-page item's `prov` list for
   this order check, and do not raise or drop Docling on an order failure.

### 5.2 Marker

1. Enable the pinned renderer's official paginated Markdown mode in the
   existing render pass.
2. Convert zero-based Marker page IDs to one-based PDF pages.
3. Preserve transient page tokens through existing cleanup, remove them with
   an exact state machine, and calculate offsets after cleanup.
4. Prove that token removal reproduces legacy unpaginated cleaned Markdown
   exactly.

### 5.3 GROBID

1. Use `convert_tei_to_markdown_with_provenance()` from the pinned parser
   package.
2. Resolve multi-page TEI spans to their first coordinate page while retaining
   all candidate pages.
3. Leave emitted ranges without usable coordinates residual for final
   resolution.
4. Obtain the authoritative PDF page count independently so every coordinate
   and final choice can be range-checked.

### 5.4 Source-sidecar contract

Persist one `pdfx-source-page-provenance` record per extractor containing:

- schema and contract version;
- extractor/parser versions and relevant options;
- PDF, native artifact, and exact Markdown SHA-256 digests;
- expected PDF page count;
- ordered UTF-8 byte ranges with `page_number`, `candidate_pages`, method,
  native IDs, or residual reason;
- record SHA-256.

Keep the current `page_coverage` receipt separate. It proves extractor/page
inventory completeness; `page_provenance` maps Markdown bytes to pages.

Write source page maps before the native manifest and bind filename, digest,
size, and media type into that manifest. Bump `EXTRACTION_CONFIG_VERSION` from
6 to 7 so caches without required source maps re-extract.

## 6. Final Merged Page Map

Build the final sidecar solely by intersecting source page ranges with the
existing exact merge audit:

1. Translate every selected source-backed audit interval through that source's
   page map.
2. Assign deterministic transformations through a finite ownership table:
   - heading markers and emphasis delimiters inherit owned content;
   - generated bibliography/figure headings with parser emission provenance
     inherit their first emitted entry directly; PDFX owns any remaining
     unbacked generated heading through the first following entry;
   - reference separators inherit the following reference;
   - terminal newline inherits preceding content.
3. Use existing merge-region candidate spans for alternative-extractor page
   evidence. Do not rerun structural alignment.
4. Coalesce adjacent residual publication bytes sharing the same page choices.
5. Resolve residual publication text through the bounded model path below.
6. Coalesce adjacent final ranges only when page, method, source/operation, and
   evidence identity all match.

The `pdfx-merged-page-provenance` sidecar contains:

- exact PDF, merged Markdown, audit, merge-contract, and source-map digests;
- a contiguous partition of all merged Markdown UTF-8 bytes;
- one valid `page_number` for every range;
- `candidate_pages`, evidence tier, source/operation identity, and evidence
  digest;
- summary byte/range counts by extractor and resolution method;
- record SHA-256.

Persist the page map inside the manifest-last merge bundle. A successful
merged job requires durable upload of both `merged.md` and the final page
sidecar.

## 7. Residual Page LLM

Add a `page_resolution` model-policy role fixed to `gpt-5.6-luna` with medium
reasoning.

For each residual range, gather only existing bounded evidence:

- exact residual text and immediate source context;
- native candidate pages and coordinate summaries;
- page-local excerpts sliced from precomputed source maps;
- already-aligned alternative candidate spans;
- neighboring direct page anchors.

The evidence builder uses interval lookups and exact byte slices. It performs
no semantic Markdown parse, fuzzy match, heuristic role regex, PDF-wide text
search, or new alignment pass.

The model request contains a digest, range IDs, numbered page choices, and
supporting evidence. Its structured response returns the same digest and one
integer choice per range. Persist replayable request/response receipts without
duplicating publication text in metrics.

Batch and evidence-size bounds must be environment-configurable and documented
with their defaults. Process all residual batches while finalization time
remains; any unprocessed or failed selection uses the deterministic fallback
defined in Section 3.

## 8. Public API and Persistence

1. Add `page_provenance` to the download-method enum.
2. Serve
   `GET /api/v1/extract/{process_id}/download/page_provenance` as
   `application/json`.
3. Add `artifacts_json.page_provenance` and expose it through artifact URL
   responses.
4. Upload source page maps alongside native extractor artifacts for audit and
   replay.
5. Verify the complete local merge bundle before serving merged Markdown,
   audit, or page provenance.
6. Bump the merge contract because the committed bundle and required inputs
   change.
7. Keep `merged.md` free of inline page syntax and byte-identical to the
   pre-feature merge.

The AI Curation sidecar consumer, chunk-to-byte-range mapping, and Weaviate
changes are explicitly deferred to a separate goal after PDFX is proven.

## 9. Tests and Release Evidence

### Parser and extractor tests

- [x] Docling's 51-, 27-, and 25-page captures reproduce every expected safe
  transition and the current Markdown SHA exactly.
- [x] A pinned real Docling fixture with primary provenance order `1,2,1,3`
  preserves exact Markdown but produces residual rather than wrong `direct`
  page evidence; a monotonic fixture retains direct ranges.
- [x] Docling Markdown export and order validation use the same explicit
  content layers and picture traversal, with byte identity against the current
  pinned default proven by test.
- [x] Marker fixtures cover tables, lists, blank pages, images/links, and the
  terminal page while preserving current cleaned Markdown exactly.
- [x] GROBID directly maps at least 95% of source Markdown bytes on both Debbie
  captures and maps every coordinate-bearing reference.
- [x] Parser and extractor outputs remain official ABC Markdown.

  Here “remain” is a regression criterion at each existing contract boundary:
  GROBID parser and final merged outputs pass the official validator/reader,
  while the Docling/Marker source audit artifacts remain byte-identical and do
  not acquire new diagnostics. Rewriting pre-existing noncanonical raw Marker
  source would violate the stronger byte-identity requirement and is not part
  of the public final-output contract.

### Contract and mutation tests

- [x] Every range satisfies `0 <= start < end <= markdown_size`.
- [x] Final ranges exactly partition every merged byte without gaps or overlap.
- [x] Every final page is an integer within the PDF page count.
- [x] Cross-page blocks select their starting page and retain all candidates.
- [x] Wrong PDF, Markdown, native, audit, contract, or sidecar digests reject
  reuse.
- [x] Reversed, overlapping, missing, and out-of-bounds ranges are rejected.
- [x] Missing source maps invalidate extractor caches under v7.
- [x] Invalid model digests, missing decisions, duplicate decisions, and
  invented page choices are rejected.
- [x] Missing/failed model calls produce recorded deterministic fallbacks, not
  unnumbered output.

### Performance and architecture tests

- [x] Page provenance adds zero `read_markdown()` calls.
- [x] Page provenance adds zero `validate_markdown()` calls beyond the existing
  authoritative conversion/final-output gates.
- [x] Page provenance adds zero RapidFuzz calls and zero new structural scans.
- [x] Extractor page capture occurs in the existing Markdown emission pass.
- [x] Cache validation uses hashes, schemas, ranges, and receipts rather than
  publication-text reparsing.

### Real evidence

- [x] Build a deterministic masked holdout across all three captures covering
  each extractor and observed structural kind. No wrong LLM page choice is
  allowed before deployment.
- [x] Independently inspect every real residual selection from the three
  captures against native/PDF evidence.
- [x] Canary `8395208_J390188.pdf`, `8395484_J390190.pdf`, and
  `8394599_J390144.pdf` end to end on parser 1.7.2.
- [x] Confirm successful durable `merged` and `page_provenance` downloads.
- [x] Confirm metrics expose direct, LLM, and fallback byte/range counts plus
  LLM usage and cost.

## 10. Avoidance of Over-Engineering

Every checkbox blocks release if violated:

- [x] No inline page comments or Markdown rewrites.
- [x] No new heuristic publication-role regex.
- [x] No per-page Docling export concatenation.
- [x] No final-document structural scan, repeated reader comparison, or
  cross-source fuzzy page vote.
- [x] No general text-edit engine, provenance plugin framework, or second
  Markdown parser.
- [x] No vision/page-image pipeline in this goal.
- [x] No JATS provenance expansion or sentence-segmentation rollout.
- [x] No compatibility layer, migration framework, feature flag, or rollback
  machinery beyond required cache/contract versioning.
- [x] The LLM sees only residual ranges and bounded application-owned choices.
- [x] Existing `page_coverage` qualification behavior remains separate.
- [x] PR #42 and `05687ea` remain evidence only; their rescanning/projection
  implementation is not cherry-picked.
- [x] Every changed production file maps to the parser hook, one extractor
  adapter, the page-sidecar contract, residual page selection, persistence, or
  the public download.
- [x] Tests cover observed and reachable behavior without an exhaustive
  theoretical Cartesian matrix.

## 11. Implementation, PR, and Deployment Order

1. Commit this goal document before implementation work.
2. Implement and validate the parser branch.
3. Run the mandatory local review gate for the parser diff.
4. Push and open the parser PR only after the local reviewer is satisfied.
5. Run bounded Claude review, merge, tag, and publish parser 1.7.2.
6. Implement PDFX against the exact published parser pin.
7. Run focused/full PDFX tests and the exact production-artifact evidence.
8. Run the mandatory local review gate for the PDFX diff.
9. Push and open the PDFX PR only after the local reviewer is satisfied.
10. Run bounded Claude review and required GitHub checks.
11. Merge without waiting for additional external human approval once every
    required gate passes.
12. Deploy PDFX, run the three canaries, and monitor extraction failures,
    sidecar validation, fallback counts, latency, and LLM cost.
    For each canary, explicitly record Docling source direct/residual byte
    counts and final `byte_counts_by_method`; inspect the final pages rather
    than treating healthy-looking method counts as proof of correctness.
13. Close PR #42 as superseded after the replacement is deployed and verified.
14. Create a separate AI Curation goal only after the PDFX artifact contract is
    production-proven.

Claude review framing:

> Review this PR against the goal document, its stated acceptance criteria,
> and the preserved production evidence. Ground requested changes in a
> reachable defect, violated contract, or concrete data-integrity risk.
> Recommend the smallest complete correction. Do not broaden the work into
> inline page syntax, semantic rescanning, generalized provenance
> infrastructure, vision, sentence segmentation, JATS changes, compatibility
> layers, migrations, or speculative edge-case matrices. Record unrelated
> ideas as non-blocking follow-ups.

Implement supported Blockers and Material corrections. Implement a High-value
simplification only when it removes concrete present complexity or risk.
Request another Claude round only after material code changes. Stop when no
supported Blocker or Material correction remains.

## 12. Mandatory Final Goal Review

- [x] At the end of each implementation PR, spawn a **GPT-5.6 Sol sub-agent
  with xhigh reasoning**.
- [x] Its prompt **MUST explicitly invoke `$max-review-skill`** and identify
  this goal document, the final diff, preserved production captures,
  acceptance criteria, and Avoidance of Over-Engineering checklist.
- [x] Require evidence-backed finding labels and the smallest complete
  correction. The reviewer must not invent theoretical edge cases,
  generalized frameworks, or unreachable test combinations.
- [x] Resolve every supported Blocker, Material correction, and High-value
  simplification.
- [x] If material code changes follow, rerun affected tests and repeat the same
  GPT-5.6 Sol/xhigh `$max-review-skill` review.
- [x] Do not proceed to Claude or declare the PR ready until the local verdict
  is `Accept` or `Accept with follow-ups` with no supported Blocker, Material
  correction, or High-value simplification outstanding.
- [x] After the local gate, iterate with Claude only under the bounded rules in
  Section 11. No additional external human approval is required for merge and
  deployment once tests, checks, reviews, and canaries pass.

## 13. Resume Checkpoint

Point a new Codex session at this document and say: **Resume the PDFX primary
page-provenance goal from Section 13.**

Verified state as of 2026-08-23:

- Parser PR #2 is merged and tag `v1.7.0` is pushed:
  <https://github.com/alliance-genome/agr_abc_document_parsers/pull/2>
- Valerio approved the change and added PyPI user `ctabone` as an owner:
  <https://github.com/alliance-genome/agr_abc_document_parsers/pull/2#issuecomment-5382525044>
- Parser PR #3 is merged as
  `37a63876db1bd5345a5664df8e980fd57031cc95`; tag `v1.7.1` is pushed and the
  release is published and verified from the public PyPI Simple Index:
  <https://github.com/alliance-genome/agr_abc_document_parsers/pull/3>
- Parser PR #3 passed 539 tests with 4 skips and 3 deselections, the repeated
  GPT-5.6 Sol/xhigh `$max-review-skill` gate, and a bounded Claude Opus review.
  Both reviews accepted with no supported Blocker, Material correction, or
  High-value simplification. The exact three TEI replays remain byte-identical
  and produce generated figure/reference heading pages 3/14, 4/10, and 5/13.
- Parser PR #4 merged as
  `1875dcabf0ff690ac501c9279f4e3ce210647cfb`; tag `v1.7.2` is pushed and the
  release is published and verified from its public PyPI wheel:
  <https://github.com/alliance-genome/agr_abc_document_parsers/pull/4>
- Parser PR #4 passed 543 tests with 4 skips and 3 deselections. Repeated
  GPT-5.6 Sol/xhigh `$max-review-skill` and bounded Claude Opus reviews accepted
  with no remaining blocker, material correction, or production-code change.
  Exact replay remained byte-identical and independently PDF-ground-truthed
  Acknowledgments page 9 plus Funding/Availability page 13.
- PDFX PR #46 merged as
  `b2d73a45388027a64dbc069b0ec4009d20bc3463` and deployed successfully after
  retrying an AWS `g6.2xlarge` capacity failure in a different availability
  zone. The active backend AMI is `ami-04e8e82f37589faa5`:
  <https://github.com/alliance-genome/agr_pdf_extraction_service/pull/46>
- The exact 1.7.0 production canaries all completed successfully:
  `8395208_J390188.pdf` process
  `0c12d3a5-5f19-4dbf-aa40-24af541210de`, `8395484_J390190.pdf` process
  `6c8f7f78-9755-4bc0-8fb1-acf36b37792e`, and `8394599_J390144.pdf` process
  `5b49999d-62b2-4ce8-94c0-8ab5953b5f20`. Their durable merged/sidecar
  downloads and digest/range bindings passed.
- Inspection of every residual selection found one wrong result in the
  25-page canary: `## Figure Legends` received page 13 from choices `[5, 13]`,
  although its first emitted figure legend is on PDF page 5. Funding and
  Availability were correctly selected as page 13. Parser 1.7.1 is the bounded
  correction, and PDFX integration branch
  `fix/generated-heading-page-provenance-integration` updates its exact pin,
  implementation digest, and native-manifest parser version.
- PDFX PR #47 merged as
  `343e30d58e354071c888a55efc840ca5e0590a02` and deployed successfully on
  backend AMI `ami-0cc9379de43451279` with parser 1.7.1 and implementation
  digest `41ce835298863d25a30c733cd245580f3d782eb943470fda096e5389e5914ad2`:
  <https://github.com/alliance-genome/agr_pdf_extraction_service/pull/47>
- The exact parser-1.7.1 replacement canaries completed as process IDs
  `9f175123-a540-4086-b002-6810323fb00c`,
  `4ebce48a-ec77-415a-82d4-402822ba9ff9`, and
  `09d8c3f0-a5dc-48ce-9869-47c9b7faeb95`. They proved the Figure Legends fix,
  but exposed three remaining generated-back-matter mistakes: the 27-page
  Acknowledgments heading selected page 26 instead of native page 9; the
  25-page Funding heading selected page 12 instead of native page 13; and its
  Availability heading selected page 25 instead of native page 13. Parser
  1.7.2 is the bounded programmatic correction.
- Do not repeat completed parser work or PR #46 review work unless a concrete
  new finding appears.
- PDFX PR #48 merged as
  `94dd55556235079a8b5ddac5ce49c5374678d766`; deployment run `32625401103`
  published immutable image/AMI pair
  `94dd55556235079a8b5ddac5ce49c5374678d766` /
  `ami-06ade111a5fc13fa4`. Parser 1.7.2 and the exact implementation digest were
  verified in the live GPU runtime:
  <https://github.com/alliance-genome/agr_pdf_extraction_service/pull/48>
- The exact parser-1.7.2 replacement canaries completed as process IDs
  `a057665d-7785-4fcc-9b2c-6b08252c8bac`,
  `7e1e09c2-8c1a-4f56-a70e-ab9f7e8e79f1`, and
  `ff06f11b-7319-45b8-b708-e47356b66b6f`. All final/download/digest/range/ABC
  checks passed, no page-resolution LLM or fallback range was needed, and the
  corrected heading pages are 9 plus 5/13/13 as required. The detailed hashes,
  method counts, cost, holdout, and shutdown evidence are in the implementation
  evidence ledger above.
- The production queue is empty and the GPU ASG has desired capacity zero with
  no remaining instance.

Completion state:

1. Close superseded PDFX PR #42 after this evidence-only checkpoint is merged.
2. Do not add the AI Curation/Weaviate consumer here. Start that work only as a
   separate goal using this production-proven sidecar contract.
