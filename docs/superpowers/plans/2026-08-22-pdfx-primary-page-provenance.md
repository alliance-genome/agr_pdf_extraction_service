# PDFX Primary Page-Provenance Goal

**Date:** 2026-08-22

**Status:** Implementation and code review complete; release gates remain

**PDFX implementation base:** `origin/main` at `9747452`

**Parser implementation base:** `agr_abc_document_parsers` `origin/main` at `a37100e` / `v1.6.0`

**Reference evidence only:** PDFX PR #42 and commit `05687ea`

**Official goal:** Active for the complete parser/PDFX/review/release/deploy
sequence in this document. This document remains the scope and acceptance
authority for that goal.

### Implementation evidence ledger

- Parser PR #2 merged as `7efc257bb858449fab9e4d96f17cfa031a9402cb`;
  tag `v1.7.0` is pushed and PDFX pins `agr-abc-document-parsers==1.7.0`.
- Parser source digest is pinned by PDFX as
  `192f912fff47fe79e6a3118a60530cfd00a07944a06c2394ced59fa47e82095c`.
- Parser validation: 537 passed, 4 skipped, 3 deselected; required Sol/max and
  bounded Claude reviews accepted the parser change.
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
- Parser publication remains operationally blocked: PyPI currently resolves
  only 1.6.0 and no publisher credential is materialized on this host. A clean
  PDFX build/deploy cannot proceed until 1.7.0 is published.
- Exact Debbie PDFs are no longer recoverable from the terminated worker
  volumes or known durable stores. Their three exact MD5 identities are
  retained for the required post-deployment canary gate; no local replay claim
  may substitute for that gate.

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
- An additional production-style capture: 24 pages.

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
7. Release and tag `agr-abc-document-parsers==1.7.0`; PDFX pins that exact
   version.

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
   - generated bibliography/figure headings inherit the first following entry;
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

- [ ] Docling's 51-, 27-, and 24-page captures reproduce every expected safe
  transition and the current Markdown SHA exactly.
- [x] A pinned real Docling fixture with primary provenance order `1,2,1,3`
  preserves exact Markdown but produces residual rather than wrong `direct`
  page evidence; a monotonic fixture retains direct ranges.
- [x] Docling Markdown export and order validation use the same explicit
  content layers and picture traversal, with byte identity against the current
  pinned default proven by test.
- [x] Marker fixtures cover tables, lists, blank pages, images/links, and the
  terminal page while preserving current cleaned Markdown exactly.
- [ ] GROBID directly maps at least 95% of source Markdown bytes on both Debbie
  captures and maps every coordinate-bearing reference.
- [ ] Parser and extractor outputs remain official ABC Markdown.

### Contract and mutation tests

- [ ] Every range satisfies `0 <= start < end <= markdown_size`.
- [ ] Final ranges exactly partition every merged byte without gaps or overlap.
- [ ] Every final page is an integer within the PDF page count.
- [ ] Cross-page blocks select their starting page and retain all candidates.
- [ ] Wrong PDF, Markdown, native, audit, contract, or sidecar digests reject
  reuse.
- [ ] Reversed, overlapping, missing, and out-of-bounds ranges are rejected.
- [ ] Missing source maps invalidate extractor caches under v7.
- [ ] Invalid model digests, missing decisions, duplicate decisions, and
  invented page choices are rejected.
- [ ] Missing/failed model calls produce recorded deterministic fallbacks, not
  unnumbered output.

### Performance and architecture tests

- [ ] Page provenance adds zero `read_markdown()` calls.
- [ ] Page provenance adds zero `validate_markdown()` calls beyond the existing
  authoritative conversion/final-output gates.
- [ ] Page provenance adds zero RapidFuzz calls and zero new structural scans.
- [ ] Extractor page capture occurs in the existing Markdown emission pass.
- [ ] Cache validation uses hashes, schemas, ranges, and receipts rather than
  publication-text reparsing.

### Real evidence

- [ ] Build a deterministic masked holdout across all three captures covering
  each extractor and observed structural kind. No wrong LLM page choice is
  allowed before deployment.
- [ ] Independently inspect every real residual selection from the three
  captures against native/PDF evidence.
- [ ] Canary `8395208_J390188.pdf`, `8395484_J390190.pdf`, and the 24-page
  capture end to end.
- [ ] Confirm successful durable `merged` and `page_provenance` downloads.
- [ ] Confirm metrics expose direct, LLM, and fallback byte/range counts plus
  LLM usage and cost.

## 10. Avoidance of Over-Engineering

Every checkbox blocks release if violated:

- [ ] No inline page comments or Markdown rewrites.
- [ ] No new heuristic publication-role regex.
- [ ] No per-page Docling export concatenation.
- [ ] No final-document structural scan, repeated reader comparison, or
  cross-source fuzzy page vote.
- [ ] No general text-edit engine, provenance plugin framework, or second
  Markdown parser.
- [ ] No vision/page-image pipeline in this goal.
- [ ] No JATS provenance expansion or sentence-segmentation rollout.
- [ ] No compatibility layer, migration framework, feature flag, or rollback
  machinery beyond required cache/contract versioning.
- [ ] The LLM sees only residual ranges and bounded application-owned choices.
- [ ] Existing `page_coverage` qualification behavior remains separate.
- [ ] PR #42 and `05687ea` remain evidence only; their rescanning/projection
  implementation is not cherry-picked.
- [ ] Every changed production file maps to the parser hook, one extractor
  adapter, the page-sidecar contract, residual page selection, persistence, or
  the public download.
- [ ] Tests cover observed and reachable behavior without an exhaustive
  theoretical Cartesian matrix.

## 11. Implementation, PR, and Deployment Order

1. Commit this goal document before implementation work.
2. Implement and validate the parser branch.
3. Run the mandatory local review gate for the parser diff.
4. Push and open the parser PR only after the local reviewer is satisfied.
5. Run bounded Claude review, merge, tag, and publish parser 1.7.0.
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
