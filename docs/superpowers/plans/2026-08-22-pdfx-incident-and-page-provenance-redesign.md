# PDFX Incident and Page-Provenance Redesign Plan

**Date:** 2026-08-22  
**Status:** Planning; no implementation approved  
**Clean implementation base:** `origin/main` at `673ab52`  
**Reference implementation only:** PR #42 / `fix/style-selection-receipt-consistency`

## 1. Why We Are Restarting

PR #42 reached a functionally correct state and reproduced both reported
production failures, but its page-marker hardening moved PDFX in the wrong
architectural direction:

- it added repeated whole-document reads during page projection;
- bundle replay repeated those reads again;
- the reads invoke a line-oriented, regex-heavy semantic parser;
- PDFX became responsible for discovering where another package interprets a
  transport comment as publication data;
- correctness checks accumulated faster than the underlying transport contract
  was clarified.

This restart preserves the production evidence and learned failure modes, but
does not reuse PR #42 as an implementation base. Every change must be
re-derived from `origin/main` against the constraints in this document.

## 2. Production Evidence We Must Preserve

### 2.1 Positive-style receipt failures

- `8394599_J390144.pdf`
  - process `36734948-ce4d-455e-a2aa-944ab1ca76d1`
  - `positive style model-selection receipt is inconsistent`
- `8395484_J390190.pdf`
  - process `ebdb9353-8046-49ce-b914-d22e337742bf`
  - `positive style model-selection receipt is inconsistent`

Observed root cause: claims sharing one model-selection call can later carry
different replay-derived values for:

- `style_selection_donor_ordinal`
- `style_selection_target_ordinal`
- `style_selection_order_crossing`

Those fields are not part of the model request or choice. The real receipt
fields—request digest, candidates, response choice, selected candidate, model,
reasoning effort, and trace—must remain strictly validated.

### 2.2 Deterministic-provenance failure

- `8395208_J390188.pdf`
  - process `bf645eca-f163-4bfb-96de-ce8bbf62ca2d`
  - `deterministic markup provenance is invalid`

Observed root cause: a generated `## References` insertion included leading
blank-line bytes in one deterministic audit span. A later role permutation
split that span at the heading boundary. The fragments retained a digest for
the complete insertion and failed final audit validation.

The correction must make deterministic operations composable without adding a
generic “rehash any clipped generated span” escape hatch. Content-bearing
markers, including numbered reference markers, must remain fail-closed.

### 2.3 Page provenance failures and hazards

The prepared page-provenance work in commit `05687ea` established useful source
evidence, but review found these hazards:

- stale native sidecars can be reused with a different PDF;
- equal or conflicting cross-source page votes can be misleading;
- inserting comments inside tables, lists, references, or nested list content
  changes Markdown structure;
- inserting a page comment before an H1 can cause the Alliance reader to lose
  the title and body;
- inserting a page comment immediately after the H1 can be interpreted as an
  author;
- a skipped inline marker cannot, by itself, represent an “unassigned” page
  span;
- replay must not depend silently on a different parser or fuzzy-matching
  implementation.

These are design inputs, not a mandate to reproduce PR #42’s defenses.

## 3. Architectural Constraints

### 3.1 Whole-document scan budget

Before implementation, measure and document the current `origin/main` scan
count for one merge and one cache validation. Then enforce:

- The incident-fix PR adds **zero** whole-document scans.
- Page-provenance generation adds **zero direct `read_markdown()` calls** in
  PDFX.
- Page-provenance generation should reuse structural units already computed by
  the merge pipeline. If reuse is impossible, it may add at most one linear
  structural pass, with a benchmark and explicit justification.
- Immediate persistence validation must not rerun the entire semantic reader,
  fuzzy alignment, or structural scan merely to reproduce a result generated
  moments earlier.
- Cache loading validates identities, hashes, ranges, ordering, and receipt
  schema without reparsing publication Markdown by default.
- Any optional deep replay mode must be explicit, diagnostic-only, and measured.

Tests should instrument the relevant functions and fail if the agreed scan
budget is exceeded.

### 3.2 Regex policy

- Do not add heuristic publication-role regexes to PDFX.
- Exact transport-token recognition may use one anchored, linear expression or
  an equivalent string parser.
- Document roles remain owned by the authoritative Alliance parser.
- PDFX structural code should consume typed units or parser output rather than
  independently guessing authors, headings, references, tables, or lists.

### 3.3 Ownership boundaries

- PDFX owns extraction provenance, byte ranges, source identities, and merge
  audit receipts.
- `agr-abc-document-parsers` owns the semantics of Alliance Markdown.
- If inline page comments are part of Alliance Markdown transport, the parser
  must explicitly define and test their semantics.
- A PDFX workaround must not reverse-engineer the parser by comparing repeated
  before/after parses.

### 3.4 Fail-closed behavior

- A real model choice mutation must fail validation.
- A content-bearing deterministic span with an inconsistent digest or range
  must fail validation.
- A native sidecar with the wrong PDF digest must not be reused.
- Conflicting page evidence must remain unresolved.
- “Unassigned” page provenance must be represented explicitly in metadata; it
  must not be implied through the absence of an inline marker.

## 4. Proposed Workstreams and PR Boundaries

Do not combine these into one large PR.

### PR A: Minimal PDFX incident fixes

Scope:

1. Canonicalize style-selection receipts by excluding exactly the three known
   replay-derived diagnostics from call grouping.
2. Preserve strict validation of the actual request and choice fields.
3. Make generated bibliography/figure heading text and any required leading
   whitespace separate deterministic operations so later interval permutation
   cannot split one content-bearing digest.
4. Define narrow validators for each generated operation.
5. Add exact regressions for both Debbie failure classes.

Constraints:

- No page-provenance behavior.
- No new parser calls.
- No new fuzzy-alignment calls.
- No cache-contract bump unless the persisted receipt shape actually changes.
- No generic deterministic-span rebinding.

Acceptance criteria:

- Exact `8395208` artifacts complete a no-model merge and final audit
  validation.
- Exact `8395484` artifacts replay duplicate style decisions successfully.
- Mutating the real style response choice fails closed.
- Clipping `12. ` to `2. ` remains invalid.
- Scan-count tests show no increase from `origin/main`.

### PR B: Alliance parser transport contract

Repository: the repository that publishes `agr-abc-document-parsers`.

Scope:

1. Decide whether `<!-- page: N -->` is an official transport token.
2. If yes, make `read_markdown()` explicitly recognize and ignore it in every
   permitted position without changing the surrounding `Document` model.
3. Make `validate_markdown()` either accept the token or report an explicit
   placement error.
4. Define where the token is legal relative to:
   - the H1/title and front matter;
   - ordinary paragraphs and headings;
   - tables;
   - ordered and unordered lists, including lazy continuations;
   - references;
   - fenced code blocks and captions.
5. Add parser tests proving that inserting legal page tokens does not change
   title, authors, sections, paragraphs, lists, tables, figures, or references.
6. Publish a new pinned parser version and implementation digest.

If the parser maintainers do not want inline page tokens, stop here and use
sidecar-only page provenance in PR C.

Acceptance criteria:

- One parse of marked Markdown produces the same publication model as the
  unmarked input.
- Illegal placements fail with a specific rule rather than being interpreted
  as publication content.
- No new heuristic role inference is introduced.
- Parser performance is measured against representative large PDFX Markdown.

### PR C: Page provenance with a declared transport contract

Prerequisite: PR B is released, or a sidecar-only decision is recorded.

Scope:

1. Restore native page evidence from `05687ea` selectively, re-deriving each
   change against current `main`.
2. Bind every native sidecar and page-coverage record to:
   - extractor artifact digest;
   - exact PDF SHA-256;
   - schema/contract version.
3. Reuse structural units already computed during merge.
4. Resolve pages from direct source/audit overlap first.
5. Treat ties, missing evidence, and cross-source contradictions as explicit
   unresolved intervals in a sidecar receipt.
6. Emit inline page tokens only where the parser contract explicitly permits
   them.
7. Keep unresolved page boundaries in metadata even when no inline token is
   emitted.
8. Validate the receipt using hashes, monotonic ranges, identities, and token
   audit spans—not repeated full-document semantic parsing.

Acceptance criteria:

- Zero direct PDFX `read_markdown()` calls are added by page projection.
- The agreed structural scan budget is enforced by tests.
- Removing transport tokens yields byte-for-byte publication text.
- The parser’s one-pass publication model is unchanged by legal tokens.
- Tables, lists, references, captions, and code blocks are never split at an
  illegal location.
- Unresolved intervals remain explicit and ordered.
- Stale sidecars and dependency/contract mismatches fail with clear errors.

## 5. Proposed Data Flow

```text
extractor Markdown + native sidecars + exact PDF digest
                         |
                         v
existing merge structural analysis (computed once, then reused)
                         |
                         +--> publication merge + deterministic audit
                         |
                         +--> page evidence join
                                  |
                                  +--> resolved legal boundaries
                                  +--> explicit unresolved intervals
                         |
                         v
optional parser-supported transport-token insertion
                         |
                         v
existing final Alliance validation (single authoritative path)
                         |
                         v
hash/range/schema validation of persisted receipts
```

The page-evidence join must not call the semantic reader to discover whether a
token is safe. Safety comes from the parser contract and typed placement data.

## 6. Measurement Plan

Before coding PR A or PR C, add a temporary benchmark harness or test spy that
records:

- `scan_structural_units()` calls and total input bytes scanned;
- `read_markdown()` calls and total input bytes parsed;
- `validate_markdown()` calls and total input bytes validated;
- RapidFuzz alignment calls and compared unit counts;
- wall-clock merge and bundle-validation time.

Measure at least:

- a small synthetic article;
- exact `8395208` cached artifacts;
- exact `8395484` cached artifacts;
- the largest approved non-sensitive cached article available locally.

Record baseline and proposed counts in each PR description. A faster wall-clock
result does not excuse an unbounded increase in scans.

## 7. Test Strategy

### Incident fixtures

Keep raw production PDFs and extracted publication text out of Git. Build
content-free or synthetic regressions that preserve the failing audit/receipt
shape. Run the exact cached artifacts locally as release evidence.

### Required mutation tests

- Change a real style response choice: reject.
- Change only a replay ordinal diagnostic: canonical call grouping remains
  consistent, while exact event replay still detects unauthorized mutation.
- Clip a numbered reference marker: reject.
- Split or permute an atomic generated heading: retain valid audit ownership or
  reject before delivery.
- Change the PDF digest under a native sidecar: reject/re-extract.
- Reverse or overlap page intervals: reject.
- Introduce a contradictory page vote: record unresolved, do not guess.

### Parser-contract tests

For every legal token position, compare the complete `Document` model before
and after insertion. For every illegal position, assert a named validation
rule. Do this once in the parser repository, not repeatedly at PDFX runtime.

## 8. Cache and Migration Policy

- PR A should avoid invalidating extractor caches.
- A changed persisted merge-receipt schema requires a new merge contract ID.
- A parser version/digest change requires regeneration only for artifacts whose
  replay semantics depend on that parser.
- Page sidecars must include their own contract and PDF identity so they can be
  invalidated independently of expensive extractor output.
- Do not silently accept old inline page-marker bundles under a new parser
  contract.

## 9. Rollout Plan

1. Merge and deploy PR A independently.
2. Re-run the three Debbie PDFs and confirm the incident errors are gone.
3. Observe merge timing and audit failures before introducing page changes.
4. Complete and release PR B, or explicitly choose sidecar-only provenance.
5. Implement PR C against the declared contract and scan budget.
6. Canary page provenance on cached/non-sensitive PDFs.
7. Deploy without interrupting active or queued production jobs.
8. Monitor merge duration, cache replay failures, audit rejection reasons, and
   GPU idle behavior.

## 10. Disposition of PR #42

PR #42 remains valuable as:

- an incident notebook;
- a source of exact regression ideas;
- evidence of unsafe Markdown placements;
- proof that both production failures can be reproduced locally.

It is not the implementation base for this redesign. Do not cherry-pick its
page-projection guard, repeated reader comparison, or full projection replay.
Individual minimal changes may be reimplemented only after they are justified
against this plan and the scan budget.

## 11. Decisions Required Before Implementation

1. Are inline page comments an official Alliance Markdown transport feature?
2. If yes, which repository owns their syntax and legal placement?
3. If no, which consumer reads the page-provenance sidecar?
4. Is full computational replay required for merge bundles, or are exact
   identities, hashes, monotonic ranges, and audit reconciliation sufficient?
5. What scan-count baseline and maximum are acceptable for the largest PDFs?
6. Should PR #42 be closed immediately as superseded, or retained as a draft
   reference until PR A lands?

No implementation begins until these decisions and the scan budget are
reviewed.
