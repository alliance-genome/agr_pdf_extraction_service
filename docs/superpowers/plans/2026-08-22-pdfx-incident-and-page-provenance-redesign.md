# PDFX Incident and Page-Provenance Redesign Plan

**Date:** 2026-08-22  
**Status:** PR A #43 and PR B #44 merged, deployed, and production-canary verified; PR C/D gated
**Clean implementation base:** `origin/main` at `9c07feb`
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

**Decision:** this is a purely programmatic fix. Receipt consistency asks
whether recorded hashes, candidates, choices, models, and traces agree; an LLM
must not arbitrate conflicts in its own audit trail. Differences limited to the
three replay-derived diagnostics above are normalized for call grouping. Any
disagreement in the real request or choice fields fails closed. If safe
fallback delivery is considered later, its selection must also be
deterministic—for example, a separately verified baseline extractor output—not
another model call.

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

### PR A: Minimal style-receipt incident fix

Scope:

1. Canonicalize style-selection receipts by excluding exactly the three known
   replay-derived diagnostics from call grouping.
2. Preserve strict validation of the actual request and choice fields.
3. Keep receipt grouping and failure handling entirely programmatic; do not
   introduce an LLM retry, judge, or tie-breaker for receipt consistency.
4. Add exact regressions for the two style-receipt failures.

Constraints:

- No deterministic-transformation or page-provenance behavior.
- No new parser calls.
- No new fuzzy-alignment calls.
- No new LLM calls or prompts.
- No cache-contract bump unless the persisted receipt shape actually changes.
- No audit-operation redesign.

Acceptance criteria:

- [x] Exactly the three named replay-derived diagnostics are excluded from
  call grouping; no other `style_selection_*` field is excluded.
- [x] Persisted style events are not rewritten or normalized.
- [ ] Synthetic regressions matching `8394599` and `8395484` succeed; exact
  cached artifacts succeed as local release evidence.
- [x] Changing a real request digest, candidate or selected identity, response
  choice, model metadata, or matching trace still fails closed.
- [x] Changing a persisted replay diagnostic without its corresponding
  replay-derived value still fails exact event reconciliation.
- [x] Receipt validation invokes no model provider or model-selection
  resolver.
- [x] No parser, fuzzy-alignment, structural-scan, audit-schema, or
  cache-contract behavior changes.

Current implementation evidence:

- [x] A synthetic duplicate-call regression failed on `origin/main` with the
  production error and passes after excluding only the three replay-derived
  diagnostics.
- [x] The exact preserved `8395484` cache (`927dd77e…`) completes merge and
  full `validate_merge_artifacts()` replay: 65 style groups, 50 duplicate
  groups, 11 raw diagnostic-only inconsistencies, and zero canonical receipt
  inconsistencies.
- [x] Exact-cache mutation evidence rejects a changed response choice during
  call grouping and rejects a changed donor ordinal during complete replay.
- [x] `test_merge_artifact.py` plus `test_document_skeleton.py`: 99 passed.
- [x] Broader backend suite in the available sibling virtual environment: 446
  passed and 6 skipped. The Torch-dependent Marker test module and the
  Docling-package-metadata test module were excluded because those optional
  dependencies are not installed in that environment; neither module
  exercises this receipt-grouping change.
- [ ] Re-run the exact `8394599` artifacts. Its former backend cache hash is
  known (`c8081bce…`), but that cache was on the terminated production worker
  and is not in the preserved local capture; recover the cache or obtain the
  source PDF before checking this item.
- [x] Final local GPT-5.6 Sol/xhigh review invoked `$max-review-skill` and
  returned `Accept with follow-ups`, with no Blocker, Material correction, or
  High-value simplification. The reviewer independently reran the 99-test
  focused slice; unavailable `8394599` release evidence was the sole
  non-code follow-up.
- [x] Claude reviewed PR #43 against the bounded contract and reported no
  blockers, confirmed the change is not overengineered, and identified four
  explicitly non-blocking notes. The producer comment and extra unit test are
  optional coverage/documentation ideas; checklist splitting is optional
  presentation; and the existing replay spy remains a defensible explicit
  assertion. None supplies evidence for expanding the patch or requesting a
  second review round.

Deployment evidence:

- [x] PR #43 merged as `9c07febfa92b1b3cdeda3f190f5d7d6f5bcac1e6`.
- [x] The main deployment workflow completed successfully and published that
  exact immutable backend image tag and its baked AMI.
- [x] Production canary `e4f0617a-0422-4b79-97a4-00a1cbe4d0f3` used the
  repository-owned `deploy/aws/ami/test-sample.pdf`, woke the new backend,
  completed all three extractors, and returned a 200 merged download.
- [x] Bootstrap reported `marker_models=ok` and `GPU worker CUDA: HEALTHY
  (NVIDIA L4)`; deep health was healthy with an empty durable queue after the
  job.
- [x] The GPU ASG returned to desired capacity zero with no instances, queued
  work, or active work. The unchanged proxy task was recycled after the
  documented manual ASG scale-down so its in-memory lifecycle state again
  reported `ec2=stopped`.
- [ ] Operational follow-up outside PR A/B: the documented direct ASG
  scale-down leaves a ready proxy's lifecycle label stale until that process
  resynchronizes, and extractor-specific download requests can wake a stopped
  backend. Address that lifecycle/runbook behavior separately; it is not part
  of either provenance failure.

### PR B: Repository-wide deterministic transformation contract

The `8395208` failure is the first acceptance case, not the boundary of this
work. Apply one composition rule across the existing `deterministic_markup`
audit emitters and their shared copy, replacement, permutation,
reconciliation, newline-normalization, and final-validation paths. This does
not include source-selection policy, role inference, model receipts,
event-only decisions, or every post-extraction byte path.

Core invariant:

> No later composition step—replacement, deletion, permutation, or
> normalization—may retain only part of a content-bearing deterministic audit
> atom.

Source-backed spans remain sliceable when their exact source interval is
adjusted and validated. Boundary whitespace is emitted separately with
declared ownership. If a proposed boundary lands inside a content atom, the
composition must move the complete atom according to that ownership or reject
before constructing output. It must never silently clip bytes and assign the
old digest, range, or semantic identity to the fragments.

Scope:

1. Record the bounded inventory of the five current `deterministic_markup`
   emitter sites and map each to its producer, reachable composition path,
   validator, event behavior, and focused test.
2. Extend the existing audit-entry contract with the missing atomicity and
   boundary-ownership policy. Use a small internal typed operation only if the
   inventory proves the current tuples cannot express those creation-time
   facts.
3. If a type is needed, store only operation kind, input range, replacement
   bytes, and boundary ownership. Derive output range, digest, and
   transformation ID; do not persist a validator identity.
4. Represent meaningful generated content and incidental boundary whitespace
   as separate operations at creation time.
5. Make every current composition boundary preserve complete content atoms or
   reject the edit before constructing output. Preserve exact slicing of
   source-backed spans.
6. Keep validator selection static and code-owned by operation kind, using one
   source of truth for the audit-emitting operation shapes.
7. Reconcile the composed audit against the final byte partition, exact
   digests, and operation events without rescanning Markdown semantics.
8. Keep deletion-only events reconciled without creating zero-length final
   audit entries.

Constraints:

- No LLM calls or semantic judgment.
- No regex-based reconstruction of document roles.
- No new `read_markdown()`, fuzzy-alignment, or whole-document scan passes.
- No generic "rehash the clipped fragment" escape hatch.
- No general text-edit engine or operation abstraction beyond what the current
  emitters and composition paths require.
- No reduction of the existing immediate-generation/persistence validation
  duplication; measure and propose that separately if it remains worthwhile.

Acceptance criteria:

- [x] The PR description contains the bounded current-operation inventory and
  maps each item to its producer, reachable composition path, validator, event
  behavior, and focused test.
- [x] Every later composition path rejects partial retention of a
  content-bearing deterministic atom before output construction.
- [x] Exact source-backed subrange slicing remains supported and
  byte-validated.
- [x] In the `8395208` path, optional separator bytes preceding
  `## References` are emitted separately and remain left-owned;
  `## References\n\n` moves intact with the reference container.
- [x] Figure Legends uses the equivalent declared ownership rule.
- [x] Exact `8395208` artifacts complete a no-model merge and final
  `_validate_audit()`.
- [x] A synthetic Figure Legends analogue completes with intact heading
  ownership.
- [x] Clipping `12. ` to `2. ` remains invalid.
- [x] Deletions remain event-reconciled and create no zero-length final audit
  entries.
- [x] Final audit entries exactly partition output and retain valid ranges,
  digests, operation kinds, and event reconciliation.
- [x] Existing valid role-order, heading, table, reference, emphasis, newline,
  and fallback fixtures preserve their visible output and prior success or
  failure behavior, except where an already-emitted deterministic operation
  was rejected solely because final validation omitted its declared shape.
  The reachable `alliance_title_composite_join` path is the one additional
  success correction and now has final-audit regression coverage.
- [x] Existing valid fixtures preserve their Alliance reader model; PR B does
  not change publication-role decisions or output ordering.
- [x] No current scan, parser, fuzzy-alignment, or model-call count increases.
- [x] No persisted contract bump occurs unless separately justified.

Current PR B inventory and evidence:

| Emitter | Producer and reachable composition | Static validator/event | Focused evidence |
|---|---|---|---|
| Final newline atom | `normalize_trailing_newline()` after merge finalization | exact LF; finalization warning, no transformation ID | `test_deterministic_markdown_repair.py` |
| Missing table separator atom | `_render_missing_table_separators()` through skeleton rendering and later role composition | generated GFM separator plus transformation event | merge-service table repair regression |
| Selected-skeleton heading atom | `render_document_skeleton()` through later replacements/permutations | one-to-six `#` bytes plus transformation event | document-skeleton heading regressions |
| Structural edit atom | `_replace_deterministic_markup()` for current copy/replacement/role-order paths | operation-specific static shape plus transformation event | role, reference, Figure Legends, front matter, and table regressions |
| Native emphasis delimiter atom | `project_native_emphasis()` through final audit validation | exact `*` plus projection event | native emphasis projection regressions |

All five emitters now call one `deterministic_audit_entry()` builder. The same
static operation-shape table drives final validation. `_copy_audit_interval()`
permits exact slicing only for source-backed spans and rejects any partial
generated atom before appending bytes; replacement, permutation, role-slot,
and emphasis paths all compose through that copy primitive. Final newline
normalization performs the equivalent atomicity check on its direct rewrite.
No general edit engine, semantic rescanning, role reconstruction, LLM route,
fallback, migration, or page-provenance behavior was added.

The inserted References and Figure Legends helpers now create at most two
ordinary edits at the same insertion point: optional `\n`/`\n\n` with the
operation's declared left ownership, followed by one complete heading atom.
The exact `8395208` output therefore retains one valid left-boundary receipt
and one intact `## References\n\n` receipt instead of splitting one receipt
after composition.

Fresh-`origin/main` measurement and proposed counts (calls/bytes or calls/
compared characters) are:

| Case | Structural scan | `read_markdown` | `validate_markdown` | RapidFuzz ratio | Alignment calls (baseline/alternative units) | Result |
|---|---:|---:|---:|---:|---:|---|
| Small synthetic, before = after | 37 / 4,791 | 45 / 5,832 | 18 / 2,340 | 155 / 4,722 | 8 (44 / 38) | success; identical output SHA-256 |
| Large generated non-sensitive fixture, before = after | 37 / 1,334,048 | 45 / 1,622,511 | 18 / 649,002 | 471,950 / 33,206,502 | 8 (6,626 / 6,620) | success; identical output SHA-256 |
| Exact `8395484`, before = after | 28 / 1,934,387 | 26 / 1,707,760 | 11 / 768,976 | 252,691 / 138,997,858 | 1,346 (160,048 / 4,902) | success; identical output SHA-256 |
| Exact `8395208`, before | 27 / 4,227,063 | 24 / 3,510,580 | 10 / 1,590,325 | 425,234 / 166,583,164 | 747 (251,534 / 3,396) | failed during audit validation |
| Exact `8395208`, after | 29 / 4,581,307 | 26 / 3,823,604 | 11 / 1,767,485 | 802,402 / 329,777,696 | 1,490 (501,182 / 5,613) | success, 177,160 bytes and 928 valid audit entries |

The three already-successful paths have exactly unchanged call and byte/unit
counts, and all runs use `llm=None` (zero provider/model calls). The repaired
`8395208` path reaches two existing post-audit validation phases that
`origin/main` never reached because it terminated at the invalid receipt; the
patch adds no scan, parser, validator, alignment, or provider call site.
Wall-clock observations were 0.034s, 2.818s, 15.314s (failed), and 12.211s for
the four baseline cases, versus 0.034s, 2.810s, 25.913s (completed), and
13.660s after the patch. These are diagnostic observations, not CI limits.

Validation currently recorded:

- [x] Focused deterministic/document/merge/artifact tests: 156 passed, 6
  skipped.
- [x] Broad available backend suite: 452 passed, 6 skipped. Only the
  unavailable Torch Marker and Docling package-metadata modules were excluded;
  neither exercises deterministic composition.
- [x] Existing successful small, large, and exact `8395484` output SHA-256
  values are identical to fresh `origin/main`.
- [x] Exact `8395208` completes with no model and passes final
  `_validate_audit()`.
- [x] Mandatory GPT-5.6 Sol/xhigh review invoked `$max-review-skill` and
  returned `Accept` with no Blocker, Material correction, High-value
  simplification, Optional simplification, or follow-up. The reviewer
  independently reran both test slices, checked old same-contract composite
  receipts, and found the central operation-shape module proportional to the
  five existing emitters rather than a general edit framework.
- [x] Claude's bounded PR #44 review found the core change correct and no
  blocker. Its concrete observation that `alliance_title_composite_join` now
  changes a latent hard failure into success was accepted: the existing title
  composition test now runs final audit validation and the intentional success
  correction is recorded above. Its non-blocking suggestion to replace the
  builder's `ValueError` with `ConsensusContractError` was not implemented:
  every current reachable caller already catches `Exception`, so no present
  failure or acceptance-criterion violation supports widening the code diff.

PR B deployment evidence:

- [x] PR #44 merged as `9324afdc947ee05e0a9304aedb4deb436da6ca7f`.
- [x] Main Build and Deploy run `32552863851` initially failed before
  publication because AWS had no `g6.2xlarge` capacity in `us-east-1c`. One
  failed-job retry succeeded without a code or configuration change; the
  capacity failure was operational and no partial AMI publication occurred.
- [x] SSM now publishes the matched pair `ami-0c68817944e447e45` and immutable
  backend image tag `9324afdc947ee05e0a9304aedb4deb436da6ca7f`.
- [x] Production canary `d04d473c-c8a5-4f65-b8ac-7333a6b40707` used the
  repository-owned `deploy/aws/ami/test-sample.pdf`, cleared cache, completed
  Docling, GROBID, Marker, and the merge, and returned a 200 merged download
  (148 bytes, SHA-256 `516a5581302e80aaa325f5ed0493fc1d391a83e2d10ba48610129048dc3590fe`).
- [x] The canary instance booted from the exact published AMI. Bootstrap
  reported `marker_models=ok` and `GPU worker CUDA: HEALTHY (NVIDIA L4)`;
  deep health reported an empty queue and no active work after completion.
- [x] The GPU ASG returned to desired/count zero. The unchanged
  `pdfx-proxy:56` task was recycled after manual scale-down so final idle
  health reports `ec2=stopped`, queue depth zero, and no active jobs.
- [x] Final scope audit: PR C and PR D were not started. Both remain gated on
  the page-transport contract decisions and prerequisites documented below.

### PR C: Alliance parser transport contract

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
sidecar-only page provenance in PR D.

Acceptance criteria:

- [ ] One parse of marked Markdown produces the same publication model as the
  unmarked input.
- [ ] Illegal placements fail with a specific rule rather than being
  interpreted as publication content.
- [ ] No new heuristic role inference is introduced.
- [ ] Parser performance is measured against representative large PDFX
  Markdown.

### PR D: Page provenance with a declared transport contract

Prerequisite: PR C is released, or a sidecar-only decision is recorded.

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
9. Treat missing, stale, or invalid auxiliary page evidence as unavailable;
   never reuse it, but do not fail an otherwise valid publication merge solely
   because optional page provenance is unavailable.

Acceptance criteria:

- [ ] Zero direct PDFX `read_markdown()` calls are added by page projection.
- [ ] The agreed structural scan budget is enforced by tests.
- [ ] Removing transport tokens yields byte-for-byte publication text.
- [ ] The parser’s one-pass publication model is unchanged by legal tokens.
- [ ] Tables, lists, references, captions, and code blocks are never split at
  an illegal location.
- [ ] Unresolved intervals remain explicit and ordered.
- [ ] A wrong PDF, Markdown, artifact, or contract digest prevents sidecar
  reuse and records a specific provenance failure.
- [ ] Missing or invalid auxiliary page evidence leaves publication text
  unchanged and affected page intervals explicitly unresolved.
- [ ] Existing valid extraction or failsafe delivery is not converted into a
  terminal failure solely because auxiliary page provenance is unavailable.

## 5. Avoidance of Over-Engineering — All Boxes Required

This gate applies during implementation and review. An unchecked item blocks
the affected PR.

- [ ] Every changed production file maps to a named acceptance criterion or
  one of the three documented production incidents.
- [ ] PR A contains no deterministic-operation, page-provenance, parser,
  cache, or merge-policy redesign.
- [ ] PR B is limited to existing `deterministic_markup` emitters and their
  current composition, reconciliation, and validation paths.
- [ ] PR B does not redesign source selection, publication-role inference,
  event-only receipts, parser behavior, or page provenance.
- [ ] The implementation reuses the existing replacement and permutation
  boundaries; it adds no general text-edit engine, planner, DSL, plugin system,
  or dynamic validator mechanism.
- [ ] Validator selection is static and code-owned by operation kind;
  persisted input cannot select executable validation behavior.
- [ ] The persisted audit/event schema and merge contract ID remain unchanged
  unless the implementation proves the current schema cannot express required
  ownership. Any change requires separate justification.
- [ ] No generic clipped-fragment rehash path is added for content-bearing
  deterministic spans.
- [ ] Tests cover shared invariants and reachable current sequences; no
  exhaustive theoretical operation/permutation matrix is added.
- [ ] No feature flag, compatibility branch, migration, fallback engine, or
  rollback machinery is added without a present persisted-schema requirement.
- [ ] Wall-clock measurements remain release evidence rather than flaky CI
  thresholds.
- [ ] Each PR lists and justifies every changed file; PR C/D files do not
  appear in PR A/B.

## 6. Proposed Data Flow

```text
extractor Markdown + native sidecars + exact PDF digest
                         |
                         v
existing merge structural analysis (computed once, then reused)
                         |
                         +--> publication merge
                         |        |
                         |        +--> deterministic audit operations
                         |                 |
                         |                 +--> explicit ownership policy
                         |                 +--> atom-preserving composition
                         |                 +--> final byte/audit reconciliation
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

## 7. Measurement Plan

PR A needs focused test spies proving the receipt-grouping change adds no
parser, fuzzy-alignment, structural-scan, or provider calls. It does not require
a new multi-artifact benchmark harness.

Before coding PR B or PR D, record one reusable `origin/main` baseline with a
temporary benchmark harness or test spy that records:

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

Record baseline and proposed counts in the relevant PR description. A faster
wall-clock result does not excuse an unbounded increase in scans. Do not make
wall-clock measurements blocking CI thresholds.

## 8. Test Strategy

### Incident fixtures

Keep raw production PDFs and extracted publication text out of Git. Build
content-free or synthetic regressions that preserve the failing audit/receipt
shape. Run the exact cached artifacts locally as release evidence.

### Required mutation tests

- Change a real style response choice: reject.
- Change only a replay ordinal diagnostic: canonical call grouping remains
  consistent, while exact event replay still detects unauthorized mutation.
- Clip a numbered reference marker: reject.
- Try every byte boundary of one representative content atom: move it whole or
  reject before output construction.
- Prove declared ownership for representative boundary whitespace.
- Prove exact subrange slicing remains valid for a representative source-backed
  span.
- Exercise one reachable References/Figure Legends composition through final
  bytes, audit partition, digests, and event reconciliation.
- Do not add a Cartesian suite for operation/permutation pairs that cannot
  occur in the current pipeline.
- Change the PDF digest under a native sidecar: reject/re-extract.
- Reverse or overlap page intervals: reject.
- Introduce a contradictory page vote: record unresolved, do not guess.

### Parser-contract tests

For every legal token position, compare the complete `Document` model before
and after insertion. For every illegal position, assert a named validation
rule. Do this once in the parser repository, not repeatedly at PDFX runtime.

## 9. Cache and Migration Policy

- PR A and PR B should avoid invalidating extractor caches unless PR B changes
  a persisted merge-receipt schema.
- A changed persisted merge-receipt schema requires a new merge contract ID.
- A parser version/digest change requires regeneration only for artifacts whose
  replay semantics depend on that parser.
- Page sidecars must include their own contract and PDF identity so they can be
  invalidated independently of expensive extractor output.
- Do not silently accept old inline page-marker bundles under a new parser
  contract.

## 10. Rollout Plan

1. Approve PR A's own checklist, merge and deploy it independently, then rerun
   `8394599` and `8395484`.
2. Record PR B's bounded emitter/composition inventory and structural baseline,
   implement only that scope, and run exact `8395208` as release evidence.
3. Deploy PR B and observe merge timing and audit failures before introducing
   page changes.
4. Complete and release PR C, or explicitly choose sidecar-only provenance.
5. Implement PR D against the declared contract and scan budget.
6. Canary page provenance on cached/non-sensitive PDFs.
7. Deploy without interrupting active or queued production jobs.
8. Monitor merge duration, cache replay failures, audit rejection reasons, and
   GPU idle behavior.

## 11. Disposition of PR #42

PR #42 remains valuable as:

- an incident notebook;
- a source of exact regression ideas;
- evidence of unsafe Markdown placements;
- proof that both production failures can be reproduced locally.

It is not the implementation base for this redesign. Do not cherry-pick its
page-projection guard, repeated reader comparison, or full projection replay.
Individual minimal changes may be reimplemented only after they are justified
against this plan and the scan budget.

## 12. Decisions Required Before Implementation

1. Are inline page comments an official Alliance Markdown transport feature?
2. If yes, which repository owns their syntax and legal placement?
3. If no, which consumer reads the page-provenance sidecar?
4. Is full computational replay required for merge bundles, or are exact
   identities, hashes, monotonic ranges, and audit reconciliation sufficient?
5. What scan-count baseline and maximum are acceptable for the largest PDFs?
6. Should PR #42 be closed immediately as superseded, or retained as a draft
   reference until PR A lands?

The atom-preserving deterministic-operation contract is decided by this plan;
it is not limited to References and Figure Legends. PR A can begin once its own
checklist is approved. PR B additionally requires its bounded inventory and
scan baseline. Only PR C and PR D wait for the page-transport decision. Any
proposal to reduce duplicate generation/persistence validation is a separate
workstream and does not hitchhike into these incident fixes.

## 13. Mandatory Final Review and Proportional PR Iteration

This is the final completion gate for every implementation PR produced from
this plan. It applies after implementation and focused validation, before the
PR is presented as ready for human review.

### 13.1 Local final-review gate

- [x] Run the complete acceptance checklist for the applicable PR and record
  the results.
- [x] Request a read-only review from a **GPT-5.6 Sol sub-agent with xhigh
  reasoning** against the final branch diff and surrounding code.
- [x] The review request **MUST explicitly invoke `$max-review-skill`** and
  identify this plan, the applicable PR acceptance criteria, the production
  failure, and behavior that must remain unchanged.
- [x] Require the skill's evidence rule and finding labels. The reviewer must
  not invent theoretical edge cases, generalized frameworks, or tests for
  unreachable combinations.
- [x] Resolve every supported Blocker, Material correction, and High-value
  simplification with the smallest complete change.
- [x] Optional simplifications and unrelated observations are recorded but do
  not expand the implementation PR.
- [x] If material code changes after that review, rerun the affected focused
  tests and one final Max review of the resulting diff.
- [x] Do not open or mark the PR ready until the final verdict is `Accept` or
  `Accept with follow-ups` with no unresolved Blocker, Material correction, or
  High-value simplification.

### 13.2 Opening the PR and requesting Claude review

- [x] Open the PR only after the local final-review gate passes.
- [x] Include the applicable acceptance checklist, production evidence, test
  results, scan/call-count evidence, changed-file justification, and explicit
  exclusions in the PR description.
- [x] Ask Claude to review correctness, behavior preservation, integration,
  and proportionality against this bounded contract.
- [x] Tell Claude explicitly that review comments are not a request to broaden
  the work into page provenance, parser changes, merge-policy redesign, a
  general text-edit framework, exhaustive edge-case matrices, fallbacks, or
  compatibility machinery unless a concrete reachable defect requires it.

Use this framing in the initial Claude request:

> Review this PR against its stated acceptance criteria and the documented
> production failure. Preserve existing behavior outside that contract.
> Ground each requested change in a reachable failing path, violated acceptance
> criterion, existing supported contract, or concrete data-integrity risk.
> Recommend the smallest complete correction. Do not request speculative edge
> cases, exhaustive test combinations, generalized editing infrastructure,
> page-provenance work, parser redesign, fallbacks, migrations, or compatibility
> layers unless the current diff makes one demonstrably necessary. Mark useful
> unrelated ideas as non-blocking follow-ups.

### 13.3 Claude iteration stop rules

- [x] Classify each Claude finding using the same evidence standard as
  `$max-review-skill`; do not implement a suggestion merely because it was
  suggested.
- [x] Implement supported Blockers and Material corrections. Implement a
  High-value simplification only when it removes concrete present complexity
  or risk introduced by this PR.
- [x] Do not implement Optional simplifications or outside-scope observations
  in the incident PR; record them separately if they remain useful.
- [x] Request another Claude review only after a material code change that
  could affect its earlier conclusion. Do not request ceremonial additional
  rounds after documentation-only replies or disposition of unsupported
  comments.
- [x] Before every additional round, compare the diff with the accepted scope
  and the Avoidance of Over-Engineering gate. Remove or narrow unsupported
  growth before asking again.
- [x] Stop iterating when no supported Blocker or Material correction remains.
  Zero comments, theoretical completeness, and reviewer exhaustion are not
  completion criteria.

### 13.4 Merge authorization

After the mandatory GPT-5.6 Sol/xhigh Max-review gate and bounded Claude review
both finish with no supported blocker or material correction, no additional
external human approval is required. The operator explicitly authorized the
agent on 2026-08-22 to use the repository-ruleset bypass to merge and deploy
these implementation PRs once their required tests, evidence, checks, and
review gates pass. This authorization does not waive any acceptance criterion,
validation, deployment health check, or review stop rule above.

If Claude begins expanding scope or producing variants of already-dispositioned
theoretical concerns, reply with this boundary rather than implementing them:

> Thanks. We are deliberately holding this PR to the documented production
> failure and accepted checklist. Please identify the concrete reachable path,
> violated acceptance criterion, existing behavior regression, or
> data-integrity risk that makes this change necessary now. If none applies, we
> will record it as a non-blocking follow-up rather than expand this PR.

If a Claude finding does identify a new reachable correctness or data-integrity
failure, pause the review loop, add that evidence to the plan and acceptance
criteria, obtain the smallest proportional correction, and rerun the local
GPT-5.6 Sol/xhigh Max-review gate before returning to Claude.
