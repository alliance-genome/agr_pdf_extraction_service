"""Immutable source spans and selection-only model contracts."""

from __future__ import annotations

import hashlib
import heapq
import unicodedata
from dataclasses import dataclass
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


SourceName = Literal["grobid", "docling", "marker"]
CandidateType = Literal[
    "prose",
    "whitespace",
    "markdown",
    "heading",
    "citation",
    "equation",
    "table_cell",
    "table_row",
    "figure_caption",
]
SelectionAction = Literal["select_candidates", "keep_baseline"]
MergeQuality = Literal[
    "consensus",
    "terra_selected",
    "sol_selected",
    "baseline_fallback",
]

_SOURCE_NAMES = frozenset({"grobid", "docling", "marker"})
_CANDIDATE_TYPES = frozenset(
    {
        "prose",
        "whitespace",
        "markdown",
        "heading",
        "citation",
        "equation",
        "table_cell",
        "table_row",
        "figure_caption",
    }
)
_ALLOWED_C0 = frozenset({"\t", "\n", "\r"})
_RISKY_BIDI_FORMAT_CONTROLS = frozenset(
    {
        "\u061c",  # Arabic letter mark
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u202a",  # embeddings/overrides and pop directional formatting
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",  # directional isolates and pop directional isolate
        "\u2067",
        "\u2068",
        "\u2069",
    }
)
_MAX_DIAGNOSTIC_CODEPOINTS = 8


class ConsensusContractError(ValueError):
    """Raised when source provenance or a selection decision is invalid."""


def forbidden_control_characters(text: str) -> tuple[str, ...]:
    """Return forbidden Unicode control characters in document order.

    Horizontal tab, line feed, and carriage return are the only ``Cc``
    characters permitted in scientific Markdown.
    """

    return tuple(
        char
        for char in text
        if unicodedata.category(char) == "Cc" and char not in _ALLOWED_C0
    )


def unsafe_unicode_characters(text: str) -> tuple[str, ...]:
    """Return controls, risky bidi formatting, and Unicode noncharacters."""

    unsafe = []
    for char in text:
        codepoint = ord(char)
        is_noncharacter = (
            0xFDD0 <= codepoint <= 0xFDEF
            or codepoint & 0xFFFF in {0xFFFE, 0xFFFF}
        )
        if (
            (unicodedata.category(char) == "Cc" and char not in _ALLOWED_C0)
            or unicodedata.category(char) == "Cf"
            or char in _RISKY_BIDI_FORMAT_CONTROLS
            or is_noncharacter
        ):
            unsafe.append(char)
    return tuple(unsafe)


def require_publishable_text(text: str, *, candidate_id: str) -> None:
    """Reject source text that cannot safely enter a completed artifact."""

    unsafe = unsafe_unicode_characters(text)
    if unsafe:
        samples = tuple(dict.fromkeys(unsafe))[:_MAX_DIAGNOSTIC_CODEPOINTS]
        codepoints = ", ".join(f"U+{ord(char):04X}" for char in samples)
        suffix = "" if len(unsafe) <= len(samples) else f" (+{len(unsafe) - len(samples)} more)"
        raise ConsensusContractError(
            f"candidate {candidate_id!r} contains {len(unsafe)} unsafe Unicode "
            f"character(s): {codepoints}{suffix}"
        )


@dataclass(frozen=True)
class SourceArtifact:
    """Immutable UTF-8 bytes from one extractor plus their verified digest."""

    source: SourceName
    raw_utf8: bytes
    digest: str

    def __post_init__(self) -> None:
        if self.source not in _SOURCE_NAMES:
            raise ConsensusContractError(f"unknown extractor source: {self.source!r}")
        if not isinstance(self.raw_utf8, bytes):
            raise ConsensusContractError("raw_utf8 must be bytes")
        try:
            self.raw_utf8.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ConsensusContractError("source artifact is not valid UTF-8") from exc

        actual_digest = hashlib.sha256(self.raw_utf8).hexdigest()
        if self.digest != actual_digest:
            raise ConsensusContractError(
                f"source artifact digest mismatch: expected {self.digest!r}, got {actual_digest!r}"
            )

    @classmethod
    def from_text(cls, source: SourceName, text: str) -> "SourceArtifact":
        return cls.from_bytes(source, text.encode("utf-8"))

    @classmethod
    def from_bytes(cls, source: SourceName, raw_utf8: bytes) -> "SourceArtifact":
        """Preserve already-materialized extractor bytes without newline folding."""

        return cls(
            source=source,
            raw_utf8=raw_utf8,
            digest=hashlib.sha256(raw_utf8).hexdigest(),
        )

    @property
    def text(self) -> str:
        return self.raw_utf8.decode("utf-8", errors="strict")


@dataclass(frozen=True)
class SourceSpanCandidate:
    """One exact, occurrence-specific byte span offered to the merger."""

    candidate_id: str
    occurrence_id: str
    structural_unit_id: str
    source: SourceName
    artifact_digest: str
    byte_start: int
    byte_end: int
    candidate_type: CandidateType
    comparison_key: str
    visible_text_digest: str | None = None
    non_emphasis_ast_digest: str | None = None
    emphasis_occurrence_ids: tuple[str, ...] = ()
    clean: bool = True
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("candidate_id", "occurrence_id", "structural_unit_id"):
            if not getattr(self, field_name):
                raise ConsensusContractError(f"{field_name} must not be empty")
        if self.source not in _SOURCE_NAMES:
            raise ConsensusContractError(f"unknown extractor source: {self.source!r}")
        if self.candidate_type not in _CANDIDATE_TYPES:
            raise ConsensusContractError(
                f"unknown candidate type: {self.candidate_type!r}"
            )
        if len(self.artifact_digest) != 64 or any(
            char not in "0123456789abcdef" for char in self.artifact_digest
        ):
            raise ConsensusContractError("artifact_digest must be a lowercase SHA-256 digest")
        if self.byte_start < 0 or self.byte_end <= self.byte_start:
            raise ConsensusContractError(
                f"invalid byte range [{self.byte_start}, {self.byte_end})"
            )
        if self.clean and self.invalid_reason:
            raise ConsensusContractError("clean candidates cannot have invalid_reason")
        if not self.clean and not self.invalid_reason:
            raise ConsensusContractError("invalid candidates require invalid_reason")
        formatting_digests = (
            self.visible_text_digest,
            self.non_emphasis_ast_digest,
        )
        if any(formatting_digests) and not all(formatting_digests):
            raise ConsensusContractError("formatting evidence digests must be paired")
        for digest in (*formatting_digests, *self.emphasis_occurrence_ids):
            if digest is not None and (
                len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ConsensusContractError(
                    "formatting evidence must use lowercase SHA-256 digests"
                )
        if self.emphasis_occurrence_ids and not all(formatting_digests):
            raise ConsensusContractError(
                "emphasis occurrences require complete formatting evidence"
            )

    def extract_bytes(self, artifact: SourceArtifact) -> bytes:
        """Reproduce this candidate's exact source bytes after provenance checks."""

        if artifact.source != self.source:
            raise ConsensusContractError(
                f"candidate {self.candidate_id!r} belongs to {self.source}, not {artifact.source}"
            )
        if artifact.digest != self.artifact_digest:
            raise ConsensusContractError(
                f"candidate {self.candidate_id!r} artifact digest does not match"
            )
        if self.byte_end > len(artifact.raw_utf8):
            raise ConsensusContractError(
                f"candidate {self.candidate_id!r} exceeds its source artifact"
            )

        raw_span = artifact.raw_utf8[self.byte_start:self.byte_end]
        try:
            text = raw_span.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ConsensusContractError(
                f"candidate {self.candidate_id!r} splits a UTF-8 code point"
            ) from exc

        if not self.clean:
            raise ConsensusContractError(
                f"candidate {self.candidate_id!r} is invalid: {self.invalid_reason}"
            )
        require_publishable_text(text, candidate_id=self.candidate_id)
        return raw_span


class RegionSelectionDecision(BaseModel):
    """Selection-only decision for one bounded baseline region."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    region_id: str = Field(min_length=1)
    action: SelectionAction
    candidate_ids: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_action_contract(self) -> "RegionSelectionDecision":
        if self.action == "keep_baseline" and self.candidate_ids:
            raise ValueError("keep_baseline must not include candidate_ids")
        if self.action == "select_candidates" and not self.candidate_ids:
            raise ValueError("select_candidates requires at least one candidate_id")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate_ids must not contain duplicates")
        if any(not candidate_id for candidate_id in self.candidate_ids):
            raise ValueError("candidate_ids must not contain empty values")
        return self


class SelectionDecisionResponse(BaseModel):
    """Structured model response with no publication-text field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decisions: tuple[RegionSelectionDecision, ...]

    @model_validator(mode="after")
    def validate_unique_regions(self) -> "SelectionDecisionResponse":
        region_ids = [decision.region_id for decision in self.decisions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("decisions must not contain duplicate region IDs")
        return self


class CandidateSelectionEvidence(BaseModel):
    """Read-only source evidence shown to the selector model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    source: SourceName
    artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    occurrence_id: str = Field(min_length=1)
    structural_unit_id: str = Field(min_length=1)
    candidate_type: CandidateType
    byte_start: int = Field(ge=0)
    byte_end: int = Field(gt=0)
    visible_text_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    non_emphasis_ast_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    emphasis_occurrence_ids: tuple[str, ...] = Field(default_factory=tuple)
    display: str = Field(max_length=4096)

    @model_validator(mode="after")
    def validate_byte_range(self) -> "CandidateSelectionEvidence":
        if self.byte_end <= self.byte_start:
            raise ValueError("byte_end must be greater than byte_start")
        if bool(self.visible_text_digest) != bool(self.non_emphasis_ast_digest):
            raise ValueError("formatting evidence digests must be paired")
        if self.emphasis_occurrence_ids and not self.visible_text_digest:
            raise ValueError("emphasis occurrences require formatting evidence")
        if any(
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            for digest in self.emphasis_occurrence_ids
        ):
            raise ValueError("emphasis occurrences must be SHA-256 digests")
        return self


class CandidateSelectionRegionRequest(BaseModel):
    """One strict region graph serialized for model selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    region_id: str = Field(min_length=1)
    baseline_candidate_id: str = Field(min_length=1)
    candidates: tuple[CandidateSelectionEvidence, ...] = Field(min_length=1, max_length=32)
    valid_paths: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_region_graph(self) -> "CandidateSelectionRegionRequest":
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("region candidates must have unique candidate IDs")
        if self.baseline_candidate_id not in candidate_ids:
            raise ValueError("baseline candidate is outside the region graph")
        if (self.baseline_candidate_id,) not in self.valid_paths:
            raise ValueError("valid_paths must include the single-candidate baseline path")
        if len(self.valid_paths) != len(set(self.valid_paths)):
            raise ValueError("valid_paths must not contain duplicates")
        allowed = set(candidate_ids)
        for path in self.valid_paths:
            if not path:
                raise ValueError("valid_paths must not contain an empty path")
            if len(path) > 16:
                raise ValueError("a valid path must not exceed 16 candidates")
            if len(path) != len(set(path)):
                raise ValueError("a valid path must not repeat a candidate ID")
            if set(path) - allowed:
                raise ValueError("valid path contains a candidate outside the region graph")
        return self


class CandidateSelectionRequest(BaseModel):
    """Strict model input with ordered, non-overlapping region ownership."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    regions: tuple[CandidateSelectionRegionRequest, ...] = Field(max_length=8)

    @model_validator(mode="after")
    def validate_document_graph(self) -> "CandidateSelectionRequest":
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("candidate-selection request has duplicate region IDs")
        candidate_owners: dict[str, str] = {}
        for region in self.regions:
            for candidate in region.candidates:
                owner = candidate_owners.setdefault(candidate.candidate_id, region.region_id)
                if owner != region.region_id:
                    raise ValueError(
                        f"candidate {candidate.candidate_id!r} is reused across regions"
                    )
        if sum(len(region.candidates) for region in self.regions) > 128:
            raise ValueError("candidate-selection request exceeds 128 candidates")
        if len(self.model_dump_json().encode("utf-8")) > 131_072:
            raise ValueError("candidate-selection request exceeds 128 KiB")
        return self


@dataclass(frozen=True)
class RegionCandidateGraph:
    """Executable ordered paths for one bounded document region."""

    region_id: str
    baseline_candidate_id: str
    valid_paths: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.region_id or not self.baseline_candidate_id:
            raise ConsensusContractError("region and baseline candidate IDs must not be empty")
        if not self.valid_paths or (self.baseline_candidate_id,) not in self.valid_paths:
            raise ConsensusContractError(
                "region graph must include the single-candidate baseline path"
            )
        if len(self.valid_paths) != len(set(self.valid_paths)):
            raise ConsensusContractError("region graph paths must be unique")
        for path in self.valid_paths:
            if not path or len(path) != len(set(path)):
                raise ConsensusContractError(
                    "region graph paths must be non-empty and occurrence-unique"
                )


@dataclass(frozen=True)
class CandidateGraph:
    """Ordered regions with exclusive ownership of candidate occurrences."""

    regions: tuple[RegionCandidateGraph, ...]

    def __post_init__(self) -> None:
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ConsensusContractError("candidate graph region IDs must be unique")
        owners: dict[str, str] = {}
        for region in self.regions:
            for path in region.valid_paths:
                for candidate_id in path:
                    owner = owners.setdefault(candidate_id, region.region_id)
                    if owner != region.region_id:
                        raise ConsensusContractError(
                            f"candidate {candidate_id!r} is reused across regions"
                        )


@dataclass(frozen=True)
class CandidateSelectionBatchPlan:
    """Bounded selector requests plus regions that must retain baseline."""

    requests: tuple[CandidateSelectionRequest, ...]
    fallback_region_ids: tuple[str, ...]



class CandidateStore:
    """In-memory registry that resolves only verified source-backed spans."""

    def __init__(
        self,
        artifacts: Mapping[tuple[SourceName, str], SourceArtifact],
        candidates: Mapping[str, SourceSpanCandidate],
    ) -> None:
        self._artifacts = dict(artifacts)
        self._candidates = dict(candidates)
        for candidate_id, candidate in self._candidates.items():
            if candidate_id != candidate.candidate_id:
                raise ConsensusContractError(
                    f"candidate registry key {candidate_id!r} does not match "
                    f"candidate ID {candidate.candidate_id!r}"
                )
        for artifact_key, artifact in self._artifacts.items():
            expected_key = (artifact.source, artifact.digest)
            if artifact_key != expected_key:
                raise ConsensusContractError(
                    f"artifact registry key {artifact_key!r} does not match {expected_key!r}"
                )

    def candidate_bytes(self, candidate_id: str) -> bytes:
        try:
            candidate = self._candidates[candidate_id]
        except KeyError as exc:
            raise ConsensusContractError(f"unknown candidate ID: {candidate_id!r}") from exc

        artifact_key = (candidate.source, candidate.artifact_digest)
        try:
            artifact = self._artifacts[artifact_key]
        except KeyError as exc:
            raise ConsensusContractError(
                f"missing source artifact for candidate {candidate_id!r}"
            ) from exc
        return candidate.extract_bytes(artifact)

    def candidate_metadata(self, candidate_id: str) -> SourceSpanCandidate:
        """Return immutable provenance metadata for a registered candidate."""

        try:
            return self._candidates[candidate_id]
        except KeyError as exc:
            raise ConsensusContractError(f"unknown candidate ID: {candidate_id!r}") from exc

    def source_artifacts(self) -> tuple[SourceArtifact, ...]:
        """Return the immutable source artifacts for deterministic validation."""

        return tuple(
            artifact
            for _key, artifact in sorted(
                self._artifacts.items(),
                key=lambda item: item[0],
            )
        )

    def validate_graph(self, graph: CandidateGraph) -> None:
        """Validate topology and provenance before any region can be assembled."""

        occurrence_owners: dict[str, str] = {}
        spans_by_artifact: dict[
            tuple[SourceName, str],
            list[tuple[int, int, str]],
        ] = {}
        for region in graph.regions:
            region_candidate_ids = {
                candidate_id
                for path in region.valid_paths
                for candidate_id in path
            }
            try:
                region_candidates = [
                    self._candidates[candidate_id]
                    for candidate_id in region_candidate_ids
                ]
            except KeyError as exc:
                raise ConsensusContractError(
                    f"region graph references unknown candidate {exc.args[0]!r}"
                ) from exc
            structural_units = {
                candidate.structural_unit_id for candidate in region_candidates
            }
            if len(structural_units) != 1:
                raise ConsensusContractError(
                    f"region {region.region_id!r} spans multiple structural units"
                )
            baseline_candidate = self._candidates.get(region.baseline_candidate_id)
            if baseline_candidate is None:
                raise ConsensusContractError(
                    f"region {region.region_id!r} references an unknown baseline candidate"
                )
            incompatible_types = {
                candidate.candidate_type
                for candidate in region_candidates
                if candidate.candidate_type != baseline_candidate.candidate_type
            }
            if incompatible_types:
                raise ConsensusContractError(
                    f"region {region.region_id!r} mixes baseline candidate type "
                    f"{baseline_candidate.candidate_type!r} with incompatible types"
                )
            for candidate in region_candidates:
                occurrence_owner = occurrence_owners.setdefault(
                    candidate.occurrence_id,
                    region.region_id,
                )
                if occurrence_owner != region.region_id:
                    raise ConsensusContractError(
                        f"occurrence {candidate.occurrence_id!r} is reused across regions"
                    )
                spans_by_artifact.setdefault(
                    (candidate.source, candidate.artifact_digest), []
                ).append(
                    (candidate.byte_start, candidate.byte_end, region.region_id)
                )
            for path in region.valid_paths:
                candidates = [self._candidates[candidate_id] for candidate_id in path]
                occurrence_ids = [candidate.occurrence_id for candidate in candidates]
                if len(occurrence_ids) != len(set(occurrence_ids)):
                    raise ConsensusContractError(
                        f"region {region.region_id!r} path repeats an occurrence"
                    )
                for previous, current in zip(candidates, candidates[1:]):
                    same_artifact = (
                        previous.source == current.source
                        and previous.artifact_digest == current.artifact_digest
                    )
                    if same_artifact and current.byte_start < previous.byte_end:
                        raise ConsensusContractError(
                            f"region {region.region_id!r} path is overlapping or non-monotonic"
                        )
                for candidate in candidates:
                    self.candidate_bytes(candidate.candidate_id)

        # Detect cross-region overlap in O(n log n) per artifact. Multiple
        # alternative spans inside one region may overlap by design.
        for spans in spans_by_artifact.values():
            active: list[tuple[int, str]] = []
            active_owner_counts: dict[str, int] = {}
            for start, end, owner in sorted(spans):
                while active and active[0][0] <= start:
                    _, expired_owner = heapq.heappop(active)
                    active_owner_counts[expired_owner] -= 1
                    if active_owner_counts[expired_owner] == 0:
                        del active_owner_counts[expired_owner]
                if active_owner_counts and (
                    len(active_owner_counts) > 1 or owner not in active_owner_counts
                ):
                    raise ConsensusContractError(
                        "source span overlaps candidates owned by different regions"
                    )
                heapq.heappush(active, (end, owner))
                active_owner_counts[owner] = active_owner_counts.get(owner, 0) + 1

    def _build_selection_region_request(
        self,
        region: RegionCandidateGraph,
    ) -> CandidateSelectionRegionRequest:
        ordered_candidate_ids = tuple(
            dict.fromkeys(
                candidate_id
                for path in region.valid_paths
                for candidate_id in path
            )
        )
        evidence = []
        for candidate_id in ordered_candidate_ids:
            candidate = self._candidates[candidate_id]
            evidence.append(
                CandidateSelectionEvidence(
                    candidate_id=candidate.candidate_id,
                    source=candidate.source,
                    artifact_digest=candidate.artifact_digest,
                    occurrence_id=candidate.occurrence_id,
                    structural_unit_id=candidate.structural_unit_id,
                    candidate_type=candidate.candidate_type,
                    byte_start=candidate.byte_start,
                    byte_end=candidate.byte_end,
                    visible_text_digest=candidate.visible_text_digest,
                    non_emphasis_ast_digest=candidate.non_emphasis_ast_digest,
                    emphasis_occurrence_ids=candidate.emphasis_occurrence_ids,
                    display=self.candidate_bytes(candidate_id).decode("utf-8", errors="strict"),
                )
            )
        return CandidateSelectionRegionRequest(
            region_id=region.region_id,
            baseline_candidate_id=region.baseline_candidate_id,
            candidates=tuple(evidence),
            valid_paths=region.valid_paths,
        )

    def build_selection_request_batches(
        self,
        graph: CandidateGraph,
    ) -> CandidateSelectionBatchPlan:
        """Create deterministic bounded batches; oversize regions keep baseline."""

        self.validate_graph(graph)
        batches: list[CandidateSelectionRequest] = []
        fallback_region_ids: list[str] = []
        pending: list[CandidateSelectionRegionRequest] = []
        for region in graph.regions:
            try:
                request_region = self._build_selection_region_request(region)
            except (ConsensusContractError, ValidationError):
                fallback_region_ids.append(region.region_id)
                continue
            try:
                CandidateSelectionRequest(regions=tuple(pending + [request_region]))
            except ValidationError:
                if pending:
                    batches.append(CandidateSelectionRequest(regions=tuple(pending)))
                    pending = []
                try:
                    CandidateSelectionRequest(regions=(request_region,))
                except ValidationError:
                    fallback_region_ids.append(region.region_id)
                    continue
            pending.append(request_region)
        if pending:
            batches.append(CandidateSelectionRequest(regions=tuple(pending)))
        return CandidateSelectionBatchPlan(
            requests=tuple(batches),
            fallback_region_ids=tuple(fallback_region_ids),
        )

    def build_selection_request(self, graph: CandidateGraph) -> CandidateSelectionRequest:
        """Serialize one already-bounded graph for a single model request."""

        if not graph.regions:
            self.validate_graph(graph)
            return CandidateSelectionRequest(regions=())
        plan = self.build_selection_request_batches(graph)
        if plan.fallback_region_ids or len(plan.requests) != 1:
            raise ConsensusContractError(
                "candidate graph requires batching or contains an oversized region"
            )
        return plan.requests[0]

    def resolve_region(
        self,
        decision: RegionSelectionDecision,
        *,
        region: RegionCandidateGraph,
    ) -> bytes:
        """Execute a model decision without accepting any model-authored text."""

        self.validate_graph(CandidateGraph(regions=(region,)))
        if decision.region_id != region.region_id:
            raise ConsensusContractError("selection decision does not match the region graph")
        if decision.action == "keep_baseline":
            return self.candidate_bytes(region.baseline_candidate_id)

        selected_path = tuple(decision.candidate_ids)
        if selected_path not in region.valid_paths:
            raise ConsensusContractError(
                f"selection is not an executable path in region {region.region_id!r}"
            )
        return b"".join(self.candidate_bytes(candidate_id) for candidate_id in selected_path)
