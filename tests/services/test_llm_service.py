import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.source_contracts import (
    CandidateGraph,
    CandidateStore,
    RegionCandidateGraph,
    SourceArtifact,
    SourceSpanCandidate,
)
from app.services.llm_service import (
    CandidateSelectionFailure,
    LLM,
    TokenAccumulator,
    _NumberedRegionDecision,
    compute_cost,
)


def _selection_contract():
    artifacts = {}
    candidates = {}
    for candidate_id, source, text in (
        ("baseline", "grobid", "Gene Gg1."),
        ("preferred", "marker", "Gene Gγ1."),
    ):
        artifact = SourceArtifact.from_text(source, text)
        artifacts[(source, artifact.digest)] = artifact
        candidates[candidate_id] = SourceSpanCandidate(
            candidate_id=candidate_id,
            occurrence_id=f"occurrence-{candidate_id}",
            structural_unit_id="unit-1",
            source=source,
            artifact_digest=artifact.digest,
            byte_start=0,
            byte_end=len(artifact.raw_utf8),
            candidate_type="prose",
            comparison_key=text.casefold(),
        )
    region = RegionCandidateGraph(
        region_id="region-1",
        baseline_candidate_id="baseline",
        valid_paths=(("baseline",), ("preferred",)),
    )
    return CandidateStore(artifacts, candidates), CandidateGraph(regions=(region,))


def _llm_with_parsed(parsed, *, refusal=None, usage=None):
    llm = object.__new__(LLM)
    llm.client = MagicMock()
    llm.usage = TokenAccumulator()
    llm.openai_timeout_seconds = 30.0
    llm.openai_max_retries = 1
    llm.selection_call_traces = []
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage,
    )
    llm.client.chat.completions.parse.return_value = completion
    return llm


def _parsed(payload):
    return SimpleNamespace(model_dump=lambda **_kwargs: payload)


def test_llm_client_uses_bounded_timeout_and_retries():
    with patch("app.services.llm_service.OpenAI") as openai_cls:
        llm = LLM(
            api_key="key",
            openai_timeout_seconds=12.5,
            openai_max_retries=0,
        )

    openai_cls.assert_called_once_with(api_key="key", timeout=12.5, max_retries=0)
    assert llm.openai_timeout_seconds == 12.5
    assert llm.openai_max_retries == 0


@pytest.mark.parametrize(
    ("timeout", "retries"),
    [(0, 1), (-1, 1), (float("inf"), 1), (601, 1), (10, -1), (10, 3)],
)
def test_llm_client_rejects_unbounded_transport(timeout, retries):
    with patch("app.services.llm_service.OpenAI") as openai_cls:
        with pytest.raises(ValueError):
            LLM(
                api_key="key",
                openai_timeout_seconds=timeout,
                openai_max_retries=retries,
            )
    openai_cls.assert_not_called()


def test_selector_returns_application_owned_path_from_number_only():
    store, graph = _selection_contract()
    request = store.build_selection_request(graph)
    payload, _choices = LLM._numbered_request(request)
    llm = _llm_with_parsed(
        _parsed(
            {
                "request_sha256": payload["request_sha256"],
                "decisions": ({"region_id": "region-1", "choice": 1},),
            }
        )
    )

    response = llm.resolve_candidate_selections(store, graph, raise_on_failure=True)

    assert response.decisions[0].candidate_ids == ("preferred",)
    assert response.decisions[0].action == "select_candidates"
    call = llm.client.chat.completions.parse.call_args.kwargs
    assert call["model"] == "gpt-5.6-terra"
    assert call["reasoning_effort"] == "medium"
    assert call["response_format"].__name__ == "_NumberedSelectionResponse"
    assert len(llm.selection_call_traces) == 1
    trace = llm.selection_call_traces[0]
    assert trace["outcome"] == "valid"
    assert trace["response"]["decisions"] == [
        {"region_id": "region-1", "choice": 1}
    ]
    assert trace["request"]["regions"][0]["candidates"][0][
        "display_sha256"
    ]
    assert "display" not in trace["request"]["regions"][0]["candidates"][0]


def test_selector_choice_zero_retains_baseline():
    store, graph = _selection_contract()
    request = store.build_selection_request(graph)
    payload, _choices = LLM._numbered_request(request)
    llm = _llm_with_parsed(
        _parsed(
            {
                "request_sha256": payload["request_sha256"],
                "decisions": ({"region_id": "region-1", "choice": 0},),
            }
        )
    )

    response = llm.resolve_candidate_selections(store, graph, raise_on_failure=True)

    assert response.decisions[0].action == "keep_baseline"
    assert response.decisions[0].candidate_ids == ()


@pytest.mark.parametrize(
    "response",
    [
        {"request_sha256": "0" * 64, "decisions": ({"region_id": "region-1", "choice": 1},)},
        {"request_sha256": "a" * 64, "decisions": ({"region_id": "region-1", "choice": 2},)},
        {
            "request_sha256": "a" * 64,
            "decisions": ({"region_id": "region-1", "choice": 1, "text": "invented"},),
        },
    ],
)
def test_selector_rejects_digest_unknown_choice_and_authored_text(response):
    store, graph = _selection_contract()
    llm = _llm_with_parsed(_parsed(response))

    with pytest.raises(CandidateSelectionFailure) as exc_info:
        llm.resolve_candidate_selections(store, graph, raise_on_failure=True)

    assert exc_info.value.reason == "no_valid_terra_selection"


def test_sol_selector_uses_exact_5_6_sol_high():
    store, graph = _selection_contract()
    request = store.build_selection_request(graph)
    payload, _choices = LLM._numbered_request(request)
    llm = _llm_with_parsed(
        _parsed(
            {
                "request_sha256": payload["request_sha256"],
                "decisions": ({"region_id": "region-1", "choice": 0},),
            }
        )
    )

    llm.resolve_candidate_selections(
        store,
        graph,
        use_sol=True,
        raise_on_failure=True,
    )

    call = llm.client.chat.completions.parse.call_args.kwargs
    assert call["model"] == "gpt-5.6-sol"
    assert call["reasoning_effort"] == "high"


def test_bounded_id_choice_returns_existing_skeleton_id_with_sol_high():
    llm = _llm_with_parsed(None)

    def parse(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        parsed = _parsed(
            {
                "request_sha256": payload["request_sha256"],
                "decisions": ({"region_id": "skeleton_conflict", "choice": 1},),
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))],
            usage=None,
        )

    llm.client.chat.completions.parse.side_effect = parse
    choices = [
        {
            "candidate_id": "skeleton-b",
            "projection_id": "projection-b",
            "payload_byte_count": 120,
            "display": "bounded candidate B",
        },
        {
            "candidate_id": "skeleton-a",
            "projection_id": "projection-a",
            "payload_byte_count": 100,
            "display": "bounded candidate A",
        },
    ]

    selected = llm.resolve_bounded_id_choice(
        reason="skeleton_conflict",
        baseline_id="skeleton-b",
        choices=choices,
    )

    assert selected == "skeleton-a"
    call = llm.client.chat.completions.parse.call_args.kwargs
    assert call["model"] == "gpt-5.6-sol"
    assert call["reasoning_effort"] == "high"
    assert call["response_format"].__name__ == "_NumberedSelectionResponse"
    request = json.loads(call["messages"][1]["content"])
    assert request["regions"][0]["keep_baseline_choice"] == 0
    assert {item["candidate_id"] for item in request["regions"][0]["candidates"]} == {
        "skeleton-a",
        "skeleton-b",
    }
    assert {item["display"] for item in request["regions"][0]["candidates"]} == {
        "bounded candidate A",
        "bounded candidate B",
    }
    assert llm.selection_call_traces[0]["call_type"] == "bounded_id_selection_sol"
    assert all(
        "display" not in item
        for item in llm.selection_call_traces[0]["request"]["regions"][0]["candidates"]
    )


def test_bounded_id_choice_receipt_binds_selected_id_to_numeric_trace():
    llm = _llm_with_parsed(None)

    def parse(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        parsed = _parsed({
            "request_sha256": payload["request_sha256"],
            "decisions": ({"region_id": "style-tie", "choice": 1},),
        })
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))],
            usage=None,
        )

    llm.client.chat.completions.parse.side_effect = parse
    receipt = llm.resolve_bounded_id_choice_with_receipt(
        reason="style-tie",
        baseline_id="none",
        choices=[
            {"candidate_id": "none", "display": "none"},
            {"candidate_id": "target", "display": "target"},
        ],
    )

    assert receipt == {
        "selected_candidate_id": "target",
        "request_sha256": llm.selection_call_traces[0]["request"]["request_sha256"],
        "response_choice": 1,
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
    }


def test_bounded_id_choice_accepts_none_plus_sixty_three_targets():
    llm = _llm_with_parsed(None)

    def parse(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        parsed = _parsed({
            "request_sha256": payload["request_sha256"],
            "decisions": ({"region_id": "style-capacity", "choice": 63},),
        })
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                parsed=parsed, refusal=None
            ))],
            usage=None,
        )

    llm.client.chat.completions.parse.side_effect = parse
    choices = [
        {"candidate_id": "none", "display": "none"},
        *(
            {"candidate_id": f"target-{index:02d}", "display": f"target {index}"}
            for index in range(63)
        ),
    ]

    selected = llm.resolve_bounded_id_choice(
        reason="style-capacity",
        baseline_id="none",
        choices=choices,
    )

    assert selected == "target-62"
    request = json.loads(
        llm.client.chat.completions.parse.call_args.kwargs["messages"][1]["content"]
    )
    assert len(request["regions"][0]["candidates"]) == 64


def test_numbered_response_schema_uses_the_last_executable_choice_index():
    assert _NumberedRegionDecision.model_json_schema()["properties"]["choice"][
        "maximum"
    ] == 63
    assert _NumberedRegionDecision(region_id="style-capacity", choice=63).choice == 63
    with pytest.raises(ValueError):
        _NumberedRegionDecision(region_id="style-capacity", choice=64)


def test_bounded_id_choice_rejects_more_than_sixty_four_total_choices():
    llm = _llm_with_parsed(None)
    choices = [
        {"candidate_id": "none", "display": "none"},
        *(
            {"candidate_id": f"target-{index:02d}", "display": f"target {index}"}
            for index in range(64)
        ),
    ]

    with pytest.raises(ValueError, match="between one and 64 choices"):
        llm.resolve_bounded_id_choice(
            reason="style-over-capacity",
            baseline_id="none",
            choices=choices,
        )

    llm.client.chat.completions.parse.assert_not_called()


@pytest.mark.parametrize(
    "response_mutation",
    ("wrong_digest", "unknown_choice", "authored_text"),
)
def test_bounded_id_choice_rejects_unbound_or_nonclosed_response(response_mutation):
    llm = _llm_with_parsed(None)

    def parse(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        response = {
            "request_sha256": payload["request_sha256"],
            "decisions": [{"region_id": "skeleton_conflict", "choice": 0}],
        }
        if response_mutation == "wrong_digest":
            response["request_sha256"] = "0" * 64
        elif response_mutation == "unknown_choice":
            response["decisions"][0]["choice"] = 9
        else:
            response["decisions"][0]["text"] = "authored publication text"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                parsed=_parsed(response), refusal=None
            ))],
            usage=None,
        )

    llm.client.chat.completions.parse.side_effect = parse

    with pytest.raises(CandidateSelectionFailure):
        llm.resolve_bounded_id_choice(
            reason="skeleton_conflict",
            baseline_id="skeleton-a",
            choices=[
                {"candidate_id": "skeleton-a", "projection_id": "projection-a"},
                {"candidate_id": "skeleton-b", "projection_id": "projection-b"},
            ],
        )


def test_token_accounting_and_cost_reject_unknown_model():
    accumulator = TokenAccumulator()
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        prompt_tokens_details=SimpleNamespace(cached_tokens=2),
    )
    accumulator.record(usage, "source_path_selection_terra", "gpt-5.6-terra")
    summary = accumulator.summary()
    assert summary["total_tokens"] == 15
    assert accumulator.tokens_for_types("source_path_selection_terra") == 15

    with pytest.raises(ValueError, match="No pricing entry"):
        compute_cost(summary, {})
