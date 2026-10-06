# LOG-C2-029 — unit tests for the state schema and the inner workflow nodes.
#
# The caller contract, the output boundary, the retrieval service and the
# end-to-end path have their own files:
#   tests/unit/test_caller_contract.py
#   tests/unit/test_output_boundary.py
#   tests/unit/test_retrieval.py
#   tests/proof_of_boundary/test_invoke_e2e.py
#
# The node modules import emit_trace_event at module level. The published SDK
# ships a real `shared` package, so it is not stubbed — an autouse fixture
# mutes the function in each node module after import instead.

import json

import pytest
from framework.schemas.agent_status import AgentStatus
from unittest.mock import patch

from src.nodes.answer_generate import SCOPE_NOTE, SOURCES_HEADING, AnswerGenerateNode
from src.nodes.logistics_policy_kb_retrieve import LogisticsPolicyKBRetrieveNode
from src.nodes.query_normalize import QueryNormalizeNode
from src.nodes.response_validate import ResponseValidateNode
from src.schemas.state import State, finite_in_range, from_json, to_json


@pytest.fixture(autouse=True)
def _mute_audit():
    modules = [
        "src.nodes.query_normalize",
        "src.nodes.logistics_policy_kb_retrieve",
        "src.nodes.answer_generate",
        "src.nodes.response_validate",
    ]
    patches = []
    for module in modules:
        patcher = patch(f"{module}.emit_trace_event", lambda *a, **k: None)
        patcher.start()
        patches.append(patcher)
    yield
    for patcher in patches:
        patcher.stop()


_CHUNKS = [
    {
        "text": "Cold chain consignments travel between 2 and 8 degrees Celsius.",
        "source": "carrier_policy/cold_chain.md",
        "document_id": "cold_chain",
        "title": "Cold chain handling",
        "score": 0.9,
        "last_verified": "2026-05-01",
    },
    {
        "text": "A temperature logger rides inside the packaging.",
        "source": "carrier_policy/logging.md",
        "document_id": "logging",
        "title": "Temperature logging",
        "score": 0.7,
        "last_verified": "2026-03-02",
    },
]


class TestState:
    """State must stay a flat, checkpoint-safe TypedDict."""

    def test_state_extends_the_framework_state(self):
        # TypedDicts reject issubclass(), so the relationship is checked
        # structurally: the framework's own fields are visible on the subclass
        # alongside this template's, and the type is still a plain dict.
        assert issubclass(State, dict)
        for field in ("user_input", "input_context", "status", "formatted_output"):
            assert field in State.__annotations__, f"{field} was lost from the state schema"
        for field in ("query", "validated_context", "retrieved_chunks", "out_of_scope"):
            assert field in State.__annotations__

    def test_state_instantiates_as_a_plain_dict(self):
        state = State(user_input="q", query="q")
        assert isinstance(state, dict)
        assert state["user_input"] == "q"

    def test_json_helpers_round_trip(self):
        value = {"carrier": "yamato_transport", "top_k": 3}
        assert from_json(to_json(value)) == value

    def test_from_json_survives_malformed_input(self):
        assert from_json("{not json", default={}) == {}
        assert from_json(None, default=[]) == []

    @pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", float("nan"), float("inf"), True, None, [1]])
    def test_finite_parser_rejects_unusable_numbers(self, value):
        assert finite_in_range(value, 0, 10) is None

    def test_finite_parser_accepts_in_range_numbers(self):
        assert finite_in_range("3", 1, 5) == 3.0
        assert finite_in_range(5, 1, 5) == 5.0
        assert finite_in_range(6, 1, 5) is None


class TestQueryNormalizeNode:
    def setup_method(self):
        self.node = QueryNormalizeNode()

    def test_filler_is_stripped_and_whitespace_collapsed(self):
        result = self.node.execute({"user_input": "Please tell me   about the PACKAGING rules"})
        assert result["status"] == AgentStatus.SUCCESS
        assert result["normalized_query"] == "the packaging rules"

    def test_carrier_variants_resolve_to_a_canonical_slug(self):
        result = self.node.execute({"user_input": "What is the Kuroneko pickup cut off?"})
        assert json.loads(result["extracted_context"])["carrier"] == "yamato_transport"

    def test_route_codes_are_extracted_and_upper_cased(self):
        result = self.node.execute({"user_input": "rules for hnd-osa"})
        assert json.loads(result["extracted_context"])["route"] == "HND-OSA"

    def test_validated_caller_values_override_the_text(self):
        result = self.node.execute(
            {
                "user_input": "What is the Kuroneko pickup cut off?",
                "validated_context": to_json({"carrier": "sagawa_express", "route_code": "KIX-FUK"}),
            }
        )
        context = json.loads(result["extracted_context"])
        assert context["carrier"] == "sagawa_express"
        assert context["route"] == "KIX-FUK"

    def test_empty_input_is_an_error(self):
        result = self.node.execute({"user_input": "   "})
        assert result["status"] == AgentStatus.SUCCESS
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert result["error_log"]

    def test_non_string_input_is_an_error(self):
        result = self.node.execute({"user_input": {"q": 1}})
        assert result["status"] == AgentStatus.SUCCESS
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")


class TestLogisticsPolicyKBRetrieveNode:
    def setup_method(self):
        self.node = LogisticsPolicyKBRetrieveNode()

    def test_a_matching_question_returns_passages(self):
        result = self.node.execute({"normalized_query": "cold chain handling requirements"})
        assert result["status"] == AgentStatus.SUCCESS
        assert result["out_of_scope"] is False
        assert json.loads(result["retrieved_chunks"])

    def test_caller_documents_are_searched_instead_of_the_bundled_corpus(self):
        result = self.node.execute(
            {
                "normalized_query": "tender cut off on the trunk route",
                "validated_context": to_json(
                    {
                        "policy_documents": [
                            {
                                "document_id": "trunk",
                                "title": "Trunk route",
                                "source": "carrier_policy/trunk.md",
                                "text": "The trunk route tender cut off is 17:00.",
                                "last_verified": "2026-05-04",
                            }
                        ]
                    }
                ),
            }
        )
        [chunk] = json.loads(result["retrieved_chunks"])
        assert chunk["source"] == "carrier_policy/trunk.md"

    def test_a_live_data_question_is_out_of_scope_before_any_search(self):
        result = self.node.execute({"normalized_query": "what is the eta for my parcel"})
        assert result["status"] == AgentStatus.SUCCESS
        assert result["out_of_scope"] is True
        assert result["out_of_scope_reason"] == "live_data_request"
        assert json.loads(result["retrieved_chunks"]) == []

    def test_no_match_is_out_of_scope_not_a_guess(self):
        result = self.node.execute({"normalized_query": "which sonata did beethoven publish"})
        assert result["status"] == AgentStatus.SUCCESS
        assert result["out_of_scope"] is True
        assert result["out_of_scope_reason"] == "no_kb_match"

    def test_a_missing_normalised_question_is_an_error(self):
        result = self.node.execute({})
        assert result["status"] == AgentStatus.ERROR
        assert result["out_of_scope"] is True

    def test_an_unusable_caller_top_k_falls_back_to_the_default(self):
        result = self.node.execute(
            {
                "normalized_query": "cold chain handling requirements",
                "validated_context": to_json({"top_k": float("nan")}),
            }
        )
        assert result["status"] == AgentStatus.SUCCESS
        assert json.loads(result["retrieved_chunks"])


class TestAnswerGenerateNode:
    def setup_method(self):
        self.node = AnswerGenerateNode()

    def test_an_answer_is_composed_with_sources_and_the_scope_note(self):
        result = self.node.execute({"retrieved_chunks": to_json(_CHUNKS)})
        assert result["status"] == AgentStatus.SUCCESS
        assert SOURCES_HEADING in result["answer"]
        assert SCOPE_NOTE in result["answer"]
        assert result["result"] == result["answer"]
        assert json.loads(result["citations"]) == [
            "carrier_policy/cold_chain.md",
            "carrier_policy/logging.md",
        ]

    def test_the_most_recent_verification_date_is_reported(self):
        result = self.node.execute({"retrieved_chunks": to_json(_CHUNKS)})
        assert result["last_verified"] == "2026-05-01"

    def test_the_question_is_not_echoed_into_the_answer(self):
        question = "What temperature applies to a cold chain consignment on this route?"
        result = self.node.execute({"user_input": question, "retrieved_chunks": to_json(_CHUNKS)})
        assert question not in result["answer"]

    def test_out_of_scope_generates_nothing(self):
        result = self.node.execute({"out_of_scope": True, "retrieved_chunks": to_json([])})
        assert result["status"] == AgentStatus.SUCCESS
        assert "answer" not in result and "citations" not in result

    def test_passages_without_a_source_are_an_error_not_an_uncited_answer(self):
        chunks = [dict(chunk, source="  ") for chunk in _CHUNKS]
        result = self.node.execute({"retrieved_chunks": to_json(chunks)})
        assert result["status"] == AgentStatus.ERROR
        assert "answer" not in result

    def test_no_passages_at_all_is_an_error(self):
        result = self.node.execute({"retrieved_chunks": to_json([])})
        assert result["status"] == AgentStatus.ERROR

    def test_the_length_ceiling_trims_the_body_never_the_citations(self):
        long_chunks = [dict(_CHUNKS[0], text="policy text. " * 400)]
        result = self.node.execute(
            {
                "retrieved_chunks": to_json(long_chunks),
                "validated_context": to_json({"max_answer_chars": 600}),
            }
        )
        assert result["status"] == AgentStatus.SUCCESS
        assert len(result["answer"]) <= 600
        assert SOURCES_HEADING in result["answer"]
        assert SCOPE_NOTE in result["answer"]


class TestResponseValidateNode:
    def setup_method(self):
        self.node = ResponseValidateNode()

    def _answer(self, body: str) -> str:
        return (
            f"{body}\n\n---\n{SOURCES_HEADING}\n- carrier_policy/cold_chain.md\n\n"
            f"*Last verified: 2026-05-01*\n\n*{SCOPE_NOTE}*"
        )

    def test_a_clean_answer_passes(self):
        answer = self._answer("Cold chain consignments travel between 2 and 8 degrees Celsius.")
        result = self.node.execute({"result": answer})
        assert result["status"] == AgentStatus.SUCCESS
        assert result["validation_passed"] is True
        assert result["validated_answer"] == answer

    def test_the_standing_scope_note_does_not_scope_the_answer_out(self):
        """The note names live status by design; screening it would decline
        every answer the agent gives."""
        result = self.node.execute({"result": self._answer("Packaging rules for fragile goods.")})
        assert result["validation_passed"] is True

    def test_an_answer_body_asserting_live_status_is_withheld(self):
        result = self.node.execute({"result": self._answer("The current status of the parcel is in transit.")})
        assert result["status"] == AgentStatus.SUCCESS
        assert result["out_of_scope"] is True
        assert result["out_of_scope_reason"] == "live_data_request"
        assert result["validation_passed"] is False
        assert result["validated_answer"].strip()

    def test_a_question_already_out_of_scope_gets_redirect_text(self):
        result = self.node.execute({"result": "", "out_of_scope": True, "out_of_scope_reason": "no_kb_match"})
        assert result["status"] == AgentStatus.SUCCESS
        assert result["validation_passed"] is False
        assert result["validated_answer"].strip()

    def test_an_empty_answer_is_an_error(self):
        result = self.node.execute({"result": "   "})
        assert result["status"] == AgentStatus.ERROR

    def test_an_over_long_answer_is_an_error(self):
        result = self.node.execute({"result": "x" * 3000, "validated_context": to_json({"max_answer_chars": 500})})
        assert result["status"] == AgentStatus.ERROR
        assert result["validation_passed"] is False

    def test_restricted_content_withholds_the_answer(self):
        result = self.node.execute({"result": self._answer("api_key = sk-abcdefghijklmnop1234567890")})
        assert result["status"] == AgentStatus.ERROR
        assert "validated_answer" not in result

    def test_an_unusable_caller_ceiling_falls_back_to_the_default(self):
        answer = self._answer("Packaging rules for fragile goods.")
        result = self.node.execute({"result": answer, "validated_context": to_json({"max_answer_chars": "NaN"})})
        assert result["status"] == AgentStatus.SUCCESS


class TestRuntimeConfig:
    """The values declared in config/config.yaml must actually govern the run.

    The manifest is flat and carries no runtime block, so a reader pointed at it
    finds nothing and every declared value goes silently unused — the graph then
    runs on framework defaults while the file says otherwise. These tests pin
    that the declared values are loaded and that they reach the inner graph,
    which is constructed separately and would otherwise get no config at all.
    """

    def test_declared_values_are_loaded_by_the_outer_graph(self):
        from src.graph.graph import LogisticsPolicyQAAgent, _runtime_config

        declared = _runtime_config()
        assert declared, "config/config.yaml declared nothing — the reader is pointed at the wrong file"
        agent = LogisticsPolicyQAAgent()
        for key, value in declared.items():
            assert agent.config.get(key) == value

    def test_an_explicit_config_still_wins(self):
        from src.graph.graph import LogisticsPolicyQAAgent

        assert LogisticsPolicyQAAgent(config={"max_retry": 1}).config == {"max_retry": 1}

    def test_declared_values_reach_the_inner_graph(self):
        from src.graph.graph import LogisticsPolicyQAGraphNode, _runtime_config

        # The entry point resolves config/config.yaml and threads it in at
        # register_nodes() time; the node never opens the file itself.
        declared = _runtime_config()
        inner = LogisticsPolicyQAGraphNode(runtime_config=declared).get_subgraph()
        for key, value in declared.items():
            assert inner.config.get(key) == value
            assert inner.config["configurable"].get(key) == value

    @pytest.mark.parametrize(
        "config",
        [{"max_retry": -1}, {"max_retry": 10}, {"max_retry": True}, {"timeout_s": 0}, {"timeout_s": 601}],
    )
    def test_an_unusable_declared_value_fails_at_compile_time(self, config):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        with pytest.raises(ValueError):
            DomainWorkflowGraph(config=config).compile()
