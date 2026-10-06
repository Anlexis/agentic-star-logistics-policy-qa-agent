# LOG-C2-029 — Proof-of-Boundary
#
# Three guarantees this agent must never break, asserted against the compiled
# graph (the merged behaviour, not any single node's intent):
#
#   B-1  Scope boundary. A question that needs live shipment data is
#          declined. The agent answers from static policy text, so a guess
#          dressed as an answer would be worse than a refusal.
#
#   B-3  Grounding boundary. No policy statement leaves the agent without
#          the sources it came from. Passages with no source produce an error,
#          never an uncited answer.
#
#   B-6  Content boundary. Restricted content — credentials, national-ID and
#          payment-card forms — never crosses the output boundary, whatever
#          put it into the corpus.
#
# The node modules import emit_trace_event at module level. The published SDK
# ships a real `shared` package, so it is not stubbed — an autouse fixture
# mutes the function in each node module after import instead.

import json

import pytest
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from unittest.mock import patch

from src.graph.graph import LogisticsPolicyQAAgent
from src.nodes.answer_generate import SOURCES_HEADING, AnswerGenerateNode
from src.nodes.post_process_node import PostProcessNode
from src.schemas.state import to_json


@pytest.fixture(autouse=True)
def _mute_audit():
    modules = [
        "src.nodes.answer_generate",
        "src.nodes.response_validate",
        "src.nodes.post_process_node",
    ]
    patches = []
    for module in modules:
        patcher = patch(f"{module}.emit_trace_event", lambda *a, **k: None)
        patcher.start()
        patches.append(patcher)
    yield
    for patcher in patches:
        patcher.stop()


@pytest.fixture(scope="module")
def agent():
    graph = LogisticsPolicyQAAgent()
    graph.compile()
    return graph


def _ask(agent, question, input_context=None):
    ctx = InvocationContext(session_id="pob", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
    return agent.invoke(question, ctx=ctx, input_context=input_context or {})


# ---------------------------------------------------------------------------
# B-1 / B-2: scope boundary
# ---------------------------------------------------------------------------


class TestScopeBoundary:
    @pytest.mark.parametrize(
        "question",
        [
            "What is the ETA for shipment 4521?",
            "Give me the current status of my consignment.",
            "Where is my parcel right now?",
            "I need real-time visibility on this order.",
        ],
    )
    def test_a_live_data_question_never_returns_a_policy_answer(self, agent, question):
        result = _ask(agent, question)
        output = result.get("output") or ""
        assert "outside the logistics policy corpus" in output
        assert SOURCES_HEADING not in output

    def test_a_policy_question_using_the_same_vocabulary_is_answered(self, agent):
        """The screen must not decline genuine policy work."""
        docs = [
            {
                "document_id": "tracking_numbers",
                "title": "Tracking number formats",
                "source": "carrier_policy/tracking_numbers.md",
                "text": "A tracking number is twelve digits printed on the consignment note.",
                "last_verified": "2026-04-01",
            }
        ]
        output = _ask(agent, "What format does a tracking number use?", {"policy_documents": docs}).get("output")
        assert "twelve digits" in output


# ---------------------------------------------------------------------------
# B-3 / B-4 / B-5: grounding boundary
# ---------------------------------------------------------------------------


class TestGroundingBoundary:
    def test_every_answer_carries_its_sources(self, agent):
        output = _ask(agent, "What are the cold chain handling requirements?").get("output")
        assert SOURCES_HEADING in output
        assert "logistics_policy/cold_chain_handling.md" in output

    def test_passages_without_a_source_produce_an_error_not_an_uncited_answer(self):
        chunks = [
            {"source": "  ", "text": "Some uncited policy text.", "last_verified": "2026-01-01"},
            {"source": "", "text": "More uncited text.", "last_verified": "2026-01-02"},
        ]
        result = AnswerGenerateNode().execute({"retrieved_chunks": to_json(chunks)})
        assert result["status"] == AgentStatus.ERROR
        assert result.get("error_log")
        assert "answer" not in result

    def test_an_answer_that_lost_its_sources_is_withheld_at_the_boundary(self):
        result = PostProcessNode().execute(
            {"validated_answer": "Cold chain moves between 2 and 8 degrees.", "validation_passed": True}
        )
        assert result["status"] == AgentStatus.ERROR
        assert "Cold chain moves" not in result["formatted_output"]


# ---------------------------------------------------------------------------
# B-6 / B-7: content boundary
# ---------------------------------------------------------------------------


class TestContentBoundary:
    @pytest.mark.parametrize(
        "leak",
        [
            "sk-abcdefghijklmnop1234567890",
            "Bearer abcdefghijklmnopqrstuvwxyz012345",
            "123-45-6789",
            "4111 1111 1111 1111",
        ],
    )
    def test_restricted_content_in_the_corpus_never_reaches_the_caller(self, agent, leak):
        docs = [
            {
                "document_id": "leaky",
                "title": "Packaging standards",
                "source": "carrier_policy/leaky.md",
                "text": f"Packaging standards for fragile cartons. Contact detail: {leak}",
                "last_verified": "2026-04-01",
            }
        ]
        result = _ask(agent, "What packaging standards apply to fragile cartons?", {"policy_documents": docs})
        assert leak not in json.dumps(result, default=str)

    def test_the_caller_question_is_never_echoed_into_the_answer(self, agent):
        question = "What are the cold chain handling requirements for a frozen consignment?"
        assert question not in (_ask(agent, question).get("output") or "")
