# LOG-C2-029 — the external output boundary (PostProcessNode).
#
# Whatever this node returns is what leaves the agent, so the published output
# contract is asserted here for EVERY representation the answer can take, not
# only the convenient one:
#
#   * the caller's own words are never echoed back verbatim;
#   * restricted content anywhere in the payload — including nested inside the
#     citation list — withholds the whole answer;
#   * an answer that lost its sources is withheld, one that lost only the scope
#     note is repaired;
#   * a scope redirect, which cites nothing by design, is recognised from the
#     workflow's own flag and passes;
#   * this domain's identifiers pass through byte-identical.
#
# The last one is the point of the "no numeric grid" decision recorded in
# src/nodes/post_process_node.py: a rounding grid identifies money by treating
# any standalone three-letter uppercase word as a currency marker, which in
# logistics text is a route, port or carrier code. These tests pin that
# identifiers are not rewritten.

import pytest
from framework.schemas.agent_status import AgentStatus

from src.nodes.answer_generate import SCOPE_NOTE, SOURCES_HEADING
from src.nodes.post_process_node import PostProcessNode, enforce_output_contract

_BODY = "**Logistics policy answer**\n\n**HND-OSA trunk route rules**\n\nTender before 17:00."
_FOOTER = (
    f"\n\n---\n{SOURCES_HEADING}\n- carrier_policy/route_hnd_osa.md\n\n*Last verified: 2026-05-04*\n\n*{SCOPE_NOTE}*"
)
_ANSWER = _BODY + _FOOTER


def _run(state):
    return PostProcessNode().execute(state)


def _released(state):
    result = _run(state)
    assert result["status"] == AgentStatus.SUCCESS, result.get("error_log")
    return result["formatted_output"]


def _withheld(result):
    assert result["status"] == AgentStatus.ERROR
    assert result.get("error_log")
    return result["formatted_output"]


class TestReleasePath:
    def test_a_contract_compliant_answer_is_released_unchanged(self):
        assert _released({"validated_answer": _ANSWER, "validation_passed": True}) == _ANSWER

    def test_an_empty_answer_is_withheld(self):
        assert "withheld" in _withheld(_run({"validated_answer": "   "})).lower()


class TestEchoedQuestionRedaction:
    def test_the_callers_question_is_redacted_when_echoed_back(self):
        question = "What are the tender rules on route HND-OSA before cut off?"
        released = _released(
            {
                "validated_answer": f"{_BODY}\n\n{question}{_FOOTER}",
                "validation_passed": True,
                "user_input": question,
            }
        )
        assert question not in released
        assert "[REDACTED]" in released

    def test_corpus_vocabulary_shared_with_a_short_question_is_not_redacted(self):
        """A single term is corpus vocabulary; redacting it would delete the
        answer's own subject."""
        released = _released({"validated_answer": _ANSWER, "validation_passed": True, "user_input": "HND-OSA"})
        assert released == _ANSWER

    def test_the_validated_question_is_redacted_too(self):
        question = "Which packaging rules apply to a fragile carton shipment?"
        released = _released(
            {
                "validated_answer": f"{_BODY}\n\n{question}{_FOOTER}",
                "validation_passed": True,
                "validated_input": question,
            }
        )
        assert question not in released


class TestRestrictedContent:
    @pytest.mark.parametrize(
        "leak",
        [
            "sk-abcdefghijklmnop1234567890",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r",
            "Bearer abcdefghijklmnopqrstuvwxyz012345",
            "password: hunter2hunter2",
            "123-45-6789",
            "4111 1111 1111 1111",
        ],
    )
    def test_a_leak_in_the_answer_withholds_the_whole_answer(self, leak):
        result = _run({"validated_answer": _BODY + f"\n\n{leak}" + _FOOTER, "validation_passed": True})
        assert leak not in _withheld(result)

    def test_a_leak_nested_in_the_citation_list_is_caught(self):
        result = _run(
            {
                "validated_answer": _ANSWER,
                "validation_passed": True,
                "citations": '["carrier_policy/a.md", "token=abcdefgh12345678"]',
            }
        )
        assert "abcdefgh12345678" not in _withheld(result)

    def test_ordinary_policy_text_is_not_mistaken_for_a_leak(self):
        answer = (
            "**Logistics policy answer**\n\nContainer MSKU 4512345 moves on route HND-OSA "
            "with reference REF-2026-0041 and a 12-digit tracking number." + _FOOTER
        )
        assert _released({"validated_answer": answer, "validation_passed": True}) == answer


class TestOutputContractEnforcement:
    def test_an_answer_without_sources_is_withheld(self):
        result = _run({"validated_answer": _BODY, "validation_passed": True})
        assert "withheld" in _withheld(result).lower()

    def test_an_empty_sources_block_is_withheld(self):
        answer = f"{_BODY}\n\n---\n{SOURCES_HEADING}\n\n*{SCOPE_NOTE}*"
        _withheld(_run({"validated_answer": answer, "validation_passed": True}))

    def test_a_missing_scope_note_is_restored(self):
        answer = f"{_BODY}\n\n---\n{SOURCES_HEADING}\n- carrier_policy/route_hnd_osa.md"
        released = _released({"validated_answer": answer, "validation_passed": True})
        assert SCOPE_NOTE in released

    def test_a_scope_redirect_is_released_without_citations(self):
        redirect = "This question is outside the logistics policy corpus."
        assert _released({"validated_answer": redirect, "validation_passed": False}) == redirect

    def test_the_contract_helper_reports_each_failure_kind(self):
        assert enforce_output_contract(_BODY)[1] == "missing_sources"
        repaired, finding = enforce_output_contract(f"{_BODY}\n\n---\n{SOURCES_HEADING}\n- carrier_policy/a.md")
        assert finding == "missing_scope_note" and SCOPE_NOTE in repaired
        assert enforce_output_contract(_ANSWER) == (_ANSWER, None)


class TestIdentifiersSurviveTheBoundary:
    """No numeric rounding grid runs here, so the identifier forms this domain
    quotes must come out exactly as they went in."""

    @pytest.mark.parametrize(
        "identifier",
        [
            "HND-OSA",
            "TYO-NGO",
            "MSKU 4512345",
            "REF-2026-0041",
            "LOG-SHIP-20260712-001",
            "UN 1263",
            "JPY 1,234",
            "17:00",
            "30 kg",
            "1200pcs",
            "6109.10",
        ],
    )
    def test_identifier_is_byte_identical(self, identifier):
        answer = f"**Logistics policy answer**\n\nThe governing reference is {identifier}." + _FOOTER
        assert identifier in _released({"validated_answer": answer, "validation_passed": True})

    def test_a_numbered_section_after_a_three_letter_code_keeps_its_number(self):
        """A currency-style grammar with a whitespace-spanning delimiter would
        rewrite this heading; nothing here may."""
        answer = (
            "**Logistics policy answer**\n\nRoute code: HND\n\n3. Exception handling\n\n"
            "Raise the exception the same working day." + _FOOTER
        )
        released = _released({"validated_answer": answer, "validation_passed": True})
        assert "3. Exception handling" in released
