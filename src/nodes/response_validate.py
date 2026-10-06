"""AgentCore Platform v1.0"""

# Node contract:
#   - extend FunctionNode; implement execute(state, config) -> dict
#   - return ONLY the fields this node changes (never the full state)
#   - return AgentStatus enum constants — never plain strings
#   - never import from other agents
#
# LOG-C2-029 — ResponseValidateNode
# Last node of the inner workflow: check the composed answer before it leaves
# the workflow for the agent's external boundary.
#
# Checks, in order:
#   1. scope pass-through — a question already settled as out of scope gets the
#      redirect text, not a guessed answer;
#   2. live-data backstop — an answer BODY that asserts live shipment status is
#      withheld and replaced with the redirect text. Only the body is screened:
#      the standing scope note in the footer says the words "live status" by
#      design, and screening it would scope out every answer the agent gives;
#   3. non-empty;
#   4. length ceiling (the caller's, within the published bounds);
#   5. restricted-content scan (shared with the external boundary).

import logging
from typing import Any, ClassVar, Dict, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, OUTPUT_BLOCKED, TOO_LONG

from src.schemas.state import (
    DEFAULT_MAX_ANSWER_CHARS,
    MAX_ANSWER_CHARS_MAX,
    MAX_ANSWER_CHARS_MIN,
    finite_in_range,
    from_json,
)
from src.services.service import is_live_data_request, scan_for_restricted_content

logger = logging.getLogger(__name__)

# Separator between the answer body and the standing footer (sources,
# verification date, scope note) written by AnswerGenerateNode.
_FOOTER_SEPARATOR = "\n\n---\n"

# Returned when the answer itself asserted live shipment data.
_LIVE_DATA_REDIRECT = (
    "This question needs live shipment data — current status, location or estimated "
    "arrival — which a static policy corpus cannot answer reliably. Use a shipment "
    "tracking service for live data; this agent answers what the policies say."
)

# Returned when an earlier step already settled the question as out of scope.
_OUT_OF_SCOPE_REDIRECT = (
    "This question is outside the logistics policy corpus. For live shipment "
    "tracking, estimated arrival or other real-time operational data, use a "
    "shipment tracking service."
)


def _answer_body(text: str) -> str:
    """Return the part of the answer above the standing footer."""
    return text.split(_FOOTER_SEPARATOR, 1)[0]


class ResponseValidateNode(FunctionNode):
    """Check the composed answer before it leaves the inner workflow.

    Trust: the caller's trust level is decided at the agent's entry boundary
    (PreProcessNode); the inner workflow is unreachable except through it.

    Input state keys:
        result / answer:     the composed answer
        out_of_scope:        (optional) set by an earlier step
        out_of_scope_reason: (optional) reason code from that step
        validated_context:   JSON-serialised validated caller options

    Output state keys (partial dict):
        validated_answer:    str  — the answer, or the redirect text
        validation_passed:   bool — True only when every check passed
        out_of_scope:        bool — set here on the live-data backstop
        out_of_scope_reason: str  — reason code when set here
        status:              AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:           list[str] (on error only)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # The request was already found unacceptable upstream: this run
        # completes without a result, so there is nothing for this step to
        # do. Returning the marker keeps it on the node's own result dict,
        # which is what the output gate inspects.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}

        emit_progress("Checking the request...")
        validated: Dict[str, Any] = from_json(state.get("validated_context"), {}) or {}
        ceiling = finite_in_range(
            validated.get("max_answer_chars", DEFAULT_MAX_ANSWER_CHARS),
            MAX_ANSWER_CHARS_MIN,
            MAX_ANSWER_CHARS_MAX,
        )
        max_chars = int(ceiling) if ceiling is not None else DEFAULT_MAX_ANSWER_CHARS

        # -- 1. scope pass-through ---------------------------------------
        if state.get("out_of_scope"):
            logger.info(
                "ResponseValidateNode: out of scope (%s) — returning the redirect text",
                state.get("out_of_scope_reason"),
            )
            emit_trace_event(
                "response_validate_complete",
                {
                    "branch": "scope_passthrough",
                    "reason": state.get("out_of_scope_reason"),
                    "validation_passed": False,
                },
                state,
            )
            return {
                "validated_answer": _OUT_OF_SCOPE_REDIRECT,
                "validation_passed": False,
                "status": AgentStatus.SUCCESS.value,
            }

        answer = state.get("result") or state.get("answer") or ""
        if not isinstance(answer, str):
            answer = ""

        # -- 2. live-data backstop ---------------------------------------
        if is_live_data_request(_answer_body(answer)):
            logger.warning("ResponseValidateNode: the answer body asserts live shipment data")
            emit_trace_event(
                "response_validate_complete",
                {
                    "branch": "live_data_scope",
                    "out_of_scope_reason": "live_data_request",
                    "validation_passed": False,
                },
                state,
            )
            return {
                "out_of_scope": True,
                "out_of_scope_reason": "live_data_request",
                "validated_answer": _LIVE_DATA_REDIRECT,
                "validation_passed": False,
                "status": AgentStatus.SUCCESS.value,
            }

        # -- 3. non-empty ------------------------------------------------
        if not answer.strip():
            logger.warning("ResponseValidateNode: the answer is empty")
            emit_trace_event(
                "response_validate_complete",
                {"branch": "empty_result", "validation_passed": False},
                state,
            )
            emit_progress(EMPTY_INPUT)
            return {
                "validation_passed": False,
                "status": AgentStatus.ERROR.value,
                "error_log": ["ResponseValidateNode: the answer is empty."],
            }

        # -- 4. length ceiling -------------------------------------------
        if len(answer) > max_chars:
            logger.warning(
                "ResponseValidateNode: the answer is %d characters, over the %d ceiling",
                len(answer),
                max_chars,
            )
            emit_trace_event(
                "response_validate_complete",
                {
                    "branch": "length_exceeded",
                    "answer_chars": len(answer),
                    "max_chars": max_chars,
                    "validation_passed": False,
                },
                state,
            )
            emit_progress(TOO_LONG)
            return {
                "validation_passed": False,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"ResponseValidateNode: the answer exceeds the {max_chars}-character ceiling."],
            }

        # -- 5. restricted content ---------------------------------------
        violation = scan_for_restricted_content(answer)
        if violation:
            logger.error("ResponseValidateNode: answer withheld — restricted content (%s)", violation)
            emit_trace_event(
                "response_validate_complete",
                {"branch": "restricted_content", "finding": violation, "validation_passed": False},
                state,
            )
            emit_progress(OUTPUT_BLOCKED)
            return {
                "validation_passed": False,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"ResponseValidateNode: the answer was withheld — restricted content ({violation})."],
            }

        logger.info("ResponseValidateNode: answer accepted (%d characters)", len(answer))
        emit_trace_event(
            "response_validate_complete",
            {"branch": "pass", "answer_chars": len(answer), "validation_passed": True},
            state,
        )
        return {
            "validated_answer": answer,
            "validation_passed": True,
            "status": AgentStatus.SUCCESS.value,
        }
