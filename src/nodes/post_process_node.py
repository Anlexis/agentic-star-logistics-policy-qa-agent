"""AgentCore Platform v1.0"""

# LOG-C2-029 — PostProcessNode
# Outer backbone post_process slot: the external-output boundary. Whatever this
# node returns is what leaves the agent, so the published output contract is
# enforced HERE rather than trusted from the step that composed the answer.
#
# Four layers, in this order:
#
#   (1) echoed-question redaction — the answer is a rendering of policy
#       passages, never a copy of what the caller typed. A verbatim reappearance
#       of the caller's question as a whole phrase is replaced with [REDACTED]
#       plus an audit event;
#   (2) restricted-content scan — API keys, JWTs, bearer tokens, credential
#       assignments, national-ID and payment-card forms anywhere in the payload
#       withhold the WHOLE answer (sanitised stub, ERROR status). The scan
#       recurses into the citation list and any nested structure, so a
#       violation cannot hide below the top-level string;
#   (3) output contract — every answer this agent emits carries the sources it
#       came from and the standing scope note. A payload that lost its sources
#       is withheld (they cannot be invented); a payload that lost only the
#       scope note has it restored, with an audit event either way;
#   (4) the layer-2 scan repeated, so the final bytes leaving the agent are
#       never an unscanned surface.
#
# Why there is no numeric rounding grid here. Some templates render monetary
# aggregates and snap them onto a rounding grid on the way out. This one emits
# no monetary aggregate at all — it answers policy questions in the words of
# the policy — and the grammar such a grid uses reads any standalone
# three-letter uppercase word as a currency marker. In this domain those words
# are the content: route codes (HND-OSA), carrier and port codes, container and
# consignment references. A grid here would rewrite the identifiers the answer
# exists to quote, so the invariant enforced below is this template's own —
# cited, scope-marked, free of restricted content — and identifiers pass
# through byte-identical.

import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, OUTPUT_BLOCKED, TOO_LONG

from src.nodes.answer_generate import SCOPE_NOTE, SOURCES_HEADING
from src.schemas.state import from_json
from src.services.service import scan_for_restricted_content

logger = logging.getLogger(__name__)

# State fields holding the caller's own words. None of them belongs in the
# answer: the answer is composed from retrieved policy passages.
_CALLER_TEXT_FIELDS = ("user_input", "validated_input")

# A caller's text counts as "echoed" when the whole of it reappears as a
# PHRASE — at least this many characters and containing a space. A single term
# ("packaging", "cold chain") is corpus vocabulary and will legitimately appear
# in a policy passage; redacting it would delete the answer's own subject.
_ECHO_MIN_CHARS = 20

_REDACTION = "[REDACTED]"

_WITHHELD_STUB = (
    "[This answer was withheld at the output boundary: it did not satisfy the "
    "output contract. Review the policy sources supplied with the request and retry.]"
)


def _redact_echoed_question(text: str, state: Dict[str, Any]) -> tuple[str, List[str]]:
    """Replace any verbatim echo of the caller's own text. Returns (text, fields)."""
    redacted: List[str] = []
    for field in _CALLER_TEXT_FIELDS:
        value = state.get(field)
        if not isinstance(value, str):
            continue
        phrase = value.strip()
        if len(phrase) < _ECHO_MIN_CHARS or " " not in phrase:
            continue
        if phrase in text:
            text = text.replace(phrase, _REDACTION)
            redacted.append(field)
    return text, redacted


def enforce_output_contract(text: str) -> tuple[str, Optional[str]]:
    """Enforce the published answer contract. Returns (text, finding).

    A finding of "missing_sources" means the answer cannot be repaired and must
    be withheld — citations cannot be invented. "missing_scope_note" is
    repaired in place and reported so the repair is audited.
    """
    heading_at = text.find(SOURCES_HEADING)
    if heading_at < 0:
        return text, "missing_sources"
    tail = text[heading_at + len(SOURCES_HEADING) :]
    if not any(line.strip().startswith("-") and line.strip(" -") for line in tail.splitlines()):
        return text, "missing_sources"
    if SCOPE_NOTE not in text:
        return f"{text}\n\n*{SCOPE_NOTE}*", "missing_scope_note"
    return text, None


# Caller-facing wording for a run that completed without an answer. The marker
# is an internal reason code; this maps it to the sentence the caller sees.
# Static sentences only - no request value is ever substituted, so nothing the
# caller sent can be reflected back through this path.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Enforce the output contract on the bytes that leave the agent.

    Input state keys:
        result:            the answer composed by the workflow
        validated_answer:  the answer after the workflow's own checks
        validation_passed: False for a scope redirect, which carries no citation
        citations:         JSON-serialised source list, scanned with the answer
        user_input /
        validated_input:   the caller's own text, redacted if echoed back

    Output state keys (partial dict):
        formatted_output: str — the final answer, redirect text, or withheld stub
        status:           AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:        list[str] (on withhold only)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        emit_progress("Finalising the response...")

        # The run completed without an answer because the request could not be
        # accepted as written. Report the reason as the response: the caller
        # needs to know what to change, and an empty body would leave them with
        # nothing. Status stays SUCCESS - the run did what it could with the
        # request it was given, and the caller can correct it and send again on
        # the same conversation.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "formatted_output": message,
                "result": message,
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
            }
        answer = state.get("validated_answer") or state.get("result") or ""
        if not isinstance(answer, str):
            answer = ""
        citations = from_json(state.get("citations"), []) or []

        if not answer.strip():
            emit_trace_event("output_withheld", {"reason": "empty_answer"}, state)
            emit_progress(OUTPUT_BLOCKED)
            return {
                "formatted_output": _WITHHELD_STUB,
                "status": AgentStatus.ERROR.value,
                "error_log": ["PostProcessNode: no answer reached the output boundary."],
            }

        # -- 1. echoed-question redaction --------------------------------
        answer, redacted_fields = _redact_echoed_question(answer, state)
        if redacted_fields:
            logger.warning("PostProcessNode: redacted echoed caller text (%s)", ",".join(redacted_fields))
            emit_trace_event("output_caller_text_redacted", {"fields": redacted_fields}, state)

        # -- 2. restricted-content scan (answer + citations) --------------
        finding = scan_for_restricted_content({"answer": answer, "citations": citations})
        if finding:
            return self._withhold(finding, state)

        # -- 3. output contract ------------------------------------------
        # A scope redirect makes no policy claim and cites nothing, so the
        # citation rule does not apply to it. That path is recognised from the
        # workflow's own flag, never guessed at from the text.
        is_redirect = state.get("validation_passed") is False
        if not is_redirect:
            answer, contract_finding = enforce_output_contract(answer)
            if contract_finding == "missing_sources":
                return self._withhold("missing_sources", state)
            if contract_finding:
                logger.warning("PostProcessNode: restored the scope note on the answer")
                emit_trace_event(
                    "output_contract_repaired",
                    {"repair": contract_finding, "answer_chars": len(answer)},
                    state,
                )

        # -- 4. re-scan the final bytes -----------------------------------
        finding = scan_for_restricted_content(answer)
        if finding:
            return self._withhold(finding, state)

        emit_trace_event(
            "output_released",
            {"answer_chars": len(answer), "citation_count": len(citations), "redirect": is_redirect},
            state,
        )
        return {"formatted_output": answer, "status": AgentStatus.SUCCESS.value}

    def _withhold(self, finding: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Withhold the whole answer. The finding names the rule, never the value."""
        logger.error("PostProcessNode: answer withheld at the output boundary (%s)", finding)
        emit_trace_event("output_withheld", {"reason": finding}, state)
        emit_progress(OUTPUT_BLOCKED)
        return {
            "formatted_output": _WITHHELD_STUB,
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PostProcessNode: the answer was withheld at the output boundary ({finding})."],
        }
