"""AgentCore Platform v1.0"""

# LOG-C2-029 — PreProcessNode
# Outer backbone pre_process slot: the caller-data contract for this agent.
#
# Everything a caller can send arrives here and nowhere else:
#
#   input          the natural-language policy question
#   input_context  the per-invocation options object:
#                    policy_documents  the policy corpus to answer FROM
#                    carrier / route_code / shipment_class  retrieval scoping
#                    top_k / score_threshold / max_answer_chars  retrieval controls
#                    channel  the calling surface, recorded for audit
#
# Contract rules, all enforced below:
#   * every caller-controlled NUMBER is parsed by a finite, bounded parser —
#     "NaN" and "Infinity" parse through float() and arrive intact through raw
#     JSON, and every comparison against NaN is False, so an unchecked value
#     would silently disable the control it was meant to set;
#   * every caller-controlled string that SELECTS behaviour (carrier,
#     shipment class, channel, document id) is locked to an inert identifier —
#     free text in such a field is caller-controlled output and log injection;
#   * caller free text that is genuinely free text (the question, document
#     titles and bodies) is length-capped and stripped of control characters,
#     and refused outright when it carries an instruction-override directive;
#   * a rejected value is NEVER echoed back — the error names the field only;
#   * anything invalid fails CLOSED: the run stops here with an error status
#     rather than continuing on a partially trusted context.
#
# The instruction-override refusal is this template's OWN guarantee. The
# platform screens hostile input ahead of execute() as well, but a template
# whose only defence is the platform's returns a normal answer wherever that
# screen is absent or configured off. The check below runs inside execute(),
# so calling execute() directly still refuses.

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import INVALID_VALUE

from src.schemas.state import (
    DEFAULT_MAX_ANSWER_CHARS,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    MAX_ANSWER_CHARS_MAX,
    MAX_ANSWER_CHARS_MIN,
    MAX_DOCUMENT_SOURCE_CHARS,
    MAX_DOCUMENT_TEXT_CHARS,
    MAX_DOCUMENT_TITLE_CHARS,
    MAX_POLICY_DOCUMENTS,
    MAX_QUERY_CHARS,
    SCORE_THRESHOLD_MAX,
    SCORE_THRESHOLD_MIN,
    TOP_K_MAX,
    TOP_K_MIN,
    finite_in_range,
    to_json,
)

logger = logging.getLogger(__name__)

# Caller strings that select behaviour must be inert identifiers — lowercase
# alphanumerics/underscore, bounded length.
_INERT_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]{1,32}$")

# Route codes are the airport/port-pair form used across carrier policy
# documents: three uppercase letters, a hyphen, three uppercase letters.
_ROUTE_CODE_RE = re.compile(r"^[A-Z]{3}-[A-Z]{3}$")

# A document source is a corpus path/citation label, not free prose.
_SOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/#-]{0,119}$")

# ISO calendar date, the only accepted verification-date form.
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Control characters are stripped from every free-text field. Tab and newline
# survive — policy text is paragraph text — everything else is removed rather
# than rendered into an answer.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Instruction-override directives. Every alternative is anchored on a full
# phrase, not a lone verb: real logistics policy prose says "act as agent of
# record" and "override the standard route" in ordinary sentences, and a screen
# that fires on those refuses genuine work — the failure direction that
# actually blocks users.
_INSTRUCTION_OVERRIDE_RE = re.compile(
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instruction|instructions|prompt|prompts|rule|rules|direction|directions)"
    r"|disregard\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instruction|instructions|prompt|prompts|rule|rules)"
    r"|forget\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instruction|instructions|prompt|prompts)"
    r"|(?:reveal|show|print|repeat|output|disclose)\s+(?:me\s+)?(?:your|the)\s+"
    r"(?:system\s+prompt|system\s+message|initial\s+prompt|hidden\s+instructions)"
    r"|you\s+are\s+now\s+(?:a|an)\s"
    r"|act\s+as\s+(?:if\s+you\s+are\s+)?(?:a\s+|an\s+)?(?:developer|admin|root)\s+mode"
    r"|override\s+(?:your|the)\s+(?:instruction|instructions|rules|safety)",
    re.IGNORECASE,
)


def _sanitise_free_text(value: str, limit: int) -> str:
    """Strip control characters and cap the length of a free-text field."""
    return _CONTROL_CHARS_RE.sub("", value)[:limit].strip()


def _inert(value: Any, field: str) -> Tuple[Optional[str], Optional[str]]:
    """Validate an inert-identifier field. Returns (value, error)."""
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, f"PreProcessNode: {field} must be a short lowercase identifier."
    candidate = value.strip().lower()
    if not candidate:
        return None, None
    if not _INERT_IDENTIFIER_RE.match(candidate):
        return None, f"PreProcessNode: {field} must be a short lowercase identifier."
    return candidate, None


def _route_code(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Validate the route code. Returns (value, error)."""
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, "PreProcessNode: input_context.route_code must look like 'HND-OSA'."
    candidate = value.strip().upper()
    if not candidate:
        return None, None
    if not _ROUTE_CODE_RE.match(candidate):
        return None, "PreProcessNode: input_context.route_code must look like 'HND-OSA'."
    return candidate, None


def _bounded_int(value: Any, field: str, lo: int, hi: int, default: int) -> Tuple[int, Optional[str]]:
    """Validate a caller integer through the finite+bounded parser."""
    if value is None:
        return default, None
    parsed = finite_in_range(value, lo, hi)
    if parsed is None or parsed != int(parsed):
        return default, f"PreProcessNode: {field} must be a whole number between {lo} and {hi}."
    return int(parsed), None


def _bounded_float(value: Any, field: str, lo: float, hi: float, default: float) -> Tuple[float, Optional[str]]:
    """Validate a caller float through the finite+bounded parser."""
    if value is None:
        return default, None
    parsed = finite_in_range(value, lo, hi)
    if parsed is None:
        return default, f"PreProcessNode: {field} must be a number between {lo} and {hi}."
    return parsed, None


def _validate_document(raw: Any, index: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate one caller-supplied policy document. Returns (document, error).

    The position in the list is named in any error, never the content.
    """
    field = f"input_context.policy_documents[{index}]"
    if not isinstance(raw, dict):
        return None, f"PreProcessNode: {field} must be an object."

    document_id, err = _inert(raw.get("document_id"), f"{field}.document_id")
    if err:
        return None, err
    if not document_id:
        return None, f"PreProcessNode: {field}.document_id is required."

    source = raw.get("source")
    if not isinstance(source, str) or not _SOURCE_RE.match(source.strip()):
        return None, (
            f"PreProcessNode: {field}.source is required and must be a citation "
            f"label of at most {MAX_DOCUMENT_SOURCE_CHARS} characters."
        )

    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, f"PreProcessNode: {field}.text is required."
    if len(text) > MAX_DOCUMENT_TEXT_CHARS:
        return None, (f"PreProcessNode: {field}.text exceeds the " f"{MAX_DOCUMENT_TEXT_CHARS}-character limit.")
    if _INSTRUCTION_OVERRIDE_RE.search(text):
        return None, f"PreProcessNode: {field}.text refused — instruction-override content."

    title = raw.get("title")
    if title is not None and not isinstance(title, str):
        return None, f"PreProcessNode: {field}.title must be text."
    if isinstance(title, str) and _INSTRUCTION_OVERRIDE_RE.search(title):
        return None, f"PreProcessNode: {field}.title refused — instruction-override content."

    last_verified = raw.get("last_verified")
    if last_verified is not None:
        if not isinstance(last_verified, str) or not _ISO_DATE_RE.match(last_verified.strip()):
            return None, f"PreProcessNode: {field}.last_verified must be an ISO date (YYYY-MM-DD)."
        last_verified = last_verified.strip()

    return (
        {
            "document_id": document_id,
            "source": source.strip(),
            "title": _sanitise_free_text(title, MAX_DOCUMENT_TITLE_CHARS) if isinstance(title, str) else "",
            "text": _sanitise_free_text(text, MAX_DOCUMENT_TEXT_CHARS),
            "last_verified": last_verified or "",
        },
        None,
    )


def _validate_documents(raw: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Validate the caller's policy corpus. Returns (documents, error)."""
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return [], "PreProcessNode: input_context.policy_documents must be a list."
    if len(raw) > MAX_POLICY_DOCUMENTS:
        return [], (
            f"PreProcessNode: input_context.policy_documents accepts at most " f"{MAX_POLICY_DOCUMENTS} entries."
        )
    documents: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        document, err = _validate_document(entry, index)
        if err or document is None:
            return [], err or f"PreProcessNode: input_context.policy_documents[{index}] is invalid."
        if document["document_id"] in seen:
            return [], (f"PreProcessNode: input_context.policy_documents[{index}].document_id " f"is a duplicate.")
        seen.add(document["document_id"])
        documents.append(document)
    return documents, None


class PreProcessNode(FunctionNode):
    """Validate the question and the caller options before any domain work.

    Input state keys:
        user_input:    str  — the policy question (required)
        input_context: dict — per-invocation options (all fields optional)

    Output state keys (partial dict):
        validated_input:   str  — the sanitised question
        validated_context: str  — JSON-serialised, fully validated options
        enriched_context:  dict — channel/source metadata for downstream audit
        status:            AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:         list[str] — field-naming errors, on ERROR only
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        emit_progress("Checking the request...")
        raw_query: Any = state.get("user_input", "")
        raw_context: Any = state.get("input_context", {})

        if not isinstance(raw_query, str):
            return self._reject(["PreProcessNode: the question must be text."], state)
        query = _sanitise_free_text(raw_query, MAX_QUERY_CHARS)
        if not query:
            return self._reject(["PreProcessNode: the question is empty."], state)
        if len(raw_query) > MAX_QUERY_CHARS:
            return self._reject(
                [f"PreProcessNode: the question exceeds the {MAX_QUERY_CHARS}-character limit."],
                state,
            )
        if _INSTRUCTION_OVERRIDE_RE.search(query):
            # Refused here, in the node that owns the caller contract, so the
            # refusal holds however the agent is hosted.
            return self._reject(
                ["PreProcessNode: the question was refused — instruction-override content."],
                state,
            )

        if raw_context is None:
            raw_context = {}
        if not isinstance(raw_context, dict):
            return self._reject(["PreProcessNode: input_context must be an object."], state)

        errors: List[str] = []

        carrier, err = _inert(raw_context.get("carrier"), "input_context.carrier")
        if err:
            errors.append(err)
        shipment_class, err = _inert(raw_context.get("shipment_class"), "input_context.shipment_class")
        if err:
            errors.append(err)
        channel, err = _inert(raw_context.get("channel"), "input_context.channel")
        if err:
            errors.append(err)
        route_code, err = _route_code(raw_context.get("route_code"))
        if err:
            errors.append(err)

        top_k, err = _bounded_int(raw_context.get("top_k"), "input_context.top_k", TOP_K_MIN, TOP_K_MAX, DEFAULT_TOP_K)
        if err:
            errors.append(err)
        score_threshold, err = _bounded_float(
            raw_context.get("score_threshold"),
            "input_context.score_threshold",
            SCORE_THRESHOLD_MIN,
            SCORE_THRESHOLD_MAX,
            DEFAULT_SCORE_THRESHOLD,
        )
        if err:
            errors.append(err)
        max_answer_chars, err = _bounded_int(
            raw_context.get("max_answer_chars"),
            "input_context.max_answer_chars",
            MAX_ANSWER_CHARS_MIN,
            MAX_ANSWER_CHARS_MAX,
            DEFAULT_MAX_ANSWER_CHARS,
        )
        if err:
            errors.append(err)

        documents, err = _validate_documents(raw_context.get("policy_documents"))
        if err:
            errors.append(err)

        if errors:
            return self._reject(errors, state)

        validated: Dict[str, Any] = {
            "carrier": carrier,
            "route_code": route_code,
            "shipment_class": shipment_class,
            "channel": channel or "unknown",
            "top_k": top_k,
            "score_threshold": score_threshold,
            "max_answer_chars": max_answer_chars,
            "policy_documents": documents,
        }

        emit_trace_event(
            "caller_contract_accepted",
            {
                "query_chars": len(query),
                "document_count": len(documents),
                "top_k": top_k,
                "channel": validated["channel"],
            },
            state,
        )
        logger.info(
            "PreProcessNode: accepted question (%d chars) with %d caller document(s)",
            len(query),
            len(documents),
        )

        return {
            "validated_input": query,
            "validated_context": to_json(validated),
            "enriched_context": {
                "source": "LogisticsPolicyQAAgent",
                "channel": validated["channel"],
                "document_count": len(documents),
            },
            "status": AgentStatus.SUCCESS.value,
        }

    def _reject(self, errors: List[str], state: Dict[str, Any]) -> Dict[str, Any]:
        """Fail closed. Errors name the offending field, never its value."""
        emit_trace_event(
            "caller_contract_rejected",
            {"error_count": len(errors), "fields": [e.split(":", 1)[-1].strip()[:60] for e in errors]},
            state,
        )
        logger.warning("PreProcessNode: declined caller input (%d finding(s))", len(errors))
        emit_progress(INVALID_VALUE)
        # The caller can correct these values and send the request again, so the
        # run completes carrying the reason rather than terminating and
        # surfacing only an exception type. Errors name the field, never a value.
        return {
            "status": AgentStatus.SUCCESS.value,
            "error_code": "INVALID_REQUEST",
            "error_log": errors,
        }
