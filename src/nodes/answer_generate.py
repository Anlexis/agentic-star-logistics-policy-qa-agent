"""AgentCore Platform v1.0"""

# Node contract:
#   - extend FunctionNode; implement execute(state, config) -> dict
#   - return ONLY the fields this node changes (never the full state)
#   - return AgentStatus enum constants — never plain strings
#   - never import from other agents
#
# LOG-C2-029 — AnswerGenerateNode
# Third node of the inner workflow: turn the retrieved passages into the answer.
#
# Two paths:
#   (a) out_of_scope is already set — pass through, generate nothing;
#   (b) otherwise, compose the answer from the retrieved passages.
#
# Composition is deterministic: the answer is the retrieved policy text, the
# sources it came from and the date those sources were last verified. Nothing
# is added that the corpus did not say, so the answer can be checked against
# its citations line by line. Wire a model into _compose_answer() to phrase the
# same material in prose; the citation and scope-note rules below are what the
# output boundary enforces either way.
#
# Citations are MANDATORY. If passages were retrieved but none of them names a
# source, the run fails rather than emitting an uncited policy statement.
#
# The question is NOT echoed back into the answer: the caller's own text
# reappearing verbatim in the output is caller-controlled content on an
# external surface, and the output boundary redacts it.

import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import PROCESSING_FAILED

from src.schemas.state import (
    DEFAULT_MAX_ANSWER_CHARS,
    MAX_ANSWER_CHARS_MAX,
    MAX_ANSWER_CHARS_MIN,
    finite_in_range,
    from_json,
    to_json,
)

logger = logging.getLogger(__name__)

# The output contract this template publishes. Both strings are enforced at the
# external boundary (src/nodes/post_process_node.py), so a change in how the
# answer is composed cannot silently drop either of them.
SOURCES_HEADING = "**Sources**"
SCOPE_NOTE = (
    "Answered from static logistics policy text. It does not reflect the live "
    "status, location or estimated arrival of any shipment."
)


def _parse_chunks(raw: Any) -> List[Dict[str, Any]]:
    """Coerce the retrieved passages to a list of dicts."""
    value = from_json(raw, [])
    if isinstance(value, list):
        return [chunk for chunk in value if isinstance(chunk, dict)]
    return []


def _extract_citations(chunks: List[Dict[str, Any]]) -> List[str]:
    """Return the deduplicated source identifiers of *chunks*, in rank order."""
    seen: set[str] = set()
    citations: List[str] = []
    for chunk in chunks:
        source = str(chunk.get("source", "")).strip()
        if source and source not in seen:
            seen.add(source)
            citations.append(source)
    return citations


def _extract_last_verified(chunks: List[Dict[str, Any]]) -> str:
    """Return the most recent verification date across *chunks*.

    ISO dates compare correctly as strings, so the maximum is the latest.
    """
    dates = [str(chunk.get("last_verified", "")).strip() for chunk in chunks]
    dates = [date for date in dates if date]
    return max(dates) if dates else ""


def _compose_answer(
    chunks: List[Dict[str, Any]],
    citations: List[str],
    last_verified: str,
    max_chars: int,
) -> str:
    """Build the answer from the retrieved passages.

    The body is the retrieved policy text under its document title; the footer
    carries the sources, the verification date and the scope note. The body is
    trimmed — never the footer — when the caller's length ceiling is reached,
    so an answer can lose detail but never its citations.
    """
    blocks: List[str] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        title = str(chunk.get("title", "")).strip()
        blocks.append(f"**{title}**\n\n{text}" if title else text)

    body = (
        "\n\n".join(blocks)
        if blocks
        else (
            "The retrieved policy sources carry no readable text for this question. "
            "Consult the logistics compliance owner for the governing policy."
        )
    )

    citation_lines = "\n".join(f"- {citation}" for citation in citations)
    verified_note = f"Last verified: {last_verified}" if last_verified else "Last verified: unknown"
    footer = f"\n\n---\n{SOURCES_HEADING}\n{citation_lines}\n\n*{verified_note}*\n\n*{SCOPE_NOTE}*"

    header = "**Logistics policy answer**\n\n"
    budget = max_chars - len(header) - len(footer)
    if budget < 0:
        # The footer alone exceeds the ceiling: keep the contract, drop the body.
        return header + footer
    if len(body) > budget:
        body = body[:budget].rstrip()
    return header + body + footer


class AnswerGenerateNode(FunctionNode):
    """Compose the policy answer from the retrieved passages.

    Trust: the caller's trust level is decided at the agent's entry boundary
    (PreProcessNode); the inner workflow is unreachable except through it.

    Input state keys:
        out_of_scope:      bool — True routes to the pass-through path
        retrieved_chunks:  str  — JSON-serialised ranked passages
        validated_context: str  — JSON-serialised validated caller options
                                  (max_answer_chars)

    Output state keys (partial dict):
        answer:        str — the composed answer (normal path only)
        result:        str — the same text; the output boundary reads result
        citations:     str — JSON-serialised source list (normal path only)
        last_verified: str — most recent source verification date
        status:        AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:     list[str] (on error only)
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

        emit_progress("Composing the answer...")
        # -- Path (a): nothing to answer from -----------------------------
        if state.get("out_of_scope"):
            logger.info("AnswerGenerateNode: out of scope — nothing generated")
            emit_trace_event(
                "answer_generate_complete",
                {"out_of_scope": True, "citation_count": 0},
                state,
            )
            return {"status": AgentStatus.SUCCESS.value}

        # -- Path (b): compose from the retrieved passages ----------------
        chunks = _parse_chunks(state.get("retrieved_chunks"))
        if not chunks:
            emit_progress(PROCESSING_FAILED)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["AnswerGenerateNode: no retrieved passage to answer from."],
            }

        citations = _extract_citations(chunks)
        if not citations:
            # An answer without citations is a policy claim nobody can check.
            logger.error("AnswerGenerateNode: retrieved passages name no source")
            emit_progress(PROCESSING_FAILED)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    "AnswerGenerateNode: the retrieved passages name no source, so no "
                    "citable answer can be produced."
                ],
            }

        validated: Dict[str, Any] = from_json(state.get("validated_context"), {}) or {}
        ceiling = finite_in_range(
            validated.get("max_answer_chars", DEFAULT_MAX_ANSWER_CHARS),
            MAX_ANSWER_CHARS_MIN,
            MAX_ANSWER_CHARS_MAX,
        )
        max_chars = int(ceiling) if ceiling is not None else DEFAULT_MAX_ANSWER_CHARS

        last_verified = _extract_last_verified(chunks)
        answer_text = _compose_answer(chunks, citations, last_verified, max_chars)

        logger.info(
            "AnswerGenerateNode: answer composed — passages=%d citations=%d chars=%d",
            len(chunks),
            len(citations),
            len(answer_text),
        )
        emit_trace_event(
            "answer_generate_complete",
            {
                "citation_count": len(citations),
                "chunk_count": len(chunks),
                "last_verified": last_verified,
                "answer_chars": len(answer_text),
            },
            state,
        )

        return {
            "answer": answer_text,
            "result": answer_text,
            "citations": to_json(citations),
            "last_verified": last_verified,
            "status": AgentStatus.SUCCESS.value,
        }
