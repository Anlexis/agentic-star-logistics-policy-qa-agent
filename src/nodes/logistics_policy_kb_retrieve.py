"""AgentCore Platform v1.0"""

# Node contract:
#   - extend FunctionNode; implement execute(state, config) -> dict
#   - return ONLY the fields this node changes (never the full state)
#   - return AgentStatus enum constants — never plain strings
#   - read inputs via state.get(...) — read-only
#   - never import from other agents
#
# LOG-C2-029 — LogisticsPolicyKBRetrieveNode
# Second node of the inner workflow: find the policy passages that answer the
# question.
#
# Where the corpus comes from:
#   * the caller may send its own policy documents with the request — those are
#     validated at the entry boundary and searched here;
#   * with no documents sent, the template's bundled reference corpus is
#     searched instead, so a fresh checkout answers real questions.
# Either way the same deterministic relevance search runs (PolicyCorpusService)
# and the passages, scores and citations returned are computed from the text
# actually supplied.
#
# Two outcomes have no answer to generate:
#   * the question asks for live shipment data, which policy text cannot supply;
#   * nothing in the corpus clears the relevance threshold.
# Both set out_of_scope=True with a SUCCESS status — the run behaved correctly,
# the question simply is not answerable from policy text.

import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import RETRIEVAL_FAILED

from src.schemas.state import (
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    SCORE_THRESHOLD_MAX,
    SCORE_THRESHOLD_MIN,
    TOP_K_MAX,
    TOP_K_MIN,
    finite_in_range,
    from_json,
    to_json,
)
from src.services.service import PolicyCorpusService, is_live_data_request

logger = logging.getLogger(__name__)


class LogisticsPolicyKBRetrieveNode(FunctionNode):
    """Retrieve the policy passages relevant to the question.

    Trust: the caller's trust level is decided at the agent's entry boundary
    (PreProcessNode); the inner workflow is unreachable except through it.

    Input state keys:
        normalized_query:  str — the retrieval key from QueryNormalizeNode
        extracted_context: str — JSON-serialised {carrier, route, shipment_class}
        validated_context: str — JSON-serialised validated caller options
                                 (policy_documents, top_k, score_threshold)

    Output state keys (partial dict):
        retrieved_chunks:    str  — JSON-serialised ranked passages
        out_of_scope:        bool — True on a live-data question or an empty result
        out_of_scope_reason: str  — "live_data_request" | "no_kb_match"
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

        emit_progress("Searching the knowledge base...")
        normalized_query: Any = state.get("normalized_query", "")
        if not isinstance(normalized_query, str) or not normalized_query.strip():
            emit_progress(RETRIEVAL_FAILED)
            return {
                "retrieved_chunks": to_json([]),
                "out_of_scope": True,
                "out_of_scope_reason": "no_kb_match",
                "status": AgentStatus.ERROR.value,
                "error_log": ["LogisticsPolicyKBRetrieveNode: no normalised question to search with."],
            }
        normalized_query = normalized_query.strip()

        # A live-data question is settled before any search: policy text cannot
        # answer it, and a partial match would look like an answer.
        if is_live_data_request(normalized_query):
            emit_trace_event(
                "kb_retrieve_out_of_scope",
                {"reason": "live_data_request"},
                state,
            )
            return {
                "retrieved_chunks": to_json([]),
                "out_of_scope": True,
                "out_of_scope_reason": "live_data_request",
                "status": AgentStatus.SUCCESS.value,
            }

        validated: Dict[str, Any] = from_json(state.get("validated_context"), {}) or {}
        context: Dict[str, Any] = from_json(state.get("extracted_context"), {}) or {}

        # The entry boundary already bounded these; re-parsing here keeps the
        # node correct when it is exercised on its own, and never widens a bound.
        top_k_value = finite_in_range(validated.get("top_k", DEFAULT_TOP_K), TOP_K_MIN, TOP_K_MAX)
        top_k = int(top_k_value) if top_k_value is not None else DEFAULT_TOP_K
        threshold = finite_in_range(
            validated.get("score_threshold", DEFAULT_SCORE_THRESHOLD),
            SCORE_THRESHOLD_MIN,
            SCORE_THRESHOLD_MAX,
        )
        score_threshold = threshold if threshold is not None else DEFAULT_SCORE_THRESHOLD

        documents = validated.get("policy_documents") or None
        chunks: List[Dict[str, Any]] = PolicyCorpusService().search(
            query=normalized_query,
            documents=documents,
            context=context,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        if not chunks:
            logger.info(
                "LogisticsPolicyKBRetrieveNode: no passage cleared the threshold " "(corpus=%s, top_k=%d)",
                "caller" if documents else "bundled",
                top_k,
            )
            emit_trace_event(
                "kb_retrieve_out_of_scope",
                {"reason": "no_kb_match", "corpus": "caller" if documents else "bundled", "top_k": top_k},
                state,
            )
            return {
                "retrieved_chunks": to_json([]),
                "out_of_scope": True,
                "out_of_scope_reason": "no_kb_match",
                "status": AgentStatus.SUCCESS.value,
            }

        logger.info(
            "LogisticsPolicyKBRetrieveNode: %d passage(s) retrieved (corpus=%s, top_k=%d)",
            len(chunks),
            "caller" if documents else "bundled",
            top_k,
        )
        emit_trace_event(
            "kb_retrieve_complete",
            {
                "chunk_count": len(chunks),
                "corpus": "caller" if documents else "bundled",
                "top_k": top_k,
                "top_score": chunks[0]["score"],
            },
            state,
        )

        return {
            "retrieved_chunks": to_json(chunks),
            "out_of_scope": False,
            "status": AgentStatus.SUCCESS.value,
        }
