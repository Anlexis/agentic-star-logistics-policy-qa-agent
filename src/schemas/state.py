"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a Pydantic BaseModel. LangGraph
# checkpoints use msgpack serialization; Pydantic objects cause silent
# corruption. Extend AgentState with agent-specific fields only. Do NOT add
# credentials, secrets, or Pydantic models.
#
# LOG-C2-029 — Logistics Policy Q&A Agent
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph). Fields below cover both layers.
#
# dict/list payloads are stored JSON-serialised (Optional[str]) so every value
# that reaches a checkpoint is a msgpack-safe primitive.
#
# This module also owns the caller-input bounds shared by the validation node,
# the retrieval node and the tests, so there is ONE definition of each limit.

import json
import math
from typing import Any, Optional

from framework.schemas.agent_state import AgentState

# ---------------------------------------------------------------------------
# Caller-input bounds — the published contract for POST /invoke input_context.
# Every bound is enforced by PreProcessNode and documented in docs/02_design.md.
# ---------------------------------------------------------------------------

# Raw question text.
MAX_QUERY_CHARS = 2_000

# Caller-supplied policy corpus.
MAX_POLICY_DOCUMENTS = 20
MAX_DOCUMENT_TEXT_CHARS = 8_000
MAX_DOCUMENT_TITLE_CHARS = 200
MAX_DOCUMENT_SOURCE_CHARS = 120

# Retrieval controls.
TOP_K_MIN = 1
TOP_K_MAX = 20
SCORE_THRESHOLD_MIN = 0.0
SCORE_THRESHOLD_MAX = 1.0

# Answer length ceiling requested by the caller.
MAX_ANSWER_CHARS_MIN = 500
MAX_ANSWER_CHARS_MAX = 20_000

# Defaults applied when the caller supplies nothing.
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.35
DEFAULT_MAX_ANSWER_CHARS = 10_000


def to_json(value: Any) -> Optional[str]:
    """Serialize a value to a JSON string for State storage."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Any, default: Any = None) -> Any:
    """Deserialize a JSON string from State storage.

    Accepts an already-deserialised value unchanged so a node is robust to
    either representation reaching it.
    """
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def finite_in_range(value: Any, lo: float, hi: float) -> Optional[float]:
    """Parse a caller-controlled numeric: FINITE float within [lo, hi], else None.

    Rejects bools, non-numerics, and — critically — non-finite values: float()
    happily parses "NaN"/"Infinity" (and Python's json accepts bare NaN in
    request bodies), and IEEE NaN comparisons are always False, which turns a
    threshold check into a silent pass. Every caller-supplied number comes
    through here so an unusable value fails CLOSED instead of disabling the
    check it was meant to control.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


class State(AgentState):
    """Flat TypedDict for LOG-C2-029.

    All shared fields (user_input, input_context, status, session_id,
    node_history, error_log, result, formatted_output, hitl_*, etc.) are
    inherited from AgentState and are NOT re-declared here — re-declaring an
    inherited field with a narrower type breaks the state contract.
    """

    # ------------------------------------------------------------------
    # Input layer — the question and the validated caller options
    # ------------------------------------------------------------------

    # Raw natural-language question from the user.
    query: str

    # JSON-serialised dict of caller options AFTER validation by
    # PreProcessNode. This is the only channel by which caller-supplied data
    # reaches the inner workflow, and every field in it has already been
    # bounds-checked:
    #   {"carrier": str|None, "route_code": str|None, "shipment_class": str|None,
    #    "channel": str, "top_k": int, "score_threshold": float,
    #    "max_answer_chars": int, "policy_documents": [ {...}, ... ]}
    validated_context: Optional[str]

    # Optional carrier identifier supplied by the caller (inert slug).
    carrier: Optional[str]

    # Optional route code scoping the policy lookup (e.g. "HND-OSA").
    route_code: Optional[str]

    # Optional shipment class context (inert slug, e.g. "cold_chain").
    shipment_class: Optional[str]

    # ------------------------------------------------------------------
    # Normalization layer — QueryNormalizeNode
    # ------------------------------------------------------------------

    # Normalized query text: filler-stripped, whitespace-collapsed, lowercased.
    normalized_query: Optional[str]

    # JSON-serialised retrieval context resolved from the query text and the
    # validated caller options: {"carrier": ..., "route": ..., "shipment_class": ...}
    extracted_context: Optional[str]

    # ------------------------------------------------------------------
    # Retrieval layer — LogisticsPolicyKBRetrieveNode
    # ------------------------------------------------------------------

    # JSON-serialised ranked policy passages:
    # [{"text": str, "source": str, "score": float, "last_verified": str}]
    retrieved_chunks: Optional[str]

    # True when the query falls outside what a static policy corpus can answer
    # (live tracking / ETA), or when retrieval returned nothing.
    # Represented as out_of_scope=True with a SUCCESS status: the run behaved
    # correctly, the question was simply not answerable from policy text.
    out_of_scope: bool

    # Machine-readable reason when out_of_scope is True
    # (e.g. "no_kb_match", "live_data_request").
    out_of_scope_reason: Optional[str]

    # ------------------------------------------------------------------
    # Answer layer — AnswerGenerateNode
    # ------------------------------------------------------------------

    # Generated policy answer text presented to the end user.
    answer: Optional[str]

    # JSON-serialised list of source identifiers backing the answer.
    citations: Optional[str]

    # ISO date of the most recently verified source used (e.g. "2026-03-01").
    last_verified: Optional[str]

    # ------------------------------------------------------------------
    # Validation layer — ResponseValidateNode
    # ------------------------------------------------------------------

    # Answer text after the in-workflow content checks.
    validated_answer: Optional[str]

    # Set when the run completes WITHOUT producing an answer because the
    # caller's request could not be accepted as written - a rejection the
    # caller can correct and retry. The run still completes: nothing is
    # retrieved, no answer is assembled, and the domain audit event for the
    # rejection is still emitted. Carrying this as a completion marker rather
    # than a terminal error is what lets the caller see the reason and send a
    # corrected request on the same conversation.
    error_code: Optional[str]

    # True when the answer passed the content checks unchanged.
    validation_passed: Optional[bool]
