"""AgentCore Platform v1.0"""

# Node contract:
#   - extend FunctionNode; implement execute(state, config) -> dict
#   - return ONLY the fields this node changes (never the full state)
#   - return AgentStatus enum constants — never plain strings
#   - never import from other agents
#
# LOG-C2-029 — QueryNormalizeNode
# First node of the inner workflow: normalise the question and resolve the
# retrieval context (carrier, route, shipment class).
#
# Normalisation is deterministic and rule-based — no model call. Carrier and
# shipment-class name variants map onto canonical slugs, filler phrases are
# dropped and whitespace is collapsed, so the same question always produces the
# same retrieval key.
#
# Caller-supplied scoping values arrive through `validated_context`, which the
# entry boundary has already bounds-checked, and take precedence over anything
# inferred from the question text.

import logging
import re
from typing import Any, ClassVar, Dict, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INVALID_VALUE

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Carrier name normalisation
# ---------------------------------------------------------------------------

# Maps common abbreviations / variants onto a canonical carrier slug.
_CARRIER_ALIASES: Dict[str, str] = {
    "yamato": "yamato_transport",
    "kuroneko": "yamato_transport",
    "yamato transport": "yamato_transport",
    "sagawa": "sagawa_express",
    "sagawa express": "sagawa_express",
    "佐川": "sagawa_express",
    "fukuyama": "fukuyama_freight",
    "fukuyama freight": "fukuyama_freight",
    "seino": "seino_transportation",
    "seino transportation": "seino_transportation",
    "nittsu": "nippon_express",
    "nippon express": "nippon_express",
    "nexp": "nippon_express",
}

# ---------------------------------------------------------------------------
# Shipment class normalisation
# ---------------------------------------------------------------------------

_SHIPMENT_CLASS_ALIASES: Dict[str, str] = {
    "cold": "cold_chain",
    "cold chain": "cold_chain",
    "冷蔵": "cold_chain",
    "frozen": "frozen",
    "冷凍": "frozen",
    "hazmat": "hazardous",
    "dangerous goods": "hazardous",
    "危険物": "hazardous",
    "oversized": "oversized",
    "heavy": "oversized",
    "fragile": "fragile",
    "standard": "standard",
    "regular": "standard",
}

# ---------------------------------------------------------------------------
# Route code pattern (e.g. "HND-OSA", "TYO→NGO", "KIX~FUK")
# ---------------------------------------------------------------------------

_ROUTE_RE = re.compile(r"\b([A-Z]{3})\s*[-→~]\s*([A-Z]{3})\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Filler-phrase removal
# ---------------------------------------------------------------------------

_FILLER_RE = re.compile(
    r"\b(please|could you|can you|kindly|tell me|let me know|i want to know|about)\b",
    re.IGNORECASE,
)


def _normalize_query(raw: str) -> str:
    """Strip filler phrases, collapse whitespace, lowercase."""
    text = _FILLER_RE.sub("", raw)
    text = " ".join(text.split())
    return text.lower().strip()


def _extract_carrier(text: str) -> Optional[str]:
    """Return the canonical carrier slug if a known variant appears in *text*."""
    lower = text.lower()
    for alias, canonical in _CARRIER_ALIASES.items():
        if alias in lower:
            return canonical
    return None


def _extract_shipment_class(text: str) -> Optional[str]:
    """Return the canonical shipment class if a known variant appears in *text*."""
    lower = text.lower()
    for alias, canonical in _SHIPMENT_CLASS_ALIASES.items():
        if alias in lower:
            return canonical
    return None


def _extract_route(text: str) -> Optional[str]:
    """Return an 'AAA-BBB' route code if a route pattern appears in *text*."""
    match = _ROUTE_RE.search(text)
    if match:
        return f"{match.group(1).upper()}-{match.group(2).upper()}"
    return None


class QueryNormalizeNode(FunctionNode):
    """Normalise the question and resolve the retrieval context.

    Trust: the caller's trust level is decided once, at the agent's entry
    boundary (PreProcessNode); the inner workflow is unreachable except through
    it, so the inner nodes do not re-gate the caller.

    Input state keys:
        user_input:        str — the question (already sanitised at the boundary)
        validated_context: str — JSON-serialised validated caller options;
                                 carrier / route_code / shipment_class here take
                                 precedence over values inferred from the text

    Output state keys (partial dict):
        normalized_query:  str — filler-stripped, whitespace-collapsed, lowercased
        extracted_context: str — JSON-serialised {carrier, route, shipment_class}
        status:            AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:         list[str] (on error only)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        emit_progress("Preparing the query...")
        raw_query: Any = state.get("user_input", "")

        if not isinstance(raw_query, str):
            emit_progress(INVALID_VALUE)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "INVALID_REQUEST",
                "error_log": ["QueryNormalizeNode: the question must be text."],
            }

        if not raw_query.strip():
            emit_progress(EMPTY_INPUT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["QueryNormalizeNode: the question is empty."],
            }

        normalized_query = _normalize_query(raw_query)

        # Values inferred from the question text …
        carrier = _extract_carrier(raw_query)
        route = _extract_route(raw_query)
        shipment_class = _extract_shipment_class(raw_query)

        # … are overridden by the caller's validated scoping values.
        validated: Dict[str, Any] = from_json(state.get("validated_context"), {}) or {}
        if validated.get("carrier"):
            carrier = validated["carrier"]
        if validated.get("route_code"):
            route = validated["route_code"]
        if validated.get("shipment_class"):
            shipment_class = validated["shipment_class"]

        # Omit empty values to keep the state lean.
        extracted_context: Dict[str, Any] = {}
        if carrier is not None:
            extracted_context["carrier"] = carrier
        if route is not None:
            extracted_context["route"] = route
        if shipment_class is not None:
            extracted_context["shipment_class"] = shipment_class

        logger.info(
            "QueryNormalizeNode: normalized_length=%d context_keys=%s",
            len(normalized_query),
            list(extracted_context.keys()),
        )

        emit_trace_event(
            "query_normalize_complete",
            {
                "normalized_chars": len(normalized_query),
                "context_keys": sorted(extracted_context.keys()),
            },
            state,
        )

        return {
            "normalized_query": normalized_query,
            "extracted_context": to_json(extracted_context),
            "status": AgentStatus.SUCCESS.value,
        }
