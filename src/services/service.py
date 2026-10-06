"""AgentCore Platform v1.0"""

# LOG-C2-029 — domain service layer.
#
# Holds the two pieces of domain logic the nodes share, so neither is defined
# twice and both can be tested on their own:
#
#   PolicyCorpusService  deterministic retrieval over a policy corpus
#   is_live_data_request  the scope screen that separates a policy question
#                         from a live shipment-tracking question
#
# No business routing, no credentials, no state access — nodes pass values in
# and get values back.

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Scope screen
# ---------------------------------------------------------------------------

# A static policy corpus can describe what the rules ARE; it cannot say where a
# parcel is right now. Questions of the second kind are answered honestly ("out
# of scope") instead of being guessed at from policy text.
#
# Every alternative is anchored on a whole phrase or a whole word. An earlier
# substring form matched "eta" inside "detailed" and "live" inside "delivery",
# so ordinary policy answers were scoped out — the failure direction that
# silently blocks real work.
_LIVE_DATA_RE = re.compile(
    r"\beta\b"
    r"|\bcurrent status\b"
    r"|\breal[\s-]?time\b"
    r"|\blive (?:tracking|status|location|data|update|updates)\b"
    r"|\btrack(?:ing)? (?:this|my|the|our) (?:shipment|parcel|package|order|consignment)\b"
    r"|\bwhere is (?:my|the|our) (?:shipment|parcel|package|order|consignment)\b"
    r"|\bcurrent(?:ly)? (?:located|in transit)\b",
    re.IGNORECASE,
)


def is_live_data_request(text: str) -> bool:
    """True when *text* asks for (or asserts) live shipment data."""
    return bool(text) and bool(_LIVE_DATA_RE.search(text))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words carry no retrieval signal; keeping them would let any question
# score against any passage.
_STOPWORDS = frozenset(
    """a an and any are as at be by can do does for from has have how i if in
    is it its me my of on or our should so than that the their there these this
    to us was we what when where which who why will with you your""".split()
)

# Boost weights for the scoping tokens resolved upstream.
_CARRIER_BOOST = 0.15
_ROUTE_BOOST = 0.10
_CLASS_BOOST = 0.10

# Longest passage returned for one document.
_MAX_PASSAGE_CHARS = 1_200

# The corpus the template ships with, so a fresh checkout answers real
# questions before any corpus of your own is wired in. Replace it — or send
# `policy_documents` with the request — to answer from your own policies.
BUNDLED_CORPUS: List[Dict[str, str]] = [
    {
        "document_id": "packaging_standards",
        "title": "General packaging standards",
        "source": "logistics_policy/packaging_standards.md",
        "text": (
            "Cartons must be sealed on all seams and marked with the consignee name and "
            "destination route code. Fragile goods require corner protection and a "
            "double-wall carton. Palletised freight is shrink-wrapped and strapped in two "
            "directions.\n\n"
            "Any carton exceeding 30 kg carries a two-person lift label. Cartons that "
            "cannot be stacked are marked as top load and are placed last."
        ),
        "last_verified": "2026-03-01",
    },
    {
        "document_id": "cold_chain_handling",
        "title": "Cold chain and frozen handling",
        "source": "logistics_policy/cold_chain_handling.md",
        "text": (
            "Cold chain consignments move between 2 and 8 degrees Celsius and frozen "
            "consignments below minus 18 degrees Celsius. A temperature logger travels "
            "inside the packaging and its record is filed with the delivery note.\n\n"
            "A cold chain carton is packed with pre-conditioned gel packs and must not "
            "stand on an open dock for longer than fifteen minutes. A break in the "
            "temperature record is reported as an exception before the goods are released."
        ),
        "last_verified": "2026-03-01",
    },
    {
        "document_id": "hazardous_goods",
        "title": "Hazardous goods acceptance",
        "source": "logistics_policy/hazardous_goods.md",
        "text": (
            "Hazardous consignments are accepted only with a signed shipper declaration "
            "quoting the UN number, the packing group and the net quantity per package. "
            "Limited quantity shipments carry the limited quantity mark on two opposing "
            "faces.\n\n"
            "Lithium batteries shipped with equipment require a state of charge below 30 "
            "percent. Air movement of hazardous goods needs carrier approval before the "
            "booking is confirmed."
        ),
        "last_verified": "2026-03-01",
    },
    {
        "document_id": "service_levels",
        "title": "Route service levels and exception handling",
        "source": "logistics_policy/service_levels.md",
        "text": (
            "Domestic trunk routes commit to next day delivery when the consignment is "
            "tendered before the daily cut off. Regional routes commit to two working "
            "days. Weather suspensions extend the commitment by the length of the "
            "suspension.\n\n"
            "An exception is raised when a consignment misses its committed service level. "
            "The raising branch records the cause code and the recovery plan, and the "
            "consignee is notified the same working day."
        ),
        "last_verified": "2026-03-01",
    },
    {
        "document_id": "claims_and_damage",
        "title": "Damage and loss claims",
        "source": "logistics_policy/claims_and_damage.md",
        "text": (
            "A damage claim is filed within seven working days of delivery and must "
            "include the delivery note, photographs of the packaging and a description of "
            "the damage. Concealed damage is reported within three working days.\n\n"
            "A loss claim is opened after a consignment has been unlocated for ten working "
            "days. Claims are assessed against the declared packaging standard for the "
            "shipment class."
        ),
        "last_verified": "2026-03-01",
    },
]


def _fold(token: str) -> str:
    """Fold a regular plural onto its singular form.

    One rule, applied identically to the question and the corpus, so
    "standards" matches "standard" and "process" is left alone. Anything
    cleverer needs a stemmer, which is a dependency this template does not
    take on for a lexical baseline.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> List[str]:
    """Lowercase content tokens of *text*, function words removed, plurals folded."""
    return [_fold(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _coverage(query_tokens: Iterable[str], passage_tokens: Iterable[str]) -> float:
    """Share of the question's content tokens present in the passage."""
    wanted = set(query_tokens)
    if not wanted:
        return 0.0
    return len(wanted & set(passage_tokens)) / len(wanted)


def _passages(text: str) -> List[str]:
    """Split a document into paragraph-sized passages."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class PolicyCorpusService:
    """Deterministic relevance search over a logistics policy corpus.

    Scoring is a lexical coverage measure — the share of the question's content
    words that a passage actually contains — plus fixed boosts when the
    passage also mentions the carrier, route or shipment class resolved for the
    request. It is reproducible: the same question over the same corpus always
    returns the same passages in the same order, so a policy answer can be
    audited after the fact.

    The default threshold asks that about a third of the question's content
    words appear in the passage; raise it for a precise corpus, lower it for a
    sparse one. Wire a vector or hybrid search client in here to answer from a larger
    corpus; the return shape is what the rest of the pipeline depends on.
    """

    def search(
        self,
        query: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        score_threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """Return up to *top_k* ranked passages scoring at or above the threshold.

        Each result is
        ``{"text", "source", "document_id", "title", "score", "last_verified"}``.
        """
        corpus: List[Dict[str, Any]] = list(documents) if documents else list(BUNDLED_CORPUS)
        query_tokens = _tokens(query)
        if not query_tokens or not corpus:
            return []

        context = context or {}
        carrier = str(context.get("carrier") or "")
        route = str(context.get("route") or "")
        shipment_class = str(context.get("shipment_class") or "")

        results: List[Dict[str, Any]] = []
        for document in corpus:
            text = str(document.get("text", ""))
            if not text.strip():
                continue

            # The title describes the whole document, so it counts towards every
            # passage in it.
            title_tokens = _tokens(str(document.get("title", "")))
            best_score = 0.0
            best_passage = ""
            for passage in _passages(text):
                score = _coverage(query_tokens, _tokens(passage) + title_tokens)
                if score > best_score:
                    best_score, best_passage = score, passage
            if not best_passage:
                continue

            haystack = f"{text} {document.get('title', '')}".lower()
            if carrier and carrier.replace("_", " ") in haystack:
                best_score += _CARRIER_BOOST
            if route and route.lower() in haystack:
                best_score += _ROUTE_BOOST
            if shipment_class and shipment_class.replace("_", " ") in haystack:
                best_score += _CLASS_BOOST
            best_score = min(best_score, 1.0)

            if best_score < score_threshold:
                continue

            results.append(
                {
                    "text": best_passage[:_MAX_PASSAGE_CHARS],
                    "source": str(document.get("source", "")),
                    "document_id": str(document.get("document_id", "")),
                    "title": str(document.get("title", "")),
                    "score": round(best_score, 4),
                    "last_verified": str(document.get("last_verified", "")),
                }
            )

        results.sort(key=lambda chunk: (-float(chunk["score"]), str(chunk["document_id"])))
        return results[:top_k]


# ---------------------------------------------------------------------------
# Restricted-content screen
# ---------------------------------------------------------------------------

# Content that must never appear in an answer, whatever produced it. Policy
# prose, carrier names, route codes and container numbers are legitimate
# answer content, so this set targets secrets and the two identifier forms a
# logistics policy answer never needs: national-ID and payment-card numbers.
#
# One definition, applied twice: the workflow checks the answer it composed,
# and the external boundary checks the exact bytes that leave the agent. Two
# independent applications of the same rule cannot drift apart.
_RESTRICTED_PATTERNS: List[Any] = [
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/-]{20,}", re.IGNORECASE)),
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
    ("national_id", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("payment_card", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
]

# Defensive cap against pathological or self-referential structures; a
# legitimate answer payload never nests this deep.
_MAX_SCAN_DEPTH = 12


def scan_for_restricted_content(value: Any, depth: int = 0) -> Optional[str]:
    """Return the name of the first restricted form found in *value*, else None.

    Accepts any value and recurses into dicts, lists, tuples and sets, so a
    violation buried inside a structured payload — a citation list, a nested
    metadata dict — is caught as surely as one in a top-level string. Dict keys
    are schema field names, not caller content, so only values are scanned.
    """
    if depth > _MAX_SCAN_DEPTH:
        return None
    if isinstance(value, str):
        for name, pattern in _RESTRICTED_PATTERNS:
            if pattern.search(value):
                return str(name)
        return None
    if isinstance(value, dict):
        for item in value.values():
            hit = scan_for_restricted_content(item, depth + 1)
            if hit:
                return hit
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            hit = scan_for_restricted_content(item, depth + 1)
            if hit:
                return hit
        return None
    return None
