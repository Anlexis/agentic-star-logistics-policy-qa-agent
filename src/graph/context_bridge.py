"""AgentCore Platform v1.0"""

# src/graph/context_bridge.py — carries the VALIDATED caller options across the
# outer -> inner graph boundary.
#
# Why this exists: GraphNode.execute() invokes the inner graph as
# `subgraph.invoke(user_input, session_id=..., ctx=...)` and does not forward
# the outer state's caller context, so an inner-node read of the caller's
# options would always see an empty dict through the full nested graph. The
# sanctioned subclass hooks bridge it:
#
#   LogisticsPolicyQAGraphNode.extract_input(state)   [before subgraph.invoke]
#       -> set_caller_context(<validated options>)
#   DomainWorkflowGraph._extra_initial_state()        [inside subgraph.invoke]
#       -> returns {"validated_context": <validated options>}
#
# Only PreProcessNode's validated output travels this channel — never the raw
# request body — so the inner workflow reads bounds-checked values by
# construction.
#
# A ContextVar keeps the hand-off correct per thread/task, so concurrent
# invocations in one process cannot see each other's context.

from contextvars import ContextVar
from typing import Any, Dict, Optional

_CALLER_CONTEXT: ContextVar[Optional[Dict[str, Any]]] = ContextVar("log_c2_029_caller_context", default=None)


def set_caller_context(context: Optional[Dict[str, Any]]) -> None:
    """Stash the validated caller options for the imminent inner-graph invoke."""
    _CALLER_CONTEXT.set(dict(context) if context else {})


def get_caller_context() -> Dict[str, Any]:
    """Read (without consuming) the stashed options; {} when none was set."""
    return _CALLER_CONTEXT.get() or {}
