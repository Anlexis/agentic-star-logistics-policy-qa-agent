# Boundary: human-in-the-loop interrupt propagation.
#
# This boundary test verifies that a human-in-the-loop interrupt raised inside
# the agent graph propagates ACROSS the backbone / subgraph boundary up to the
# invoking caller — so an external orchestrator can pause the run, collect a
# human decision and resume it. The behaviour is only meaningful for templates
# that opt into cross-boundary propagation: a main-slot GraphNode declaring
# `propagate_hitl = True`, or an explicit interrupt() checkpoint on the
# five-node backbone.
#
# LOG-C2-029 does NOT: LogisticsPolicyQAGraphNode declares
# `propagate_hitl = False`, the inner workflow is a linear, rule-based policy
# Q&A pipeline with no interrupt() checkpoint, and the backbone runs to
# completion. There is therefore no propagation behaviour to assert, so this
# ships as a skip stub — a real, importable module that skips with a clear
# reason (never `assert True`), ready to be filled in if and when the
# propagation path is wired end to end.

import importlib

import pytest


def _hitl_propagation_enabled() -> bool:
    """True iff this template opts into cross-boundary interrupt propagation.

    Detected by inspecting the classes DEFINED in ``src/graph/graph.py`` for a
    node or graph subclass declaring ``propagate_hitl = True``. Imported
    defensively so collection never errors when the SDK or the graph module is
    unavailable — the test then simply skips.
    """
    try:
        graph_mod = importlib.import_module("src.graph.graph")
    except Exception:
        return False
    for obj in vars(graph_mod).values():
        if (
            isinstance(obj, type)
            and getattr(obj, "__module__", None) == graph_mod.__name__
            and getattr(obj, "propagate_hitl", False) is True
        ):
            return True
    return False


_HITL_PROPAGATION_ENABLED = _hitl_propagation_enabled()

_SKIP_REASON = (
    "cross-boundary human-in-the-loop propagation is not implemented for this "
    "template (LogisticsPolicyQAGraphNode.propagate_hitl=False; no interrupt() "
    "checkpoint) — skip stub"
)


@pytest.mark.skipif(not _HITL_PROPAGATION_ENABLED, reason=_SKIP_REASON)
class TestHitlInterruptPropagation:
    """A human-in-the-loop interrupt must propagate across the graph boundary.

    Skipped for this template — cross-boundary propagation is not enabled, so
    there is no behaviour to verify. The real assertion belongs here once the
    propagation path is wired end to end.
    """

    def test_hitl_interrupt_propagates_to_caller(self):
        # Reached only when a graph class declares propagate_hitl=True. The real
        # assertion (invoke -> assert the interrupt surfaces to the caller ->
        # resume) is implemented at that point.
        raise AssertionError("the real assertion is not yet implemented for a propagation-enabled template")
