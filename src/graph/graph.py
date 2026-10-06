"""AgentCore Platform v1.0"""

# LOG-C2-029 — Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Architecture:
#
#   Outer backbone (fixed — do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |
#                                             +-- retry -> pre_process
#
#   The `main` slot is a GraphNode subclass (LogisticsPolicyQAGraphNode) that
#   delegates the whole domain workflow to DomainWorkflowGraph (inner BaseGraph).
#
#   Domain complexity is fully encapsulated inside the inner graph; the outer
#   backbone stays a thin, uniform shell.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- validated-options hand-off across the boundary
#
# Rules enforced:
#   * LogisticsPolicyQAAgent inherits AgentBaseGraph (framework base class)
#   * super().register_nodes() is called first (fills initialize + finalize)
#   * LogisticsPolicyQAGraphNode occupies self._nodes["main"]
#   * merge_output() returns only changed keys
#   * add_edges() is NOT overridden on the outer graph

import os
from typing import Any, ClassVar, Dict, Optional, cast

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus

from src.graph.context_bridge import set_caller_context
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json

# Runtime parameters live in config/config.yaml at the repository root (three
# levels up from this file: src/graph/graph.py -> src/graph -> src -> root).
# config/agent.yaml is the static manifest and carries no runtime block, so a
# reader pointed at it would find nothing and every declared value would go
# silently unused.
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "config.yaml",
)


def _runtime_config() -> Dict[str, Any]:
    """Return the runtime parameters declared in config/config.yaml.

    Best-effort: a missing or unparseable file yields {} so graph construction
    never breaks — the graph and its nodes then fall back to their declared
    defaults. PyYAML is loaded lazily; it is a framework runtime dependency, so
    importing it on demand avoids a hard module-load coupling.
    """
    try:
        import yaml

        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        return cast(Dict[str, Any], loaded) if isinstance(loaded, dict) else {}
    except Exception:
        return {}


class LogisticsPolicyQAGraphNode(GraphNode):
    """GraphNode occupying the `main` slot of LogisticsPolicyQAAgent.

    Wraps DomainWorkflowGraph (the inner Cat 2 BaseGraph). The backbone calls
    it after pre_process and before post_process.

    Contracts:
      get_subgraph()    instantiate and return DomainWorkflowGraph
      extract_input()   return the question string handed to the inner invoke()
                        and stash the validated caller options for it
      merge_output()    map sub_result fields into the outer state delta
                        (changed keys only)
      error_strategy    "propagate": re-raise inner errors as SubgraphError
    """

    def __init__(self, runtime_config: dict[str, Any] | None = None) -> None:
        """Receive the runtime config from the outer graph.

        A BaseNode has no config back-reference of its own, so the outer
        AgentBaseGraph reads `self.config` and threads it in here at
        register_nodes() time. Static construction input - not mutable state.
        """
        # A non-mapping runtime config degrades to {} instead of raising: reading and
        # parsing config/config.yaml belongs to the entry point, and this node only has
        # to survive whatever it is handed.
        self._runtime_config = dict(runtime_config) if isinstance(runtime_config, dict) else {}

    # "propagate": re-raise inner graph exceptions as SubgraphError (fail fast).
    # "handle": call on_subgraph_error() instead — for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: human-in-the-loop interrupts stay contained in the inner graph.
    propagate_hitl: ClassVar[bool] = False

    def get_subgraph(self) -> Any:
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported inside the method to avoid a circular
        import at module load time.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to work on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: Dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the question string passed into inner_graph.invoke().

        The framework hands the inner graph a STRING, and does not forward the
        caller's options with it. PreProcessNode has already validated those
        options into `validated_context`; they are stashed here so the inner
        graph's `_extra_initial_state()` can seed them (src/graph/context_bridge.py).
        Only validated data crosses this boundary.
        """
        set_caller_context(from_json(state.get("validated_context"), {}) or {})

        validated = state.get("validated_input")
        if isinstance(validated, str) and validated.strip():
            return validated
        user_input = state.get("user_input", "")
        return user_input if isinstance(user_input, str) else ""

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the declared runtime parameters to the inner graph.

        The inner graph is constructed here, so without this it would be built
        with no config at all and the values declared in config/config.yaml
        would never reach it. Only keys the file actually declares are
        forwarded, at the top level (where the inner graph validates them) and
        under `configurable` (where a node reads them).
        """
        cfg = self._runtime_config
        declared = {"max_retry": cfg.get("max_retry"), "timeout_s": cfg.get("timeout_s")}
        forwarded = {key: value for key, value in declared.items() if value is not None}
        return {**forwarded, "configurable": dict(forwarded)}

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner graph's output into the outer state delta.

        `sub_result` is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys — never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          inner get_output()   emits: answer, result, status, validated_answer,
                                      validation_passed, out_of_scope,
                                      out_of_scope_reason, citations,
                                      last_verified, trace_id
          this merge_output()  reads:  the same names

        answer / result   the policy answer text (result is what post_process reads)
        validated_answer  the answer after the in-workflow content checks
        validation_passed True when those checks passed unchanged
        out_of_scope      True when the question is not answerable from policy text
        citations         JSON-serialised source list backing the answer
        last_verified     ISO date of the most recently verified source used
        """
        return {
            "answer": sub_result.get("answer"),
            # post_process reads state["result"]; map it from the answer.
            "result": sub_result.get("result") or sub_result.get("answer"),
            "status": sub_result.get("status"),
            # Outer reason wins: a reason settled before the inner run is the
            # real one, and a plain sub_result.get() would erase it.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
            "validated_answer": sub_result.get("validated_answer"),
            "validation_passed": sub_result.get("validation_passed"),
            "out_of_scope": sub_result.get("out_of_scope"),
            "out_of_scope_reason": sub_result.get("out_of_scope_reason"),
            "citations": sub_result.get("citations"),
            "last_verified": sub_result.get("last_verified"),
        }


class LogisticsPolicyQAAgent(AgentBaseGraph):
    """Outer graph for LOG-C2-029 (Cat 2).

    Inherits AgentBaseGraph directly. Domain logic is fully encapsulated in
    LogisticsPolicyQAGraphNode (the `main` slot), which delegates to
    DomainWorkflowGraph.

    Backbone (fixed):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills initialize and finalize
      - pre_process:  PreProcessNode  (caller-data contract)
      - main:         LogisticsPolicyQAGraphNode (delegates to the inner graph)
      - post_process: PostProcessNode (external-output boundary)

    add_edges() is NOT overridden — backbone wiring belongs to the framework.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Construct the agent with the declared runtime parameters.

        A caller (or the agent registry) may pass an explicit config; otherwise
        the runtime block in config/config.yaml is loaded, so a declared value
        such as max_retry actually governs the backbone instead of silently
        falling back to a framework default.
        """
        super().__init__(config if config is not None else _runtime_config())

    @property
    def name(self) -> str:
        """Agent identifier registered with the agent registry."""
        return "log_c2_029"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default initialize node (schema_version, session_id,
        trust_level) and finalize node (response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = LogisticsPolicyQAGraphNode(runtime_config=self.config)
        self._nodes["post_process"] = PostProcessNode()


# Back-compat alias — callers may reference either name.
Graph = LogisticsPolicyQAAgent

# Re-exported for the HTTP adapter's type annotations.
__all__ = ["Graph", "LogisticsPolicyQAAgent", "LogisticsPolicyQAGraphNode"]
