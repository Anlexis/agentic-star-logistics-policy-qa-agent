"""AgentCore Platform v1.0"""

# LOG-C2-029 — DomainWorkflowGraph (inner BaseGraph)
#
# The inner graph of the Cat 2 two-layer architecture. It holds the whole
# logistics policy Q&A workflow:
#
#   START -> query_normalize -> kb_retrieve -> answer_generate
#         -> response_validate -> END
#
# LogisticsPolicyQAGraphNode.get_subgraph() (src/graph/graph.py) builds it, and
# get_output() below shapes the dict that node's merge_output() consumes.
#
# Rules enforced:
#   * inherits BaseGraph (fully custom topology — no forced backbone)
#   * implements every BaseGraph abstract method
#   * register_nodes() does NOT call super() (it is abstract in BaseGraph)
#   * does NOT register initialize / finalize (outer backbone concerns)
#   * get_output() is designed together with the outer merge_output()

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus

from src.graph.context_bridge import get_caller_context
from src.nodes.answer_generate import AnswerGenerateNode
from src.nodes.logistics_policy_kb_retrieve import LogisticsPolicyKBRetrieveNode
from src.nodes.query_normalize import QueryNormalizeNode
from src.nodes.response_validate import ResponseValidateNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for LOG-C2-029.

    Pipeline (linear):
        START
          -> query_normalize      (QueryNormalizeNode)
          -> kb_retrieve          (LogisticsPolicyKBRetrieveNode)
          -> answer_generate      (AnswerGenerateNode)
          -> response_validate    (ResponseValidateNode)
          -> END

    Every node is a FunctionNode subclass returning partial-dict state updates.
    """

    # -- Identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "log_c2_029_policy_qa_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict shared across the inner and outer graph."""
        return State

    # -- Config validation --------------------------------------------------

    def _validate_config(self) -> None:
        """Validate the inner graph's config before compilation.

        Runtime parameters (max_retry, timeout_s) are forwarded from
        config/config.yaml by the wrapping node. Every key is optional — each
        domain node resolves its own defaults — but a key that IS declared must
        be usable, and a deployment mistake should surface at compile time
        rather than mid-run.
        """
        config = self.config or {}
        max_retry = config.get("max_retry")
        if max_retry is not None and (
            not isinstance(max_retry, int) or isinstance(max_retry, bool) or not 0 <= max_retry < 10
        ):
            raise ValueError(f"{type(self).__name__}: max_retry must be an integer between 0 and 9")
        timeout_s = config.get("timeout_s")
        if timeout_s is not None and (
            isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s <= 600
        ):
            raise ValueError(f"{type(self).__name__}: timeout_s must be a number between 0 and 600")

    # -- Caller-options hand-off -------------------------------------------

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the validated caller options into the inner initial state.

        The framework does not forward the outer graph's caller context into a
        subgraph invoke, so the outer GraphNode stashes the VALIDATED options
        and this hook reads them back (src/graph/context_bridge.py). The inner
        nodes therefore see bounds-checked values or nothing at all.
        """
        return {"validated_context": to_json(get_caller_context())}

    # -- Node registration --------------------------------------------------

    def register_nodes(self) -> None:
        """Register all four domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract. initialize
        and finalize are outer backbone concerns and are not registered here.
        """
        self._nodes["query_normalize"] = QueryNormalizeNode()
        self._nodes["kb_retrieve"] = LogisticsPolicyKBRetrieveNode()
        self._nodes["answer_generate"] = AnswerGenerateNode()
        self._nodes["response_validate"] = ResponseValidateNode()

    # -- Edge wiring --------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear policy Q&A topology.

        Each step passes its partial-dict output into the shared State. The
        topology is intentionally linear — no conditional branching between
        domain nodes — so route() exists to satisfy the abstract contract but
        add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "query_normalize")
        self._sg.add_edge("query_normalize", "kb_retrieve")
        self._sg.add_edge("kb_retrieve", "answer_generate")
        self._sg.add_edge("answer_generate", "response_validate")
        self._sg.add_edge("response_validate", END)

    # -- Routing ------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing — required by the BaseGraph contract.

        This topology is linear, so the method is never called at runtime. It
        returns END on an error status so an unexpected call cannot re-enter a
        processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "response_validate"

    # -- Output shape -------------------------------------------------------

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the dict returned to the outer graph as `sub_result`.

        LogisticsPolicyQAGraphNode.merge_output() consumes exactly these keys;
        the two are designed together so field names cannot drift apart.
        """
        output: Dict[str, Any] = {
            "answer": state.get("answer"),
            "result": state.get("result") or state.get("answer"),
            "validated_answer": state.get("validated_answer"),
            "validation_passed": state.get("validation_passed"),
            "out_of_scope": state.get("out_of_scope"),
            "out_of_scope_reason": state.get("out_of_scope_reason"),
            "citations": state.get("citations"),
            "last_verified": state.get("last_verified"),
            "status": state.get("status"),
            # Carried explicitly: the boundary only moves the keys named here.
            "error_code": state.get("error_code"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
        return output

    # -- Lifecycle helpers --------------------------------------------------

    def get_state_class(self) -> type:
        """Return the State TypedDict used by both graphs."""
        return State
