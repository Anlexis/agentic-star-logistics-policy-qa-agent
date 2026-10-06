# Boundary: invoke execution order and the trust gate.
#
# BaseNode.__call__() is both the graph entry point and a security boundary: it
# runs the trust gate, then the audit start event, the input gate, execute(),
# the output gate and the audit completion event — in that order, for every
# concrete node under src/nodes/. A node that reordered or skipped a step would
# still pass its own unit tests, so the order is asserted here directly.
#
# The negative case matters as much: an under-privileged caller is refused
# before execute() runs and before any normal lifecycle event is emitted.
# Assertions are on BEHAVIOUR — refused, nothing executed — never on the
# framework's exact refusal wording, which is not this template's to fix.

import importlib
import inspect
import pkgutil
from typing import ClassVar

import pytest
from framework.nodes.base_node import BaseNode
from framework.schemas.trust_level import TrustLevel


class _PrivilegedTrustGateFixture(BaseNode):
    """Always-present privileged node used to prove the trust-gate refusal."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _security_gate_input(self, state):
        return state

    def execute(self, state):
        return {"status": "success"}

    def _security_gate_output(self, result):
        return result


def _trust_predecessor(required: TrustLevel) -> TrustLevel:
    """Return a lower valid trust level; fail loudly if the framework adds one."""
    predecessors = {
        TrustLevel.VERIFIED_EXTERNAL: TrustLevel.ANONYMOUS,
        TrustLevel.INTERNAL: TrustLevel.VERIFIED_EXTERNAL,
    }
    try:
        return predecessors[required]
    except KeyError as exc:
        raise AssertionError(f"no lower trust level defined for {required!r}") from exc


def _discover_node_classes() -> list[type]:
    """Import every module under src/nodes/ and collect concrete BaseNode subclasses."""
    try:
        pkg = importlib.import_module("src.nodes")
    except ImportError as exc:
        pytest.fail(f"PB-6 cannot import src.nodes; framework/template setup is broken: {exc}")

    discovered = []
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix="src.nodes."):
        module = importlib.import_module(modname)
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseNode)
                and attr is not BaseNode
                and attr.__module__ == modname
                and not inspect.isabstract(attr)
            ):
                discovered.append(attr)
    return discovered


class TestInvokeOrder:
    """__call__ runs: trust gate -> node_start -> input gate -> execute()
    -> output gate -> node_complete."""

    def test_an_under_privileged_caller_is_refused_before_execute(self, monkeypatch):
        """An always-present privileged node proves the refusal path."""
        import framework.nodes.base_node as base_node_module

        events: list[str] = []
        execute_calls: list[object] = []
        monkeypatch.setattr(
            base_node_module,
            "emit_trace_event",
            lambda event_type, _payload, _state: events.append(event_type),
        )
        original_execute = _PrivilegedTrustGateFixture.execute

        def spy_execute(self, state):
            execute_calls.append(state)
            return original_execute(self, state)

        monkeypatch.setattr(_PrivilegedTrustGateFixture, "execute", spy_execute)
        result = _PrivilegedTrustGateFixture()(
            {
                "caller_trust_level": _trust_predecessor(_PrivilegedTrustGateFixture.required_trust_level).value,
                "correlation_id": "tc08-s1-denial",
            }
        )

        # Behaviour, not wording: refused, nothing executed, nothing published.
        assert result["status"] == "error"
        assert result.get("error_log"), "a refusal must be recorded"
        assert any("trust" in entry.lower() for entry in result["error_log"])
        assert events and all(event != "node_complete" for event in events)
        assert not execute_calls

    def test_call_order_for_every_node(self, monkeypatch):
        node_classes = _discover_node_classes()
        if not node_classes:
            pytest.skip("no concrete BaseNode subclasses found under src/nodes/")

        import framework.nodes.base_node as base_node_module

        failures: list[str] = []
        for node_cls in node_classes:
            order: list[str] = []
            monkeypatch.setattr(
                base_node_module,
                "emit_trace_event",
                lambda event_type, _payload, _state, _o=order: _o.append(f"event:{event_type}"),
            )

            for method_name, label in (
                ("_security_gate_input", "security_gate_input"),
                ("execute", "execute"),
                ("_security_gate_output", "security_gate_output"),
            ):
                original = getattr(node_cls, method_name)

                def spy(self, arg, _o=order, _label=label, _orig=original):
                    _o.append(_label)
                    return _orig(self, arg)

                monkeypatch.setattr(node_cls, method_name, spy)

            instance = node_cls()
            state = {
                "caller_trust_level": node_cls.required_trust_level.value,
                "correlation_id": "pb6-invoke-order-test",
            }
            instance(state)

            expected = [
                "event:node_start",
                "security_gate_input",
                "execute",
                "security_gate_output",
                "event:node_complete",
            ]

            if order != expected:
                failures.append(
                    f"{node_cls.__name__}: invoke order violation.\n" f"expected: {expected}\nactual:   {order}"
                )

        assert not failures, "\n\n".join(failures)


class TestEntryTrustBoundary:
    """The agent's declared minimum caller trust is enforced at its entry node."""

    def test_the_entry_node_refuses_an_anonymous_caller(self, monkeypatch):
        from src.nodes.pre_process_node import PreProcessNode

        import framework.nodes.base_node as base_node_module

        monkeypatch.setattr(base_node_module, "emit_trace_event", lambda *a, **k: None)
        result = PreProcessNode()(
            {
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
                "user_input": "What packaging standards apply?",
                "correlation_id": "trust-boundary",
            }
        )
        assert result["status"] == "error"
        assert "validated_input" not in result

    def test_the_entry_node_admits_a_verified_external_caller(self, monkeypatch):
        from src.nodes.pre_process_node import PreProcessNode

        import framework.nodes.base_node as base_node_module

        monkeypatch.setattr(base_node_module, "emit_trace_event", lambda *a, **k: None)
        result = PreProcessNode()(
            {
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
                "user_input": "What packaging standards apply?",
                "input_context": {},
                "correlation_id": "trust-boundary",
            }
        )
        assert result.get("validated_input") == "What packaging standards apply?"
