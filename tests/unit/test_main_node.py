# LOG-C2-029 — governance: the placeholder MainNode stays removed.
#
# The starter project's MainNode was dead code here — never wired into the
# nested graph, whose `main` slot is LogisticsPolicyQAGraphNode. This test
# governs its removal so the dead node cannot be reintroduced silently, and
# replaces the former MainNode unit tests.
#
# The file is deleted rather than emptied: an empty placeholder still registers
# as a backbone-named module in src/nodes/ carrying no class and no
# required_trust_level, which every node scan then has to special-case. So the
# assertions below are about absence of the module AND of the file.

import importlib.util
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MAIN_NODE_PATH = os.path.join(_REPO_ROOT, "src", "nodes", "main_node.py")


def test_main_node_module_removed():
    """src.nodes.main_node must not exist.

    find_spec returns None for a missing submodule of an existing package, so a
    reintroduced file — empty or not — fails here.
    """
    assert importlib.util.find_spec("src.nodes.main_node") is None, (
        "src/nodes/main_node.py was removed as dead code — do not reintroduce it; "
        "the `main` slot is LogisticsPolicyQAGraphNode in src/graph/graph.py."
    )


def test_main_node_file_absent():
    """The source file itself must be gone, not merely emptied."""
    assert not os.path.exists(_MAIN_NODE_PATH), (
        f"{_MAIN_NODE_PATH} still exists. An empty placeholder still registers as a "
        "backbone-named node module in src/nodes/ with no required_trust_level; "
        "delete the file instead of blanking it."
    )
