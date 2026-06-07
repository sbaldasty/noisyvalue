from collections import defaultdict
from weakref import WeakSet

import pytest

import src.core as core_module


@pytest.fixture(autouse=True)
def clear_node_registry():
    core_module._SYMBOL_NODES.clear()
    core_module._SYMBOL_ASSOCIATED_NODES = defaultdict(WeakSet)
