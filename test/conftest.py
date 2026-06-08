from collections import defaultdict
from weakref import WeakSet

import pytest
import sympy as sp

import src.core as core_module

from src.core import NoisyFloat


@pytest.fixture(autouse=True)
def clear_node_registry():
    core_module._SYMBOL_NODES.clear()
    core_module._SYMBOL_ASSOCIATED_NODES = defaultdict(WeakSet)


def rooted_float(obs, expr, eqns=()):
    expr = sp.sympify(expr)
    eqns = tuple(sp.sympify(eqn) for eqn in eqns)
    root = core_module.Node.derived(definition=expr, constraints=eqns)
    return NoisyFloat.from_node(obs=obs, root=root, expr=expr)
