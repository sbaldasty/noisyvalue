import sympy as sp

import src.core as core_module

from src.core import NoisyFloat


def rooted_float(obs, expr, eqns=(), depends_on=()):
    expr = sp.sympify(expr)
    eqns = tuple(sp.sympify(eqn) for eqn in eqns)
    root = core_module.Node.derived(definition=expr, constraints=eqns, depends_on=depends_on)
    return NoisyFloat.from_node(obs=obs, root=root, expr=expr)
