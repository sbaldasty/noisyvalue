import sympy as sp

from src.core import DerivedNode
from src.core import NoisyFloat


def rooted_float(obs, expr, eqns=(), depends_on=()):
    expr = sp.sympify(expr)
    eqns = tuple(sp.sympify(eqn) for eqn in eqns)
    root = DerivedNode(definition=expr, constraints=eqns, depends_on=depends_on)
    return NoisyFloat.from_node(obs=obs, root=root, expr=expr)
