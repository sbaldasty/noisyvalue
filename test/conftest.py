import sympy as sp

from src.core import NoisyFloat
from src.graph import DerivedNode, NormalNode, BinomialNode
from sympy import Basic, sympify


def from_node(obs, root, expr=None):
    expr = root.expr if expr is None else sp.sympify(expr)
    if expr != root.expr:
        root = DerivedNode.operational(expr, deps=[root])
    return NoisyFloat(obs, root)


def rooted_float(obs, expr, eqns=(), depends_on=()):
    expr = sp.sympify(expr)
    eqns = tuple(sp.sympify(eqn) for eqn in eqns)
    root = DerivedNode(expr, constraints=eqns, depends_on=depends_on)
    return from_node(obs=obs, root=root, expr=expr)


def _to_expr(value):
    if isinstance(value, Basic):
        return value
    if isinstance(value, (int, float)):
        return value

    from src.core import NoisyFloat, NoisyInt

    if isinstance(value, (NoisyFloat, NoisyInt)):
        return value.expr

    expr = sympify(value)
    if isinstance(expr, Basic):
        return expr
    raise TypeError(f"Unsupported value type: {type(value)}")


def gaussian(loc, scale):
    return NormalNode(_to_expr(loc), _to_expr(scale))


def binomial(n, p):
    return BinomialNode(_to_expr(n), _to_expr(p))
