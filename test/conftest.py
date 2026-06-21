import sympy as sp

from src.core import NoisyFloat
from src.graph import DerivedNode, NormalNoiseNode, BinomialNoiseNode
from sympy import Basic, sympify


def rooted_float(obs, expr, eqns=(), depends_on=()):
    expr = sp.sympify(expr)
    eqns = tuple(sp.sympify(eqn) for eqn in eqns)
    root = DerivedNode(expr, constraints=eqns, depends_on=depends_on)
    return NoisyFloat.from_node(obs=obs, root=root, expr=expr)


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
    return NormalNoiseNode(_to_expr(loc), _to_expr(scale))


def binomial(n, p):
    return BinomialNoiseNode(_to_expr(n), _to_expr(p))
