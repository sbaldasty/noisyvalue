from .core import _preferred_value_expr
from .core import NoisyFloat
from .core import NoisyInt
from .util import fresh_name
from sympy.stats import Binomial
from sympy.stats import Normal


def _to_expr(value):
    if isinstance(value, (NoisyFloat, NoisyInt)):
        return _preferred_value_expr(value)
    if isinstance(value, (int, float)):
        return value
    raise TypeError(f"Unsupported value type: {type(value)}")


def binomial(n, p):
    return binomial_factory(n, p)()


def binomial_factory(n, p):
    n = _to_expr(n)
    p = _to_expr(p)
    return lambda: Binomial(fresh_name(), n, p)


def gaussian(loc, scale):
    loc = _to_expr(loc)
    scale = _to_expr(scale)
    return lambda: Normal(fresh_name(), loc, scale)
