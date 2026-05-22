from sympy import Max
from sympy import Min

from .core import _as_noisy_float
from .core import _combine_float


def _fold_float(values, op):
    if not values:
        raise ValueError("requires at least one value")

    result = _as_noisy_float(values[0])
    for value in values[1:]:
        result = _combine_float(result, _as_noisy_float(value), op)
    return result


def noisy_min(*values):
    return _fold_float(values, Min)


def noisy_max(*values):
    return _fold_float(values, Max)
