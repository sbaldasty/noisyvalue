import numpy as np

from .core import _as_noisy_float
from .core import _combine_float
from sympy import Max
from sympy import Min


def _fold_float(values, op):
    if not values:
        raise ValueError("Requires at least one value")

    result = _as_noisy_float(values[0])
    for value in values[1:]:
        result = _combine_float(result, _as_noisy_float(value), op)
    return result


def noisy_min(*values):
    return _fold_float(values, Min)


def noisy_max(*values):
    return _fold_float(values, Max)


def odds_ratio(a, b, c, d, correction=0.5):
    a = _as_noisy_float(a) + correction
    b = _as_noisy_float(b) + correction
    c = _as_noisy_float(c) + correction
    d = _as_noisy_float(d) + correction

    if min(float(a), float(b), float(c), float(d)) <= 0:
        raise ValueError("Corrected inputs must be non-negative")

    return (a * d) / (b * c)


def odds_ratio_ci(a, b, c, d, n=10000, alpha=0.05, correction=0.5, seed=None):
    or_nf = odds_ratio(a, b, c, d, correction=correction)
    draws = np.asarray(or_nf.sample_n(n=n, seed=seed), dtype=float)
    draws = draws[np.isfinite(draws) & (draws > 0)]
    if draws.size == 0:
        raise ValueError("No valid odds ratio draws")

    q_low, q_high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(q_low), float(q_high)
