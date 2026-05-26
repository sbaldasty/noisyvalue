import numpy as np

from .core import _as_noisy_float
from .core import _combine_float
from .core import as_noisy_float_array
from .core import sample_n
from .core import sample_shaped
from dataclasses import dataclass
from functools import cache
from numpy.random import Generator
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


class OddsRatio:
    def __init__(self, tbl):
        self.tbl = as_noisy_float_array(tbl)
        assert self.tbl.shape == (2, 2)

    def sample(self, n=1000, rng=None, lib="scipy"):
        # Number of samples (odds ratio draws) must be positive
        n = int(n)
        assert n > 0

        # Initialize random number generator for binomial sampling
        if not isinstance(rng, Generator):
            rng = np.random.default_rng(rng)

        # Sample flattened table with differential privacy uncertainty
        tbl = self.tbl.ravel()
        dp_draws = sample_shaped(tbl, n, lib, rng, axis=0)

        # For collecting valid odds ratio draws
        or_draws = []

        for a_draw, b_draw, c_draw, d_draw in dp_draws:
            row0_total = int(round(a_draw + b_draw))
            row1_total = int(round(c_draw + d_draw))
            if row0_total <= 0 or row1_total <= 0:
                continue

            p0 = a_draw / (a_draw + b_draw)
            p1 = c_draw / (c_draw + d_draw)
            if not np.isfinite(p0) or not np.isfinite(p1):
                continue
            p0 = float(np.clip(p0, 0.0, 1.0))
            p1 = float(np.clip(p1, 0.0, 1.0))

            a_eff = float(rng.binomial(row0_total, p0))
            b_eff = float(row0_total - a_eff)
            c_eff = float(rng.binomial(row1_total, p1))
            d_eff = float(row1_total - c_eff)

            numerator = a_eff * d_eff
            denominator = b_eff * c_eff
            if denominator <= 0 or numerator <= 0:
                continue

            or_draw = numerator / denominator
            if np.isfinite(or_draw) and or_draw > 0:
                or_draws.append(or_draw)

        self.samples = np.asarray(or_draws, dtype=float)
        self.confidence_interval.cache_clear()
        return self

    @cache
    def ratio(self):
        a, b, c, d = self.tbl.ravel()

        # All components must be positive, this also precludes DBZ
        if min(float(a), float(b), float(c), float(d)) <= 0:
            return None

        # Odds ratio computation
        return (a * d) / (b * c)

    @cache
    def confidence_interval(self, a=0.05):
        a = float(a)
        assert 0.0 <= a <= 1.0

        # Get samples if not already done
        if self.samples is None:
            self.samples = self.tbl.sample_n()

        # Not enough valid odds ratio samples
        if self.samples.size == 0:
            raise ValueError("No valid odds ratio draws")

        # Compute confidence interval
        lo, hi = np.quantile(self.samples, [a / 2.0, 1.0 - a / 2.0])
        return float(lo), float(hi)
 