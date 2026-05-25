import numpy as np
from statsmodels.stats.contingency_tables import Table2x2

from .core import _as_noisy_float
from .core import _combine_float
from .release import noisy_float
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


class NoisyTable2x2:
    def __init__(self, table):
        table = np.asarray(table, dtype=object)
        if table.shape != (2, 2):
            raise ValueError("table must be a 2x2 structure")

        self.a = _as_noisy_float(table[0, 0])
        self.b = _as_noisy_float(table[0, 1])
        self.c = _as_noisy_float(table[1, 0])
        self.d = _as_noisy_float(table[1, 1])

    @classmethod
    def from_cells(cls, a, b, c, d):
        return cls([[a, b], [c, d]])

    @property
    def table(self):
        return np.array([[self.a, self.b], [self.c, self.d]], dtype=object)


    def sample_n(self, n, seed=None):
        a_draws = self.a.sample_n(n=n, seed=seed)
        b_draws = self.b.sample_n(n=n, seed=seed)
        c_draws = self.c.sample_n(n=n, seed=seed)
        d_draws = self.d.sample_n(n=n, seed=seed)

        return np.array([[a_draws, b_draws], [c_draws, d_draws]], dtype=object)


    def oddsratio(self, corr=0.0):
        a = self.a + corr
        b = self.b + corr
        c = self.c + corr
        d = self.d + corr

        if min(float(a), float(b), float(c), float(d)) <= 0:
            return None

        return (a * d) / (b * c)


    def oddsratio_confint(self,
        n=10000,
        alpha=0.05,
        corr=0.0,
        seed=None):

        if n <= 0:
            raise ValueError("n must be positive")

        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")

        rng = seed
        if isinstance(seed, int):
            rng = np.random.default_rng(seed)

        a_draws = np.asarray(self.a.sample_n(n=n, seed=rng), dtype=float)
        b_draws = np.asarray(self.b.sample_n(n=n, seed=rng), dtype=float)
        c_draws = np.asarray(self.c.sample_n(n=n, seed=rng), dtype=float)
        d_draws = np.asarray(self.d.sample_n(n=n, seed=rng), dtype=float)

        sampling_rng = np.random.default_rng(seed)
        or_draws = []

        for a_draw, b_draw, c_draw, d_draw in zip(a_draws, b_draws, c_draws, d_draws):
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

            a_eff = float(sampling_rng.binomial(row0_total, p0))
            b_eff = float(row0_total - a_eff)
            c_eff = float(sampling_rng.binomial(row1_total, p1))
            d_eff = float(row1_total - c_eff)

            # TODO This is just odds ratio, reuse it?
            numerator = (a_eff + correction) * (d_eff + correction)
            denominator = (b_eff + correction) * (c_eff + correction)
            if denominator <= 0 or numerator <= 0:
                continue

            or_draw = numerator / denominator
            if np.isfinite(or_draw) and or_draw > 0:
                or_draws.append(or_draw)

        if not or_draws:
            raise ValueError("No valid odds ratio draws")

        q_low, q_high = np.quantile(or_draws, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(q_low), float(q_high)
