import numpy as np

from .core import as_noisy_float
from .core import _combine_float
from .core import as_noisy_float_array
from .core import sample_shaped
from numpy.random import Generator
from sympy import Max
from sympy import Min


def _fold_float(values, op):
    if not values:
        raise ValueError("Requires at least one value")

    result = as_noisy_float(values[0])
    for value in values[1:]:
        result = _combine_float(result, as_noisy_float(value), op)
    return result


def _odds_ratio(a, b, c, d):
    # All values must be positive
    if min(float(a), float(b), float(c), float(d)) <= 0.0:
        return None

    # Odds ratio calculation
    result = (a * d) / (b * c)

    # Validate just the observation if noisy
    if not np.isfinite(float(result)) or float(result) <= 0.0:
        return None

    # Can be a noisy float or just a float
    return result


def noisy_min(*values):
    return _fold_float(values, Min)


def noisy_max(*values):
    return _fold_float(values, Max)


class OddsRatio:
    def __init__(self, tbl):
        self.samples = None

        # Enforce noisy floats in contingency table and correct shape
        self.tbl = as_noisy_float_array(tbl)
        assert self.tbl.shape == (2, 2)

    def sample(self, n=10000, rng=None, lib="scipy"):
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
        for grp0_yes, grp0_no, grp1_yes, grp1_no in dp_draws:

            # Total counts for each group
            grp0 = int(round(grp0_yes + grp0_no))
            grp1 = int(round(grp1_yes + grp1_no))

            # Throw out samples with non-positive group sizes
            if grp0 <= 0 or grp1 <= 0:
                continue

            # Ratios of "yes" outcomes within groups
            grp0_yes_ratio = grp0_yes / (grp0_yes + grp0_no)
            grp1_yes_ratio = grp1_yes / (grp1_yes + grp1_no)

            grp0_yes_draw = rng.binomial(grp0, grp0_yes_ratio)
            grp0_no_draw = grp0 - grp0_yes_draw
            grp1_yes_draw = rng.binomial(grp1, grp1_yes_ratio)
            grp1_no_draw = grp1 - grp1_yes_draw

            # Calculate odds ratio and keep if valid
            or_draw = _odds_ratio(grp0_yes_draw, grp0_no_draw, grp1_yes_draw, grp1_no_draw)
            if or_draw is not None:
                or_draws.append(or_draw)

        # Cache ratios for later confidence interval calculation
        self.samples = np.asarray(or_draws, dtype=float)

        # Just for convenient chaining for analysts
        return self

    def ratio(self):
        # Extract values from the contingency table
        grp0_yes_draw, grp0_no_draw, grp1_yes_draw, grp1_no_draw = self.tbl.ravel()

        # Odds ratio should come back as a noisy float
        return _odds_ratio(grp0_yes_draw, grp0_no_draw, grp1_yes_draw, grp1_no_draw)

    def confidence_interval(self, a=0.05):
        a = float(a)
        assert 0.0 <= a <= 1.0

        # Get samples if not already done
        if self.samples is None:
            self.sample()

        # Not enough valid odds ratio samples
        if self.samples.size == 0:
            raise ValueError("No valid odds ratio draws")

        # Compute confidence interval
        lo, hi = np.quantile(self.samples, [a / 2.0, 1.0 - a / 2.0])
        return float(lo), float(hi)
