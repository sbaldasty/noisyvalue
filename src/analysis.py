from . import noise
from .core import NoisyFloat
from .core import NoisyInt
from .core import GraphBuilder
from .core import Node
from .core import as_noisy_float
from .core import _preferred_value_expr
from .core import _combine_float
from .core import as_noisy_float_array
from .core import sample_float_array
import sympy as sp
from .util import fresh_name
from sympy import And
from sympy import Piecewise
from sympy import nan
from numpy import quantile
from numpy import asarray
from numpy import isfinite
from numpy.random import Generator
from numpy.random import default_rng
from sympy.stats import Binomial
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
    if not isfinite(float(result)) or float(result) <= 0.0:
        return None

    # Can be a noisy float or just a float
    return result


def _binomial_draw(total, yes_ratio):
    total = int(total)
    yes_ratio = float(yes_ratio)

    if total <= 0 or yes_ratio < 0.0 or yes_ratio > 1.0:
        return None

    expr = Binomial(fresh_name(), total, yes_ratio)
    obs = int(round(total * yes_ratio))
    root = NoisyInt.from_node(obs, Node(symbol=expr, depends_on=(), constraints=(), law=expr, role="noise"), expr=expr)
    return root


def _symbolic_odds_ratio(a, b, c, d):
    a = as_noisy_float(a)
    b = as_noisy_float(b)
    c = as_noisy_float(c)
    d = as_noisy_float(d)

    if min(float(a), float(b), float(c), float(d)) <= 0.0:
        return None

    a_expr = _preferred_value_expr(a)
    b_expr = _preferred_value_expr(b)
    c_expr = _preferred_value_expr(c)
    d_expr = _preferred_value_expr(d)

    expr = Piecewise(
        (
            (a_expr * d_expr) / (b_expr * c_expr),
            And(a_expr > 0, b_expr > 0, c_expr > 0, d_expr > 0),
        ),
        (nan, True),
    )
    result = (a * d) / (b * c)
    return type(result).from_node(float(result), result.root, expr=expr)


def odds_ratio(tbl):
    tbl = as_noisy_float_array(tbl)
    assert tbl.shape == (2, 2)
    assert isfinite(asarray([float(value) for value in tbl.ravel()], dtype=float)).all()

    grp0_yes, grp0_no, grp1_yes, grp1_no = tbl.ravel()

    grp0_total = (grp0_yes + grp0_no).round_nearest()
    grp1_total = (grp1_yes + grp1_no).round_nearest()
    grp0_ratio = grp0_yes / (grp0_yes + grp0_no)
    grp1_ratio = grp1_yes / (grp1_yes + grp1_no)

    builder = GraphBuilder(grp0_yes, grp0_no, grp1_yes, grp1_no)
    grp0_yes_draw = grp0_yes.round_nearest().resample(noise.binomial(grp0_total, grp0_ratio))
    grp1_yes_draw = grp1_yes.round_nearest().resample(noise.binomial(grp1_total, grp1_ratio))
    builder.include_values(grp0_yes_draw, grp1_yes_draw)

    grp0_no_draw = grp0_total - grp0_yes_draw
    grp1_no_draw = grp1_total - grp1_yes_draw

    valid = _preferred_value_expr(
        (grp0_total > 0)
        & (grp1_total > 0)
        & (grp0_yes + grp0_no > 0)
        & (grp1_yes + grp1_no > 0)
        & (grp0_ratio >= 0)
        & (grp0_ratio <= 1)
        & (grp1_ratio >= 0)
        & (grp1_ratio <= 1)
        & (grp0_yes_draw > 0)
        & (grp0_no_draw > 0)
        & (grp1_yes_draw > 0)
        & (grp1_no_draw > 0))

    ratio_draw = (grp0_yes_draw * grp1_no_draw) / (grp0_no_draw * grp1_yes_draw)
    expr = Piecewise((_preferred_value_expr(ratio_draw), valid), (nan, True))

    root = builder.derived(definition=expr)

    obs_valid = bool(
        (grp0_yes > 0)
        & (grp0_no > 0)
        & (grp1_yes > 0)
        & (grp1_no > 0))

    obs_or = float((grp0_yes * grp1_no) / (grp0_no * grp1_yes)) if obs_valid else nan
    return NoisyFloat.from_node(obs_or, root)


def noisy_min(*values):
    return _fold_float(values, Min)


def noisy_max(*values):
    return _fold_float(values, Max)


class MonteCarlo:
    def __init__(self):
        self.samples = None

    def sample(self, n=1000, rng=None, lib="scipy"):
        raise NotImplementedError("Must be implemented by subclass")

    def confidence_interval(self, a=0.05):
        a = float(a)
        assert 0.0 <= a <= 1.0

        # Get samples if not already done
        if self.samples is None:
            self.sample()

        # Not enough valid samples
        if self.samples.size == 0:
            raise ValueError("No valid draws")

        # Compute confidence interval
        lo, hi = quantile(self.samples, [a / 2.0, 1.0 - a / 2.0])
        return float(lo), float(hi)


class OddsRatio(MonteCarlo):
    def __init__(self, tbl):
        super().__init__()

        # Enforce noisy floats in contingency table and correct shape
        self.tbl = as_noisy_float_array(tbl)
        assert self.tbl.shape == (2, 2)

    def sample(self, n=10000, rng=None, lib="scipy"):
        # Number of samples (odds ratio draws) must be positive
        n = int(n)
        assert n > 0

        # Initialize random number generator for binomial sampling
        if not isinstance(rng, Generator):
            rng = default_rng(rng)

        # Sample flattened table with differential privacy uncertainty
        tbl = self.tbl.ravel()
        dp_draws = sample_float_array(tbl, n, lib, rng, axis=0)

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

            # Represent the sampling uncertainty symbolically, then sample it once.
            grp0_yes_draw = _binomial_draw(grp0, grp0_yes_ratio)
            grp1_yes_draw = _binomial_draw(grp1, grp1_yes_ratio)
            if grp0_yes_draw is None or grp1_yes_draw is None:
                continue

            grp0_no_draw = grp0 - grp0_yes_draw
            grp1_no_draw = grp1 - grp1_yes_draw

            # Calculate odds ratio and keep if valid
            or_draw = _symbolic_odds_ratio(grp0_yes_draw, grp0_no_draw, grp1_yes_draw, grp1_no_draw)
            if or_draw is not None:
                sampled_or = float(or_draw.sample(n=1, rng=rng)[0])
                if isfinite(sampled_or) and sampled_or > 0.0:
                    or_draws.append(sampled_or)

        # Cache ratios for later confidence interval calculation
        self.samples = asarray(or_draws, dtype=float)

        # Just for convenient chaining for analysts
        return self

    def _composed_or_value(self):
        return odds_ratio(self.tbl)

    def sample2(self, n=10000, rng=None, lib="scipy"):
        n = int(n)
        assert n > 0

        if not isinstance(rng, Generator):
            rng = default_rng(rng)

        composed_or = self._composed_or_value()
        if composed_or is None:
            self.samples = asarray([], dtype=float)
            return self

        draws = composed_or.sample(n=n, rng=rng)
        valid = isfinite(draws) & (draws > 0.0)
        self.samples = asarray(draws[valid], dtype=float)
        return self

    def ratio(self):
        # Extract values from the contingency table
        grp0_yes_draw, grp0_no_draw, grp1_yes_draw, grp1_no_draw = self.tbl.ravel()

        # Odds ratio should come back as a noisy float
        return _odds_ratio(grp0_yes_draw, grp0_no_draw, grp1_yes_draw, grp1_no_draw)
