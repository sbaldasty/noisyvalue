from .core import NoisyFloat
from .core import NoisyInt
from .core import Unknown
from .core import as_noisy_float
from .core import _preferred_value_expr
from .core import _combine_float
from .core import as_noisy_float_array
from .core import sample_float_array
import sympy as sp
from .util import fresh_name
from sympy import And
from sympy import Piecewise
from sympy import Symbol
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
    root = NoisyInt.from_unknown(obs, Unknown(symbol=expr, depends_on=(), constraints=(), law=expr, role="noise"), expr=expr)
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
    return type(result).from_unknown(float(result), result.root, expr=expr)


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
        grp0_yes, grp0_no, grp1_yes, grp1_no = self.tbl.ravel()

        grp0_yes_expr = _preferred_value_expr(grp0_yes)
        grp0_no_expr = _preferred_value_expr(grp0_no)
        grp1_yes_expr = _preferred_value_expr(grp1_yes)
        grp1_no_expr = _preferred_value_expr(grp1_no)

        g0_yes_obs = float(grp0_yes)
        g0_no_obs = float(grp0_no)
        g1_yes_obs = float(grp1_yes)
        g1_no_obs = float(grp1_no)

        grp0_obs_total = int(round(g0_yes_obs + g0_no_obs))
        grp1_obs_total = int(round(g1_yes_obs + g1_no_obs))
        if grp0_obs_total <= 0 or grp1_obs_total <= 0:
            return None

        grp0_obs_ratio = g0_yes_obs / (g0_yes_obs + g0_no_obs)
        grp1_obs_ratio = g1_yes_obs / (g1_yes_obs + g1_no_obs)
        if not (0.0 <= grp0_obs_ratio <= 1.0 and 0.0 <= grp1_obs_ratio <= 1.0):
            return None

        grp0_yes_obs_draw = int(round(grp0_obs_total * grp0_obs_ratio))
        grp1_yes_obs_draw = int(round(grp1_obs_total * grp1_obs_ratio))
        grp0_no_obs_draw = grp0_obs_total - grp0_yes_obs_draw
        grp1_no_obs_draw = grp1_obs_total - grp1_yes_obs_draw
        if min(grp0_yes_obs_draw, grp0_no_obs_draw, grp1_yes_obs_draw, grp1_no_obs_draw) <= 0:
            return None

        obs_or = (grp0_yes_obs_draw * grp1_no_obs_draw) / (grp0_no_obs_draw * grp1_yes_obs_draw)
        if not isfinite(obs_or) or obs_or <= 0.0:
            return None

        grp0_total_expr = sp.floor(grp0_yes_expr + grp0_no_expr + sp.Rational(1, 2))
        grp1_total_expr = sp.floor(grp1_yes_expr + grp1_no_expr + sp.Rational(1, 2))
        grp0_ratio_expr = grp0_yes_expr / (grp0_yes_expr + grp0_no_expr)
        grp1_ratio_expr = grp1_yes_expr / (grp1_yes_expr + grp1_no_expr)

        grp0_yes_symbol = Symbol(fresh_name())
        grp1_yes_symbol = Symbol(fresh_name())

        grp0_yes_node = Unknown(
            symbol=grp0_yes_symbol,
            depends_on=(grp0_yes.root, grp0_no.root),
            constraints=(),
            law=Binomial(fresh_name(), grp0_total_expr, grp0_ratio_expr),
            role="noise",
        )
        grp1_yes_node = Unknown(
            symbol=grp1_yes_symbol,
            depends_on=(grp1_yes.root, grp1_no.root),
            constraints=(),
            law=Binomial(fresh_name(), grp1_total_expr, grp1_ratio_expr),
            role="noise",
        )

        grp0_no_expr = grp0_total_expr - grp0_yes_symbol
        grp1_no_expr = grp1_total_expr - grp1_yes_symbol

        valid = And(
            grp0_total_expr > 0,
            grp1_total_expr > 0,
            grp0_yes_expr + grp0_no_expr > 0,
            grp1_yes_expr + grp1_no_expr > 0,
            grp0_ratio_expr >= 0,
            grp0_ratio_expr <= 1,
            grp1_ratio_expr >= 0,
            grp1_ratio_expr <= 1,
            grp0_yes_symbol > 0,
            grp0_no_expr > 0,
            grp1_yes_symbol > 0,
            grp1_no_expr > 0,
        )

        expr = Piecewise(
            ((grp0_yes_symbol * grp1_no_expr) / (grp0_no_expr * grp1_yes_symbol), valid),
            (nan, True),
        )

        root = Unknown(
            symbol=Symbol(fresh_name()),
            depends_on=(
                grp0_yes.root,
                grp0_no.root,
                grp1_yes.root,
                grp1_no.root,
                grp0_yes_node,
                grp1_yes_node,
            ),
            constraints=(),
            law=None,
            role="derived",
        )

        return NoisyFloat.from_unknown(obs_or, root, expr=expr)

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
