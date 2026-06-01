from . import noise
from .core import NoisyFloat
from .core import as_noisy_float
from .core import _preferred_value_expr
from .core import _combine_float
from .core import as_noisy_float_array
from .core import _derived_node
from sympy import Piecewise
from sympy import nan
from numpy import asarray
from numpy import isfinite
from sympy import Max
from sympy import Min


def _fold_float(values, op):
    if not values:
        raise ValueError("Requires at least one value")

    result = as_noisy_float(values[0])
    for value in values[1:]:
        result = _combine_float(result, as_noisy_float(value), op)
    return result


def odds_ratio(tbl):
    # Must have 2x2 table of noisy floats with postive finite observations
    tbl = as_noisy_float_array(tbl)
    assert tbl.shape == (2, 2)
    assert isfinite(asarray([float(value) for value in tbl.ravel()], dtype=float)).all()

    # Compute totals and ratios as noisy floats for each group
    grp0_yes, grp0_no, grp1_yes, grp1_no = tbl.ravel()
    grp0_total = (grp0_yes + grp0_no).round_nearest()
    grp1_total = (grp1_yes + grp1_no).round_nearest()
    grp0_ratio = grp0_yes / (grp0_yes + grp0_no)
    grp1_ratio = grp1_yes / (grp1_yes + grp1_no)

    # Observation for the result is calculation over inputs or NaN if nonpositive counts
    obs_valid = bool((grp0_yes > 0) & (grp0_no > 0) & (grp1_yes > 0) & (grp1_no > 0))
    obs_or = float((grp0_yes * grp1_no) / (grp0_no * grp1_yes)) if obs_valid else nan

    # Chain symbolic binomial distributions for sampling uncertainty
    grp0_yes_draw = grp0_yes.round_nearest().resample(noise.binomial(grp0_total, grp0_ratio))
    grp1_yes_draw = grp1_yes.round_nearest().resample(noise.binomial(grp1_total, grp1_ratio))
    grp0_no_draw = grp0_total - grp0_yes_draw
    grp1_no_draw = grp1_total - grp1_yes_draw

    # Basically deferred sample filtering
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
    root = _derived_node(definition=expr)
    return NoisyFloat.from_node(obs_or, root)


def noisy_min(*values):
    return _fold_float(values, Min)


def noisy_max(*values):
    return _fold_float(values, Max)
