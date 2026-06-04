from . import noise
from .core import NoisyBool
from .core import as_noisy_float
from .core import _combine_float
from .core import as_noisy_float_array
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


def as_contingency_table(tbl):
    # Elements should be noisy floats if they aren't already
    tbl = as_noisy_float_array(tbl)
    # Must be 2D with positive dimensions
    assert tbl.ndim == 2
    assert tbl.shape[0] > 0 and tbl.shape[1] > 0
    # All observations must be finite
    assert isfinite(asarray([float(value) for value in tbl.ravel()], dtype=float)).all()
    # Return a fresh copy
    return tbl


def chi_squared(tbl):
    tbl = as_contingency_table(tbl)

    n_rows, n_cols = tbl.shape
    row_totals = tuple(sum(tbl[i, :]) for i in range(n_rows))
    col_totals = tuple(sum(tbl[:, j]) for j in range(n_cols))
    total = sum(row_totals)

    stat = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            expected = (row_totals[i] * col_totals[j]) / total
            stat += (tbl[i, j] - expected) ** 2 / expected

    valid = NoisyBool.TRUE
    for row_total in row_totals:
        valid &= row_total > 0
    for col_total in col_totals:
        valid &= col_total > 0
    for cell in tbl.ravel():
        valid &= cell >= 0

    return stat.guarded(valid)


def contingency_table_predictive(tbl):
    tbl = as_contingency_table(tbl)

    n_rows, n_cols = tbl.shape
    predictive = []

    for i in range(n_rows):
        row = tuple(tbl[i, :])
        row_draws = []
        remaining_total = sum(row).round_nearest()
        remaining_mass = sum(row)

        for j in range(n_cols - 1):
            cell = row[j]
            prob = cell / remaining_mass
            draw = cell.round_nearest().resample(noise.binomial(remaining_total, prob))
            remaining_total = remaining_total - draw
            remaining_mass = remaining_mass - cell
            row_draws.append(draw)

        row_draws.append(remaining_total)
        predictive.append(row_draws)

    return as_noisy_float_array(predictive)


def odds_ratio(tbl):
    tbl = as_contingency_table(tbl)
    assert tbl.shape == (2, 2)

    grp0_yes, grp0_no, grp1_yes, grp1_no = tbl.ravel()
    stat = (grp0_yes * grp1_no) / (grp0_no * grp1_yes)
    valid = (grp0_yes > 0) & (grp0_no > 0) & (grp1_yes > 0) & (grp1_no > 0)
    return stat.guarded(valid)


def noisy_min(*values):
    return _fold_float(values, Min)


def noisy_max(*values):
    return _fold_float(values, Max)
