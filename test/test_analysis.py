import numpy as np

import src.analysis as analysis
import src.noise as noise
from statsmodels.stats.contingency_tables import Table2x2


def test_noisy_odds_ratio_matches_closed_form_for_plain_floats():
    result = analysis.noisy_odds_ratio(65.0, 109.0, 243.0, 1348.0, correction=0.0)
    expected = (65.0 * 1348.0) / (109.0 * 243.0)

    assert float(result) == expected


def test_oddsratio_confint_noisy_matches_statsmodels_for_plain_floats():
    ci_low, ci_high = analysis.oddsratio_confint_noisy(
        65.0,
        109.0,
        243.0,
        1348.0,
        n=1000,
        seed=123,
        correction=0.0,
    )

    sampled_table = np.array([[65.0, 109.0], [243.0, 1348.0]], dtype=float)
    expected_low, expected_high = Table2x2(sampled_table, shift_zeros=False).oddsratio_confint()

    assert ci_low == expected_low
    assert ci_high == expected_high


def test_quantile_ci_without_sampling_collapses_for_plain_floats():
    q_low, q_high = analysis.oddsratio_confint_noisy_quantile(
        65.0,
        109.0,
        243.0,
        1348.0,
        n=1000,
        seed=123,
        correction=0.0,
        include_sampling=False,
    )

    expected_or = (65.0 * 1348.0) / (109.0 * 243.0)
    assert q_low == expected_or
    assert q_high == expected_or


def test_quantile_ci_with_sampling_keeps_sampling_uncertainty():
    q_low, q_high = analysis.oddsratio_confint_noisy_quantile(
        65.0,
        109.0,
        243.0,
        1348.0,
        n=4000,
        seed=123,
        correction=0.0,
        include_sampling=True,
    )

    assert q_low < q_high
    assert q_low > 0


def test_quantile_ci_with_noisy_inputs_runs_and_returns_float_bounds():
    rng = np.random.default_rng(42)
    noise_factory = noise.gaussian(loc=0, scale=1)

    a = analysis._as_noisy_float(65.0)
    b = analysis._as_noisy_float(109.0)
    c = analysis._as_noisy_float(243.0)
    d = analysis._as_noisy_float(1348.0)

    # Add explicit release noise layer to inputs.
    from src.release import noisy_float

    a = noisy_float(float(a), noise_factory, seed=rng)
    b = noisy_float(float(b), noise_factory, seed=rng)
    c = noisy_float(float(c), noise_factory, seed=rng)
    d = noisy_float(float(d), noise_factory, seed=rng)

    q_low, q_high = analysis.oddsratio_confint_noisy_quantile(
        a,
        b,
        c,
        d,
        n=2000,
        seed=123,
        correction=0.0,
        include_sampling=True,
    )

    assert isinstance(q_low, float)
    assert isinstance(q_high, float)
    assert q_low < q_high
