import numpy as np
import pytest
import sympy as sp
from scipy.stats import chi2_contingency

import src.analysis as analysis

from src.core import NoisyFloat
from src.core import Node
from sympy.stats import Normal
from sympy.stats.rv import random_symbols
from src.util import fresh_name


def _rooted_float(obs, expr, thetas=(), eqns=()):
    eqns = tuple(sp.sympify(eqn) for eqn in eqns)
    theta_nodes = tuple(
        Node(symbol=sp.sympify(theta), depends_on=(), constraints=(), law=None, definition=None, role="latent")
        for theta in sorted(set(thetas), key=str)
    )
    random_rvs = set(random_symbols(expr)) | {
        rv for eqn in eqns for rv in random_symbols(eqn)
    }
    noise_nodes = tuple(
        Node(symbol=rv, depends_on=(), constraints=(), law=rv, definition=None, role="noise")
        for rv in sorted(random_rvs, key=str)
    )
    root = Node(
        symbol=sp.Symbol(f"root_{fresh_name()}"),
        depends_on=theta_nodes + noise_nodes,
        constraints=eqns,
        law=None,
        definition=None,
        role="derived",
    )
    return NoisyFloat.from_node(obs=obs, root=root, expr=expr)


def test_noisy_min_and_noisy_max_for_plain_floats_match_python_min_max():
    lo = analysis.noisy_min(3.0, -1.5, 8.0)
    hi = analysis.noisy_max(3.0, -1.5, 8.0)

    assert isinstance(lo, NoisyFloat)
    assert isinstance(hi, NoisyFloat)
    assert float(lo) == -1.5
    assert float(hi) == 8.0


def test_noisy_min_raises_for_empty_input():
    with pytest.raises(ValueError, match="Requires at least one value"):
        analysis.noisy_min()


def test_noisy_max_combines_noisy_value_metadata():
    theta = sp.Symbol("theta_fold")
    constraints = [theta - 1.0]
    a = _rooted_float(obs=1.0, expr=theta, thetas={theta}, eqns=constraints)
    b = _rooted_float(obs=2.0, expr=2.0 * theta, thetas={theta}, eqns=constraints)

    out = analysis.noisy_max(a, b)

    assert isinstance(out, NoisyFloat)
    assert float(out) == 2.0
    assert out.root.latent_symbols() == {theta}
    assert len(out.root.all_constraints()) >= 2


def test_odds_ratio_enforces_2x2_shape():
    with pytest.raises(AssertionError):
        analysis.odds_ratio([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_chi_squared_enforces_2d_shape():
    with pytest.raises(AssertionError):
        analysis.chi_squared([1.0, 2.0, 3.0])


def test_chi_squared_matches_scipy_for_plain_floats():
    table = [[65.0, 109.0], [243.0, 1348.0]]
    result = analysis.chi_squared(table)
    expected = chi2_contingency(table, correction=False).statistic

    assert isinstance(result, NoisyFloat)
    assert float(result) == pytest.approx(expected)


def test_chi_squared_returns_nan_when_a_row_has_no_mass():
    result = analysis.chi_squared([[1.0, 2.0], [0.0, 0.0]])

    assert isinstance(result, NoisyFloat)
    assert np.isnan(float(result))


def test_chi_squared_builds_single_noisy_float_with_propagated_uncertainty():
    theta = sp.Symbol("theta_chi_squared")
    noisy_a = _rooted_float(obs=5.0, expr=theta, thetas={theta}, eqns=[theta - 5.0])

    result = analysis.chi_squared([[noisy_a, 7.0], [11.0, 13.0]])

    assert isinstance(result, NoisyFloat)
    assert theta in result.root.latent_symbols()

    draws = result.sample(n=128, rng=123)
    assert draws.shape == (128,)
    assert np.all(np.isfinite(draws))


def test_odds_ratio_matches_closed_form_for_plain_floats():
    ratio = analysis.odds_ratio([[65.0, 109.0], [243.0, 1348.0]])
    expected = (65.0 * 1348.0) / (109.0 * 243.0)

    assert isinstance(ratio, NoisyFloat)
    assert float(ratio) == pytest.approx(expected)


def test_odds_ratio_sample_keeps_only_valid_draws():
    ratio = analysis.odds_ratio([[5.0, 7.0], [11.0, 13.0]])

    draws = ratio.sample(n=400, rng=123)
    finite = draws[np.isfinite(draws)]

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (400,)
    assert finite.size > 0
    assert np.all(finite > 0.0)


def test_odds_ratio_sample_with_zero_n_returns_empty_array():
    ratio = analysis.odds_ratio([[1.0, 2.0], [3.0, 4.0]])

    draws = ratio.sample(n=0, rng=123)

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (0,)


def test_odds_ratio_builds_single_noisy_float_with_propagated_uncertainty():
    theta = sp.Symbol("theta_odds_ratio")
    eps = Normal("eps_odds_ratio", 0, 1)

    noisy_a = _rooted_float(obs=5.0, expr=theta, thetas={theta}, eqns=[theta + eps - 5.0])

    ratio = analysis.odds_ratio([[noisy_a, 7.0], [11.0, 13.0]])

    assert isinstance(ratio, NoisyFloat)
    assert theta in ratio.root.latent_symbols()
    assert any(node.role == "noise" and node.law is not None for node in ratio.root.closure())

    draws = ratio.sample(n=128, rng=123)
    assert draws.shape == (128,)
    assert np.all(np.isfinite(draws))


def test_odds_ratio_returns_nan_observation_when_observed_ratio_is_invalid():
    ratio = analysis.odds_ratio([[1.0, 0.0], [2.0, 3.0]])

    assert isinstance(ratio, NoisyFloat)
    assert np.isnan(float(ratio))
    assert isinstance(ratio.root, Node)


def test_odds_ratio_sampling_handles_out_of_range_noisy_probabilities():
    # This table implies grp0_ratio = 5 / (5 + -1) = 1.25, which is out of
    # bounds for Binomial p. Sampling should stay robust and return NaNs.
    ratio = analysis.odds_ratio([[5.0, -1.0], [11.0, 13.0]])

    draws = ratio.sample(n=64, rng=123)

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (64,)
    assert np.all(np.isnan(draws))
