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


def test_as_contingency_table_returns_noisy_contingency_table():
    table = analysis.NoisyContingencyTable([[1.0, 2.0], [3.0, 4.0]])

    assert isinstance(table, analysis.NoisyContingencyTable)
    assert table.tbl.shape == (2, 2)


def test_odds_ratio_enforces_2x2_shape():
    with pytest.raises(AssertionError):
        analysis.NoisyContingencyTable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).odds_ratio()


def test_noisy_contingency_table_odds_ratio_enforces_2x2_shape():
    with pytest.raises(AssertionError):
        analysis.NoisyContingencyTable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).odds_ratio()


def test_chi_squared_enforces_2d_shape():
    with pytest.raises(AssertionError):
        analysis.NoisyContingencyTable([1.0, 2.0, 3.0]).chi_squared()


def test_chi_squared_matches_scipy_for_plain_floats():
    table = [[65.0, 109.0], [243.0, 1348.0]]
    result = analysis.NoisyContingencyTable(table).chi_squared()
    expected = chi2_contingency(table, correction=False).statistic

    assert isinstance(result, NoisyFloat)
    assert float(result) == pytest.approx(expected)


def test_noisy_contingency_table_repeatability_on_same_input():
    table_data = [[65.0, 109.0], [243.0, 1348.0]]

    table = analysis.NoisyContingencyTable(table_data)
    chi_a = table.chi_squared()
    or_a = table.odds_ratio()
    chi_b = table.chi_squared()
    or_b = table.odds_ratio()

    assert float(chi_a) == pytest.approx(float(chi_b))
    assert float(or_a) == pytest.approx(float(or_b))


def test_chi_squared_returns_nan_when_a_row_has_no_mass():
    result = analysis.NoisyContingencyTable([[1.0, 2.0], [0.0, 0.0]]).chi_squared()

    assert isinstance(result, NoisyFloat)
    assert np.isnan(float(result))


def test_chi_squared_builds_single_noisy_float_with_propagated_uncertainty():
    theta = sp.Symbol("theta_chi_squared")
    noisy_a = _rooted_float(obs=5.0, expr=theta, thetas={theta}, eqns=[theta - 5.0])

    result = analysis.NoisyContingencyTable([[noisy_a, 7.0], [11.0, 13.0]]).chi_squared()

    assert isinstance(result, NoisyFloat)
    assert theta in result.root.latent_symbols()

    draws = result.sample(n=128, rng=123).draws
    assert draws.shape == (128,)
    assert np.all(np.isfinite(draws))


def test_odds_ratio_matches_closed_form_for_plain_floats():
    ratio = analysis.NoisyContingencyTable([[65.0, 109.0], [243.0, 1348.0]]).odds_ratio()
    expected = (65.0 * 1348.0) / (109.0 * 243.0)

    assert isinstance(ratio, NoisyFloat)
    assert float(ratio) == pytest.approx(expected)


def test_odds_ratio_sample_keeps_only_valid_draws():
    ratio = analysis.NoisyContingencyTable([[5.0, 7.0], [11.0, 13.0]]).odds_ratio()
    expected = (5.0 * 13.0) / (7.0 * 11.0)

    draws = ratio.sample(n=400, rng=123).draws

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (400,)
    assert np.all(np.isfinite(draws))
    assert np.allclose(draws, expected)


def test_odds_ratio_sample_with_zero_n_returns_empty_array():
    ratio = analysis.NoisyContingencyTable([[1.0, 2.0], [3.0, 4.0]]).odds_ratio()

    draws = ratio.sample(n=0, rng=123).draws

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (0,)

def test_odds_ratio_builds_single_noisy_float_with_propagated_uncertainty():
    theta = sp.Symbol("theta_odds_ratio")
    eps = Normal("eps_odds_ratio", 0, 1)

    noisy_a = _rooted_float(obs=5.0, expr=theta, thetas={theta}, eqns=[theta + eps - 5.0])

    ratio = analysis.NoisyContingencyTable([[noisy_a, 7.0], [11.0, 13.0]]).odds_ratio()

    assert isinstance(ratio, NoisyFloat)
    assert theta in ratio.root.latent_symbols()
    assert any(node.role == "noise" and node.law is not None for node in ratio.root.closure())

    draws = ratio.sample(n=128, rng=123).draws
    assert draws.shape == (128,)
    assert np.all(np.isfinite(draws))


def test_odds_ratio_returns_nan_observation_when_observed_ratio_is_invalid():
    ratio = analysis.NoisyContingencyTable([[1.0, 0.0], [2.0, 3.0]]).odds_ratio()

    assert isinstance(ratio, NoisyFloat)
    assert np.isnan(float(ratio))
    assert isinstance(ratio.root, Node)


def test_odds_ratio_sampling_handles_out_of_range_noisy_probabilities():
    # Invalid observed counts produce NaNs through the validity gate.
    ratio = analysis.NoisyContingencyTable([[5.0, -1.0], [11.0, 13.0]]).odds_ratio()

    draws = ratio.sample(n=64, rng=123).draws

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (64,)
    assert np.all(np.isnan(draws))


def test_contingency_table_predictive_supports_general_2d_shape():
    table = [[10.0, 20.0, 30.0], [9.0, 3.0, 8.0]]

    predictive = analysis.NoisyContingencyTable(table).with_sampling_uncertainty().tbl

    assert predictive.shape == (2, 3)
    assert np.isfinite(np.asarray([float(value) for value in predictive.ravel()], dtype=float)).all()


def test_with_sampling_uncertainty_returns_noisy_contingency_table():
    table = analysis.NoisyContingencyTable([[10.0, 20.0, 30.0], [9.0, 3.0, 8.0]])

    predictive = table.with_sampling_uncertainty()

    assert isinstance(predictive, analysis.NoisyContingencyTable)
    assert predictive.tbl.shape == (2, 3)
    assert np.isfinite(np.asarray([float(value) for value in predictive.tbl.ravel()], dtype=float)).all()


def test_chi_squared_accepts_predictive_contingency_table():
    table = [[65.0, 109.0], [243.0, 1348.0]]

    stat = analysis.NoisyContingencyTable(table).with_sampling_uncertainty().chi_squared()
    draws = stat.sample(n=256, rng=123).draws

    assert isinstance(stat, NoisyFloat)
    assert draws.shape == (256,)
    assert np.isfinite(draws).all()
    assert np.std(draws) > 0.0


def test_odds_ratio_accepts_predictive_contingency_table():
    table = [[65.0, 109.0], [243.0, 1348.0]]

    ratio = analysis.NoisyContingencyTable(table).with_sampling_uncertainty().odds_ratio()
    draws = ratio.sample(n=256, rng=123).draws
    finite = draws[np.isfinite(draws)]

    assert isinstance(ratio, NoisyFloat)
    assert draws.shape == (256,)
    assert finite.size > 0
    assert np.std(finite) > 0.0
