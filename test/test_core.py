import numpy as np
import sympy as sp

from sympy.stats import Normal

from src.core import NoisyFloat
from src.core import noisy_value_sampler
from src.core import float_array_sampler
from src.core import sample_noisy_values
from src.core import sample_float_array


def test_joint_sampling_preserves_shared_latent_dependency():
    theta = sp.Symbol("theta")
    eps_obs = Normal("eps_obs", 0, 1)

    noisy_a = NoisyFloat(
        expr=theta,
        obs=0.0,
        thetas={theta},
        eqns=[theta + eps_obs - 1.0],
    )
    noisy_b = NoisyFloat(
        expr=2.0 * theta,
        obs=0.0,
        thetas={theta},
        eqns=[theta + eps_obs - 1.0],
    )

    draws_a, draws_b = sample_noisy_values(noisy_a, noisy_b, n=2000, rng=123)

    assert draws_a.shape == (2000,)
    assert draws_b.shape == (2000,)
    assert np.allclose(draws_b, 2.0 * draws_a)


def test_joint_sampling_returns_single_array_for_single_value():
    x = NoisyFloat(obs=7.0, expr=7.0, thetas=set(), eqns=[])
    draws = sample_noisy_values(x, n=5, rng=123)

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (5,)
    assert np.all(draws == 7.0)


def test_prepared_sampler_preserves_shared_latent_dependency():
    theta = sp.Symbol("theta")
    eps_obs = Normal("eps_obs_prepared", 0, 1)

    noisy_a = NoisyFloat(
        expr=theta,
        obs=0.0,
        thetas={theta},
        eqns=[theta + eps_obs - 1.0],
    )
    noisy_b = NoisyFloat(
        expr=2.0 * theta,
        obs=0.0,
        thetas={theta},
        eqns=[theta + eps_obs - 1.0],
    )

    prepared = noisy_value_sampler(noisy_a, noisy_b)
    draws_a, draws_b = prepared.sample(n=2000, rng=123)

    assert draws_a.shape == (2000,)
    assert draws_b.shape == (2000,)
    assert np.allclose(draws_b, 2.0 * draws_a)


def test_prepared_sampler_matches_direct_sampling_for_same_seed():
    theta = sp.Symbol("theta_match")
    eps_obs = Normal("eps_obs_match", 0, 1)

    noisy_a = NoisyFloat(
        expr=theta,
        obs=0.0,
        thetas={theta},
        eqns=[theta + eps_obs - 1.0],
    )
    noisy_b = NoisyFloat(
        expr=theta + 3.0,
        obs=0.0,
        thetas={theta},
        eqns=[theta + eps_obs - 1.0],
    )

    direct_a, direct_b = sample_noisy_values(noisy_a, noisy_b, n=250, rng=777)
    prepared = noisy_value_sampler(noisy_a, noisy_b)
    prepared_a, prepared_b = prepared.sample(n=250, rng=777)

    assert np.allclose(prepared_a, direct_a)
    assert np.allclose(prepared_b, direct_b)


def test_sample_shaped_returns_table_shape_plus_sample_axis():
    table = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=object)
    draws = sample_float_array(table, n=11, rng=123)

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (2, 2, 11)
    assert draws.dtype == float


def test_sample_shaped_preserves_shared_dependency_across_cells():
    theta = sp.Symbol("theta_table")
    eps_obs = Normal("eps_obs_table", 0, 1)

    a = NoisyFloat(expr=theta, obs=0.0, thetas={theta}, eqns=[theta + eps_obs - 1.0])
    b = NoisyFloat(expr=2.0 * theta, obs=0.0, thetas={theta}, eqns=[theta + eps_obs - 1.0])
    table = np.array([[a, b]], dtype=object)

    draws = sample_float_array(table, n=300, rng=123)
    assert draws.shape == (1, 2, 300)
    assert np.allclose(draws[0, 1, :], 2.0 * draws[0, 0, :])


def test_prepared_shaped_sampler_moves_sample_axis():
    table = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=object)
    prepared = float_array_sampler(table)
    draws = prepared.sample(n=7, rng=123, axis=0)

    assert draws.shape == (7, 2, 2)
