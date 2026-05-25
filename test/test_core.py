import numpy as np
import sympy as sp

from sympy.stats import Normal

from src.core import NoisyFloat
from src.core import prepare_sampler
from src.core import sample_n


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

    draws_a, draws_b = sample_n(noisy_a, noisy_b, n=2000, rng=123)

    assert draws_a.shape == (2000,)
    assert draws_b.shape == (2000,)
    assert np.allclose(draws_b, 2.0 * draws_a)


def test_joint_sampling_returns_single_array_for_single_value():
    x = NoisyFloat(obs=7.0, expr=7.0, thetas=set(), eqns=[])
    draws = sample_n(x, n=5, rng=123)

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

    prepared = prepare_sampler(noisy_a, noisy_b)
    draws_a, draws_b = prepared.sample_n(n=2000, rng=123)

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

    direct_a, direct_b = sample_n(noisy_a, noisy_b, n=250, rng=777)
    prepared = prepare_sampler(noisy_a, noisy_b)
    prepared_a, prepared_b = prepared.sample_n(n=250, rng=777)

    assert np.allclose(prepared_a, direct_a)
    assert np.allclose(prepared_b, direct_b)
