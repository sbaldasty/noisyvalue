import numpy as np
import pytest
import conftest as noise
import sympy as sp

from conftest import rooted_float
from conftest import from_node
from noisyvalue.core import NoisyInt
from noisyvalue.sample import noisy_value_sampler
from noisyvalue.sample import sample_noisy_values
from noisyvalue.graph import DerivedNode
from noisyvalue.graph import LatentNode
from noisyvalue.graph import GaussianNode


def test_joint_sampling_preserves_shared_latent_dependency():
    theta_node = LatentNode()
    theta = theta_node.expr
    eps_obs_node = noise.gaussian(0, 1)
    eps_obs = eps_obs_node.expr

    constraints = [theta + eps_obs - 1.0]
    noisy_a = rooted_float(obs=0.0, expr=theta, eqns=constraints, deps=(theta_node, eps_obs_node))
    noisy_b = rooted_float(obs=0.0, expr=2.0 * theta, eqns=constraints, deps=(theta_node, eps_obs_node))

    batch_a, batch_b = sample_noisy_values(noisy_a, noisy_b, n=2000, rng=123)
    draws_a = batch_a.draws
    draws_b = batch_b.draws

    assert draws_a.shape == (2000,)
    assert draws_b.shape == (2000,)
    assert np.allclose(draws_b, 2.0 * draws_a)


def test_integer_noisy_values_sample_as_integers():
    count = NoisyInt.lift(3)
    draws = sample_noisy_values(count, n=5, rng=123)[0].draws

    assert isinstance(draws, np.ndarray)
    assert draws.dtype == int
    assert np.all(draws == 3)


def test_prepared_sampler_preserves_shared_latent_dependency():
    theta_node = LatentNode()
    theta = theta_node.expr
    eps_obs_node = noise.gaussian(0, 1)
    eps_obs = eps_obs_node.expr

    constraints = [theta + eps_obs - 1.0]
    noisy_a = rooted_float(obs=0.0, expr=theta, eqns=constraints, deps=(theta_node, eps_obs_node))
    noisy_b = rooted_float(obs=0.0, expr=2.0 * theta, eqns=constraints, deps=(theta_node, eps_obs_node))

    prepared = noisy_value_sampler(noisy_a, noisy_b)
    batch_a, batch_b = prepared.sample(n=2000, rng=123)
    draws_a = batch_a.draws
    draws_b = batch_b.draws

    assert draws_a.shape == (2000,)
    assert draws_b.shape == (2000,)
    assert np.allclose(draws_b, 2.0 * draws_a)


def test_prepared_sampler_matches_direct_sampling_for_same_seed():
    theta_node = LatentNode()
    theta = theta_node.expr
    eps_obs_node = noise.gaussian(0, 1)
    eps_obs = eps_obs_node.expr

    constraints = [theta + eps_obs - 1.0]
    noisy_a = rooted_float(obs=0.0, expr=theta, eqns=constraints, deps=(theta_node, eps_obs_node))
    noisy_b = rooted_float(obs=0.0, expr=theta + 3.0, eqns=constraints, deps=(theta_node, eps_obs_node))

    direct_a, direct_b = sample_noisy_values(noisy_a, noisy_b, n=250, rng=777)
    prepared = noisy_value_sampler(noisy_a, noisy_b)
    prepared_a, prepared_b = prepared.sample(n=250, rng=777)

    assert np.allclose(prepared_a.draws, direct_a.draws)
    assert np.allclose(prepared_b.draws, direct_b.draws)


def test_sampler_uses_root_constraints():
    theta_node = LatentNode()
    theta = theta_node.expr
    root = DerivedNode(
        expr=theta,
        constraints=(theta - 3.5,),
        deps=(theta_node,),
    )

    value = from_node(obs=3.5, root=root, expr=theta)
    draws = sample_noisy_values(value, n=6, rng=123)[0].draws

    assert draws.shape == (6,)
    assert np.all(draws == 3.5)


def test_sampler_resolves_multilayer_law_dependencies():
    z1 = noise.gaussian(0, 1)
    z1_symbol = z1.expr
    z2 = GaussianNode.create(deps=(z1,), loc=z1_symbol, scale=1, low=-sp.oo, high=sp.oo)
    root = DerivedNode(
        expr=z2.expr,
        deps=(z2,),
    )

    value = from_node(obs=0.0, root=root, expr=z2.expr)
    draws = sample_noisy_values(value, n=4000, rng=123)[0].draws

    assert draws.shape == (4000,)
    assert np.all(np.isfinite(draws))
    assert np.var(draws) == pytest.approx(2.0, rel=0.25)


def test_sampler_uses_root_output_definition():
    theta_node = LatentNode()
    theta = theta_node.expr
    root = DerivedNode(
        expr=theta,
        constraints=(theta - 4.0,),
        deps=(theta_node,),
    )

    value = from_node(obs=4.0, root=root, expr=theta + 9.0)
    assert value._root.expr == theta + 9.0
    assert value._root.constraints == frozenset({theta - 4.0})
    draws = sample_noisy_values(value, n=8, rng=123)[0].draws

    assert draws.shape == (8,)
    assert np.all(draws == 13.0)

