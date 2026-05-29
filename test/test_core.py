import numpy as np
import pytest
import sympy as sp

from sympy.stats import Normal
from sympy.stats.rv import random_symbols

from src.core import NoisyFloat
from src.core import NoisyInt
from src.core import Node
from src.core import as_noisy_float
from src.core import as_noisy_int
from src.core import noisy_value_sampler
from src.core import float_array_sampler
from src.core import sample_noisy_values
from src.core import sample_float_array
from src.util import fresh_name


def _rooted_float(obs, expr, thetas=(), eqns=()):
    eqns = tuple(sp.sympify(eqn) for eqn in eqns)
    theta_nodes = tuple(
        Node(symbol=sp.sympify(theta), depends_on=(), constraints=(), law=None, role="latent")
        for theta in sorted(set(thetas), key=str)
    )
    random_rvs = set(random_symbols(expr)) | {
        rv for eqn in eqns for rv in random_symbols(eqn)
    }
    noise_nodes = tuple(
        Node(symbol=rv, depends_on=(), constraints=(), law=rv, role="noise")
        for rv in sorted(random_rvs, key=str)
    )
    root = Node(
        symbol=sp.Symbol(f"root_{fresh_name()}"),
        depends_on=theta_nodes + noise_nodes,
        constraints=eqns,
        law=None,
        role="derived",
    )
    return NoisyFloat.from_node(obs=obs, root=root, expr=expr)


def test_joint_sampling_preserves_shared_latent_dependency():
    theta = sp.Symbol("theta")
    eps_obs = Normal("eps_obs", 0, 1)

    constraints = [theta + eps_obs - 1.0]
    noisy_a = _rooted_float(obs=0.0, expr=theta, thetas={theta}, eqns=constraints)
    noisy_b = _rooted_float(obs=0.0, expr=2.0 * theta, thetas={theta}, eqns=constraints)

    draws_a, draws_b = sample_noisy_values(noisy_a, noisy_b, n=2000, rng=123)

    assert draws_a.shape == (2000,)
    assert draws_b.shape == (2000,)
    assert np.allclose(draws_b, 2.0 * draws_a)


def test_joint_sampling_returns_single_array_for_single_value():
    x = as_noisy_float(7.0)
    draws = sample_noisy_values(x, n=5, rng=123)

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (5,)
    assert np.all(draws == 7.0)


def test_literal_conversions_use_native_root_model():
    converted = as_noisy_float(7.0)

    assert isinstance(converted.root, Node)
    assert converted.root.constraints == ()
    assert float(converted) == 7.0


def test_integer_noisy_values_sample_as_integers():
    count = as_noisy_int(3)
    draws = sample_noisy_values(count, n=5, rng=123)

    assert isinstance(draws, np.ndarray)
    assert draws.dtype == int
    assert np.all(draws == 3)


def test_prepared_sampler_preserves_shared_latent_dependency():
    theta = sp.Symbol("theta")
    eps_obs = Normal("eps_obs_prepared", 0, 1)

    constraints = [theta + eps_obs - 1.0]
    noisy_a = _rooted_float(obs=0.0, expr=theta, thetas={theta}, eqns=constraints)
    noisy_b = _rooted_float(obs=0.0, expr=2.0 * theta, thetas={theta}, eqns=constraints)

    prepared = noisy_value_sampler(noisy_a, noisy_b)
    draws_a, draws_b = prepared.sample(n=2000, rng=123)

    assert draws_a.shape == (2000,)
    assert draws_b.shape == (2000,)
    assert np.allclose(draws_b, 2.0 * draws_a)


def test_prepared_sampler_matches_direct_sampling_for_same_seed():
    theta = sp.Symbol("theta_match")
    eps_obs = Normal("eps_obs_match", 0, 1)

    constraints = [theta + eps_obs - 1.0]
    noisy_a = _rooted_float(obs=0.0, expr=theta, thetas={theta}, eqns=constraints)
    noisy_b = _rooted_float(obs=0.0, expr=theta + 3.0, thetas={theta}, eqns=constraints)

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

    constraints = [theta + eps_obs - 1.0]
    a = _rooted_float(obs=0.0, expr=theta, thetas={theta}, eqns=constraints)
    b = _rooted_float(obs=0.0, expr=2.0 * theta, thetas={theta}, eqns=constraints)
    table = np.array([[a, b]], dtype=object)

    draws = sample_float_array(table, n=300, rng=123)
    assert draws.shape == (1, 2, 300)
    assert np.allclose(draws[0, 1, :], 2.0 * draws[0, 0, :])


def test_prepared_shaped_sampler_moves_sample_axis():
    table = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=object)
    prepared = float_array_sampler(table)
    draws = prepared.sample(n=7, rng=123, axis=0)

    assert draws.shape == (7, 2, 2)


def test_node_is_immutable_and_validates_role():
    theta = sp.Symbol("theta_node")
    node = Node(symbol=theta, role="latent")

    with pytest.raises(Exception):
        node.role = "noise"

    with pytest.raises(ValueError):
        Node(symbol=theta, role="not_a_role")


def test_noisyvalue_from_node_uses_root_and_constraints():
    theta = sp.Symbol("theta_from_node")
    root = Node(
        symbol=theta,
        depends_on=(),
        constraints=(theta - 2.0,),
        law=None,
        role="latent",
    )

    value = NoisyFloat.from_node(obs=2.0, root=root)

    assert value.root is root
    assert value.root.latent_symbols() == {theta}
    assert value.root.all_constraints() == (theta - 2.0,)


def test_sampler_uses_root_constraints():
    theta = sp.Symbol("theta_root_only")
    root = Node(
        symbol=theta,
        depends_on=(),
        constraints=(theta - 3.5,),
        law=None,
        role="latent",
    )

    value = NoisyFloat.from_node(obs=3.5, root=root, expr=theta)
    draws = value.sample(n=6, rng=123)

    assert draws.shape == (6,)
    assert np.all(draws == 3.5)


def test_sampler_resolves_multilayer_law_dependencies():
    z1_symbol = sp.Symbol("z1_layered")
    z2_symbol = sp.Symbol("z2_layered")

    z1 = Node(
        symbol=z1_symbol,
        depends_on=(),
        constraints=(),
        law=Normal("law_z1_layered", 0, 1),
        role="noise",
    )
    z2 = Node(
        symbol=z2_symbol,
        depends_on=(z1,),
        constraints=(),
        law=Normal("law_z2_layered", z1_symbol, 1),
        role="noise",
    )
    root = Node(
        symbol=sp.Symbol(f"root_{fresh_name()}"),
        depends_on=(z2,),
        constraints=(),
        law=None,
        role="derived",
    )

    value = NoisyFloat.from_node(obs=0.0, root=root, expr=z2_symbol)
    draws = value.sample(n=4000, rng=123)

    assert draws.shape == (4000,)
    assert np.all(np.isfinite(draws))
    assert np.var(draws) == pytest.approx(2.0, rel=0.25)


def test_sampler_uses_root_output_definition():
    theta = sp.Symbol("theta_stale_expr")
    root = Node(
        symbol=theta,
        depends_on=(),
        constraints=(theta - 4.0,),
        law=None,
        role="latent",
    )

    value = NoisyFloat.from_node(obs=4.0, root=root, expr=theta + 9.0)
    assert value.root.definition == theta + 9.0
    assert value.root.constraints == ()
    draws = value.sample(n=8, rng=123)

    assert draws.shape == (8,)
    assert np.all(draws == 13.0)
