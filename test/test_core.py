import numpy as np
import pytest
import src.noise as noise
import sympy as sp

from numpy.random import default_rng
from conftest import rooted_float
from src.core import NoisyFloat
from src.core import NoisyInt
from src.core import Node
from src.core import noisy_value_sampler
from src.core import float_array_sampler
from src.core import sample_noisy_values
from src.core import sample_float_array
from src.core import NoisyFloat
from src.core import Node
from sympy.stats import Binomial
from sympy.stats import Normal


_rng_factory = lambda: default_rng(42)


def test_joint_sampling_preserves_shared_latent_dependency():
    theta_node = Node.latent()
    theta = theta_node.symbol
    eps_obs_node = Node.noise(law=Normal("eps_obs", 0, 1))
    eps_obs = eps_obs_node.symbol

    constraints = [theta + eps_obs - 1.0]
    noisy_a = rooted_float(obs=0.0, expr=theta, eqns=constraints, depends_on=(theta_node, eps_obs_node))
    noisy_b = rooted_float(obs=0.0, expr=2.0 * theta, eqns=constraints, depends_on=(theta_node, eps_obs_node))

    batch_a, batch_b = sample_noisy_values(noisy_a, noisy_b, n=2000, rng=123)
    draws_a = batch_a.draws
    draws_b = batch_b.draws

    assert draws_a.shape == (2000,)
    assert draws_b.shape == (2000,)
    assert np.allclose(draws_b, 2.0 * draws_a)


def test_literal_conversions_use_native_root_model():
    converted = NoisyFloat.lift(7.0)

    assert isinstance(converted._root, Node)
    assert converted._root.constraints == ()
    assert float(converted) == 7.0


def test_integer_noisy_values_sample_as_integers():
    count = NoisyInt.lift(3)
    draws = sample_noisy_values(count, n=5, rng=123)[0].draws

    assert isinstance(draws, np.ndarray)
    assert draws.dtype == int
    assert np.all(draws == 3)


def test_prepared_sampler_preserves_shared_latent_dependency():
    theta_node = Node.latent()
    theta = theta_node.symbol
    eps_obs_node = Node.noise(law=Normal("eps_obs_prepared", 0, 1))
    eps_obs = eps_obs_node.symbol

    constraints = [theta + eps_obs - 1.0]
    noisy_a = rooted_float(obs=0.0, expr=theta, eqns=constraints, depends_on=(theta_node, eps_obs_node))
    noisy_b = rooted_float(obs=0.0, expr=2.0 * theta, eqns=constraints, depends_on=(theta_node, eps_obs_node))

    prepared = noisy_value_sampler(noisy_a, noisy_b)
    batch_a, batch_b = prepared.sample(n=2000, rng=123)
    draws_a = batch_a.draws
    draws_b = batch_b.draws

    assert draws_a.shape == (2000,)
    assert draws_b.shape == (2000,)
    assert np.allclose(draws_b, 2.0 * draws_a)


def test_prepared_sampler_matches_direct_sampling_for_same_seed():
    theta_node = Node.latent()
    theta = theta_node.symbol
    eps_obs_node = Node.noise(law=Normal("eps_obs_match", 0, 1))
    eps_obs = eps_obs_node.symbol

    constraints = [theta + eps_obs - 1.0]
    noisy_a = rooted_float(obs=0.0, expr=theta, eqns=constraints, depends_on=(theta_node, eps_obs_node))
    noisy_b = rooted_float(obs=0.0, expr=theta + 3.0, eqns=constraints, depends_on=(theta_node, eps_obs_node))

    direct_a, direct_b = sample_noisy_values(noisy_a, noisy_b, n=250, rng=777)
    prepared = noisy_value_sampler(noisy_a, noisy_b)
    prepared_a, prepared_b = prepared.sample(n=250, rng=777)

    assert np.allclose(prepared_a.draws, direct_a.draws)
    assert np.allclose(prepared_b.draws, direct_b.draws)


def test_sample_shaped_returns_table_shape_plus_sample_axis():
    table = np.array([
        [NoisyFloat.lift(1.0), NoisyFloat.lift(2.0)],
        [NoisyFloat.lift(3.0), NoisyFloat.lift(4.0)]], dtype=object)
    draws = sample_float_array(table, n=11, rng=123)

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (2, 2, 11)
    assert draws.dtype == float


def test_sample_shaped_preserves_shared_dependency_across_cells():
    theta_node = Node.latent()
    theta = theta_node.symbol
    eps_obs_node = Node.noise(law=Normal("eps_obs_table", 0, 1))
    eps_obs = eps_obs_node.symbol

    constraints = [theta + eps_obs - 1.0]
    a = rooted_float(obs=0.0, expr=theta, eqns=constraints, depends_on=(theta_node, eps_obs_node))
    b = rooted_float(obs=0.0, expr=2.0 * theta, eqns=constraints, depends_on=(theta_node, eps_obs_node))
    table = np.array([[a, b]], dtype=object)

    draws = sample_float_array(table, n=300, rng=123)
    assert draws.shape == (1, 2, 300)
    assert np.allclose(draws[0, 1, :], 2.0 * draws[0, 0, :])


def test_prepared_shaped_sampler_moves_sample_axis():
    table = np.array([
        [NoisyFloat.lift(1.0), NoisyFloat.lift(2.0)],
        [NoisyFloat.lift(3.0), NoisyFloat.lift(4.0)]], dtype=object)
    prepared = float_array_sampler(table)
    draws = prepared.sample(n=7, rng=123, axis=0)

    assert draws.shape == (7, 2, 2)


def test_node_factories_reject_direct_role_override():
    theta = sp.Symbol("theta_node")

    with pytest.raises(TypeError):
        Node.latent(symbol=theta)


def test_noisyvalue_from_node_uses_root_and_constraints():
    theta_node = Node.latent()
    theta = theta_node.symbol
    root = Node.derived(
        constraints=(theta - 2.0,),
        definition=theta,
        depends_on=(theta_node,),
    )

    value = NoisyFloat.from_node(obs=2.0, root=root)

    assert value._root is root
    assert value._root.latent_symbols() == {theta}
    assert value._root.all_constraints() == (theta - 2.0,)


def test_sampler_uses_root_constraints():
    theta_node = Node.latent()
    theta = theta_node.symbol
    root = Node.derived(
        constraints=(theta - 3.5,),
        definition=theta,
        depends_on=(theta_node,),
    )

    value = NoisyFloat.from_node(obs=3.5, root=root, expr=theta)
    draws = value.sample(n=6, rng=123).draws

    assert draws.shape == (6,)
    assert np.all(draws == 3.5)


def test_noisyfloat_round_nearest_for_deterministic_value():
    value = NoisyFloat.lift(2.6)
    rounded = value.round_nearest()

    assert isinstance(rounded, NoisyInt)
    assert int(rounded) == 3
    draws = rounded.sample(n=5, rng=123).draws
    assert np.all(draws == 3)


def test_noisyfloat_round_nearest_tie_uses_floor_plus_half_rule():
    theta_node = Node.latent()
    theta = theta_node.symbol
    root = Node.derived(
        constraints=(theta - 2.5,),
        definition=theta,
        depends_on=(theta_node,),
    )

    value = NoisyFloat.from_node(obs=2.5, root=root, expr=theta)
    rounded = value.round_nearest()

    assert int(rounded) == 3
    draws = rounded.sample(n=4, rng=123).draws
    assert np.all(draws == 3)


def test_noisyfloat_divide_by_zero_returns_inf_observation():
    symbol = sp.Symbol("symbol")
    node = Node.derived(definition=symbol)
    x = NoisyFloat(1.0, node)
    y = NoisyFloat(0.0, node)

    z = x / y

    assert isinstance(z, NoisyFloat)
    assert np.isinf(float(z))


def test_noisyfloat_zero_divide_zero_returns_nan_observation():
    symbol = sp.Symbol("symbol")
    node = Node.derived(definition=symbol)
    x = NoisyFloat(0.0, node)
    y = NoisyFloat(0.0, node)

    z = x / y

    assert isinstance(z, NoisyFloat)
    assert np.isnan(float(z))


def test_noisyfloat_guarded_returns_value_when_guard_true():
    x = NoisyFloat.lift(3.5)

    guarded = x.guarded(True)

    assert isinstance(guarded, NoisyFloat)
    assert float(guarded) == pytest.approx(3.5)
    draws = guarded.sample(n=5, rng=123).draws
    assert np.all(draws == 3.5)


def test_noisyfloat_guarded_returns_nan_when_guard_false():
    x = NoisyFloat.lift(3.5)

    guarded = x.guarded(False)

    assert isinstance(guarded, NoisyFloat)
    assert np.isnan(float(guarded))
    draws = guarded.sample(n=5, rng=123).draws
    assert np.all(np.isnan(draws))


def test_noisyfloat_guarded_preserves_uncertainty_when_guard_is_noisy_bool():
    theta_node = Node.latent()
    theta = theta_node.symbol
    eps_node = Node.noise(law=Normal("eps_guarded", 0, 1))
    eps = eps_node.symbol
    constraints = [theta + eps - 4.0]

    value = rooted_float(obs=4.0, expr=theta, eqns=constraints, depends_on=(theta_node, eps_node))
    guarded = value.guarded(value > 0)

    assert isinstance(guarded, NoisyFloat)
    draws = guarded.sample(n=128, rng=123).draws
    finite = draws[np.isfinite(draws)]
    assert draws.shape == (128,)
    assert finite.size > 0


def test_noisyfloat_power_supports_plain_exponent():
    x = NoisyFloat.lift(3.0)

    z = x ** 2

    assert isinstance(z, NoisyFloat)
    assert float(z) == pytest.approx(9.0)


def test_noisyfloat_reverse_power_supports_plain_base():
    x = NoisyFloat.lift(3.0)

    z = 2.0 ** x

    assert isinstance(z, NoisyFloat)
    assert float(z) == pytest.approx(8.0)


def test_noisyfloat_invalid_real_power_returns_nan_observation():
    x = NoisyFloat.lift(-1.0)
    y = NoisyFloat.lift(0.5)

    z = x ** y

    assert isinstance(z, NoisyFloat)
    assert np.isnan(float(z))


def test_noisyint_resample_replaces_value_with_new_law():
    count = NoisyInt.lift(5)
    resampled = count.resample(Binomial("k_resample", 2, 0.5))

    assert isinstance(resampled, NoisyInt)
    assert int(resampled) == 5
    assert any(node.role == "noise" and node.law is not None for node in resampled._root.closure())

    draws = resampled.sample(n=128, rng=123).draws
    assert draws.dtype == int
    assert np.all(draws >= 0)
    assert np.all(draws <= 2)


def test_noisyint_resample_preserves_upstream_dependency():
    theta_node = Node.latent()
    theta = theta_node.symbol
    root = Node.derived(
        constraints=(theta - 3.0,),
        definition=theta,
        depends_on=(theta_node,),
    )
    count = NoisyInt.from_node(obs=3, root=root, expr=theta)

    resampled = count.resample(Binomial("k_resample_theta", theta, 0.5))

    assert theta in resampled._root.latent_symbols()


def test_noisyint_resample_invalid_binomial_parameter_yields_nan_draws():
    theta_node = Node.latent()
    theta = theta_node.symbol
    root = Node.derived(
        constraints=(theta - 1.5,),
        definition=theta,
        depends_on=(theta_node,),
    )
    count = NoisyInt.from_node(obs=3, root=root, expr=sp.Integer(10))
    resampled = count.resample(Binomial("k_bad_binomial", 10, theta))

    noisy_float = NoisyFloat.from_node(obs=float(int(resampled)), root=resampled._root, expr=resampled._root.symbol)
    draws = noisy_float.sample(n=16, rng=123).draws

    assert draws.shape == (16,)
    assert np.all(np.isnan(draws))


def test_sampler_resolves_multilayer_law_dependencies():
    z1 = Node.noise(
        law=Normal("law_z1_layered", 0, 1),
    )
    z1_symbol = z1.symbol
    z2 = Node.noise(
        law=Normal("law_z2_layered", z1_symbol, 1),
        depends_on=(z1,),
    )
    root = Node.derived(
        definition=z2.symbol,
        depends_on=(z2,),
    )

    value = NoisyFloat.from_node(obs=0.0, root=root, expr=z2.symbol)
    draws = value.sample(n=4000, rng=123).draws

    assert draws.shape == (4000,)
    assert np.all(np.isfinite(draws))
    assert np.var(draws) == pytest.approx(2.0, rel=0.25)


def test_sampler_uses_root_output_definition():
    theta_node = Node.latent()
    theta = theta_node.symbol
    root = Node.derived(
        constraints=(theta - 4.0,),
        definition=theta,
        depends_on=(theta_node,),
    )

    value = NoisyFloat.from_node(obs=4.0, root=root, expr=theta + 9.0)
    assert value._root.definition == theta + 9.0
    assert value._root.constraints == ()
    draws = value.sample(n=8, rng=123).draws

    assert draws.shape == (8,)
    assert np.all(draws == 13.0)


def test_node_derived_uses_explicit_dependencies():
    theta = Node.latent()

    eps_rv = Normal("eps_gb", 0, 1)
    eps = Node.noise(law=eps_rv)

    value = Node.derived(definition=theta.symbol + eps.symbol, depends_on=(theta, eps))

    assert {dep.symbol for dep in value.depends_on} == {theta.symbol, eps.symbol}


def test_node_noise_uses_explicit_dependencies():
    theta = Node.latent()

    law = Normal("z_law_gb", theta.symbol, 1)
    z = Node.noise(law=law, depends_on=(theta,))

    assert {dep.symbol for dep in z.depends_on} == {theta.symbol}


def test_node_uses_fresh_symbols_for_each_node():
    first = Node.latent()
    second = Node.latent()

    assert first.symbol != second.symbol


def test_node_derived_wires_wrapper_and_symbol_nodes_explicitly():
    theta_node = Node.latent()
    theta = theta_node.symbol
    base_root = Node.latent(constraints=(theta - 2.0,), definition=theta)
    wrapped = NoisyFloat.from_node(obs=3.0, root=base_root, expr=theta + 1.0)

    out = Node.derived(definition=theta + 2.0, depends_on=(wrapped._root, theta_node))

    dep_symbols = {dep.symbol for dep in out.depends_on}
    assert wrapped._root.symbol in dep_symbols
    assert theta in dep_symbols


def test_draw_obs():
    '''
    The observed value of a NoisyFloat is its true value plus noise.
    '''
    noise_factory = noise.gaussian(loc=0, scale=1)()
    noisy_float = NoisyFloat.draw(5.0, noise_factory, seed=_rng_factory())
    expected_noise = _rng_factory().normal(loc=0, scale=1)
    assert float(noisy_float) == expected_noise + 5.0


def test_draw_uses_root_node():
    noise_factory = noise.gaussian(loc=0, scale=1)()
    noisy_float = NoisyFloat.draw(5.0, noise_factory, seed=_rng_factory())

    assert isinstance(noisy_float._root, Node)
    assert noisy_float._root.role == "derived"
    assert noisy_float._root.latent_symbols()
