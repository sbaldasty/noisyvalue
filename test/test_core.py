import numpy as np
import pytest
import sympy as sp

from sympy.stats import Binomial
from sympy.stats import Normal
from sympy.stats.rv import random_symbols

from src.core import NoisyFloat
from src.core import NoisyBool
from src.core import NoisyInt
from src.core import Node
from src.core import _derived_node
from src.core import _latent_node
from src.core import _noise_node
from src.core import as_noisy_bool
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


def test_noisybool_true_false_constants_exist_and_are_boolean():
    assert isinstance(NoisyBool.TRUE, NoisyBool)
    assert isinstance(NoisyBool.FALSE, NoisyBool)
    assert bool(NoisyBool.TRUE)
    assert not bool(NoisyBool.FALSE)


def test_noisybool_true_false_constants_match_literal_conversion_singletons():
    assert NoisyBool.TRUE is as_noisy_bool(True)
    assert NoisyBool.FALSE is as_noisy_bool(False)


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


def test_noisyfloat_round_nearest_for_deterministic_value():
    value = as_noisy_float(2.6)
    rounded = value.round_nearest()

    assert isinstance(rounded, NoisyInt)
    assert int(rounded) == 3
    draws = rounded.sample(n=5, rng=123)
    assert np.all(draws == 3)


def test_noisyfloat_round_nearest_tie_uses_floor_plus_half_rule():
    theta = sp.Symbol("theta_round_tie")
    root = Node(
        symbol=theta,
        depends_on=(),
        constraints=(theta - 2.5,),
        law=None,
        role="latent",
    )

    value = NoisyFloat.from_node(obs=2.5, root=root, expr=theta)
    rounded = value.round_nearest()

    assert int(rounded) == 3
    draws = rounded.sample(n=4, rng=123)
    assert np.all(draws == 3)


def test_noisyfloat_divide_by_zero_returns_inf_observation():
    x = as_noisy_float(1.0)
    y = as_noisy_float(0.0)

    z = x / y

    assert isinstance(z, NoisyFloat)
    assert np.isinf(float(z))


def test_noisyfloat_zero_divide_zero_returns_nan_observation():
    x = as_noisy_float(0.0)
    y = as_noisy_float(0.0)

    z = x / y

    assert isinstance(z, NoisyFloat)
    assert np.isnan(float(z))


def test_noisyfloat_power_supports_plain_exponent():
    x = as_noisy_float(3.0)

    z = x ** 2

    assert isinstance(z, NoisyFloat)
    assert float(z) == pytest.approx(9.0)


def test_noisyfloat_reverse_power_supports_plain_base():
    x = as_noisy_float(3.0)

    z = 2.0 ** x

    assert isinstance(z, NoisyFloat)
    assert float(z) == pytest.approx(8.0)


def test_noisyfloat_invalid_real_power_returns_nan_observation():
    x = as_noisy_float(-1.0)
    y = as_noisy_float(0.5)

    z = x ** y

    assert isinstance(z, NoisyFloat)
    assert np.isnan(float(z))


def test_noisyint_resample_replaces_value_with_new_law():
    count = as_noisy_int(5)
    resampled = count.resample(Binomial("k_resample", 2, 0.5))

    assert isinstance(resampled, NoisyInt)
    assert int(resampled) == 5
    assert any(node.role == "noise" and node.law is not None for node in resampled.root.closure())

    draws = resampled.sample(n=128, rng=123)
    assert draws.dtype == int
    assert np.all(draws >= 0)
    assert np.all(draws <= 2)


def test_noisyint_resample_preserves_upstream_dependency():
    theta = sp.Symbol("theta_resample")
    root = Node(
        symbol=theta,
        depends_on=(),
        constraints=(theta - 3.0,),
        law=None,
        role="latent",
    )
    count = NoisyInt.from_node(obs=3, root=root, expr=theta)

    resampled = count.resample(Binomial("k_resample_theta", theta, 0.5))

    assert theta in resampled.root.latent_symbols()


def test_noisyint_resample_invalid_binomial_parameter_yields_nan_draws():
    theta = sp.Symbol("theta_bad_binomial")
    root = Node(
        symbol=theta,
        depends_on=(),
        constraints=(theta - 1.5,),
        law=None,
        role="latent",
    )
    count = NoisyInt.from_node(obs=3, root=root, expr=sp.Integer(10))
    resampled = count.resample(Binomial("k_bad_binomial", 10, theta))

    noisy_float = NoisyFloat.from_node(obs=float(int(resampled)), root=resampled.root, expr=resampled.root.symbol)
    draws = noisy_float.sample(n=16, rng=123)

    assert draws.shape == (16,)
    assert np.all(np.isnan(draws))


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


def test_registry_infers_derived_dependencies_from_definition():
    theta = _latent_node("theta_gb", constraints=(sp.Symbol("theta_gb") - 1.0,))

    eps_rv = Normal("eps_gb", 0, 1)
    eps = _noise_node(eps_rv, law=eps_rv)

    value = _derived_node("value_gb", definition=theta.symbol + eps.symbol)

    assert {dep.symbol for dep in value.depends_on} == {theta.symbol, eps.symbol}


def test_registry_infers_noise_dependencies_from_law_parameters():
    theta = _latent_node("theta_law_gb", constraints=(sp.Symbol("theta_law_gb") - 2.0,))

    law = Normal("z_law_gb", theta.symbol, 1)
    z = _noise_node("z_gb", law=law)

    assert {dep.symbol for dep in z.depends_on} == {theta.symbol}


def test_registry_rejects_duplicate_symbol_registration():
    existing = _latent_node("dup_gb")

    with pytest.raises(ValueError, match="already registered"):
        _derived_node("dup_gb", definition=sp.Integer(1))

    assert existing.symbol == sp.Symbol("dup_gb")


def test_registry_auto_includes_wrapper_roots_from_expression_symbols():
    theta = sp.Symbol("theta_gb_input")
    base_root = Node(
        symbol=theta,
        depends_on=(),
        constraints=(theta - 2.0,),
        law=None,
        role="latent",
    )
    wrapped = NoisyFloat.from_node(obs=3.0, root=base_root, expr=theta + 1.0)

    out = _derived_node("out_gb_input", definition=theta + 2.0)

    dep_symbols = {dep.symbol for dep in out.depends_on}
    assert wrapped.root.symbol in dep_symbols
    assert theta in dep_symbols
