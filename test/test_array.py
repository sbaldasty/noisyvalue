import numpy as np

from conftest import rooted_float
from src.array import float_array_sampler
from src.array import sample_float_array
from src.core import NoisyFloat
from src.graph import LatentNode
import src.graph as noise

def test_prepared_shaped_sampler_moves_sample_axis():
    table = np.array([
        [NoisyFloat.lift(1.0), NoisyFloat.lift(2.0)],
        [NoisyFloat.lift(3.0), NoisyFloat.lift(4.0)]], dtype=object)
    prepared = float_array_sampler(table)
    draws = prepared.sample(n=7, rng=123, axis=0)

    assert draws.shape == (7, 2, 2)


def test_sample_shaped_returns_table_shape_plus_sample_axis():
    table = np.array([
        [NoisyFloat.lift(1.0), NoisyFloat.lift(2.0)],
        [NoisyFloat.lift(3.0), NoisyFloat.lift(4.0)]], dtype=object)
    draws = sample_float_array(table, n=11, rng=123)

    assert isinstance(draws, np.ndarray)
    assert draws.shape == (2, 2, 11)
    assert draws.dtype == float


def test_sample_shaped_preserves_shared_dependency_across_cells():
    theta_node = LatentNode()
    theta = theta_node.symbol
    eps_obs_node = noise.gaussian(0, 1)
    eps_obs = eps_obs_node.symbol

    constraints = [theta + eps_obs - 1.0]
    a = rooted_float(obs=0.0, expr=theta, eqns=constraints, depends_on=(theta_node, eps_obs_node))
    b = rooted_float(obs=0.0, expr=2.0 * theta, eqns=constraints, depends_on=(theta_node, eps_obs_node))
    table = np.array([[a, b]], dtype=object)

    draws = sample_float_array(table, n=300, rng=123)
    assert draws.shape == (1, 2, 300)
    assert np.allclose(draws[0, 1, :], 2.0 * draws[0, 0, :])
