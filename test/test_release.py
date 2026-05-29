import src.noise as noise
import src.release as release
from src.core import NoisyFloat
from src.core import Node

import numpy as np

from numpy.random import default_rng


_rng_factory = lambda: default_rng(42)


def test_float_obs():
    '''
    The observed value of a NoisyFloat is its true value plus noise.
    '''
    noise_factory = noise.gaussian(loc=0, scale=1)
    noisy_float = release.noisy_float(5.0, noise_factory, seed=_rng_factory())
    expected_noise = _rng_factory().normal(loc=0, scale=1)
    assert float(noisy_float) == expected_noise + 5.0


def test_noisy_float_uses_root_node():
    noise_factory = noise.gaussian(loc=0, scale=1)
    noisy_float = release.noisy_float(5.0, noise_factory, seed=_rng_factory())

    assert isinstance(noisy_float.root, Node)
    assert noisy_float.root.role == "derived"
    assert noisy_float.root.latent_symbols()


def test_noisy_float_array_shape_and_type():
    noise_factory = noise.gaussian(loc=0, scale=1)
    true_tbl = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

    noisy_tbl = release.noisy_float_array(true_tbl, noise_factory, seed=_rng_factory())

    assert isinstance(noisy_tbl, np.ndarray)
    assert noisy_tbl.shape == true_tbl.shape
    assert noisy_tbl.dtype == object
    assert all(isinstance(value, NoisyFloat) for value in noisy_tbl.reshape(-1))
