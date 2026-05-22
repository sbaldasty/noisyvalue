import src.noise as noise
import src.release as release

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
