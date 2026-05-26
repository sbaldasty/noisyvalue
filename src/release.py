import numpy as np

from .core import NoisyFloat
from .util import fresh_name
from sympy import Symbol
from sympy import sympify
from sympy.stats import sample
from sympy.stats.rv import random_symbols


def noisy_float(true_value, noise_factory, **sample_kwargs):
    noise_rv = noise_factory()

    if set(random_symbols(noise_rv)) != {noise_rv}:
        raise TypeError("noise_factory must return a random variable")

    theta = Symbol(fresh_name())
    measurement = theta + noise_rv
    obs_expr = measurement.subs({theta: sympify(true_value)})
    obs = float(sample(obs_expr, **sample_kwargs))
    eqns = [measurement - obs]

    return NoisyFloat(obs, theta, {theta}, eqns)


def noisy_float_array(true_tbl, noise_factory, **sample_kwargs):
    values = np.asarray(true_tbl, dtype=object)
    noisy_flat = np.array(
        [noisy_float(value, noise_factory, **sample_kwargs) for value in values.reshape(-1)],
        dtype=object,
    )
    return noisy_flat.reshape(values.shape)