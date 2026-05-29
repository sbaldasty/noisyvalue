import numpy as np

from .core import NoisyFloat
from .core import Unknown
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
    obs_expr = theta + noise_rv
    obs = float(sample(obs_expr.subs({theta: sympify(true_value)}), **sample_kwargs))

    theta_node = Unknown(symbol=theta, depends_on=(), constraints=(), law=None, role="latent")
    noise_node = Unknown(symbol=noise_rv, depends_on=(), constraints=(), law=noise_rv, role="noise")
    root = Unknown(
        symbol=Symbol(fresh_name()),
        depends_on=(theta_node, noise_node),
        constraints=(obs_expr - obs,),
        law=None,
        role="derived",
    )

    return NoisyFloat.from_unknown(obs, root, expr=theta)


def noisy_float_array(true_tbl, noise_factory, **sample_kwargs):
    values = np.asarray(true_tbl, dtype=object)
    noisy_flat = np.array(
        [noisy_float(value, noise_factory, **sample_kwargs) for value in values.reshape(-1)],
        dtype=object,
    )
    return noisy_flat.reshape(values.shape)