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
    observed_expr = measurement.subs({theta: sympify(true_value)})
    observed = float(sample(observed_expr, **sample_kwargs))

    equations = [measurement - observed]
    return NoisyFloat(observed, theta, thetas={theta}, eqns=equations)
