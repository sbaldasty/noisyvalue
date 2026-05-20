from sympy import Symbol
from sympy import sympify
from sympy.stats import sample
from sympy.stats.rv import random_symbols

from .core import NoisyFloat

_theta_counter = 0
_noise_counter = 0


def _fresh_theta():
    global _theta_counter
    name = f"theta_{_theta_counter}"
    _theta_counter += 1
    return Symbol(name)


def _fresh_noise_name(prefix="R"):
    global _noise_counter
    name = f"{prefix}{_noise_counter}"
    _noise_counter += 1
    return name


def from_noise_rv(true_value, noise_rv, **sample_kwargs):
    """
    Build a NoisyValue from any SymPy random variable.

    The returned `expr` is the latent value (`theta`), while the measurement
    mechanism is encoded in `equations` as `theta + noise - observed = 0`.
    This makes downstream sampling reflect analyst belief about the true
    quantity rather than release-to-release spread.
    """
    noise_symbols = random_symbols(noise_rv)
    if len(noise_symbols) != 1 or noise_rv not in noise_symbols:
        raise TypeError("noise_rv must be a single SymPy random variable")

    theta = _fresh_theta()
    measurement_expr = theta + noise_rv
    observed_expr = measurement_expr.subs({theta: sympify(true_value)})
    observed = float(sample(observed_expr, **sample_kwargs))

    equations = [measurement_expr - observed]
    return NoisyFloat(observed, theta, thetas={theta}, eqns=equations)


def from_distribution(
    true_value,
    dist_builder,
    *dist_args,
    name_prefix="R",
    **dist_kwargs,
):
    """
    Build a NoisyValue from a SymPy distribution constructor.

    Example: NoisyValue.from_distribution(10, Exponential, 2)
    """
    name = _fresh_noise_name(name_prefix)
    noise_rv = dist_builder(name, *dist_args, **dist_kwargs)
    return from_noise_rv(true_value, noise_rv)
