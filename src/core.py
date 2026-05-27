import sympy as sp
import numpy as np
from dataclasses import dataclass
from typing import Any

from sympy import Abs
from sympy import And
from sympy import Not
from sympy import Or
from sympy import sympify
from sympy.stats import sample
from sympy.stats.rv import random_symbols


def _combine_float(a, b, op):
    b = as_noisy_float(b)
    obs = op(a._obs, b._obs)
    expr = op(a._expr, b._expr)
    thetas = a._thetas | b._thetas
    eqns = a._eqns + b._eqns
    return NoisyFloat(obs, expr, thetas, eqns)


def _combine_bool(a, b, obs_op, expr_op):
    b = as_noisy_bool(b)
    obs = obs_op(a._obs, b._obs)
    expr = expr_op(a._expr, b._expr)
    thetas = a._thetas | b._thetas
    eqns = a._eqns + b._eqns
    return NoisyBool(obs, expr, thetas, eqns)


def _compare_float(a, b, op):
    b = as_noisy_float(b)
    obs = op(a._obs, b._obs)
    expr = op(a._expr, b._expr)
    thetas = a._thetas | b._thetas
    eqns = a._eqns + b._eqns
    return NoisyBool(obs, expr, thetas, eqns)


def _lift_unary_bool(x, obs_fn, expr_fn):
    x = as_noisy_bool(x)
    return NoisyBool(obs_fn(bool(x._obs)), expr_fn(x._expr), x._thetas, x._eqns)


def _lift_unary_float(x, obs_fn, expr_fn):
    x = as_noisy_float(x)
    return NoisyFloat(obs_fn(float(x._obs)), expr_fn(x._expr), x._thetas, x._eqns)


def _solve_theta_substitutions(thetas, eqns):
    if not thetas:
        return {}

    equations = [sp.Eq(eq, 0) for eq in eqns]
    theta_list = list(thetas)

    sol = sp.solve(equations, theta_list, dict=True)
    chosen = sol[0]
    missing = set(thetas) - set(chosen.keys())
    if missing:
        raise ValueError(f"Latent variables are underidentified: {missing}")

    return chosen


def as_noisy_bool(value):
    if isinstance(value, NoisyBool):
        return value
    if isinstance(value, (bool, np.bool_)):
        return NoisyBool(value, sympify(bool(value)), [], [])
    raise TypeError(f"Expected bool or NoisyBool, got {type(value).__name__}")


def as_noisy_float(value):
    if isinstance(value, NoisyFloat):
        return value
    expr = sympify(value)
    return NoisyFloat(float(expr), expr, [], [])


def as_noisy_float_array(array):
    values = np.asarray(array, dtype=object)
    flat = values.reshape(-1)
    converted = np.array([as_noisy_float(value) for value in flat], dtype=object)
    return converted.reshape(values.shape)


def as_noisy_value(value):
    if isinstance(value, NoisyValue):
        return value
    if isinstance(value, (bool, np.bool_)):
        return as_noisy_bool(value)
    return as_noisy_float(value)


def noisy_value_sampler(
    *values: Any,
    library: str = "scipy",
    **sample_kwargs: Any,
):
    """Prepare a reusable joint sampler for one or more noisy values.

    The returned object caches symbolic setup work and can be reused for
    repeated `sample_n` calls with different sample sizes or RNG seeds.
    """
    if not values:
        raise ValueError("At least one value is required")

    noisy_values = tuple(as_noisy_value(value) for value in values)

    all_thetas = set().union(*(value._thetas for value in noisy_values))
    all_eqns = [eqn for value in noisy_values for eqn in value._eqns]
    theta_substitutions = _solve_theta_substitutions(all_thetas, all_eqns)

    rhs_noise_vars = {
        rv for rhs in theta_substitutions.values() for rv in random_symbols(rhs)
    }
    predictive_noise_vars = {
        rv for value in noisy_values for rv in random_symbols(value._expr)
    }
    all_noise_vars = sorted(rhs_noise_vars | predictive_noise_vars, key=str)

    return NoisyValueSampler(
        nvs=noisy_values,
        subst=theta_substitutions,
        vars=all_noise_vars,
        lib=library,
        sample_kwargs=sample_kwargs,
    )

def sample_noisy_values(
    *values: Any,
    n: int = 1000,
    library: str = "scipy",
    rng: Any = None,
    **sample_kwargs: Any,
):
    """Jointly sample one or more noisy values.

    Shared latent variables and shared random symbols are sampled once per draw,
    then reused across all requested values to preserve dependencies.
    """
    prepared = noisy_value_sampler(*values, library=library, **sample_kwargs)
    return prepared.sample(n=n, rng=rng)


def float_array_sampler(
    values: Any,
    library: str = "scipy",
    **sample_kwargs: Any,
):
    """Prepare a reusable sampler for tensor-like value collections.

    The prepared sampler returns arrays with the same base shape as `values`
    plus one sample axis.
    """
    values_array = np.asarray(values, dtype=object)
    if values_array.size == 0:
        raise ValueError("At least one value is required")

    prepared = noisy_value_sampler(
        *values_array.reshape(-1).tolist(),
        library=library,
        **sample_kwargs,
    )
    return FloatArraySampler(delegate=prepared, shape=values_array.shape)


def sample_float_array(
    values: Any,
    n: int = 1000,
    library: str = "scipy",
    rng: Any = None,
    axis: int = -1,
    **sample_kwargs: Any,
) -> np.ndarray:
    """Jointly sample a tensor-like collection of values.

    Returns a float numpy array with shape `values.shape + (n,)` by default.
    Use `axis` to move the sample dimension.
    """
    prepared = float_array_sampler(values, library=library, **sample_kwargs)
    return prepared.sample(n=n, rng=rng, sample_axis=axis)


class NoisyValue:
    def __init__(self, obs, expr, thetas, eqns):
        self._obs = obs
        self._expr = sympify(expr)
        self._thetas = frozenset(thetas)
        self._eqns = tuple(eqns)

    def __repr__(self):
        return f"~{self._obs})"

    def _solve_theta_substitutions(self):
        return _solve_theta_substitutions(self._thetas, self._eqns)

    def sample(self, n=1000, rng=None):
        return noisy_value_sampler(self).sample(n, rng)


class NoisyFloat(NoisyValue):
    def __init__(self, obs, expr, thetas, eqns):
        super().__init__(float(obs), expr, thetas, eqns)

    def __float__(self):
        return self._obs

    def __int__(self):
        return int(self._obs)

    def __abs__(self):
        return _lift_unary_float(self, abs, Abs)

    def __add__(self, other):
        return _combine_float(self, other, lambda a, b: a + b)

    def __radd__(self, other):
        return _combine_float(self, other, lambda a, b: b + a)

    def __sub__(self, other):
        return _combine_float(self, other, lambda a, b: a - b)

    def __rsub__(self, other):
        return _combine_float(self, other, lambda a, b: b - a)

    def __mul__(self, other):
        return _combine_float(self, other, lambda a, b: a * b)

    def __rmul__(self, other):
        return _combine_float(self, other, lambda a, b: b * a)

    def __truediv__(self, other):
        return _combine_float(self, other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return _combine_float(self, other, lambda a, b: b / a)

    def __lt__(self, other):
        return _compare_float(self, other, lambda a, b: a < b)

    def __le__(self, other):
        return _compare_float(self, other, lambda a, b: a <= b)

    def __gt__(self, other):
        return _compare_float(self, other, lambda a, b: a > b)

    def __ge__(self, other):
        return _compare_float(self, other, lambda a, b: a >= b)

    def __eq__(self, other):
        return _compare_float(self, other, lambda a, b: a == b)

    def __ne__(self, other):
        return _compare_float(self, other, lambda a, b: a != b)

    def exp(self):
        return _lift_unary_float(self, np.exp, sp.exp)

    def log(self):
        return _lift_unary_float(self, np.log, sp.log)

    def sqrt(self):
        return _lift_unary_float(self, np.sqrt, sp.sqrt)


class NoisyBool(NoisyValue):
    def __init__(self, obs, expr, thetas, eqns):
        super().__init__(bool(obs), expr, thetas, eqns)

    def __bool__(self):
        return self._obs

    def __and__(self, other):
        return _combine_bool(self, other, lambda a, b: a and b, And)

    def __rand__(self, other):
        return _combine_bool(self, other, lambda a, b: b and a, And)

    def __or__(self, other):
        return _combine_bool(self, other, lambda a, b: a or b, Or)

    def __ror__(self, other):
        return _combine_bool(self, other, lambda a, b: b or a, Or)

    def __invert__(self):
        return _lift_unary_bool(self, lambda a: not a, Not)


class NoisyValueSampler:
    def __init__(self, nvs, subst, vars, lib="scipy", **kwargs):
        self.noisy_values = tuple(nvs)
        self.theta_substitutions = dict(subst)
        self.all_noise_vars = tuple(vars)
        self.library = lib
        self.sample_kwargs = dict(kwargs or {})

    def __post_init__(self) -> None:
        object.__setattr__(self, "noisy_values", tuple(self.noisy_values))
        object.__setattr__(self, "theta_substitutions", dict(self.theta_substitutions))
        object.__setattr__(self, "all_noise_vars", tuple(self.all_noise_vars))
        object.__setattr__(self, "sample_kwargs", dict(self.sample_kwargs or {}))

    def sample(self, n: int = 1000, rng: Any = None):
        dtypes = tuple(type(value._obs) for value in self.noisy_values)

        if n <= 0:
            empty = tuple(np.array([], dtype=dtype) for dtype in dtypes)
            return empty[0] if len(empty) == 1 else empty

        if self.all_noise_vars:
            noise_draws = {
                rv: np.asarray(
                    sample(
                        rv,
                        size=n,
                        library=self.library,
                        seed=rng,
                        **self.sample_kwargs,
                    )
                )
                for rv in self.all_noise_vars
            }
        else:
            noise_draws = {}

        outputs = [np.empty(n, dtype=dtype) for dtype in dtypes]

        for idx in range(n):
            draws = {rv: noise_draws[rv][idx] for rv in self.all_noise_vars}
            theta_values = {
                theta: rhs.subs(draws)
                for theta, rhs in self.theta_substitutions.items()
            }

            for out_idx, noisy_value in enumerate(self.noisy_values):
                sampled_expr = noisy_value._expr.subs(theta_values).subs(draws)
                outputs[out_idx][idx] = dtypes[out_idx](sampled_expr)

        result = tuple(outputs)
        return result[0] if len(result) == 1 else result


class FloatArraySampler:
    def __init__(self, delegate, shape):
        self._delegate = delegate
        self._shape = shape

    def sample(self, n=1000, rng=None, sample_axis=-1):
    
        raw = self._delegate.sample(n=n, rng=rng)
        if isinstance(raw, tuple):
            flat = np.stack(raw, axis=0)
        else:
            flat = raw[np.newaxis, :]

        shaped = np.asarray(flat.reshape(self._shape + (n,)), dtype=float)
        if sample_axis == -1:
            return shaped

        ndim = len(self._shape) + 1
        axis = sample_axis if sample_axis >= 0 else sample_axis + ndim
        if axis < 0 or axis >= ndim:
            raise np.AxisError(sample_axis, ndim=ndim)
        return np.moveaxis(shaped, -1, axis)
