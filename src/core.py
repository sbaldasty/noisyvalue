import sympy as sp
import numpy as np

from sympy import Abs
from sympy import And
from sympy import Eq
from sympy import Not
from sympy import Or
from sympy import sympify
from sympy.stats import sample
from sympy.stats.rv import random_symbols


def _combine_float(x, y, op):
    y = as_noisy_float(y)
    obs = op(x._obs, y._obs)
    expr = op(x._expr, y._expr)
    thetas = x._thetas | y._thetas
    eqns = x._eqns + y._eqns
    return NoisyFloat(obs, expr, thetas, eqns)


def _combine_bool(x, y, obs_op, expr_op):
    y = as_noisy_bool(y)
    obs = obs_op(x._obs, y._obs)
    expr = expr_op(x._expr, y._expr)
    thetas = x._thetas | y._thetas
    eqns = x._eqns + y._eqns
    return NoisyBool(obs, expr, thetas, eqns)


def _compare_float(x, y, op):
    y = as_noisy_float(y)
    obs = op(x._obs, y._obs)
    expr = op(x._expr, y._expr)
    thetas = x._thetas | y._thetas
    eqns = x._eqns + y._eqns
    return NoisyBool(obs, expr, thetas, eqns)


def _lift_unary_bool(x, obs_fn, expr_fn):
    x = as_noisy_bool(x)
    return NoisyBool(obs_fn(x._obs), expr_fn(x._expr), x._thetas, x._eqns)


def _lift_unary_float(x, obs_fn, expr_fn):
    x = as_noisy_float(x)
    return NoisyFloat(obs_fn(x._obs), expr_fn(x._expr), x._thetas, x._eqns)


def _solve_theta_substitutions(thetas, eqns):
    if not thetas:
        return {}

    equations = [Eq(eq, 0) for eq in eqns]
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


def noisy_value_sampler(*vals, lib="scipy", **kwargs):
    """Prepare a reusable joint sampler for one or more noisy values.

    The returned object caches symbolic setup work and can be reused for
    repeated `sample_n` calls with different sample sizes or RNG seeds.
    """
    if not vals:
        raise ValueError("At least one value is required")

    noisy_values = tuple(as_noisy_value(value) for value in vals)

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

    return NoisyValueSampler(noisy_values, subs=theta_substitutions, vars=all_noise_vars, lib=lib, **kwargs)


def sample_noisy_values(*vals, n=1000, lib="scipy", rng=None, **kwargs):
    """Jointly sample one or more noisy values.

    Shared latent variables and shared random symbols are sampled once per draw,
    then reused across all requested values to preserve dependencies.
    """
    sampler = noisy_value_sampler(*vals, lib=lib, **kwargs)
    return sampler.sample(n=n, rng=rng)


def float_array_sampler(vals, lib="scipy", **kwargs):
    """Prepare a reusable sampler for tensor-like value collections.

    The prepared sampler returns arrays with the same base shape as `values`
    plus one sample axis.
    """
    values_array = np.asarray(vals, dtype=object)
    if values_array.size == 0:
        raise ValueError("At least one value is required")

    prepared = noisy_value_sampler(
        *values_array.reshape(-1).tolist(),
        lib=lib,
        **kwargs,
    )
    return FloatArraySampler(prepared, values_array.shape)


def sample_float_array(vals, n=1000, lib="scipy", rng=None, axis=-1, **kwargs):
    """Jointly sample a tensor-like collection of values.

    Returns a float numpy array with shape `values.shape + (n,)` by default.
    Use `axis` to move the sample dimension.
    """
    sampler = float_array_sampler(vals, lib=lib, **kwargs)
    return sampler.sample(n, rng, axis)


class NoisyValue:
    def __init__(self, obs, expr, thetas, eqns):
        self._obs = obs
        self._expr = sympify(expr)
        self._thetas = frozenset(thetas)
        self._eqns = tuple(eqns)

    def __repr__(self):
        return f"~{self._obs}"

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
    def __init__(self, vals, subs, vars, lib="scipy", **kwargs):
        self._vals = tuple(vals)
        self._subs = dict(subs)
        self._vars = tuple(vars)
        self._lib = lib
        self._kwargs = dict(kwargs)

    def sample(self, n=1000, rng=None):
        dtypes = tuple(type(value._obs) for value in self._vals)

        if n <= 0:
            empty = tuple(np.array([], dtype=dtype) for dtype in dtypes)
            return empty[0] if len(empty) == 1 else empty

        if self._vars:
            noise_draws = {
                rv: np.asarray(
                    sample(
                        rv,
                        size=n,
                        library=self._lib,
                        seed=rng,
                        **self._kwargs,
                    )
                )
                for rv in self._vars
            }
        else:
            noise_draws = {}

        outputs = [np.empty(n, dtype=dtype) for dtype in dtypes]

        for idx in range(n):
            draws = {rv: noise_draws[rv][idx] for rv in self._vars}
            theta_values = {
                theta: rhs.subs(draws)
                for theta, rhs in self._subs.items()
            }

            for out_idx, noisy_value in enumerate(self._vals):
                sampled_expr = noisy_value._expr.subs(theta_values).subs(draws)
                outputs[out_idx][idx] = dtypes[out_idx](sampled_expr)

        result = tuple(outputs)
        return result[0] if len(result) == 1 else result


class FloatArraySampler:
    def __init__(self, delegate, shape):
        self._delegate = delegate
        self._shape = shape

    def sample(self, n=1000, rng=None, axis=-1):
        raw = self._delegate.sample(n=n, rng=rng)
        if isinstance(raw, tuple):
            flat = np.stack(raw, axis=0)
        else:
            flat = raw[np.newaxis, :]

        shaped = np.asarray(flat.reshape(self._shape + (n,)), dtype=float)
        if axis == -1:
            return shaped

        ndim = len(self._shape) + 1
        axis = axis if axis >= 0 else axis + ndim
        if axis < 0 or axis >= ndim:
            raise np.AxisError(axis, ndim=ndim)
        return np.moveaxis(shaped, -1, axis)
