import sympy as sp
import numpy as np

from sympy import Abs
from sympy import And
from sympy import Not
from sympy import Or
from sympy import sympify
from sympy.stats import sample
from sympy.stats.rv import random_symbols


def _as_noisy_float(value):
    if isinstance(value, NoisyFloat):
        return value
    expr = sympify(value)
    return NoisyFloat(float(expr), expr, [], [])


def _as_noisy_bool(value):
    if isinstance(value, NoisyBool):
        return value
    if isinstance(value, (bool, np.bool_)):
        return NoisyBool(value, sympify(bool(value)), [], [])
    raise TypeError(f"Expected bool or NoisyBool, got {type(value).__name__}")


def _combine_float(a, b, op):
    b = _as_noisy_float(b)
    obs = op(a.obs, b.obs)
    expr = op(a.expr, b.expr)
    thetas = a.thetas | b.thetas
    eqns = a.eqns + b.eqns
    return NoisyFloat(obs, expr, thetas, eqns)


def _combine_bool(a, b, obs_op, expr_op):
    b = _as_noisy_bool(b)
    obs = obs_op(a.obs, b.obs)
    expr = expr_op(a.expr, b.expr)
    thetas = a.thetas | b.thetas
    eqns = a.eqns + b.eqns
    return NoisyBool(obs, expr, thetas, eqns)


def _compare_float(a, b, op):
    b = _as_noisy_float(b)
    obs = op(a.obs, b.obs)
    expr = op(a.expr, b.expr)
    thetas = a.thetas | b.thetas
    eqns = a.eqns + b.eqns
    return NoisyBool(obs, expr, thetas, eqns)


def _lift_unary(x, obs_fn, expr_fn):
    x = _as_noisy_float(x)
    return NoisyFloat(obs_fn(float(x.obs)), expr_fn(x.expr), x.thetas, x.eqns)


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


def _as_noisy_value(value):
    if isinstance(value, NoisyValue):
        return value
    if isinstance(value, (bool, np.bool_)):
        return _as_noisy_bool(value)
    return _as_noisy_float(value)


class PreparedSampler:
    def __init__(
        self,
        noisy_values,
        theta_substitutions,
        all_noise_vars,
        library="scipy",
        sample_kwargs=None,
    ):
        self._noisy_values = tuple(noisy_values)
        self._dtypes = tuple(type(value.obs) for value in self._noisy_values)
        self._theta_substitutions = dict(theta_substitutions)
        self._all_noise_vars = tuple(all_noise_vars)
        self._library = library
        self._sample_kwargs = dict(sample_kwargs or {})

    def sample_n(self, n=1000, rng=None):
        if n <= 0:
            empty = tuple(np.array([], dtype=dtype) for dtype in self._dtypes)
            return empty[0] if len(empty) == 1 else empty

        if self._all_noise_vars:
            noise_draws = {
                rv: np.asarray(
                    sample(
                        rv,
                        size=n,
                        library=self._library,
                        seed=rng,
                        **self._sample_kwargs,
                    )
                )
                for rv in self._all_noise_vars
            }
        else:
            noise_draws = {}

        outputs = [np.empty(n, dtype=dtype) for dtype in self._dtypes]

        for idx in range(n):
            draws = {rv: noise_draws[rv][idx] for rv in self._all_noise_vars}
            theta_values = {
                theta: rhs.subs(draws)
                for theta, rhs in self._theta_substitutions.items()
            }

            for out_idx, noisy_value in enumerate(self._noisy_values):
                sampled_expr = noisy_value.expr.subs(theta_values).subs(draws)
                outputs[out_idx][idx] = self._dtypes[out_idx](sampled_expr)

        result = tuple(outputs)
        return result[0] if len(result) == 1 else result


def prepare_sampler(*values, library="scipy", **sample_kwargs):
    """Prepare a reusable joint sampler for one or more noisy values.

    The returned object caches symbolic setup work and can be reused for
    repeated `sample_n` calls with different sample sizes or RNG seeds.
    """
    if not values:
        raise ValueError("At least one value is required")

    noisy_values = [_as_noisy_value(value) for value in values]

    all_thetas = set().union(*(value.thetas for value in noisy_values))
    all_eqns = [eqn for value in noisy_values for eqn in value.eqns]
    theta_substitutions = _solve_theta_substitutions(all_thetas, all_eqns)

    rhs_noise_vars = {
        rv for rhs in theta_substitutions.values() for rv in random_symbols(rhs)
    }
    predictive_noise_vars = {
        rv for value in noisy_values for rv in random_symbols(value.expr)
    }
    all_noise_vars = sorted(rhs_noise_vars | predictive_noise_vars, key=str)

    return PreparedSampler(
        noisy_values=noisy_values,
        theta_substitutions=theta_substitutions,
        all_noise_vars=all_noise_vars,
        library=library,
        sample_kwargs=sample_kwargs,
    )


def sample_n(*values, n=1000, library="scipy", rng=None, **sample_kwargs):
    """Jointly sample one or more noisy values.

    Shared latent variables and shared random symbols are sampled once per draw,
    then reused across all requested values to preserve dependencies.
    """
    prepared = prepare_sampler(*values, library=library, **sample_kwargs)
    return prepared.sample_n(n=n, rng=rng)


class NoisyValue:
    def __init__(self, obs, expr, thetas, eqns):
        self.expr = sympify(expr)
        self.obs = obs
        self.thetas = set(thetas)
        self.eqns = list(eqns)

    def __repr__(self):
        return f"~{self.obs})"

    # TODO Remove this?
    def _solve_theta_substitutions(self):
        return _solve_theta_substitutions(self.thetas, self.eqns)


class NoisyFloat(NoisyValue):
    def __init__(self, obs, expr, thetas, eqns):
        super().__init__(float(obs), expr, thetas, eqns)

    def __float__(self):
        return self.obs

    def __abs__(self):
        return _lift_unary(self, abs, Abs)

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
        return _lift_unary(self, np.exp, sp.exp)

    def log(self):
        return _lift_unary(self, np.log, sp.log)

    def sqrt(self):
        return _lift_unary(self, np.sqrt, sp.sqrt)


class NoisyBool(NoisyValue):
    def __init__(self, obs, expr, thetas, eqns):
        super().__init__(bool(obs), expr, thetas, eqns)

    def __bool__(self):
        return self.obs

    def __and__(self, other):
        return _combine_bool(self, other, lambda a, b: a and b, And)

    def __rand__(self, other):
        return _combine_bool(self, other, lambda a, b: b and a, And)

    def __or__(self, other):
        return _combine_bool(self, other, lambda a, b: a or b, Or)

    def __ror__(self, other):
        return _combine_bool(self, other, lambda a, b: b or a, Or)

    def __invert__(self):
        return NoisyBool(not self.obs, Not(self.expr), self.thetas, self.eqns)
