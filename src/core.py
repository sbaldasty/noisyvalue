import sympy as sp
import numpy as np

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


class NoisyValue:
    def __init__(self, obs, expr, thetas, eqns):
        self.expr = sympify(expr)
        self.obs = obs
        self.thetas = set(thetas)
        self.eqns = list(eqns)

    def __repr__(self):
        return f"~{self.obs})"

    def _solve_theta_substitutions(self):
        if not self.thetas:
            return {}
        if not self.eqns:
            raise ValueError("No equations available to solve for latent variables")

        eqs = [sp.Eq(eq, 0) for eq in self.eqns]
        thetas = list(self.thetas)

        sol = sp.solve(eqs, thetas, dict=True)
        if not sol:
            raise ValueError("Could not solve for latent variables")

        chosen = sol[0]
        missing = self.thetas - set(chosen.keys())
        if missing:
            raise ValueError(f"Latent variables are underidentified: {missing}")

        return chosen

    def sample_n(self, n=1000, library="scipy", seed=None, **sample_kwargs):
        dtype = type(self.obs)
        if n <= 0:
            return np.array([], dtype=dtype)

        sample_seed = seed
        if isinstance(seed, int):
            sample_seed = np.random.default_rng(seed)

        if not self.thetas:
            expr = self.expr
            if not random_symbols(expr):
                return np.full(n, dtype(expr), dtype=dtype)
            values = sample(expr, size=n, library=library, seed=sample_seed, **sample_kwargs)
            return np.asarray(values, dtype=dtype)

        sol = self._solve_theta_substitutions()
        rhs_noise_vars = list({rv for rhs in sol.values() for rv in random_symbols(rhs)})
        predictive_noise_vars = list(random_symbols(self.expr))

        samples = []
        for _ in range(n):
            rhs_noise_draws = {
                rv: float(sample(rv, library=library, seed=sample_seed, **sample_kwargs))
                for rv in rhs_noise_vars
            }
            theta_values = {
                theta: float(rhs.subs(rhs_noise_draws))
                for theta, rhs in sol.items()
            }
            predictive_noise_draws = {
                rv: float(sample(rv, library=library, seed=sample_seed, **sample_kwargs))
                for rv in predictive_noise_vars
            }

            value = dtype(self.expr.subs(theta_values).subs(predictive_noise_draws))
            samples.append(value)

        return np.asarray(samples, dtype=dtype)


class NoisyFloat(NoisyValue):
    def __init__(self, obs, expr, thetas, eqns):
        super().__init__(float(obs), expr, thetas, eqns)

    def __float__(self):
        return self.obs

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
