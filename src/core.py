import sympy as sp
import numpy as np

from sympy import And
from sympy import Not
from sympy import Or
from sympy.stats import sample
from sympy.stats.rv import random_symbols


def _as_noisy_float(value):
    if isinstance(value, NoisyFloat):
        return value
    expr = sp.sympify(value)
    return NoisyFloat(expr, float(expr), thetas=set(), equations=[])


def _as_noisy_bool(value):
    if isinstance(value, NoisyBool):
        return value
    if isinstance(value, (bool, np.bool_)):
        return NoisyBool(sp.sympify(bool(value)), bool(value), thetas=set(), equations=[])
    raise TypeError(f"Expected bool or NoisyBool, got {type(value).__name__}")


def _combine_float(a, b, op):
    b = _as_noisy_float(b)
    expr = op(a.expr, b.expr)
    observed = op(a.observed, b.observed)
    thetas = a.thetas | b.thetas
    equations = a.equations + b.equations
    return NoisyFloat(expr, observed, thetas, equations)


def _combine_bool(a, b, expr_op, observed_op):
    b = _as_noisy_bool(b)
    expr = expr_op(a.expr, b.expr)
    observed = observed_op(a.observed, b.observed)
    thetas = a.thetas | b.thetas
    equations = a.equations + b.equations
    return NoisyBool(expr, observed, thetas, equations)


def _compare_float(a, b, op):
    b = _as_noisy_float(b)
    expr = op(a.expr, b.expr)
    observed = bool(op(a.observed, b.observed))
    thetas = a.thetas | b.thetas
    equations = a.equations + b.equations
    return NoisyBool(expr, observed, thetas, equations)


def _evaluate_random_expr(expr, rng, library="scipy", **sample_kwargs):
    """Substitute one fresh numeric draw for each random symbol in expr."""
    value = expr
    for rv in random_symbols(value):
        value = value.subs(rv, float(sample(rv, library=library, seed=rng, **sample_kwargs)))
    return value


class NoisyValue:
    def __init__(self, expr, observed, thetas, equations):
        self.expr = sp.sympify(expr)
        self.observed = observed
        self.thetas = set() if thetas is None else set(thetas)
        self.equations = [] if equations is None else list(equations)

    def __repr__(self):
        return f"NoisyValue(expr={self.expr}, observed={self.observed})"

    def _solve_theta_substitutions(self):
        if not self.thetas:
            return {}
        if not self.equations:
            raise ValueError("No equations available to solve for latent variables")

        eqs = [sp.Eq(eq, 0) for eq in self.equations]
        thetas = list(self.thetas)

        sol = sp.solve(eqs, thetas, dict=True)
        if not sol:
            raise ValueError("Could not solve for latent variables")

        chosen = sol[0]
        missing = self.thetas - set(chosen.keys())
        if missing:
            raise ValueError(f"Latent variables are underidentified: {missing}")

        return chosen


class NoisyFloat(NoisyValue):
    def __init__(self, expr, observed, thetas, equations):
        observed_value = float(observed)
        if equations is None:
            equations = [sp.sympify(expr) - observed_value]
        super().__init__(expr, observed_value, thetas=thetas, equations=equations)

    def __repr__(self):
        return f"NoisyFloat(expr={self.expr}, observed={self.observed})"

    def __float__(self):
        return self.observed

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

    def sample_n(self, n=1000, library="scipy", seed=None, **sample_kwargs):
        if n <= 0:
            return np.array([], dtype=float)

        sample_seed = seed
        if isinstance(seed, int):
            sample_seed = np.random.default_rng(seed)

        if not self.thetas:
            expr = self.expr
            if not random_symbols(expr):
                return np.full(n, float(expr), dtype=float)
            values = sample(expr, size=n, library=library, seed=sample_seed, **sample_kwargs)
            return np.asarray(values, dtype=float)

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

            value = float(self.expr.subs(theta_values).subs(predictive_noise_draws))
            samples.append(value)

        return np.asarray(samples, dtype=float)


class NoisyBool(NoisyValue):
    def __init__(self, expr, observed, thetas, equations):
        super().__init__(expr, bool(observed), thetas=thetas, equations=equations)

    def __repr__(self):
        return f"NoisyBool(expr={self.expr}, observed={self.observed})"

    def __bool__(self):
        return self.observed >= 0.5

    def __and__(self, other):
        return _combine_bool(self, other, lambda a, b: And(a, b), lambda a, b: a and b)

    def __rand__(self, other):
        return _combine_bool(self, other, lambda a, b: And(b, a), lambda a, b: b and a)

    def __or__(self, other):
        return _combine_bool(self, other, lambda a, b: Or(a, b), lambda a, b: a or b)

    def __ror__(self, other):
        return _combine_bool(self, other, lambda a, b: Or(b, a), lambda a, b: b or a)

    def __invert__(self):
        return NoisyBool(Not(self.expr), not self.observed, self.thetas, self.equations)

    def sample_n(self, n=1000, library="scipy", seed=None, **sample_kwargs):
        if n <= 0:
            return np.array([], dtype=bool)

        sample_seed = seed
        if isinstance(seed, int):
            sample_seed = np.random.default_rng(seed)

        if not self.thetas:
            expr = self.expr
            if not random_symbols(expr):
                return np.full(n, bool(expr), dtype=bool)

            samples = []
            for _ in range(n):
                samples.append(bool(_evaluate_random_expr(expr, sample_seed, library=library, **sample_kwargs)))
            return np.asarray(samples, dtype=bool)

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

            value = self.expr.subs(theta_values).subs(predictive_noise_draws)
            samples.append(bool(value))

        return np.asarray(samples, dtype=bool)
