import sympy as sp
import numpy as np

from sympy import Max
from sympy import Min
from sympy import Symbol
from sympy.stats import sample
from sympy.stats.rv import random_symbols

_theta_counter = 0
_noise_counter = 0


def fresh_theta(tag=None):
    global _theta_counter
    name = f"theta_{_theta_counter}" if tag is None else f"theta_{tag}_{_theta_counter}"
    _theta_counter += 1
    return Symbol(name)


def fresh_noise_name(prefix="R"):
    global _noise_counter
    name = f"{prefix}{_noise_counter}"
    _noise_counter += 1
    return name


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


class NoisyValue:
    def __init__(self, expr, observed, thetas=None, equations=None):
        self.expr = sp.sympify(expr)
        self.observed = observed
        self.thetas = set() if thetas is None else set(thetas)
        self.equations = [] if equations is None else list(equations)

    @property
    def constraints(self):
        return self.equations

    @constraints.setter
    def constraints(self, value):
        self.equations = list(value)

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

    def eliminate_thetas(self, noise_cloner=None):
        expr = self.expr
        sol = self._solve_theta_substitutions()
        substitutions = {}
        clone_cache = {}

        for theta, rhs in sol.items():
            if noise_cloner is not None:
                replacements = {}
                for rv in random_symbols(rhs):
                    if rv not in clone_cache:
                        cloned = noise_cloner(rv)
                        cloned_symbols = random_symbols(cloned)
                        if len(cloned_symbols) != 1 or cloned not in cloned_symbols:
                            raise TypeError("noise_cloner must return a single SymPy random variable")
                        clone_cache[rv] = cloned
                    replacements[rv] = clone_cache[rv]
                rhs = rhs.subs(replacements)

            substitutions[theta] = rhs

        return expr.subs(substitutions)


class NoisyFloat(NoisyValue):
    def __init__(self, expr, observed, thetas=None, equations=None):
        observed_value = float(observed)
        if equations is None:
            equations = [sp.sympify(expr) - observed_value]
        super().__init__(expr, observed_value, thetas=thetas, equations=equations)

    def __repr__(self):
        return f"NoisyFloat(expr={self.expr}, observed={self.observed})"

    @classmethod
    def from_noise_rv(cls, true_value, noise_rv, provenance=None, **sample_kwargs):
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

        theta = fresh_theta(provenance)
        measurement_expr = theta + noise_rv
        observed_expr = measurement_expr.subs({theta: sp.sympify(true_value)})
        observed = float(sample(observed_expr, **sample_kwargs))

        equations = [measurement_expr - observed]
        return cls(theta, observed, thetas={theta}, equations=equations)

    @classmethod
    def from_distribution(
        cls,
        true_value,
        dist_builder,
        *dist_args,
        provenance=None,
        name_prefix="R",
        **dist_kwargs,
    ):
        """
        Build a NoisyValue from a SymPy distribution constructor.

        Example: NoisyValue.from_distribution(10, Exponential, 2)
        """
        name = fresh_noise_name(name_prefix)
        noise_rv = dist_builder(name, *dist_args, **dist_kwargs)
        return cls.from_noise_rv(true_value, noise_rv, provenance=provenance)


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

    def minimum(self, other):
        return _combine_float(self, other, lambda a, b: Min(a, b))

    def maximum(self, other):
        return _combine_float(self, other, lambda a, b: Max(a, b))

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

    def sample_n(self, n=1000, noise_cloner=None, library="scipy", seed=None, **sample_kwargs):
        if n <= 0:
            return np.array([], dtype=float)

        sample_seed = seed
        if isinstance(seed, int):
            sample_seed = np.random.default_rng(seed)

        if noise_cloner is not None:
            expr = self.eliminate_thetas(noise_cloner=noise_cloner)
            if not random_symbols(expr):
                return np.full(n, float(expr), dtype=float)

            try:
                values = sample(expr, size=n, library=library, seed=sample_seed, **sample_kwargs)
                return np.asarray(values, dtype=float)
            except Exception:
                samples = []
                for _ in range(n):
                    samples.append(float(sample(expr, library=library, seed=sample_seed, **sample_kwargs)))
                return np.asarray(samples, dtype=float)

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
    def __init__(self, expr, observed, thetas=None, equations=None):
        super().__init__(expr, bool(observed), thetas=thetas, equations=equations)

    def __repr__(self):
        return f"NoisyBool(expr={self.expr}, observed={self.observed})"

    def __bool__(self):
        # TODO Should come from observed value
        raise Exception("not implemented")

    def __and__(self, other):
        return _combine_bool(self, other, lambda a, b: sp.And(a, b), lambda a, b: a and b)

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        return _combine_bool(self, other, lambda a, b: sp.Or(a, b), lambda a, b: a or b)

    def __ror__(self, other):
        return self.__or__(other)

    def __invert__(self):
        return NoisyBool(sp.Not(self.expr), not self.observed, self.thetas, self.equations)

    def sample_n(self, n=1000, noise_cloner=None, library="scipy", seed=None, **sample_kwargs):
        if n <= 0:
            return np.array([], dtype=bool)

        sample_seed = seed
        if isinstance(seed, int):
            sample_seed = np.random.default_rng(seed)

        if noise_cloner is not None:
            expr = self.eliminate_thetas(noise_cloner=noise_cloner)
            if not random_symbols(expr):
                return np.full(n, bool(expr), dtype=bool)

            samples = []
            for _ in range(n):
                samples.append(bool(_evaluate_random_expr(expr, sample_seed, library=library, **sample_kwargs)))
            return np.asarray(samples, dtype=bool)

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

    def prob_true(self, n=4000, noise_cloner=None, library="scipy", seed=None, **sample_kwargs):
        draws = self.sample_n(
            n=n,
            noise_cloner=noise_cloner,
            library=library,
            seed=seed,
            **sample_kwargs,
        )
        if draws.size == 0:
            return float("nan")
        return float(np.mean(draws))

    def decide(self, threshold=0.5, n=4000, noise_cloner=None, library="scipy", seed=None, **sample_kwargs):
        p = self.prob_true(
            n=n,
            noise_cloner=noise_cloner,
            library=library,
            seed=seed,
            **sample_kwargs,
        )
        return bool(p >= float(threshold))


def _evaluate_random_expr(expr, rng, library="scipy", **sample_kwargs):
    """Substitute one fresh numeric draw for each random symbol in expr."""
    value = expr
    for rv in random_symbols(value):
        value = value.subs(rv, float(sample(rv, library=library, seed=rng, **sample_kwargs)))
    return value


def noisy_min(*values):
    """Return a NoisyValue representing the pointwise minimum of all inputs."""
    if not values:
        raise ValueError("noisy_min requires at least one value")

    result = _as_noisy_float(values[0])
    for value in values[1:]:
        result = result.minimum(value)
    return result


def noisy_max(*values):
    """Return a NoisyValue representing the pointwise maximum of all inputs."""
    if not values:
        raise ValueError("noisy_max requires at least one value")

    result = _as_noisy_float(values[0])
    for value in values[1:]:
        result = result.maximum(value)
    return result
