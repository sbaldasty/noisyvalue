import numpy as np

from sympy import sympify
from sympy.stats import Normal
from sympy.stats.frv_types import BinomialDistribution, rv

from .core import _preferred_value_expr
from .core import NoisyFloat, NoisyInt
from .util import fresh_name
from sympy import Basic


def _to_expr(value):
    if isinstance(value, (NoisyFloat, NoisyInt)):
        return _preferred_value_expr(value)
    if isinstance(value, Basic):
        return value
    if isinstance(value, (int, float)):
        return value
    expr = sympify(value)
    if isinstance(expr, Basic):
        return expr
    raise TypeError(f"Unsupported value type: {type(value)}")


class NoiseSource:
    @property
    def free_symbols(self):
        raise NotImplementedError

    def instantiate(self, resolved):
        """Return a new NoiseSource with parameter expressions substituted."""
        raise NotImplementedError

    def sample(self, rng, size=None):
        """Sample using NumPy. Parameters must be numeric (call instantiate first)."""
        raise NotImplementedError

    def param_exprs(self):
        """Return parameter expressions as a tuple, in the order expected by sample_arrays."""
        raise NotImplementedError

    def sample_arrays(self, rng, *param_arrays):
        """Sample given already-evaluated numpy arrays for each param in param_exprs()."""
        raise NotImplementedError

    def sympy_rv(self):
        """Return a SymPy RV for this distribution (used for visualization)."""
        raise NotImplementedError


class NormalNoiseSource(NoiseSource):
    def __init__(self, loc, scale):
        self._loc = sympify(loc)
        self._scale = sympify(scale)

    @property
    def free_symbols(self):
        return self._loc.free_symbols | self._scale.free_symbols

    def instantiate(self, resolved):
        return NormalNoiseSource(self._loc.subs(resolved), self._scale.subs(resolved))

    def sample(self, rng, size=None):
        return rng.normal(float(self._loc), float(self._scale), size=size)

    def param_exprs(self):
        return (self._loc, self._scale)

    def sample_arrays(self, rng, loc, scale):
        loc = np.asarray(loc, dtype=float)
        scale = np.asarray(scale, dtype=float)
        valid = np.isfinite(loc) & (scale > 0)
        if np.all(valid):
            return rng.normal(loc, scale)
        result = rng.normal(np.where(valid, loc, 0.0), np.where(valid, scale, 1.0))
        return np.where(valid, result, np.nan)

    def sympy_rv(self):
        return Normal(fresh_name(), self._loc, self._scale)


class BinomialNoiseSource(NoiseSource):
    def __init__(self, n, p):
        self._n = sympify(n)
        self._p = sympify(p)

    @property
    def free_symbols(self):
        return self._n.free_symbols | self._p.free_symbols

    def instantiate(self, resolved):
        return BinomialNoiseSource(self._n.subs(resolved), self._p.subs(resolved))

    def sample(self, rng, size=None):
        try:
            n_val = int(self._n)
            p_val = float(self._p)
        except (TypeError, ValueError):
            return np.nan if size is None else np.full(size, np.nan, dtype=float)
        if n_val < 0 or not np.isfinite(p_val) or p_val < 0.0 or p_val > 1.0:
            return np.nan if size is None else np.full(size, np.nan, dtype=float)
        return rng.binomial(n_val, p_val, size=size)

    def param_exprs(self):
        return (self._n, self._p)

    def sample_arrays(self, rng, n, p):
        n = np.asarray(n, dtype=float)
        p = np.asarray(p, dtype=float)
        valid = (n >= 0) & np.isfinite(p) & (p >= 0) & (p <= 1)
        result = np.asarray(rng.binomial(
            np.where(valid, n, 0).astype(int),
            np.where(valid, p, 0.5),
        ), dtype=float)
        return np.where(valid, result, np.nan)

    def sympy_rv(self):
        return rv(fresh_name(), BinomialDistribution, self._n, self._p, check=False)


def gaussian(loc, scale):
    return NormalNoiseSource(_to_expr(loc), _to_expr(scale))


def binomial(n, p):
    return BinomialNoiseSource(_to_expr(n), _to_expr(p))
