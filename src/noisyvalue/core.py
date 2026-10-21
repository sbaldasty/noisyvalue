import operator as op
import sympy as sp
import numpy as np

from sympy import Abs
from sympy import And
from sympy import Max
from sympy import Min
from sympy import Not
from sympy import Or
from sympy import Piecewise
from sympy import Pow
from sympy import Rational
from sympy import sympify

from . import util
from .graph import GaussianNode
from .graph import LaplaceNode
from .graph import BinomialNode
from .graph import DiscreteGaussianNode
from .graph import DiscreteLaplaceNode
from .graph import DerivedNode
from .graph import LatentNode
from .graph import Node


class NoisyValue:
    """An observation paired with a symbolic posterior over its true value.

    Every noisy value carries its full probabilistic provenance as a DAG of
    `Node`s, so values built from one another stay correlated under joint
    sampling. Arithmetic on the subclasses composes the observed scalar and
    the symbolic expression in tandem. Use `sample_noisy_values` to draw from
    the posterior.

    Examples
    --------
    >>> from noisyvalue import NoisyFloat, sample_noisy_values
    >>> x = NoisyFloat.gaussian(10, 2, obs=11.5)
    >>> y = 2 * x + 1
    >>> y
    ~24.0
    >>> sample_noisy_values(y, n=1000, rng=0)[0].draws.shape
    (1000,)
    """
    def __init__(self, obs, root):
        util.require_instance(Node, root)
        """`obs` is the observed scalar; `root` is the `Node` carrying its provenance."""
        self._obs = obs
        self._root = root

    def __repr__(self):
        """Show the observation, prefixed with `~` to mark it as noisy."""
        return f"~{self._obs}"

    def __float__(self):
        """The observed value as a `float`; the posterior is ignored."""
        return float(self._obs)

    def __int__(self):
        """The observed value as an `int`; the posterior is ignored."""
        return int(self._obs)

    def __bool__(self):
        """Truth of the observed value; the posterior is ignored."""
        return bool(self._obs)

    @property
    def expr(self):
        """The SymPy expression for this value in terms of its nodes' symbols."""
        return self._root.expr

    def bin_op(self, x, out_cls, obs_op, expr_op=None, rev=False):
        x = type(self).lift(x, accept=NoisyValue)
        lhs = x if rev else self
        rhs = self if rev else x

        if expr_op is None:
            expr_op = obs_op

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            obs = obs_op(lhs._obs, rhs._obs)

        expr = expr_op(lhs.expr, rhs.expr)
        root = DerivedNode.operational(expr, deps=[lhs._root, rhs._root])
        return out_cls(obs, root)

    def unary_op(self, out_cls, obs_op, expr_op):
        root = DerivedNode.operational(expr_op(self.expr), deps=[self._root])
        return out_cls(obs_op(self._obs), root)

    @classmethod
    def observe(cls, obs, noise_node):
        """Interpret an existing observation as unknown true value plus noise.

        The returned value's posterior treats `obs` as a realization of
        `theta + noise`, where `theta` is a latent true value with a flat
        prior and `noise` is described by `noise_node`.

        Examples
        --------
        >>> from noisyvalue import NoisyFloat, sample_noisy_values
        >>> noise = NoisyFloat.gaussian(0, 2, obs=0.0)._root
        >>> theta = NoisyFloat.observe(5.0, noise)
        >>> lo, hi = sample_noisy_values(theta, n=20000, rng=0)[0].credible_interval(0.95)
        >>> round(float(lo), 1), round(float(hi), 1)
        (1.1, 8.9)
        """
        theta_node = LatentNode()
        theta = theta_node.expr
        root = DerivedNode(
            theta,
            constraints=(theta + noise_node.expr - sympify(obs),),
            deps=(theta_node, noise_node))
        return cls(obs, root)

    @classmethod
    def draw(cls, true_value, noise_node, rng=None):
        """Simulate an observation of `true_value` and interpret it via `observe`.

        The observation is `true_value` plus a draw from `noise_node`, sampled
        with `rng` (a seed or generator, as accepted by `util.generator`).

        Examples
        --------
        >>> from noisyvalue import NoisyFloat
        >>> noise = NoisyFloat.gaussian(0, 1, obs=0.0)._root
        >>> NoisyFloat.draw(3.0, noise, rng=1)
        ~3.345584192064786
        """
        rng = util.generator(rng)
        obs = sympify(true_value) + noise_node.sample(rng)
        return cls.observe(obs, noise_node)

    @classmethod
    def lift(cls, value, accept=None):
        """Wrap a plain `value` as a constant of this class.

        A value that is already an instance of `accept` (default `cls`) is
        returned unchanged. Raises `TypeError` if `accept` is not a
        `NoisyValue` subclass.

        Examples
        --------
        >>> from noisyvalue import NoisyFloat
        >>> NoisyFloat.lift(3)
        ~3.0
        """
        accept = cls if accept is None else accept
        if not isinstance(accept, type) or not issubclass(accept, NoisyValue):
            raise TypeError(f"Expected accept to be subclass of NoisyValue, got {accept!r}")
        return value if isinstance(value, accept) else cls(value, DerivedNode(value))


class NoisyNumber(NoisyValue):
    """Numeric noisy value: supports arithmetic, comparisons and `guarded`.

    Operations between two integers stay integer (`NoisyInt`), except true
    division and powers, which give `NoisyFloat`. Comparisons give `NoisyBool`.

    Examples
    --------
    >>> from noisyvalue import NoisyInt
    >>> count = NoisyInt.binomial(10, 0.5, obs=4)
    >>> type(count + 1).__name__, type(count + 1.5).__name__
    ('NoisyInt', 'NoisyFloat')
    >>> count / 2
    ~2.0
    >>> count > 3
    ~True
    """
    def __init__(self, obs, root):
        super().__init__(obs, root)

    def __abs__(self):
        return self.unary_op(type(self), abs, Abs)

    def __add__(self, other):
        return self.bin_op(other, _numeric_result_cls(self, other), op.add)

    def __radd__(self, other):
        return self.bin_op(other, _numeric_result_cls(self, other), op.add, rev=True)

    def __sub__(self, other):
        return self.bin_op(other, _numeric_result_cls(self, other), op.sub)

    def __rsub__(self, other):
        return self.bin_op(other, _numeric_result_cls(self, other), op.sub, rev=True)

    def __mul__(self, other):
        return self.bin_op(other, _numeric_result_cls(self, other), op.mul)

    def __rmul__(self, other):
        return self.bin_op(other, _numeric_result_cls(self, other), op.mul, rev=True)

    def __truediv__(self, other):
        return self.bin_op(other, NoisyFloat, np.divide, op.truediv)

    def __rtruediv__(self, other):
        return self.bin_op(other, NoisyFloat, np.divide, op.truediv, rev=True)

    def __pow__(self, other):
        return self.bin_op(other, NoisyFloat, np.power, Pow)

    def __rpow__(self, other):
        return self.bin_op(other, NoisyFloat, np.power, Pow, rev=True)

    def __lt__(self, other):
        return self.bin_op(other, NoisyBool, op.lt)

    def __le__(self, other):
        return self.bin_op(other, NoisyBool, op.le)

    def __gt__(self, other):
        return self.bin_op(other, NoisyBool, op.gt)

    def __ge__(self, other):
        return self.bin_op(other, NoisyBool, op.ge)

    def __eq__(self, other):
        """Compare as noisy values, giving a `NoisyBool` rather than a plain `bool`."""
        return self.bin_op(other, NoisyBool, op.eq)

    def __ne__(self, other):
        return self.bin_op(other, NoisyBool, op.ne)

    def guarded(self, guard, fallback=sp.nan):
        """Restrict this value to the event `guard`.

        Where `guard` (a `NoisyBool` or plain bool) holds the result equals this
        value; elsewhere it equals `fallback`. The observation is `fallback` if
        the observed guard is false.

        Examples
        --------
        >>> from noisyvalue import NoisyFloat
        >>> x = NoisyFloat.gaussian(0, 1, obs=2.0)
        >>> x.guarded(x > 0, fallback=0)
        ~2.0
        >>> x.guarded(x < 0, fallback=0)
        ~0.0
        """
        guard = NoisyBool.lift(guard)
        fallback = sympify(fallback)

        obs = self._obs if bool(guard) else fallback
        expr = Piecewise(
            (self.expr, guard.expr),
            (fallback, True))

        root = DerivedNode.operational(expr, deps=[self._root, guard._root])
        return type(self)(obs, root)


class NoisyFloat(NoisyNumber):
    """Real-valued noisy value, with `exp`, `log`, `sqrt` and rounding.

    Examples
    --------
    >>> from noisyvalue import NoisyFloat
    >>> NoisyFloat.gaussian(0, 1, obs=2.0).exp().log()
    ~2.0
    """
    def __init__(self, obs, root):
        super().__init__(float(obs), root)

    def exp(self):
        return self.unary_op(NoisyFloat, np.exp, sp.exp)

    def log(self):
        return self.unary_op(NoisyFloat, np.log, sp.log)

    def round_nearest(self):
        """Round to the nearest integer, with halves rounding up, as a `NoisyInt`."""
        obs = np.floor(self._obs + 0.5)
        expr = sp.floor(self.expr + Rational(1, 2))
        root = DerivedNode.operational(expr, deps=[self._root])
        return NoisyInt(obs, root)

    def sqrt(self):
        return self.unary_op(NoisyFloat, np.sqrt, sp.sqrt)

    @classmethod
    def gaussian(cls, loc, scale, low=-sp.oo, high=sp.oo, obs=None, rng=None):
        """A normally distributed noisy value, optionally truncated to `[low, high]`.

        `loc` and `scale` may be numbers or noisy values, giving hierarchical
        noise. If `obs` is omitted it is drawn using `rng`.

        Examples
        --------
        >>> from noisyvalue import NoisyFloat
        >>> NoisyFloat.gaussian(0, 1, obs=0.3)
        ~0.3
        >>> float(NoisyFloat.gaussian(0, 1, low=0, rng=0)) >= 0
        True
        """
        loc = NoisyFloat.lift(loc)
        scale = NoisyFloat.lift(scale)
        deps = [v._root for v in (loc, scale) if v.expr.free_symbols]
        node = GaussianNode.create(
            deps=deps, loc=loc.expr, scale=scale.expr, low=low, high=high)
        if obs is None:
            rng = util.generator(rng)
            obs = GaussianNode._draw(
                rng, float(loc), float(scale), float(low), float(high))
        return cls(obs, node)

    @classmethod
    def laplace(cls, loc, scale, low=-sp.oo, high=sp.oo, obs=None, rng=None):
        """A Laplace distributed noisy value, optionally truncated to `[low, high]`.

        `loc` and `scale` may be numbers or noisy values, giving hierarchical
        noise. If `obs` is omitted it is drawn using `rng`.

        Examples
        --------
        >>> from noisyvalue import NoisyFloat
        >>> NoisyFloat.laplace(0, 1, low=0, obs=0.5)
        ~0.5
        """
        loc = NoisyFloat.lift(loc)
        scale = NoisyFloat.lift(scale)
        deps = [v._root for v in (loc, scale) if v.expr.free_symbols]
        node = LaplaceNode.create(
            deps=deps, loc=loc.expr, scale=scale.expr, low=low, high=high)
        if obs is None:
            rng = util.generator(rng)
            obs = LaplaceNode._draw(
                rng, float(loc), float(scale), float(low), float(high))
        return cls(obs, node)


class NoisyInt(NoisyNumber):
    """Integer-valued noisy value, with constructors for discrete noise."""
    def __init__(self, obs, root):
        super().__init__(int(obs), root)

    def __index__(self):
        return self._obs

    @classmethod
    def binomial(cls, n, p, obs=None, rng=None):
        """A binomial noisy value: successes in `n` trials of probability `p`.

        `n` and `p` may be numbers or noisy values. If `obs` is omitted it is
        drawn using `rng`.

        Examples
        --------
        >>> from noisyvalue import NoisyInt
        >>> NoisyInt.binomial(10, 0.5, obs=4)
        ~4
        >>> 0 <= int(NoisyInt.binomial(10, 0.5, rng=0)) <= 10
        True
        """
        n = NoisyInt.lift(n)
        p = NoisyFloat.lift(p)
        deps = [v._root for v in (n, p) if v.expr.free_symbols]
        node = BinomialNode.create(deps=deps, trials=n.expr, prob=p.expr)
        if obs is None:
            rng = util.generator(rng)
            obs = rng.binomial(int(n), float(p))
        return cls(obs, node)

    @classmethod
    def discrete_gaussian(cls, scale, loc=sp.Integer(0), low=-sp.oo, high=sp.oo, obs=None, rng=None):
        """A discrete Gaussian noisy value on the integers, optionally truncated to `[low, high]`.

        `scale` may be a number or a noisy value. If `obs` is omitted it is
        drawn using `rng`.

        Examples
        --------
        >>> from noisyvalue import NoisyInt
        >>> NoisyInt.discrete_gaussian(3, obs=2)
        ~2
        """
        scale = NoisyFloat.lift(scale)
        deps = [v._root for v in (scale,) if v.expr.free_symbols]
        node = DiscreteGaussianNode.create(
            deps=deps, loc=loc, scale=scale.expr, low=low, high=high)
        if obs is None:
            rng = util.generator(rng)
            obs = int(DiscreteGaussianNode._draw(
                rng, float(loc), float(scale), low=float(low), high=float(high)))
        return cls(obs, node)

    @classmethod
    def discrete_laplace(cls, scale, loc=sp.Integer(0), low=-sp.oo, high=sp.oo, obs=None, rng=None):
        """A discrete Laplace noisy value on the integers, optionally truncated to `[low, high]`.

        `scale` may be a number or a noisy value. If `obs` is omitted it is
        drawn using `rng`.

        Examples
        --------
        >>> from noisyvalue import NoisyInt
        >>> NoisyInt.discrete_laplace(2, obs=-1)
        ~-1
        """
        scale = NoisyFloat.lift(scale)
        deps = [v._root for v in (scale,) if v.expr.free_symbols]
        node = DiscreteLaplaceNode.create(
            deps=deps, loc=loc, scale=scale.expr, low=low, high=high)
        if obs is None:
            rng = util.generator(rng)
            obs = int(DiscreteLaplaceNode._draw(
                rng, float(loc), float(scale), low=float(low), high=float(high)))
        return cls(obs, node)


class NoisyBool(NoisyValue):
    """Boolean noisy value, combined with `&`, `|` and `~`.

    Examples
    --------
    >>> from noisyvalue import NoisyFloat
    >>> x = NoisyFloat.gaussian(0, 1, obs=2.0)
    >>> (x > 0) & ~(x > 5)
    ~True
    """
    def __init__(self, obs, root):
        super().__init__(bool(obs), root)

    def __and__(self, other):
        return self.bin_op(other, NoisyBool, op.and_, And)

    def __rand__(self, other):
        return self.bin_op(other, NoisyBool, op.and_, And, rev=True)

    def __or__(self, other):
        return self.bin_op(other, NoisyBool, op.or_, Or)

    def __ror__(self, other):
        return self.bin_op(other, NoisyBool, op.or_, Or, rev=True)

    def __invert__(self):
        return self.unary_op(NoisyBool, op.not_, Not)


def is_int_like(x):
    """Whether `x` is a `NoisyInt` or a plain (non-bool) integer."""
    if isinstance(x, NoisyInt):
        return True
    if isinstance(x, NoisyValue):
        return False
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _numeric_result_cls(a, b):
    return NoisyInt if is_int_like(a) and is_int_like(b) else NoisyFloat


def _fold_numbers(values, op):
    # Accept either several values or one iterable
    if len(values) == 1:
        try:
            values = tuple(values[0])
        except TypeError:
            pass
    if not values:
        raise ValueError("Requires at least one value")

    # Lift everything first so a plain float is never coerced to the int type
    values = [(NoisyInt if is_int_like(v) else NoisyFloat).lift(v, accept=NoisyNumber)
              for v in values]
    result = values[0]
    for value in values[1:]:
        result = result.bin_op(value, _numeric_result_cls(result, value), op)
    return result


def noisy_min(*values):
    """Minimum of noisy values, as a new noisy value.

    Accepts several values or a single iterable, like the built-in `min`.
    The posterior of the result reflects the uncertainty about which
    input is smallest, not just the observed winner. The result is a
    `NoisyInt` if every input is an integer, otherwise a `NoisyFloat`.
    Raises `ValueError` if there are no inputs.

    Examples
    --------
    >>> from noisyvalue import NoisyInt, noisy_min
    >>> a = NoisyInt.discrete_gaussian(3, obs=2)
    >>> noisy_min(a, 5)
    ~2
    >>> noisy_min([a, 5.5])
    ~2.0
    """
    return _fold_numbers(values, Min)


def noisy_max(*values):
    """Maximum of noisy values, as a new noisy value.

    Accepts several values or a single iterable, like the built-in `max`.
    The posterior of the result reflects the uncertainty about which
    input is largest, not just the observed winner. The result is a
    `NoisyInt` if every input is an integer, otherwise a `NoisyFloat`.
    Raises `ValueError` if there are no inputs.

    Examples
    --------
    >>> from noisyvalue import NoisyInt, noisy_max
    >>> a = NoisyInt.discrete_gaussian(3, obs=2)
    >>> noisy_max(a, 5)
    ~5
    >>> noisy_max([a, 5.5])
    ~5.5
    """
    return _fold_numbers(values, Max)
