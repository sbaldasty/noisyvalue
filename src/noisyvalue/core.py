import operator as op
import sympy as sp
import numpy as np

from sympy import Abs
from sympy import And
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
    def __init__(self, obs, root):
        util.require_instance(Node, root)
        self._obs = obs
        self._root = root

    def __repr__(self):
        return f"~{self._obs}"

    def __float__(self):
        return float(self._obs)

    def __int__(self):
        return int(self._obs)

    def __bool__(self):
        return bool(self._obs)

    @property
    def expr(self):
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
        rng = util.generator(rng)
        obs = sympify(true_value) + noise_node.sample(rng)
        return cls.observe(obs, noise_node)

    @classmethod
    def lift(cls, value, accept=None):
        accept = cls if accept is None else accept
        if not isinstance(accept, type) or not issubclass(accept, NoisyValue):
            raise TypeError(f"Expected accept to be subclass of NoisyValue, got {accept!r}")
        return value if isinstance(value, accept) else cls(value, DerivedNode(value))


def _is_int_like(x):
    if isinstance(x, NoisyInt):
        return True
    if isinstance(x, NoisyValue):
        return False
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _numeric_result_cls(a, b):
    return NoisyInt if _is_int_like(a) and _is_int_like(b) else NoisyFloat


class NoisyNumber(NoisyValue):
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
        return self.bin_op(other, NoisyBool, op.eq)

    def __ne__(self, other):
        return self.bin_op(other, NoisyBool, op.ne)

    def guarded(self, guard, fallback=sp.nan):
        guard = NoisyBool.lift(guard)
        fallback = sympify(fallback)

        obs = self._obs if bool(guard) else fallback
        expr = Piecewise(
            (self.expr, guard.expr),
            (fallback, True))

        root = DerivedNode.operational(expr, deps=[self._root, guard._root])
        return type(self)(obs, root)


class NoisyFloat(NoisyNumber):
    def __init__(self, obs, root):
        super().__init__(float(obs), root)

    def exp(self):
        return self.unary_op(NoisyFloat, np.exp, sp.exp)

    def log(self):
        return self.unary_op(NoisyFloat, np.log, sp.log)

    def round_nearest(self):
        obs = np.floor(self._obs + 0.5)
        expr = sp.floor(self.expr + Rational(1, 2))
        root = DerivedNode.operational(expr, deps=[self._root])
        return NoisyInt(obs, root)

    def sqrt(self):
        return self.unary_op(NoisyFloat, np.sqrt, sp.sqrt)

    @classmethod
    def gaussian(cls, loc, scale, low=-sp.oo, high=sp.oo, obs=None, rng=None):
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
    def __init__(self, obs, root):
        super().__init__(int(obs), root)

    def __index__(self):
        return self._obs

    @classmethod
    def binomial(cls, n, p, obs=None, rng=None):
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
