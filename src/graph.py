import numpy as np
import util

from sympy import Symbol
from sympy import sympify
from sympy.stats import Normal
from sympy.stats.frv_types import BinomialDistribution, rv

from .util import fresh_name


class Parameter:
    def __init__(self, index=None):
        self.index = index
        self.name = None
        self.owner = None

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return obj.params[self.index]


class Node:
    def __init__(self, deps=()):
        self.expr = Symbol(fresh_name())
        self.deps = util.as_tuple(deps, Node)

    def closure(self):
        seen = set()
        ordered = []

        def walk(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            ordered.append(node)
            for dep in node.deps:
                walk(dep)

        walk(self)
        return tuple(ordered)

    def latent_symbols(self):
        return {node.expr for node in self.closure() if isinstance(node, LatentNode)}

    def all_constraints(self):
        return frozenset(
            c
            for node in self.closure()
            if isinstance(node, DerivedNode)
            for c in node.constraints
        )


class LatentNode(Node):
    pass


class DerivedNode(Node):
    def __init__(self, expr, constraints=(), deps=()):
        super().__init__(deps)
        self.expr = sympify(expr)
        self.constraints = frozenset(sympify(x) for x in constraints)

    @classmethod
    def operational(cls, expr, deps=()):
        flat_deps = []
        flat_eqns = []
        for node in deps:
            if isinstance(node, DerivedNode):
                flat_eqns.extend(node.constraints)
                flat_deps.extend(node.deps)
            else:
                flat_deps.append(node)
        return cls(expr, frozenset(flat_eqns), frozenset(flat_deps))


class NoiseNode(Node):
    def __init__(self, params, deps=()):
        super().__init__(deps)
        self.params = list(params)

    def __init_subclass__(cls):
        super().__init_subclass__()
        params = [x for x in cls.__dict__.values() if isinstance(x, Parameter)]
        params.sort(key=lambda p: p.index)
        cls._parameters = tuple(params)

    def param_symbols(self):
        return {s for p in self.params for s in p.free_symbols}

    def sample(self, rng, size=None, resolved=None):
        raise NotImplementedError

    def sympy_rv(self):
        raise NotImplementedError

    @classmethod
    def create(cls, deps=(), **kwargs):
        params = []
        for p in cls._parameters:
            if p.name in kwargs:
                params.append(sympify(kwargs[p.name]))
            else:
                raise TypeError(f"Missing parameter: {p.name}")
        return cls(params, deps)

    @classmethod
    def sample_arrays(cls, rng, *param_arrays):
        raise NotImplementedError


class NormalNode(NoiseNode):
    loc = Parameter(0)
    scale = Parameter(1)

    def sample(self, rng, size=None, resolved=()):
        loc = float(self.loc.subs(resolved))
        scale = float(self.scale.subs(resolved))
        return rng.normal(loc, scale, size=size)

    def sympy_rv(self):
        return Normal(fresh_name(), self.loc, self.scale)

    @classmethod
    def sample_arrays(cls, rng, loc, scale):
        loc = np.asarray(loc, dtype=float)
        scale = np.asarray(scale, dtype=float)
        valid = np.isfinite(loc) & (scale > 0)
        if np.all(valid):
            return rng.normal(loc, scale)
        result = rng.normal(np.where(valid, loc, 0.0), np.where(valid, scale, 1.0))
        return np.where(valid, result, np.nan)


class BinomialNode(NoiseNode):
    trials = Parameter(0)
    prob = Parameter(1)

    def sample(self, rng, size=None, resolved=()):
        try:
            trials = int(self.trials.subs(resolved))
            prob = float(self.prob.subs(resolved))
        except (TypeError, ValueError):
            return np.nan if size is None else np.full(size, np.nan, dtype=float)
        if trials < 0 or not np.isfinite(prob) or prob < 0.0 or prob > 1.0:
            return np.nan if size is None else np.full(size, np.nan, dtype=float)
        return rng.binomial(trials, prob, size=size)

    def sympy_rv(self):
        return rv(fresh_name(), BinomialDistribution, self.trials, self.prob, check=False)

    @classmethod
    def sample_arrays(cls, rng, n, p):
        n = np.asarray(n, dtype=float)
        p = np.asarray(p, dtype=float)
        valid = (n >= 0) & np.isfinite(p) & (p >= 0) & (p <= 1)
        result = np.asarray(rng.binomial(
            np.where(valid, n, 0).astype(int),
            np.where(valid, p, 0.5),
        ), dtype=float)
        return np.where(valid, result, np.nan)


def topological_sort_law_nodes(law_nodes):
    law_symbols = {node.expr for node in law_nodes}
    by_symbol = {node.expr: node for node in law_nodes}
    predecessors = {
        node.expr: {
            dep.expr
            for dep in node.deps
            if not isinstance(dep, DerivedNode) and dep.expr in law_symbols
        }
        for node in law_nodes
    }
    ordered = []
    resolved = set()
    remaining = set(law_symbols)
    while remaining:
        ready = {sym for sym in remaining if predecessors[sym] <= resolved}
        for sym in sorted(ready, key=str):
            ordered.append(by_symbol[sym])
            resolved.add(sym)
            remaining.discard(sym)
    return tuple(ordered)
