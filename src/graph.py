import numpy as np
import util

from sympy import Symbol
from sympy import sympify
from sympy.stats import Normal
from sympy.stats.frv_types import BinomialDistribution, rv

from .util import fresh_name


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


class NoiseNode(Node):
    def param_exprs(self):
        raise NotImplementedError

    def param_symbols(self):
        return {s for p in self.param_exprs() for s in p.free_symbols}

    def sample(self, rng, size=None, resolved=None):
        raise NotImplementedError

    def sympy_rv(self):
        raise NotImplementedError

    @classmethod
    def sample_arrays(cls, rng, *param_arrays):
        raise NotImplementedError


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


class NormalNode(NoiseNode):
    def __init__(self, loc, scale, deps=()):
        super().__init__(deps)
        self.loc = sympify(loc)
        self.scale = sympify(scale)

    def param_exprs(self):
        return (self.loc, self.scale)

    def sample(self, rng, size=None, resolved=None):
        loc = self.loc if resolved is None else self.loc.subs(resolved)
        scale = self.scale if resolved is None else self.scale.subs(resolved)
        return rng.normal(float(loc), float(scale), size=size)

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
    def __init__(self, n, p, deps=()):
        super().__init__(deps)
        self.trials = sympify(n)
        self.prob = sympify(p)

    def param_exprs(self):
        return (self.trials, self.prob)

    def sample(self, rng, size=None, resolved=None):
        n = self.trials if resolved is None else self.trials.subs(resolved)
        p = self.prob if resolved is None else self.prob.subs(resolved)
        try:
            n_val = int(n)
            p_val = float(p)
        except (TypeError, ValueError):
            return np.nan if size is None else np.full(size, np.nan, dtype=float)
        if n_val < 0 or not np.isfinite(p_val) or p_val < 0.0 or p_val > 1.0:
            return np.nan if size is None else np.full(size, np.nan, dtype=float)
        return rng.binomial(n_val, p_val, size=size)

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
