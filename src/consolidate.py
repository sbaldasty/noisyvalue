import sympy as sp
from sympy import sympify

from .core import (
    _as_node,
    _preferred_value_expr,
    _sampler_inputs_from_roots,
    _solve_theta_substitutions,
    DerivedNode,
    NoiseNode,
    NoisyValue,
)
from .noise import NormalNoiseSource


def _extract_coeff_symbol(expr):
    """Return (coefficient, symbol) if expr is c*sym or sym, else (None, None)."""
    if isinstance(expr, sp.Symbol):
        return sp.Integer(1), expr
    if isinstance(expr, sp.Mul):
        nums = [a for a in expr.args if a.is_number]
        syms = [a for a in expr.args if isinstance(a, sp.Symbol)]
        rest = [a for a in expr.args if not a.is_number and not isinstance(a, sp.Symbol)]
        if len(syms) == 1 and not rest:
            coeff = sp.Mul(*nums) if nums else sp.Integer(1)
            return coeff, syms[0]
    return None, None


class ConsolidationRule:
    """Base class for noise-combination rules used by consolidate()."""

    def matches(self, expr, symbol_to_node, eligible):
        raise NotImplementedError

    def apply(self, expr, symbol_to_node, eligible):
        raise NotImplementedError


class NormalSumRule(ConsolidationRule):
    """Collapse a linear combination of independent normal noise symbols into one."""

    def _parse(self, expr, symbol_to_node, eligible):
        if not isinstance(expr, sp.Add):
            return None, None
        normal_terms = []
        other_args = []
        for arg in expr.args:
            coeff, sym = _extract_coeff_symbol(arg)
            if (
                sym is not None
                and sym in eligible
                and sym in symbol_to_node
                and isinstance(symbol_to_node[sym], NoiseNode)
                and isinstance(symbol_to_node[sym].source, NormalNoiseSource)
                and not symbol_to_node[sym].depends_on
            ):
                normal_terms.append((coeff, symbol_to_node[sym]))
            else:
                other_args.append(arg)
        if len(normal_terms) < 2:
            return None, None
        return normal_terms, other_args

    def matches(self, expr, symbol_to_node, eligible):
        terms, _ = self._parse(expr, symbol_to_node, eligible)
        return terms is not None

    def apply(self, expr, symbol_to_node, eligible):
        normal_terms, other_args = self._parse(expr, symbol_to_node, eligible)
        combined_mu = sum(c * node.source._loc for c, node in normal_terms)
        combined_sigma = sp.sqrt(sum((c * node.source._scale) ** 2 for c, node in normal_terms))
        new_node = NoiseNode(NormalNoiseSource(combined_mu, combined_sigma))
        symbol_to_node[new_node.symbol] = new_node
        for _, node in normal_terms:
            eligible.discard(node.symbol)
        return sp.Add(new_node.symbol, *other_args)


DEFAULT_RULES = [NormalSumRule()]


def consolidate(*values, rules=None):
    """Return copies of values backed by a simplified noise graph.

    Noise symbols that can be combined by a rule are merged into fewer nodes.
    The returned values have the same observed values and equivalent marginal
    distributions, but share no nodes with the originals.

    Noise symbols that appear in more than one value's resolved expression are
    left untouched, preserving correlations between the consolidated values.
    """
    assert values and all(isinstance(v, NoisyValue) for v in values)
    if rules is None:
        rules = DEFAULT_RULES

    # Resolve latent variables so each expression is purely over noise symbols.
    all_thetas, theta_eqns, _, _, _ = _sampler_inputs_from_roots(values)
    theta_subs = _solve_theta_substitutions(all_thetas, theta_eqns)
    resolved_exprs = tuple(
        sympify(_preferred_value_expr(v)).subs(theta_subs) for v in values
    )

    # Build a mutable symbol → node lookup; rules may add new entries.
    symbol_to_node = {}
    for v in values:
        for node in _as_node(v).closure():
            symbol_to_node[node.symbol] = node

    # A noise symbol is eligible for combination only if it appears exactly once
    # across the joint expression, ensuring no cross-value correlation is broken.
    joint = sp.Tuple(*resolved_exprs)
    eligible = {
        sym
        for sym, node in symbol_to_node.items()
        if isinstance(node, NoiseNode)
        and not node.depends_on
        and joint.count(sym) == 1
    }

    def _query(expr):
        return any(rule.matches(expr, symbol_to_node, eligible) for rule in rules)

    def _value(expr):
        for rule in rules:
            if rule.matches(expr, symbol_to_node, eligible):
                return rule.apply(expr, symbol_to_node, eligible)
        return expr

    new_joint = joint.replace(_query, _value)
    new_exprs = new_joint.args

    result = []
    for v, new_expr in zip(values, new_exprs):
        dep_nodes = [node for sym, node in symbol_to_node.items() if sym in new_expr.free_symbols]
        root = DerivedNode(definition=new_expr, depends_on=dep_nodes)
        result.append(type(v)(v._obs, root))
    return tuple(result)
