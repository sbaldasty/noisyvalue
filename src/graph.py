import sympy as sp

from sympy import Symbol
from sympy import sympify

from .util import fresh_name


class Node:
    def __init__(self, depends_on=()):
        self.symbol = Symbol(fresh_name())
        self.depends_on = tuple(depends_on)
        if not all(isinstance(node, Node) for node in self.depends_on):
            raise TypeError("depends_on must contain Node instances")

    def closure(self):
        seen = set()
        ordered = []

        def walk(node):
            if node.symbol in seen:
                return
            seen.add(node.symbol)
            ordered.append(node)
            for dep in node.depends_on:
                walk(dep)

        walk(self)
        return tuple(ordered)

    def latent_symbols(self):
        return {node.symbol for node in self.closure() if isinstance(node, LatentNode)}

    def all_constraints(self):
        return tuple(
            c
            for node in self.closure()
            if isinstance(node, DerivedNode)
            for c in node.constraints
        )


class LatentNode(Node):
    pass


class NoiseNode(Node):
    pass


class DerivedNode(Node):
    def __init__(self, definition, constraints=(), depends_on=()):
        super().__init__(depends_on=depends_on)
        self.definition = sympify(definition)
        self.constraints = tuple(sympify(x) for x in constraints)


def topological_sort_law_nodes(law_nodes):
    law_symbols = {node.symbol for node in law_nodes}
    by_symbol = {node.symbol: node for node in law_nodes}
    predecessors = {
        node.symbol: {
            dep.symbol
            for dep in node.depends_on
            if not isinstance(dep, DerivedNode) and dep.symbol in law_symbols
        }
        for node in law_nodes
    }
    ordered = []
    resolved = set()
    remaining = set(law_symbols)
    while remaining:
        ready = {sym for sym in remaining if predecessors[sym] <= resolved}
        if not ready:
            raise ValueError(f"Cycle in law node dependencies: {remaining}")
        for sym in sorted(ready, key=str):
            ordered.append(by_symbol[sym])
            resolved.add(sym)
            remaining.discard(sym)
    return tuple(ordered)
