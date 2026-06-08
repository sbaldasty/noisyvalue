from collections import defaultdict
from weakref import WeakSet

import pytest
import sympy as sp

import src.core as core_module
from sympy.stats.rv import random_symbols

from src.core import NoisyFloat
from src.core import Node


@pytest.fixture(autouse=True)
def clear_node_registry():
    core_module._SYMBOL_NODES.clear()
    core_module._SYMBOL_ASSOCIATED_NODES = defaultdict(WeakSet)


def _lookup_registered_node(symbol, *, role=None):
    symbol = sp.sympify(symbol)

    node = core_module._SYMBOL_NODES.get(symbol)
    if node is not None and (role is None or node.role == role):
        return node

    associated = core_module._SYMBOL_ASSOCIATED_NODES.get(symbol)
    if not associated:
        return None

    candidates = sorted(associated, key=lambda item: str(item.symbol))
    if role is None:
        return candidates[0]

    role_candidates = [candidate for candidate in candidates if candidate.role == role]
    if not role_candidates:
        return None

    if role == "latent":
        return role_candidates[0]

    if role == "noise":
        for candidate in role_candidates:
            if candidate.law == symbol and not candidate.depends_on:
                return candidate
        for candidate in role_candidates:
            if candidate.law == symbol:
                return candidate

    return role_candidates[0]


def rooted_float(obs, expr, thetas=(), eqns=()):
    eqns = tuple(sp.sympify(eqn) for eqn in eqns)

    theta_nodes = []
    theta_substitutions = {}
    for theta in sorted(set(thetas), key=str):
        theta = sp.sympify(theta)
        node = _lookup_registered_node(theta, role="latent") or Node.latent()
        core_module._SYMBOL_ASSOCIATED_NODES[theta].add(node)
        theta_nodes.append(node)
        theta_substitutions[theta] = node.symbol

    random_rvs = set(random_symbols(expr)) | {
        rv for eqn in eqns for rv in random_symbols(eqn)
    }
    noise_substitutions = {}
    noise_nodes = []
    for rv in sorted(random_rvs, key=str):
        node = _lookup_registered_node(rv, role="noise") or Node.noise(law=rv)
        core_module._SYMBOL_ASSOCIATED_NODES[rv].add(node)
        noise_nodes.append(node)
        noise_substitutions[rv] = node.symbol

    substitutions = {**theta_substitutions, **noise_substitutions}
    expr = sp.sympify(expr).subs(substitutions)
    eqns = tuple(eqn.subs(substitutions) for eqn in eqns)

    root = Node.derived(
        constraints=eqns,
        definition=expr,
    )
    return NoisyFloat.from_node(obs=obs, root=root, expr=expr)
