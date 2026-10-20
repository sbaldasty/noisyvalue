import pytest
import conftest as noise
import sympy as sp

from conftest import from_node
from noisyvalue.graph import DerivedNode
from noisyvalue.graph import LatentNode
from noisyvalue.graph import GaussianNode


def test_node_constructors_reject_unknown_kwargs():
    with pytest.raises(TypeError):
        LatentNode(symbol=sp.Symbol("theta_node"))


def test_node_derived_uses_explicit_dependencies():
    theta = LatentNode()
    eps = noise.gaussian(0, 1)

    value = DerivedNode(expr=theta.expr + eps.expr, deps=(theta, eps))

    assert {dep.expr for dep in value.deps} == {theta.expr, eps.expr}


def test_node_noise_uses_explicit_dependencies():
    theta = LatentNode()

    z = GaussianNode.create(deps=(theta,), loc=theta.expr, scale=1, low=-sp.oo, high=sp.oo)

    assert {dep.expr for dep in z.deps} == {theta.expr}


def test_node_uses_fresh_symbols_for_each_node():
    first = LatentNode()
    second = LatentNode()

    assert first.expr != second.expr


def test_node_derived_wires_wrapper_and_symbol_nodes_explicitly():
    theta_node = LatentNode()
    theta = theta_node.expr
    base_root = DerivedNode(expr=theta, constraints=(theta - 2.0,), deps=(theta_node,))
    wrapped = from_node(obs=3.0, root=base_root, expr=theta + 1.0)

    out = DerivedNode(expr=theta + 2.0, deps=(wrapped._root, theta_node))

    dep_symbols = {dep.expr for dep in out.deps}
    assert wrapped._root.expr in dep_symbols
    assert theta in dep_symbols

