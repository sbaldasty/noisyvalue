import matplotlib
import numpy as np
import pytest
import sympy as sp
import src.core as core_module

from sympy.stats import Normal
from sympy.stats.rv import random_symbols

from src.core import NoisyFloat
from src.core import Node
from src.visual import plot_posterior
from src.util import fresh_name


matplotlib.use("Agg")


def _rooted_float(obs, expr, thetas=(), eqns=()):
    eqns = tuple(sp.sympify(eqn) for eqn in eqns)
    def _lookup_registered_node(symbol):
        node = core_module._SYMBOL_NODES.get(symbol)
        if node is not None:
            return node
        associated = core_module._SYMBOL_ASSOCIATED_NODES.get(symbol)
        if associated:
            return next(iter(sorted(associated, key=lambda item: str(item.symbol))))
        return None

    theta_nodes = []
    theta_substitutions = {}
    for theta in sorted(set(thetas), key=str):
        theta = sp.sympify(theta)
        node = _lookup_registered_node(theta) or Node.latent()
        core_module._SYMBOL_ASSOCIATED_NODES[theta].add(node)
        theta_nodes.append(node)
        theta_substitutions[theta] = node.symbol
    random_rvs = set(random_symbols(expr)) | {
        rv for eqn in eqns for rv in random_symbols(eqn)
    }
    noise_nodes = []
    noise_substitutions = {}
    for rv in sorted(random_rvs, key=str):
        node = _lookup_registered_node(rv) or Node.noise(law=rv)
        core_module._SYMBOL_ASSOCIATED_NODES[rv].add(node)
        noise_nodes.append(node)
        noise_substitutions[rv] = node.symbol

    substitutions = {**theta_substitutions, **noise_substitutions}
    expr = sp.sympify(expr).subs(substitutions)
    eqns = tuple(eqn.subs(substitutions) for eqn in eqns)

    root = Node.derived(
        depends_on=tuple(theta_nodes) + tuple(noise_nodes),
        constraints=eqns,
        definition=expr,
    )
    return NoisyFloat.from_node(obs=obs, root=root, expr=expr)


def test_plot_posteriors_for_composed_expression_returns_density_curve():
    theta_0 = sp.Symbol("theta_0")
    theta_1 = sp.Symbol("theta_1")
    eps_0 = Normal("eps_0", 0, 1)
    eps_1 = Normal("eps_1", 0, 2)
    eps_pred = Normal("eps_pred", 0, 0.5)

    observed_0 = 3.0
    observed_1 = -1.0

    # Posterior of a composed expression in two latent dimensions.
    noisy = _rooted_float(
        obs=0.0,
        expr=theta_0 * theta_1 + eps_pred,
        thetas={theta_0, theta_1},
        eqns=[theta_0 + eps_0 - observed_0, theta_1 + eps_1 - observed_1],
    )

    result = plot_posterior(noisy, quadrature_points=7, grid_size=220)

    assert "ax" in result
    assert "curves" in result
    assert len(result["curves"]) == 1

    curve = result["curves"][0]
    x = curve["x"]
    density = curve["density"]

    assert isinstance(x, np.ndarray)
    assert isinstance(density, np.ndarray)
    assert x.shape == density.shape
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(density))
    assert np.all(density >= 0.0)

    area = np.trapezoid(density, x)
    assert area == pytest.approx(1.0, rel=0.08)


def test_plot_posteriors_supports_multiple_values():
    theta = sp.Symbol("theta")
    eps_obs = Normal("eps_obs", 0, 1)
    eps_pred = Normal("eps_pred", 0, 1)

    noisy_a = _rooted_float(obs=0.0, expr=theta + eps_pred, thetas={theta}, eqns=[theta + eps_obs - 1.0])
    noisy_b = _rooted_float(obs=0.0, expr=2 * theta + eps_pred, thetas={theta}, eqns=[theta + eps_obs - 1.0])

    result = plot_posterior(noisy_a, noisy_b, quadrature_points=7, grid_size=180)

    assert len(result["curves"]) == 2
