import matplotlib
import numpy as np
import pytest
import sympy as sp

from sympy.stats import Normal

from src.core import NoisyFloat
from src.visual import plot_posterior


matplotlib.use("Agg")


def test_plot_posteriors_for_composed_expression_returns_density_curve():
    theta_0 = sp.Symbol("theta_0")
    theta_1 = sp.Symbol("theta_1")
    eps_0 = Normal("eps_0", 0, 1)
    eps_1 = Normal("eps_1", 0, 2)
    eps_pred = Normal("eps_pred", 0, 0.5)

    observed_0 = 3.0
    observed_1 = -1.0

    # Posterior of a composed expression in two latent dimensions.
    noisy = NoisyFloat(
        expr=theta_0 * theta_1 + eps_pred,
        obs=0.0,
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

    noisy_a = NoisyFloat(
        expr=theta + eps_pred,
        obs=0.0,
        thetas={theta},
        eqns=[theta + eps_obs - 1.0],
    )
    noisy_b = NoisyFloat(
        expr=2 * theta + eps_pred,
        obs=0.0,
        thetas={theta},
        eqns=[theta + eps_obs - 1.0],
    )

    result = plot_posterior(noisy_a, noisy_b, quadrature_points=7, grid_size=180)

    assert len(result["curves"]) == 2
