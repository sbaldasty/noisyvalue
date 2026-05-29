import itertools
import numpy as np
import sympy as sp

from sympy.stats import quantile
from sympy.stats.rv import random_symbols

from src.core import _filter_theta_equations
from src.core import _solve_theta_substitutions
from src.core import _preferred_value_expr


def _weighted_quantile(values, weights, q):
    if len(values) == 0:
        raise ValueError("Cannot compute quantiles of an empty sample")

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")

    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("weights must sum to a positive finite value")

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / total

    q = float(np.clip(q, 0.0, 1.0))
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def _quantile_nodes_for_rv(rv, quadrature_points, eps=1e-8):
    qfun = quantile(rv)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)

    # Map Gauss-Legendre nodes from [-1, 1] to [0, 1].
    u = 0.5 * (nodes + 1.0)
    u = np.clip(u, eps, 1.0 - eps)
    w = 0.5 * weights

    rv_values = []
    for ui in u:
        try:
            rv_values.append(float(sp.N(qfun(float(ui)))))
        except Exception as exc:
            raise ValueError(
                f"Could not evaluate quantile for random symbol {rv}. "
                "This method requires continuous distributions with numeric quantiles."
            ) from exc

    return np.asarray(rv_values, dtype=float), np.asarray(w, dtype=float)


def _compute_posterior_quadrature_points(noisy_value, quadrature_points=17, max_grid_points=300000):
    if quadrature_points < 2:
        raise ValueError("quadrature_points must be at least 2")

    thetas = noisy_value.root.latent_symbols()
    constraints = noisy_value.root.all_constraints()
    theta_constraints = _filter_theta_equations(constraints, thetas)

    if thetas:
        sol = _solve_theta_substitutions(thetas, theta_constraints)
        rhs_noise_vars = list({rv for rhs in sol.values() for rv in random_symbols(rhs)})
    else:
        sol = {}
        rhs_noise_vars = []

    value_expr = _preferred_value_expr(noisy_value)
    predictive_noise_vars = list(random_symbols(value_expr))
    integration_rvs = sorted(set(rhs_noise_vars) | set(predictive_noise_vars), key=str)

    if not integration_rvs:
        value = float(value_expr)
        return np.asarray([value], dtype=float), np.asarray([1.0], dtype=float)

    total_points = quadrature_points ** len(integration_rvs)
    if total_points > max_grid_points:
        raise ValueError(
            "Deterministic quadrature grid is too large for this expression. "
            f"Requested {total_points} points; lower quadrature_points or simplify expression."
        )

    rv_to_nodes = {}
    rv_to_weights = {}
    for rv in integration_rvs:
        nodes, weights = _quantile_nodes_for_rv(rv, quadrature_points)
        rv_to_nodes[rv] = nodes
        rv_to_weights[rv] = weights

    z_values = np.empty(total_points, dtype=float)
    point_weights = np.empty(total_points, dtype=float)

    point_index = 0
    for combo in itertools.product(range(quadrature_points), repeat=len(integration_rvs)):
        draws = {}
        weight = 1.0
        for rv, idx in zip(integration_rvs, combo):
            draws[rv] = rv_to_nodes[rv][idx]
            weight *= rv_to_weights[rv][idx]

        theta_values = {theta: float(rhs.subs(draws)) for theta, rhs in sol.items()}
        value = float(value_expr.subs(theta_values).subs(draws))

        z_values[point_index] = value
        point_weights[point_index] = weight
        point_index += 1

    weight_sum = float(np.sum(point_weights))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        raise ValueError("Quadrature weights are invalid")

    point_weights /= weight_sum
    return z_values, point_weights


def _estimate_weighted_density(values, weights, grid_size=500, bandwidth=None, tail_quantile=0.002):
    if grid_size < 50:
        raise ValueError("grid_size must be at least 50")

    lo = _weighted_quantile(values, weights, tail_quantile)
    hi = _weighted_quantile(values, weights, 1.0 - tail_quantile)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Density bounds are not finite")

    if hi <= lo:
        hi = lo + 1.0

    spread = hi - lo
    lo -= 0.05 * spread
    hi += 0.05 * spread
    grid = np.linspace(lo, hi, grid_size)

    mu = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mu) ** 2))
    std = float(np.sqrt(max(var, 1e-16)))
    n_eff = 1.0 / float(np.sum(weights**2))

    if bandwidth is None:
        bandwidth = 1.06 * std * (n_eff ** (-1.0 / 5.0))
    bandwidth = float(max(bandwidth, 1e-6))

    delta = (grid[:, None] - values[None, :]) / bandwidth
    kernels = np.exp(-0.5 * delta * delta) / np.sqrt(2.0 * np.pi)
    density = np.sum(weights[None, :] * kernels, axis=1) / bandwidth
    density = np.asarray(density, dtype=float)

    # Normalize numerically to protect against truncation and finite grid effects.
    area = float(np.trapezoid(density, grid))
    if area > 0.0 and np.isfinite(area):
        density /= area

    return grid, density


def plot_posterior(
    *noisy_values,
    grid_size=500,
    quadrature_points=17,
    max_grid_points=300000,
    bandwidth=None,
    tail_quantile=0.002,
    ax=None,
):
    """
    Plot posterior densities of each NoisyFloat's composed expression.

    This is deterministic (no Monte Carlo): it approximates integration over all
    random symbols using tensor-product Gauss-Legendre quadrature in quantile
    space, then uses weighted KDE to render a smooth density curve.
    """
    if not noisy_values:
        raise ValueError("plot_posteriors requires at least one NoisyFloat")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plot_posteriors") from exc

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
        created_fig = True

    curves = []
    for idx, noisy_value in enumerate(noisy_values, start=1):
        value_expr = _preferred_value_expr(noisy_value)
        z_values, z_weights = _compute_posterior_quadrature_points(
            noisy_value,
            quadrature_points=quadrature_points,
            max_grid_points=max_grid_points,
        )
        x_grid, density = _estimate_weighted_density(
            z_values,
            z_weights,
            grid_size=grid_size,
            bandwidth=bandwidth,
            tail_quantile=tail_quantile,
        )

        label = f"expr_{idx}: {sp.sstr(value_expr)}"
        ax.plot(x_grid, density, linewidth=2.0, label=label)
        curves.append(
            {
                "noisy_value": noisy_value,
                "x": x_grid,
                "density": density,
                "quadrature_values": z_values,
                "quadrature_weights": z_weights,
            }
        )

    ax.set_xlabel("Expression value")
    ax.set_ylabel("Posterior density")
    ax.set_title("Posterior Densities of Composed Expressions")
    if len(curves) > 1:
        ax.legend(loc="best")

    if created_fig:
        plt.tight_layout()

    return {"ax": ax, "curves": curves}
