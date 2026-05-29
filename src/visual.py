import itertools
import numpy as np
import sympy as sp

from sympy.stats import quantile
from sympy.stats.rv import random_symbols

from src.core import _solve_theta_substitutions


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

    if thetas:
        sol = _solve_theta_substitutions(thetas, constraints)
        rhs_noise_vars = list({rv for rhs in sol.values() for rv in random_symbols(rhs)})
    else:
        sol = {}
        rhs_noise_vars = []

    predictive_noise_vars = list(random_symbols(noisy_value._expr))
    integration_rvs = sorted(set(rhs_noise_vars) | set(predictive_noise_vars), key=str)

    if not integration_rvs:
        value = float(noisy_value._expr)
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
        value = float(noisy_value._expr.subs(theta_values).subs(draws))

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

        label = f"expr_{idx}: {sp.sstr(noisy_value._expr)}"
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

# def plot_confidence_heatmap(
#     noisy_value,
#     theta_range=None,
#     observed_range=None,
#     grid_size=201,
#     n_samples=6000,
#     cmap="viridis_r",
#     ax=None,
#     seed=None,
#     library="scipy",
#     bandwidth=None,
#     contour_levels=(0.5, 0.8, 0.9, 0.95),
#     contour_color="white",
#     contour_linewidth=1.2,
#     contour_labels=True,
#     **sample_kwargs,
# ):
#     """
#     Plot a confidence-threshold heat map for the composed quantity in `noisy_value`.

#     Axes
#     -----
#     x-axis: true value of `noisy_value.expr`
#     y-axis: observed value of the same composed quantity

#     Heat value
#     ----------
#     For each (x, y), color is the smallest central-interval probability level
#     that contains y in the conditional observed distribution for that x-column.

#     This function supports multiple latent thetas and multiple equations by
#     Monte Carlo: draw latent theta vectors from all equations, then simulate a
#     fresh observed realization of the composed expression.
#     """
#     if grid_size < 2:
#         raise ValueError("grid_size must be at least 2")
#     if n_samples <= 0:
#         raise ValueError("n_samples must be > 0")
#     if not noisy_value.thetas:
#         raise ValueError("Heatmap requires at least one latent theta in NoisyValue")

#     sol = noisy_value._solve_theta_substitutions()
#     rhs_noise_vars = list({rv for rhs in sol.values() for rv in random_symbols(rhs)})

#     # Infer one measurement-noise RV per theta from equations that reference it.
#     theta_noise_rv = {}
#     for theta in noisy_value.thetas:
#         eqs_for_theta = [eq for eq in noisy_value.equations if theta in eq.free_symbols]
#         if not eqs_for_theta:
#             raise ValueError(f"No observation equation references latent {theta}")

#         chosen_rv = None
#         for eq in eqs_for_theta:
#             rvs = list(random_symbols(eq))
#             if rvs:
#                 chosen_rv = rvs[0]
#                 break
#         if chosen_rv is None:
#             raise ValueError(f"Could not infer measurement noise RV for latent {theta}")
#         theta_noise_rv[theta] = chosen_rv

#     rng = seed
#     if isinstance(seed, int):
#         rng = np.random.default_rng(seed)

#     x_true_samples = np.empty(n_samples, dtype=float)
#     y_obs_samples = np.empty(n_samples, dtype=float)

#     for i in range(n_samples):
#         rhs_noise_draws = {
#             rv: float(sample(rv, library=library, seed=rng, **sample_kwargs))
#             for rv in rhs_noise_vars
#         }
#         theta_values = {
#             theta: float(rhs.subs(rhs_noise_draws))
#             for theta, rhs in sol.items()
#         }

#         x_true_samples[i] = _evaluate_random_expr(
#             noisy_value.expr.subs(theta_values),
#             rng,
#             library=library,
#             **sample_kwargs,
#         )

#         observed_theta_values = {
#             theta: theta_values[theta] + float(sample(theta_noise_rv[theta], library=library, seed=rng, **sample_kwargs))
#             for theta in noisy_value.thetas
#         }
#         y_obs_samples[i] = _evaluate_random_expr(
#             noisy_value.expr.subs(observed_theta_values),
#             rng,
#             library=library,
#             **sample_kwargs,
#         )

#     if theta_range is None:
#         lo, hi = np.quantile(x_true_samples, [0.005, 0.995])
#         pad = 0.05 * (hi - lo if hi > lo else 1.0)
#         theta_range = (float(lo - pad), float(hi + pad))

#     if observed_range is None:
#         lo, hi = np.quantile(y_obs_samples, [0.005, 0.995])
#         pad = 0.05 * (hi - lo if hi > lo else 1.0)
#         observed_range = (float(lo - pad), float(hi + pad))

#     theta_grid = np.linspace(theta_range[0], theta_range[1], grid_size)
#     observed_grid = np.linspace(observed_range[0], observed_range[1], grid_size)

#     if bandwidth is None:
#         x_std = float(np.std(x_true_samples, ddof=1))
#         if not np.isfinite(x_std) or x_std == 0.0:
#             x_std = max(1e-6, (theta_range[1] - theta_range[0]) / 20.0)
#         bandwidth = 0.2 * x_std
#     bandwidth = float(max(bandwidth, 1e-12))

#     heat = np.empty((grid_size, grid_size), dtype=float)
#     for j, x0 in enumerate(theta_grid):
#         z = (x_true_samples - x0) / bandwidth
#         w = np.exp(-0.5 * z * z)
#         w_sum = float(np.sum(w))
#         if w_sum <= 0.0 or not np.isfinite(w_sum):
#             w = np.full_like(w, 1.0 / len(w))
#         else:
#             w /= w_sum

#         center = float(np.sum(w * y_obs_samples))
#         distances = np.abs(y_obs_samples - center)
#         query_radius = np.abs(observed_grid - center)

#         heat[:, j] = np.array([float(np.sum(w[distances <= r])) for r in query_radius], dtype=float)

#     created_fig = False
#     if ax is None:
#         _, ax = plt.subplots(figsize=(8, 6))
#         created_fig = True

#     im = ax.imshow(
#         heat,
#         origin="lower",
#         extent=[theta_grid[0], theta_grid[-1], observed_grid[0], observed_grid[-1]],
#         aspect="auto",
#         cmap=cmap,
#         vmin=0.0,
#         vmax=1.0,
#     )

#     if contour_levels:
#         levels = sorted({float(v) for v in contour_levels if 0.0 < float(v) < 1.0})
#         if levels:
#             xx, yy = np.meshgrid(theta_grid, observed_grid)
#             contours = ax.contour(
#                 xx,
#                 yy,
#                 heat,
#                 levels=levels,
#                 colors=contour_color,
#                 linewidths=contour_linewidth,
#                 alpha=0.95,
#             )
#             if contour_labels:
#                 fmt = {lv: f"{int(round(lv * 100))}%" for lv in contours.levels}
#                 ax.clabel(contours, contours.levels, inline=True, fontsize=8, fmt=fmt)

#     ax.set_xlabel("True value")
#     ax.set_ylabel("Observed value")
#     ax.set_title("Confidence-Interval Threshold Heatmap")
#     ax.plot(theta_grid, theta_grid, color="white", linestyle="--", linewidth=1.0, alpha=0.5)

#     cbar = plt.colorbar(im, ax=ax)
#     cbar.set_label("Smallest central CI probability containing observed value")

#     if created_fig:
#         plt.tight_layout()

#     return {
#         "ax": ax,
#         "heat": heat,
#         "theta_grid": theta_grid,
#         "observed_grid": observed_grid,
#         "x_true_samples": x_true_samples,
#         "y_obs_samples": y_obs_samples,
#         "bandwidth": bandwidth,
#     }
