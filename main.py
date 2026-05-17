import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from sympy import Symbol
from sympy.stats import sample
from sympy.stats.rv import random_symbols

_theta_counter = 0
_noise_counter = 0


def fresh_theta(tag=None):
    global _theta_counter
    name = f"theta_{_theta_counter}" if tag is None else f"theta_{tag}_{_theta_counter}"
    _theta_counter += 1
    return Symbol(name)


def fresh_noise_name(prefix="R"):
    global _noise_counter
    name = f"{prefix}{_noise_counter}"
    _noise_counter += 1
    return name


class NoisyValue:
    def __init__(self, expr, observed, thetas=None, equations=None):
        self.expr = sp.sympify(expr)
        self.observed = float(observed)
        self.thetas = set() if thetas is None else set(thetas)

        if equations is None:
            self.equations = [self.expr - self.observed]
        else:
            self.equations = equations

    def __repr__(self):
        return f"NoisyValue(expr={self.expr}, observed={self.observed})"

    @classmethod
    def from_noise_rv(cls, true_value, noise_rv, provenance=None, **sample_kwargs):
        """
        Build a NoisyValue from any SymPy random variable.

        The returned `expr` is the latent value (`theta`), while the measurement
        mechanism is encoded in `equations` as `theta + noise - observed = 0`.
        This makes downstream sampling reflect analyst belief about the true
        quantity rather than release-to-release spread.
        """
        noise_symbols = random_symbols(noise_rv)
        if len(noise_symbols) != 1 or noise_rv not in noise_symbols:
            raise TypeError("noise_rv must be a single SymPy random variable")

        theta = fresh_theta(provenance)
        measurement_expr = theta + noise_rv
        observed_expr = measurement_expr.subs({theta: sp.sympify(true_value)})
        observed = float(sample(observed_expr, **sample_kwargs))

        equations = [measurement_expr - observed]
        return cls(theta, observed, thetas={theta}, equations=equations)

    @classmethod
    def from_distribution(
        cls,
        true_value,
        dist_builder,
        *dist_args,
        provenance=None,
        name_prefix="R",
        **dist_kwargs,
    ):
        """
        Build a NoisyValue from a SymPy distribution constructor.

        Example: NoisyValue.from_distribution(10, Exponential, 2)
        """
        name = fresh_noise_name(name_prefix)
        noise_rv = dist_builder(name, *dist_args, **dist_kwargs)
        return cls.from_noise_rv(true_value, noise_rv, provenance=provenance)

    @staticmethod
    def _combine(a, b, op):
        if isinstance(b, NoisyValue):
            expr = op(a.expr, b.expr)
            observed = op(a.observed, b.observed)
            thetas = a.thetas | b.thetas
            equations = a.equations + b.equations
            return NoisyValue(expr, observed, thetas, equations)

        expr = op(a.expr, b)
        observed = op(a.observed, b)
        return NoisyValue(expr, observed, a.thetas, a.equations)

    def __add__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: a + b)

    def __radd__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: b + a)

    def __sub__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: a - b)

    def __rsub__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: b - a)

    def __mul__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: a * b)

    def __rmul__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: b * a)

    def __truediv__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: b / a)

    def _solve_theta_substitutions(self):
        eqs = [sp.Eq(eq, 0) for eq in self.equations]
        thetas = list(self.thetas)

        sol = sp.solve(eqs, thetas, dict=True)
        if not sol:
            raise ValueError("Could not solve for latent variables")

        chosen = sol[0]
        missing = self.thetas - set(chosen.keys())
        if missing:
            raise ValueError(f"Latent variables are underidentified: {missing}")

        return chosen

    def eliminate_thetas(self, noise_cloner=None):
        expr = self.expr
        sol = self._solve_theta_substitutions()
        substitutions = {}
        clone_cache = {}

        for theta, rhs in sol.items():
            if noise_cloner is not None:
                replacements = {}
                for rv in random_symbols(rhs):
                    if rv not in clone_cache:
                        cloned = noise_cloner(rv)
                        cloned_symbols = random_symbols(cloned)
                        if len(cloned_symbols) != 1 or cloned not in cloned_symbols:
                            raise TypeError("noise_cloner must return a single SymPy random variable")
                        clone_cache[rv] = cloned
                    replacements[rv] = clone_cache[rv]
                rhs = rhs.subs(replacements)

            substitutions[theta] = rhs

        return expr.subs(substitutions)

    def sample_n(self, n=1000, noise_cloner=None, library="scipy", seed=None, **sample_kwargs):
        if n <= 0:
            return np.array([], dtype=float)

        sample_seed = seed
        if isinstance(seed, int):
            sample_seed = np.random.default_rng(seed)

        if noise_cloner is not None:
            expr = self.eliminate_thetas(noise_cloner=noise_cloner)
            if not random_symbols(expr):
                return np.full(n, float(expr), dtype=float)

            try:
                values = sample(expr, size=n, library=library, seed=sample_seed, **sample_kwargs)
                return np.asarray(values, dtype=float)
            except Exception:
                samples = []
                for _ in range(n):
                    samples.append(float(sample(expr, library=library, seed=sample_seed, **sample_kwargs)))
                return np.asarray(samples, dtype=float)

        if not self.thetas:
            expr = self.expr
            if not random_symbols(expr):
                return np.full(n, float(expr), dtype=float)
            values = sample(expr, size=n, library=library, seed=sample_seed, **sample_kwargs)
            return np.asarray(values, dtype=float)

        sol = self._solve_theta_substitutions()
        rhs_noise_vars = list({rv for rhs in sol.values() for rv in random_symbols(rhs)})
        predictive_noise_vars = list(random_symbols(self.expr))

        samples = []
        for _ in range(n):
            rhs_noise_draws = {
                rv: float(sample(rv, library=library, seed=sample_seed, **sample_kwargs))
                for rv in rhs_noise_vars
            }
            theta_values = {
                theta: float(rhs.subs(rhs_noise_draws))
                for theta, rhs in sol.items()
            }
            predictive_noise_draws = {
                rv: float(sample(rv, library=library, seed=sample_seed, **sample_kwargs))
                for rv in predictive_noise_vars
            }

            value = float(self.expr.subs(theta_values).subs(predictive_noise_draws))
            samples.append(value)

        return np.asarray(samples, dtype=float)


def _evaluate_random_expr(expr, rng, library="scipy", **sample_kwargs):
    """Substitute one fresh numeric draw for each random symbol in expr."""
    value = expr
    for rv in random_symbols(value):
        value = value.subs(rv, float(sample(rv, library=library, seed=rng, **sample_kwargs)))
    return float(value)


def plot_confidence_heatmap(
    noisy_value,
    theta_range=None,
    observed_range=None,
    grid_size=201,
    n_samples=6000,
    cmap="viridis_r",
    ax=None,
    seed=None,
    library="scipy",
    bandwidth=None,
    contour_levels=(0.5, 0.8, 0.9, 0.95),
    contour_color="white",
    contour_linewidth=1.2,
    contour_labels=True,
    **sample_kwargs,
):
    """
    Plot a confidence-threshold heat map for the composed quantity in `noisy_value`.

    Axes
    -----
    x-axis: true value of `noisy_value.expr`
    y-axis: observed value of the same composed quantity

    Heat value
    ----------
    For each (x, y), color is the smallest central-interval probability level
    that contains y in the conditional observed distribution for that x-column.

    This function supports multiple latent thetas and multiple equations by
    Monte Carlo: draw latent theta vectors from all equations, then simulate a
    fresh observed realization of the composed expression.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if not noisy_value.thetas:
        raise ValueError("Heatmap requires at least one latent theta in NoisyValue")

    sol = noisy_value._solve_theta_substitutions()
    rhs_noise_vars = list({rv for rhs in sol.values() for rv in random_symbols(rhs)})

    # Infer one measurement-noise RV per theta from equations that reference it.
    theta_noise_rv = {}
    for theta in noisy_value.thetas:
        eqs_for_theta = [eq for eq in noisy_value.equations if theta in eq.free_symbols]
        if not eqs_for_theta:
            raise ValueError(f"No observation equation references latent {theta}")

        chosen_rv = None
        for eq in eqs_for_theta:
            rvs = list(random_symbols(eq))
            if rvs:
                chosen_rv = rvs[0]
                break
        if chosen_rv is None:
            raise ValueError(f"Could not infer measurement noise RV for latent {theta}")
        theta_noise_rv[theta] = chosen_rv

    rng = seed
    if isinstance(seed, int):
        rng = np.random.default_rng(seed)

    x_true_samples = np.empty(n_samples, dtype=float)
    y_obs_samples = np.empty(n_samples, dtype=float)

    for i in range(n_samples):
        rhs_noise_draws = {
            rv: float(sample(rv, library=library, seed=rng, **sample_kwargs))
            for rv in rhs_noise_vars
        }
        theta_values = {
            theta: float(rhs.subs(rhs_noise_draws))
            for theta, rhs in sol.items()
        }

        x_true_samples[i] = _evaluate_random_expr(
            noisy_value.expr.subs(theta_values),
            rng,
            library=library,
            **sample_kwargs,
        )

        observed_theta_values = {
            theta: theta_values[theta] + float(sample(theta_noise_rv[theta], library=library, seed=rng, **sample_kwargs))
            for theta in noisy_value.thetas
        }
        y_obs_samples[i] = _evaluate_random_expr(
            noisy_value.expr.subs(observed_theta_values),
            rng,
            library=library,
            **sample_kwargs,
        )

    if theta_range is None:
        lo, hi = np.quantile(x_true_samples, [0.005, 0.995])
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        theta_range = (float(lo - pad), float(hi + pad))

    if observed_range is None:
        lo, hi = np.quantile(y_obs_samples, [0.005, 0.995])
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        observed_range = (float(lo - pad), float(hi + pad))

    theta_grid = np.linspace(theta_range[0], theta_range[1], grid_size)
    observed_grid = np.linspace(observed_range[0], observed_range[1], grid_size)

    if bandwidth is None:
        x_std = float(np.std(x_true_samples, ddof=1))
        if not np.isfinite(x_std) or x_std == 0.0:
            x_std = max(1e-6, (theta_range[1] - theta_range[0]) / 20.0)
        bandwidth = 0.2 * x_std
    bandwidth = float(max(bandwidth, 1e-12))

    heat = np.empty((grid_size, grid_size), dtype=float)
    for j, x0 in enumerate(theta_grid):
        z = (x_true_samples - x0) / bandwidth
        w = np.exp(-0.5 * z * z)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0 or not np.isfinite(w_sum):
            w = np.full_like(w, 1.0 / len(w))
        else:
            w /= w_sum

        center = float(np.sum(w * y_obs_samples))
        distances = np.abs(y_obs_samples - center)
        query_radius = np.abs(observed_grid - center)

        heat[:, j] = np.array([float(np.sum(w[distances <= r])) for r in query_radius], dtype=float)

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
        created_fig = True

    im = ax.imshow(
        heat,
        origin="lower",
        extent=[theta_grid[0], theta_grid[-1], observed_grid[0], observed_grid[-1]],
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )

    if contour_levels:
        levels = sorted({float(v) for v in contour_levels if 0.0 < float(v) < 1.0})
        if levels:
            xx, yy = np.meshgrid(theta_grid, observed_grid)
            contours = ax.contour(
                xx,
                yy,
                heat,
                levels=levels,
                colors=contour_color,
                linewidths=contour_linewidth,
                alpha=0.95,
            )
            if contour_labels:
                fmt = {lv: f"{int(round(lv * 100))}%" for lv in contours.levels}
                ax.clabel(contours, contours.levels, inline=True, fontsize=8, fmt=fmt)

    ax.set_xlabel("True value")
    ax.set_ylabel("Observed value")
    ax.set_title("Confidence-Interval Threshold Heatmap")
    ax.plot(theta_grid, theta_grid, color="white", linestyle="--", linewidth=1.0, alpha=0.5)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Smallest central CI probability containing observed value")

    if created_fig:
        plt.tight_layout()

    return {
        "ax": ax,
        "heat": heat,
        "theta_grid": theta_grid,
        "observed_grid": observed_grid,
        "x_true_samples": x_true_samples,
        "y_obs_samples": y_obs_samples,
        "bandwidth": bandwidth,
    }
