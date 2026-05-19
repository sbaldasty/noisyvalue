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
