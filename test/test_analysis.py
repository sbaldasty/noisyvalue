import src.analysis as analysis
import src.noise as noise


def test_noisy_odds_ratio_matches_closed_form_for_plain_floats():
    result = analysis.noisy_odds_ratio(65.0, 109.0, 243.0, 1348.0, correction=0.0)
    expected = (65.0 * 1348.0) / (109.0 * 243.0)

    assert float(result) == expected


def test_quantile_ci_with_sampling_keeps_sampling_uncertainty():
    table = analysis.NoisyTable2x2.from_cells(65.0, 109.0, 243.0, 1348.0)
    q_low, q_high = analysis.oddsratio_confint(
        table,
        n=4000,
        seed=123,
        correction=0.0)

    assert q_low < q_high
    assert q_low > 0


def test_quantile_ci_with_noisy_inputs_runs_and_returns_float_bounds():
    rng = analysis.np.random.default_rng(42)
    noise_factory = noise.gaussian(loc=0, scale=1)

    # Add explicit release noise layer to inputs.
    from src.release import noisy_float

    table = analysis.NoisyTable2x2.from_cells(
        noisy_float(65.0, noise_factory, seed=rng),
        noisy_float(109.0, noise_factory, seed=rng),
        noisy_float(243.0, noise_factory, seed=rng),
        noisy_float(1348.0, noise_factory, seed=rng),
    )

    q_low, q_high = analysis.oddsratio_confint(
        table,
        n=2000,
        seed=123,
        correction=0.0,
        include_sampling=True,
    )

    assert isinstance(q_low, float)
    assert isinstance(q_high, float)
    assert q_low < q_high


def test_method_form_matches_function_form():
    table = analysis.NoisyTable2x2.from_cells(65.0, 109.0, 243.0, 1348.0)
    via_method = table.oddsratio_confint(
        n=2000,
        seed=123,
        correction=0.0,
        include_sampling=True,
    )
    via_function = analysis.oddsratio_confint(
        table,
        n=2000,
        seed=123,
        correction=0.0,
        include_sampling=True,
    )

    assert via_method == via_function
