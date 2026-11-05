"""Posteriors of coarsened (rounded) releases: uniform and triangular nodes."""
from collections import Counter

import numpy as np
import pytest
import scipy.stats

from noisyvalue import NoisyFloat, NoisyInt, plot_posterior, sample_noisy_values
from noisyvalue.graph import (DiscreteTriangularNode, DiscreteUniformNode,
                              TriangularNode, UniformNode)
from noisyvalue.io import load, save


def _draws(value, n=40000, rng=0):
    return sample_noisy_values(value, n=n, rng=rng)[0].draws


def _freqs(value, n=60000):
    counts = Counter(_draws(value, n=n).tolist())
    return {k: counts[k] / n for k in sorted(counts)}


# ── continuous ────────────────────────────────────────────────────────────────

def test_float_rounded_nearest_is_uniform_on_bin():
    x = NoisyFloat.rounded(120, base=10)
    assert isinstance(x, NoisyFloat) and float(x) == 120.0
    assert isinstance(x._root, UniformNode)
    draws = _draws(x)
    assert draws.min() >= 115 and draws.max() <= 125
    assert scipy.stats.kstest(draws, scipy.stats.uniform(115, 10).cdf).pvalue > 0.01


def test_float_rounded_random_is_triangular_twice_as_wide():
    x = NoisyFloat.rounded(120, base=10, method="random")
    assert isinstance(x._root, TriangularNode)
    draws = _draws(x)
    assert draws.min() > 110 and draws.max() < 130
    ref = scipy.stats.triang(c=0.5, loc=110, scale=20)
    assert scipy.stats.kstest(draws, ref.cdf).pvalue > 0.01


def test_float_rounded_respects_truncation():
    nearest = _draws(NoisyFloat.rounded(0, base=10, low=0))
    assert nearest.min() >= 0 and nearest.max() <= 5
    random = _draws(NoisyFloat.rounded(0, base=10, method="random", low=0))
    assert random.min() >= 0 and random.max() < 10
    # the right half of a triangle peaked at 0
    ref = scipy.stats.triang(c=0.0, loc=0, scale=10)
    assert scipy.stats.kstest(random, ref.cdf).pvalue > 0.01


def test_float_rounded_accepts_fractional_base():
    x = NoisyFloat.rounded(0.3, base=0.1)
    draws = _draws(x)
    assert draws.min() >= 0.25 - 1e-12 and draws.max() <= 0.35 + 1e-12


def test_float_triangular_truncated_matches_scipy():
    x = NoisyFloat.triangular(0, 2, low=-1, high=1.5)
    draws = _draws(x)
    ref = scipy.stats.triang(c=0.5, loc=-2, scale=4)
    lo, hi = ref.cdf(-1), ref.cdf(1.5)
    expected = lambda t: (ref.cdf(np.clip(t, -1, 1.5)) - lo) / (hi - lo)
    assert scipy.stats.kstest(draws, expected).pvalue > 0.01


def test_sum_of_independently_rounded_values_shrinks_like_irwin_hall():
    values = [NoisyFloat.rounded(float(k), base=1) for k in range(100)]
    total = sum(values[1:], values[0])
    assert float(total) == sum(range(100))
    assert _draws(total, n=20000).std() == pytest.approx(np.sqrt(100 / 12), rel=0.05)


def test_rounded_after_gaussian_noise_convolves():
    # released = round(theta + eps), so theta = (bin posterior) - eps
    theta = NoisyFloat.rounded(120, base=10) - NoisyFloat.gaussian(0, 3, obs=0)
    assert _draws(theta).std() == pytest.approx(np.sqrt(100 / 12 + 9), rel=0.03)


def test_float_uniform_with_noisy_bound_samples_hierarchically():
    high = NoisyFloat.uniform(1, 2, obs=1.5)
    x = NoisyFloat.uniform(0, high)
    a, b = sample_noisy_values(x, high, n=40000, rng=0)
    assert np.all(a.draws <= b.draws)
    assert a.draws.mean() == pytest.approx(0.75, abs=0.01)


def test_float_triangular_with_noisy_loc_samples_hierarchically():
    x = NoisyFloat.triangular(NoisyFloat.uniform(0, 1, obs=0.5), 1)
    # variance of the sum: 1/12 (uniform loc) + 1/6 (unit triangle)
    assert _draws(x).var() == pytest.approx(0.25, rel=0.05)


def test_float_constructors_draw_obs_from_rng_when_omitted():
    assert 2 <= float(NoisyFloat.uniform(2, 3, rng=0)) <= 3
    assert -1 < float(NoisyFloat.triangular(0, 1, rng=0)) < 1
    assert float(NoisyFloat.uniform(2, 3, rng=5)) == float(NoisyFloat.uniform(2, 3, rng=5))


# ── integer ───────────────────────────────────────────────────────────────────

def test_int_rounded_nearest_odd_base_is_symmetric():
    x = NoisyInt.rounded(20, base=5)
    assert isinstance(x, NoisyInt) and int(x) == 20
    freqs = _freqs(x)
    assert list(freqs) == [18, 19, 20, 21, 22]
    assert all(f == pytest.approx(0.2, abs=0.01) for f in freqs.values())


@pytest.mark.parametrize("obs, ties, support", [
    (20, "up", range(15, 25)),
    (20, "down", range(16, 26)),
    (20, "even", range(15, 26)),   # 20/10 is even: both ties come here
    (10, "even", range(6, 15)),    # 10/10 is odd: both ties go elsewhere
    (-10, "even", range(-14, -5)),
])
def test_int_rounded_nearest_even_base_tie_rules(obs, ties, support):
    freqs = _freqs(NoisyInt.rounded(obs, base=10, ties=ties))
    assert list(freqs) == list(support)
    assert all(f == pytest.approx(1 / len(support), abs=0.01) for f in freqs.values())


def test_int_rounded_nearest_matches_python_round_for_even_ties():
    for obs in range(-30, 40, 10):
        x = NoisyInt.rounded(obs, base=10, ties="even")
        for k in set(_draws(x, n=2000).tolist()):
            assert round(k / 10) * 10 == obs


def test_int_rounded_random_matches_exact_likelihood():
    freqs = _freqs(NoisyInt.rounded(30, base=5, method="random"))
    assert list(freqs) == list(range(26, 35))
    weights = {k: 1 - abs(k - 30) / 5 for k in range(26, 35)}
    total = sum(weights.values())
    for k, f in freqs.items():
        assert f == pytest.approx(weights[k] / total, abs=0.01)


def test_int_rounded_random_zero_with_nonnegativity():
    # true counts 0, 1, 2 round to 0 with probability 1, 2/3, 1/3
    freqs = _freqs(NoisyInt.rounded(0, base=3, method="random", low=0))
    assert list(freqs) == [0, 1, 2]
    for k, p in zip(freqs, (1 / 2, 1 / 3, 1 / 6)):
        assert freqs[k] == pytest.approx(p, abs=0.01)


def test_int_rounded_random_is_calibrated_against_simulated_rounding():
    # forward-simulate unbiased random rounding over a flat prior, then
    # check the posterior of each released value matches the empirical one
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 1000, size=400000)
    frac = (truth % 10) / 10
    released = 10 * (truth // 10 + (rng.uniform(size=truth.size) < frac))
    for r in (300, 500):
        empirical = Counter(truth[released == r].tolist())
        n = sum(empirical.values())
        model = _freqs(NoisyInt.rounded(r, base=10, method="random"))
        assert set(empirical) == set(model)
        for k in model:
            assert empirical[k] / n == pytest.approx(model[k], abs=0.01)


def test_int_rounded_base_one_is_a_point_mass():
    assert set(_draws(NoisyInt.rounded(7)).tolist()) == {7}
    assert set(_draws(NoisyInt.rounded(7, method="random")).tolist()) == {7}


def test_int_uniform_with_noisy_bound_samples_hierarchically():
    x = NoisyInt.uniform(0, NoisyInt.rounded(10, base=5))
    draws = _draws(x)
    assert draws.dtype.kind == "i"
    assert draws.mean() == pytest.approx(5.0, abs=0.05)


def test_int_triangular_with_noisy_loc_samples_hierarchically():
    x = NoisyInt.triangular(NoisyInt.uniform(0, 3, obs=1), 4)
    assert _draws(x).mean() == pytest.approx(1.5, abs=0.05)


def test_int_constructors_draw_obs_from_rng_when_omitted():
    assert 3 <= int(NoisyInt.uniform(3, 7, rng=0)) <= 7
    assert 2 <= int(NoisyInt.triangular(5, 3, rng=0)) <= 8


# ── validation ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("make", [
    lambda: NoisyInt.rounded(12, base=5),
    lambda: NoisyFloat.rounded(12.0, base=5),
    lambda: NoisyInt.rounded(10, base=2.5),
    lambda: NoisyInt.rounded(10, base=0),
    lambda: NoisyFloat.rounded(10, base=-5),
    lambda: NoisyInt.rounded(10, base=5, method="banker"),
    lambda: NoisyInt.rounded(10, base=10, ties="sideways"),
    lambda: NoisyInt.rounded(-10, base=5, low=0),
    lambda: NoisyFloat.rounded(-10, base=5, low=0),
    lambda: NoisyFloat.rounded(-10, base=5, method="random", low=0),
    lambda: NoisyInt.rounded(-10, base=5, method="random", low=0),
    lambda: NoisyFloat.uniform(0, float("inf")),
    lambda: NoisyFloat.uniform(3, 2),
    lambda: NoisyInt.uniform(2.2, 2.8),
])
def test_invalid_rounding_inputs_raise(make):
    with pytest.raises(ValueError):
        make()


def test_rounded_rejects_noisy_obs():
    with pytest.raises(TypeError):
        NoisyFloat.rounded(NoisyFloat.uniform(0, 1, obs=0.5))


def test_invalid_hierarchical_bounds_give_nan_not_garbage():
    high = NoisyFloat.uniform(-3, 3, obs=1)
    x = NoisyFloat.uniform(0, high)
    draws, highs = (b.draws for b in sample_noisy_values(x, high, n=2000, rng=0))
    assert np.all(np.isnan(draws) == (highs < 0))


# ── quantiles, visualization and IO ───────────────────────────────────────────

@pytest.mark.parametrize("value", [
    NoisyFloat.rounded(120, base=10),
    NoisyFloat.rounded(120, base=10, method="random", low=115),
    NoisyInt.rounded(20, base=10, ties="even"),
    NoisyInt.rounded(0, base=3, method="random", low=0),
])
def test_node_quantile_matches_sampled_quantiles(value):
    u = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    exact = value._root.quantile(u)
    sampled = np.quantile(_draws(value, n=100000), u)
    assert np.all(np.diff(exact) >= 0)
    assert exact == pytest.approx(sampled, abs=0.15)


def test_plot_posterior_accepts_rounded_values():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    ax = plot_posterior(NoisyFloat.rounded(120, base=10, method="random"))
    assert ax is not None


@pytest.mark.parametrize("value", [
    NoisyFloat.rounded(120, base=10),
    NoisyFloat.rounded(120, base=10, method="random", low=0),
    NoisyInt.rounded(20, base=10, ties="even"),
    NoisyInt.rounded(0, base=3, method="random", low=0),
])
def test_rounded_values_survive_roundtrip(tmp_path, value):
    path = tmp_path / "data.json"
    save(path, value)
    loaded = load(path)
    assert type(loaded) is type(value) and loaded._obs == value._obs
    original = _draws(value, n=5000, rng=3)
    assert np.array_equal(_draws(loaded, n=5000, rng=3), original)
