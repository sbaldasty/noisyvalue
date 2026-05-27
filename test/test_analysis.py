import numpy as np
import pytest
import sympy as sp

import src.analysis as analysis

from src.core import NoisyFloat


def test_noisy_min_and_noisy_max_for_plain_floats_match_python_min_max():
    lo = analysis.noisy_min(3.0, -1.5, 8.0)
    hi = analysis.noisy_max(3.0, -1.5, 8.0)

    assert isinstance(lo, NoisyFloat)
    assert isinstance(hi, NoisyFloat)
    assert float(lo) == -1.5
    assert float(hi) == 8.0


def test_noisy_min_raises_for_empty_input():
    with pytest.raises(ValueError, match="Requires at least one value"):
        analysis.noisy_min()


def test_noisy_max_combines_noisy_value_metadata():
    theta = sp.Symbol("theta_fold")
    a = NoisyFloat(obs=1.0, expr=theta, thetas={theta}, eqns=[theta - 1.0])
    b = NoisyFloat(obs=2.0, expr=2.0 * theta, thetas={theta}, eqns=[theta - 1.0])

    out = analysis.noisy_max(a, b)

    assert isinstance(out, NoisyFloat)
    assert float(out) == 2.0
    assert out._thetas == {theta}
    assert len(out._eqns) == 2


def test_odds_ratio_init_enforces_2x2_shape():
    with pytest.raises(AssertionError):
        analysis.OddsRatio([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_odds_ratio_ratio_matches_closed_form_for_plain_floats():
    ratio = analysis.OddsRatio([[65.0, 109.0], [243.0, 1348.0]]).ratio()
    expected = (65.0 * 1348.0) / (109.0 * 243.0)

    assert isinstance(ratio, NoisyFloat)
    assert float(ratio) == pytest.approx(expected)


def test_odds_ratio_ratio_returns_none_for_non_positive_cells():
    ratio = analysis.OddsRatio([[1.0, 0.0], [2.0, 3.0]]).ratio()
    assert ratio is None


def test_odds_ratio_sample_keeps_only_valid_draws():
    model = analysis.OddsRatio([[5.0, 7.0], [11.0, 13.0]])

    model.sample(n=400, rng=123)

    assert isinstance(model.samples, np.ndarray)
    assert model.samples.ndim == 1
    assert 0 < model.samples.size <= 400
    assert np.all(np.isfinite(model.samples))
    assert np.all(model.samples > 0.0)


def test_odds_ratio_sample_requires_positive_n():
    model = analysis.OddsRatio([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(AssertionError):
        model.sample(n=0)


def test_confidence_interval_autosamples_when_needed(monkeypatch):
    model = analysis.OddsRatio([[1.0, 2.0], [3.0, 4.0]])

    def fake_sample(self, n=1000, rng=None, lib="scipy"):
        self.samples = np.array([0.4, 0.8, 1.2, 1.6], dtype=float)
        return self

    monkeypatch.setattr(analysis.OddsRatio, "sample", fake_sample)
    lo, hi = model.confidence_interval(a=0.10)

    assert lo == pytest.approx(np.quantile([0.4, 0.8, 1.2, 1.6], 0.05))
    assert hi == pytest.approx(np.quantile([0.4, 0.8, 1.2, 1.6], 0.95))


def test_confidence_interval_validates_alpha_bounds():
    model = analysis.OddsRatio([[1.0, 2.0], [3.0, 4.0]])
    model.samples = np.array([1.0, 2.0], dtype=float)

    with pytest.raises(AssertionError):
        model.confidence_interval(a=-0.01)

    with pytest.raises(AssertionError):
        model.confidence_interval(a=1.01)


def test_confidence_interval_raises_when_no_valid_draws(monkeypatch):
    model = analysis.OddsRatio([[1.0, 2.0], [3.0, 4.0]])

    def fake_sample(self, n=1000, rng=None, lib="scipy"):
        self.samples = np.array([], dtype=float)
        return self

    monkeypatch.setattr(analysis.OddsRatio, "sample", fake_sample)

    with pytest.raises(ValueError, match="No valid odds ratio draws"):
        model.confidence_interval()
