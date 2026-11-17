import math

import numpy as np
import pandas
import pytest

duet = pytest.importorskip("duet")
import duet.pandas as dpd  # noqa: E402

from noisyvalue import dduo  # noqa: E402
from noisyvalue.core import NoisyFloat  # noqa: E402
from noisyvalue.pandas import NoisyFloatArray  # noqa: E402
from noisyvalue.release import SharedLatentRegistry  # noqa: E402
from noisyvalue.sample import sample_noisy_values  # noqa: E402


@pytest.fixture
def df(tmp_path):
    path = tmp_path / "people.csv"
    pandas.DataFrame({
        "age": [23, 35, 47, 51, 62, 18, 29, 90, 41, 38],
        "hours": [40, 50, 20, 45, 10, 30, 60, 5, 40, 35],
    }).to_csv(path, index=False)
    return dpd.read_csv(str(path))


class _Spy:
    """Stands in for every kind of filter and odometer, recording charges."""

    def __init__(self, kind, log):
        self.kind, self.log = kind, log

    def pay(self, a, b):
        self.log.append((self.kind, "pay", a, b))

    def add(self, penv):
        self.log.append((self.kind, "add", type(penv).__name__,
                         sorted((str(k), float(v)) for k, v in penv.fp.items()),
                         sorted((str(k), float(v)) for k, v in penv.sp.items())))


@pytest.fixture
def charges():
    log = []
    acct = duet.accountant
    spies = [(acct.register_filter, acct.unregister_filter, _Spy("f", log)),
             (acct.register_advf, acct.unregister_advf, _Spy("advf", log)),
             (acct.register_odometer, acct.unregister_odometer, _Spy("od", log)),
             (acct.register_advod, acct.unregister_advod, _Spy("advod", log))]
    for reg, _, spy in spies:
        reg(spy)
    yield log
    for _, unreg, spy in spies:
        unreg(spy)


def _draws(value, n=20000, seed=0):
    return sample_noisy_values(value, n=n, rng=np.random.default_rng(seed))[0].draws


# ── parity with DDuo: same noise, same accounting ──────────────────────────

@pytest.mark.parametrize("ours, theirs, params", [
    (dduo.gauss, duet.gauss, {"ε": 0.5, "δ": 1e-5}),
    (dduo.laplace, duet.laplace, {"ε": 0.5}),
    (dduo.renyi_gauss, duet.renyi_gauss, {"α": 10, "ε": 0.5}),
])
def test_scalar_release_matches_dduo(df, charges, ours, theirs, params):
    np.random.seed(7)
    expected = theirs(df.shape[0], **params)
    theirs_log = list(charges)
    charges.clear()

    np.random.seed(7)
    value = ours(df.shape[0], **params)

    assert isinstance(value, NoisyFloat)
    assert float(value) == pytest.approx(expected)
    assert charges == theirs_log


def test_vector_release_matches_dduo(df, charges):
    hours = df["hours"].clip(0, 60).values
    np.random.seed(3)
    expected = duet.renyi_gauss_vec(hours, α=10, ε=1.0)
    theirs_log = list(charges)
    charges.clear()

    np.random.seed(3)
    released = dduo.renyi_gauss_vec(hours, α=10, ε=1.0)

    assert isinstance(released, NoisyFloatArray)
    np.testing.assert_allclose([float(v) for v in released], expected)
    assert charges == theirs_log


def test_filter_still_enforced(df):
    with pytest.raises(duet.PrivacyFilterException):
        with duet.AdvEDFilterObj(0.5, 1e-3):
            for _ in range(5):
                dduo.gauss(df.shape[0], epsilon=0.3, delta=1e-5)


# ── posteriors ─────────────────────────────────────────────────────────────

def test_laplace_posterior_has_mechanism_scale(df):
    value = dduo.laplace(df.shape[0], epsilon=0.5)
    draws = _draws(value)
    # sensitivity 1, epsilon 0.5 -> Laplace scale 2 -> sd 2*sqrt(2)
    assert abs(np.median(draws) - float(value)) < 0.1
    assert abs(draws.std() - 2 * math.sqrt(2)) < 0.15


def test_gauss_posterior_has_mechanism_scale(df):
    value = dduo.gauss(df.shape[0], epsilon=0.5, delta=1e-5)
    sigma = math.sqrt(2 * math.log(1.25 / 1e-5)) / 0.5
    draws = _draws(value)
    assert abs(draws.mean() - float(value)) < 0.05 * sigma
    assert abs(draws.std() - sigma) < 0.05 * sigma


def test_low_bound_truncates_count_posterior(df):
    value = dduo.laplace(df.shape[0], epsilon=0.1, low=0)
    assert _draws(value).min() >= 0


def test_ratio_of_releases_propagates_posterior(df):
    total = dduo.laplace(df["age"].clip(0, 100).sum(), epsilon=1.0)
    count = dduo.laplace(df.shape[0], epsilon=1.0, low=0)
    mean = total / count
    assert isinstance(mean, NoisyFloat)
    assert float(mean) == pytest.approx(float(total) / float(count))
    # the sum's sensitivity is the clip width (100), so the mean's posterior
    # is much wider than the count's alone would make it
    assert _draws(mean, n=5000).std() > 5


def test_series_release_keeps_index(df):
    counts = df["age"].clip(0, 100)
    released = dduo.laplace(counts, epsilon=1.0)
    assert isinstance(released, pandas.Series)
    assert isinstance(released.array, NoisyFloatArray)
    assert list(released.index) == list(range(10))


def test_repeated_gaussian_releases_share_a_key(df):
    registry = SharedLatentRegistry()
    first = dduo.gauss(df.shape[0], epsilon=0.5, delta=1e-5, key="n", registry=registry)
    dduo.gauss(df.shape[0], epsilon=0.5, delta=1e-5, key="n", registry=registry)
    sigma = math.sqrt(2 * math.log(1.25 / 1e-5)) / 0.5
    _, variance = registry.posterior("n")
    assert variance == pytest.approx(sigma ** 2 / 2)
    # `first` was issued before the second release, and tightens in place
    assert _draws(first).std() == pytest.approx(sigma / math.sqrt(2), rel=0.05)


# ── refusals ───────────────────────────────────────────────────────────────

def test_untracked_value_is_refused():
    with pytest.raises(TypeError, match="DuetWrapper"):
        dduo.laplace(10, epsilon=1.0)


def test_unbounded_sensitivity_is_refused(df):
    # DDuo gives the product of two sensitive columns infinite sensitivity.
    # (An *unclipped* `df["age"].sum()` is not caught: DDuo's
    # `DistanceMetric.__eq__` compares only `bound`, so `LInf() == L1()` and
    # `sens_sum`'s mode assertion never fires -- the sum gets sensitivity 1.)
    with pytest.raises(ValueError, match="unbounded"):
        dduo.laplace((df["age"] * df["hours"]).sum(), epsilon=1.0)


def test_missing_and_duplicate_parameters(df):
    with pytest.raises(TypeError, match="Missing"):
        dduo.gauss(df.shape[0], epsilon=1.0)
    with pytest.raises(TypeError, match="once"):
        dduo.laplace(df.shape[0], epsilon=1.0, ε=1.0)


def test_key_shape_must_match_value(df):
    registry = SharedLatentRegistry()
    with pytest.raises(ValueError, match="keys="):
        dduo.laplace(df["age"].clip(0, 1), epsilon=1.0, key="x", registry=registry)
    with pytest.raises(ValueError, match="registry"):
        dduo.laplace(df.shape[0], epsilon=1.0, key="x")
