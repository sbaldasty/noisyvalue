import math

import numpy as np
import pandas
import pytest

duet = pytest.importorskip("duet")
if not hasattr(pytest.importorskip("duet.duet_vals"), "Release"):
    pytest.skip("needs DDuo with details=True mechanisms", allow_module_level=True)
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


# ── delegation to DDuo: same noise, same accounting ────────────────────────

@pytest.mark.parametrize("ours, theirs, params", [
    (dduo.gauss, duet.gauss, {"ε": 0.5, "δ": 1e-5}),
    (dduo.laplace, duet.laplace, {"ε": 0.5}),
    (dduo.renyi_gauss, duet.renyi_gauss, {"α": 10, "ε": 0.5}),
])
def test_scalar_release_matches_dduo(df, charges, ours, theirs, params):
    total = df["age"].clip(0, 100).sum()
    np.random.seed(7)
    expected = theirs(total, **params)
    theirs_log = list(charges)
    charges.clear()

    np.random.seed(7)
    value = ours(total, **params)

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
    value = dduo.laplace(df["hours"].clip(-10, 10).sum(), epsilon=0.5)
    draws = _draws(value)
    # rows clipped to [-10, 10] move a sum by 20 -> Laplace scale 40 -> sd 40*sqrt(2)
    assert abs(np.median(draws) - float(value)) < 2
    assert abs(draws.std() - 40 * math.sqrt(2)) < 3


def test_gauss_posterior_has_mechanism_scale(df):
    release = duet.gauss(df.shape[0], epsilon=0.5, delta=1e-5, details=True)
    value = dduo.gauss(df.shape[0], epsilon=0.5, delta=1e-5, high=1e6)
    sigma = math.sqrt(2 * math.log(1.25 / 1e-5)) / 0.5
    assert release.scale == pytest.approx(sigma)
    draws = _draws(value)
    # not testing the mean: DDuo's lower bound of 0 truncates this posterior
    assert draws.std() > 0.5 * sigma


def test_row_release_scale_comes_from_bounds(df):
    # each element of a clipped column is one row: its sensitivity is the
    # clip width, which only DDuo's release sensitivity knows
    registry = SharedLatentRegistry()
    dduo.gauss(df["hours"].clip(-30, 30).values, epsilon=0.5, delta=1e-5,
               keys=range(10), registry=registry)
    sigma = math.sqrt(2 * math.log(1.25 / 1e-5)) * 60 / 0.5
    assert registry.posterior(0)[1] == pytest.approx(sigma ** 2)


def test_count_posterior_is_nonnegative_without_low(df):
    value = dduo.laplace(df.shape[0], epsilon=0.1)
    assert _draws(value).min() >= 0


def test_clipped_column_release_is_bounded(df):
    released = dduo.laplace(df["age"].clip(0, 100), epsilon=0.1)
    draws = _draws(released.iloc[0])
    assert draws.min() >= 0 and draws.max() <= 100


def test_caller_bounds_narrow_dduo_bounds(df):
    value = dduo.laplace(df.shape[0], epsilon=0.1, low=-50, high=12)
    draws = _draws(value)
    assert draws.min() >= 0          # DDuo's 0 wins over the looser -50
    assert draws.max() <= 12         # the caller's 12 adds to DDuo's inf


def test_ratio_of_releases_propagates_posterior(df):
    total = dduo.laplace(df["age"].clip(0, 100).sum(), epsilon=1.0)
    count = dduo.laplace(df.shape[0], epsilon=1.0)
    mean = total / count
    assert isinstance(mean, NoisyFloat)
    assert float(mean) == pytest.approx(float(total) / float(count))
    # the sum's sensitivity is the clip width (100), so the mean's posterior
    # is much wider than the count's alone would make it
    assert _draws(mean, n=5000).std() > 5


def test_series_release_keeps_index(df):
    clipped = df["age"].clip(0, 100)
    released = dduo.laplace(clipped, epsilon=1.0)
    assert isinstance(released, pandas.Series)
    assert isinstance(released.array, NoisyFloatArray)
    assert list(released.index) == list(range(10))


def test_repeated_gaussian_releases_share_a_key(df):
    registry = SharedLatentRegistry()
    total = df["hours"].clip(-1, 1).sum()   # sensitivity 2, no bound on the sum
    first = dduo.gauss(total, epsilon=0.5, delta=1e-5, key="n", registry=registry)
    dduo.gauss(total, epsilon=0.5, delta=1e-5, key="n", registry=registry)
    sigma = math.sqrt(2 * math.log(1.25 / 1e-5)) * 2 / 0.5
    _, variance = registry.posterior("n")
    assert variance == pytest.approx(sigma ** 2 / 2)
    # `first` was issued before the second release, and tightens in place
    assert _draws(first).std() == pytest.approx(sigma / math.sqrt(2), rel=0.05)


def test_vector_keys(df):
    registry = SharedLatentRegistry()
    released = dduo.gauss(df["hours"].clip(0, 60).values, epsilon=0.5, delta=1e-5,
                          keys=range(10), registry=registry)
    assert len(released) == 10
    assert registry.posterior(3)[0] == pytest.approx(float(released[3]))


# ── refusals, before any budget is spent ───────────────────────────────────

def test_untracked_value_is_refused(charges):
    with pytest.raises(TypeError, match="DuetWrapper"):
        dduo.laplace(10, epsilon=1.0)
    with pytest.raises(TypeError, match="DuetWrapper"):
        dduo.renyi_gauss(10.0, alpha=10, epsilon=1.0)
    assert charges == []


@pytest.mark.parametrize("unbounded", [
    lambda df: df["age"].sum(),
    lambda df: df["age"],
])
def test_unbounded_sensitivity_is_refused(df, charges, unbounded):
    with pytest.raises(ValueError, match="unbounded"):
        dduo.laplace(unbounded(df), epsilon=1.0)
    assert charges == []


def test_zero_sensitivity_is_refused(df, charges):
    with pytest.raises(ValueError, match="Sensitivity is 0"):
        dduo.laplace(df.shape[0] * 0, epsilon=1.0)
    assert charges == []


def test_missing_and_duplicate_parameters(df, charges):
    with pytest.raises(TypeError, match="Missing"):
        dduo.gauss(df.shape[0], epsilon=1.0)
    with pytest.raises(TypeError, match="once"):
        dduo.laplace(df.shape[0], epsilon=1.0, ε=1.0)
    assert charges == []


def test_key_shape_must_match_value(df, charges):
    registry = SharedLatentRegistry()
    with pytest.raises(ValueError, match="keys="):
        dduo.laplace(df["age"].clip(0, 1), epsilon=1.0, key="x", registry=registry)
    with pytest.raises(ValueError, match="key="):
        dduo.laplace(df.shape[0], epsilon=1.0, keys=["x"], registry=registry)
    with pytest.raises(ValueError, match="registry"):
        dduo.laplace(df.shape[0], epsilon=1.0, key="x")
    with pytest.raises(ValueError, match="10 released elements for 3 keys"):
        dduo.laplace(df["age"].clip(0, 1), epsilon=1.0, keys="abc", registry=registry)
    assert charges == []
