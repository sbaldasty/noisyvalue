"""DDuo mechanisms that release `NoisyValue`s.

DDuo (Abuah et al., CSF 2021; https://github.com/uvm-plaid/dduo-python)
tracks sensitivity dynamically: values read through `duet.pandas.read_csv`
etc. are `DuetWrapper`s whose sensitivity environment follows them through
ordinary pandas/numpy code.  Its mechanisms add calibrated noise and return
plain numbers.  The functions here are drop-in replacements for those
mechanisms that return `NoisyValue`s instead, whose posteriors encode the
noise law each release was made under::

    import duet
    from noisyvalue import dduo

    df = duet.pandas.read_csv("adult.csv")
    with duet.EDOdometer() as odo:
        total = dduo.laplace(df["age"].clip(0, 100).sum(), epsilon=1.0)
        count = dduo.laplace(df.shape[0], epsilon=1.0)
        mean = total / count     # a NoisyFloat; its posterior propagates

Each function calls the corresponding `duet` mechanism with
`details=True`, so calibration, the noise draw and the charges to the
active odometers and filters are DDuo's own, and builds the posterior from
the public facts in the returned `duet.Release`: the mechanism, its noise
scale and DDuo's public bounds on the true value.  Those bounds truncate
the posterior, so counts and sums of non-negative values are known to be
non-negative without being told.  `low`/`high` add the caller's own prior
knowledge and can only narrow DDuo's bounds, never widen them.

The guarantees are DDuo's, caveats included: noise comes from NumPy's
floating-point samplers rather than a vetted DP sampler, and the classic
Gaussian calibration `sigma = sqrt(2 ln(1.25/delta)) * sensitivity /
epsilon` is only valid for `epsilon < 1`.

Two refusals are checked *before* DDuo is called, so a refused release
spends no budget: an argument that is not a `DuetWrapper` (it may be
private data that was never wrapped), and a value whose release
sensitivity is infinite (nothing calibrated to it protects anyone) or zero
(it does not depend on any tracked data source).

Because DDuo noises with continuous laws, every release is a `NoisyFloat`
(or a `NoisyFloatArray` for vector-valued arguments), even for counts.
`key`/`keys` with a `SharedLatentRegistry` give repeated releases of one
statistic a common identity, exactly as in `noisyvalue.opendp`; DDuo tracks
sensitivity, not which statistic a value is, so the key has to come from
the caller.

What is deliberately *not* wrapped: `duet.report_noisy_max` and
`duet.above_threshold` release a selection (an element or an index), not a
noisy number, so there is no additive-noise posterior to attach.  A value
released *because* a noisy test selected it has a posterior that the flat
prior used here does not account for (the winner's curse).

Released values are public -- DP is closed under post-processing -- so do
not feed a `NoisyValue` back into a computation over `DuetWrapper`s (e.g. a
noisy gradient step updating parameters that the next gradient is computed
from): DDuo and numpy see an object, not a number.  Use `float(value)` (or
`np.asarray(array, dtype=float)`) at that boundary.

Noise is drawn by DDuo from the global `np.random` state, so seeding means
`np.random.seed(...)`.  TODO: these functions used to take `rng=` (a NumPy
`Generator`), which keeps reproducible runs independent of global state;
restoring it needs DDuo's mechanisms to accept a generator.

Requires the `duet` package with `details=True` support, which is not on
PyPI; install DDuo from its repository or put a checkout on `PYTHONPATH`.
Its core needs only numpy and pandas.
"""

import math

import numpy as np
import pandas as pd

try:
    import duet.duet_core as _core
    from duet.duet_vals import DuetWrapper, Release, release_sensitivity, unwrap
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "noisyvalue.dduo requires DDuo's 'duet' package with `details=True` "
        "mechanisms (duet.duet_vals.Release); install it from "
        "https://github.com/uvm-plaid/dduo-python or put a checkout on "
        "PYTHONPATH"
    ) from e

from .pandas import NoisyFloatArray
from .release import observe_release


def gauss(val, *, epsilon=None, delta=None, ε=None, δ=None,
          key=None, keys=None, registry=None, low=None, high=None):
    """`duet.gauss`, releasing `NoisyValue`s: (epsilon, delta)-DP Gaussian
    noise with `sigma = sqrt(2 ln(1.25/delta)) * sensitivity / epsilon`.

    `ε`/`δ` are accepted as aliases, as in DDuo.  Scalars yield a
    `NoisyFloat`; list/array/Series arguments yield a `NoisyFloatArray`
    (a pandas Series of them for a Series).  See the module docstring for
    `key`/`keys`/`registry`/`low`/`high`.
    """
    eps = _alias("epsilon", epsilon, ε)
    dlt = _alias("delta", delta, δ)
    _check(val, key, keys, registry)
    release = _core.gauss(val, epsilon=eps, delta=dlt, details=True)
    return _posterior(release, key, keys, registry, low, high)


def laplace(val, *, epsilon=None, ε=None,
            key=None, keys=None, registry=None, low=None, high=None):
    """`duet.laplace`, releasing `NoisyValue`s: epsilon-DP Laplace noise of
    scale `sensitivity / epsilon`, elementwise for vector arguments.

    Repeated Laplace releases of one key cannot be combined in closed form
    and are refused by the registry; see `SharedLatentRegistry`.
    """
    eps = _alias("epsilon", epsilon, ε)
    _check(val, key, keys, registry)
    release = _core.laplace(val, epsilon=eps, details=True)
    return _posterior(release, key, keys, registry, low, high)


def renyi_gauss(val, *, alpha=None, epsilon=None, α=None, ε=None,
                key=None, keys=None, registry=None, low=None, high=None):
    """`duet.renyi_gauss`, releasing `NoisyValue`s: (alpha, epsilon)-RDP
    Gaussian noise with `sigma = sqrt(sensitivity**2 * alpha / (2 * epsilon))`.

    Unlike `duet.renyi_gauss`, which returns an untracked argument exactly,
    an argument that is not a `DuetWrapper` is refused: it may be private
    data that was never wrapped.
    """
    return _renyi(_core.renyi_gauss, val, alpha, epsilon, α, ε,
                  key, keys, registry, low, high)


def renyi_gauss_vec(val, *, alpha=None, epsilon=None, α=None, ε=None,
                    key=None, keys=None, registry=None, low=None, high=None):
    """`duet.renyi_gauss_vec`, releasing `NoisyValue`s; see `renyi_gauss`."""
    return _renyi(_core.renyi_gauss_vec, val, alpha, epsilon, α, ε,
                  key, keys, registry, low, high)


# ── internals ────────────────────────────────────────────────────────────────

def _renyi(mechanism, val, alpha, epsilon, α, ε, key, keys, registry, low, high):
    al = _alias("alpha", alpha, α)
    eps = _alias("epsilon", epsilon, ε)
    _check(val, key, keys, registry)
    release = mechanism(val, alpha=al, epsilon=eps, details=True)
    return _posterior(release, key, keys, registry, low, high)


def _alias(name, value, greek):
    if value is not None and greek is not None:
        raise TypeError(f"Pass {name} once, not under both spellings")
    value = greek if value is None else value
    if value is None:
        raise TypeError(f"Missing required privacy parameter {name!r}")
    return float(value)


def _check(val, key, keys, registry):
    """Every refusal that can be decided before DDuo spends any budget."""
    if not isinstance(val, DuetWrapper):
        raise TypeError(
            f"Expected a DuetWrapper (a value whose sensitivity DDuo tracks), "
            f"got {type(val).__name__}")
    sens = float(release_sensitivity(val))
    if math.isinf(sens):
        raise ValueError(
            "Sensitivity is unbounded -- DDuo cannot bound how far one "
            "change to the data moves this value; clip it (or the data it "
            "is computed from) before releasing")
    if not sens > 0:
        raise ValueError(
            f"Sensitivity is {sens}: the value does not depend on any tracked "
            "data source, so there is nothing to protect by adding noise")

    raw = unwrap(val)
    vector = np.ndim(raw) > 0
    if vector and key is not None:
        raise ValueError("Vector-valued releases take keys=, not key=")
    if not vector and keys is not None:
        raise ValueError("Scalar releases take key=, not keys=")
    if (key is not None or keys is not None) and registry is None:
        raise ValueError("Sharing statistics by key requires a registry")
    if keys is not None and len(tuple(keys)) != np.size(raw):
        raise ValueError(
            f"Got {np.size(raw)} released elements for {len(tuple(keys))} keys")


def _narrow(public, caller, tighter):
    """DDuo's public bound, narrowed by the caller's; None if unbounded."""
    bound = public if caller is None else tighter(public, caller)
    return None if math.isinf(bound) else bound


def _posterior(release, key, keys, registry, low, high):
    assert isinstance(release, Release)
    lo = _narrow(float(release.bounds[0]), low, max)
    hi = _narrow(float(release.bounds[1]), high, min)

    def build(obs, k):
        params = dict(mechanism=release.mechanism, scale=release.scale,
                      discrete=False, low=lo, high=hi)
        if k is not None:
            return registry.observe(k, obs, **params)
        return observe_release(obs, **params)

    value = release.value
    if np.ndim(value) == 0:
        return build(float(value), key)

    noisy = np.asarray(value, dtype=float).ravel()
    ks = tuple(keys) if keys is not None else (None,) * len(noisy)
    array = NoisyFloatArray._from_sequence(
        [build(float(x), k) for x, k in zip(noisy, ks)])
    if isinstance(value, pd.Series):
        return pd.Series(array, index=value.index, name=value.name)
    return array
