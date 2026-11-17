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
        count = dduo.laplace(df.shape[0], epsilon=1.0, low=0)
        mean = total / count     # a NoisyFloat; its posterior propagates

Each mechanism reproduces DDuo's own calibration, noise draw and privacy
accounting exactly -- the active odometers and filters in
`duet.duet_core.accountant` are charged just as the corresponding `duet`
function would charge them -- and differs only in what it returns.  DDuo
itself is not modified, and only its public sensitivity metadata
(`val.senv.get_max()`) is read.  Consequently the guarantees are DDuo's,
caveats included: noise comes from NumPy's floating-point samplers rather
than a vetted DP sampler, the classic Gaussian calibration
`sigma = sqrt(2 ln(1.25/delta)) * sensitivity / epsilon` is only valid for
`epsilon < 1`, and DDuo does not check that the tracked distance metric
(L1/L2) matches the mechanism.

Because DDuo noises with continuous laws, every release is a `NoisyFloat`
(or a `NoisyFloatArray` for vector-valued arguments), even for counts.
Pass `low=0` for counts to truncate the posterior at zero.  `key`/`keys`
with a `SharedLatentRegistry` give repeated releases of one statistic a
common identity, exactly as in `noisyvalue.opendp`; DDuo tracks
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

Requires the `duet` package, which is not on PyPI; install DDuo from its
repository or put a checkout on `PYTHONPATH`.  Its core needs only numpy
and pandas.
"""

import math

import numpy as np
import pandas as pd

try:
    import duet.duet_core as _core
    from duet.duet_vals import DuetWrapper, unwrap
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "noisyvalue.dduo requires DDuo's 'duet' package; install it from "
        "https://github.com/uvm-plaid/dduo-python or put a checkout on "
        "PYTHONPATH"
    ) from e

from .pandas import NoisyFloatArray
from .release import observe_release


def gauss(val, *, epsilon=None, delta=None, ε=None, δ=None,
          key=None, keys=None, registry=None, low=None, high=None, rng=None):
    """`duet.gauss`, releasing `NoisyValue`s: (epsilon, delta)-DP Gaussian
    noise with `sigma = sqrt(2 ln(1.25/delta)) * sensitivity / epsilon`.

    `ε`/`δ` are accepted as aliases, as in DDuo.  Scalars yield a
    `NoisyFloat`; list/array/Series arguments yield a `NoisyFloatArray`
    (a pandas Series of them for a Series).  See the module docstring for
    `key`/`keys`/`registry`/`low`/`high`; `rng` is a NumPy `Generator`,
    defaulting to the global `np.random` state DDuo itself draws from.
    """
    eps = _alias("epsilon", epsilon, ε)
    dlt = _alias("delta", delta, δ)
    sens = _sensitivity(val)
    sigma = math.sqrt(2 * math.log(1.25 / dlt)) * sens / eps
    _charge_ed(val, eps, dlt)
    return _release(val, "gaussian", sigma, key, keys, registry, low, high, rng)


def laplace(val, *, epsilon=None, ε=None,
            key=None, keys=None, registry=None, low=None, high=None, rng=None):
    """`duet.laplace`, releasing `NoisyValue`s: epsilon-DP Laplace noise of
    scale `sensitivity / epsilon`, elementwise for vector arguments.

    Repeated Laplace releases of one key cannot be combined in closed form
    and are refused by the registry; see `SharedLatentRegistry`.
    """
    eps = _alias("epsilon", epsilon, ε)
    scale = _sensitivity(val) / eps
    _charge_ed(val, eps, 0)
    return _release(val, "laplace", scale, key, keys, registry, low, high, rng)


def renyi_gauss(val, *, alpha=None, epsilon=None, α=None, ε=None,
                key=None, keys=None, registry=None, low=None, high=None, rng=None):
    """`duet.renyi_gauss` / `duet.renyi_gauss_vec`, releasing `NoisyValue`s:
    (alpha, epsilon)-RDP Gaussian noise with
    `sigma = sqrt(sensitivity**2 * alpha / (2 * epsilon))`.

    Unlike `duet.renyi_gauss`, which silently returns an untracked argument
    unchanged, an argument that is not a `DuetWrapper` is refused: it may be
    private data that was never wrapped.
    """
    al = _alias("alpha", alpha, α)
    eps = _alias("epsilon", epsilon, ε)
    sens = _sensitivity(val)
    sigma = math.sqrt(sens ** 2 * al / (2 * eps))
    for f in _core.accountant.fs:
        f.pay(al, eps)
    for o in _core.accountant.ods:
        o.add(val.senv.renyi_shrug(al, eps))
    return _release(val, "gaussian", sigma, key, keys, registry, low, high, rng)


renyi_gauss_vec = renyi_gauss


# ── internals ────────────────────────────────────────────────────────────────

def _alias(name, value, greek):
    if value is not None and greek is not None:
        raise TypeError(f"Pass {name} once, not under both spellings")
    value = greek if value is None else value
    if value is None:
        raise TypeError(f"Missing required privacy parameter {name!r}")
    return float(value)


def _sensitivity(val):
    if not isinstance(val, DuetWrapper):
        raise TypeError(
            f"Expected a DuetWrapper (a value whose sensitivity DDuo tracks), "
            f"got {type(val).__name__}")
    sens = float(val.senv.get_max())
    if math.isinf(sens):
        raise ValueError(
            "Sensitivity is unbounded -- DDuo assigns infinite sensitivity to "
            "e.g. comparisons and products of two sensitive values; clip "
            "before aggregating")
    if not sens > 0:
        raise ValueError(
            f"Sensitivity is {sens}: the value does not depend on any tracked "
            "data source, so there is nothing to protect by adding noise")
    return sens


def _charge_ed(val, eps, dlt):
    """Charge (eps, dlt) exactly as `duet.gauss`/`duet.laplace` do."""
    acct = _core.accountant
    for f in acct.fs:
        f.pay(eps, dlt)
    for f in acct.advfs:
        f.pay(eps, dlt)
    for o in acct.ods:
        o.add(val.senv.shrug(eps, dlt))
    for o in acct.advods:
        o.add(val.senv.shrug(eps, dlt))


def _draw(rng, mechanism, scale, size):
    src = np.random if rng is None else rng
    if mechanism == "gaussian":
        return src.normal(loc=0.0, scale=scale, size=size)
    return src.laplace(loc=0.0, scale=scale, size=size)


def _release(val, mechanism, scale, key, keys, registry, low, high, rng):
    raw = unwrap(val)
    vector = hasattr(raw, "__len__") and not isinstance(raw, (str, bytes))
    if vector and key is not None:
        raise ValueError("Vector-valued releases take keys=, not key=")
    if not vector and keys is not None:
        raise ValueError("Scalar releases take key=, not keys=")
    if (key is not None or keys is not None) and registry is None:
        raise ValueError("Sharing statistics by key requires a registry")

    def build(obs, k):
        if k is not None:
            return registry.observe(k, obs, mechanism=mechanism, scale=scale,
                                    discrete=False, low=low, high=high)
        return observe_release(obs, mechanism=mechanism, scale=scale,
                               discrete=False, low=low, high=high)

    if not vector:
        return build(float(raw) + float(_draw(rng, mechanism, scale, None)), key)

    exact = np.asarray(raw, dtype=float).ravel()
    ks = tuple(keys) if keys is not None else (None,) * len(exact)
    if len(ks) != len(exact):
        raise ValueError(f"Got {len(exact)} released elements for {len(ks)} keys")
    noisy = exact + _draw(rng, mechanism, scale, len(exact))
    array = NoisyFloatArray._from_sequence(
        [build(float(x), k) for x, k in zip(noisy, ks)])
    if isinstance(raw, pd.Series):
        return pd.Series(array, index=raw.index, name=raw.name)
    return array
