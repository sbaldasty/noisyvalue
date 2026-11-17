"""Interpreting additive-noise releases as `NoisyValue`s, independent of
whichever DP library performed the release.

`observe_release` turns one released number plus its mechanism's public
metadata (family, scale, discreteness) into a `NoisyValue` whose posterior
under a flat prior is the noise law re-centred at the observation.
`SharedLatentRegistry` gives statistics stable identities across releases so
repeated measurements of one statistic combine in closed form.  Backends --
`noisyvalue.opendp`, `noisyvalue.dduo` -- perform the release and call into
this module; nothing here depends on any DP library.
"""

import math

import sympy as sp

from .core import NoisyFloat, NoisyInt
from .graph import DiscreteGaussianNode, DiscreteLaplaceNode
from .graph import GaussianNode, LaplaceNode


_NODE_CLASSES = {
    ("gaussian", True): DiscreteGaussianNode,
    ("gaussian", False): GaussianNode,
    ("laplace", True): DiscreteLaplaceNode,
    ("laplace", False): LaplaceNode,
}


def _release_variance(mechanism, discrete, scale):
    """Sampling variance of the mechanism's noise, in parameter space.

    For the Gaussian family this is scale**2 for the continuous and discrete
    laws alike -- what matters downstream is that products of Gaussian kernels
    combine exactly through these parameters, which holds on the lattice too.
    The discrete Laplace formula matches
    `dataset.solve.DiscreteLaplaceFamily.variance_from_scale`.
    """
    scale = float(scale)
    if mechanism == "gaussian":
        return scale ** 2
    if discrete:
        p = math.exp(-1.0 / scale)
        return 2.0 * p / (1.0 - p) ** 2
    return 2.0 * scale ** 2


# A support bound this far outside the posterior's centre lies beyond the
# noise's sampled/informative range entirely, so it is dropped as provably
# inert (same reasoning as `dataset.solve.LatticeFamily._drop_inert`).
_INERT_FLOOR = 60.0
_INERT_SCALES = {"gaussian": 8.0, "laplace": 32.0}


def _posterior_node(mechanism, discrete, center, scale, low=None, high=None):
    """A noise node whose law *is* the statistic's posterior.

    Under a flat prior a single additive-noise measurement leaves the noise
    law re-centred at the observation, so the posterior is encoded directly
    as a node rather than through a latent-plus-constraint pair (the same
    choice `dataset/solve.py` makes for release cells).  `low`/`high` are
    prior knowledge that the true value cannot fall outside them (e.g. zero
    as a lower bound for counts, or a released statistic's own declared
    sensitivity bounds) -- both `GaussianNode`/`LaplaceNode` and their
    discrete-lattice counterparts accept truncation the same way.
    """
    node_cls = _NODE_CLASSES[(mechanism, discrete)]
    center = float(center)
    scale = float(scale)
    slack = max(_INERT_FLOOR, _INERT_SCALES[mechanism] * scale)
    if low is not None and low < center - slack:
        low = None
    if high is not None and high > center + slack:
        high = None
    to_bound = (lambda v: sp.Integer(int(v))) if discrete else (lambda v: sp.Float(float(v)))
    params = {
        "loc": sp.Float(center),
        "scale": sp.Float(scale),
        "low": -sp.oo if low is None else to_bound(low),
        "high": sp.oo if high is None else to_bound(high),
    }
    return node_cls.create(**params)


def _issue(obs, node, discrete):
    cls = NoisyInt if discrete else NoisyFloat
    return cls(obs, node)


def observe_release(obs, *, mechanism, scale, discrete=None, low=None, high=None):
    """Interpret one already-released number as a `NoisyValue`.

    `mechanism` is `"gaussian"` or `"laplace"`; `discrete` selects the
    integer-lattice law and defaults to whether `obs` is an integer;
    `low`/`high` are prior knowledge that the true value cannot fall outside
    them, e.g. zero as a lower bound for counts, or declared sensitivity
    bounds on a released statistic.  The value is unshared -- use a
    `SharedLatentRegistry` to give repeated measurements of one statistic a
    common identity.
    """
    if mechanism not in ("gaussian", "laplace"):
        raise ValueError(f"Unknown mechanism: {mechanism!r}")
    if discrete is None:
        discrete = isinstance(obs, int)
    node = _posterior_node(mechanism, discrete, obs, scale, low=low, high=high)
    return _issue(obs, node, discrete)


class _Statistic:
    __slots__ = ("mechanism", "discrete", "center", "variance", "node")

    def __init__(self, mechanism, discrete, center, variance, node):
        self.mechanism = mechanism
        self.discrete = discrete
        self.center = center
        self.variance = variance
        self.node = node


class SharedLatentRegistry:
    """Stable identities for released statistics, so repeated measurements
    of one statistic share a posterior.

    Every value issued for a key is backed by the *same* node object, so all
    of them are one random variable to the joint sampler, and a later
    measurement of the key tightens every previously issued value in place:
    a registry's values are live views of each statistic's current posterior,
    not snapshots of the release that produced them.

    Combination is closed-form and Gaussian-family only: the product of two
    Gaussian kernels (continuous or on the integer lattice) is again one, at
    the precision-weighted centre.  Laplace posteriors do not stay in family
    under a second measurement, so a repeated Laplace key is refused -- the
    same boundary `dataset/solve.py` draws for its pair closed form.
    """

    def __init__(self):
        self._stats = {}
        self._folded_sums = set()

    def observe(self, key, obs, *, mechanism, scale, discrete=None, low=None, high=None):
        """Record a release of the statistic `key` and return its value.

        `low`/`high` bound the support (see `observe_release`); they are
        fixed by the key's first observation and ignored afterwards.
        """
        if mechanism not in ("gaussian", "laplace"):
            raise ValueError(f"Unknown mechanism: {mechanism!r}")
        if discrete is None:
            discrete = isinstance(obs, int)
        stat = self._stats.get(key)
        if stat is None:
            node = _posterior_node(mechanism, discrete, obs, scale, low=low, high=high)
            self._stats[key] = _Statistic(
                mechanism, discrete, float(obs),
                _release_variance(mechanism, discrete, scale), node)
            return _issue(obs, node, discrete)
        if (stat.mechanism, stat.discrete) != (mechanism, discrete):
            raise ValueError(
                f"Key {key!r} was first observed as "
                f"{'discrete ' if stat.discrete else ''}{stat.mechanism}; "
                f"a {'discrete ' if discrete else ''}{mechanism} measurement "
                "cannot be the same statistic")
        self._combine(stat, float(obs),
                      _release_variance(mechanism, discrete, scale))
        return _issue(obs, stat.node, discrete)

    def add_estimate(self, key, center, variance):
        """Fold an independent unbiased estimate into `key`'s posterior.

        The estimate must be independent of every measurement already folded
        into the key.  The estimate enters as numbers, not nodes, so any
        correlation between the source of the estimate and this key's value
        is not represented in the graph -- joint samples of the two remain
        independent even though the marginals are right.
        """
        self._combine(self._require(key), float(center), float(variance))

    def add_sum_estimate(self, total_key, part_keys):
        """Fold the sum of some parts' posteriors into a total's posterior.

        The parts' current centres and variances are summed and folded into
        `total_key` as one independent estimate -- correct when the parts
        were measured independently of the total's own measurements.  For
        discrete parts the sum of lattice laws is treated as Gaussian with
        the summed variance, a close approximation.  As with `add_estimate`,
        the induced correlation between total and parts is not represented.
        """
        fold = (total_key, tuple(part_keys))
        if fold in self._folded_sums:
            raise ValueError(
                f"The sum of {list(part_keys)} was already folded into "
                f"{total_key!r}; folding the same evidence twice would "
                "double-count it")
        parts = [self._require(k) for k in part_keys]
        self.add_estimate(
            total_key,
            sum(p.center for p in parts),
            sum(p.variance for p in parts))
        self._folded_sums.add(fold)

    def posterior(self, key):
        """(center, variance) of the statistic's current posterior."""
        stat = self._require(key)
        return stat.center, stat.variance

    def current(self, key):
        """A fresh value viewing `key`'s current posterior.

        Its observation is the posterior centre (rounded for discrete
        statistics), which is generally *not* any single released number.
        """
        stat = self._require(key)
        obs = round(stat.center) if stat.discrete else stat.center
        return _issue(obs, stat.node, stat.discrete)

    def _require(self, key):
        stat = self._stats.get(key)
        if stat is None:
            raise KeyError(f"No statistic observed under key {key!r}")
        return stat

    @staticmethod
    def _combine(stat, center, variance):
        if stat.mechanism != "gaussian":
            raise NotImplementedError(
                "Combining repeated measurements is closed-form only for the "
                "Gaussian family; a second Laplace measurement of one "
                "statistic has a posterior outside the library's noise "
                "families. Use distinct keys, or a Gaussian mechanism.")
        if variance <= 0:
            raise ValueError(f"Estimate variance must be positive: {variance}")
        precision = 1.0 / stat.variance + 1.0 / variance
        combined = (stat.center / stat.variance + center / variance) / precision
        stat.center = combined
        stat.variance = 1.0 / precision
        # Node params are read at sample time, so updating them in place
        # retunes every value already issued for this statistic.
        stat.node.params[0] = sp.Float(combined)
        stat.node.params[1] = sp.Float(math.sqrt(stat.variance))
