"""Choosing released counts to answer a query, as linear algebra over base cells.

Every released count covers a set of *base cells* -- the finest cells of the
histogram the release measures.  Writing each count's support as a column of
an incidence matrix `A` (base cells x counts), a query as a vector `q` over
the same base cells, and a selection of counts as a genome `g`, the two
objectives the evolutionary search trades off are one matrix product each:

- **coverage**: `sum_a w_a * |(A g)_a - q_a|`, the number of base cells the
  selection gets wrong.  A missed query cell costs the same as an extra cell,
  and a cell counted twice costs one -- the same metric as
  `evolve.fitness_coverage`, which this reproduces exactly on its releases.
- **noise**: `sum_s v_s * g_s**2`, the variance of the summed counts, assuming
  the counts' noise is independent (true of the Census NMF measurements and of
  `evolve`'s synthetic releases).  Closed form, so no sampling and no jitter.

Both work unchanged for a *signed* genome (entries in {-1, 0, 1}), where a
count can also be subtracted: "18 and over" is "total" minus "under 18".

Base cells can be numerous (the DHC person histogram has 1,227,744), so a
problem built from a catalog does not use them directly.  On each axis it
groups base levels that no candidate count and not the query can tell apart,
and an atom of the problem is one such group per axis.  `w_a` is the number
of base cells an atom stands for, so coverage is still counted in base cells.

A problem plugs into `evolve`'s algorithms as the `release` argument, with
`coverage_objective` / `noise_objective` as the objectives: those algorithms
only pass `release` through to the objective functions.
"""

import itertools as ite
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sps
import sympy as sp

from .. import util
from ..graph import GaussianNode
from ..sample import sample_noisy_values


class CoverProblem:
    """Candidate counts, the base cells each covers, and the query to answer.

    Args:
        incidence: atoms x slices matrix; entry (a, s) is how many times
            slice s counts atom a (almost always 0 or 1).
        query: per-atom multiplicity the query asks for (0 or 1).
        variances: per-slice noise variance.
        weights: base cells each atom stands for (default 1 each).
        slice_labels / atom_labels: descriptions, for reporting.
    """

    def __init__(self, incidence, query, variances, *, weights=None,
                 slice_labels=None, atom_labels=None):
        self.incidence = sps.csr_array(incidence, dtype=np.int64)
        n_atoms, n_slices = self.incidence.shape
        self.query = np.asarray(query, dtype=np.int64)
        self.variances = np.asarray(variances, dtype=np.float64)
        self.weights = (np.ones(n_atoms, dtype=np.int64) if weights is None
                        else np.asarray(weights, dtype=np.int64))
        if self.query.shape != (n_atoms,) or self.weights.shape != (n_atoms,):
            raise ValueError("query and weights need one entry per atom")
        if self.variances.shape != (n_slices,):
            raise ValueError("variances need one entry per slice")
        self.slice_labels = (tuple(range(n_slices)) if slice_labels is None
                             else tuple(slice_labels))
        self.atom_labels = (tuple(range(n_atoms)) if atom_labels is None
                            else tuple(atom_labels))

    def __repr__(self):
        return (f"<CoverProblem: {self.n_slices:,} candidate counts over "
                f"{self.n_atoms:,} atoms ({int(self.weights.sum()):,} base cells); "
                f"query is {int(self.weights @ self.query):,} base cells>")

    @property
    def n_atoms(self):
        return self.incidence.shape[0]

    @property
    def n_slices(self):
        return self.incidence.shape[1]

    genome_length = n_slices

    # ── objectives ──────────────────────────────────────────────────────────
    # Each accepts one genome (shape [n_slices]) or a population of them
    # (shape [pop, n_slices]) and returns a scalar or a [pop] array.

    def counts(self, genomes):
        """How many times the selection counts each atom."""
        g = np.asarray(genomes, dtype=np.int64)
        return (self.incidence @ g.T).T

    def coverage(self, genomes):
        """Base cells the selection gets wrong: missed, extra, or double-counted."""
        return self.missing(genomes) + self.extra(genomes)

    def missing(self, genomes):
        """Base cells the query asks for that the selection does not count."""
        return np.clip(self.query - self.counts(genomes), 0, None) @ self.weights

    def extra(self, genomes):
        """Base cells the selection counts beyond what the query asks for."""
        return np.clip(self.counts(genomes) - self.query, 0, None) @ self.weights

    def noise(self, genomes):
        """Variance of the summed (or differenced) counts."""
        g = np.asarray(genomes, dtype=np.float64)
        return g ** 2 @ self.variances

    # ── reporting ───────────────────────────────────────────────────────────

    def describe(self, genome):
        """The counts a genome selects, with sign, and its objective values."""
        genome = np.asarray(genome)
        chosen = np.flatnonzero(genome)
        return {
            "selected": [(self.slice_labels[s], int(genome[s])) for s in chosen],
            "coverage": int(self.coverage(genome)),
            "missing": int(self.missing(genome)),
            "extra": int(self.extra(genome)),
            "noise": float(self.noise(genome)),
        }


# ── objectives in `evolve`'s (release, query, genome) signature ─────────────

def coverage_objective(problem, query, genome):
    """`evolve`-compatible coverage objective; the query lives in the problem."""
    return int(problem.coverage(genome))


def noise_objective(problem, query, genome):
    """`evolve`-compatible noise objective; the query lives in the problem."""
    return float(problem.noise(genome))


# ── from `evolve`'s synthetic releases ──────────────────────────────────────

def from_release(release, query, *, variances=None, rng=None, n_draws=10_000):
    """A cover problem for an `evolve.Release` and `evolve.Query`.

    Atoms are the release's `(Variable, value)` pieces.  Each slice's variance
    is read exactly when its count is an untruncated Gaussian with a constant
    scale (every slice `evolve` builds), and estimated once from `n_draws`
    samples otherwise.  Pass `variances` to override.
    """
    pieces = ite.chain(query.subpop, *(s.subpop for s in release.slices))
    index = {piece: a for a, piece in enumerate(dict.fromkeys(pieces))}

    rows, cols = [], []
    for s, slice_ in enumerate(release.slices):
        for piece in slice_.subpop:
            rows.append(index[piece])
            cols.append(s)
    # duplicate (row, col) pairs sum, so a slice naming a piece twice counts it twice
    incidence = sps.coo_array(
        (np.ones(len(rows), dtype=np.int64), (rows, cols)),
        shape=(len(index), len(release.slices)))

    wanted = Counter(query.subpop)
    q = np.array([wanted[piece] for piece in index], dtype=np.int64)

    if variances is None:
        rng = util.generator(rng)
        variances = [_variance(s.count, rng, n_draws) for s in release.slices]

    return CoverProblem(
        incidence, q, variances,
        slice_labels=[tuple(v for _, v in s.subpop) for s in release.slices],
        atom_labels=[f"{var.name}={value}" for var, value in index])


def _variance(value, rng, n_draws):
    node = value._root
    if (isinstance(node, GaussianNode) and not node.scale.free_symbols
            and node.low == -sp.oo and node.high == sp.oo):
        return float(node.scale) ** 2
    draws = sample_noisy_values(value, n=n_draws, rng=rng)[0].draws
    return float(np.var(draws))


# ── from a dataset catalog ──────────────────────────────────────────────────

def from_product(product, query, *, measurements=None, variances=1.0,
                 prune=True):
    """A cover problem over a `dataset.Product`'s measurements.  Catalog only.

    Args:
        product: e.g. `census.dhc_2020`.
        query: `{axis: base levels}` -- the query is every base cell whose
            level on each listed axis is in its set; unlisted axes are summed
            over.  Levels are labels from `product.axis(name).levels`; see
            `levels_for` to state them in a measurement's own binning.
        measurements: names of the measurements to draw counts from
            (default: all of them).
        variances: one variance for every count, or `{measurement: variance}`.
            The catalog carries none; read them from a view, e.g.
            `view.query(name)["variance"].iloc[0]`.  They are per geography,
            so a problem describes one area.
        prune: keep only counts that overlap the query.  A count that does
            not overlap only adds extra cells, so this never removes a useful
            candidate -- unless signed genomes are wanted, where a disjoint
            count can still help cancel.  Set False for those.
    """
    universe = product.universe
    names = (list(product.measurements()["measurement"]) if measurements is None
             else list(measurements))
    ms = [product.measurement(n) for n in names]
    if not isinstance(variances, dict):
        variances = {m.name: float(variances) for m in ms}

    wanted = []
    for axis in universe.axes:
        levels = query.get(axis.name)
        if levels is None:
            wanted.append(frozenset(range(axis.size)))
        else:
            wanted.append(frozenset(_level_index(axis, level) for level in levels))
    unknown = set(query) - set(universe.axis_names)
    if unknown:
        raise KeyError(f"no axes {sorted(unknown)} in {universe.name!r}")

    # Candidate positions per measurement and axis.
    positions = []
    for m in ms:
        per_axis = []
        for a, binning in enumerate(m.binnings):
            part = binning.partition
            per_axis.append(part.touching(wanted[a]) if prune
                            else tuple(range(len(part))))
        positions.append(per_axis)

    # Per axis, group base levels no candidate and not the query distinguish.
    classes = []        # per axis: list of tuples of base-level indices
    class_of = []       # per axis: {base level: class index}
    for a, axis in enumerate(universe.axes):
        signatures = {}
        for level in range(axis.size):
            sig = [level in wanted[a]]
            for m, per_axis in zip(ms, positions):
                block = m.binnings[a].partition.block_of(level)
                sig.append(block if block in per_axis[a] else None)
            if not sig[0] and all(b is None for b in sig[1:]):
                continue        # no candidate and not the query: irrelevant
            signatures.setdefault(tuple(sig), []).append(level)
        groups = list(signatures.values())
        classes.append(groups)
        class_of.append({lvl: c for c, group in enumerate(groups) for lvl in group})
    shape = tuple(len(groups) for groups in classes)

    # One column per candidate cell: the atoms it covers.
    rows, cols, labels, slice_vars = [], [], [], []
    for m, per_axis in zip(ms, positions):
        covered = []
        for a, binning in enumerate(m.binnings):
            covered.append({
                p: np.array(sorted({class_of[a][lvl]
                                    for lvl in binning.partition.groups[p]}),
                            dtype=np.int64)
                for p in per_axis[a]})
        for cell in ite.product(*per_axis):
            atoms = np.ravel_multi_index(
                np.ix_(*(covered[a][p] for a, p in enumerate(cell))), shape).ravel()
            rows.append(atoms)
            cols.append(np.full(atoms.size, len(labels), dtype=np.int64))
            labels.append((m.name, tuple(
                binning.labels[p] for binning, p in zip(m.binnings, cell))))
            slice_vars.append(variances[m.name])

    n_atoms = int(np.prod(shape))
    rows = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    incidence = sps.csr_array(
        (np.ones(rows.size, dtype=np.int64), (rows, cols)),
        shape=(n_atoms, len(labels)))

    weights = np.ones(1, dtype=np.int64)
    in_query = np.ones(1, dtype=bool)
    for a, groups in enumerate(classes):
        weights = np.multiply.outer(weights, [len(g) for g in groups])
        in_query = np.logical_and.outer(
            in_query, [g[0] in wanted[a] for g in groups])
    weights, in_query = weights.ravel(), in_query.ravel()

    # Drop atoms that only exist as a product of relevant per-axis classes:
    # covered by no candidate and outside the query, they cost nothing.
    used = (np.diff(incidence.indptr) > 0) | in_query
    keep = np.flatnonzero(used)
    atom_labels = [
        tuple(tuple(universe.axes[a].levels[lvl] for lvl in classes[a][c])
              for a, c in enumerate(np.unravel_index(k, shape)))
        for k in keep] if keep.size <= 10_000 else None

    return CoverProblem(
        incidence[keep], in_query[keep].astype(np.int64), slice_vars,
        weights=weights[keep], slice_labels=labels, atom_labels=atom_labels)


def levels_for(product, measurement, filters):
    """Translate `{axis: labels of measurement's binning}` into base levels.

    `census.dhc_table_variables()` states each published cell this way --
    `{"sex": ("male",), "age": ("15-17",)}` against `sex*age_38_groups` -- and
    this turns it into a `from_product` query.
    """
    m = product.measurement(measurement)
    out = {}
    for axis, binning in zip(product.universe.axes, m.binnings):
        if axis.name not in filters:
            continue
        position = {label: p for p, label in enumerate(binning.labels)}
        out[axis.name] = tuple(
            axis.levels[lvl]
            for label in filters[axis.name]
            for lvl in binning.partition.groups[position[label]])
    return out


def selection_frame(product, problem, genome):
    """The selected counts of a `from_product` problem as a frame.

    Columns are `measurement`, one per axis (in the labels a `View` frame
    uses), and `sign`, so it merges against a view frame on `measurement` plus
    the axis columns to pull the selected cells' noisy values for summing.
    """
    axes = product.universe.axis_names
    rows = []
    for (name, labels), sign in problem.describe(genome)["selected"]:
        rows.append({"measurement": name, **dict(zip(axes, labels)), "sign": sign})
    return pd.DataFrame(rows, columns=["measurement", *axes, "sign"])


def _level_index(axis, level):
    if isinstance(level, (int, np.integer)):
        return int(level)
    try:
        return axis.levels.index(level)
    except ValueError:
        raise KeyError(f"axis {axis.name!r} has no level {level!r}") from None
