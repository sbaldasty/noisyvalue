import os

import numpy as np
import pytest

from noisyvalue import census
from noisyvalue.census import dhc_2020, dhc_table_variables
from noisyvalue.core import NoisyFloat
from noisyvalue.dataset import cover
from noisyvalue.dataset.evolve import (
    Query, Release, Slice, Variable, fitness_coverage_old,
    pareto_evolutionary_algorithm, query_qty, synthetic_release_chaotic,
    synthetic_release_hierarchical)


# ── synthetic releases ──────────────────────────────────────────────────────

def test_coverage_matches_evolve_fitness_coverage():
    rng = np.random.default_rng(1)
    release = synthetic_release_chaotic(
        n_vals=40, n_slices=25, n_members_min=1, n_members_max=8, rng=rng)
    query = query_qty(release, qty=10, rng=rng)
    problem = cover.from_release(release, query)

    genomes = rng.integers(0, 2, size=(200, problem.n_slices))
    expected = [fitness_coverage_old(release, query, list(g)) for g in genomes]
    assert list(problem.coverage(genomes)) == expected


def test_noise_is_exact_sum_of_variances():
    v = Variable("color", ("red", "green", "blue"))
    slices = (
        Slice(((v, "red"),), NoisyFloat.gaussian(obs=1.0, loc=1.0, scale=1.0)),
        Slice(((v, "green"),), NoisyFloat.gaussian(obs=1.0, loc=1.0, scale=3.0)),
        Slice(((v, "blue"),), NoisyFloat.gaussian(obs=1.0, loc=1.0, scale=6.0)),
    )
    problem = cover.from_release(Release((v,), slices), Query(((v, "red"),)))
    assert problem.noise([1, 0, 0]) == 1.0
    assert problem.noise([0, 1, 1]) == 45.0
    assert list(problem.noise([[1, 1, 1], [0, 0, 0]])) == [46.0, 0.0]


def test_signed_genome_subtracts():
    # two levels, three children: parent minus one child covers the other two
    release = synthetic_release_hierarchical(n_levels=2, n_children=3)
    first, second, third, parent = release.slices
    query = Query(second.subpop + third.subpop)
    problem = cover.from_release(release, query)
    assert problem.coverage([0, 1, 1, 0]) == 0
    assert problem.coverage([-1, 0, 0, 1]) == 0
    assert problem.coverage([0, 0, 0, 1]) == 1          # one extra cell
    assert problem.noise([-1, 0, 0, 1]) == problem.noise([0, 1, 1, 0])


def test_plugs_into_pareto_evolutionary_algorithm():
    rng = np.random.default_rng(0)
    release = synthetic_release_hierarchical(n_levels=3, n_children=2)
    query = Query(release.slices[0].subpop + release.slices[1].subpop)
    problem = cover.from_release(release, query)
    fronts, _, _ = pareto_evolutionary_algorithm(
        release=problem, query=None, genome_length=problem.genome_length,
        total_generations=30, num_parents=20, num_children=20,
        tournament_size=2, archive_size=20, rng=rng,
        objectives={"noise": cover.noise_objective,
                    "coverage": cover.coverage_objective})
    best = min(fronts[-1], key=lambda i: i.fitnesses["coverage"])
    # the two leaves, or their parent alone (cheaper: one count, not two)
    assert best.fitnesses["coverage"] == 0
    assert best.fitnesses["noise"] == 1.0


# ── the DHC catalog (no data needed) ────────────────────────────────────────

NOT_DETAILED = [m for m in dhc_2020.measurements()["measurement"]
                if m != "detailed"]


def _crosswalk_genome(problem, entry):
    """The genome selecting exactly the cells `dhc_table_variables` sums."""
    axes = dhc_2020.universe.axis_names
    filters = {k: v for k, v in entry.items() if k not in ("query", "label")}
    genome = np.zeros(problem.n_slices, dtype=np.int64)
    for s, (name, labels) in enumerate(problem.slice_labels):
        if name == dhc_2020.measurement(entry["query"]).name and all(
                labels[axes.index(axis)] in levels
                for axis, levels in filters.items()):
            genome[s] = 1
    return genome


@pytest.mark.parametrize("code", [
    "P12_001N", "P12_002N", "P12_004N", "P12_007N", "P12_025N", "P12_049N"])
def test_p12_crosswalk_covers_exactly(code):
    entry = dhc_table_variables()[code]
    filters = {k: v for k, v in entry.items() if k not in ("query", "label")}
    query = cover.levels_for(dhc_2020, entry["query"], filters)
    problem = cover.from_product(dhc_2020, query, measurements=NOT_DETAILED)

    genome = _crosswalk_genome(problem, entry)
    assert genome.sum() > 0
    assert problem.coverage(genome) == 0


def test_dhc_coverage_counts_base_cells():
    query = {"sex": ("male",), "age": ("5", "6", "7", "8", "9")}
    problem = cover.from_product(dhc_2020, query)
    other_dims = 42 * 2 * 63                           # relgq x hispanic x cenrace
    assert problem.weights @ problem.query == 5 * other_dims
    assert problem.weights.sum() == dhc_2020.universe.cells

    male_total = np.array([
        name == "sex*hispanic" and labels[1] == "male"
        for name, labels in problem.slice_labels], dtype=np.int64)
    assert problem.missing(male_total) == 0
    assert problem.extra(male_total) == (116 - 5) * other_dims


def test_prune_keeps_only_overlapping_counts():
    query = {"sex": ("male",)}
    pruned = cover.from_product(dhc_2020, query, measurements=["sex*hispanic"])
    full = cover.from_product(
        dhc_2020, query, measurements=["sex*hispanic"], prune=False)
    assert pruned.n_slices == 2 and full.n_slices == 4


# ── against the real release ───────────────────────────────────────────────

requires_dhc_vt = pytest.mark.skipif(
    not os.path.isdir(os.path.join(
        census.DEFAULT_DHC_ROOT, "US_DHCP_PROD", "County.parquet", "DPQuery",
        "State=050")),
    reason='DHC Vermont counties not present; run fetch_dhc("county", state="VT")')


@requires_dhc_vt
def test_selection_pulls_values_from_a_view():
    view = dhc_2020.view("county", state="50", county="007")
    frame = view.query("sex*age_38_groups")
    variance = float(frame["variance"].iloc[0])

    query = {"sex": ("male",), "age": ("5", "6", "7", "8", "9")}
    problem = cover.from_product(
        dhc_2020, query, measurements=["sex*age_38_groups", "sex*hispanic"],
        variances={"sex*age_38_groups": variance, "sex*hispanic": variance})
    genome = np.array([
        name == "sex*age_38_groups" for name, _ in problem.slice_labels],
        dtype=np.int64)
    assert problem.coverage(genome) == 0
    assert problem.noise(genome) == pytest.approx(5 * variance)

    selected = cover.selection_frame(dhc_2020, problem, genome)
    axes = list(dhc_2020.universe.axis_names)
    merged = frame.merge(selected, on=["measurement", *axes])
    assert len(merged) == 5
    total = sum(value * sign for value, sign in zip(merged["value"], merged["sign"]))
    assert float(total) == merged["value"].astype(float).sum()
