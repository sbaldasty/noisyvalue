import itertools as ite
import numpy as np
import operator

from dataclasses import dataclass
from functools import reduce

from ..core import NoisyFloat
from ..core import NoisyValue
from ..sample import sample_noisy_values


@dataclass(frozen=True)
class Variable:
    name: str
    values: tuple[str]


@dataclass
class Slice:
    subpop: dict[Variable, tuple[str]]
    count: NoisyValue


@dataclass
class Release:
    variables: tuple[Variable]
    slices: tuple[Slice]


@dataclass
class Query:
    subpop: dict[Variable, tuple[str]]


def merge_slices(slices):
    merged_subpop = reduce(
        lambda acc, d: {k: acc.get(k, ()) + d.get(k, ()) for k in set(acc) | set(d)}, 
        (s.subpop for s in slices), 
        {})
    merged_count = sum(s.count for s in slices)
    return Slice(subpop=merged_subpop, count=merged_count)


def synthetic_release_disjoint(*, n_vals):
    """
    Generate a synthetic release over a single variable with a slice for each value.
    For such releases the genetic algorithm can easily identify the optimal genome,
    as each variable corresponds to a distinct subpopulation. It need not consider the noise
    associated with the slices.

    Args:
        n_vals (int): Number of values for the single variable.

    Returns:
        Release: A synthetic release object with disjoint subpopulations.
    """
    variable = Variable("X", tuple(f"value{i}" for i in range(n_vals)))
    slices = [Slice({variable: (f"value{i}",)}, NoisyFloat.gaussian(obs=1.0, loc=1.0, scale=1.0)) for i in range(n_vals)]
    return Release(variables=(variable,), slices=tuple(slices))


def synthetic_release_hierarchical(*, n_levels, n_children):
    """
    Generate a synthetic release over a single variable with a hierarchical structure.
    At the bottom level, each value corresponds to a distinct subpopulation, similar to the
    disjoint case. Higher levels aggregate these subpopulations, creating a
    hierarchy of slices.

    Args:
        n_levels (int): Number of hierarchical levels.
        n_children (int): Number of variables that combine to form a higher-level slice.

    Returns:
        Release: A synthetic release object with hierarchical subpopulations.
    """
    bottom = synthetic_release_disjoint(n_vals=n_children ** (n_levels - 1))
    all_slices = list(bottom.slices)
    lvl = list(bottom.slices)
    for _ in range(n_levels - 1):
        batches = ite.batched(lvl, n_children, strict=True)
        lvl = [merge_slices(s) for s in batches]
        all_slices.extend(lvl)
    return Release(variables=bottom.variables, slices=all_slices)


def synthetic_release_chaotic(*, n_vals, n_slices, n_members_min, n_members_max):
    """
    Generate a synthetic release with completely random subpopulations over a single variable.
    Each slice is composed of a random selection of base slices.

    Args:
        n_vals (int): Number of values for the single variable in the base release.
        n_slices (int): Number of slices to generate.
        n_members_min (int): Minimum number of base slices to combine for each new slice.
        n_members_max (int): Maximum number of base slices to combine for each new slice.

    Returns:
        Release: A synthetic release object with chaotic subpopulations.
    """
    base = synthetic_release_disjoint(n_vals=n_vals)
    slices = []
    for _ in range(n_slices):
        n_members = np.random.randint(n_members_min, n_members_max + 1)
        components = np.random.choice(base.slices, size=n_members, replace=False)
        slices.append(merge_slices(components))
    return Release(variables=tuple(base.variables), slices=tuple(slices))


def fitness_noise(release, genome):
    total = sum(release.slices[i].count for i, g in enumerate(genome) if g)
    samples = sample_noisy_values(total, n=100)[0]
    return np.var(samples.draws)
