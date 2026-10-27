# Some vocabulary...
#
# Value: str
# Variable: (name: str, set[Value])
## Subpopulation: dict[Variable, set[Value]]
# Count: tuple(Subpopulation, NoisyValue)
# Query: dict[Variable, set[Value]]  --Conjunctive query, working definition
# Release: list[Count]
# Genome: list[int]  --same length as release

import numpy as np

from dataclasses import dataclass

from ..core import NoisyValue
from ..sample import sample_noisy_values

@dataclass(frozen=True)
class Variable:
    name: str
    values: frozenset[str]

@dataclass
class Count:
    subpopulation: dict[Variable, frozenset[str]]
    count: NoisyValue

class Problem:
    pass

def fitness_noise(release, genome):
    total = sum(release[i].count for i, g in enumerate(genome) if g)
    samples = sample_noisy_values(total, n=100)[0]
    return np.var(samples.draws)
