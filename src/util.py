import numpy as np

from itertools import count
from numpy.random import Generator

_counter = count()

def fresh_name():
    return f"id_{next(_counter)}"

def reset_name_provider():
    global _counter
    _counter = count()

def generator(rng):
    return rng if isinstance(rng, Generator) else np.random.default_rng(rng)
