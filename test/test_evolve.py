import pytest

from noisyvalue.core import NoisyFloat
from noisyvalue.dataset.evolve import Count
from noisyvalue.dataset.evolve import Variable
from noisyvalue.dataset.evolve import fitness_noise

# FIXME This is not deterministic!
def test_noise_fitness():
    v = Variable("color", set("red", "green", "blue"))
    release = {
        Count({ v: set("red")}, NoisyFloat.gaussian(obs=12.1, loc=12.1, scale=1.0)),
        Count({ v: set("green")}, NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Count({ v: set("green")}, NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Count({ v: set("green")}, NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Count({ v: set("blue")}, NoisyFloat.gaussian(obs=100.1, loc=100.1, scale=6.0)),
    }
    f1 = fitness_noise(release, [1, 0, 0, 0, 0])
    f2 = fitness_noise(release, [0, 1, 0, 0, 0])
    f3 = fitness_noise(release, [0, 0, 0, 0, 1])
    f4 = fitness_noise(release, [0, 1, 1, 1, 0])
    assert f1 < f2
    assert f2 < f3
    assert f3 < f4
