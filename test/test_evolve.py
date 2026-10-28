from frozendict import frozendict
from noisyvalue.core import NoisyFloat
from noisyvalue.dataset.evolve import Release
from noisyvalue.dataset.evolve import Slice
from noisyvalue.dataset.evolve import Variable
from noisyvalue.dataset.evolve import fitness_noise
from noisyvalue.dataset.evolve import synthetic_release_disjoint
from noisyvalue.dataset.evolve import synthetic_release_hierarchical
from noisyvalue.dataset.evolve import synthetic_release_chaotic

# FIXME This is not deterministic!
def test_noise_fitness():
    v = Variable("color", ("red", "green", "blue"))
    slices = [
        Slice(frozendict({ v: ("red",)}), NoisyFloat.gaussian(obs=12.1, loc=12.1, scale=1.0)),
        Slice(frozendict({ v: ("green",)}), NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Slice(frozendict({ v: ("green",)}), NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Slice(frozendict({ v: ("green",)}), NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Slice(frozendict({ v: ("green",)}), NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Slice(frozendict({ v: ("green",)}), NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Slice(frozendict({ v: ("blue",)}), NoisyFloat.gaussian(obs=100.1, loc=100.1, scale=6.0)),
    ]
    release = Release(variables=(v,), slices=tuple(slices))
    f1 = fitness_noise(release, [1, 0, 0, 0, 0, 0, 0])
    f2 = fitness_noise(release, [0, 1, 0, 0, 0, 0, 0])
    f3 = fitness_noise(release, [0, 0, 0, 0, 0, 0, 1])
    f4 = fitness_noise(release, [0, 1, 1, 1, 1, 1, 0])
    assert f1 < f2
    assert f2 < f3
    assert f3 < f4


def test_noise_fitness_zero():
    release = Release(variables=(), slices=())
    fitness = fitness_noise(release, [0, 0, 0, 0])
    assert fitness == 0.0


def test_synthetic_release_disjoint():
    release = synthetic_release_disjoint(n_vals=3)
    assert len(release.variables) == 1
    assert len(release.slices) == 3


def test_synthetic_release_hierarchical():
    release = synthetic_release_hierarchical(n_levels=3, n_children=2)
    assert len(release.variables) == 1
    assert len(release.slices) > 0
    print(release)

def test_synthetic_release_chaotic():
    release = synthetic_release_chaotic(n_vals=3, n_slices=5, n_members_min=1, n_members_max=3)
    assert len(release.variables) == 1
    assert len(release.slices) == 5
    print(release)
    