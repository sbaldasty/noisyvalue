# Noise tracking in differentially private data

Work in progress!

https://chatgpt.com/c/69baecdd-f518-832e-8309-62f7a3a23167

## Current prototype API

`NoisyValue` tracks:
- A symbolic random expression (`expr`)
- One realized observation (`observed`)
- Latent symbols (`thetas`) constrained by algebraic equations (`equations`)

### Construction

Use any SymPy distribution-backed random variable:

```python
import sympy as sp
from sympy.stats import Exponential

x = NoisyValue.gaussian(10, 1)
y = NoisyValue.from_distribution(5, Exponential, 2)
# equivalent lower-level API:
# rv = Exponential("E0", 2)
# y = NoisyValue.from_noise_rv(5, rv)
```

### Sampling

`sample_n` supports mixed distributions without per-distribution branching and
can jointly sample multiple expressions while preserving shared dependencies:

```python
from src.core import sample_n

z = x * y
samples = sample_n(z, n=1000, rng=123)

joint_x, joint_y = sample_n(x, y, n=1000, rng=123)
```

By default, sampling uses independent noise draws for:
- latent-theta reconstruction from constraints
- predictive expression evaluation

This keeps uncertainty propagation working for arbitrary SymPy random variables.

### Min/max composition

Use explicit composition APIs for minima and maxima:

```python
from main import noisy_min, noisy_max

m1 = x.minimum(y)
m2 = x.maximum(y)

# n-ary helpers accept NoisyValue instances and scalars
m3 = noisy_min(x, y, 11)
m4 = noisy_max(x, y, 11)

samples_min = sample_n(m1, n=2000, rng=123)
samples_max = sample_n(m2, n=2000, rng=123)
```

These operations compose symbolic expressions via `sympy.Min`/`sympy.Max` and
preserve all latent-theta constraints, so downstream sampling continues to
propagate uncertainty through the branch structure.

We intentionally do not overload comparison operators (`<`, `>`, `<=`, `>=`) to
drive `min`/`max`. For uncertain values, comparisons are probabilistic rather
than single booleans, so boolean ordering can be inconsistent and produce
incorrect inference behavior.

### Important semantic note

This is algebraic elimination + forward Monte Carlo, not exact Bayesian
conditioning.

Also the current implementation assumes the random variables that represent
noise are all independent of each other. Not a problem for differential
privacy, but it could be a problem to generalize beyond differential privacy.
There would have to be support for "noise cloning".