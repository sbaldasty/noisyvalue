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

x = NoisyValue.gaussian(10, 1, provenance="A")
y = NoisyValue.from_distribution(5, Exponential, 2, provenance="B")
# equivalent lower-level API:
# rv = Exponential("E0", 2)
# y = NoisyValue.from_noise_rv(5, rv, provenance="B")
```

### Sampling

`sample_n` supports mixed distributions without per-distribution branching:

```python
z = x * y
samples = z.sample_n(1000, seed=123)
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

samples_min = m1.sample_n(2000, seed=123)
samples_max = m2.sample_n(2000, seed=123)
```

These operations compose symbolic expressions via `sympy.Min`/`sympy.Max` and
preserve all latent-theta constraints, so downstream sampling continues to
propagate uncertainty through the branch structure.

We intentionally do not overload comparison operators (`<`, `>`, `<=`, `>=`) to
drive `min`/`max`. For uncertain values, comparisons are probabilistic rather
than single booleans, so boolean ordering can be inconsistent and produce
incorrect inference behavior.

### Optional symbolic cloning

If you need symbolic elimination with fresh noise symbols, pass a `noise_cloner`
to `eliminate_thetas` or `sample_n`. The cloner must map one random variable to
one random variable.

### Important semantic note

This is algebraic elimination + forward Monte Carlo, not exact Bayesian
conditioning.
