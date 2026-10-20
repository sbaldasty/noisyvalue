import numpy as np
import sympy as sp
from sympy import sympify

from . import util
from .core import NoisyValue
from .core import _solve_theta_substitutions
from .core import _sampler_inputs_from_roots
from .consolidate import consolidate
from .graph import topological_sort_law_nodes


def _cast_draws(arr, dtype):
    arr = np.asarray(arr)
    if np.issubdtype(dtype, np.integer) and np.isnan(arr).any():
        raise ValueError(
            f"NaN in draws cannot be cast to {dtype.__name__}; an upstream noise "
            "source likely produced an invalid parameter (e.g. a probability "
            "outside [0, 1])"
        )
    return arr.astype(dtype)


def noisy_value_sampler(*vals):
    """Prepare a reusable joint sampler for one or more noisy values.

    The returned object caches symbolic setup work and can be reused for
    repeated `sample_n` calls with different sample sizes or RNG seeds.
    """
    vals = util.as_nonempty_tuple(vals, NoisyValue)
    vals = consolidate(*vals)

    (
        all_thetas,
        all_eqns,
        law_nodes,
        independent_noise_symbols,
        independent_noise,
    ) = _sampler_inputs_from_roots(vals)

    theta_substitutions = _solve_theta_substitutions(all_thetas, all_eqns)
    value_exprs = tuple(value.expr for value in vals)

    return NoisyValueSampler(
        vals,
        exprs=value_exprs,
        subs=theta_substitutions,
        independent_noise=independent_noise,
        law_nodes=law_nodes,
    )


def sample_noisy_values(*vals, n=1000, rng=None):
    """Jointly sample one or more noisy values.

    Shared latent variables and shared noise symbols are sampled once per draw,
    then reused across all requested values to preserve dependencies.
    """
    sampler = noisy_value_sampler(*vals)
    return sampler.sample(n=n, rng=rng)


class NoisyValueSampler:
    def __init__(self, vals, exprs, subs, independent_noise, law_nodes=()):
        self._vals = tuple(vals)
        self._exprs = tuple(exprs)
        self._subs = dict(subs)
        self._independent_noise = dict(independent_noise)
        self._law_nodes = topological_sort_law_nodes(law_nodes)

        # Pre-substitute latent solutions into outputs once.
        self._resolved_exprs = tuple(sympify(expr).subs(self._subs) for expr in self._exprs)

        law_symbols = {node.expr for node in self._law_nodes}
        self._theta_static = []
        self._theta_dynamic = []
        for theta, rhs in self._subs.items():
            rhs_expr = sympify(rhs)
            if rhs_expr.free_symbols & law_symbols:
                self._theta_dynamic.append((theta, rhs_expr))
            else:
                self._theta_static.append((theta, rhs_expr))
        self._theta_static = tuple(self._theta_static)
        self._theta_dynamic = tuple(self._theta_dynamic)

        eval_symbols = set(self._independent_noise.keys())
        eval_symbols |= {node.expr for node in self._law_nodes}
        self._eval_symbols = tuple(sorted(eval_symbols, key=str))
        self._resolved_expr_eval_fns = ()
        try:
            self._resolved_expr_eval_fns = tuple(
                sp.lambdify(self._eval_symbols, expr, modules="numpy")
                for expr in self._resolved_exprs
            )
        except Exception:
            self._resolved_expr_eval_fns = ()

        # Precompile per-law-node parameter lambdify functions for the vectorized path.
        # _law_batch_entries is None if compilation fails (signals scalar fallback).
        available = list(sorted(self._independent_noise.keys(), key=str))
        law_batch_entries = []
        try:
            for node in self._law_nodes:
                syms = tuple(available)
                param_fns = tuple(
                    sp.lambdify(syms, sympify(expr).subs(self._subs), modules="numpy")
                    for expr in node.params
                )
                law_batch_entries.append((node, syms, param_fns))
                available.append(node.expr)
            self._law_batch_entries = tuple(law_batch_entries)
        except Exception:
            self._law_batch_entries = None

    def sample(self, n=1000, rng=None):
        dtypes = tuple(type(value._obs) for value in self._vals)
        rng = util.generator(rng)

        if n <= 0:
            return tuple(SampleBatch(np.array([], dtype=dtype)) for dtype in dtypes)

        noise_draws = {
            sym: node.sample(rng, size=n)
            for sym, node in self._independent_noise.items()
        }

        if self._resolved_expr_eval_fns and self._law_batch_entries is not None:
            all_draws = dict(noise_draws)
            for node, syms, param_fns in self._law_batch_entries:
                args = tuple(all_draws[sym] for sym in syms)
                param_arrays = tuple(np.broadcast_to(fn(*args), (n,)) for fn in param_fns)
                all_draws[node.expr] = node.sample_arrays(rng, *param_arrays)
            eval_args = tuple(all_draws[sym] for sym in self._eval_symbols)
            return tuple(
                SampleBatch(_cast_draws(np.broadcast_to(fn(*eval_args), (n,)), dtype))
                for fn, dtype in zip(self._resolved_expr_eval_fns, dtypes)
            )

        outputs = [np.empty(n, dtype=dtype) for dtype in dtypes]

        for idx in range(n):
            draws = {sym: noise_draws[sym][idx] for sym in self._independent_noise}
            theta_values = {
                theta: rhs.subs(draws)
                for theta, rhs in self._theta_static
            }
            if self._theta_dynamic:
                theta_values.update({
                    theta: rhs.subs(draws)
                    for theta, rhs in self._theta_dynamic
                })

            resolved_values = dict(draws)
            resolved_values.update(theta_values)

            for node in self._law_nodes:
                sampled_value = node.sample(rng, resolved=resolved_values)
                draws[node.expr] = sampled_value
                resolved_values[node.expr] = sampled_value
                if self._theta_dynamic:
                    theta_values.update({
                        theta: rhs.subs(draws)
                        for theta, rhs in self._theta_dynamic
                    })
                    resolved_values.update(theta_values)

            if self._resolved_expr_eval_fns:
                eval_args = tuple(draws.get(sym, 0) for sym in self._eval_symbols)
                for out_idx, eval_fn in enumerate(self._resolved_expr_eval_fns):
                    outputs[out_idx][idx] = _cast_draws(eval_fn(*eval_args), dtypes[out_idx])
            else:
                for out_idx, sampled_value_expr in enumerate(self._resolved_exprs):
                    sampled_expr = sampled_value_expr.subs(draws)
                    outputs[out_idx][idx] = dtypes[out_idx](sampled_expr)

        return tuple(SampleBatch(x) for x in outputs)


class SampleBatch:
    def __init__(self, draws):
        draws = np.asarray(draws)
        if draws.ndim != 1:
            raise ValueError(f"Expected 1D draws array, got shape {draws.shape}")
        self.draws = draws

    def credible_interval(self, p=0.95):
        alpha = (1.0 - p) / 2.0
        return np.nanquantile(self.draws, [alpha, 1.0 - alpha], method="linear")

    def mean(self):
        return np.mean(self.draws)
