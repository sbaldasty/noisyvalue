from dataclasses import dataclass
from collections import defaultdict
import operator
import sympy as sp
import sympy.stats as spstats
import numpy as np

from sympy import Abs
from sympy import And
from sympy import Eq
from sympy import Not
from sympy import Or
from sympy import Pow
from sympy import Symbol
from sympy import sympify
from sympy.stats import sample
from sympy.stats.rv import random_symbols
from weakref import WeakValueDictionary
from weakref import WeakSet

from .util import fresh_name

_SYMBOL_NODES = WeakValueDictionary()
_SYMBOL_ASSOCIATED_NODES = defaultdict(WeakSet)


@dataclass(frozen=True)
class Node:
    symbol: sp.Basic
    depends_on: tuple["Node", ...] = ()
    constraints: tuple[sp.Expr, ...] = ()
    law: sp.Expr | None = None
    definition: sp.Expr | None = None
    role: str = "derived"

    def __post_init__(self):
        if self.role not in {"latent", "noise", "derived"}:
            raise ValueError(f"Invalid role: {self.role}")

        symbol = sympify(self.symbol)
        if not isinstance(symbol, sp.Basic):
            raise TypeError("Node.symbol must be a sympy expression atom")
        object.__setattr__(self, "symbol", symbol)

        deps = tuple(self.depends_on)
        if not all(isinstance(dep, Node) for dep in deps):
            raise TypeError("Node.depends_on must contain Node instances")
        object.__setattr__(self, "depends_on", deps)

        constraints = tuple(sympify(expr) for expr in self.constraints)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "law", None if self.law is None else sympify(self.law))
        object.__setattr__(self, "definition", self.symbol if self.definition is None else sympify(self.definition))

    def closure(self):
        seen = set()
        ordered = []

        def walk(node):
            if node.symbol in seen:
                return
            seen.add(node.symbol)
            ordered.append(node)
            for dep in node.depends_on:
                walk(dep)

        walk(self)
        return tuple(ordered)

    def latent_symbols(self):
        return {node.symbol for node in self.closure() if node.role == "latent"}

    def all_constraints(self):
        all_constraints = []
        for node in self.closure():
            all_constraints.extend(node.constraints)
        return tuple(all_constraints)

    @classmethod
    def latent(cls, symbol=None, *, constraints=(), depends_on=None, definition=None):
        symbol = _normalize_symbol(symbol)
        inferred = _resolve_depends_on(
            depends_on,
            expressions=(definition, *constraints),
            exclude_symbols={symbol},
        )
        node = cls(
            symbol=symbol,
            depends_on=inferred,
            constraints=constraints,
            definition=definition,
            role="latent",
        )
        return _register_node(node)

    @classmethod
    def noise(cls, symbol=None, *, law, constraints=(), depends_on=None, definition=None):
        symbol = _normalize_symbol(symbol)
        inferred = _resolve_depends_on(
            depends_on,
            expressions=(law, definition, *constraints),
            exclude_symbols={symbol},
        )
        node = cls(
            symbol=symbol,
            depends_on=inferred,
            constraints=constraints,
            law=law,
            definition=definition,
            role="noise",
        )
        return _register_node(node)

    @classmethod
    def derived(cls, symbol=None, *, definition, constraints=(), depends_on=None):
        symbol = _normalize_symbol(symbol)
        inferred = _resolve_depends_on(
            depends_on,
            expressions=(definition, *constraints),
            exclude_symbols={symbol},
        )
        node = cls(
            symbol=symbol,
            depends_on=inferred,
            constraints=constraints,
            definition=definition,
            role="derived",
        )
        return _register_node(node)


def _is_registerable_symbol(symbol):
    if isinstance(symbol, Symbol):
        return True
    try:
        return bool(random_symbols(sympify(symbol)))
    except Exception:
        return False


def _register_node(node, *, strict=True):
    if not isinstance(node, Node):
        raise TypeError(f"Expected Node, got {type(node).__name__}")

    if not _is_registerable_symbol(node.symbol):
        return node

    existing = _SYMBOL_NODES.get(node.symbol)
    if existing is not None and existing is not node and strict:
        raise ValueError(f"A different node is already registered for symbol: {node.symbol}")

    if existing is None:
        _SYMBOL_NODES[node.symbol] = node

    associated_symbols = {node.symbol}
    associated_symbols |= _extract_symbols(node.definition, node.law, *node.constraints)
    for symbol in associated_symbols:
        if _is_registerable_symbol(symbol):
            _SYMBOL_ASSOCIATED_NODES[symbol].add(node)

    return node


def _register_closure(root):
    root = root if isinstance(root, Node) else _as_node(root)
    for node in root.closure():
        _register_node(node, strict=False)
    return root


def _normalize_symbol(symbol):
    if symbol is None:
        return Symbol(fresh_name())
    if isinstance(symbol, str):
        return Symbol(symbol)
    return sympify(symbol)


def _dedupe_nodes(nodes):
    seen = set()
    ordered = []
    for node in nodes:
        if not isinstance(node, Node):
            raise TypeError(f"Node dependencies must contain Node instances, got {type(node).__name__}")
        if node.symbol in seen:
            continue
        seen.add(node.symbol)
        ordered.append(node)
    return tuple(ordered)


def _extract_symbols(*expressions):
    symbols = set()
    for expr in expressions:
        if expr is None:
            continue
        expr = sympify(expr)
        symbols |= set(expr.free_symbols)

        pspace = getattr(expr, "pspace", None)
        distribution = getattr(pspace, "distribution", None)
        if distribution is not None:
            for arg in distribution.args:
                arg_expr = sympify(arg)
                symbols |= set(arg_expr.free_symbols)
                for rv in random_symbols(arg_expr):
                    symbols.add(rv)
                    rv_symbol = getattr(rv, "symbol", None)
                    if rv_symbol is not None:
                        symbols.add(sympify(rv_symbol))

        for rv in random_symbols(expr):
            symbols.add(rv)
            rv_symbol = getattr(rv, "symbol", None)
            if rv_symbol is not None:
                symbols.add(sympify(rv_symbol))
    return symbols


def _resolve_depends_on(explicit_depends_on, expressions, exclude_symbols):
    if explicit_depends_on is not None:
        return _dedupe_nodes(tuple(explicit_depends_on))

    symbols = _extract_symbols(*expressions)
    symbols -= set(exclude_symbols)

    deps = []
    for symbol in sorted(symbols, key=str):
        direct = _SYMBOL_NODES.get(symbol)
        if direct is not None:
            deps.append(direct)

        associated = _SYMBOL_ASSOCIATED_NODES.get(symbol)
        if associated is not None:
            deps.extend(sorted(associated, key=lambda node: str(node.symbol)))

    return _dedupe_nodes(tuple(deps))


def _as_node(value):
    root = getattr(value, "_root", None)
    if not isinstance(root, Node):
        raise TypeError(f"Expected value with Node root, got {type(value).__name__}")
    return root


def _lift_unary_bool(x, obs_fn, expr_fn):
    x = NoisyBool.from_value(x)
    return NoisyBool.from_node(obs_fn(x._obs), _as_node(x), expr=expr_fn(_preferred_value_expr(x)))


def _lift_unary_float(x, obs_fn, expr_fn):
    x = NoisyFloat.from_value(x)
    return NoisyFloat.from_node(obs_fn(x._obs), _as_node(x), expr=expr_fn(_preferred_value_expr(x)))


def _solve_theta_substitutions(thetas, eqns):
    if not thetas:
        return {}

    equations = []
    for eq in eqns:
        eq = sympify(eq)
        if isinstance(eq, sp.Equality):
            equations.append(eq)
        else:
            equations.append(Eq(eq, 0))
    theta_list = list(thetas)

    sol = sp.solve(equations, theta_list, dict=True)
    if not sol:
        raise ValueError(f"Could not solve latent variables from constraints: {thetas}")
    chosen = sol[0]
    missing = set(thetas) - set(chosen.keys())
    if missing:
        raise ValueError(f"Latent variables are underidentified: {missing}")

    return chosen


def _instantiate_law(law, substitutions):
    law = sympify(law)
    pspace = getattr(law, "pspace", None)
    distribution = getattr(pspace, "distribution", None)
    if distribution is None:
        return law.subs(substitutions)

    ctor_name = distribution.__class__.__name__.replace("Distribution", "")
    ctor = getattr(spstats, ctor_name, None)
    if ctor is None:
        return law.subs(substitutions)

    args = tuple(sympify(arg).subs(substitutions) for arg in distribution.args)
    return ctor(fresh_name(), *args)


def _try_fast_numpy_sample(rv, rng, *, size=None):
    """Try to sample common RVs through NumPy for speed.

    Returns `None` when the RV is not recognized or parameters are not
    numerically instantiated, so callers can fall back to SymPy sampling.
    """
    pspace = getattr(rv, "pspace", None)
    distribution = getattr(pspace, "distribution", None)
    if distribution is None:
        return None

    dist_name = distribution.__class__.__name__

    if dist_name == "BinomialDistribution":
        if len(distribution.args) < 2:
            return None
        n_arg = distribution.args[0]
        p_arg = distribution.args[1]
        try:
            n = int(sympify(n_arg))
            p = float(sympify(p_arg))
        except (TypeError, ValueError):
            return None

        if n < 0 or not np.isfinite(p) or p < 0.0 or p > 1.0:
            return None

        return rng.binomial(n, p, size=size)

    if dist_name == "NormalDistribution":
        mu_arg, sigma_arg = distribution.args
        try:
            mu = float(sympify(mu_arg))
            sigma = float(sympify(sigma_arg))
        except (TypeError, ValueError):
            return None

        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma < 0.0:
            return None

        return rng.normal(mu, sigma, size=size)

    return None


def _sample_rv(rv, rng, *, lib, kwargs, next_seed, size=None):
    fast = _try_fast_numpy_sample(rv, rng, size=size)
    if fast is not None:
        arr = np.asarray(fast)
        if size is None:
            return arr.item()
        return arr

    sampled = sample(
        rv,
        size=size,
        library=lib,
        seed=next_seed(),
        **kwargs,
    )
    if size is None:
        return sampled
    return np.asarray(sampled)


def _is_binomial_rv(rv):
    pspace = getattr(rv, "pspace", None)
    distribution = getattr(pspace, "distribution", None)
    if distribution is None:
        return False
    return distribution.__class__.__name__ == "BinomialDistribution"


def _expanded_definitions(root):
    expanded = {}
    for node in reversed(root.closure()):
        expanded[node.symbol] = sympify(node.definition).subs(expanded)
    return expanded


def _preferred_value_expr(noisy_value):
    root = _as_node(noisy_value)
    expanded = _expanded_definitions(root)
    return expanded[root.symbol]


def _filter_theta_equations(eqns, thetas):
    """Keep only equations suitable for solving latent symbols.

    We keep equations that involve only latent symbols, or latent symbols plus
    random symbols. Equations with other deterministic non-latent symbols are
    excluded from latent substitution solving.
    """
    thetas = set(thetas)
    theta_eqns = []
    for eqn in eqns:
        eqn = sympify(eqn)
        non_latent_symbols = set(eqn.free_symbols) - thetas
        if not non_latent_symbols:
            theta_eqns.append(eqn)
            continue

        random_related_symbols = set()
        for rv in random_symbols(eqn):
            random_related_symbols.add(rv)
            rv_symbol = getattr(rv, "symbol", None)
            if rv_symbol is not None:
                random_related_symbols.add(rv_symbol)

        if non_latent_symbols.issubset(random_related_symbols):
            theta_eqns.append(eqn)

    return tuple(theta_eqns)


def as_noisy_float_array(array):
    values = np.asarray(array, dtype=object)
    flat = values.reshape(-1)
    converted = np.array([NoisyFloat.from_value(value) for value in flat], dtype=object)
    return converted.reshape(values.shape)


def as_noisy_value(value):
    if isinstance(value, NoisyValue):
        return value
    if isinstance(value, (bool, np.bool_)):
        return NoisyBool.from_value(value)
    if isinstance(value, (int, np.integer)):
        return NoisyInt.from_value(value)
    return NoisyFloat.from_value(value)


def _sampler_inputs_from_roots(values):
    all_thetas = set()
    all_eqns = []
    root_noise_vars = set()
    law_nodes = {}

    for value in values:
        root = _as_node(value)
        all_thetas |= root.latent_symbols()
        all_eqns.extend(root.all_constraints())
        for node in root.closure():
            if node.law is None:
                continue
            if node.symbol == node.law:
                # Exogenous RV: sample once per draw index and reuse directly.
                root_noise_vars |= set(random_symbols(node.law))
                continue

            # Law node: sample node.symbol once all dependencies are resolved.
            law_nodes[node.symbol] = node

            # Include random symbols from distribution parameters as upstream noise vars.
            pspace = getattr(node.law, "pspace", None)
            distribution = getattr(pspace, "distribution", None)
            if distribution is not None:
                for arg in distribution.args:
                    root_noise_vars |= set(random_symbols(arg))
            else:
                root_noise_vars |= set(random_symbols(node.law))

    theta_eqns = _filter_theta_equations(all_eqns, all_thetas)

    ordered_law_nodes = tuple(law_nodes[symbol] for symbol in sorted(law_nodes, key=str))
    return all_thetas, theta_eqns, root_noise_vars, ordered_law_nodes


def noisy_value_sampler(*vals, lib="scipy", **kwargs):
    """Prepare a reusable joint sampler for one or more noisy values.

    The returned object caches symbolic setup work and can be reused for
    repeated `sample_n` calls with different sample sizes or RNG seeds.
    """
    if not vals:
        raise ValueError("At least one value is required")

    noisy_values = tuple(as_noisy_value(value) for value in vals)

    all_thetas, all_eqns, root_noise_vars, law_nodes = _sampler_inputs_from_roots(noisy_values)

    theta_substitutions = _solve_theta_substitutions(all_thetas, all_eqns)

    rhs_noise_vars = {
        rv for rhs in theta_substitutions.values() for rv in random_symbols(rhs)
    }
    value_exprs = tuple(_preferred_value_expr(value) for value in noisy_values)
    predictive_noise_vars = {
        rv for expr in value_exprs for rv in random_symbols(expr)
    }
    all_noise_vars = sorted(rhs_noise_vars | predictive_noise_vars | root_noise_vars, key=str)

    return NoisyValueSampler(
        noisy_values,
        exprs=value_exprs,
        subs=theta_substitutions,
        vars=all_noise_vars,
        law_nodes=law_nodes,
        lib=lib,
        **kwargs,
    )


def sample_noisy_values(*vals, n=1000, lib="scipy", rng=None, **kwargs):
    """Jointly sample one or more noisy values.

    Shared latent variables and shared random symbols are sampled once per draw,
    then reused across all requested values to preserve dependencies.
    """
    sampler = noisy_value_sampler(*vals, lib=lib, **kwargs)
    return sampler.sample(n=n, rng=rng)


def float_array_sampler(vals, lib="scipy", **kwargs):
    """Prepare a reusable sampler for tensor-like value collections.

    The prepared sampler returns arrays with the same base shape as `values`
    plus one sample axis.
    """
    values_array = np.asarray(vals, dtype=object)
    if values_array.size == 0:
        raise ValueError("At least one value is required")

    prepared = noisy_value_sampler(
        *values_array.reshape(-1).tolist(),
        lib=lib,
        **kwargs,
    )
    return FloatArraySampler(prepared, values_array.shape)


def sample_float_array(vals, n=1000, lib="scipy", rng=None, axis=-1, **kwargs):
    """Jointly sample a tensor-like collection of values.

    Returns a float numpy array with shape `values.shape + (n,)` by default.
    Use `axis` to move the sample dimension.
    """
    sampler = float_array_sampler(vals, lib=lib, **kwargs)
    return sampler.sample(n, rng, axis)


class NoisyValue:
    def __init__(self, obs, root):
        if not isinstance(root, Node):
            raise TypeError(f"Expected Node root, got {type(root).__name__}")
        self._obs = obs
        self._root = root

    def __repr__(self):
        return f"~{self._obs}"

    def __float__(self):
        return float(self._obs)

    def __int__(self):
        return int(self._obs)

    def __bool__(self):
        return bool(self._obs)

    @classmethod
    def from_node(cls, obs, root, expr=None):
        if not isinstance(root, Node):
            raise TypeError(f"Expected Node root, got {type(root).__name__}")

        expr = root.symbol if expr is None else sympify(expr)
        if expr != root.symbol:
            output_symbol = Symbol(fresh_name())
            root = Node.derived(
                symbol=output_symbol,
                depends_on=(root,),
                definition=expr,
            )
        _register_closure(root)
        return cls(obs, root)

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        expr = sympify(value)
        root = Node.derived(definition=expr)
        return cls.from_node(expr, root)

    def sample(self, n=1000, lib="scipy", rng=None, **kwargs):
        return noisy_value_sampler(self, lib=lib, **kwargs).sample(n, rng)[0]

    def credible_interval(self, p=0.95, n=1000, lib="scipy", rng=None, **kwargs):
        return self.sample(n=n, lib=lib, rng=rng, **kwargs).credible_interval(p)

    def bin_op(self, x, out_cls, obs_op, expr_op=None, rev=False):
        x = type(self).from_value(x)
        lhs = x if rev else self
        rhs = self if rev else x

        if expr_op is None:
            expr_op = obs_op

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            obs = obs_op(lhs._obs, rhs._obs)

        expr = expr_op(_preferred_value_expr(lhs), _preferred_value_expr(rhs))
        root = Node.derived(depends_on=(_as_node(self), _as_node(x)), definition=expr)
        return out_cls.from_node(obs, root, expr)


class NoisyFloat(NoisyValue):
    def __init__(self, obs, root):
        super().__init__(float(obs), root)

    def guarded(self, guard, fallback=sp.nan):
        guard = NoisyBool.from_value(guard)
        fallback = sympify(fallback)

        obs = self._obs if bool(guard) else float(fallback)
        expr = sp.Piecewise(
            (_preferred_value_expr(self), _preferred_value_expr(guard)),
            (fallback, True),
        )
        root = Node.derived(definition=expr)
        return NoisyFloat.from_node(obs, root)

    def __abs__(self):
        return _lift_unary_float(self, abs, Abs)

    def __add__(self, other):
        return self.bin_op(other, NoisyFloat, operator.add)

    def __radd__(self, other):
        return self.bin_op(other, NoisyFloat, operator.add, rev=True)

    def __sub__(self, other):
        return self.bin_op(other, NoisyFloat, operator.sub)

    def __rsub__(self, other):
        return self.bin_op(other, NoisyFloat, operator.sub, rev=True)

    def __mul__(self, other):
        return self.bin_op(other, NoisyFloat, operator.mul)

    def __rmul__(self, other):
        return self.bin_op(other, NoisyFloat, operator.mul, rev=True)

    def __truediv__(self, other):
        return self.bin_op(other, NoisyFloat, np.divide, operator.truediv)

    def __rtruediv__(self, other):
        return self.bin_op(other, NoisyFloat, np.divide, operator.truediv, rev=True)

    def __pow__(self, other):
        return self.bin_op(other, NoisyFloat, np.power, Pow)

    def __rpow__(self, other):
        return self.bin_op(other, NoisyFloat, np.power, Pow, rev=True)

    def __lt__(self, other):
        return self.bin_op(other, NoisyBool, operator.lt)

    def __le__(self, other):
        return self.bin_op(other, NoisyBool, operator.le)

    def __gt__(self, other):
        return self.bin_op(other, NoisyBool, operator.gt)

    def __ge__(self, other):
        return self.bin_op(other, NoisyBool, operator.ge)

    def __eq__(self, other):
        return self.bin_op(other, NoisyBool, operator.eq)

    def __ne__(self, other):
        return self.bin_op(other, NoisyBool, operator.ne)

    def exp(self):
        return _lift_unary_float(self, np.exp, sp.exp)

    def log(self):
        return _lift_unary_float(self, np.log, sp.log)

    def round_nearest(self):
        expr = sp.floor(_preferred_value_expr(self) + sp.Rational(1, 2))
        obs = int(np.floor(float(self._obs) + 0.5))
        return NoisyInt.from_node(obs, self._root, expr=expr)

    def sqrt(self):
        return _lift_unary_float(self, np.sqrt, sp.sqrt)


class NoisyInt(NoisyFloat):
    def __init__(self, obs, root):
        NoisyValue.__init__(self, int(obs), root)

    def __index__(self):
        return self._obs

    def resample(self, law, *, obs=None):
        if callable(law):
            law = law()
        law = sympify(law)

        noise_node = Node.noise(law=law)

        if obs is None:
            obs = self._obs
        return NoisyInt.from_node(int(obs), noise_node, expr=noise_node.symbol)


class NoisyBool(NoisyValue):
    def __init__(self, obs, root):
        super().__init__(bool(obs), root)

    def __and__(self, other):
        return self.bin_op(other, NoisyBool, operator.and_, And)

    def __rand__(self, other):
        return self.bin_op(other, NoisyBool, operator.and_, And, rev=True)

    def __or__(self, other):
        return self.bin_op(other, NoisyBool, operator.or_, Or)

    def __ror__(self, other):
        return self.bin_op(other, NoisyBool, operator.or_, Or, rev=True)

    def __invert__(self):
        return _lift_unary_bool(self, operator.not_, Not)


class NoisyValueSampler:
    def __init__(self, vals, exprs, subs, vars, law_nodes=(), lib="scipy", **kwargs):
        self._vals = tuple(vals)
        self._exprs = tuple(exprs)
        self._subs = dict(subs)
        self._vars = tuple(vars)
        self._law_nodes = tuple(law_nodes)
        self._lib = lib
        self._kwargs = dict(kwargs)

        # Pre-substitute latent solutions into outputs once to avoid repeated
        # per-draw theta substitution.
        self._resolved_exprs = tuple(sympify(expr).subs(self._subs) for expr in self._exprs)

        law_symbols = {node.symbol for node in self._law_nodes}
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

        eval_symbols = set(self._vars)
        eval_symbols |= {node.symbol for node in self._law_nodes}
        self._eval_symbols = tuple(sorted(eval_symbols, key=str))
        self._resolved_expr_eval_fns = ()
        try:
            self._resolved_expr_eval_fns = tuple(
                sp.lambdify(self._eval_symbols, expr, modules="numpy")
                for expr in self._resolved_exprs
            )
        except Exception:
            # Keep robust symbolic fallback for unsupported expressions.
            self._resolved_expr_eval_fns = ()

    def sample(self, n=1000, rng=None):
        dtypes = tuple(type(value._obs) for value in self._vals)

        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        def next_seed():
            return int(rng.integers(0, np.iinfo(np.int64).max))

        if n <= 0:
            return tuple(SampleBatch(np.array([], dtype=dtype)) for dtype in dtypes)

        if self._vars:
            noise_draws = {
                rv: _sample_rv(
                    rv,
                    rng,
                    lib=self._lib,
                    kwargs=self._kwargs,
                    next_seed=next_seed,
                    size=n,
                )
                for rv in self._vars
            }
        else:
            noise_draws = {}

        outputs = [np.empty(n, dtype=dtype) for dtype in dtypes]

        for idx in range(n):
            draws = {rv: noise_draws[rv][idx] for rv in self._vars}
            theta_values = {
                theta: rhs.subs(draws)
                for theta, rhs in self._theta_static
            }
            if self._theta_dynamic:
                theta_values.update({
                    theta: rhs.subs(draws)
                    for theta, rhs in self._theta_dynamic
                })

            unresolved = list(self._law_nodes)
            resolved_values = dict(draws)
            resolved_values.update(theta_values)

            while unresolved:
                next_unresolved = []
                progress = False
                for node in unresolved:
                    if node.symbol in resolved_values:
                        progress = True
                        continue

                    unmet = []
                    for dep in node.depends_on:
                        # Derived structural parents (law=None, non-latent) do not
                        # represent a sampled numeric variable by themselves.
                        if dep.law is None and dep.role != "latent":
                            continue
                        if dep.symbol not in resolved_values:
                            unmet.append(dep)

                    if unmet:
                        next_unresolved.append(node)
                        continue

                    try:
                        sampled_law = _instantiate_law(node.law, resolved_values)
                        sampled_value = _sample_rv(
                            sampled_law,
                            rng,
                            lib=self._lib,
                            kwargs=self._kwargs,
                            next_seed=next_seed,
                        )
                    except (TypeError, ValueError):
                        if _is_binomial_rv(node.law):
                            sampled_value = np.nan
                        else:
                            raise
                    draws[node.symbol] = sampled_value
                    resolved_values[node.symbol] = sampled_value

                    if self._theta_dynamic:
                        theta_values.update({
                            theta: rhs.subs(draws)
                            for theta, rhs in self._theta_dynamic
                        })
                        resolved_values.update(theta_values)
                    progress = True

                if not progress:
                    missing = {
                        node.symbol: sorted(
                            str(dep.symbol)
                            for dep in node.depends_on
                            if not (
                                dep.law is None and dep.role != "latent"
                            )
                            and dep.symbol not in resolved_values
                        )
                        for node in next_unresolved
                    }
                    raise ValueError(
                        "Could not resolve law dependencies during sampling: "
                        f"{missing}"
                    )

                unresolved = next_unresolved

            if self._resolved_expr_eval_fns:
                eval_args = tuple(draws[symbol] for symbol in self._eval_symbols)
                for out_idx, eval_fn in enumerate(self._resolved_expr_eval_fns):
                    outputs[out_idx][idx] = dtypes[out_idx](eval_fn(*eval_args))
            else:
                for out_idx, sampled_value_expr in enumerate(self._resolved_exprs):
                    sampled_expr = sampled_value_expr.subs(draws)
                    outputs[out_idx][idx] = dtypes[out_idx](sampled_expr)

        return tuple(SampleBatch(x) for x in outputs)


class FloatArraySampler:
    def __init__(self, delegate, shape):
        self._delegate = delegate
        self._shape = shape

    def sample(self, n=1000, rng=None, axis=-1):
        raw = self._delegate.sample(n=n, rng=rng)
        if isinstance(raw, tuple):
            flat = np.stack([batch.draws for batch in raw], axis=0)
        else:
            flat = raw.draws[np.newaxis, :]

        shaped = np.asarray(flat.reshape(self._shape + (n,)), dtype=float)
        if axis == -1:
            return shaped

        ndim = len(self._shape) + 1
        axis = axis if axis >= 0 else axis + ndim
        if axis < 0 or axis >= ndim:
            raise np.AxisError(axis, ndim=ndim)
        return np.moveaxis(shaped, -1, axis)


class SampleBatch:
    def __init__(self, draws):
        draws = np.asarray(draws)
        assert draws.ndim == 1
        self.draws = draws

    def credible_interval(self, p=0.95):
        alpha = (1 - p) / 2
        return np.quantile(self.draws, [alpha, 1 - alpha], method="linear")
