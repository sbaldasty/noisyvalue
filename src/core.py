from dataclasses import dataclass
import sympy as sp
import sympy.stats as spstats
import numpy as np

from sympy import Abs
from sympy import And
from sympy import Eq
from sympy import Not
from sympy import Or
from sympy import Symbol
from sympy import sympify
from sympy.stats import sample
from sympy.stats.rv import random_symbols

from .util import fresh_name


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

        if self._has_cycle():
            raise ValueError("Node dependency graph contains a cycle")

    def _has_cycle(self):
        visited = set()
        in_stack = set()

        def visit(node):
            if node.symbol in in_stack:
                return True
            if node.symbol in visited:
                return False
            visited.add(node.symbol)
            in_stack.add(node.symbol)
            for dep in node.depends_on:
                if visit(dep):
                    return True
            in_stack.remove(node.symbol)
            return False

        return visit(self)

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

def _as_node(value):
    root = getattr(value, "root", None)
    if not isinstance(root, Node):
        raise TypeError(f"Expected value with Node root, got {type(value).__name__}")
    return root


def _derive_node(*parents):
    parent_nodes = tuple(_as_node(parent) for parent in parents)
    return Node(
        symbol=Symbol(fresh_name()),
        depends_on=parent_nodes,
        constraints=(),
        law=None,
        role="derived",
    )


# Backward-compatible internal aliases.
_as_unknown = _as_node
_derive_unknown = _derive_node


def _combine_float(x, y, op):
    y = as_noisy_float(y)
    obs = op(x._obs, y._obs)
    expr = op(_preferred_value_expr(x), _preferred_value_expr(y))
    root = _derive_node(x, y)
    return NoisyFloat.from_node(obs, root, expr=expr)


def _combine_bool(x, y, obs_op, expr_op):
    y = as_noisy_bool(y)
    obs = obs_op(x._obs, y._obs)
    expr = expr_op(_preferred_value_expr(x), _preferred_value_expr(y))
    root = _derive_node(x, y)
    return NoisyBool.from_node(obs, root, expr=expr)


def _compare_float(x, y, op):
    y = as_noisy_float(y)
    obs = op(x._obs, y._obs)
    expr = op(_preferred_value_expr(x), _preferred_value_expr(y))
    root = _derive_node(x, y)
    return NoisyBool.from_node(obs, root, expr=expr)


def _lift_unary_bool(x, obs_fn, expr_fn):
    x = as_noisy_bool(x)
    return NoisyBool.from_node(obs_fn(x._obs), _as_node(x), expr=expr_fn(_preferred_value_expr(x)))


def _lift_unary_float(x, obs_fn, expr_fn):
    x = as_noisy_float(x)
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


def as_noisy_bool(value):
    if isinstance(value, NoisyBool):
        return value
    if isinstance(value, (bool, np.bool_)):
        expr = sympify(bool(value))
        root = Node(symbol=expr, depends_on=(), constraints=(), law=None, role="derived")
        return NoisyBool.from_node(bool(value), root, expr=expr)
    raise TypeError(f"Expected bool or NoisyBool, got {type(value).__name__}")


def as_noisy_float(value):
    if isinstance(value, NoisyFloat):
        return value
    expr = sympify(value)
    root = Node(symbol=expr, depends_on=(), constraints=(), law=None, role="derived")
    return NoisyFloat.from_node(float(expr), root, expr=expr)


def as_noisy_int(value):
    if isinstance(value, NoisyInt):
        return value
    expr = sympify(value)
    root = Node(symbol=expr, depends_on=(), constraints=(), law=None, role="derived")
    return NoisyInt.from_node(int(expr), root, expr=expr)


def as_noisy_float_array(array):
    values = np.asarray(array, dtype=object)
    flat = values.reshape(-1)
    converted = np.array([as_noisy_float(value) for value in flat], dtype=object)
    return converted.reshape(values.shape)


def as_noisy_value(value):
    if isinstance(value, NoisyValue):
        return value
    if isinstance(value, (bool, np.bool_)):
        return as_noisy_bool(value)
    if isinstance(value, (int, np.integer)):
        return as_noisy_int(value)
    return as_noisy_float(value)


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

    @classmethod
    def from_node(cls, obs, root, expr=None):
        if not isinstance(root, Node):
            raise TypeError(f"Expected Node root, got {type(root).__name__}")

        expr = root.symbol if expr is None else sympify(expr)
        if expr != root.symbol:
            output_symbol = Symbol(fresh_name())
            root = Node(
                symbol=output_symbol,
                depends_on=(root,),
                constraints=(),
                law=None,
                definition=expr,
                role="derived",
            )
        return cls(obs, root)

    @classmethod
    def from_unknown(cls, obs, root, expr=None):
        return cls.from_node(obs, root, expr)

    @property
    def root(self):
        return self._root

    def __repr__(self):
        return f"~{self._obs}"

    def sample(self, n=1000, rng=None):
        return noisy_value_sampler(self).sample(n, rng)


class NoisyFloat(NoisyValue):
    def __init__(self, obs, root):
        super().__init__(float(obs), root)

    def __float__(self):
        return self._obs

    def __int__(self):
        return int(self._obs)

    def __abs__(self):
        return _lift_unary_float(self, abs, Abs)

    def __add__(self, other):
        return _combine_float(self, other, lambda a, b: a + b)

    def __radd__(self, other):
        return _combine_float(self, other, lambda a, b: b + a)

    def __sub__(self, other):
        return _combine_float(self, other, lambda a, b: a - b)

    def __rsub__(self, other):
        return _combine_float(self, other, lambda a, b: b - a)

    def __mul__(self, other):
        return _combine_float(self, other, lambda a, b: a * b)

    def __rmul__(self, other):
        return _combine_float(self, other, lambda a, b: b * a)

    def __truediv__(self, other):
        return _combine_float(self, other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return _combine_float(self, other, lambda a, b: b / a)

    def __lt__(self, other):
        return _compare_float(self, other, lambda a, b: a < b)

    def __le__(self, other):
        return _compare_float(self, other, lambda a, b: a <= b)

    def __gt__(self, other):
        return _compare_float(self, other, lambda a, b: a > b)

    def __ge__(self, other):
        return _compare_float(self, other, lambda a, b: a >= b)

    def __eq__(self, other):
        return _compare_float(self, other, lambda a, b: a == b)

    def __ne__(self, other):
        return _compare_float(self, other, lambda a, b: a != b)

    def exp(self):
        return _lift_unary_float(self, np.exp, sp.exp)

    def log(self):
        return _lift_unary_float(self, np.log, sp.log)

    def sqrt(self):
        return _lift_unary_float(self, np.sqrt, sp.sqrt)


class NoisyInt(NoisyFloat):
    def __init__(self, obs, root):
        NoisyValue.__init__(self, int(obs), root)

    def __float__(self):
        return float(self._obs)

    def __int__(self):
        return self._obs

    def __index__(self):
        return self._obs


class NoisyBool(NoisyValue):
    def __init__(self, obs, root):
        super().__init__(bool(obs), root)

    def __bool__(self):
        return self._obs

    def __and__(self, other):
        return _combine_bool(self, other, lambda a, b: a and b, And)

    def __rand__(self, other):
        return _combine_bool(self, other, lambda a, b: b and a, And)

    def __or__(self, other):
        return _combine_bool(self, other, lambda a, b: a or b, Or)

    def __ror__(self, other):
        return _combine_bool(self, other, lambda a, b: b or a, Or)

    def __invert__(self):
        return _lift_unary_bool(self, lambda a: not a, Not)


class NoisyValueSampler:
    def __init__(self, vals, exprs, subs, vars, law_nodes=(), lib="scipy", **kwargs):
        self._vals = tuple(vals)
        self._exprs = tuple(exprs)
        self._subs = dict(subs)
        self._vars = tuple(vars)
        self._law_nodes = tuple(law_nodes)
        self._lib = lib
        self._kwargs = dict(kwargs)

    def sample(self, n=1000, rng=None):
        dtypes = tuple(type(value._obs) for value in self._vals)

        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        def next_seed():
            return int(rng.integers(0, np.iinfo(np.int64).max))

        if n <= 0:
            empty = tuple(np.array([], dtype=dtype) for dtype in dtypes)
            return empty[0] if len(empty) == 1 else empty

        if self._vars:
            noise_draws = {
                rv: np.asarray(
                    sample(
                        rv,
                        size=n,
                        library=self._lib,
                        seed=next_seed(),
                        **self._kwargs,
                    )
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
                for theta, rhs in self._subs.items()
            }

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

                    sampled_law = _instantiate_law(node.law, resolved_values)
                    sampled_value = sample(
                        sampled_law,
                        library=self._lib,
                        seed=next_seed(),
                        **self._kwargs,
                    )
                    draws[node.symbol] = sampled_value
                    resolved_values[node.symbol] = sampled_value

                    theta_values = {
                        theta: rhs.subs(draws)
                        for theta, rhs in self._subs.items()
                    }
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

            for out_idx, sampled_value_expr in enumerate(self._exprs):
                sampled_expr = sampled_value_expr.subs(theta_values).subs(draws)
                outputs[out_idx][idx] = dtypes[out_idx](sampled_expr)

        result = tuple(outputs)
        return result[0] if len(result) == 1 else result


class FloatArraySampler:
    def __init__(self, delegate, shape):
        self._delegate = delegate
        self._shape = shape

    def sample(self, n=1000, rng=None, axis=-1):
        raw = self._delegate.sample(n=n, rng=rng)
        if isinstance(raw, tuple):
            flat = np.stack(raw, axis=0)
        else:
            flat = raw[np.newaxis, :]

        shaped = np.asarray(flat.reshape(self._shape + (n,)), dtype=float)
        if axis == -1:
            return shaped

        ndim = len(self._shape) + 1
        axis = axis if axis >= 0 else axis + ndim
        if axis < 0 or axis >= ndim:
            raise np.AxisError(axis, ndim=ndim)
        return np.moveaxis(shaped, -1, axis)
