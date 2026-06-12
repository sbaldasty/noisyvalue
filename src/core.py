import operator as op
import sympy as sp
import numpy as np

from sympy import Abs, And, Eq, Not, Or, Pow, Symbol
from sympy import sympify

from .util import fresh_name


def _as_node(value):
    root = getattr(value, "_root", None)
    if not isinstance(root, Node):
        raise TypeError(f"Expected value with Node root, got {type(value).__name__}")
    return root


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


def _expanded_definitions(root):
    expanded = {}
    for node in reversed(root.closure()):
        expanded[node.symbol] = sympify(node.definition).subs(expanded)
    return expanded


def _preferred_value_expr(noisy_value):
    root = _as_node(noisy_value)
    expanded = _expanded_definitions(root)
    return expanded[root.symbol]


def _filter_theta_equations(eqns, thetas, independent_noise_symbols):
    """Keep only equations suitable for solving latent symbols.

    We keep equations whose non-latent free symbols are all independent noise
    symbols (plain symbols of noise nodes with no dependencies).
    """
    thetas = set(thetas)
    theta_eqns = []
    for eqn in eqns:
        eqn = sympify(eqn)
        non_latent_symbols = eqn.free_symbols - thetas
        if not non_latent_symbols:
            theta_eqns.append(eqn)
            continue
        if non_latent_symbols.issubset(independent_noise_symbols):
            theta_eqns.append(eqn)
    return tuple(theta_eqns)


def _sampler_inputs_from_roots(values):
    all_thetas = set()
    all_eqns = []
    dependent_law_nodes = {}
    all_nodes = {}

    for value in values:
        root = _as_node(value)
        for node in root.closure():
            all_nodes[node.symbol] = node
        all_thetas |= root.latent_symbols()
        all_eqns.extend(root.all_constraints())
        for node in root.closure():
            if node.source is None or not node.depends_on:
                continue
            dependent_law_nodes[node.symbol] = node

    independent_noise = {
        node.symbol: node.source
        for node in all_nodes.values()
        if node.role == "noise" and node.source is not None and not node.depends_on
    }
    independent_noise_symbols = set(independent_noise.keys())
    theta_eqns = _filter_theta_equations(all_eqns, all_thetas, independent_noise_symbols)

    ordered_law_nodes = tuple(
        dependent_law_nodes[sym] for sym in sorted(dependent_law_nodes, key=str)
    )
    return all_thetas, theta_eqns, ordered_law_nodes, independent_noise_symbols, independent_noise


def noisy_value_sampler(*vals):
    """Prepare a reusable joint sampler for one or more noisy values.

    The returned object caches symbolic setup work and can be reused for
    repeated `sample_n` calls with different sample sizes or RNG seeds.
    """
    assert vals and all(isinstance(x, NoisyValue) for x in vals)

    (
        all_thetas,
        all_eqns,
        law_nodes,
        independent_noise_symbols,
        independent_noise,
    ) = _sampler_inputs_from_roots(vals)

    theta_substitutions = _solve_theta_substitutions(all_thetas, all_eqns)
    value_exprs = tuple(_preferred_value_expr(value) for value in vals)

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


class Node:
    def __init__(self, role, definition=None, source=None, constraints=(), depends_on=()):
        assert role in {"latent", "noise", "derived"}
        self.role = role
        self.symbol = Symbol(fresh_name())
        self.source = source
        self.definition = self.symbol if definition is None else sympify(definition)
        self.constraints = tuple(sympify(x) for x in constraints)
        self.depends_on = tuple(depends_on)
        if not all(isinstance(node, Node) for node in self.depends_on):
            raise TypeError("depends_on must contain Node instances")

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
    def latent(cls, *, constraints=(), definition=None, depends_on=()):
        return Node("latent", definition=definition, constraints=constraints, depends_on=depends_on)

    @classmethod
    def noise(cls, source, *, constraints=(), definition=None, depends_on=()):
        return Node("noise", definition=definition, source=source, constraints=constraints, depends_on=depends_on)

    @classmethod
    def derived(cls, definition, *, constraints=(), depends_on=()):
        return cls("derived", definition=definition, constraints=constraints, depends_on=depends_on)


class NoisyValue:
    def __init__(self, obs, root):
        assert isinstance(root, Node)
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
        if expr != root.symbol and expr != root.definition:
            root = Node.derived(definition=expr, depends_on=(root,))

        return cls(obs, root)

    @classmethod
    def draw(cls, true_value, noise_source, rng=None):
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        theta_node = Node.latent()
        noise_node = Node.noise(noise_source)
        theta = theta_node.symbol
        noise_sym = noise_node.symbol
        obs_noise = float(noise_source.sample(rng))
        obs = float(sympify(true_value)) + obs_noise
        root = Node.derived(
            constraints=(theta + noise_sym - obs,),
            definition=theta,
            depends_on=(theta_node, noise_node))
        return cls(obs, root)

    @classmethod
    def lift(cls, value, accept=None):
        accept = cls if accept is None else accept
        assert issubclass(accept, NoisyValue)
        return value if isinstance(value, accept) else cls(value, Node.derived(value))

    def sample(self, n=1000, rng=None):
        return noisy_value_sampler(self).sample(n, rng)[0]

    def credible_interval(self, p=0.95, n=1000, rng=None):
        return self.sample(n=n, rng=rng).credible_interval(p)

    def bin_op(self, x, out_cls, obs_op, expr_op=None, rev=False):
        x = type(self).lift(x)
        lhs = x if rev else self
        rhs = self if rev else x

        if expr_op is None:
            expr_op = obs_op

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            obs = obs_op(lhs._obs, rhs._obs)

        expr = expr_op(_preferred_value_expr(lhs), _preferred_value_expr(rhs))
        root = Node.derived(expr, depends_on=(lhs._root, rhs._root))
        return out_cls(obs, root)

    def unary_op(self, out_cls, obs_op, expr_op):
        return out_cls.from_node(obs_op(self._obs), self._root, expr=expr_op(_preferred_value_expr(self)))


class NoisyNumber(NoisyValue):
    def __init__(self, obs, root):
        super().__init__(obs, root)

    def __abs__(self):
        return self.unary_op(NoisyFloat, abs, Abs)

    def __add__(self, other):
        return self.bin_op(other, NoisyFloat, op.add)

    def __radd__(self, other):
        return self.bin_op(other, NoisyFloat, op.add, rev=True)

    def __sub__(self, other):
        return self.bin_op(other, NoisyFloat, op.sub)

    def __rsub__(self, other):
        return self.bin_op(other, NoisyFloat, op.sub, rev=True)

    def __mul__(self, other):
        return self.bin_op(other, NoisyFloat, op.mul)

    def __rmul__(self, other):
        return self.bin_op(other, NoisyFloat, op.mul, rev=True)

    def __truediv__(self, other):
        return self.bin_op(other, NoisyFloat, np.divide, op.truediv)

    def __rtruediv__(self, other):
        return self.bin_op(other, NoisyFloat, np.divide, op.truediv, rev=True)

    def __pow__(self, other):
        return self.bin_op(other, NoisyFloat, np.power, Pow)

    def __rpow__(self, other):
        return self.bin_op(other, NoisyFloat, np.power, Pow, rev=True)

    def __lt__(self, other):
        return self.bin_op(other, NoisyBool, op.lt)

    def __le__(self, other):
        return self.bin_op(other, NoisyBool, op.le)

    def __gt__(self, other):
        return self.bin_op(other, NoisyBool, op.gt)

    def __ge__(self, other):
        return self.bin_op(other, NoisyBool, op.ge)

    def __eq__(self, other):
        return self.bin_op(other, NoisyBool, op.eq)

    def __ne__(self, other):
        return self.bin_op(other, NoisyBool, op.ne)

    def guarded(self, guard, fallback=sp.nan):
        guard = NoisyBool.lift(guard)
        fallback = sympify(fallback)

        obs = self._obs if bool(guard) else float(fallback)
        expr = sp.Piecewise(
            (_preferred_value_expr(self), _preferred_value_expr(guard)),
            (fallback, True))

        root = Node.derived(expr, depends_on=(self._root, guard._root))
        return type(self)(obs, root)


class NoisyFloat(NoisyNumber):
    def __init__(self, obs, root):
        super().__init__(float(obs), root)

    def exp(self):
        return self.unary_op(NoisyFloat, np.exp, sp.exp)

    def log(self):
        return self.unary_op(NoisyFloat, np.log, sp.log)

    def round_nearest(self):
        expr = sp.floor(_preferred_value_expr(self) + sp.Rational(1, 2))
        obs = np.floor(self._obs + 0.5)
        return NoisyInt.from_node(obs, self._root, expr=expr)

    def sqrt(self):
        return self.unary_op(NoisyFloat, np.sqrt, sp.sqrt)


class NoisyInt(NoisyNumber):
    def __init__(self, obs, root):
        super().__init__(int(obs), root)

    def __index__(self):
        return self._obs

    def resample(self, source, *, obs=None):
        noise_node = Node.noise(source, depends_on=(self._root,))
        if obs is None:
            obs = self._obs
        return NoisyInt.from_node(int(obs), noise_node, expr=noise_node.symbol)


class NoisyBool(NoisyValue):
    def __init__(self, obs, root):
        super().__init__(bool(obs), root)

    def __and__(self, other):
        return self.bin_op(other, NoisyBool, op.and_, And)

    def __rand__(self, other):
        return self.bin_op(other, NoisyBool, op.and_, And, rev=True)

    def __or__(self, other):
        return self.bin_op(other, NoisyBool, op.or_, Or)

    def __ror__(self, other):
        return self.bin_op(other, NoisyBool, op.or_, Or, rev=True)

    def __invert__(self):
        return self.unary_op(NoisyBool, op.not_, Not)


class NoisyValueSampler:
    def __init__(self, vals, exprs, subs, independent_noise, law_nodes=()):
        self._vals = tuple(vals)
        self._exprs = tuple(exprs)
        self._subs = dict(subs)
        self._independent_noise = dict(independent_noise)
        self._law_nodes = tuple(law_nodes)

        # Pre-substitute latent solutions into outputs once.
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

        eval_symbols = set(self._independent_noise.keys())
        eval_symbols |= {node.symbol for node in self._law_nodes}
        self._eval_symbols = tuple(sorted(eval_symbols, key=str))
        self._resolved_expr_eval_fns = ()
        try:
            self._resolved_expr_eval_fns = tuple(
                sp.lambdify(self._eval_symbols, expr, modules="numpy")
                for expr in self._resolved_exprs
            )
        except Exception:
            self._resolved_expr_eval_fns = ()

    def sample(self, n=1000, rng=None):
        dtypes = tuple(type(value._obs) for value in self._vals)

        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        if n <= 0:
            return tuple(SampleBatch(np.array([], dtype=dtype)) for dtype in dtypes)

        noise_draws = {
            sym: source.sample(rng, size=n)
            for sym, source in self._independent_noise.items()
        }

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

            unresolved = list(self._law_nodes)
            while unresolved:
                next_unresolved = []
                progress = False
                for node in unresolved:
                    if node.symbol in resolved_values:
                        progress = True
                        continue

                    unmet = []
                    for dep in node.depends_on:
                        if dep.source is None and dep.role != "latent":
                            continue
                        if dep.symbol not in resolved_values:
                            unmet.append(dep)

                    if unmet:
                        next_unresolved.append(node)
                        continue

                    sampled_value = node.source.instantiate(resolved_values).sample(rng)
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
                            if not (dep.source is None and dep.role != "latent")
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
                eval_args = tuple(draws.get(sym, 0) for sym in self._eval_symbols)
                for out_idx, eval_fn in enumerate(self._resolved_expr_eval_fns):
                    outputs[out_idx][idx] = dtypes[out_idx](eval_fn(*eval_args))
            else:
                for out_idx, sampled_value_expr in enumerate(self._resolved_exprs):
                    sampled_expr = sampled_value_expr.subs(draws)
                    outputs[out_idx][idx] = dtypes[out_idx](sampled_expr)

        return tuple(SampleBatch(x) for x in outputs)


class SampleBatch:
    def __init__(self, draws):
        draws = np.asarray(draws)
        assert draws.ndim == 1
        self.draws = draws

    def credible_interval(self, p=0.95):
        alpha = (1.0 - p) / 2.0
        return np.quantile(self.draws, [alpha, 1.0 - alpha], method="linear")

    def mean(self):
        return np.mean(self.draws)
