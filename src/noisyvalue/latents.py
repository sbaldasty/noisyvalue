"""Helpers for solving latent symbols from the constraints in a node DAG.

Shared by sampling, consolidation, and visualization, which all need to turn
the latents and constraints under some root nodes into symbolic substitutions.
"""
import sympy as sp

from typing import NamedTuple

from sympy import Eq, Equality
from sympy import sympify

from .graph import NoiseNode


def solve_theta_substitutions(thetas, eqns):
    """Solve latent symbols symbolically from their constraint equations.

    `eqns` may mix `Equality` objects and bare expressions; the latter are
    treated as `expr == 0`. Returns a dict mapping each symbol in `thetas` to
    an expression over the remaining (noise) symbols, or `{}` if there are no
    latents. Raises `ValueError` if the system has no solution or leaves any
    latent underidentified. If `sympy.solve` returns several solutions, the
    first is used.
    """
    if not thetas:
        return {}

    equations = []
    for eq in eqns:
        eq = sympify(eq)
        if isinstance(eq, Equality):
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


def filter_theta_equations(eqns, thetas, independent_noise_symbols):
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


class SamplerInputs(NamedTuple):
    """Everything a sampler needs from the DAGs under some root nodes.

    - `thetas`: the set of latent symbols across all roots.
    - `theta_eqns`: constraint equations usable for solving the latents, i.e.
      those whose non-latent symbols are all independent noise symbols (see
      `filter_theta_equations`).
    - `law_nodes`: dependent noise nodes (those with `deps`), sorted by
      symbol name.
    - `independent_noise_symbols`: the symbols of noise nodes with no deps.
    - `independent_noise`: the same noise nodes, keyed by their symbol.
    """
    thetas: set
    theta_eqns: tuple
    law_nodes: tuple
    independent_noise_symbols: set
    independent_noise: dict


def sampler_inputs_from_roots(roots):
    """Collect a `SamplerInputs` from the DAGs under `roots`.

    Walks the closure of every root node, so nodes shared between roots are
    counted once.
    """
    all_thetas = set()
    all_eqns = set()
    dependent_law_nodes = {}
    all_nodes = {}

    for root in roots:
        for node in root.closure():
            all_nodes[node.expr] = node
        all_thetas |= root.latent_symbols()
        all_eqns.update(root.all_constraints())
        for node in root.closure():
            if not isinstance(node, NoiseNode) or not node.deps:
                continue
            dependent_law_nodes[node.expr] = node

    independent_noise = {
        node.expr: node
        for node in all_nodes.values()
        if isinstance(node, NoiseNode) and not node.deps
    }
    independent_noise_symbols = set(independent_noise.keys())
    theta_eqns = filter_theta_equations(all_eqns, all_thetas, independent_noise_symbols)

    ordered_law_nodes = tuple(
        dependent_law_nodes[sym] for sym in sorted(dependent_law_nodes, key=str)
    )
    return SamplerInputs(
        thetas=all_thetas,
        theta_eqns=theta_eqns,
        law_nodes=ordered_law_nodes,
        independent_noise_symbols=independent_noise_symbols,
        independent_noise=independent_noise,
    )
