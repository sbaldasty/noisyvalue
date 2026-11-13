import copy
import itertools as ite
import numpy as np
import operator


from dataclasses import dataclass
from functools import reduce

from ..core import NoisyFloat
from ..core import NoisyValue
from ..sample import sample_noisy_values
from .. import util

@dataclass(frozen=True)
class Variable:
    name: str
    values: tuple[str]


@dataclass
class Slice:
    subpop: tuple[tuple[Variable, str]]
    count: NoisyValue


@dataclass
class Release:
    variables: tuple[Variable]
    slices: tuple[Slice]


@dataclass
class Query:
    subpop: tuple[tuple[Variable, str]]
    

def merge_slices(slices):
    merged_subpop = reduce(
        lambda acc, d: acc + d,
        (s.subpop for s in slices), 
        ())
    obs_sum = sum(float(s.count) for s in slices)
    # TODO Parameterize noise scale / revisit NoisyFloat creation if we have to
    merged_count = NoisyFloat.gaussian(obs=obs_sum, loc=obs_sum, scale=1.0)
    return Slice(subpop=tuple(merged_subpop), count=merged_count)


def synthetic_release_disjoint(*, n_vals):
    """
    Generate a synthetic release over a single variable with a slice for each value.
    For such releases the genetic algorithm can easily identify the optimal genome,
    as each variable corresponds to a distinct subpopulation. It need not consider the noise
    associated with the slices.

    Args:
        n_vals (int): Number of values for the single variable.

    Returns:
        Release: A synthetic release object with disjoint subpopulations.
    """
    variable = Variable("X", tuple(f"value{i}" for i in range(n_vals)))
    slices = [Slice(
        subpop=((variable, f"value{i}"),),
        count=NoisyFloat.gaussian(obs=1.0, loc=1.0, scale=1.0)) for i in range(n_vals)]
    return Release(variables=(variable,), slices=tuple(slices))


def synthetic_release_hierarchical(*, n_levels, n_children):
    """
    Generate a synthetic release over a single variable with a hierarchical structure.
    At the bottom level, each value corresponds to a distinct subpopulation, similar to the
    disjoint case. Higher levels aggregate these subpopulations, creating a
    hierarchy of slices.

    Args:
        n_levels (int): Number of hierarchical levels.
        n_children (int): Number of variables that combine to form a higher-level slice.

    Returns:
        Release: A synthetic release object with hierarchical subpopulations.
    """
    bottom = synthetic_release_disjoint(n_vals=n_children ** (n_levels - 1))
    all_slices = list(bottom.slices)
    lvl = list(bottom.slices)
    for _ in range(n_levels - 1):
        batches = ite.batched(lvl, n_children, strict=True)
        lvl = [merge_slices(s) for s in batches]
        all_slices.extend(lvl)
    return Release(variables=bottom.variables, slices=all_slices)


def synthetic_release_chaotic(*, n_vals, n_slices, n_members_min, n_members_max, rng=None):
    """
    Generate a synthetic release with completely random subpopulations over a single variable.
    Each slice is composed of a random selection of base slices.

    Args:
        n_vals (int): Number of values for the single variable in the base release.
        n_slices (int): Number of slices to generate.
        n_members_min (int): Minimum number of base slices to combine for each new slice.
        n_members_max (int): Maximum number of base slices to combine for each new slice.

    Returns:
        Release: A synthetic release object with chaotic subpopulations.
    """
    rng = util.generator(rng)
    base = synthetic_release_disjoint(n_vals=n_vals)
    slices = []
    for _ in range(n_slices):
        n_members = rng.integers(n_members_min, n_members_max + 1)
        components = rng.choice(base.slices, size=n_members, replace=False)
        slices.append(merge_slices(components))
    return Release(variables=tuple(base.variables), slices=tuple(slices))


def query_nothing():
    """Generate an empty query."""
    return Query(subpop=())


def subpop_parts(slices):
    chain = ite.chain.from_iterable(s.subpop for s in slices)
    return tuple(dict.fromkeys(chain)) # Preserves order


def query_everything(release):
    """Generate a query that asks for every possible part of the population."""
    return Query(subpop=subpop_parts(release.slices))


def query_qty(release, qty, rng=None):
    """Generate a query for qty randomly selected unique pieces."""
    rng = util.generator(rng)
    all_possible = query_everything(release).subpop
    selection = rng.choice(len(all_possible), size=qty, replace=False)
    return Query(subpop=tuple(all_possible[i] for i in selection))


def fitness_noise(release, query, genome):
    counts = [release.slices[i].count for i, g in enumerate(genome) if g]
    total = NoisyFloat.lift(sum(counts), accept=NoisyValue)
    samples = sample_noisy_values(total, n=100)[0]
    return np.var(samples.draws)

# TODO: Is this even possible?
def slice_penalty(slices):
    counts = [penalized_slice.count for penalized_slice in slices]
    total_penalty = 0
    for count in counts:
        samples = sample_noisy_values(count, n=100)[0]
        total_penalty += np.max(samples.draws)
    return total_penalty
    
def fitness_coverage(release, query, genome):
    phenotype_include = [release.slices[i].subpop for i,g in enumerate(genome) if g == 1]
    phenotype_exclude = [release.slices[i].subpop for i,g in enumerate(genome) if g == -1]
    penalty = 0
    for subpop in subpop_parts(release.slices):
        in_query = 1 if subpop in query.subpop else 0
        num_appearances = 0
        for phenotype_subpop in phenotype_include:
            if subpop in phenotype_subpop:
                num_appearances += 1
        for phenotype_subpop in phenotype_exclude:
            if subpop in phenotype_subpop:
                num_appearances -= 1
        penalty += abs(in_query - num_appearances)
    return penalty


class Individual:
    def __init__(self, fitness_function, genome_length, release, query, fitnesses={"noise": 0, "coverage": 0},objectives={"noise": fitness_noise, "coverage": fitness_coverage}, rng=None):
        rng = util.generator(rng)
        self.fitness = 0
        self.fitness_function = fitness_function
        self.genome = [rng.integers(0, 2) for _ in range(genome_length)]
        self.release = release
        self.query = query
        self.fitnesses = fitnesses
        self.objectives = objectives
        self.sparsity = 0
        self.rank = 9999999

    def clone(self):
        """Create a copy with its own genome and fitnesses, sharing the release and query."""
        child = copy.copy(self)
        child.genome = list(self.genome)
        child.fitnesses = dict(self.fitnesses)
        return child

    def eval_fitness(self):
        self.fitness = self.fitness_function(self.release, self.genome)
    def eval_pareto_fitness(self):
        for name, fitness_func in self.objectives.items():
            self.fitnesses[name] = fitness_func(self.release, self.query, self.genome)
    def eval_weighted_fitness(self):
        weight = 1 / len(self.fitnesses)
        for _, fitness_func in self.objectives.items():
            self.fitness += weight * fitness_func(self.release, self.query, self.genome)
        

def evolutionary_algorithm(fitness_function=None, release=synthetic_release_disjoint(n_vals=3), query=Query(subpop=()),total_generations=100, num_parents=10, num_children=10, genome_length=10, num_elements_to_mutate=1, crossover=False, tournament_size=4, num_tournament_winners=2, rng=None):
    """ Evolutinary Algorithm (adapted from CS6520 assignments)
        
        Args: 
            fitness_funciton (callable function): That return the fitness of a genome 
                            given the release and genome as input parameters.
            release (Release): A synthetic release object with prespecified subpopulations.
            query (Query): A collection of subpopuations the user wishes to retrieve counts of from the release.
            total_generation (int): Number of total iterations for stopping condition.
            num_parents (int): Number of parents we downselect to at each generation (mu).
            num_childre (int): Number of children (note: parents not included in this count) that we baloon to each generation (lambda).
            genome_length (int): Length of bit string genome to be evoloved.
            num_elements_to_mutate (int): Number of alleles to modify during mutation (0 = no mutation).
            crossover (bool): Whether to perform crossover when generating children.
            tournament_size (int): Number of individuals competing in each tournament.
            num_tournament_winners (int): Number of individuals selected as future parents from each tournament (must be less than tournament_size).
        
        Returns:
            fitness_over_time (numpy array): Track record of the top fitness value at each generation.
            solutions_over_time (numpy array): Track record of the top genome value at each generation.
    """
    rng = util.generator(rng)
    # initialize record keeping
    solution = None # best genome so far
    solution_fitness = 999999 # fitness of best genome so far
    fitness_over_time = np.zeros(total_generations)
    solutions_over_time = np.zeros((total_generations,genome_length))

    # the initialization proceedure
    population = [] # keep population of individuals in a list
    for i in range(num_parents): # only create parents for initialization (the mu in mu+lambda)
        population.append(Individual(fitness_function, genome_length, release, query, rng=rng)) # generate new random individuals as parents

    # get population fitness
    for i in range(len(population)):
        population[i].eval_fitness() # evaluate the fitness of each parent
    
    for generation_num in range(total_generations): # repeat
        # the modification procedure
        new_children = [] # keep children separate for now (lambda in mu+lambda)
        while len(new_children) < num_children:
            # inheretance
            [parent1, parent2] = rng.choice(population, size=2) # pick 2 random parents
            child1 = parent1.clone() # initialize children as perfect copies of their parents
            child2 = parent2.clone()

            # crossover (for a later deliverable)
            # N/A

            # mutation
            for this_child in [child1, child2]:
                loci_indices = []
                while len(loci_indices) < num_elements_to_mutate:
                    index = rng.integers(genome_length) # randomly select the location in the child bit string to mutate
                    if index not in loci_indices:
                        loci_indices.append(index)
                for loci in loci_indices:
                    this_child.genome[loci] = (this_child.genome[loci] + 1) % 2 # flip the bit at the chosen location
            new_children.extend((child1,child2)) # add children to the new_children list

        # the assessement procedure
        for i in range(len(new_children)):
            new_children[i].eval_fitness() # assign fitness to each child

        # selection procedure
        population += new_children # combine parents with new children (the + in mu+lambda)
        population = sorted(population, key=lambda individual: individual.fitness, reverse = False) # sort the full population by each individual's fitness (from highers to lowest)
        
        # tournament selection
        new_population = []
        new_population.append(population[0])
        while len(new_population) < num_parents:
            tournament = rng.choice(population, size = tournament_size)
            tournament = sorted(tournament, key=lambda individual: individual.fitness, reverse = False)
            new_population.extend(tournament[:num_tournament_winners])
        population = new_population

        # record keeping
        if population[0].fitness < solution_fitness: # if the new parent is the best found so far
            solution = population[0].genome # update best solution records
            solution_fitness = population[0].fitness
            solution_generation = generation_num # possibly unnecessary?
        fitness_over_time[generation_num] = solution_fitness # record the fitness of the current best over evolutionary time
        solutions_over_time[generation_num,:] = solution

    return fitness_over_time, solutions_over_time

def pareto_domination(a, b, objectives={"noise":fitness_noise, "coverage":fitness_coverage}):
    dominates = False
    for objective in objectives.keys():
        # We are assuming smaller is better for noise and inverse of coverage
        if a.fitnesses[objective] < b.fitnesses[objective]: 
            dominates = True # A might dominate B
        elif b.fitnesses[objective] < a.fitnesses[objective]:
            return False # A definitely doesn't dominate B 
    return dominates


def undominated(individual, population, objectives):
    """Whether the individual is undominated in the population with respect to the objectives."""
    return not any(
        pareto_domination(other, individual, objectives)
        for other in population
        if other != individual)


def pareto_front(population, objectives = {"noise":fitness_noise, "coverage":fitness_coverage}):
    """The undominated individuals in the population with respect to the objectives."""
    return [
        individual for individual in population
        if undominated(individual, population, objectives)]


# TODO:
def get_ranks(population, release, query, objectives={"noise":fitness_noise, "coverage":fitness_coverage}):
    pop_prime = copy.deepcopy(population)
    ranks = []
    i = 1
    while len(pop_prime) > 0:
        rank_i = pareto_front(pop_prime, objectives)
        for individual in rank_i:
            individual.rank = i
            pop_prime.remove(individual)
        i += 1
        ranks.append(rank_i)
    return ranks

def get_sparsity(front, release, query, objectives={"noise":fitness_noise, "coverage":fitness_coverage}):
    for individual in front:
        individual.sparsity = 0

    for objective in objectives.keys():
        front_copy = copy.copy(front)
        front_copy = sorted(front_copy, key=lambda x: x.fitnesses[objective], reverse=False)
        # Ends are very sparse
        front_copy[0].sparsity = 9999999
        front_copy[-1].sparsity = 9999999
        for j in range(2, len(front_copy) - 1):
            front_copy[j].sparsity = front_copy[j].sparsity + (front_copy[j+1].fitnesses[objective] - front_copy[j-1].fitnesses[objective]) / 1 # TODO: Replace 1 with range
    return front

# Referencing NSGA-II pseudocode from textbook
# TODO: num_tournament_winners is not used, as textbook uses size of 2 and 1 winner always
def pareto_evolutionary_algorithm(fitness_function=None, release=synthetic_release_disjoint(n_vals=3), query=Query(subpop=()),total_generations=100, num_parents=10, num_children=10, genome_length=10, num_elements_to_mutate=1, crossover=False, tournament_size=4, num_tournament_winners=2, archive_size=10, objectives={"noise": fitness_noise, "coverage": fitness_coverage}, rng=None):
    rng = util.generator(rng)
    # initialize record keeping
    solution = None # best genome so far
    solution_fitness = 999999 # fitness of best genome so far
    fitness_over_time = []
    solutions_over_time = np.zeros((total_generations,genome_length))
    best_front = []
    ranks = []
    fronts_over_time = []
    pop_over_time =[]

    # the initialization proceedure
    population = [] # keep population of individuals in a list
    for i in range(num_parents): # only create parents for initialization (the mu in mu+lambda)
        population.append(Individual(fitness_function, genome_length, release, query, fitnesses={ "noise": 0,"coverage": 0 }, objectives=objectives, rng=rng)) # generate new random individuals as parents

    archive = []

    for generation_num in range(total_generations): # repeat
        # get population fitness
        for i in range(len(population)):
            population[i].eval_pareto_fitness() # evaluate the fitness of each parent
        
        # Combine population and archive
        population = list(set(population).union(archive))

        # Get pareto front
        best_front = pareto_front(population, objectives)

        # Compute all ranks (turn this into property of Individual)
        ranks = get_ranks(population, release, query, objectives)

        archive = []
        for front_rank in ranks:
            front = get_sparsity(front_rank, release, query, objectives)
            rank_by_sparsity = sorted(front, key=lambda x: x.sparsity, reverse=True)
            if len(archive) + len(front_rank) >= archive_size: # Last front rank to load into archive
                sparsest = rank_by_sparsity[0:archive_size - len(archive)]
                archive = list(set(archive).union(sparsest))
                break
            else:
                archive = list(set(archive).union(front_rank)) # Add entire rank!

        # "Breeding" step
        breeding_pop = archive
        new_children = [] # keep children separate for now (lambda in mu+lambda)
        while len(new_children) < num_children:
            # inheretance
            [parent1, parent2] = rng.choice(breeding_pop, size=2) # pick 2 random parents
            child1 = parent1.clone() # initialize children as perfect copies of their parents
            child2 = parent2.clone()

            # crossover (for a later deliverable)
            # N/A

            # mutation
            for this_child in [child1, child2]:
                loci_indices = []
                while len(loci_indices) < num_elements_to_mutate:
                    index = rng.integers(genome_length) # randomly select the location in the child bit string to mutate
                    if index not in loci_indices:
                        loci_indices.append(index)
                for loci in loci_indices:
                    this_child.genome[loci] = ((this_child.genome[loci] + 2) % 3) - 1 # flip the bit at the chosen location
            new_children.extend((child1,child2)) # add children to the new_children list

        # the assessement procedure
        for i in range(len(new_children)):
            new_children[i].eval_pareto_fitness() # assign fitness to each child
        
        # Selection
        breeding_pop += new_children # combine parents with new children (the + in mu+lambda)
        new_population = []
        while len(new_population) < num_parents:
            best = breeding_pop[rng.integers(len(breeding_pop))] # start with random best
            for i in range(2, tournament_size):
                next = breeding_pop[rng.integers(len(breeding_pop))]
                if next.rank < best.rank:
                    best = next
                elif next.rank == best.rank:
                    if next.sparsity > best.sparsity:
                        best = next
            new_population.append(best)

        population = new_population
        best_front = pareto_front(population, objectives)
        fronts_over_time.append(best_front)

        # record keeping
        if population[0].fitness < solution_fitness: # if the new parent is the best found so far
            solution = population[0].genome # update best solution records
            solution_fitness = population[0].fitness
            solution_generation = generation_num # possibly unnecessary?
        # solutions_over_time[generation_num,:] = solution
        
        fitness_over_time.append([(x.fitnesses["noise"], x.fitnesses["coverage"]) for x in best_front])
        pop_over_time.append(population)

    return fronts_over_time, fitness_over_time, pop_over_time




def weighted_evolutionary_algorithm(fitness_function=None, release=synthetic_release_disjoint(n_vals=3), query=Query(subpop=()),total_generations=100, num_parents=10, num_children=10, genome_length=10, num_elements_to_mutate=1, crossover=False, tournament_size=4, num_tournament_winners=2, objectives={"noise": fitness_noise, "coverage": fitness_coverage}, rng=None):
    """ Evolutinary Algorithm (adapted from CS6520 assignments)
        
        Args: 
            fitness_funciton (callable function): That return the fitness of a genome 
                            given the release and genome as input parameters.
            release (Release): A synthetic release object with prespecified subpopulations.
            query (Query): A collection of subpopuations the user wishes to retrieve counts of from the release.
            total_generation (int): Number of total iterations for stopping condition.
            num_parents (int): Number of parents we downselect to at each generation (mu).
            num_childre (int): Number of children (note: parents not included in this count) that we baloon to each generation (lambda).
            genome_length (int): Length of bit string genome to be evoloved.
            num_elements_to_mutate (int): Number of alleles to modify during mutation (0 = no mutation).
            crossover (bool): Whether to perform crossover when generating children.
            tournament_size (int): Number of individuals competing in each tournament.
            num_tournament_winners (int): Number of individuals selected as future parents from each tournament (must be less than tournament_size).
        
        Returns:
            fitness_over_time (numpy array): Track record of the top fitness value at each generation.
            solutions_over_time (numpy array): Track record of the top genome value at each generation.
    """
    rng = util.generator(rng)
    # initialize record keeping
    solution = None # best genome so far
    solution_fitness = 999999 # fitness of best genome so far
    fitness_over_time = np.zeros(total_generations)
    solutions_over_time = np.zeros((total_generations,genome_length))

    # the initialization proceedure
    population = [] # keep population of individuals in a list
    for i in range(num_parents): # only create parents for initialization (the mu in mu+lambda)
        population.append(Individual(fitness_function, genome_length, release, query, objectives=objectives, rng=rng)) # generate new random individuals as parents

    # get population fitness
    for i in range(len(population)):
        population[i].eval_weighted_fitness() # evaluate the fitness of each parent
    
    for generation_num in range(total_generations): # repeat
        # the modification procedure
        new_children = [] # keep children separate for now (lambda in mu+lambda)
        while len(new_children) < num_children:
            # inheretance
            [parent1, parent2] = rng.choice(population, size=2) # pick 2 random parents
            child1 = parent1.clone() # initialize children as perfect copies of their parents
            child2 = parent2.clone()

            # crossover (for a later deliverable)
            # N/A

            # mutation
            for this_child in [child1, child2]:
                loci_indices = []
                while len(loci_indices) < num_elements_to_mutate:
                    index = rng.integers(genome_length) # randomly select the location in the child bit string to mutate
                    if index not in loci_indices:
                        loci_indices.append(index)
                for loci in loci_indices:
                    this_child.genome[loci] = (this_child.genome[loci] + 1) % 2 # flip the bit at the chosen location
            new_children.extend((child1,child2)) # add children to the new_children list

        # the assessement procedure
        for i in range(len(new_children)):
            new_children[i].eval_weighted_fitness() # assign fitness to each child

        # selection procedure
        population += new_children # combine parents with new children (the + in mu+lambda)
        population = sorted(population, key=lambda individual: individual.fitness, reverse = False) # sort the full population by each individual's fitness (from highers to lowest)
        
        # tournament selection
        new_population = []
        new_population.append(population[0])
        while len(new_population) < num_parents:
            tournament = rng.choice(population, size = tournament_size)
            tournament = sorted(tournament, key=lambda individual: individual.fitness, reverse = False)
            new_population.extend(tournament[:num_tournament_winners])
        population = new_population

        # record keeping
        if population[0].fitness < solution_fitness: # if the new parent is the best found so far
            solution = population[0].genome # update best solution records
            solution_fitness = population[0].fitness
            solution_generation = generation_num # possibly unnecessary?
        fitness_over_time[generation_num] = solution_fitness # record the fitness of the current best over evolutionary time
        solutions_over_time[generation_num,:] = solution

    return fitness_over_time, solutions_over_time