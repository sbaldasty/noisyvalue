import copy
import itertools as ite
import matplotlib.pyplot as plt
import numpy as np
import operator
import scikits.bootstrap as bootstrap


from dataclasses import dataclass
from functools import reduce

from ..core import NoisyFloat
from ..core import NoisyValue
from ..sample import sample_noisy_values


@dataclass(frozen=True)
class Variable:
    name: str
    values: tuple[str]


@dataclass
class Slice:
    subpop: dict[Variable, tuple[str]]
    count: NoisyValue


@dataclass
class Release:
    variables: tuple[Variable]
    slices: tuple[Slice]


@dataclass
class Query:
    subpop: dict[Variable, tuple[str]]


class Individual:
    def __init__(self, fitness_function, genome_length, release):
        self.fitness = 0
        self.fitness_function = fitness_function
        self.genome = [0 for _ in range(genome_length)]
        self.release = release
    def evaluate_fitness(self):
        self.fitness = self.fitness_function(self.release, self.genome)


def merge_slices(slices):
    merged_subpop = reduce(
        lambda acc, d: {k: acc.get(k, ()) + d.get(k, ()) for k in set(acc) | set(d)}, 
        (s.subpop for s in slices), 
        {})
    merged_count = sum(s.count for s in slices)
    return Slice(subpop=merged_subpop, count=merged_count)


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
    slices = [Slice({variable: (f"value{i}",)}, NoisyFloat.gaussian(obs=1.0, loc=1.0, scale=1.0)) for i in range(n_vals)]
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


def synthetic_release_chaotic(*, n_vals, n_slices, n_members_min, n_members_max):
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
    base = synthetic_release_disjoint(n_vals=n_vals)
    slices = []
    for _ in range(n_slices):
        n_members = np.random.randint(n_members_min, n_members_max + 1)
        components = np.random.choice(base.slices, size=n_members, replace=False)
        slices.append(merge_slices(components))
    return Release(variables=tuple(base.variables), slices=tuple(slices))


def fitness_noise(release, genome):
    counts = [release.slices[i].count for i, g in enumerate(genome) if g]
    if not counts:
        return 0.0

    total = sum(counts)
    samples = sample_noisy_values(total, n=100)[0]
    return np.var(samples.draws)


def evolutionary_algorithm(fitness_function=None, release=synthetic_release_disjoint(n_vals=3), total_generations=100, num_parents=10, num_children=10, genome_length=10, num_elements_to_mutate=1, crossover=False, tournament_size=4, num_tournament_winners=2):
    """ Evolutinary Algorithm (adapted from CS6520 assignments)
        
        Args: 
            fitness_funciton (callable function): That return the fitness of a genome 
                            given the genome as an input parameter.
            release (Release): A synthetic release object with prespecified subpopulations.
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
    # initialize record keeping
    solution = None # best genome so far
    solution_fitness = 999999 # fitness of best genome so far
    fitness_over_time = np.zeros(total_generations)
    solutions_over_time = np.zeros((total_generations,genome_length))

    # the initialization proceedure
    population = [] # keep population of individuals in a list
    for i in range(num_parents): # only create parents for initialization (the mu in mu+lambda)
        population.append(Individual(fitness_function, genome_length, release)) # generate new random individuals as parents

    # get population fitness
    for i in range(len(population)):
        population[i].eval_fitness() # evaluate the fitness of each parent
    
    for generation_num in range(total_generations): # repeat
        # the modification procedure
        new_children = [] # keep children separate for now (lambda in mu+lambda)
        while len(new_children) < num_children:
            # inheretance
            [parent1, parent2] = np.random.choice(population, size=2) # pick 2 random parents
            child1 = copy.deepcopy(parent1) # initialize children as perfect copies of their parents
            child2 = copy.deepcopy(parent2)

            # crossover (for a later deliverable)
            # N/A

            # mutation
            for this_child in [child1, child2]:
                loci_indices = []
                while len(loci_indices) < num_elements_to_mutate:
                    index = np.random.randint(genome_length) # randomly select the location in the child bit string to mutate
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
        population = sorted(population, key=lambda individual: individual.fitness, reverse = True) # sort the full population by each individual's fitness (from highers to lowest)
        
        # tournament selection
        new_population = []
        new_population.append(population[0])
        while len(new_population) < num_parents:
            tournament = np.random.choice(population, size = tournament_size)
            tournament = sorted(tournament, key=lambda individual: individual.fitness, reverse = True)
            new_population.extend(tournament[:num_tournament_winners])
        population = new_population

        # record keeping
        if population[0].fitness > solution_fitness: # if the new parent is the best found so far
            solution = population[0].genome # update best solution records
            solution_fitness = population[0].fitness
            solution_generation = generation_num # possibly unnecessary?
        fitness_over_time[generation_num] = solution_fitness # record the fitness of the current best over evolutionary time
        solutions_over_time[generation_num,:] = solution

    return fitness_over_time, solutions_over_time

def plot_mean_and_bootstrapped_ci_over_time(input_data = None, name = "change me", x_label = "change me", y_label="change me", y_limit = None, plot_bootstrap = True):
    """
    
    Parameters: 
        input_data (numpy array of shape (max_k, num_repitions)): Solution metric to plot.
        name (string): Name for legend.
        x_label (string): x axis label.
        y_label (string): y axis label.
    
    Returns:
        None
    """
    fig, ax = plt.subplots() # generate figure and axes
    if isinstance(name, str): 
        name = [name]; input_data = [input_data]

    for this_name in name:
        print("plotting",this_name)
        this_input_data = input_data[this_name]
        total_generations = this_input_data.shape[1]

        if plot_bootstrap:
            boostrap_ci_generation_found = np.zeros((2,total_generations))
            for this_gen in range(total_generations):
                if this_gen%10==0: 
                    print(this_gen)
                boostrap_ci_generation_found[:,this_gen] = bootstrap.ci(this_input_data[:,this_gen], np.mean, alpha=0.05)
        ax.plot(np.arange(total_generations), np.mean(this_input_data,axis=0), label = this_name) # plot the fitness over time
        if plot_bootstrap:
            ax.fill_between(np.arange(total_generations),boostrap_ci_generation_found[0,:], boostrap_ci_generation_found[1,:],alpha=0.3) # plot, and fill, the confidence interval for fitness over time
        ax.set_xlabel(x_label) # add axes labels
        ax.set_ylabel(y_label)
        if y_limit: 
            ax.set_ylim(y_limit[0],y_limit[1])
        plt.legend(loc='best'); # add legend