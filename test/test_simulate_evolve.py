from noisyvalue.dataset.evolve import evolutionary_algorithm, pareto_evolutionary_algorithm, pareto_front, weighted_evolutionary_algorithm, query_qty
from noisyvalue.dataset.evolve import Individual, Variable, Slice, NoisyFloat, Release
import numpy as np
from noisyvalue.dataset.evolve import fitness_noise, fitness_coverage
from noisyvalue.dataset.evolve import synthetic_release_disjoint
from noisyvalue.dataset.evolve import synthetic_release_hierarchical
from noisyvalue.dataset.evolve import synthetic_release_chaotic
from noisyvalue.dataset.evolve import Query
import warnings
import matplotlib.pyplot as plt
import scikits.bootstrap as bootstrap

warnings.filterwarnings('ignore') 
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
    plt.show()

experiment_results = {}
solutions_results = {}

###################################
# Original single objective testing
###################################
def single_obj_test():
    num_runs = 20
    total_generations = 100
    num_elements_to_mutate = 1
    genome_length = 15
    num_parents = 50
    num_children = 50
    tournament_size = 10
    num_tournament_winners = 5
    # release = synthetic_release_chaotic(n_vals=3, n_slices=genome_length, n_members_min=1, n_members_max=3)
    release = synthetic_release_disjoint(n_vals=genome_length)
    names = ["50 select 5", "50 select 20", "20 select 5", "20 select 20"]
    t_sizes = [50, 50, 20, 20]
    t_winners = [5, 20, 5, 20]
    for i in range(len(names)):
        run_name = names[i]
        experiment_results[run_name] = np.zeros((num_runs, total_generations))
        solutions_results[run_name] = np.zeros((num_runs, total_generations, genome_length))
        tournament_size = t_sizes[i]
        num_tournament_winners = t_winners[i]
        for run_num in range(num_runs):
            fitness_over_time, solutions_over_time = evolutionary_algorithm(fitness_function=fitness_noise, 
                                                                            release=release,
                                                                            total_generations=total_generations,
                                                                            num_parents=num_parents,
                                                                            num_children=num_children,
                                                                            genome_length=genome_length,
                                                                            num_elements_to_mutate=num_elements_to_mutate,
                                                                            crossover=False,
                                                                            tournament_size=tournament_size,
                                                                            num_tournament_winners=num_tournament_winners)
            experiment_results[run_name][run_num] = fitness_over_time
            solutions_results[run_name][run_num] = solutions_over_time
            print(run_name, run_num, fitness_over_time[-1])

    # plotting
    plot_mean_and_bootstrapped_ci_over_time(input_data = experiment_results, name = names,x_label = "Generation", y_label = "Fitness", plot_bootstrap = True)


###################################
# Multi-objective testing
###################################
def multi_obj_test():
    num_runs = 1#20
    total_generations = 100
    num_elements_to_mutate = 1
    genome_length = 15
    num_parents = 50
    num_children = 50
    tournament_size = 2 # these are commonly used by the textbook, (2, 1) but might not be optimal
    num_tournament_winners = 1
    archive_size = 50
    release = synthetic_release_chaotic(n_vals=300, n_slices=genome_length, n_members_min=10, n_members_max=30)
    # release = synthetic_release_disjoint(n_vals=genome_length)
    # release = synthetic_release_hierarchical(n_levels=3, n_children=genome_length)
    query = query_qty(release, qty=30)
    objectives = {"noise": fitness_noise, "coverage": fitness_coverage}

    # More interesting query/release
    v = Variable("color", ("red", "green", "blue"))
    query = Query(subpop=((v, "red"), (v, "green")))
    slices = [
        Slice(((v, "red"),), NoisyFloat.gaussian(obs=12.1, loc=12.1, scale=1.0)),
        Slice(((v, "green"),), NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Slice(((v, "green"),), NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Slice(((v, "green"),), NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Slice(((v, "green"),), NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Slice(((v, "green"),), NoisyFloat.gaussian(obs=4.4, loc=4.4, scale=3.0)),
        Slice(((v, "blue"),), NoisyFloat.gaussian(obs=100.1, loc=100.1, scale=6.0)),
    ]
    release = Release(variables=(v,), slices=tuple(slices))
    genome_length = 7

    names = ["2 select 1 pareto"]
    for i in range(len(names)):
        run_name = names[i]
        experiment_results[run_name] = np.zeros((num_runs, total_generations))
        solutions_results[run_name] = np.zeros((num_runs, total_generations, genome_length))
        for run_num in range(num_runs):
            fronts_over_time, fitness_over_time, pop_over_time = pareto_evolutionary_algorithm(fitness_function=None, 
                                                                                release=release, 
                                                                                query=query,
                                                                                total_generations=total_generations, 
                                                                                num_parents=num_parents, 
                                                                                num_children=num_children, 
                                                                                genome_length=genome_length, 
                                                                                num_elements_to_mutate=num_elements_to_mutate, 
                                                                                crossover=False, 
                                                                                tournament_size=tournament_size, 
                                                                                num_tournament_winners=num_tournament_winners, 
                                                                                archive_size=archive_size, 
                                                                                objectives=objectives)
            
            # fitness_over_time[-1] gives array of (noise fitness, coverage fitness) tuples for front
            print(run_name, run_num)
            xs = []
            ys = []
            front = fronts_over_time[-1]
            print(f"{front[0].fitnesses} {front[0].genome}")
            # Plot all "dominated" points in final population (TODO: do we even care about rest of pop, or do we prefer the previous fronts?)
            for pop in pop_over_time:
                for ind in pop:
                    if ind not in front:
                        xs.append(ind.fitnesses["noise"])
                        ys.append(ind.fitnesses["coverage"])
            plt.scatter(xs, ys, marker="x")
            xs = []
            ys = []
            # Plot the final front
            for ind in fronts_over_time[-1]:
                # print("noise: ", ind.fitnesses["noise"], " coverage: ", ind.fitnesses["coverage"])
                xs.append(ind.fitnesses["noise"])
                ys.append(ind.fitnesses["coverage"])
            plt.scatter(xs, ys) 
            plt.xlabel("Noise")
            plt.ylabel("Coverage")
            plt.title(run_name)
        plt.show()
        
###################################
# Weighted multi objective testing
###################################
def weighted_multi_test():
    num_runs = 20
    total_generations = 100
    num_elements_to_mutate = 1
    genome_length = 15
    num_parents = 50
    num_children = 50
    tournament_size = 10
    num_tournament_winners = 5
    release = synthetic_release_chaotic(n_vals=3, n_slices=genome_length, n_members_min=1, n_members_max=3)
    # release = synthetic_release_disjoint(n_vals=genome_length)
    # release = synthetic_release_hierarchical(n_levels=3, n_children=genome_length)
    query = query = Query(subpop=())
    objectives = {"noise": fitness_noise, "coverage": fitness_coverage}    
    # t_sizes = [50, 50, 20, 20]
    # t_winners = [5, 20, 5, 20]
    names = ["2 select 1 weighted"]

    for i in range(len(names)):
        run_name = names[i]
        experiment_results[run_name] = np.zeros((num_runs, total_generations))
        solutions_results[run_name] = np.zeros((num_runs, total_generations, genome_length))
        # tournament_size = t_sizes[i]
        # num_tournament_winners = t_winners[i]
        for run_num in range(num_runs):
            fitness_over_time, solutions_over_time = weighted_evolutionary_algorithm(fitness_function=fitness_noise, 
                                                                            release=release,
                                                                            query=query,
                                                                            total_generations=total_generations,
                                                                            num_parents=num_parents,
                                                                            num_children=num_children,
                                                                            genome_length=genome_length,
                                                                            num_elements_to_mutate=num_elements_to_mutate,
                                                                            crossover=False,
                                                                            tournament_size=tournament_size,
                                                                            num_tournament_winners=num_tournament_winners,
                                                                            objectives=objectives)
            experiment_results[run_name][run_num] = fitness_over_time
            solutions_results[run_name][run_num] = solutions_over_time
            print(run_name, run_num, fitness_over_time[-1])

    # plotting
    plot_mean_and_bootstrapped_ci_over_time(input_data = experiment_results, name = names,x_label = "Generation", y_label = "Fitness", plot_bootstrap = True)


# TODO: Make a plotting function to visualize the pareto front found at end of each run
def pareto_rank_visual():
    release = synthetic_release_chaotic(n_vals=3, n_slices=15, n_members_min=1, n_members_max=3)
    query = query = Query(subpop=())
    objectives = {"noise": fitness_noise, "coverage": fitness_coverage}

    test_pop = []
    for i in range(50):
        i1 = Individual(None,10, release, query, fitnesses = {"noise": 0, "coverage": 0}, objectives=objectives)
        i1.fitnesses["noise"] = np.random.randint(-10, 10)
        i1.fitnesses["coverage"] = np.random.randint(-10, 10)
        test_pop.append(i1)

    front = pareto_front(test_pop, objectives)
    xs = []
    ys = []
    for i in test_pop:
        xs.append(i.fitnesses["noise"])
        ys.append(i.fitnesses["coverage"])
    plt.scatter(xs, ys, marker="x", color="r")
    xs = []
    ys = []
    for i in front:
        xs.append(i.fitnesses["noise"])
        ys.append(i.fitnesses["coverage"])
    plt.scatter(xs, ys, marker="o", color='g')
    plt.show()

# multi_obj_test()
weighted_multi_test()
