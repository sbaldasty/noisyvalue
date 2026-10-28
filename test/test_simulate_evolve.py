from noisyvalue.dataset.evolve import evolutionary_algorithm
from noisyvalue.dataset.evolve import Individual
import numpy as np
from noisyvalue.dataset.evolve import fitness_noise
from noisyvalue.dataset.evolve import synthetic_release_disjoint
from noisyvalue.dataset.evolve import synthetic_release_hierarchical
from noisyvalue.dataset.evolve import synthetic_release_chaotic
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

num_runs = 20
total_generations = 100
num_elements_to_mutate = 1
genome_length = 15
num_parents = 50
num_children = 50
tournament_size = 10
num_tournament_winners = 5
release = synthetic_release_disjoint(n_vals=genome_length)

names = ["50 select 5", "50 select 20", "20 select 5", "20 select 20"]
t_sizes = [50, 50, 20, 20]
t_winners = [5, 20, 5, 20]
for i in range(4):
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

# genome of all 0s gives error "Expected all items to be NoisyValue"