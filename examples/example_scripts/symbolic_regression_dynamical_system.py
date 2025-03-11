"""
# Symbolic regression of a dynamical system

In this example, Kozax is applied to recover the state equations of the Lotka-Volterra system. The candidate solutions are integrated as a system of differential equations, after which the 
predictions are compared to the true observations to determine a fitness score.
"""

# Specify the cores to use for XLA
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'

import jax
import diffrax
import jax.numpy as jnp
import jax.random as jr
import diffrax

from kozax.genetic_programming import GeneticProgramming
from kozax.fitness_functions.ODE_fitness_function import ODEFitnessFunction
from kozax.environments.SR_environments.lotka_volterra import LotkaVolterra

"""
First the data is generated, consisting of initial conditions, time points and the true observations. Kozax provides the Lotka-Volterra environment, which is integrated with Diffrax.
"""

def get_data(key, env, dt, T, batch_size=20):
    x0s = env.sample_init_states(batch_size, key)
    ts = jnp.arange(0, T, dt)

    def solve(env, ts, x0):
        solver = diffrax.Dopri5()
        dt0 = 0.001
        saveat = diffrax.SaveAt(ts=ts)

        system = diffrax.ODETerm(env.drift)

        # Solve the system given an initial conditions
        sol = diffrax.diffeqsolve(system, solver, ts[0], ts[-1], dt0, x0, saveat=saveat, max_steps=500, 
                                  adjoint=diffrax.DirectAdjoint(), stepsize_controller=diffrax.PIDController(atol=1e-7, rtol=1e-7, dtmin=0.001))
        
        return sol.ys

    ys = jax.vmap(solve, in_axes=[None, None, 0])(env, ts, x0s) #Parallelize over the batch dimension
    
    return x0s, ts, ys

"""
For the fitness function, we used the ODEFitnessFunction that uses Diffrax to integrate candidate solutions. It is possible to select the solver, time step, number of steps and a 
stepsize controller to balance efficiency and accuracy. To ensure convergence of the genetic programming algorithm, constant optimization is applied to the best candidates at every 
generation. The constant optimization is performed with a couple of simple evolutionary steps that adjust the values of the constants in a candidate. The hyperparameters that define the 
constant optimization are `constant_optimization_N_offspring`, `constant_optimization_steps`, `optimize_constants_elite` and `constant_step_size_init`.
"""

if __name__ == "__main__":
    key = jr.PRNGKey(0)
    data_key, gp_key = jr.split(key)

    T = 30
    dt = 0.2
    env = LotkaVolterra()

    # Simulate the data
    data = get_data(data_key, env, dt, T, batch_size=4)
    x0s, ts, ys = data

    #Define the nodes and hyperparameters
    operator_list = [
            ("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
            ("-", lambda x, y: jnp.subtract(x, y), 2, 0.1), 
            ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5), 
        ]

    variable_list = [["x" + str(i) for i in range(env.n_var)]]
    layer_sizes = jnp.array([env.n_var])

    population_size = 100
    num_populations = 10
    num_generations = 50

    #Initialize the fitness function and the genetic programming strategy
    fitness_function = ODEFitnessFunction(solver=diffrax.Dopri5(), dt0 = 0.01, stepsize_controller=diffrax.PIDController(atol=1e-6, rtol=1e-6, dtmin=0.001), max_steps=300)

    strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, layer_sizes, num_populations = num_populations,
                        size_parsimony=0.003, constant_optimization_method="evolution", constant_optimization_N_offspring = 25, constant_optimization_steps = 5, 
                        optimize_constants_elite=100, constant_step_size_init=0.1, constant_step_size_decay=0.99)
    """
    Kozax provides a fit function that receives the data and a random key. However, it is also possible to run Kozax with an easy loop consisting of evaluating and evolving. 
    This is useful as different input data can be provided during evaluation. In symbolic regression of dynamical systems, it helps to first optimize on a small part of the time points, 
    and provide the full data trajectories only after a couple of generations.    
    """

    # Sample the initial population
    population = strategy.initialize_population(gp_key)

    # Define the number of timepoints to include in the data
    end_ts = int(ts.shape[0]/2)

    for g in range(num_generations):
        if g == 25: # After 25 generations, use the full data
            end_ts = ts.shape[0]

        key, eval_key, sample_key = jr.split(key, 3)
        # Evaluate the population on the data, and return the fitness
        fitness, population = strategy.evaluate_population(population, (x0s, ts[:end_ts], ys[:,:end_ts]), eval_key)

        # Print the best solution in the population in this generation
        best_fitness, best_solution = strategy.get_statistics(g)
        print(f"In generation {g+1}, best fitness = {best_fitness:.4f}, best solution = {strategy.expression_to_string(best_solution)}")

        # Evolve the population until the last generation. The fitness should be given to the evolve function.
        if g < (num_generations-1):
            population = strategy.evolve_population(population, fitness, sample_key)

    strategy.print_pareto_front()

    """
    Instead of using evolution to optimize the constants, Kozax also offers gradient-based optimization. For gradient optimization, it is possible to specify the optimizer, the number of 
    candidates to apply constant optimization to, the initial learning rate and the learning rate decay over generation. These two methods are provided as either can be more effective or 
    efficient for different problems.
    """

    import optax

    strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, layer_sizes, num_populations = num_populations,
                        size_parsimony=0.003, constant_optimization_method="gradient", constant_optimization_steps = 15, optimizer_class = optax.adam,
                        optimize_constants_elite=100, constant_step_size_init=0.025, constant_step_size_decay=0.95)

    key = jr.PRNGKey(0)
    data_key, gp_key = jr.split(key)

    T = 30
    dt = 0.2
    env = LotkaVolterra()

    # Simulate the data
    data = get_data(data_key, env, dt, T, batch_size=4)
    x0s, ts, ys = data

    # Sample the initial population
    population = strategy.initialize_population(gp_key)

    # Define the number of timepoints to include in the data
    end_ts = int(ts.shape[0]/2)

    for g in range(num_generations):
        if g == 25: # After 25 generations, use the full data
            end_ts = ts.shape[0]

        key, eval_key, sample_key = jr.split(key, 3)
        # Evaluate the population on the data, and return the fitness
        fitness, population = strategy.evaluate_population(population, (x0s, ts[:end_ts], ys[:,:end_ts]), eval_key)

        # Print the best solution in the population in this generation
        best_fitness, best_solution = strategy.get_statistics(g)
        print(f"In generation {g+1}, best fitness = {best_fitness:.4f}, best solution = {strategy.expression_to_string(best_solution)}")

        # Evolve the population until the last generation. The fitness should be given to the evolve function.
        if g < (num_generations-1):
            population = strategy.evolve_population(population, fitness, sample_key)

    strategy.print_pareto_front()