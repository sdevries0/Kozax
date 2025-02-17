"""
Control policy optimization

In this example, a symbolic policy is evolved for the pendulum swingup task. Gymnax is used for simulation of the pendulum environment, showing that Kozax can easily be extended to external libraries.
"""

# Specify the cores to use for XLA
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'
import jax
import jax.numpy as jnp
import jax.random as jr
import gymnax
import matplotlib.pyplot as plt

from kozax.genetic_programming import GeneticProgramming
from kozax.fitness_functions.Gymnax_fitness_function import GymFitnessFunction

if __name__ == "__main__":
    """
    Kozax provides a simple fitness function for Gymnax environments, which is used in this example.
    """

    #Define hyperparameters
    population_size = 100
    num_populations = 5
    num_generations = 50
    batch_size = 16

    fitness_function = GymFitnessFunction("Pendulum-v1")

    #Define operators and variables
    operator_list = [
        ("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
        ("-", lambda x, y: jnp.subtract(x, y), 2, 0.1), 
        ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5), 
        ]

    variable_list = [[f"y{i}" for i in range(fitness_function.env.observation_space(fitness_function.env_params).shape[0])]]

    #Initialize strategy
    strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, num_populations=num_populations)

    key = jr.PRNGKey(0)
    data_key, gp_key = jr.split(key, 2)

    # The data comprises keys need to initialize the batch of environments.
    batch_keys = jr.split(data_key, batch_size)

    strategy.fit(gp_key, batch_keys, verbose=True)

    """
    ## Visualize best solution

    We can visualize the sin and cos position in a trajectory using the best solution.
    """

    env, env_params = gymnax.make('Pendulum-v1')
    key = jr.PRNGKey(2)
    obs, env_state = env.reset(key)
    all_obs = []
    treward = []
    actions = []

    done = False

    sin = jnp.sin
    cos = jnp.cos

    T=199
    for t in range(T):

        y0, y1, y2 = obs
        action = 2.92*y0*(-6.56*y1 - 1.29*y2)
        obs, env_state, reward, done, _ = env.step(
                    jr.fold_in(key, t), env_state, action, env_params
                )
        all_obs.append(obs)
        treward.append(reward)
        actions.append(action)

    all_obs = jnp.array(all_obs)
    plt.plot(all_obs[:,0], label='cos(x)')
    plt.plot(all_obs[:,1], label='sin(x)')
    plt.legend()
    plt.show()