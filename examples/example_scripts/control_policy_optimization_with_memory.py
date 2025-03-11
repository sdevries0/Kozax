"""
Memory-based control policy optimization

In this more advanced example, we will optimize a symbolic control policy that is extended with a dynamic latent memory (check out this paper for more details: https://arxiv.org/abs/2406.02765).
The memory is defined by a set of differential equations, consisting of a tree for each latent unit. The memory updates every time step, and the control policy maps the memory to a control 
signal via an additional readout tree. This setup is applied to the partially observable acrobot swingup task, in which the angular velocity is hidden.
"""

# Specify the cores to use for XLA
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'

import jax
import jax.numpy as jnp
import diffrax
import jax.random as jr
import matplotlib.pyplot as plt
from jax import Array
from typing import Tuple, Callable
import copy

from kozax.genetic_programming import GeneticProgramming
from kozax.environments.control_environments.acrobot import Acrobot

"""
First we generate data, consisting of initial conditions, keys for noise, targets and parameters of the environment.
"""

def get_data(key, env, batch_size, dt, T, param_setting):
    init_key, noise_key2, param_key = jr.split(key, 3)
    x0, targets = env.sample_init_states(batch_size, init_key)
    obs_noise_keys = jr.split(noise_key2, batch_size)
    ts = jnp.arange(0, T, dt)

    params = env.sample_params(batch_size, param_setting, ts, param_key)
    return x0, ts, targets, obs_noise_keys, params

"""
The evaluator class parallelizes the simulation of the control loop over the batched data. In each trajectory, a coupled dynamical system is simulated using diffrax, integrating both the 
dynamic memory and the dynamic environment. At every time step, the state of the environment is mapped to an observations and the latent memory is mapped to a control signal. Afterwards, 
the state equation of the memory and environment state is computed. When the simulation is done, the fitness is computed given the control and environment state.
"""

class Evaluator:
    def __init__(self, env, state_size: int, dt0: float, solver=diffrax.Euler(), max_steps: int = 16**4, stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()) -> None:
        """Evaluates dynamic symbolic policies in control tasks.

        Args:
            env: Environment on which the candidate is evaluated.
            state_size: Dimensionality of the hidden state.
            dt0: Initial step size for integration.
            solver: Solver used for integration (default: diffrax.Euler()).
            max_steps: The maximum number of steps that can be used in integration (default: 16**4).
            stepsize_controller: Controller for the stepsize during integration (default: diffrax.ConstantStepSize()).

        Attributes:
            env: Environment on which the candidate is evaluated.
            max_fitness: Max fitness which is assigned when a trajectory returns an invalid value.
            state_size: Dimensionality of the hidden state.
            obs_size: Dimensionality of the observations.
            control_size: Dimensionality of the control.
            latent_size: Dimensionality of the state of the environment.
            dt0: Initial step size for integration.
            solver: Solver used for integration.
            max_steps: The maximum number of steps that can be used in integration.
            stepsize_controller: Controller for the stepsize during integration.
        """
        self.env = env
        self.max_fitness = 1e4
        self.state_size = state_size
        self.obs_size = env.n_obs
        self.control_size = env.n_control_inputs
        self.latent_size = env.n_var*env.n_dim
        self.dt0 = dt0
        self.solver = solver
        self.max_steps = max_steps
        self.stepsize_controller = stepsize_controller

    def __call__(self, candidate: Array, data: Tuple, tree_evaluator: Callable) -> float:
        """Evaluates the candidate on a task.

        Args:
            candidate: The coefficients of the candidate.
            data: The data required to evaluate the candidate.
            tree_evaluator: Function for evaluating trees.

        Returns:
            Fitness of the candidate.
        """
        _, _, _, _, fitness = jax.vmap(self.evaluate_trajectory, in_axes=[None, 0, None, 0, 0, 0, None])(candidate, *data, tree_evaluator)

        fitness = jnp.mean(fitness)
        return fitness
      
    def evaluate_trajectory(self, candidate: Array, x0: Array, ts: Array, target: float, obs_noise_key: jr.PRNGKey, params: Tuple, tree_evaluator: Callable) -> Tuple[Array, Array, Array, Array, float]:
        """Solves the coupled differential equation of the system and controller. 
        The differential equation of the system is defined in the environment and the differential equation 
        of the control is defined by the set of trees.

        Args:
            candidate: Candidate with trees for the hidden state and readout.
            x0: Initial state of the system.
            ts: Time points on which the controller is evaluated.
            target: Target position that the system should reach.
            obs_noise_key: Key to generate noisy observations.
            params: Parameters that define the environment.
            tree_evaluator: Function for evaluating trees.

        Returns:
            States, observations, control, activities of the hidden state of the candidate and the fitness of the candidate.
        """
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        state_equation = candidate[:self.state_size]
        readout = candidate[self.state_size:]
        
        solver = self.solver
        dt0 = self.dt0
        saveat = diffrax.SaveAt(ts=ts)

        # Concatenate the initial state of the system with the initial state of the latent memory
        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])

        system = diffrax.ODETerm(self._drift)
        
        # Solve the coupled system of the environment and the controller
        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, _x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=self.max_steps, event=diffrax.Event(self.env.cond_fn_nan), 
            args=(env, state_equation, readout, obs_noise_key, target, tree_evaluator), stepsize_controller=self.stepsize_controller, throw=False
        )

        xs = sol.ys[:,:self.latent_size]
        # Get observations of the state at every time step
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))

        activities = sol.ys[:,self.latent_size:]
        # Get control actions at every time step
        us = jax.vmap(lambda y, a, tar: tree_evaluator(readout, jnp.concatenate([y, a, jnp.zeros(self.control_size), target])), in_axes=[0,0,None])(ys, activities, target)

        # Compute the fitness of the candidate in this trajectory
        fitness = env.fitness_function(xs, us[:,None], target, ts)

        return xs, ys, us, activities, fitness
    
    def _drift(self, t, x_a, args):
        env, state_equation, readout, obs_noise_key, target, tree_evaluator = args
        x = x_a[:self.latent_size]
        a = x_a[self.latent_size:]

        _, y = env.f_obs(obs_noise_key, (t, x)) #Get observations from system
        u = tree_evaluator(readout, jnp.concatenate([jnp.zeros(self.obs_size), a, jnp.zeros(self.control_size), target])) #Compute control action from latent memory

        u = jnp.atleast_1d(u)

        dx = env.drift(t, x, u) #Apply control to system and get system change
        da = tree_evaluator(state_equation, jnp.concatenate([y, a, u, target])) #Compute change in latent memory

        return jnp.concatenate([dx, da])

"""
Here we define the hyperparameters, operators, variables and initialize the strategy. The control policy will consist of two latent variables and a readout layer.
`layer_sizes` allows us to define different types of tree, where `variable_list` contains different sets of input variables for each type of tree. 
The readout layer will only receive the latent states, while the inputs to the state equations consists of the observations, control signal and latent states.
"""

if __name__ == "__main__":
    key = jr.PRNGKey(0)
    gp_key, data_key = jr.split(key)
    batch_size = 8
    T = 40
    dt = 0.2
    param_setting = "Constant"

    env = Acrobot(n_obs = 2) # Partial observably Acrobot

    data = get_data(data_key, env, batch_size, dt, T, param_setting)
    
    #Define hyperparameters
    population_size = 100
    num_populations = 10
    num_generations = 50
    state_size = 2

    #Define expressions
    operator_list = [("+", lambda x, y: x + y, 2, 0.5), 
                    ("-", lambda x, y: x - y, 2, 0.1),
                    ("*", lambda x, y: x * y, 2, 0.5),
                    ("sin", lambda x: jnp.sin(x), 1, 0.1),
                    ("cos", lambda x: jnp.cos(x), 1, 0.1)]

    variable_list = [["x" + str(i) for i in range(env.n_obs)] + ["a1", "a2", "u"], ["a1", "a2"]]

    layer_sizes = jnp.array([state_size, env.n_control_inputs])

    #Define evaluator
    fitness_function = Evaluator(env, state_size, 0.05, solver=diffrax.Dopri5(), stepsize_controller=diffrax.PIDController(atol=1e-4, rtol=1e-4, dtmin=0.001), max_steps=1000)

    #Initialize strategy
    strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, layer_sizes, num_populations = num_populations, size_parsimony=0.003)

    strategy.fit(gp_key, data, verbose=True)

    """
    ## Visualize best solution
    """

    #Generate test_data
    data = get_data(jr.PRNGKey(10), env, 1, 0.01, T, param_setting)
    x0s, ts, _, _, params = data

    best_candidate = strategy.pareto_front[1][21]

    xs, ys, us, activities, fitness = jax.vmap(fitness_function.evaluate_trajectory, in_axes=[None, 0, None, 0, 0, 0, None])(best_candidate, *data, strategy.tree_evaluator)

    plt.plot(ts, -jnp.cos(xs[0,:,0]), color = f"C{0}", label="first link")
    plt.plot(ts, -jnp.cos(xs[0,:,0]) - jnp.cos(xs[0,:,0] + xs[0,:,1]), color = f"C{1}", label="second link")
    plt.hlines(1.5, ts[0], ts[-1], linestyles='dashed', color = "black")

    plt.legend(loc="best")
    plt.show()