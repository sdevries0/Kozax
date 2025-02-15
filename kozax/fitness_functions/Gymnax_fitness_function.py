"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import jax
from jaxtyping import Array
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple, Callable, Any
from kozax.fitness_functions.base_fitness_function import BaseFitnessFunction
import gymnax

class GymFitnessFunction(BaseFitnessFunction):
    """
    Evaluator for static symbolic policies in control tasks using Gym environments.

    Parameters
    ----------
    env_name : str
        Name of the Gym environment to be used.

    Attributes
    ----------
    env : gymnax environment
        The gymnax environment.
    env_params : dict
        Parameters for the gymnax environment.
    num_steps : int
        Number of steps in the environment for each episode.

    Methods
    -------
    __call__(candidate, keys, tree_evaluator)
        Evaluates the candidate on a task.
    evaluate_trajectory(candidate, key, tree_evaluator)
        Evaluates a rollout of the candidate in the environment.
    """

    def __init__(self, env_name: str) -> None:
        self.env, self.env_params = gymnax.make(env_name)
        self.num_steps = self.env_params.max_steps_in_episode

    def __call__(self, candidate: Array, keys: Array, tree_evaluator: Callable) -> float:
        """
        Evaluates the candidate on a task.

        Parameters
        ----------
        candidate : :class:`jax.Array`
            The candidate solution to be evaluated.
        keys : :class:`jax.Array`
            Random keys for evaluation.
        tree_evaluator : :class:`Callable`
            Function for evaluating trees.

        Returns
        -------
        float
            Fitness of the candidate.
        """
        reward = jax.vmap(self.evaluate_trajectory, in_axes=(None, 0, None))(candidate, keys, tree_evaluator)
        return jnp.mean(reward)
        
    def evaluate_trajectory(self, candidate: Array, key: jr.PRNGKey, tree_evaluator: Callable) -> Tuple[Array, float]:
        """
        Evaluates a rollout of the candidate in the environment.

        Parameters
        ----------
        candidate : :class:`jax.Array`
            The candidate solution to be evaluated.
        key : :class:`jax.random.PRNGKey`
            Random key for evaluation.
        tree_evaluator : :class:`Callable`
            Function for evaluating trees.

        Returns
        -------
        reward : float
            Total reward obtained during the trajectory.
        """
        key, subkey = jr.split(key)
        state, env_state = self.env.reset(subkey, self.env_params)

        def policy(state: Array) -> Array:
            """Symbolic policy."""
            a = tree_evaluator(candidate, state)
            return a

        def step_fn(carry: Tuple[Array, Any, jr.PRNGKey], _) -> Tuple[Tuple[Array, Any, jr.PRNGKey], Tuple[Array, float, bool]]:
            """Step function for lax.scan."""
            state, env_state, key = carry

            # Select action based on policy
            action = policy(state)

            # Step the environment
            key, subkey = jr.split(key)
            next_state, next_env_state, reward, done, _ = self.env.step(
                subkey, env_state, action, self.env_params
            )

            return (next_state, next_env_state, key), (state, reward, done)

        # Run the rollout using lax.scan
        (final_carry, (states, rewards, dones)) = jax.lax.scan(
            step_fn, (state, env_state, key), None, length=self.num_steps
        )
        
        return -jnp.sum(rewards)