"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""

import jax
from typing import Tuple, Callable
from jaxtyping import Array
import jax.numpy as jnp
from kozax.fitness_functions.base_fitness_function import BaseFitnessFunction

class SymbolicRegressionFitnessFunction(BaseFitnessFunction):
    """
    Evaluator for candidates on symbolic regression tasks with x, y data.

    Methods
    -------
    __call__(candidate, data, tree_evaluator)
        Evaluates the candidate on a task.
    """

    def __call__(self, candidate: Array, data: Tuple[Array, Array], tree_evaluator: Callable) -> float:
        """
        Evaluates the candidate on a task.

        Parameters
        ----------
        candidate : :class:`jax.Array`
            The candidate solution to be evaluated.
        data : :class:`tuple` of :class:`jax.Array`
            The data required to evaluate the candidate. Tuple of (x, y) where x is the input data and y is the true output data.
        tree_evaluator : :class:`Callable`
            Function for evaluating trees.

        Returns
        -------
        float
            Fitness of the candidate.
        """
        x, y = data
        pred = jax.vmap(tree_evaluator, in_axes=[None, 0])(candidate, x)
        return jnp.mean(jnp.square(pred - y))