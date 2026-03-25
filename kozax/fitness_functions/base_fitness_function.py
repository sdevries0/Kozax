"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Callable
from jaxtyping import Array

class BaseFitnessFunction(ABC):
    """
    Abstract base class for evaluating candidates in genetic programming.

    Attributes
    ----------
    n_objectives : int
        Number of objectives for multi-objective optimization. Default is 1.
        Can be overridden in subclasses by setting as a class attribute.

    Methods
    -------
    __call__(candidate, data, tree_evaluator)
        Evaluates the candidate on a task.
    """
    
    n_objectives: int = 1

    @abstractmethod
    def __call__(self, candidate: Array, data: Tuple, tree_evaluator: Callable) -> float:
        """
        Evaluates the candidate on a task.

        Parameters
        ----------
        candidate : :class:`jax.Array`
            The candidate solution to be evaluated.
        data : :class:`tuple`
            The data required to evaluate the candidate.
        tree_evaluator : :class:`Callable`
            Function for evaluating trees.

        Returns
        -------
        float
            Fitness of the candidate.
        """
        raise NotImplementedError