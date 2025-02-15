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

from abc import ABC, abstractmethod
from typing import Tuple, Callable
from jaxtyping import Array

class BaseFitnessFunction(ABC):
    """
    Abstract base class for evaluating candidates in genetic programming.

    Methods
    -------
    __call__(candidate, data, tree_evaluator)
        Evaluates the candidate on a task.
    """

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