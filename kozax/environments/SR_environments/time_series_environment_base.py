"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""

from jax.random import PRNGKey
import abc
from typing import Tuple, Any
from jaxtyping import Array

class EnvironmentBase(abc.ABC):
    """
    Abstract base class for time series environments in symbolic regression tasks.

    Parameters
    ----------
    n_var : int
        Number of variables in the state.

    Methods
    -------
    sample_init_states(batch_size, key)
        Samples initial states for the environment.
    drift(t, state, args)
        Computes the drift function for the environment.
    diffusion(t, state, args)
        Computes the diffusion function for the environment.
    terminate_event(state, **kwargs)
        Checks if the termination condition is met.
    """

    def __init__(self, n_var: int, process_noise: float) -> None:
        self.n_var = n_var
        self.process_noise = process_noise

    @abc.abstractmethod
    def sample_init_states(self, batch_size: int, key: PRNGKey) -> Any:
        """
        Samples initial states for the environment.

        Parameters
        ----------
        batch_size : int
            Number of initial states to sample.
        key : :class:`jax.random.PRNGKey`
            Random key for sampling.

        Returns
        -------
        Any
            Initial states.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def drift(self, t: float, state: Array, args: Tuple) -> Array:
        """
        Computes the drift function for the environment.

        Parameters
        ----------
        t : float
            Current time.
        state : :class:`jax.Array`
            Current state.
        args : tuple
            Additional arguments.

        Returns
        -------
        :class:`jax.Array`
            Drift.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def diffusion(self, t: float, state: Array, args: Tuple) -> Array:
        """
        Computes the diffusion function for the environment.

        Parameters
        ----------
        t : float
            Current time.
        state : :class:`jax.Array`
            Current state.
        args : tuple
            Additional arguments.

        Returns
        -------
        :class:`jax.Array`
            Diffusion.
        """
        raise NotImplementedError

    def terminate_event(self, state: Array, **kwargs) -> bool:
        """
        Checks if the termination condition is met.

        Parameters
        ----------
        state : :class:`jax.Array`
            Current state.
        kwargs : dict
            Additional arguments.

        Returns
        -------
        bool
            True if the termination condition is met, False otherwise.
        """
        return False