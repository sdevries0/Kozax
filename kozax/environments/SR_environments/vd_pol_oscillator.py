"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from kozax.environments.SR_environments.time_series_environment_base import EnvironmentBase
from jaxtyping import Array
from typing import Tuple

class VanDerPolOscillator(EnvironmentBase):
    """
    Van der Pol Oscillator environment for symbolic regression tasks.

    Parameters
    ----------
    process_noise : float, optional
        Standard deviation of the process noise. Default is 0.

    Attributes
    ----------
    init_mu : :class:`jax.Array`
        Mean of the initial state distribution.
    init_sd : :class:`jax.Array`
        Standard deviation of the initial state distribution.
    mu : float
        Parameter mu of the Van der Pol system.
    V : :class:`jax.Array`
        Process noise covariance matrix.

    Methods
    -------
    sample_init_states(batch_size, key)
        Samples initial states for the environment.
    drift(t, state, args)
        Computes the drift function for the environment.
    diffusion(t, state, args)
        Computes the diffusion function for the environment.
    """

    def __init__(self, process_noise: float = 0) -> None:
        n_var = 2
        super().__init__(n_var, process_noise)

        self.init_mu = jnp.array([0, 0])
        self.init_sd = jnp.array([1.0, 1.0])

        self.mu = 1
        self.V = self.process_noise * jnp.eye(self.n_var)

    def sample_init_states(self, batch_size: int, key: jrandom.PRNGKey) -> Array:
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
        :class:`jax.Array`
            Initial states.
        """
        return self.init_mu + self.init_sd * jrandom.normal(key, shape=(batch_size, 2))
    
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
        return jnp.array([state[1], self.mu * (1 - state[0]**2) * state[1] - state[0]])

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
        return self.V