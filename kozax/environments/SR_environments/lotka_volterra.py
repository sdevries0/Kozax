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

class LotkaVolterra(EnvironmentBase):
    """
    Lotka-Volterra environment for symbolic regression tasks.

    Parameters
    ----------
    process_noise : float, optional
        Standard deviation of the process noise. Default is 0.

    Attributes
    ----------
    init_mu : :class:`jax.Array`
        Mean of the initial state distribution.
    init_sd : float
        Standard deviation of the initial state distribution.
    alpha : float
        Parameter alpha of the Lotka-Volterra system.
    beta : float
        Parameter beta of the Lotka-Volterra system.
    delta : float
        Parameter delta of the Lotka-Volterra system.
    gamma : float
        Parameter gamma of the Lotka-Volterra system.
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

        self.init_mu = jnp.array([10, 10])
        self.init_sd = 2

        self.alpha = 1.1
        self.beta = 0.4
        self.delta = 0.1
        self.gamma = 0.4
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
        return jrandom.uniform(key, shape=(batch_size, 2), minval=5, maxval=15)
    
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
        return jnp.array([self.alpha * state[0] - self.beta * state[0] * state[1], self.delta * state[0] * state[1] - self.gamma * state[1]])

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