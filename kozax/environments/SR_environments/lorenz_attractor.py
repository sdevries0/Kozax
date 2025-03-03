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

class LorenzAttractor(EnvironmentBase):
    """
    Lorenz Attractor environment for symbolic regression tasks.

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
    sigma : float
        Parameter sigma of the Lorenz system.
    rho : float
        Parameter rho of the Lorenz system.
    beta : float
        Parameter beta of the Lorenz system.
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
        n_var = 3
        super().__init__(n_var, process_noise)

        self.init_mu = jnp.array([1, 1, 1])
        self.init_sd = 0.1

        self.sigma = 10
        self.rho = 28
        self.beta = 8 / 3
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
        return self.init_mu + self.init_sd * jrandom.normal(key, shape=(batch_size, 3))

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
        return jnp.array([self.sigma * (state[1] - state[0]), state[0] * (self.rho - state[2]) - state[1], state[0] * state[1] - self.beta * state[2]])

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

class Lorenz96(EnvironmentBase):
    """
    Lorenz 96 environment for symbolic regression tasks.

    Parameters
    ----------
    n_dim : int, optional
        Number of dimensions. Default is 3.
    process_noise : float, optional
        Standard deviation of the process noise. Default is 0.

    Attributes
    ----------
    F : float
        Forcing term.
    init_mu : :class:`jax.Array`
        Mean of the initial state distribution.
    init_sd : float
        Standard deviation of the initial state distribution.
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

    def __init__(self, n_dim: int = 3, process_noise: float = 0) -> None:
        n_var = n_dim
        super().__init__(n_var, process_noise)

        self.F = 8
        self.init_mu = jnp.ones(self.n_var) * self.F
        self.init_sd = 0.1

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
        return self.init_mu + self.init_sd * jrandom.normal(key, shape=(batch_size, self.n_var))

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
        f = lambda x_cur, x_next, x_prev1, x_prev2: (x_next - x_prev2) * x_prev1 - x_cur + self.F
        return jax.vmap(f)(state, jnp.roll(state, -1), jnp.roll(state, 1), jnp.roll(state, 2))

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