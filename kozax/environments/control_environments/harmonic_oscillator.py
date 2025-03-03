"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from kozax.environments.control_environments.control_environment_base import EnvironmentBase
from jaxtyping import Array
from typing import Tuple

class HarmonicOscillator(EnvironmentBase):
    """
    Harmonic Oscillator environment for control tasks.

    Parameters
    ----------
    process_noise : float
        Standard deviation of the process noise.
    obs_noise : float
        Standard deviation of the observation noise.
    n_obs : int, optional
        Number of observations. Default is 2.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    n_var : int
        Number of variables in the state.
    n_control_inputs : int
        Number of control inputs.
    n_targets : int
        Number of targets.
    mu0 : :class:`jax.Array`
        Mean of the initial state distribution.
    P0 : :class:`jax.Array`
        Covariance matrix of the initial state distribution.
    q : float
        Process noise parameter.
    r : float
        Observation noise parameter.
    Q : :class:`jax.Array`
        Process noise covariance matrix.
    R : :class:`jax.Array`
        Observation noise covariance matrix.

    Methods
    -------
    sample_init_states(batch_size, key)
        Samples initial states for the environment.
    sample_params(batch_size, mode, ts, key)
        Samples parameters for the environment.
    initialize_parameters(params, ts)
        Initializes the parameters of the environment.
    drift(t, state, args)
        Computes the drift function for the environment.
    diffusion(t, state, args)
        Computes the diffusion function for the environment.
    fitness_function(state, control, target, ts)
        Computes the fitness function for the environment.
    cond_fn_nan(t, y, args, **kwargs)
        Checks for NaN or infinite values in the state.
    """

    def __init__(self, process_noise: float = 0.0, obs_noise: float = 0.0, n_obs: int = 2) -> None:
        self.n_dim = 1
        self.n_var = 2
        self.n_control_inputs = 1
        self.n_targets = 1
        self.mu0 = jnp.zeros(self.n_var)
        self.P0 = jnp.eye(self.n_var) * jnp.array([3.0, 1.0])
        super().__init__(process_noise, obs_noise, self.n_var, self.n_control_inputs, self.n_dim, n_obs)

        self.q = self.r = 0.5
        self.Q = jnp.array([[self.q, 0], [0, 0]])
        self.R = jnp.array([[self.r]])

    def sample_init_states(self, batch_size: int, key: jrandom.PRNGKey) -> Tuple[Array, Array]:
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
        x0 : :class:`jax.Array`
            Initial states.
        targets : :class:`jax.Array`
            Target states.
        """
        init_key, target_key = jrandom.split(key)
        x0 = self.mu0 + jrandom.normal(init_key, shape=(batch_size, self.n_var)) @ self.P0
        targets = jrandom.uniform(target_key, shape=(batch_size, self.n_targets), minval=-3, maxval=3)
        return x0, targets
    
    def sample_params(self, batch_size: int, mode: str, ts: Array, key: jrandom.PRNGKey) -> Tuple[Array, Array]:
        """
        Samples parameters for the environment.

        Parameters
        ----------
        batch_size : int
            Number of parameters to sample.
        mode : str
            Mode for sampling parameters. Options are "Constant", "Different", "Changing".
        ts : :class:`jax.Array`
            Time steps.
        key : :class:`jax.random.PRNGKey`
            Random key for sampling.

        Returns
        -------
        omegas : :class:`jax.Array`
            Sampled omega parameters.
        zetas : :class:`jax.Array`
            Sampled zeta parameters.
        """
        omega_key, zeta_key, args_key = jrandom.split(key, 3)
        if mode == "Constant":
            omegas = jnp.ones((batch_size))
            zetas = jnp.zeros((batch_size))
        elif mode == "Different":
            omegas = jrandom.uniform(omega_key, shape=(batch_size,), minval=0.0, maxval=2.0)
            zetas = jrandom.uniform(zeta_key, shape=(batch_size,), minval=0.0, maxval=1.5)
        elif mode == "Changing":
            decay_factors = jrandom.uniform(args_key, shape=(batch_size, 2), minval=0.98, maxval=1.02)
            init_omegas = jrandom.uniform(omega_key, shape=(batch_size,), minval=0.5, maxval=1.5)
            init_zetas = jrandom.uniform(zeta_key, shape=(batch_size,), minval=0.0, maxval=1.0)
            omegas = jax.vmap(lambda o, d, t: o * (d ** t), in_axes=[0, 0, None])(init_omegas, decay_factors[:, 0], ts)
            zetas = jax.vmap(lambda z, d, t: z * (d ** t), in_axes=[0, 0, None])(init_zetas, decay_factors[:, 1], ts)
        return omegas, zetas

    def initialize_parameters(self, params: Tuple[Array, Array], ts: Array) -> None:
        """
        Initializes the parameters of the environment.

        Parameters
        ----------
        params : tuple of :class:`jax.Array`
            Parameters to initialize.
        ts : :class:`jax.Array`
            Time steps.
        """
        omega, zeta = params
        self.A = jnp.array([[0, 1], [-omega, -zeta]])
        self.b = jnp.array([[0.0, 1.0]]).T
        self.G = jnp.array([[0, 0], [0, 1]])
        self.V = self.process_noise * self.G
        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise * jnp.eye(self.n_obs)

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
        return self.A @ state + self.b @ args
    
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
    
    def fitness_function(self, state: Array, control: Array, target: Array, ts: Array) -> float:
        """
        Computes the fitness function for the environment.

        Parameters
        ----------
        state : :class:`jax.Array`
            Current state.
        control : :class:`jax.Array`
            Control inputs.
        target : :class:`jax.Array`
            Target states.
        ts : :class:`jax.Array`
            Time steps.

        Returns
        -------
        float
            Fitness value.
        """
        x_d = jnp.array([jnp.squeeze(target), 0])
        u_d = -jnp.linalg.pinv(self.b) @ self.A @ x_d
        costs = jax.vmap(lambda _state, _u: (_state - x_d).T @ self.Q @ (_state - x_d) + (_u - u_d) @ self.R @ (_u - u_d))(state, control)
        return jnp.sum(costs)
    
    def cond_fn_nan(self, t: float, y: Array, args: Tuple, **kwargs) -> float:
        """
        Checks for NaN or infinite values in the state.

        Parameters
        ----------
        t : float
            Current time.
        y : :class:`jax.Array`
            Current state.
        args : tuple
            Additional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        float
            -1.0 if NaN or infinite values are found, 1.0 otherwise.
        """
        return jnp.where(jnp.any(jnp.isinf(y) + jnp.isnan(y)), -1.0, 1.0)