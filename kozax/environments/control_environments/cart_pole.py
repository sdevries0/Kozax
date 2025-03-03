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

class CartPole(EnvironmentBase):
    """
    CartPole environment for control tasks.

    Parameters
    ----------
    process_noise : float
        Standard deviation of the process noise.
    obs_noise : float
        Standard deviation of the observation noise.
    n_obs : int, optional
        Number of observations. Default is 4.

    Attributes
    ----------
    n_var : int
        Number of variables in the state.
    n_control_inputs : int
        Number of control inputs.
    n_targets : int
        Number of targets.
    n_dim : int
        Number of dimensions.
    init_bounds : :class:`jax.Array`
        Bounds for initial state sampling.
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
    terminate_event(state, **kwargs)
        Checks if the termination condition is met.
    """

    def __init__(self, process_noise: float = 0.0, obs_noise: float = 0.0, n_obs: int = 4) -> None:
        self.n_var = 4
        self.n_control_inputs = 1
        self.n_targets = 0
        self.n_dim = 1
        self.init_bounds = jnp.array([0.05, 0.05, 0.05, 0.05])
        super().__init__(process_noise, obs_noise, self.n_var, self.n_control_inputs, self.n_dim, n_obs)

        self.Q = jnp.array(0)
        self.R = jnp.array([[0.0]])

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
        x0 = jrandom.uniform(init_key, shape=(batch_size, self.n_var), minval=-self.init_bounds, maxval=self.init_bounds)
        targets = jnp.zeros((batch_size, self.n_targets))
        return x0, targets
    
    def sample_params(self, batch_size: int, mode: str, ts: Array, key: jrandom.PRNGKey) -> Array:
        """
        Samples parameters for the environment.

        Parameters
        ----------
        batch_size : int
            Number of parameters to sample.
        mode : str
            Mode for sampling parameters.
        ts : :class:`jax.Array`
            Time steps.
        key : :class:`jax.random.PRNGKey`
            Random key for sampling.

        Returns
        -------
        :class:`jax.Array`
            Sampled parameters.
        """
        return jnp.zeros((batch_size))

    def initialize_parameters(self, params: Array, ts: Array) -> None:
        """
        Initializes the parameters of the environment.

        Parameters
        ----------
        params : :class:`jax.Array`
            Parameters to initialize.
        ts : :class:`jax.Array`
            Time steps.
        """
        _ = params
        self.g = 9.81
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.cart_mass = 1
        
        self.G = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
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
        control = jnp.squeeze(args)
        control = jnp.clip(control, -1, 1)
        x, theta, x_dot, theta_dot = state

        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        
        theta_acc = (self.g * sin_theta - cos_theta * (
            control + self.pole_mass * self.pole_length * theta_dot**2 * sin_theta
            ) / (self.cart_mass + self.pole_mass)) / (
                self.pole_length * (4/3 - (self.pole_mass * cos_theta**2) / (self.cart_mass + self.pole_mass)))

        x_acc = (control + self.pole_mass * self.pole_length * (theta_dot**2 * sin_theta - theta_acc * cos_theta)) / (self.cart_mass + self.pole_mass)

        return jnp.array([
            x_dot,
            theta_dot,
            x_acc,
            theta_acc
        ])
    
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
        invalid_points = jax.vmap(lambda _x, _u: jnp.any(jnp.isinf(_x)) + jnp.isnan(_u))(state, control[:, 0])
        punishment = jnp.ones_like(invalid_points)

        costs = jnp.where(invalid_points, punishment, jnp.zeros_like(punishment))

        return jnp.sum(costs)
    
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
        return (jnp.abs(state[1]) > 0.2) | (jnp.abs(state[0]) > 4.8) | jnp.any(jnp.isnan(state)) | jnp.any(jnp.isinf(state))