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

class Acrobot(EnvironmentBase):
    """
    Acrobot environment for control tasks.

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
    n_control : int
        Number of control inputs.
    n_targets : int
        Number of targets.
    n_dim : int
        Number of dimensions.
    init_bounds : :class:`jax.Array`
        Bounds for initial state sampling.
    R : :class:`jax.Array`
        Control cost matrix.

    Methods
    -------
    sample_init_states(batch_size, key)
        Samples initial states for the environment.
    sample_params(batch_size, mode, ts, key)
        Samples parameters for the environment.
    f_obs(key, t_x)
        Computes the observation function.
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

    def __init__(self, process_noise: float = 0.0, obs_noise: float = 0.0, n_obs: int = 4) -> None:
        self.n_var = 4
        self.n_control_inputs = 1
        self.n_targets = 0
        self.n_dim = 1
        self.init_bounds = jnp.array([0.1, 0.1, 0.1, 0.1])
        super().__init__(process_noise, obs_noise, self.n_var, self.n_control_inputs, self.n_dim, n_obs)

        self.R = 0.1 * jnp.eye(self.n_control)

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
    
    def sample_params(self, batch_size: int, mode: str, ts: Array, key: jrandom.PRNGKey) -> Tuple[Array, Array, Array, Array]:
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
        tuple of :class:`jax.Array`
            Sampled parameters.
        """
        l1 = l2 = m1 = m2 = jnp.ones((batch_size))
        return l1, l2, m1, m2
    
    def f_obs(self, key: jrandom.PRNGKey, t_x: Tuple[float, Array]) -> Tuple[jrandom.PRNGKey, Array]:
        """
        Computes the observation function.

        Parameters
        ----------
        key : :class:`jax.random.PRNGKey`
            Random key for sampling noise.
        t_x : tuple of (float, :class:`jax.Array`)
            Tuple containing the current time and state.

        Returns
        -------
        key : :class:`jax.random.PRNGKey`
            Updated random key.
        out : :class:`jax.Array`
            Observation.
        """
        _, out = super().f_obs(key, t_x)
        out = jnp.array([(out[0] + jnp.pi) % (2 * jnp.pi) - jnp.pi, (out[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi, out[2], out[3]])[:self.n_obs]
        return key, out

    def initialize_parameters(self, params: Tuple[Array, Array, Array, Array], ts: Array) -> None:
        """
        Initializes the parameters of the environment.

        Parameters
        ----------
        params : tuple of :class:`jax.Array`
            Parameters to initialize.
        ts : :class:`jax.Array`
            Time steps.
        """
        l1, l2, m1, m2 = params
        self.l1 = l1  # [m]
        self.l2 = l2  # [m]
        self.m1 = m1  #: [kg] mass of link 1
        self.m2 = m2  #: [kg] mass of link 2
        self.lc1 = 0.5 * self.l1  #: [m] position of the center of mass of link 1
        self.lc2 = 0.5 * self.l2  #: [m] position of the center of mass of link 2
        self.moi1 = self.moi2 = 1.0
        self.g = 9.81

        self.G = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
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
        theta1, theta2, theta1_dot, theta2_dot = state

        d1 = self.m1 * self.lc1**2 + self.m2 * (self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * jnp.cos(theta2)) + self.moi1 + self.moi2
        d2 = self.m2 * (self.lc2**2 + self.l1 * self.lc2 * jnp.cos(theta2)) + self.moi2

        phi2 = self.m2 * self.lc2 * self.g * jnp.cos(theta1 + theta2 - jnp.pi/2)
        phi1 = -self.m2 * self.l1 * self.lc2 * theta2_dot**2 * jnp.sin(theta2) - 2 * self.m2 * self.l1 * self.lc2 * theta1_dot * theta2_dot * jnp.sin(theta1) \
                    + (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * jnp.cos(theta1 - jnp.pi/2) + phi2
        
        if self.n_control == 1:
            theta2_acc = (control + d2/d1 * phi1 - self.m2 * self.l1 * self.lc2 * theta1_dot**2 * jnp.sin(theta2) - phi2) \
                        / (self.m2 * self.lc2**2 + self.moi2 - d2**2 / d1)
            theta1_acc = -(d2 * theta2_acc + phi1)/d1
        else:
            c1, c2 = control
            theta2_acc = (c1 + d2/d1 * phi1 - self.m2 * self.l1 * self.lc2 * theta1_dot**2 * jnp.sin(theta2) - phi2) \
                        / (self.m2 * self.lc2**2 + self.moi2 - d2**2 / d1)
            theta1_acc = (c2 - d2 * theta2_acc - phi1)/d1

        return jnp.array([
            theta1_dot,
            theta2_dot,
            theta1_acc,
            theta2_acc
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
        reached_threshold = jax.vmap(lambda theta1, theta2: -jnp.cos(theta1) - jnp.cos(theta1 + theta2) > 1.5)(state[:,0], state[:,1])
        first_success = jnp.argmax(reached_threshold)

        control = jnp.clip(control, -1, 1)

        control_cost = jax.vmap(lambda _state, _u: _u @ self.R @ _u)(state, control)
        costs = jnp.where((ts / (ts[1] - ts[0])) > first_success, jnp.zeros_like(control_cost), control_cost)

        return (first_success + (first_success == 0) * ts.shape[0] + jnp.sum(costs))
    
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
        return jnp.where((jnp.abs(y[2]) > (4 * jnp.pi)) | (jnp.abs(y[3]) > (9 * jnp.pi)) | jnp.any(jnp.isnan(y)) | jnp.any(jnp.isinf(y)), -1.0, 1.0)