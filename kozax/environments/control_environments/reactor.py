"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""

import jax.numpy as jnp
import jax
import jax.random as jrandom
import diffrax
from kozax.environments.control_environments.control_environment_base import EnvironmentBase
from jaxtyping import Array
from typing import Tuple

class StirredTankReactor(EnvironmentBase):
    """
    Stirred Tank Reactor environment for control tasks.

    Parameters
    ----------
    process_noise : float
        Standard deviation of the process noise.
    obs_noise : float
        Standard deviation of the observation noise.
    n_obs : int, optional
        Number of observations. Default is 3.
    n_targets : int, optional
        Number of targets. Default is 1.
    max_control : :class:`jax.Array`, optional
        Maximum control values. Default is jnp.array([300]).
    external_f : callable, optional
        External influence function. Default is lambda t: 0.0.

    Attributes
    ----------
    n_var : int
        Number of variables in the state.
    n_control_inputs : int
        Number of control inputs.
    n_dim : int
        Number of dimensions.
    n_targets : int
        Number of targets.
    init_lower_bounds : :class:`jax.Array`
        Lower bounds for initial state sampling.
    init_upper_bounds : :class:`jax.Array`
        Upper bounds for initial state sampling.
    max_control : :class:`jax.Array`
        Maximum control values.
    Q : :class:`jax.Array`
        Process noise covariance matrix.
    r : :class:`jax.Array`
        Observation noise covariance matrix.
    external_f : callable
        External influence function.

    Methods
    -------
    initialize_parameters(params, ts)
        Initializes the parameters of the environment.
    sample_param_change(key, batch_size, ts, low, high)
        Samples parameter changes over time.
    sample_params(batch_size, mode, ts, key)
        Samples parameters for the environment.
    sample_init_states(batch_size, key)
        Samples initial states for the environment.
    drift(t, state, args)
        Computes the drift function for the environment.
    diffusion(t, state, args)
        Computes the diffusion function for the environment.
    fitness_function(state, control, targets, ts)
        Computes the fitness function for the environment.
    cond_fn_nan(t, y, args, **kwargs)
        Checks for NaN or infinite values in the state.
    """

    def __init__(self, process_noise: float = 0.0, obs_noise: float = 0.0, n_obs: int = 3, n_targets: int = 1, max_control: Array = jnp.array([300]), external_f: callable = lambda t: 0.0) -> None:
        if type(process_noise) == float:
            process_noise = [process_noise, process_noise]
        self.process_noise = process_noise
        self.obs_noise = obs_noise
        self.n_var = 3
        self.n_control_inputs = 1
        self.n_dim = 1
        self.n_targets = n_targets
        self.init_lower_bounds = jnp.array([275, 350, 0.5])
        self.init_upper_bounds = jnp.array([300, 375, 1.0])
        self.max_control = max_control
        super().__init__(process_noise, obs_noise, self.n_var, self.n_control_inputs, self.n_dim, n_obs)

        self.Q = jnp.array([[0, 0, 0], [0, 0.01, 0], [0, 0, 0]])
        self.r = jnp.array([[0.0001]])
        self.external_f = external_f

    def initialize_parameters(self, params: Tuple[Array, Array, Array, Array, Array, Array, Array, Array], ts: Array) -> None:
        """
        Initializes the parameters of the environment.

        Parameters
        ----------
        params : tuple of :class:`jax.Array`
            Parameters to initialize.
        ts : :class:`jax.Array`
            Time steps.
        """
        Vol, Cp, dHr, UA, q, Tf, Tcf, Volc = params
        self.Ea = 72750     # activation energy J/gmol
        self.R = 8.314      # gas constant J/gmol/K
        self.k0 = 7.2e10    # Arrhenius rate constant 1/min
        self.Vol = Vol      # Volume [L]
        self.Cp = Cp        # Heat capacity [J/g/K]
        self.dHr = dHr      # Enthalpy of reaction [J/mol]
        self.UA = UA        # Heat transfer [J/min/K]
        self.q = q          # Flowrate [L/min]
        self.Cf = 1.0       # Inlet feed concentration [mol/L]
        self.Tf = diffrax.LinearInterpolation(ts, Tf)  # Inlet feed temperature [K]
        self.Tcf = Tcf      # Coolant feed temperature [K]
        self.Volc = Volc    # Cooling jacket volume

        self.k = lambda T: self.k0 * jnp.exp(-self.Ea / self.R / T)

        self.G = jnp.eye(self.n_var) * jnp.array([6, 6, 0.05])
        self.process_noise_ts = diffrax.LinearInterpolation(ts, jnp.linspace(self.process_noise[0], self.process_noise[1], ts.shape[0]))

        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise * jnp.eye(self.n_obs) * (jnp.array([15, 15, 0.1])[:self.n_obs])

        self.max_control_ts = diffrax.LinearInterpolation(ts, jnp.hstack([mc * jnp.ones(int(ts.shape[0] // self.max_control.shape[0])) for mc in self.max_control]))
        self.external_influence = diffrax.LinearInterpolation(ts, jax.vmap(self.external_f)(ts))

    def sample_param_change(self, key: jrandom.PRNGKey, batch_size: int, ts: Array, low: float, high: float) -> Array:
        """
        Samples parameter changes over time.

        Parameters
        ----------
        key : :class:`jax.random.PRNGKey`
            Random key for sampling.
        batch_size : int
            Number of samples.
        ts : :class:`jax.Array`
            Time steps.
        low : float
            Lower bound for sampling.
        high : float
            Upper bound for sampling.

        Returns
        -------
        :class:`jax.Array`
            Sampled parameter values.
        """
        init_key, decay_key = jrandom.split(key)
        decay_factors = jrandom.uniform(decay_key, shape=(batch_size,), minval=1.01, maxval=1.02)
        init_values = jrandom.uniform(init_key, shape=(batch_size,), minval=low, maxval=high)
        values = jax.vmap(lambda v, d, t: v * (d ** t), in_axes=[0, 0, None])(init_values, decay_factors, ts)
        return values

    def sample_params(self, batch_size: int, mode: str, ts: Array, key: jrandom.PRNGKey) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        """
        Samples parameters for the environment.

        Parameters
        ----------
        batch_size : int
            Number of samples.
        mode : str
            Sampling mode. Options are "Constant", "Different", "Changing".
        ts : :class:`jax.Array`
            Time steps.
        key : :class:`jax.random.PRNGKey`
            Random key for sampling.

        Returns
        -------
        tuple of :class:`jax.Array`
            Sampled parameters.
        """
        if mode == "Constant":
            Vol = 100 * jnp.ones(batch_size)
            Cp = 239 * jnp.ones(batch_size)
            dHr = -5.0e4 * jnp.ones(batch_size)
            UA = 5.0e4 * jnp.ones(batch_size)
            q = 100 * jnp.ones(batch_size)
            Tf = 300 * jnp.ones((batch_size, ts.shape[0]))
            Tcf = 300 * jnp.ones(batch_size)
            Volc = 20.0 * jnp.ones(batch_size)
        elif mode == "Different":
            keys = jrandom.split(key, 8)
            Vol = jrandom.uniform(keys[0], (batch_size,), minval=75, maxval=150)
            Cp = jrandom.uniform(keys[1], (batch_size,), minval=200, maxval=350)
            dHr = jrandom.uniform(keys[2], (batch_size,), minval=-55000, maxval=-45000)
            UA = jrandom.uniform(keys[3], (batch_size,), minval=25000, maxval=75000)
            q = jrandom.uniform(keys[4], (batch_size,), minval=75, maxval=125)
            Tf = jnp.repeat(jrandom.uniform(keys[5], (batch_size,), minval=300, maxval=350)[:, None], ts.shape[0], axis=1)
            Tcf = jrandom.uniform(keys[6], (batch_size,), minval=250, maxval=300)
            Volc = jrandom.uniform(keys[7], (batch_size,), minval=10, maxval=30)
        elif mode == "Changing":
            keys = jrandom.split(key, 8)
            Vol = jrandom.uniform(keys[0], (batch_size,), minval=75, maxval=150)
            Cp = jrandom.uniform(keys[1], (batch_size,), minval=200, maxval=350)
            dHr = jrandom.uniform(keys[2], (batch_size,), minval=-55000, maxval=-45000)
            UA = jrandom.uniform(keys[3], (batch_size,), minval=25000, maxval=75000)
            q = jrandom.uniform(keys[4], (batch_size,), minval=75, maxval=125)
            Tf = self.sample_param_change(keys[5], batch_size, ts, 300, 350)
            Tcf = jrandom.uniform(keys[6], (batch_size,), minval=250, maxval=300)
            Volc = jrandom.uniform(keys[7], (batch_size,), minval=10, maxval=30)
        return Vol, Cp, dHr, UA, q, Tf, Tcf, Volc

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
        x0 = jrandom.uniform(init_key, shape=(batch_size, self.n_var), minval=self.init_lower_bounds, maxval=self.init_upper_bounds)
        targets = jrandom.uniform(target_key, shape=(batch_size, self.n_targets), minval=400, maxval=480)
        return x0, targets
    
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
        Tc, T, c = state
        control = jnp.squeeze(args)
        control = jnp.clip(control, 0, 300)

        dc = (self.q / self.Vol) * (self.Cf - c) - self.k(T) * c
        dT = (self.q / self.Vol) * (self.Tf.evaluate(t) - T) + (-self.dHr / self.Cp) * self.k(T) * c + (self.UA / self.Vol / self.Cp) * (Tc - T) + self.external_influence.evaluate(t)
        dTc = (control / self.Volc) * (self.Tcf - Tc) + (self.UA / self.Volc / self.Cp) * (T - Tc)

        return jnp.array([dTc, dT, dc])

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
        return self.process_noise_ts.evaluate(t) * self.G

    def fitness_function(self, state: Array, control: Array, targets: Array, ts: Array) -> float:
        """
        Computes the fitness function for the environment.

        Parameters
        ----------
        state : :class:`jax.Array`
            Current state.
        control : :class:`jax.Array`
            Control inputs.
        targets : :class:`jax.Array`
            Target states.
        ts : :class:`jax.Array`
            Time steps.

        Returns
        -------
        float
            Fitness value.
        """
        x_d = jax.vmap(lambda tar: jnp.array([0, tar, 0]))(targets)
        # x_d = jnp.array([0, targets[0], 0])
        print(state.shape, control.shape, x_d.shape)
        costs = jax.vmap(lambda _state, _u, _x_d: (_state - _x_d).T @ self.Q @ (_state - _x_d) + (_u) @ self.r @ (_u))(state, control, x_d)
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