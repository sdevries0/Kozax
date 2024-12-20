"""
Kozax: Genetic programming framework in JAX

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

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jrandom
import diffrax

from typing import Tuple, Callable
import copy

class Evaluator:
    def __init__(self, env, dt0: float, solver=diffrax.Euler(), max_steps: int = 16**4, lag_size: int = 1, stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()) -> None:
        """Evaluator for static symbolic policies in control tasks

        Attributes:
            env: Environment on which the candidate is evaluated
            max_fitness: Max fitness which is assigned when a trajectory returns an invalid value
            state_size: Dimensionality of the hidden state
            obs_size: Dimensionality of the observations
            control_size: Dimensionality of the control
            latent_size: Dimensionality of the state of the environment
            dt0: Initial step size for integration
            solver: Solver used for integration
            max_steps: The maximum number of steps that can be used in integration
            stepsize_controller: Controller for the stepsize during integration
        """
        self.env = env
        self.max_fitness = 1e6
        self.obs_size = env.n_obs
        self.control_size = env.n_control
        self.latent_size = env.n_var*env.n_dim
        self.lag_size = lag_size
        self.dt0 = dt0
        self.solver = solver
        self.max_steps = max_steps
        self.stepsize_controller = stepsize_controller

    def __call__(self, coefficients: Array, nodes: Array, data: Tuple, tree_evaluator: Callable) -> float:
        """Evaluates the candidate on a task

        :param coefficients: The coefficients of the candidate
        :param nodes: The nodes and index references of the candidate
        :param data: The data required to evaluate the candidate
        :param tree_evaluator: Function for evaluating trees

        Returns: Fitness of the candidate
        """
        _, _, _, fitness = self.evaluate_candidate(jnp.concatenate([nodes, coefficients], axis=-1), data, tree_evaluator)

        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
        fitness = jnp.mean(fitness)
        return jnp.clip(fitness,0,self.max_fitness)
    
    def evaluate_candidate(self, candidate: Array, data: Tuple, eval) -> Tuple[Array, Array, Array, float]:
        """Evaluates a candidate given a task and data

        :param candidate: Candidate that is evaluated
        :param data: The data required to evaluate the candidate
        :param tree_evaluator: Function for evaluating trees
        
        Returns: Predictions and fitness of the candidate
        """
        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0, 0, None])(candidate, *data, eval)
    
    def evaluate_control_loop(self, candidate: Array, x0: Array, ts: Array, target: Array, process_noise_key: jrandom.PRNGKey, obs_noise_key: jrandom.PRNGKey, params: Tuple, tree_evaluator: Callable) -> Tuple[Array, Array, Array, float]:
        """Solves the coupled differential equation of the system and controller. The differential equation of the system is defined in the environment and the differential equation 
        of the control is defined by the set of trees

        :param candidate: Candidate with trees for the hidden state and readout
        :param x0: Initial state of the system
        :param ts: time points on which the controller is evaluated
        :param target: Target position that the system should reach
        :param process_noise_key: Key to generate process noise
        :param obs_noise_key: Key to generate noisy observations
        :param params: Parameters that define the environment
        :param tree_evaluator: Function for evaluating trees

        Returns: States, observations, control, activities of the hidden state of the candidate and the fitness of the candidate.
        """
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        policy = candidate
        
        solver = self.solver
        dt0 = self.dt0
        saveat = diffrax.SaveAt(ts=ts)
        # solve_ts = jnp.arange(0,50,dt0)

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.latent_size,), key=process_noise_key, levy_area=diffrax.SpaceTimeLevyArea) #define process noise
        system = diffrax.MultiTerm(diffrax.ODETerm(self._drift), diffrax.ControlTerm(self._diffusion, brownian_motion))
        
        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=self.max_steps, event=diffrax.Event(self.env.cond_fn_nan), 
            args=(env, policy, obs_noise_key, target, tree_evaluator), stepsize_controller=self.stepsize_controller, throw=False
        )

        xs = sol.ys

        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))
        us = jax.vmap(lambda y, tar: tree_evaluator(policy, jnp.concatenate([y, tar])), in_axes=[0,None])(ys, target)

        # def solve(carry, t):
        #     x, lagged_obs = carry

        #     _, y = env.f_obs(obs_noise_key, (t, x))
        #     lagged_obs = jnp.concatenate([lagged_obs[1:self.lag_size], y[0,None]])#, lagged_obs[self.lag_size+1:2*self.lag_size], y[1,None]])
        #     u = tree_evaluator(policy, jnp.concatenate([lagged_obs, target]))
        #     dx = env.drift(t, x, u)
        #     new_x = x + self.dt0 * dx

        #     carry = new_x, lagged_obs
        #     return carry, (new_x, y, u)
    
        # init_carry = (x0, jnp.zeros(self.obs_size*self.lag_size))
        # _, (xs, ys, us) = jax.lax.scan(solve, init_carry, solve_ts)
        
        # xs = jax.vmap(self.interpolate, in_axes=[None, 1, None], out_axes=1)(solve_ts, xs, ts)
        # ys = jax.vmap(self.interpolate, in_axes=[None, 1, None], out_axes=1)(solve_ts, ys, ts)
        # us = jax.vmap(self.interpolate, in_axes=[None, 1, None], out_axes=1)(solve_ts, us, ts)

        fitness = env.fitness_function(xs, us, target, ts)

        return xs, ys, us, fitness
    
    #Define state equation
    def _drift(self, t, x, args):
        env, policy, obs_noise_key, target, tree_evaluator = args
        _, y = env.f_obs(obs_noise_key, (t, x)) #Get observations from system
        u = tree_evaluator(policy, jnp.concatenate([y, target]))

        dx = env.drift(t, x, u) #Apply control to system and get system change
        return dx
    
    #Define diffusion
    def _diffusion(self, t, x, args):
        env, policy, obs_noise_key, target, tree_evaluator = args

        return env.diffusion(t, x, jnp.array([0]))
    
    def interpolate(self, solve_ts, ys, eval_ts):
        return diffrax.LinearInterpolation(ts=solve_ts, ys=ys).evaluate(eval_ts)
    

class EvaluatorMT:
    def __init__(self, env, state_size: int, dt0: float, solver=diffrax.Euler(), max_steps: int = 16**4, stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()) -> None:
        """Evaluator for dynamic symbolic policies in control tasks

        Attributes:
            env: Environment on which the candidate is evaluated
            max_fitness: Max fitness which is assigned when a trajectory returns an invalid value
            state_size: Dimensionality of the hidden state
            obs_size: Dimensionality of the observations
            control_size: Dimensionality of the control
            latent_size: Dimensionality of the state of the environment
            dt0: Initial step size for integration
            solver: Solver used for integration
            max_steps: The maximum number of steps that can be used in integration
            stepsize_controller: Controller for the stepsize during integration
        """
        self.env = env
        self.max_fitness = 1e4
        self.state_size = state_size
        self.obs_size = env.n_obs
        self.control_size = env.n_control
        self.latent_size = env.n_var*env.n_dim
        self.dt0 = dt0
        self.solver = solver
        self.max_steps = max_steps
        self.stepsize_controller = stepsize_controller

    def __call__(self, candidate: Array, data: Tuple, tree_evaluator: Callable) -> float:
        """Evaluates the candidate on a task

        :param coefficients: The coefficients of the candidate
        :param nodes: The nodes and index references of the candidate
        :param data: The data required to evaluate the candidate
        :param tree_evaluator: Function for evaluating trees

        Returns: Fitness of the candidate
        """
        _, _, _, _, fitness = self.evaluate_candidate(candidate, data, tree_evaluator)

        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
        fitness = jnp.mean(fitness)
        return jnp.clip(fitness,0,self.max_fitness)
    
    def evaluate_candidate(self, candidate: Array, data: Tuple, tree_evaluator: Callable) -> Tuple[Array, Array, Array, Array, float]:
        """Evaluates a candidate given a task and data

        :param candidate: Candidate that is evaluated
        :param data: The data required to evaluate the candidate
        :param tree_evaluator: Function for evaluating trees
        
        Returns: Predictions and fitness of the candidate
        """
        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0, 0, None])(candidate, *data, tree_evaluator)
    
    def evaluate_control_loop(self, candidate: Array, x0: Array, ts: Array, target: float, process_noise_key: jrandom.PRNGKey, obs_noise_key: jrandom.PRNGKey, params: Tuple, tree_evaluator: Callable) -> Tuple[Array, Array, Array, Array, float]:
        """Solves the coupled differential equation of the system and controller. The differential equation of the system is defined in the environment and the differential equation 
        of the control is defined by the set of trees
        Inputs:
            candidate (NetworkTrees): Candidate with trees for the hidden state and readout
            x0 (float): Initial state of the system
            ts (Array[float]): time points on which the controller is evaluated
            target (float): Target position that the system should reach
            key (PRNGKey)
            params (Tuple[float]): Parameters that define the system

        Returns:
            xs (Array[float]): States of the system at every time point
            ys (Array[float]): Observations of the system at every time point
            us (Array[float]): Control of the candidate at every time point
            activities (Array[float]): Activities of the hidden state of the candidate at every time point
            fitness (float): Fitness of the candidate 
        """
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)
    
        targets = diffrax.LinearInterpolation(ts, jnp.hstack([t*jnp.ones(int(ts.shape[0]//target.shape[0])) for t in target]))

        solver = self.solver
        dt0 = self.dt0
        saveat = diffrax.SaveAt(ts=ts)

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.latent_size,), key=process_noise_key, levy_area=diffrax.SpaceTimeLevyArea) #define process noise
        system = diffrax.MultiTerm(diffrax.ODETerm(self._drift), diffrax.ControlTerm(self._diffusion, brownian_motion))
        
        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=self.max_steps, event=diffrax.Event(self.env.cond_fn_nan), 
            args=(env, candidate, obs_noise_key, targets, tree_evaluator), stepsize_controller=self.stepsize_controller, throw=True
        )

        xs = sol.ys
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))
        target_ts = jax.vmap(lambda t: targets.evaluate(t))(ts)
        
        us = jax.vmap(lambda y, tar: tree_evaluator(candidate, jnp.concatenate([y, jnp.array([tar])])), in_axes=[0,0])(ys, target_ts)


        fitness = env.fitness_function(xs, us, target_ts, ts)

        return xs, ys, us, fitness       
    
    def _drift(self, t, x, args):
        env, policy, obs_noise_key, target, tree_evaluator = args
        tar = target.evaluate(t)
        _, y = env.f_obs(obs_noise_key, (t, x)) #Get observations from system
        u = tree_evaluator(policy, jnp.concatenate([y, jnp.array([tar])]))

        dx = env.drift(t, x, u) #Apply control to system and get system change
        return dx
    
    def _diffusion(self, t, x, args):
        env, policy, obs_noise_key, target, tree_evaluator = args

        return env.diffusion(t, x, jnp.array([0]))