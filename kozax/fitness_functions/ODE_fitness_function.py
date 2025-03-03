"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""

import jax
from jaxtyping import Array
import jax.numpy as jnp
from typing import Tuple, Callable
from kozax.fitness_functions.base_fitness_function import BaseFitnessFunction
import diffrax

class ODEFitnessFunction(BaseFitnessFunction):
    """
    Evaluator for candidates on symbolic regression tasks.

    Parameters
    ----------
    solver : :class:`diffrax.AbstractSolver`, optional
        Solver used for integration. Default is `diffrax.Euler()`.
    dt0 : float, optional
        Initial step size for integration. Default is 0.01.
    max_steps : int, optional
        The maximum number of steps that can be used in integration. Default is 16**4.
    stepsize_controller : :class:`diffrax.AbstractStepSizeController`, optional
        Controller for the stepsize during integration. Default is `diffrax.ConstantStepSize()`.

    Attributes
    ----------
    dt0 : float
        Initial step size for integration.
    MSE : Callable
        Function that computes the mean squared error.
    system : :class:`diffrax.ODETerm`
        ODE term of the drift function.
    solver : :class:`diffrax.AbstractSolver`
        Solver used for integration.
    stepsize_controller : :class:`diffrax.AbstractStepSizeController`
        Controller for the stepsize during integration.
    max_steps : int
        The maximum number of steps that can be used in integration.

    Methods
    -------
    __call__(candidate, data, tree_evaluator)
        Evaluates the candidate on a task.
    evaluate_time_series(candidate, x0, ts, ys, tree_evaluator)
        Integrate the candidate as a differential equation and compute the fitness given the predictions.
    drift(t, x, args)
        Drift function for the ODE system.
    """

    def __init__(self, solver: diffrax.AbstractSolver = diffrax.Euler(), dt0: float = 0.01, max_steps: int = 16**4, stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()) -> None:
        self.dt0 = dt0
        self.MSE = lambda pred_ys, true_ys: jnp.mean(jnp.sum(jnp.abs(pred_ys - true_ys), axis=-1))/jnp.mean(jnp.abs(true_ys))
        self.system = diffrax.ODETerm(self.drift)
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.max_steps = max_steps

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
        x0, ts, ys = data
        fitness = jax.vmap(self.evaluate_time_series, in_axes=[None, 0, None, 0, None])(candidate, x0, ts, ys, tree_evaluator)
        return jnp.mean(fitness)
    
    def evaluate_time_series(self, candidate: Array, x0: Array, ts: Array, ys: Array, tree_evaluator: Callable) -> float:
        """
        Integrate the candidate as a differential equation and compute the fitness given the predictions.

        Parameters
        ----------
        candidate : :class:`jax.Array`
            Candidate that is evaluated.
        x0 : :class:`jax.Array`
            Initial conditions of the environment.
        ts : :class:`jax.Array`
            Timepoints of which the system has to be solved.
        ys : :class:`jax.Array`
            Ground truth data used to compute the fitness.
        tree_evaluator : :class:`Callable`
            Function for evaluating trees.

        Returns
        -------
        float
            Fitness of the candidate.
        """
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(
            self.system, self.solver, ts[0], ts[-1], self.dt0, x0, args=(candidate, tree_evaluator), saveat=saveat, max_steps=self.max_steps, stepsize_controller=self.stepsize_controller, 
            adjoint=diffrax.DirectAdjoint(), throw=False
        )
        pred_ys = sol.ys
        fitness = self.MSE(pred_ys, ys)
        return fitness
    
    def drift(self, t: float, x: Array, args: Tuple) -> Array:
        """
        Drift function for the ODE system.

        Parameters
        ----------
        t : float
            Current time.
        x : :class:`jax.Array`
            Current state.
        args : :class:`tuple`
            Additional arguments, including the candidate and tree evaluator.

        Returns
        -------
        :class:`jax.Array`
            Derivative of the state.
        """
        candidate, tree_evaluator = args
        dx = tree_evaluator(candidate, x)
        return dx