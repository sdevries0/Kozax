from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    def __init__(self, env, state_size, dt0):
        self.env = env
        self.state_size = state_size
        self.obs_size = env.n_obs
        self.control_size = env.n_control_inputs
        self.latent_size = env.n_var * env.n_dim
        self.dt0 = dt0
        self.max_fitness = 1e4

    @abstractmethod
    def __call__(self, candidate, data, tree_evaluator):
        raise NotImplementedError

    @abstractmethod
    def evaluate_trajectory(self, candidate, x0, ts, target, noise_key, params, tree_evaluator):
        """
        Evaluates one trajectory of the candidate within the environment.

        Returns
        -------
        tuple
            (xs, ys, us, activities, fitness)
        """
        raise NotImplementedError

    @abstractmethod
    def _drift(self, t, state, args):
        """Defines deterministic part of the system."""
        raise NotImplementedError

    @abstractmethod
    def _diffusion(self, t, state, args):
        """Defines stochastic component of the system."""
        raise NotImplementedError
