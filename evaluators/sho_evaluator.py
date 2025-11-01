import copy

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr


class SHOEvaluator:
    def __init__(self, env, state_size, dt0, reward_fn=None, solver=diffrax.Euler(), 
                max_steps=16**4, stepsize_controller=diffrax.ConstantStepSize()):
        self.env = env
        self.max_fitness = 1e4
        self.state_size = state_size
        self.obs_size = env.n_obs
        self.control_size = env.n_control_inputs
        self.latent_size = env.n_var * env.n_dim
        self.dt0 = dt0
        self.solver = solver
        self.max_steps = max_steps
        self.stepsize_controller = stepsize_controller
        self.reward_fn = reward_fn

    def __call__(self, candidate, data, tree_evaluator):
        xs, ys, us, activities, fitness, rewards = jax.vmap(
            self.evaluate_trajectory,
            in_axes=[None, 0, None, 0, 0, 0, None]
        )(candidate, *data, tree_evaluator)
        return jnp.mean(fitness)

    def evaluate_trajectory(self, candidate, x0, ts, target, noise_key, params, tree_evaluator):
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        state_equation = candidate[:self.state_size]
        readout = candidate[self.state_size:]
        saveat = diffrax.SaveAt(ts=ts)
        process_noise_key, obs_noise_key = jr.split(noise_key, 2)

        target_interp = diffrax.LinearInterpolation(ts, target)

        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])
        ode = diffrax.ODETerm(self._drift)
        control_term = diffrax.ControlTerm(
            self._diffusion,
            diffrax.UnsafeBrownianPath(shape=(env.n_var,), key=process_noise_key, levy_area=diffrax.SpaceTimeLevyArea)
            )
        system = diffrax.MultiTerm(ode, control_term)

        sol = diffrax.diffeqsolve(
            system, self.solver, ts[0], ts[-1], self.dt0, _x0, saveat=saveat,
            adjoint=diffrax.DirectAdjoint(), max_steps=self.max_steps,
            event=diffrax.Event(env.cond_fn_nan),
            args=(env, state_equation, readout, obs_noise_key, target_interp, tree_evaluator),
            stepsize_controller=self.stepsize_controller, throw=False
            )

        xs = sol.ys[:, :self.latent_size]
        activities = sol.ys[:, self.latent_size:]
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))

        target_ts = target_interp.evaluate(ts)
        if self.reward_fn is not None:
            rs = jax.vmap(lambda t, y_t, tar_t: self.reward_fn(t, y_t, tar_t))(ts, ys, target_ts)
            feedback = rs
        else:
            rs = jnp.zeros((ts.shape[0],))
            feedback = target

        us = jax.vmap(
            lambda y, a, tar: tree_evaluator(
                readout, jnp.concatenate([y, a, jnp.zeros(self.control_size), jnp.atleast_1d(tar)])
            )
        )(ys, activities, feedback)

        fitness = env.fitness_function(xs, us[:, None], target, ts)
        return xs, ys, us, activities, fitness, rs


    def _drift(self, t, x_a, args):
        env, state_equation, readout, obs_noise_key, target_interp, tree_evaluator = args
        x = x_a[:self.latent_size]
        a = x_a[self.latent_size:]
        _, y = env.f_obs(obs_noise_key, (t, x))
        target_t = target_interp.evaluate(t)
        if self.reward_fn is not None:
            feedback_t = jnp.atleast_1d(self.reward_fn(t, y, target_t))
        else:
            feedback_t = jnp.atleast_1d(target_t)
        u = tree_evaluator(
            readout, jnp.concatenate([jnp.zeros(self.obs_size), a, jnp.zeros(self.control_size), feedback_t])
        )
        u = jnp.atleast_1d(u)
        dx = env.drift(t, x, u)
        da = tree_evaluator(state_equation, jnp.concatenate([y, a, u, feedback_t]))
        return jnp.concatenate([dx, da])

    def _diffusion(self, t, x_a, args):
        env, state_equation, readout, obs_noise_key, input_interp, tree_evaluator = args
        x = x_a[:self.latent_size]
        return jnp.concatenate([env.diffusion(t, x, jnp.array([0])),
                                jnp.zeros((self.state_size, self.latent_size))])